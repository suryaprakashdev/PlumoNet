import os
import io
import base64
import shutil
import tempfile
from pathlib import Path
from starlette.applications import Starlette
from starlette.staticfiles import StaticFiles
import numpy as np
import cv2
import bentoml
from bentoml import asgi_app
with bentoml.importing():
    from inference_3d import InferencePipeline3D, load_dicom_volume

# ─── Helpers ──────────────────────────────────────────────────────────────────

def find_dicom_root(base_dir: str) -> str:
    """Walk extracted ZIP and return the first directory containing DICOM files."""
    for dirpath, dirnames, filenames in os.walk(base_dir):
        dirnames[:] = [d for d in dirnames if d not in ("gradcam_out", "__MACOSX")]
        if dirpath.endswith("__MACOSX"):
            continue
        if any(f.lower().endswith('.dcm') or ('.' not in f and not f.startswith('._'))
               for f in filenames):
            return dirpath
    return base_dir

def generate_annotated_slice(arr2d, mask2d=None, candidates=None, vmin=-1000, vmax=400):
    """
    Renders a 2D HU slice using OpenCV.
    Applies Gaussian smoothing, draws colored contours for the segmentation mask, 
    and adds text labels for nodules.
    """
    # 1. Windowing to 8-bit
    normed = np.clip((arr2d.astype(np.float32) - vmin) / (vmax - vmin), 0, 1)
    img_8u = (normed * 255).astype(np.uint8)

    # 2. Anti-aliasing / Smoothing
    img_smooth = cv2.GaussianBlur(img_8u, (3, 3), 0)

    # Convert to BGR for colored overlays
    img_bgr = cv2.cvtColor(img_smooth, cv2.COLOR_GRAY2BGR)

    # Only annotate malignant candidates (prob >= 0.5); skip benign/borderline entirely
    malignant_candidates = [c for c in (candidates or []) if c["probability"] >= 0.5]

    if mask2d is not None and mask2d.any() and malignant_candidates:
        mask_8u = (mask2d > 0.5).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        RED = (0, 0, 255)  # BGR — single color for all malignant markers

        for cand in malignant_candidates:
            prob = cand["probability"]
            cy, cx = cand["center_2d"]

            # Draw marker at centroid
            cv2.drawMarker(img_bgr, (cx, cy), RED, cv2.MARKER_CROSS, 10, 2)

            # Draw label
            label = f"#{cand['candidate_index']} ({prob:.0%})"
            cv2.putText(img_bgr, label, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, RED, 1, cv2.LINE_AA)

        # Draw contours in red only when malignant candidates are present on this slice
        cv2.drawContours(img_bgr, contours, -1, RED, 1, cv2.LINE_AA)

    # Encode to base64
    _, buffer = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buffer).decode()


# ─── BentoML Service ───────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

frontend_app = Starlette()

frontend_app.mount(
    "/",
    StaticFiles(
        directory=os.path.join(BASE_DIR, "frontend"),
        html=True,
    ),
    name="frontend",
)

@asgi_app(frontend_app, path="/ui")
@bentoml.service(
    resources={"memory": "4Gi"},
    traffic={"timeout": 600},
    http={
        "cors": {
            "enabled": True,
            "access_control_allow_origins": ["*"],
            "access_control_allow_methods": ["GET", "POST", "OPTIONS"],
            "access_control_allow_headers": ["*"],
        }
    },
)
class LungNoduleService:

    def __init__(self):
        self.pipeline = InferencePipeline3D(
            unet_onnx_path=os.path.join(BASE_DIR, "checkpoints", "unet3d.onnx"),
            resnet_onnx_path=os.path.join(BASE_DIR, "checkpoints", "resnet3d.onnx"),
        )

    @bentoml.api
    def predict(self, dicom_zip: Path) -> dict:

        with tempfile.TemporaryDirectory() as tmpdir:
            gradcam_dir = os.path.join(tmpdir, "gradcam_out")
            os.makedirs(gradcam_dir, exist_ok=True)

            shutil.unpack_archive(str(dicom_zip), tmpdir)
            dicom_root = find_dicom_root(tmpdir)

            # Load full volume BEFORE run_volume (files still on disk here)
            full_vol, _ = load_dicom_volume(dicom_root)  # (D, H, W) in HU

            # Run segmentation + classification
            result = self.pipeline.run_volume(
                dicom_root,
                aggregation="top_k",
                k=5,
                min_malignancy_prob=0.0
            )
            
            # Get the segmentation mask
            seg_mask = getattr(result, 'segmentation_mask', None)  # shape (D,H,W)

            # ── Derive patient-level result ───────────────────────────────────
            candidates = result.candidates
            patient_score = max((c.probability for c in candidates), default=0.0)
            patient_prediction = "Malignant" if patient_score >= 0.5 else "Benign"

            # ── Generate full-slice views for active slices ─────────────
            # We want to map slice indices to a base64 string and list of nodules
            # for each plane.
            candidate_views = {"axial": {}, "coronal": {}, "sagittal": {}}
            vol = result.volume_iso

            def get_candidates_in_slice(plane, idx):
                in_slice = []
                for c in candidates:
                    (z0, z1), (y0, y1), (x0, x1) = c.bbox
                    if plane == "axial" and z0 <= idx <= z1:
                        in_slice.append({"candidate_index": int(c.candidate_index), "probability": float(c.probability), "center_2d": (int(c.centroid[1]), int(c.centroid[2]))})
                    elif plane == "coronal" and y0 <= idx <= y1:
                        in_slice.append({"candidate_index": int(c.candidate_index), "probability": float(c.probability), "center_2d": (int(c.centroid[0]), int(c.centroid[2]))})
                    elif plane == "sagittal" and x0 <= idx <= x1:
                        in_slice.append({"candidate_index": int(c.candidate_index), "probability": float(c.probability), "center_2d": (int(c.centroid[0]), int(c.centroid[1]))})
                return in_slice

            for z in result.active_slices["axial"]:
                cands = get_candidates_in_slice("axial", z)
                img = generate_annotated_slice(vol[z, :, :], seg_mask[z, :, :] if seg_mask is not None else None, cands)
                candidate_views["axial"][str(z)] = {"image": img, "nodules": cands}

            for y in result.active_slices["coronal"]:
                cands = get_candidates_in_slice("coronal", y)
                img = generate_annotated_slice(vol[:, y, :], seg_mask[:, y, :] if seg_mask is not None else None, cands)
                candidate_views["coronal"][str(y)] = {"image": img, "nodules": cands}

            for x in result.active_slices["sagittal"]:
                cands = get_candidates_in_slice("sagittal", x)
                img = generate_annotated_slice(vol[:, :, x], seg_mask[:, :, x] if seg_mask is not None else None, cands)
                candidate_views["sagittal"][str(x)] = {"image": img, "nodules": cands}
        # ── Build response (outside tmpdir — all data already in memory) ──────
        return {
            "patient_score":   float(patient_score),
            "prediction":      str(patient_prediction),
            "num_candidates":  int(len(candidates)),
            "top_candidates": [
                {
                    "centroid":         [int(v) for v in c.centroid],
                    "prob":             float(c.probability),
                    "prediction":       str(c.prediction),
                    "volume_voxels":    int(c.volume_voxels),
                    "candidate_index":  int(c.candidate_index),
                }
                for c in candidates
            ],
            "candidate_views":  candidate_views,
            "metadata": {
                "patient_id":   str(result.metadata.get("patient_id", "")),
                "volume_shape": [int(v) for v in result.metadata.get("volume_shape", [])],
                "total_time_s": round(float(result.total_time_ms) / 1000, 1),
                "seg_time_s":   round(float(result.seg_time_ms) / 1000, 1),
                "cls_time_s":   round(float(result.cls_time_ms) / 1000, 1),
            },
        }