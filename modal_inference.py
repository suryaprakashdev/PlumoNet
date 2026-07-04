"""
PlumoNet Modal GPU Inference — REAL pipeline (ported from service.py)
=====================================================================

Runs the actual 3D U-Net segmentation + 3D ResNet-10 classification on GPU
and renders real annotated MPR slices (base64 PNGs) in the exact shape the
frontend expects: candidate_views[plane][slice_idx] = {"image": <b64>, "nodules": [...]}.

This replaces the placeholder. The render/slice logic here is a direct port of
the `predict` method + `generate_annotated_slice` from service.py.

Deploy with:
    modal deploy modal_inference.py

Ships into the image:
  - inference_3d.py, unet3d.py, resnet3d.py   (model + pipeline code)
  - checkpoints/unet3d.onnx, checkpoints/resnet3d.onnx  (weights)
Run `modal deploy` from the repo root so these relative paths resolve.
"""

import os
import io
import base64
import shutil
import tempfile
from typing import Dict, Any, List

import modal

# ─── Modal App ─────────────────────────────────────────────────────────────

app = modal.App("plumonet-inference")

# ─── GPU Image ─────────────────────────────────────────────────────────────
# Notes:
#  - apt: libgl1/libglib2.0-0 satisfy transitive OpenGL deps (monai/skimage/mpl).
#    opencv-python-headless itself doesn't need libGL, but keeping these avoids
#    the whole class of "libGL.so.1: cannot open shared object file" crashes.
#  - We install onnxruntime-GPU explicitly and drop the CPU onnxruntime so the
#    CUDAExecutionProvider is actually available on the A10 (otherwise inference
#    silently runs on CPU even with a GPU attached).
#  - Model code + ONNX weights are baked into the image via add_local_*.

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install_from_requirements("requirements_modal.txt")
    # Swap CPU onnxruntime (from requirements) for the GPU build:
    .pip_install("onnxruntime-gpu==1.17.1")
    # Ship pipeline + model source files:
    .add_local_file("inference_3d.py", "/root/inference_3d.py")
    .add_local_file("unet3d.py", "/root/unet3d.py")
    .add_local_file("resnet3d.py", "/root/resnet3d.py")
    # Ship ONNX weights:
    .add_local_file("checkpoints/unet3d.onnx", "/root/checkpoints/unet3d.onnx")
    .add_local_file("checkpoints/resnet3d.onnx", "/root/checkpoints/resnet3d.onnx")
)

# ─── Render helper (ported verbatim from service.py) ───────────────────────

# def get_cached_pipeline():
#     """Lazy-load pipeline once and reuse it."""
#     global _pipeline_cache
#     if _pipeline_cache is None:
#         print("[Cache] Loading InferencePipeline3D (first time only)...")
#         from inference_3d import InferencePipeline3D
#         _pipeline_cache = InferencePipeline3D(
#             unet_onnx_path="/root/checkpoints/unet3d.onnx",
#             resnet_onnx_path="/root/checkpoints/resnet3d.onnx",
#         )
#         print("[Cache] ✓ Pipeline cached in GPU memory")
#     return _pipeline_cache
def generate_annotated_slice(arr2d, mask2d=None, candidates=None, vmin=-1000, vmax=400):
    """
    Render a 2D HU slice with OpenCV: HU windowing -> 8-bit -> light smoothing,
    then draw a red crosshair + label for each malignant candidate (prob >= 0.5).
    Returns a base64-encoded PNG string.
    """
    import numpy as np
    import cv2

    # 1. HU windowing to 8-bit
    normed = np.clip((arr2d.astype(np.float32) - vmin) / (vmax - vmin), 0, 1)
    img_8u = (normed * 255).astype(np.uint8)

    # 2. Light anti-aliasing
    img_smooth = cv2.GaussianBlur(img_8u, (3, 3), 0)
    img_bgr = cv2.cvtColor(img_smooth, cv2.COLOR_GRAY2BGR)

    # 3. Markers for malignant candidates only
    RED = (0, 0, 255)  # BGR
    for cand in (candidates or []):
        if cand["probability"] < 0.5:
            continue
        prob = cand["probability"]
        cy, cx = cand["center_2d"]
        cv2.drawMarker(img_bgr, (cx, cy), RED, cv2.MARKER_CROSS, 10, 2)
        label = f"#{cand['candidate_index']} ({prob:.0%})"
        cv2.putText(img_bgr, label, (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, RED, 1, cv2.LINE_AA)

    _, buffer = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buffer).decode()


def find_dicom_root(base_dir: str) -> str:
    """Walk extracted ZIP; return first dir containing DICOM files (skip __MACOSX)."""
    for dirpath, dirnames, filenames in os.walk(base_dir):
        dirnames[:] = [d for d in dirnames if d not in ("gradcam_out", "__MACOSX")]
        if dirpath.endswith("__MACOSX"):
            continue
        if any(f.lower().endswith('.dcm') or ('.' not in f and not f.startswith('._'))
               for f in filenames):
            return dirpath
    return base_dir


# ─── GPU Inference Function ────────────────────────────────────────────────


@app.function(
    image=image,
    gpu="a100",
    secrets=[modal.Secret.from_name("azure-connection-string")],
    timeout=600,
)
def run_inference(blob_name: str) -> Dict[str, Any]:
    """Download DICOM zip from Azure, run seg+cls, render MPR slices, return payload."""
    import numpy as np  # noqa: F401  (used by helpers)
    from azure.storage.blob import BlobServiceClient

    print(f"[Modal GPU] Starting inference for: {blob_name}")

    # ── Azure connection string from Modal secret ──────────────────────────
    connection_string = (
        os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        or os.environ.get("MODAL_AZURE_STORAGE_CONNECTION_STRING")
    )
    if not connection_string:
        available = [k for k in os.environ if 'AZURE' in k or 'STORAGE' in k]
        print(f"[Modal GPU] Available Azure vars: {available}")
        raise ValueError(
            "AZURE_STORAGE_CONNECTION_STRING not found. Check the "
            "'azure-connection-string' secret in the Modal dashboard."
        )

    container = "dicom-uploads"
    bsc = BlobServiceClient.from_connection_string(connection_string)

    # ── Load pipeline (ONNX weights baked into image at /root/checkpoints) ──
    from inference_3d import InferencePipeline3D

    pipeline = InferencePipeline3D(
        unet_onnx_path="/root/checkpoints/unet3d.onnx",
        resnet_onnx_path="/root/checkpoints/resnet3d.onnx",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # ── Download + unzip ───────────────────────────────────────────────
        zip_path = os.path.join(tmpdir, "upload.zip")
        blob_client = bsc.get_blob_client(container=container, blob=blob_name)
        print(f"[Modal GPU] Downloading blob: {blob_name}")

        downloader = blob_client.download_blob()
        data = downloader.readall()
        with open(zip_path, "wb") as f:
            f.write(data)

        size = os.path.getsize(zip_path)
        head = data[:8]
        print(f"[Modal GPU] Downloaded {size} bytes ✓  first-bytes={head!r}")

        # Zip magic is b'PK\x03\x04' (or PK\x05\x06 empty / PK\x07\x08 spanned)
        if not data.startswith(b"PK"):
            raise ValueError(
                f"Downloaded blob is not a ZIP. size={size}, "
                f"first bytes={head!r}. The upload likely stored the wrong content."
            )

        shutil.unpack_archive(zip_path, tmpdir, format="zip")
        dicom_root = find_dicom_root(tmpdir)

        # pipeline = get_cached_pipeline()

        # ── Segmentation + classification ──────────────────────────────────
        result = pipeline.run_volume(
            dicom_root,
            aggregation="top_k",
            k=5,
            min_malignancy_prob=0.0,
        )

        seg_mask = getattr(result, "segmentation_mask", None)  # (D,H,W) or None
        candidates = result.candidates
        vol = result.volume_iso

        patient_score = max((c.probability for c in candidates), default=0.0)
        patient_prediction = "Malignant" if patient_score >= 0.5 else "Benign"

        # ── Per-slice candidate lookup (ported from service.py) ────────────
        def get_candidates_in_slice(plane: str, idx: int) -> List[dict]:
            in_slice = []
            for c in candidates:
                (z0, z1), (y0, y1), (x0, x1) = c.bbox
                if plane == "axial" and z0 <= idx <= z1:
                    in_slice.append({
                        "candidate_index": int(c.candidate_index),
                        "probability": float(c.probability),
                        "center_2d": (int(c.centroid[1]), int(c.centroid[2])),
                    })
                elif plane == "coronal" and y0 <= idx <= y1:
                    in_slice.append({
                        "candidate_index": int(c.candidate_index),
                        "probability": float(c.probability),
                        "center_2d": (int(c.centroid[0]), int(c.centroid[2])),
                    })
                elif plane == "sagittal" and x0 <= idx <= x1:
                    in_slice.append({
                        "candidate_index": int(c.candidate_index),
                        "probability": float(c.probability),
                        "center_2d": (int(c.centroid[0]), int(c.centroid[1])),
                    })
            return in_slice

        candidate_views: Dict[str, Dict[str, Any]] = {
            "axial": {}, "coronal": {}, "sagittal": {}
        }

        for z in result.active_slices["axial"]:
            cands = get_candidates_in_slice("axial", z)
            img = generate_annotated_slice(
                vol[z, :, :], seg_mask[z, :, :] if seg_mask is not None else None, cands)
            candidate_views["axial"][str(z)] = {"image": img, "nodules": cands}

        for y in result.active_slices["coronal"]:
            cands = get_candidates_in_slice("coronal", y)
            img = generate_annotated_slice(
                vol[:, y, :], seg_mask[:, y, :] if seg_mask is not None else None, cands)
            candidate_views["coronal"][str(y)] = {"image": img, "nodules": cands}

        for x in result.active_slices["sagittal"]:
            cands = get_candidates_in_slice("sagittal", x)
            img = generate_annotated_slice(
                vol[:, :, x], seg_mask[:, :, x] if seg_mask is not None else None, cands)
            candidate_views["sagittal"][str(x)] = {"image": img, "nodules": cands}

    # ── Response (matches service.py + frontend contract) ──────────────────
    payload = {
        "patient_score": float(patient_score),
        "prediction": str(patient_prediction),
        "num_candidates": int(len(candidates)),
        "top_candidates": [
            {
                "centroid": [int(v) for v in c.centroid],
                "prob": float(c.probability),
                "prediction": str(c.prediction),
                "volume_voxels": int(c.volume_voxels),
                "candidate_index": int(c.candidate_index),
            }
            for c in candidates
        ],
        "candidate_views": candidate_views,
        "metadata": {
            "patient_id": str(result.metadata.get("patient_id", "")),
            "volume_shape": [int(v) for v in result.metadata.get("volume_shape", [])],
            "total_time_s": round(float(result.total_time_ms) / 1000, 1),
            "seg_time_s": round(float(result.seg_time_ms) / 1000, 1),
            "cls_time_s": round(float(result.cls_time_ms) / 1000, 1),
        },
    }

    print(f"[Modal GPU] Inference complete ✓  prediction={payload['prediction']} "
          f"candidates={payload['num_candidates']}")
    return payload


if __name__ == "__main__":
    print("Deploy with:  modal deploy modal_inference.py  (run from repo root)")

