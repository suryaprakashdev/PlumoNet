"""
inference_3d.py
---------------
Phase 4: 3D inference pipeline replacing the legacy 2D ROI extraction.

Pipeline (from plan diagram):
  1. Sliding window: MONAI SlidingWindowInferer on full CT volume
  2. Candidate extraction: Threshold mask p > 0.5, 3D connected components
  3. Classify candidates: Crop 64×64×64 per candidate, 3D ResNet → prob
  4. Aggregate: max / top-k mean of candidates → patient probability

Usage:
  from inference_3d import InferencePipeline3D

  pipeline = InferencePipeline3D(
      unet_checkpoint="checkpoints/unet3d_best.pth",
      resnet_checkpoint="checkpoints/resnet3d_calibrated.pth",
  )
  result = pipeline.run_volume("/path/to/dicom_series_folder")
  print(result.summary())
"""

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import label as scipy_label
from scipy.ndimage import zoom

from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    NormalizeIntensity,
    ScaleIntensityRange,
    ToTensor,
)

from unet3d import UNet3D
from resnet3d import ResNet3D10

logger = logging.getLogger(__name__)


import onnxruntime as ort

class UNetONNXWrapper:
    """Wrapper to make ONNX session callable like a PyTorch module for SlidingWindowInferer."""
    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, 1, 64, 64, 64) torch tensor
        ort_inputs = {self.input_name: x.cpu().numpy()}
        ort_outs = self.session.run(None, ort_inputs)
        return torch.from_numpy(ort_outs[0]).to(x.device)


# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────

LUNG_HU_MIN = -1350
LUNG_HU_MAX = 150
CROP_SIZE = 64
SEG_THRESHOLD = 0.5


# ──────────────────────────────────────────────
#  Result containers
# ──────────────────────────────────────────────

@dataclass
class CandidateResult:
    """Result for a single nodule candidate."""
    candidate_index: int
    centroid: Tuple[int, int, int]     # (z, y, x) in volume coordinates
    bbox: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] # ((z0,z1), (y0,y1), (x0,x1))
    volume_voxels: int                 # number of voxels in the segmented region
    probability: float                 # malignancy probability [0, 1]
    prediction: str                    # "Malignant" | "Benign"


@dataclass
class VolumeResult3D:
    """Patient-level inference result from the 3D pipeline."""

    request_id: str
    folder_path: str
    patient_id: str

    # Patient-level aggregated prediction
    patient_probability: float
    patient_prediction: str           # "Malignant" | "Benign"
    patient_confidence: str           # "High" | "Medium" | "Low"
    aggregation_method: str

    # Segmentation info
    n_candidates_found: int
    segmentation_mask: Optional[np.ndarray] = None   # full-volume mask
    volume_iso: Optional[np.ndarray] = None          # isotropic resampled volume

    # Per-candidate details
    candidates: List[CandidateResult] = field(default_factory=list)

    # Active slices that contain nodules
    active_slices: Dict[str, List[int]] = field(default_factory=lambda: {"axial": [], "coronal": [], "sagittal": []})

    # Timing
    total_time_ms: float = 0.0
    seg_time_ms: float = 0.0
    cls_time_ms: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        sep = "=" * 60
        lines = [
            sep,
            f"  3D Volume Inference — {self.request_id[:8]}",
            sep,
            f"  Patient ID    : {self.patient_id}",
            f"  Series        : {os.path.basename(self.folder_path)}",
            f"  Aggregation   : {self.aggregation_method}",
            f"  Candidates    : {self.n_candidates_found} detected",
            "",
            f"  ► PREDICTION  : {self.patient_prediction}",
            f"  ► PROBABILITY : {self.patient_probability:.4f}",
            f"  ► CONFIDENCE  : {self.patient_confidence}",
            "",
            "  Candidates:",
        ]
        for c in self.candidates:
            lines.append(
                f"    [{c.candidate_index}] "
                f"centroid={c.centroid} | "
                f"vol={c.volume_voxels}vox | "
                f"{c.prediction} | prob={c.probability:.4f}"
            )
        lines += [
            "",
            f"  Timing: seg={self.seg_time_ms:.0f}ms | "
            f"cls={self.cls_time_ms:.0f}ms | "
            f"total={self.total_time_ms:.0f}ms",
            sep,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "request_id": self.request_id,
            "patient_id": self.patient_id,
            "folder_path": self.folder_path,
            "patient_prediction": self.patient_prediction,
            "patient_probability": round(self.patient_probability, 4),
            "patient_confidence": self.patient_confidence,
            "aggregation_method": self.aggregation_method,
            "n_candidates": self.n_candidates_found,
            "candidates": [
                {
                    "index": c.candidate_index,
                    "centroid": c.centroid,
                    "volume_voxels": c.volume_voxels,
                    "probability": round(c.probability, 4),
                    "prediction": c.prediction,
                }
                for c in self.candidates
            ],
            "timing": {
                "segmentation_ms": round(self.seg_time_ms, 1),
                "classification_ms": round(self.cls_time_ms, 1),
                "total_ms": round(self.total_time_ms, 1),
            },
        }


# ──────────────────────────────────────────────
#  Volume loader (DICOM → HU array)
# ──────────────────────────────────────────────

def load_dicom_volume(folder_path: str) -> Tuple[np.ndarray, dict]:
    """
    Load a DICOM series folder into a (D, H, W) float32 HU volume.

    Returns (volume, metadata_dict).
    """
    import pydicom
    from glob import glob

    folder_path = os.path.abspath(folder_path)
    dcm_files = sorted(glob(os.path.join(folder_path, "*.dcm")))

    if not dcm_files:
        # Try all files (some DICOM series have no extension)
        dcm_files = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
        ])

    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in {folder_path}")

    slices = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f)
            if not hasattr(ds, "pixel_array"):
                continue

            img = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1))
            intercept = float(getattr(ds, "RescaleIntercept", 0))
            hu = img * slope + intercept

            try:
                z = float(ds.ImagePositionPatient[2])
            except (AttributeError, IndexError):
                z = float(getattr(ds, "InstanceNumber", len(slices)))

            slices.append((z, hu, ds))
        except Exception:
            continue

    if not slices:
        raise ValueError(f"No valid DICOM slices in {folder_path}")

    slices.sort(key=lambda x: x[0])

    volume = np.stack([s[1] for s in slices], axis=0)  # (D, H, W)

    # Extract metadata from first slice
    first_ds = slices[0][2]
    ps = getattr(first_ds, "PixelSpacing", [1.0, 1.0])
    sorted_z = [s[0] for s in slices]
    slice_thickness = (
        float(np.median(np.diff(sorted_z))) if len(sorted_z) > 1
        else float(getattr(first_ds, "SliceThickness", 1.0))
    )

    metadata = {
        "patient_id": str(getattr(first_ds, "PatientID", "")),
        "study_date": str(getattr(first_ds, "StudyDate", "")),
        "series_uid": str(getattr(first_ds, "SeriesInstanceUID", "")),
        "pixel_spacing": (float(ps[0]), float(ps[1])),
        "slice_thickness": abs(slice_thickness),
        "n_slices": len(slices),
        "volume_shape": volume.shape,
    }

    return volume, metadata


def resample_to_isotropic(volume: np.ndarray,
                          spacing: Tuple[float, float, float],
                          target: float = 1.0) -> Tuple[np.ndarray, Tuple]:
    """Resample volume to isotropic spacing using scipy zoom."""
    scale = tuple(s / target for s in spacing)
    resampled = zoom(volume, scale, order=1)
    return resampled, scale


# ──────────────────────────────────────────────
#  3D Inference Pipeline
# ──────────────────────────────────────────────

class InferencePipeline3D:
    """
    End-to-end 3D inference pipeline:
      DICOM → resample → U-Net segmentation → candidate extraction →
      ResNet classification → aggregation → patient prediction

    Parameters
    ----------
    unet_checkpoint    : path to unet3d_best.pth
    resnet_checkpoint  : path to resnet3d_calibrated.pth (or resnet3d_best.pth)
    device             : torch device (auto-detected if None)
    seg_threshold      : probability threshold for segmentation mask (default: 0.5)
    min_candidate_voxels : minimum voxels for a valid candidate (default: 10)
    """

    def __init__(
        self,
        unet_onnx_path: str,
        resnet_onnx_path: str,
        device: Optional[torch.device] = None,
        seg_threshold: float = SEG_THRESHOLD,
        min_candidate_voxels: int = 10,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.seg_threshold = seg_threshold
        self.min_candidate_voxels = min_candidate_voxels

        # Setup ONNX providers
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']

        # Load U-Net ONNX
        unet_session = ort.InferenceSession(unet_onnx_path, providers=providers)
        self.unet = UNetONNXWrapper(unet_session)
        logger.info("Loaded U-Net ONNX ← %s", unet_onnx_path)

        # Load ResNet ONNX
        self.resnet_session = ort.InferenceSession(resnet_onnx_path, providers=providers)
        self.resnet_input_name = self.resnet_session.get_inputs()[0].name
        logger.info("Loaded ResNet3D ONNX ← %s", resnet_onnx_path)

        # MONAI SlidingWindowInferer for full-volume segmentation
        self.inferer = SlidingWindowInferer(
            roi_size=(CROP_SIZE, CROP_SIZE, CROP_SIZE),
            sw_batch_size=4,
            overlap=0.25,
            mode="gaussian",
            progress=True,
        )

        # Preprocessing transform for classifier input
        self.cls_transform = Compose([
            ScaleIntensityRange(
                a_min=LUNG_HU_MIN, a_max=LUNG_HU_MAX,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            NormalizeIntensity(channel_wise=True),
        ])

        logger.info("InferencePipeline3D ready | device=%s", self.device)

    def run_volume(
        self,
        folder_path: str,
        aggregation: str = "top_k",
        k: int = 5,
        classification_threshold: float = 0.5,
        min_malignancy_prob: float = 0.0,
    ) -> VolumeResult3D:
        """
        Run end-to-end 3D inference on a DICOM series folder.

        Parameters
        ----------
        folder_path              : directory containing .dcm files
        aggregation              : "max" | "mean" | "top_k"
        k                        : top-k for aggregation
        classification_threshold : decision boundary (default: 0.5)
        min_malignancy_prob      : filter out nodules with prob below this
        """
        request_id = str(uuid.uuid4())
        t0 = time.perf_counter()

        logger.info("3D Inference | request=%s | folder=%s",
                     request_id[:8], os.path.basename(folder_path))

        # ── 1. Load DICOM volume ──────────────
        volume, meta = load_dicom_volume(folder_path)
        patient_id = meta["patient_id"]

        # ── 2. Resample to isotropic 1mm ──────
        spacing = (
            meta["slice_thickness"],
            meta["pixel_spacing"][0],
            meta["pixel_spacing"][1],
        )

        # ── Diagnostic: volume loading ──
        print(f"\n  [DEBUG] === VOLUME LOADING ===")
        print(f"  [DEBUG] Raw volume shape (D,H,W): {volume.shape}")
        print(f"  [DEBUG] Spacing (D,H,W): {spacing}")
        print(f"  [DEBUG] HU range: [{volume.min():.1f}, {volume.max():.1f}]")

        volume_iso, scale_factors = resample_to_isotropic(volume, spacing)

        print(f"  [DEBUG] Resampled volume shape: {volume_iso.shape}")
        print(f"  [DEBUG] Scale factors: {scale_factors}")

        logger.info("Resampled %s → %s (spacing %s → 1mm iso)",
                     volume.shape, volume_iso.shape, spacing)

        # ── 3. Sliding window segmentation ────
        t_seg = time.perf_counter()
        seg_mask = self._segment_volume(volume_iso)
        seg_time = (time.perf_counter() - t_seg) * 1000

        # ── 4. Extract candidates ─────────────
        candidates_info = self._extract_candidates(seg_mask)
        logger.info("Found %d candidates above threshold", len(candidates_info))

        # ── 5. Classify each candidate ────────
        t_cls = time.perf_counter()
        candidate_results = self._classify_candidates(
            volume_iso, candidates_info, classification_threshold, min_malignancy_prob
        )
        cls_time = (time.perf_counter() - t_cls) * 1000

        # ── Collect active slices ─────────────
        active_slices = {"axial": set(), "coronal": set(), "sagittal": set()}
        for cand in candidate_results:
            (z0, z1), (y0, y1), (x0, x1) = cand.bbox
            active_slices["axial"].update(range(z0, z1 + 1))
            active_slices["coronal"].update(range(y0, y1 + 1))
            active_slices["sagittal"].update(range(x0, x1 + 1))
        
        active_slices = {k: sorted(list(v)) for k, v in active_slices.items()}

        # ── 6. Aggregate to patient level ─────
        if candidate_results:
            probs = [c.probability for c in candidate_results]
            patient_prob, agg_label = self._aggregate(
                probs, aggregation, k
            )
        else:
            patient_prob = 0.0
            agg_label = aggregation

        patient_pred = ("Malignant" if patient_prob > classification_threshold
                        else "Benign")
        patient_conf = self._confidence_tier(
            patient_prob, classification_threshold
        )

        total_ms = (time.perf_counter() - t0) * 1000

        result = VolumeResult3D(
            request_id=request_id,
            folder_path=folder_path,
            patient_id=patient_id,
            patient_probability=patient_prob,
            patient_prediction=patient_pred,
            patient_confidence=patient_conf,
            aggregation_method=agg_label,
            n_candidates_found=len(candidate_results),
            segmentation_mask=seg_mask,
            volume_iso=volume_iso,
            candidates=candidate_results,
            active_slices=active_slices,
            total_time_ms=total_ms,
            seg_time_ms=seg_time,
            cls_time_ms=cls_time,
            metadata=meta,
        )

        logger.info(
            "3D Inference complete | %s | prob=%.4f | candidates=%d | time=%.0fms",
            patient_pred, patient_prob, len(candidate_results), total_ms
        )
        return result

    # ── Segmentation ──────────────────────────

    def _segment_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Run U-Net segmentation on the full volume using SlidingWindowInferer.

        Returns binary mask (D, H, W).

        Preprocessing must match training transforms in monai_dataset_3d.py:
          1. ScaleIntensityRange: HU [-1350, 150] → [0, 1]
          2. NormalizeIntensity(channel_wise=True): zero-mean, unit-std
        """
        # Step 1: Window and scale to [0, 1] — matches ScaleIntensityRanged
        vol_windowed = np.clip(volume, LUNG_HU_MIN, LUNG_HU_MAX)
        vol_normed = (vol_windowed - LUNG_HU_MIN) / (LUNG_HU_MAX - LUNG_HU_MIN)

        # Step 2: Zero-mean, unit-std — matches NormalizeIntensityd(channel_wise=True)
        mean = vol_normed.mean()
        std = vol_normed.std()
        if std > 0:
            vol_normed = (vol_normed - mean) / std

        # ── Diagnostic: input statistics ──
        print(f"\n  [DEBUG] Input volume shape: {volume.shape}")
        print(f"  [DEBUG] HU range: [{volume.min():.1f}, {volume.max():.1f}]")
        print(f"  [DEBUG] After norm: mean={vol_normed.mean():.4f}, "
              f"std={vol_normed.std():.4f}, "
              f"range=[{vol_normed.min():.4f}, {vol_normed.max():.4f}]")

        tensor = torch.from_numpy(vol_normed).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,D,H,W)

        with torch.no_grad():
            output = self.inferer(tensor, self.unet)

        # ── Diagnostic: raw model output ──
        raw_logits = output.squeeze().cpu().numpy()
        print(f"  [DEBUG] Raw logit range: [{raw_logits.min():.4f}, {raw_logits.max():.4f}]")
        print(f"  [DEBUG] Raw logit mean: {raw_logits.mean():.4f}, std: {raw_logits.std():.4f}")

        probs = torch.sigmoid(output).squeeze().cpu().numpy()

        # ── Diagnostic: probability distribution ──
        print(f"  [DEBUG] Sigmoid prob range: [{probs.min():.6f}, {probs.max():.6f}]")
        print(f"  [DEBUG] Sigmoid prob mean: {probs.mean():.6f}")
        for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
            pct = 100 * (probs > t).mean()
            print(f"  [DEBUG]   voxels > {t}: {pct:.4f}%")

        mask = (probs > self.seg_threshold).astype(np.bool_)

        print(f"  [DEBUG] Seg threshold: {self.seg_threshold}")
        print(f"  [DEBUG] Mask positive voxels: {mask.sum()} / {mask.size} "
              f"({100 * mask.mean():.4f}%)")

        logger.debug("Segmentation mask: %d positive voxels (%.2f%%)",
                      mask.sum(), 100 * mask.mean())
        return mask

    # ── Candidate extraction ──────────────────

    def _extract_candidates(
        self, mask: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Extract 3D connected components from the segmentation mask.

        Returns list of dicts with 'label', 'centroid', 'volume' keys.
        """
        labeled_array, n_features = scipy_label(mask)
        candidates = []

        for i in range(1, n_features + 1):
            component = (labeled_array == i)
            vol = int(component.sum())

            if vol < self.min_candidate_voxels:
                continue

            coords = np.argwhere(component)
            centroid = tuple(coords.mean(axis=0).astype(int))
            z_min, y_min, x_min = coords.min(axis=0)
            z_max, y_max, x_max = coords.max(axis=0)
            bbox = ((int(z_min), int(z_max)), (int(y_min), int(y_max)), (int(x_min), int(x_max)))

            candidates.append({
                "label": i,
                "centroid": centroid,
                "bbox": bbox,
                "volume": vol,
            })

        # Sort by volume (largest first)
        candidates.sort(key=lambda c: c["volume"], reverse=True)
        return candidates

    # ── Candidate classification ──────────────

    def _preprocess_crop(self, crop: np.ndarray) -> torch.Tensor:
        """
        Apply training-matched preprocessing to a 64³ HU crop.
        Returns (1, 1, 64, 64, 64) tensor on self.device.
        """
        # Step 1: ScaleIntensityRange
        crop_windowed = np.clip(crop, LUNG_HU_MIN, LUNG_HU_MAX)
        crop_normed = ((crop_windowed - LUNG_HU_MIN) /
                       (LUNG_HU_MAX - LUNG_HU_MIN))

        # Step 2: NormalizeIntensity (zero-mean, unit-std)
        mean = crop_normed.mean()
        std = crop_normed.std()
        if std > 0:
            crop_normed = (crop_normed - mean) / std

        tensor = torch.from_numpy(crop_normed).float()
        return tensor.unsqueeze(0).unsqueeze(0).to(self.device)

    def _classify_candidates(
        self,
        volume: np.ndarray,
        candidates: List[Dict],
        threshold: float,
        min_prob: float = 0.0,
    ) -> List[CandidateResult]:
        """
        Crop 64³ around each candidate centroid and classify with ResNet ONNX.
        """
        results = []

        for idx, cand in enumerate(candidates):
            centroid = cand["centroid"]
            bbox = cand["bbox"]

            # Extract and preprocess 64³ crop
            crop = self._extract_crop(volume, centroid)
            tensor = self._preprocess_crop(crop) # (1, 1, 64, 64, 64) torch.Tensor

            # Classify via ONNX
            ort_inputs = {self.resnet_input_name: tensor.cpu().numpy()}
            ort_outs = self.resnet_session.run(None, ort_inputs)
            logit = ort_outs[0][0][0] # it's (1, 1)
            
            # Apply sigmoid
            prob = 1.0 / (1.0 + np.exp(-logit))

            if prob < min_prob:
                continue

            prediction = "Malignant" if prob > threshold else "Benign"

            results.append(CandidateResult(
                candidate_index=idx,
                centroid=centroid,
                bbox=bbox,
                volume_voxels=cand["volume"],
                probability=float(prob),
                prediction=prediction,
            ))

        # Sort by probability (most suspicious first)
        results.sort(key=lambda r: r.probability, reverse=True)

        # Re-index after sorting
        for i, r in enumerate(results):
            r.candidate_index = i

        return results

    def _extract_crop(self, volume: np.ndarray,
                      center: Tuple[int, int, int]) -> np.ndarray:
        """Extract a 64³ crop centred at the given coordinates."""
        half = CROP_SIZE // 2
        crop = np.zeros((CROP_SIZE, CROP_SIZE, CROP_SIZE), dtype=np.float32)

        for dim in range(3):
            if center[dim] < 0 or center[dim] >= volume.shape[dim]:
                return crop  # out of bounds → return zeros

        slices_src = []
        slices_dst = []
        for dim in range(3):
            s = max(0, center[dim] - half)
            e = min(volume.shape[dim], center[dim] + half)
            ds = s - (center[dim] - half)
            de = CROP_SIZE - ((center[dim] + half) - e)
            slices_src.append(slice(s, e))
            slices_dst.append(slice(ds, de))

        crop[slices_dst[0], slices_dst[1], slices_dst[2]] = \
            volume[slices_src[0], slices_src[1], slices_src[2]]

        return crop

    # ── Aggregation ───────────────────────────

    def _aggregate(
        self,
        probabilities: List[float],
        strategy: str = "top_k",
        k: int = 5,
    ) -> Tuple[float, str]:
        """Combine per-candidate probabilities into a patient-level score."""
        probs = np.array(probabilities, dtype=np.float32)

        if strategy == "max":
            return float(probs.max()), "max"
        elif strategy == "mean":
            return float(probs.mean()), "mean"
        elif strategy == "top_k":
            k_actual = min(k, len(probs))
            top = np.sort(probs)[::-1][:k_actual]
            return float(top.mean()), f"top_{k_actual}_mean"
        else:
            raise ValueError(f"Unknown aggregation: {strategy}")

    def _confidence_tier(self, prob: float, threshold: float) -> str:
        distance = abs(prob - threshold)
        if distance > 0.3:
            return "High"
        elif distance > 0.15:
            return "Medium"
        return "Low"

    # ── Batch inference ───────────────────────

    def run_batch(
        self,
        folder_paths: List[str],
        **kwargs,
    ) -> List[VolumeResult3D]:
        """Process multiple patients sequentially."""
        results = []
        for i, folder in enumerate(folder_paths):
            logger.info("Batch %d/%d: %s", i+1, len(folder_paths),
                        os.path.basename(folder))
            try:
                result = self.run_volume(folder, **kwargs)
                results.append(result)
            except Exception as exc:
                logger.error("Failed on %s: %s", folder, exc)
        return results


# ──────────────────────────────────────────────
#  CLI entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="3D inference pipeline")
    p.add_argument("--dicom_dir", required=True,
                   help="DICOM series folder to process")
    p.add_argument("--unet_ckpt", default="checkpoints/unet3d_best.pth")
    p.add_argument("--resnet_ckpt", default="checkpoints/resnet3d_calibrated.pth")
    p.add_argument("--aggregation", default="top_k",
                   choices=["max", "mean", "top_k"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--no_gradcam", action="store_true",
                   help="Skip Grad-CAM generation")
    p.add_argument("--gradcam_top_k", type=int, default=3,
                   help="Generate Grad-CAM for top-k candidates")
    p.add_argument("--output_dir", default="output",
                   help="Directory to save Grad-CAM overlays")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)

    pipeline = InferencePipeline3D(
        unet_checkpoint=args.unet_ckpt,
        resnet_checkpoint=args.resnet_ckpt,
    )

    result = pipeline.run_volume(
        args.dicom_dir,
        aggregation=args.aggregation,
        k=args.k,
        generate_gradcam=not args.no_gradcam,
        gradcam_top_k=args.gradcam_top_k,
    )
    print(result.summary())

    # Save Grad-CAM overlays if generated
    saved = result.save_gradcam_overlays(args.output_dir)
    if saved:
        print(f"\n  Saved {len(saved)} Grad-CAM overlays → {args.output_dir}/")
    else:
        print("\n  No Grad-CAM overlays to save (no candidates or --no_gradcam).")
