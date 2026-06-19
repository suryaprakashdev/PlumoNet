# Lung Cancer Nodule Classification — LIDC-IDRI (3D Pipeline)

An end-to-end 3D pipeline for lung nodule detection, segmentation, and malignancy
classification on LIDC-IDRI CT scans. Everything runs on full 3D volumes —
no 2D slice extraction. A **3D U-Net** segments candidate nodules, a
**3D ResNet-10** classifies each one as benign or malignant, and **3D Grad-CAM**
gives volumetric heatmaps for interpretability.

The trained models are deployed via **BentoML on Azure ML**, with the dataset
and checkpoints pulled directly from **Azure Blob Storage** rather than a local
copy of the data.

---

## Pipeline

```
DICOM in Azure Blob Storage
        │
        ▼
Stage 1 — Preprocessing
DICOM → 1mm isotropic → 64³ crops + masks (.npy)
        │
   ┌────┴────┐
   ▼         ▼
Stage 2    Stage 3
3D U-Net   3D ResNet-10
(Dice)     (AUC)
   │         │
   └────┬────┘
        ▼
Stage 4 — 3D Inference
Sliding window seg → candidates → classify → aggregate → Grad-CAM
        │
        ▼
Stage 5 — Serving
BentoML bento → Azure ML managed endpoint
```

## Project structure

```
lung_cancer_project/
├── preprocessing.py        # Stage 1: Blob Storage DICOM → 64³ .npy crops + masks
├── monai_dataset_3d.py     # MONAI CacheDataset / PersistentDataset loaders
├── unet3d.py                # 3D U-Net (segmentation)
├── resnet3d.py               # 3D ResNet-10 (classification)
├── train_unet.py            # Stage 2: U-Net training loop
├── train_classifier.py      # Stage 3: ResNet-10 training loop
├── evaluation.py            # FROC, AUC, Dice, ECE on the held-out test set
├── inference_3d.py          # Stage 4: full 3D inference pipeline
├── service.py                # BentoML service definition (Stage 5)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

NumPy is pinned to `1.26.4` — `pylidc` still relies on aliases (`np.int`,
`np.float`, `np.bool`) that were removed in NumPy 2.x.

---

## Data: Azure Blob Storage

Raw LIDC-IDRI DICOMs and all generated artifacts (`.npy` crops, checkpoints,
results) live in Blob Storage instead of on local disk. Set the connection
once:

```bash
export AZURE_STORAGE_CONNECTION_STRING="<your-connection-string>"
# or, if using a SAS token / managed identity, set AZURE_STORAGE_ACCOUNT + AZURE_STORAGE_SAS_TOKEN
```

Sync the DICOM container locally before running Stage 1 (swap in your own
storage account / container names):

```bash
az storage blob download-batch \
    --account-name <storage-account> \
    --source lidc-idri-raw \
    --destination /data/LIDC-IDRI
```

`preprocessing.py` also accepts a `--blob_container` flag to stream DICOMs
directly from Blob Storage via the `azure-storage-blob` SDK, writing the
processed `.npy` crops back to a `processed` container instead of local disk —
useful when running Stage 1 on an Azure ML compute cluster with no persistent
disk attached.

```bash
python preprocessing.py \
    --blob_container lidc-idri-raw \
    --out_container processed \
    --workers 4
```

**Output layout (in the `processed` container):**

```
volumes/
├── LIDC-IDRI-0001/
│   ├── Benign_0/...vol.npy / ...mask.npy
│   └── Malignant_1/...vol.npy / ...mask.npy
patient_splits.json
dataset_summary.json
checkpoint.txt
```

Key preprocessing steps: resampling to 1×1×1 mm isotropic spacing, MONAI
`SpatialCrop` + `ResizeWithPadOrCrop` to 64³, consensus masks from the union of
radiologist annotations, and exclusion of ambiguous nodules (avg malignancy ==
3.0) or nodules with fewer than 2 annotations.

---

## Training

**Stage 2 — 3D U-Net (segmentation)**, DiceCELoss:

```bash
python train_unet.py --blob_container processed --save_dir /data/checkpoints \
    --epochs 100 --batch_size 8
```

**Stage 3 — 3D ResNet-10 (classification)**, BCEWithLogitsLoss + pos_weight,
followed by temperature calibration:

```bash
python train_classifier.py --blob_container processed --save_dir /data/checkpoints \
    --epochs 50 --batch_size 16
```

Both scripts share the usual knobs: `--patience` for early stopping (on val
Dice / AUC), `--lr`, `--cache_rate` / `--cache_dir` for MONAI caching, and
`--no_amp` to disable mixed precision. Checkpoints (`*_best.pth`, `*_last.pth`,
`*_history.json`) are written to `--save_dir`, which can also point at a Blob
Storage–backed mount.

---

## Inference (Stage 4)

```python
from inference_3d import InferencePipeline3D

pipeline = InferencePipeline3D(
    unet_checkpoint="checkpoints/unet3d_best.pth",
    resnet_checkpoint="checkpoints/resnet3d_calibrated.pth",
)

result = pipeline.run_volume(
    "/path/to/dicom_series_folder",
    aggregation="top_k", k=5,
    generate_gradcam=True, gradcam_top_k=3,
)
print(result.summary())
result.save_gradcam_overlays("output/")
```

Steps: sliding-window U-Net segmentation → 3D connected components for
candidates → ResNet-10 classifies each 64³ crop → patient-level aggregation
(max / mean / top-k) → Grad-CAM on the most suspicious candidates, saved as
axial/coronal/sagittal PNG overlays.

---

## Deployment: BentoML on Azure ML

The calibrated ResNet-10 + U-Net pair is packaged as a BentoML service
(`service.py`) and deployed as an Azure ML managed online endpoint:

```bash
bentoml build
bentoml containerize lung_nodule_service:latest

az ml online-endpoint create -f endpoint.yml
az ml online-deployment create -f deployment.yml --all-traffic
```

The deployment config points the model and checkpoint paths at the same Blob
Storage containers used for training data, so there's a single source of
truth between training and serving. Update `endpoint.yml` / `deployment.yml`
with your workspace, resource group, and compute SKU.

---

## Evaluation

```bash
python evaluation.py --blob_container processed --ckpt_dir /data/checkpoints \
    --out_dir /data/results
```

| Metric | Target | Notes |
|---|---|---|
| AUC-ROC | ≥ 0.92 | patient-level discrimination |
| Sensitivity | ≥ 0.90 | no missed cancers |
| Dice | ≥ 0.75 | U-Net segmentation quality |
| ECE | < 0.05 | calibration error |

Writes `evaluation_metrics.json` plus ROC, FROC, Dice-distribution, and
calibration plots to `--out_dir`.

---

## Models

**3D U-Net** — 4-level encoder/decoder, channels 16→32→64→128 (bottleneck
256), input/output `(B, 1, 64, 64, 64)`, DiceCELoss, dropout 0.5 at the
bottleneck.

**3D ResNet-10** — Conv3d 7×7×7 stem + 4 ResBlock3D stages, channels
16→32→64→128 (~1.8M params), GAP → FC(128→64) → Dropout(0.5) → FC(64→1),
Grad-CAM target `model.layer4`, with learned temperature calibration on the
validation set.

---

## Design notes

- Full 3D volumes (not 2D slices) to preserve spatial context.
- 64³ isotropic crops at 1mm spacing standardize input size across scanners.
- DiceCELoss and BCEWithLogitsLoss + pos_weight handle class imbalance in
  voxel labels and benign/malignant labels respectively.
- Patient-level (not nodule-level) train/val/test splits avoid data leakage.
- SlidingWindowInferer + Gaussian blending handles arbitrary volume sizes at
  inference time.
- AMP + cosine LR annealing for training speed and stability on A100s.
- Scipy resampling instead of MONAI's `Spacing` transform, which has had
  compatibility issues on MONAI ≥ 1.3.
- Preprocessing is crash-safe: a checkpoint file tracks completed patients so
  a failed run can resume without reprocessing everything.

---

## Citation

> Armato III, S.G., et al. "The Lung Image Database Consortium (LIDC) and
> Image Database Resource Initiative (IDRI)." *Medical Physics* 38.2 (2011):
> 915-931.
