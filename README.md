# Lung Cancer Nodule Classification — LIDC-IDRI (3D Pipeline)

End-to-end **3D volumetric** pipeline for lung nodule detection, segmentation, and
malignancy classification on the LIDC-IDRI CT dataset.

The pipeline operates directly on full 3D CT volumes — no 2D slice extraction required.
A **3D U-Net** segments candidate nodules from isotropic volumes, and a
**3D ResNet-10** classifies each candidate as benign or malignant, with
**3D Grad-CAM** heatmaps for interpretability.

---

## Architecture Overview

```
DICOM Series
    │
    ▼
┌──────────────────────────────────────┐
│  Stage 1 — Preprocessing            │
│  DICOM → 1mm isotropic resampling   │
│  → 64³ nodule crops + masks (.npy)  │
└──────────────────┬───────────────────┘
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
┌────────────────┐   ┌─────────────────┐
│ Stage 2        │   │ Stage 3         │
│ 3D U-Net       │   │ 3D ResNet-10    │
│ Segmentation   │   │ Classification  │
│ DiceCELoss     │   │ BCEWithLogits   │
│ Dice ≥ 0.75   │   │ AUC ≥ 0.92     │
└────────┬───────┘   └────────┬────────┘
         │                    │
         └────────┬───────────┘
                  ▼
┌──────────────────────────────────────┐
│  Stage 4 — 3D Inference             │
│  SlidingWindowInferer → candidates  │
│  → ResNet classify → aggregate      │
│  → patient-level prediction         │
│  → 3D Grad-CAM heatmaps            │
└──────────────────────────────────────┘
```

---

## Project Structure

```
lung_cancer_project/
├── preprocessing.py        # Stage 1: DICOM → 64³ isotropic .npy crops + masks
├── monai_dataset_3d.py     # MONAI CacheDataset / PersistentDataset loaders
├── unet3d.py               # 3D U-Net model (segmentation)
├── resnet3d.py              # 3D ResNet-10 model (classification)
├── train_unet.py           # Stage 2: U-Net training loop
├── train_classifier.py     # Stage 3: ResNet-10 training loop
├── evaluation.py           # Medical imaging evaluation (FROC, AUC, Dice, ECE)
├── inference_3d.py         # Stage 4: Full 3D inference pipeline
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **NumPy version**: `requirements.txt` pins NumPy to `1.26.4` because
> `pylidc` uses deprecated aliases (`np.int`, `np.float`, `np.bool`)
> removed in NumPy 2.x.

---

## Data

1. Download the LIDC-IDRI dataset from
   [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/lidc-idri/).
2. Place all DICOM files under a single root directory,
   e.g. `/data/LIDC-IDRI/`.

---

## Run Order

### Stage 1 — 3D Preprocessing

Converts raw DICOMs into **64×64×64 isotropic 1 mm³ nodule crops** (float32 `.npy`)
with corresponding binary segmentation masks. Supports crash-safe resumption
via a checkpoint file and multiprocessing with `ProcessPoolExecutor`.

```bash
python preprocessing.py \
    --raw_dir /data/LIDC-IDRI \
    --out_dir /data/processed_3d \
    --workers 4
```

| Flag        | Default         | Description                               |
|-------------|-----------------|-------------------------------------------|
| `--raw_dir` | —               | Root of raw LIDC-IDRI DICOM data          |
| `--out_dir` | —               | Output directory for `.npy` crops         |
| `--workers` | `cpu_count - 1` | Number of parallel worker processes       |

**Output layout:**

```
/data/processed_3d/
├── volumes/
│   ├── LIDC-IDRI-0001/
│   │   ├── Benign_0/
│   │   │   ├── LIDC-IDRI-0001_nodule0_malig2.0_vol.npy    ← (64,64,64) HU crop
│   │   │   └── LIDC-IDRI-0001_nodule0_malig2.0_mask.npy   ← (64,64,64) binary mask
│   │   └── Malignant_1/
│   │       ├── LIDC-IDRI-0001_nodule1_malig4.5_vol.npy
│   │       └── LIDC-IDRI-0001_nodule1_malig4.5_mask.npy
│   └── LIDC-IDRI-0002/
│       └── ...
├── patient_splits.json     ← patient-level train/val/test splits
├── dataset_summary.json    ← per-patient stats, class balance, spacing info
└── checkpoint.txt          ← processed patient IDs (resume support)
```

**Key preprocessing steps:**

- Scipy-based resampling to 1×1×1 mm isotropic spacing
- MONAI `SpatialCrop` + `ResizeWithPadOrCrop` for exact 64³ output
- Union of multi-radiologist annotations for consensus masks
- Ambiguous nodules (average malignancy == 3.0) excluded
- Nodules with < 2 radiologist annotations excluded
- Robust DICOM loader handles flat, nested, and extensionless file structures

---

### Stage 2 — 3D U-Net Training (Segmentation)

Trains a 3D U-Net for voxel-level nodule segmentation using **DiceCELoss** (Dice + CrossEntropy).

```bash
python train_unet.py \
    --data_dir  /data/processed_3d \
    --save_dir  /data/checkpoints \
    --epochs    100 \
    --batch_size 8
```

| Flag           | Default        | Description                                        |
|----------------|----------------|----------------------------------------------------|
| `--data_dir`   | —              | Output of Stage 1                                  |
| `--save_dir`   | `checkpoints/` | Where to save `.pth` files                         |
| `--epochs`     | `100`          | Max training epochs                                |
| `--patience`   | `15`           | Early stopping patience (val Dice)                 |
| `--batch_size` | `8`            | Batch size (8 fits on A100 40 GB)                   |
| `--lr`         | `1e-3`         | Learning rate                                      |
| `--cache_rate` | `1.0`          | MONAI CacheDataset rate (1.0 = all in RAM)         |
| `--cache_dir`  | `None`         | PersistentDataset cache dir (survives restarts)    |
| `--no_amp`     | off            | Disable mixed precision (AMP) training             |

**Outputs:**

```
checkpoints/
├── unet3d_best.pth         ← best validation Dice
├── unet3d_last.pth
└── unet3d_history.json
```

---

### Stage 3 — 3D ResNet-10 Training (Classification)

Trains a 3D ResNet-10 for binary malignancy classification with
**BCEWithLogitsLoss + pos_weight**, followed by **temperature calibration**.

```bash
python train_classifier.py \
    --data_dir  /data/processed_3d \
    --save_dir  /data/checkpoints \
    --epochs    50 \
    --batch_size 16
```

| Flag           | Default        | Description                                       |
|----------------|----------------|---------------------------------------------------|
| `--data_dir`   | —              | Output of Stage 1                                 |
| `--save_dir`   | `checkpoints/` | Where to save `.pth` files                        |
| `--epochs`     | `50`           | Max training epochs                               |
| `--patience`   | `10`           | Early stopping patience (val AUC)                 |
| `--batch_size` | `16`           | Batch size                                        |
| `--lr`         | `1e-4`         | Learning rate                                     |
| `--cache_rate` | `1.0`          | MONAI CacheDataset rate                           |
| `--cache_dir`  | `None`         | PersistentDataset cache dir                       |
| `--no_amp`     | off            | Disable mixed precision                           |

**Outputs:**

```
checkpoints/
├── resnet3d_best.pth             ← best validation AUC
├── resnet3d_last.pth
├── resnet3d_calibrated.pth       ← temperature-calibrated model
├── resnet3d_history.json
└── resnet3d_test_metrics.json
```

---

### Stage 4 — 3D Inference

Full end-to-end inference on unseen DICOM CT series:

1. **Sliding window segmentation** — MONAI `SlidingWindowInferer` runs the U-Net on the full resampled volume
2. **Candidate extraction** — thresholds the probability mask and finds 3D connected components
3. **Candidate classification** — crops 64³ around each candidate centroid and classifies with ResNet-10
4. **Patient-level aggregation** — combines per-candidate probabilities (max / mean / top-k)
5. **3D Grad-CAM** — generates volumetric heatmaps for the most suspicious candidates

```python
from inference_3d import InferencePipeline3D

pipeline = InferencePipeline3D(
    unet_checkpoint="checkpoints/unet3d_best.pth",
    resnet_checkpoint="checkpoints/resnet3d_calibrated.pth",
)

result = pipeline.run_volume(
    "/path/to/dicom_series_folder",
    aggregation="top_k",
    k=5,
    generate_gradcam=True,
    gradcam_top_k=3,
)

print(result.summary())
result.save_gradcam_overlays("output/")
```

**Grad-CAM output** includes multi-plane visualisations (axial, coronal, sagittal)
for each candidate nodule, saved as PNG overlays.

---

### Evaluation

Runs all medical imaging evaluation metrics on the held-out test set:

```bash
python evaluation.py \
    --data_dir  /data/processed_3d \
    --ckpt_dir  /data/checkpoints \
    --out_dir   /data/results
```

| Metric          | Target    | Description                                  |
|-----------------|-----------|----------------------------------------------|
| **FROC**        | —         | Sensitivity at various FP/scan rates         |
| **AUC-ROC**     | ≥ 0.92    | Patient-level discrimination                 |
| **Sensitivity** | ≥ 0.90    | No missed cancers                            |
| **Dice**        | ≥ 0.75    | Segmentation quality (U-Net)                 |
| **ECE**         | < 0.05    | Expected Calibration Error                   |

**Outputs:**

```
results/
├── evaluation_metrics.json
├── roc_curve_3d.png
├── froc_curve.png
├── dice_distribution.png
└── calibration_plot.png
```

---

## Models

### 3D U-Net (Segmentation)

| Property          | Value                                            |
|-------------------|--------------------------------------------------|
| Architecture      | 4-level encoder–decoder with skip connections    |
| Channels          | 16 → 32 → 64 → 128, bottleneck 256              |
| Input             | `(B, 1, 64, 64, 64)` — single-channel HU volume |
| Output            | `(B, 1, 64, 64, 64)` — voxel probability mask   |
| Loss              | DiceCELoss (Dice + CrossEntropy)                 |
| Dropout           | 0.5 (bottleneck only)                            |
| Grad-CAM target   | N/A (segmentation model)                         |

### 3D ResNet-10 (Classification)

| Property          | Value                                            |
|-------------------|--------------------------------------------------|
| Architecture      | Conv3d 7×7×7 stem + 4 ResBlock3D stages          |
| Channels          | 16 → 32 → 64 → 128                              |
| Parameters        | ~1.8M                                            |
| Input             | `(B, 1, 64, 64, 64)` — single-channel HU volume |
| Output            | `(B, 1)` — raw logit (sigmoid applied externally)|
| Head              | GAP → FC(128→64) → Dropout(0.5) → FC(64→1)      |
| Temperature       | Learned calibration on validation set             |
| Grad-CAM target   | `model.layer4`                                   |

---

## Data Pipeline (MONAI)

The dataset module (`monai_dataset_3d.py`) provides high-performance data loading:

| Feature                  | Description                                            |
|--------------------------|--------------------------------------------------------|
| **CacheDataset**         | Caches deterministic transforms in RAM (up to `cache_rate`) |
| **PersistentDataset**    | Disk-backed cache for large datasets (survives restarts)    |
| **ThreadDataLoader**     | Multi-threaded loader for cached val/test data              |
| **Lazy file-path loading** | Stores paths, not arrays — loaded on-the-fly by `Lambdad`  |
| **Patient-level splits** | Train/val/test split by patient ID (no data leakage)        |

**Preprocessing transforms:**

```
HU volume → ScaleIntensityRange [-1350, 150] → [0, 1]
          → 3D augmentation (flip, rotate90, noise, scale, shift)
          → NormalizeIntensity (zero-mean, unit-std)
          → float32 tensor
```

---

## Key Design Choices

| Choice                              | Rationale                                                           |
|--------------------------------------|---------------------------------------------------------------------|
| Full 3D volumetric pipeline          | Preserves spatial context lost in 2D slice extraction               |
| 64³ isotropic crops at 1 mm spacing  | Standardised input size across variable patient spacings            |
| DiceCELoss for segmentation          | Combined Dice + CE handles class imbalance in voxel labels          |
| BCEWithLogitsLoss + pos_weight       | Handles benign/malignant class imbalance                            |
| WeightedRandomSampler                | Balances mini-batches without oversampling the full dataset         |
| Skip avg_malignancy == 3.0           | Ambiguous radiologist consensus excluded from labels                |
| Temperature calibration              | Post-hoc calibration improves probability reliability (ECE < 0.05)  |
| SlidingWindowInferer                 | Handles arbitrary volume sizes at inference with Gaussian blending  |
| 3D connected components             | Extracts individual nodule candidates from continuous seg mask      |
| Top-k aggregation                    | Patient-level score from most suspicious candidates                 |
| AMP (mixed precision)               | 2× training speedup on A100 GPUs                                   |
| Cosine LR annealing                 | Smooth decay avoids sharp LR drops during training                  |
| Early stopping on AUC / Dice        | Task-appropriate stopping criteria for each model                   |
| Scipy resampling                    | Avoids MONAI Spacing API incompatibility in MONAI ≥ 1.3            |
| Multiprocessing preprocessing       | ProcessPoolExecutor for parallel patient processing                 |
| Crash-safe checkpointing            | Resume after failure — completed patients are skipped               |

---

## Google Colab (A100)

The pipeline is optimised for Colab Pro with A100 GPUs:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Stage 1 — Preprocessing
!python preprocessing.py \
    --raw_dir /content/drive/MyDrive/LIDC-IDRI \
    --out_dir /content/drive/MyDrive/LIDC-3D-Processed \
    --workers 4

# Stage 2 — U-Net training
!python train_unet.py \
    --data_dir /content/drive/MyDrive/LIDC-3D-Processed \
    --save_dir /content/drive/MyDrive/checkpoints \
    --batch_size 8 --epochs 100

# Stage 3 — Classifier training
!python train_classifier.py \
    --data_dir /content/drive/MyDrive/LIDC-3D-Processed \
    --save_dir /content/drive/MyDrive/checkpoints \
    --batch_size 16 --epochs 50

# Stage 4 — Evaluation
!python evaluation.py \
    --data_dir /content/drive/MyDrive/LIDC-3D-Processed \
    --ckpt_dir /content/drive/MyDrive/checkpoints \
    --out_dir  /content/drive/MyDrive/results
```

**A100 optimisations applied automatically:**
- TF32 matmuls enabled
- cuDNN benchmarking enabled
- Mixed precision (AMP) with `GradScaler`
- `pin_memory=True` + `persistent_workers=True`

---

## Citation

> Armato III, S.G., et al. "The Lung Image Database Consortium (LIDC) and Image
> Database Resource Initiative (IDRI)." *Medical Physics* 38.2 (2011): 915-931.
