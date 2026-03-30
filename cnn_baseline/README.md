# CNN Baseline — CheXpert Plus Multi-Label Classification

A standalone CNN training module for the CheXinterpret fairness auditing project.
Trains and evaluates standard CNN architectures on the CheXpert Plus chest X-ray dataset,
producing fairness metrics in the same format as the SAE-based pipeline to enable direct comparison.

## Contents

- [Overview](#overview)
- [Quickstart](#quickstart)
- [Architectures](#architectures)
- [Training a Single Model](#training-a-single-model)
- [Hyperparameter Sweep](#hyperparameter-sweep)
- [Configuration Reference](#configuration-reference)
- [Outputs](#outputs)
- [Fairness Evaluation](#fairness-evaluation)
- [Connecting to the SAE Pipeline](#connecting-to-the-sae-pipeline)

---

## Overview

This module is kept intentionally separate from the main SAE interpretability pipeline.
Its purpose is to provide **CNN baseline models** that can be:

1. Evaluated for demographic fairness on their own
2. Used as feature extractors for SAE training (future work)
3. Compared against the SAE-based intervention results

The module reuses the same **manifest CSV** produced by the main pipeline (with pre-built
train/valid splits and demographic labels), so no data preprocessing is needed.

---

## Quickstart

```bash
# Activate the shared virtual environment
source .venv/bin/activate

# Train a single model with default settings (DenseNet121)
cnn-train --config cnn_baseline/configs/default.yaml --run-name densenet_default

# Run a hyperparameter sweep across all 4 architectures (16 trials)
cnn-sweep --config cnn_baseline/configs/sweep_quick.yaml --retrain-best
```

Both commands run from the repository root (`/home/little/Repositories/CheXinterpret/`).

---

## Architectures

Four "tried-and-tested" CNN architectures are supported, all loaded with ImageNet-pretrained
weights and fine-tuned end-to-end (no frozen layers):

| Architecture | Params | Notes |
|---|---|---|
| `densenet121` | 6M | Original CheXpert baseline (Irvin et al. 2019); memory-efficient due to dense connections |
| `resnet50` | 23M | Widely used medical imaging backbone; strong general baseline |
| `efficientnet_b4` | 17M | Good performance/efficiency tradeoff; native resolution 380px — constrained to 224px on 7GB GPU |
| `convnext_small` | 49M | Modern CNN with transformer-style design; competitive with ViTs on classification tasks |

All models output raw logits (no sigmoid) for 14 pathology classes. Use `BCEWithLogitsLoss`
with per-class positive weights for training.

### Image caching (important for spinning HDD setups)

CheXpert Plus PNG files are 3–4 MB each. Loading them from a spinning HDD results in
~0.5–1 second per image — making training impractically slow (~15 min/epoch for 134K images).

**Recommended: pre-cache images as 256px JPEGs on an SSD:**

```bash
python scripts/cache_chexpert_images.py \
    --manifest outputs/default/runs/.../manifest.csv \
    --cache-root /path/to/ssd/chexpert256 \
    --size 256
```

This creates a `manifest_cached256.csv` with updated `image_path` columns pointing to the
JPEG cache (~16 KB/image vs 3.5 MB/image → 220× smaller → reads at ~50K img/s from NVMe).
The `sweep_quick.yaml` uses this cached manifest by default.

### GPU memory at 224px / batch size 32 (RTX 2070, 7GB)

| Architecture | Forward pass VRAM | Training VRAM (est.) |
|---|---|---|
| `densenet121` | ~2.1 GB | ~4–5 GB |
| `resnet50` | ~2.1 GB | ~3–4 GB |
| `efficientnet_b4` | ~3.6 GB | ~5–6 GB |
| `convnext_small` | ~3.6 GB | ~5–6 GB |

> **Note:** At 320px, EfficientNet-B4 exceeds 7GB VRAM and will OOM. The sweep config
> is therefore fixed at 224px. For a 320px run, use DenseNet121 or ResNet50 only.

---

## Training a Single Model

```bash
cnn-train --config cnn_baseline/configs/default.yaml \
    --run-name my_run \
    [--arch densenet121|resnet50|efficientnet_b4|convnext_small] \
    [--lr 1e-4] \
    [--batch-size 32] \
    [--epochs 50] \
    [--image-size 320] \
    [--augmentation light|medium|heavy] \
    [--device cuda]
```

CLI flags override the corresponding YAML config values. The run name determines the
output subdirectory: `<output_root>/<run_name>/`.

### Augmentation levels

| Level | Transforms applied |
|---|---|
| `light` | RandomHorizontalFlip, ColorJitter (brightness ±15%, contrast ±15%) |
| `medium` | Above + RandomRotation ±10°, RandomAffine (translation ±5%) |
| `heavy` | Above + ElasticTransform, RandomErasing (20% probability) |

All levels use Resize → CenterCrop → Normalize at validation time (no augmentation).

---

## Hyperparameter Sweep

The sweep uses **Optuna** with the **TPE (Tree-structured Parzen Estimator)** sampler,
which is more sample-efficient than grid or random search for ≥4 hyperparameters.

```bash
cnn-sweep --config cnn_baseline/configs/sweep_quick.yaml \
    [--n-trials 16] \
    [--retrain-best] \
    [--run-name sweep_20260326] \
    [--device cuda]
```

`--retrain-best` automatically retrains the winning configuration from scratch for the
full epoch budget (50 epochs, patience=7) after the sweep completes.

### Search space (`sweep_quick.yaml`)

| Hyperparameter | Search space |
|---|---|
| `architecture` | categorical: all 4 CNN architectures |
| `lr` | log-uniform: [5e-5, 5e-4] |
| `batch_size` | categorical: {16, 32} |
| `weight_decay` | categorical: {1e-6, 1e-5, 1e-4} |
| `augmentation_level` | categorical: {light, medium, heavy} |
| `image_size` | categorical: {224} (fixed for GPU constraints) |

With 16 trials, TPE gets ~10 random exploration trials before fitting its model,
giving it 6 exploitation trials to converge on the best architecture/LR combination.

### Resuming an interrupted sweep

The Optuna study is stored in `<output_root>/sweep.db` (SQLite). Re-running the same
`cnn-sweep` command automatically continues from the last completed trial:

```bash
# Resume after interruption — picks up where it left off
cnn-sweep --config cnn_baseline/configs/sweep_quick.yaml --run-name sweep_20260326
```

---

## Configuration Reference

All settings live in a YAML file loaded into nested dataclasses. Example:

```yaml
seed: 42

paths:
  manifest_csv: "outputs/default/runs/.../manifest.csv"
  image_root: "/path/to/chexpert-plus/PNG/"
  output_root: "cnn_baseline/outputs"

data:
  image_size: 320          # Input resolution (px)
  num_workers: 8           # DataLoader workers
  augmentation_level: "medium"

train:
  architecture: "densenet121"
  pretrained: true         # ImageNet weights
  batch_size: 32
  lr: 1.0e-4
  weight_decay: 1.0e-5
  epochs: 50
  patience: 7              # Early stopping on val macro-AUROC
  warmup_epochs: 3         # Linear LR warmup before cosine decay
  amp: true                # Mixed precision (fp16)
  grad_clip: 1.0           # Max gradient norm
  pos_weight_mode: "auto"  # Per-pathology BCEWithLogitsLoss weights

sweep:
  n_trials: 16
  sampler: "tpe"
  architectures: [densenet121, resnet50, efficientnet_b4, convnext_small]
  lr_low: 5.0e-5
  lr_high: 5.0e-4
  batch_sizes: [16, 32]
  weight_decays: [1.0e-6, 1.0e-5, 1.0e-4]
  augmentation_levels: [light, medium, heavy]
  image_sizes: [224]
```

### Training details

**Loss function:** `BCEWithLogitsLoss` with per-pathology positive weights computed as
`n_negative / n_positive`, clamped to [0.5, 50.0]. This compensates for class imbalance
without oversampling (which is complicated for multi-label settings).

**LR schedule:** Linear warmup for `warmup_epochs`, then cosine annealing to `lr × 0.01`.

**Early stopping:** Monitors validation macro-AUROC. Stops if no improvement for `patience`
consecutive epochs; restores best checkpoint at end of training.

**AMP + gradient clipping:** Mixed precision training with `torch.amp.GradScaler`. Gradients
are unscaled before clipping to operate on true gradient magnitudes (a common correctness issue
in naive AMP implementations).

---

## Outputs

Each run produces the following under `<output_root>/<run_name>/`:

```
<run_name>/
├── best_model.pt               # Model weights + optimizer state + training state
├── train_log.csv               # Per-epoch: train_loss, val_macro_auroc, lr
├── linear_separability.csv     # Per-pathology AUROC (same schema as main project)
├── per_group_auroc.csv         # Per-pathology × attribute × group AUROC (new)
├── tpr_by_group.csv            # Per-pathology × attribute × group TPR at threshold=0.5
├── tpr_disparity.csv           # TPR disparity per (pathology, attribute) pair
├── auroc_per_pathology.png     # AUROC bar chart
├── tpr_disparity_heatmap.png   # Disparity heatmap (pathology × attribute)
├── tpr_by_sex.png              # Per-group TPR bars, by sex
├── tpr_by_age_group.png        # Per-group TPR bars, by age group
├── tpr_by_race.png             # Per-group TPR bars, by race
├── tpr_by_insurance_type.png   # Per-group TPR bars, by insurance type
└── roc_curves/
    └── roc_<Pathology>.png     # Per-pathology ROC, 2×2 subplots by demographic attribute

# Sweep runs additionally produce:
├── sweep.db                    # Optuna SQLite study (for resuming)
├── sweep_results.csv           # All trials: value + all hyperparameter values
├── best_params.yaml            # Best trial hyperparameters
└── trials/
    └── trial_NNNN/             # Per-trial outputs (same structure as a single run)
        └── train_log.csv
```

### Loading a trained model

```python
import torch
from cnn_baseline.models import build_model

ckpt = torch.load("cnn_baseline/outputs/<run_name>/best_model.pt", weights_only=False)
model = build_model("densenet121", num_classes=14, pretrained=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

---

## Fairness Evaluation

Fairness metrics are computed automatically after training, for four demographic attributes:
`sex`, `age_group`, `race`, `insurance_type`.

### Output files

| File | Description |
|---|---|
| `tpr_disparity.csv` | TPR_max − TPR_min per (pathology, attribute) — identical schema to main project |
| `tpr_by_group.csv` | Per-group TPR at threshold=0.5 for all (pathology, attribute, group) triples |
| `per_group_auroc.csv` | Per-group AUROC (threshold-free); new in this module |
| `linear_separability.csv` | Per-pathology macro AUROC — identical schema to main project |

`tpr_disparity.csv` and `linear_separability.csv` use **identical column schemas** to the
main SAE pipeline's `presentation_pipeline.py`, enabling direct comparison.

`per_group_auroc.csv` is a new metric not in the SAE pipeline. Per-group AUROC is
threshold-free and avoids calibration bias, making it more appropriate for neural network
outputs than TPR at a fixed threshold.

### Interpreting disparity

A disparity of 0 means all demographic groups have equal TPR for that pathology.
A disparity of 0.3 means the best-served group has 30 percentage-point higher TPR than
the worst-served group — a clinically significant gap.

Typical values from the SAE baseline on CheXpert Plus:
- Support Devices × insurance_type: 0.578 (highest)
- Race × Edema: 0.342
- Race × Lung Opacity: 0.328
- Age group × Cardiomegaly: 0.171

---

## Connecting to the SAE Pipeline

The eventual goal is to apply the **SAE concept ablation intervention** to CNN embeddings
to assess whether the same demographic concepts are present and removable.

Planned workflow:
1. Train best CNN from sweep (produces `best_model.pt`)
2. Extract penultimate-layer embeddings for all train/valid images
3. Train SAEs on CNN embeddings (reuse `concept_analysis_pipeline.py`)
4. Apply `fairness_intervention.py` using CNN-derived concept scores
5. Compare fairness improvement vs. the CheXagent-based results

The CNN embeddings can be extracted with:

```python
import torch
from cnn_baseline.models import build_model

model = build_model("densenet121", pretrained=False)
ckpt = torch.load("best_model.pt", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])

# Remove classification head to get 1024-dim embeddings
import torch.nn as nn
model.classifier = nn.Identity()
model.eval()

with torch.no_grad():
    embeddings = model(images)  # [B, 1024]
```

This replaces the CheXagent feature extraction step in the main pipeline.
