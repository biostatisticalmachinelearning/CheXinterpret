# CheXagent SAE Fairness Study

This project builds a reproducible pipeline for two questions:

1. Can a **Sparse Autoencoder (SAE)** learn disentangled concepts for 14 chest pathologies and 8 patient metadata elements from CheXagent vision representations?
2. If a pathology classifier on CheXagent features underperforms for specific age groups, can SAE concepts support a practical fairness-correction workflow?

## What the Pipeline Does

1. Builds a clean manifest from CheXpert Plus metadata.
2. Extracts fixed vision features with CheXagent.
3. Trains an SAE on train split features.
4. Evaluates concept disentanglement of SAE latents for pathology + metadata targets.
5. Trains baseline pathology probe on raw features and audits age-group fairness.
6. Trains pathology probe on SAE latents.
7. Applies concept-space debiasing (age-group residualization) and re-audits fairness.
8. Saves a full JSON report for comparison.

By default, the pipeline is cache-first for CheXagent embeddings:

- If `features.npz` exists, it is reused.
- New embeddings are extracted only when the cache is missing (or when `features.force_recompute: true`).

## Repository Layout

- `configs/default.yaml`: main experiment config.
- `configs/quickstart.yaml`: lightweight settings for smoke tests.
- `src/chex_sae_fairness/data/`: manifest creation and dataset utilities.
- `src/chex_sae_fairness/models/`: CheXagent feature extraction + SAE model.
- `src/chex_sae_fairness/training/`: SAE and probe trainers.
- `src/chex_sae_fairness/evaluation/`: disentanglement and fairness metrics.
- `src/chex_sae_fairness/mitigation/`: concept-space debiasing functions.
- `src/chex_sae_fairness/pipeline.py`: end-to-end study runner.
- `scripts/`: CLI wrappers.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Data and Config Requirements

Update `configs/default.yaml` with your local paths and schema:

- `paths.image_root`: directory that contains extracted PNG images.
- `paths.metadata_csv`: `df_chexpert_plus_240401.csv`.
- `paths.chexbert_labels_json`: either `chexbert_labels/findings_fixed.json` or `chexbert_labels.zip`.
- `schema.image_path_col`: usually `path_to_image`.
- `schema.pathology_cols`: should match your 14 pathology columns from CheXbert labels.
- `schema.metadata_cols`: metadata columns from the CSV.
- `schema.split_col`: can be inferred from `path_to_image` if missing.

CheXpert Plus PNG layout (official examples) uses relative image paths like:

- `train/patientXXXX/studyY/view1_frontal.jpg`
- `valid/patientXXXX/studyY/view1_frontal.jpg`

You only need one image modality for this pipeline. If using PNG:

- extract `png_chexpert_plus_chunk_*.zip`
- set `paths.image_root` to a directory containing either `train/` and `valid/`, or `PNG/train/` and `PNG/valid/`

When pathology columns are not present in the CSV, the manifest builder will merge them automatically from `paths.chexbert_labels_json` using `path_to_image`.

`uncertain_label_policy` supports:

- `zero`: map uncertain labels (`-1`) to `0`
- `one`: map uncertain labels (`-1`) to `1`
- `ignore`: convert uncertain labels to missing and drop affected rows

## Run

End-to-end:

```bash
chex-run-study --config configs/default.yaml
```

Before first run, audit your metadata/path wiring:

```bash
chex-audit-data --config configs/default.yaml --sample-size 3000
```

Modular execution:

```bash
chex-prepare-manifest --config configs/default.yaml
chex-extract-features --config configs/default.yaml
chex-train-sae --config configs/default.yaml
python scripts/run_study.py --config configs/default.yaml
```

Force-refresh cached embeddings:

```bash
chex-extract-features --config configs/default.yaml --force
```

## Outputs

All outputs are written to `paths.output_root`:

- `manifest.csv`: cleaned dataset manifest.
- `features.npz`: CheXagent feature bundle + labels/metadata.
- `sae.pt`: trained SAE checkpoint.
- `study_metrics.json`: full report with model performance, fairness gaps, disentanglement metrics, and age-associated latent rankings.

## SAE Variants

The trainer supports:

- `variant: "l1"`: standard L1-regularized sparse autoencoder.
- `variant: "topk"`: top-k sparse activation autoencoder (`topk_k` active latents per sample).

Set these in `sae` config.

## Multi-Run Benchmark (L1 vs Top-k)

Use a sweep file (example: `configs/sae_sweep.yaml`) to train many SAE configurations and compare them.

```bash
chex-run-sae-sweep --base-config configs/default.yaml --sweep-config configs/sae_sweep.yaml
```

Sweep outputs are organized by run:

- `.../sae_sweep/<run_name>/run_config.yaml`
- `.../sae_sweep/<run_name>/sae.pt`
- `.../sae_sweep/<run_name>/metrics.json`

Global comparison artifacts:

- `.../sae_sweep/summary.csv`
- `.../sae_sweep/summary.json`
- `.../sae_sweep/plots/reconstruction_mse.png`
- `.../sae_sweep/plots/pathology_correlations.png`
- `.../sae_sweep/plots/recon_vs_correlation.png`

## Fairness Correction Strategy

The provided mitigation is **concept-space residualization**:

1. Fit group means of SAE latent activations on training data by age bin.
2. Subtract each group-specific latent shift from samples in that group.
3. Retrain/evaluate pathology probes on debiased concepts.
4. Compare AUROC and equalized-odds gaps before/after.

This is intentionally interpretable and auditable: the report also ranks age-associated latent units to support targeted review.

## Notes on CheXagent Integration

CheXagent checkpoints may expose image features differently across versions. The feature extractor tries common APIs (`get_image_features`, `image_embeds`, hidden states) and raises a clear error if adaptation is required for your checkpoint.

## Sanity Check

```bash
python -m compileall src tests scripts
```
