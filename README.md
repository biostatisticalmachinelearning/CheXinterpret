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
- CheXagent checkpoints are loaded from `features.cache_dir` first; the pipeline only hits Hugging Face if the local cache is missing.

## Repository Layout

- `configs/default.yaml`: main experiment config.
- `configs/quickstart.yaml`: lightweight settings for smoke tests.
- `src/chex_sae_fairness/data/`: manifest creation and dataset utilities.
- `src/chex_sae_fairness/models/`: CheXagent feature extraction + SAE model.
- `src/chex_sae_fairness/training/`: SAE and probe trainers.
- `src/chex_sae_fairness/evaluation/`: disentanglement and fairness metrics.
- `src/chex_sae_fairness/mitigation/`: concept-space debiasing functions.
- `src/chex_sae_fairness/publication/`: core/supplement paper pipelines, tables, and figure builders.
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
- `schema.metadata_cols`: metadata columns from the CSV (CheXpert Plus commonly uses `age`, `sex`, `race`, `ethnicity`, `interpreter_needed`, `insurance_type`, `recent_bmi`, `deceased`).
- `schema.split_col`: can be inferred from `path_to_image` if missing.
- `probes.c_value` / `probes.max_iter`: logistic probe hyperparameters used for pathology classifiers.
- `fairness.debias_mode`: choose where age-concept residualization is applied (`train_and_test`, `test_only`, or `train_only`).
- `features.cache_dir`: local HuggingFace cache directory for CheXagent model/processor weights (downloaded once, then reused).

CheXpert Plus PNG layout (official examples) uses relative image paths like:

- `train/patientXXXX/studyY/view1_frontal.jpg`
- `valid/patientXXXX/studyY/view1_frontal.jpg`

You only need one image modality for this pipeline. If using PNG:

- extract `png_chexpert_plus_chunk_*.zip`
- set `paths.image_root` to a directory containing either `train/` and `valid/`, or `PNG/train/` and `PNG/valid/`
- `val/` is accepted and normalized to `valid`
- extracted chunk directories like `png_chexpert_plus_chunk_0/PNG/...` are also scanned automatically

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

By default, `chex-run-study` now runs a **comprehensive multi-SAE study**:

1. Runs an SAE sweep (L1 + top-k variants / hyperparameters).
2. Selects the best SAE run by a composite objective over reconstruction, disentanglement, correlation, and fairness-aware probe metrics.
3. Runs full baseline vs SAE vs debiased fairness analysis with the selected SAE.
4. Exports publication-ready figures.

Every run is saved into a timestamped directory so no results are overwritten:

- `<output_root>/runs/YYYYMMDD_HHMMSS/`
- key artifacts: `run_summary.json`, `configs/`, `sae_sweep/`, `workspace/study_metrics.json`, `figures/`

## Publication Pipelines

This repository now provides two dedicated, timestamped paper pipelines:

1. **Core pipeline** (`chex-run-core-paper`)
2. **Supplement pipeline** (`chex-run-supplement-paper`)

Generate a publication config template:

```bash
chex-init-paper-config --output configs/publication.yaml
```

Run core figures/tables:

```bash
chex-run-core-paper --config configs/default.yaml --publication-config configs/publication.yaml
```

Run supplementary figures/tables:

```bash
chex-run-supplement-paper --config configs/default.yaml --publication-config configs/publication.yaml
```

Outputs are written to timestamped folders:

- Core: `<output_root>/publication/core/YYYYMMDD_HHMMSS/`
- Supplement: `<output_root>/publication/supplement/YYYYMMDD_HHMMSS/`

Each pipeline writes:

- `tables/*.csv` and `tables/*.md`
- `figures/*.png`
- `<pipeline>_pipeline_summary.json`
- `reproducibility_appendix.json`

Publication-ready figures are saved under:

- `figures/sweep/`: hyperparameter ranking, disentanglement-vs-correlation, fairness-performance tradeoff, metric scorecard
- `figures/best_model/`: baseline vs SAE vs debiased performance/fairness, calibration, fairness-performance Pareto, per-group results, top age-associated latents, SAE training curve

Run a single SAE only (legacy behavior):

```bash
chex-run-study --config configs/default.yaml --single-sae
```

Live progress is logged to console and to a file by default:

- default file (comprehensive mode): `<output_root>/runs/<timestamp>/logs/run_study.log`
- default file (single-SAE mode): `<output_root>/logs/run_study.log`
- control verbosity with `--log-level` (e.g. `INFO`, `DEBUG`)
- override file path with `--log-file /path/to/run.log`

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
- `study_predictions.npz`: held-out labels/groups plus baseline/SAE/debiased score matrices for statistical and figure generation.

Classifier performance now includes:

- `macro_auroc`, `macro_accuracy`, `micro_accuracy`, `macro_brier`, `macro_ece`
- group fairness summaries including `worst_group_macro_auroc` and `worst_group_macro_accuracy`
- paired bootstrap method-comparison tests with multiple-testing corrected p-values (core pipeline)
- concept-level permutation controls with BH/Holm corrections (supplement pipeline)

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
- `.../sae_sweep/plots/worst_group_macro_auroc.png`

## Fairness Correction Strategy

The provided mitigation is **concept-space residualization**:

1. Fit group means of SAE latent activations on training data by age bin.
2. Subtract each group-specific latent shift from samples in that group.
3. Retrain/evaluate pathology probes on debiased concepts.
4. Compare AUROC and equalized-odds gaps before/after.

You can choose how the intervention is applied:

- `fairness.debias_mode: "train_and_test"`: debias both SAE train/test concepts, then train/evaluate on debiased concepts.
- `fairness.debias_mode: "test_only"`: train on original SAE concepts and debias only at test time.
- `fairness.debias_mode: "train_only"`: debias SAE train concepts only, keeping test concepts untouched.

This is intentionally interpretable and auditable: the report also ranks age-associated latent units to support targeted review.

## Notes on CheXagent Integration

CheXagent checkpoints may expose image features differently across versions. The feature extractor tries common APIs (`get_image_features`, `image_embeds`, hidden states) and raises a clear error if adaptation is required for your checkpoint.

## Sanity Check

```bash
python -m compileall src tests scripts
```
