# CheXinterpret: Methods + Figure/Tables Guide

This repository is a full research pipeline for studying sparse autoencoder (SAE) concepts learned from CheXagent embeddings, with emphasis on clinical concept alignment and fairness across age groups.

It is designed so a reader can:

1. Reproduce the full training/evaluation flow.
2. Understand the mathematical/statistical definitions used in each result.
3. Map every generated table/figure file to its exact meaning.

## 1) Scope and Research Questions

The project addresses two primary questions:

1. **Concept learning**: Can sparse latent features learned from CheXagent image embeddings recover clinically meaningful axes related to 14 pathologies and 8 metadata variables?
2. **Fairness intervention**: If a downstream pathology classifier underperforms in specific age groups, can latent-space intervention reduce group disparities while preserving overall performance?

## 2) Setup

## 2.1 Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Dependencies are managed in `pyproject.toml` and include:

- `torch`, `transformers`, `einops`, `accelerate` (CheXagent loading)
- `numpy`, `pandas`, `scikit-learn`, `scipy`
- `matplotlib`, `seaborn`

## 2.2 Data Inputs

Configure paths in `configs/default.yaml`:

- `paths.image_root`: root of extracted PNG dataset
- `paths.metadata_csv`: CheXpert Plus CSV (`df_chexpert_plus_240401.csv`)
- `paths.chexbert_labels_json`: pathology labels file or zip
- `features.cache_dir`: Hugging Face cache directory (model downloaded once and reused)

Supported PNG layouts include:

- `train/...`, `valid/...`
- `PNG/train/...`, `PNG/valid/...`
- chunked extraction directories (`png_chexpert_plus_chunk_*/...`)
- `val` alias normalized to `valid`

## 2.3 Quick Data Audit

```bash
chex-audit-data --config configs/default.yaml --sample-size 3000
```

Use this before long runs to verify path resolution rate and split discovery.

## 2.4 Clearing Cached Artifacts

Cached outputs (manifest, features, SAE checkpoints, logs, run directories) can be
cleared with:

```bash
python scripts/clear_cache.py --config configs/default.yaml
```

This clears all pipeline outputs but leaves the downloaded model weights intact.
To target specific layers only:

```bash
# Recompute manifest and features (e.g. after adding new data or changing view filter)
python scripts/clear_cache.py --config configs/default.yaml --manifest --features

# Also wipe the CheXagent model weights (~33 GB re-download required)
python scripts/clear_cache.py --config configs/default.yaml --model-weights
```

Pass `-y` to skip the confirmation prompt.

## 3) Run Flow

## 3.1 Quick pipeline test (small data slice)

Before committing to a full run (which takes days for feature extraction), validate
the entire pipeline end-to-end on a small slice of data:

```bash
chex-run-study --config configs/test.yaml
```

`configs/test.yaml` limits each split to 100 images (~300 total), uses a small SAE
(256 latent dims, 5 epochs), and reduces bootstrap samples to 20.  The full pipeline
should complete in a few minutes once model weights are cached.

The subset size is controlled by `data.max_rows_per_split` in the config.  Setting it
to `null` (or omitting it) uses the full dataset.

## 3.3 Main end-to-end run

```bash
chex-run-study --config configs/default.yaml
```

Default behavior is comprehensive:

1. Build/normalize manifest.
2. Reuse cached `features.npz` if present, otherwise extract CheXagent embeddings.
3. Run SAE hyperparameter sweep.
4. Select best SAE by composite criterion.
5. Run full baseline/SAE/debiased evaluation.
6. Export JSON summaries and figure artifacts.

## 3.4 Single-SAE run

```bash
chex-run-study --config configs/default.yaml --single-sae
```

## 3.5 Standalone Analysis Scripts

These three scripts form a sequential fairness-audit pipeline that operates on
a pre-extracted feature bundle (`features.npz`).  Run them in order after
feature extraction has completed (or after `chex-run-study` has produced
`features.npz`).

### Stage 1 — Presentation pipeline (baseline audit)

```bash
python scripts/presentation_pipeline.py --config configs/default.yaml
# Optional flags:
#   --output-dir /path/to/dir   override output location
#   --threshold 0.5             decision threshold for TPR (default 0.5)
#   --max-iter 1000             logistic regression solver iterations
```

**What it does:**

1. Loads the feature bundle from `<output_root>/features.npz`.
2. Fits a `StandardScaler` on training-split embeddings.
3. Trains one binary logistic regression (L2, lbfgs, C=1) **per pathology**
   on the training split.
4. Evaluates each probe on the **validation split** only, reporting:
   - Per-pathology AUROC ("linear separability" of the raw embeddings).
   - Sensitivity (TPR) at the chosen threshold, broken down by sex, age group,
     race, and insurance type.
5. Derives per-(pathology, attribute) TPR disparity = max_group TPR − min_group TPR.

**Outputs** (written to `<output_root>/presentation/`):

| File | Description |
|------|-------------|
| `linear_separability.csv` | AUROC, prevalence, n_positive, n_valid per pathology |
| `tpr_by_group.csv` | TPR per pathology × attribute × group |
| `tpr_disparity.csv` | max−min TPR disparity per pathology × attribute |
| `auroc_per_pathology.png` | Horizontal bar chart; bars annotated with (pos=, n=) |
| `tpr_disparity_heatmap.png` | Heatmap of TPR disparity (pathologies × attributes) |
| `tpr_by_<attribute>.png` | Clustered bar chart per attribute; Overall + per-group bars; n_positive annotated |
| `roc_curves/roc_<pathology>.png` | 2×2 grid of ROC curves — one subplot per attribute, overall (dashed) + per-group coloured curves, AUC/n in legend |

---

### Stage 2 — SAE concept analysis

```bash
python scripts/concept_analysis_pipeline.py --config configs/default.yaml
# Optional flags:
#   --output-dir /path/to/dir   override output location
#   --skip-interventions        stop after grid analysis, skip running interventions
```

**What it does:**

1. Loads the same feature bundle and extracts train-split embeddings.
2. Creates an internal 10 % hold-out from the **training split** (only) for
   SAE early stopping — the true validation split is never touched during training.
3. Trains a **4 × 5 grid of 20 TopK SAEs**:
   - *k* (number of active latents) ∈ {8, 16, 32, 64}
   - *latent_dim* ∈ {256, 512, 1024, 2048, 4096}
4. For each SAE, encodes the **training split** and computes **eta-squared
   (η²)** — the fraction of concept-activation variance explained by a
   grouping variable — for every (latent, target) pair where target is one of
   the 4 demographic attributes or one of the 14 pathologies.
5. Computes **per-pair specificity** for every (latent, attr, pathology):

   ```
   spec(latent, attr, path) = demo_η²_attr(latent) − path_η²_path(latent)
   ```

   A high-specificity concept activates strongly along demographic lines while
   remaining uninformative about pathology — the ideal fairness confound.
6. Selects the best (k, latent_dim) architecture per (attr, pathology) pair
   and saves this to `best_sae_per_pair.csv`.
7. Runs all **56 targeted fairness interventions** (one per (attr × pathology)
   pair) by invoking Stage 3 as a subprocess for each pair.
8. After all 56 pairs complete, generates 14 **summary ROC figures** — one per
   pathology — where each figure has 4 demographic quadrants, each drawn from
   the intervention that specifically targeted that (attr, pathology) pair.

**Outputs** (written to `<output_root>/presentation/sae-eval/`):

| Location | File | Description |
|----------|------|-------------|
| `k<k>_d<dim>/` | `concept_scores.csv` | Full η² + specificity table per latent |
| | `pair_overview.png` | Heatmap: max single-concept specificity per (attr, path) pair |
| | `activation_dist_<attr>.png` | Box plots of top-5 demo concepts: activation by group (pos vs neg) |
| | `activation_dist_pathology_<sae>.png` | Top-5 pathology concepts by path_η²: pos-vs-neg activation distributions |
| | `per_pair/top10_<attr>_<path>.png` | Top-10 concepts for each pair: 3-bar clusters (demo η², path η², specificity) |
| `grid/` | `grid_summary.csv` | One row per SAE: mean specificity per (attr, path) pair |
| | `best_sae_per_pair.csv` | Best (k, latent_dim) for every (attr, path) pair |
| | `best_sae_heatmap.png` | (attr × path) heatmap: best architecture name |
| | `specificity_heatmap.png` | k × dim grid coloured by mean pair-specificity |
| | `scatter_demo_vs_path.png` | Mean demo_η² vs mean path_η² per SAE |
| | `pair_heatmaps/spec_<attr>_<path>.png` | k × dim specificity grid per pair |
| `interventions/` | *(see Stage 3)* | One subdirectory per targeted pair |
| `interventions/summary/` | `roc_summary_<pathology>.png` | 14 summary ROC figures (4-quadrant, per-pair targeted ROC curves) |

---

### Stage 3 — Targeted fairness intervention

Stage 3 is invoked automatically by Stage 2 for all 56 pairs.  It can also be
run standalone for a single (attr, pathology) pair:

```bash
python scripts/fairness_intervention.py \
    --config configs/default.yaml \
    --attr sex \
    --pathology "Pleural Effusion" \
    --threshold 0.02    # fraction of concepts to ablate
```

**What it does:**

1. Loads the feature bundle and the best SAE checkpoint for the target
   (attr, pathology) pair (identified from `best_sae_per_pair.csv`).
2. Ranks all SAE latents by their **specificity score** for the target pair
   (computed using whichever mode was used in Stage 2 — see section 4.12).
3. **Mean-ablates** the top-specificity concepts: replaces their activations
   with the training-set mean for every split (train + validation).
4. Evaluates three conditions on the **validation split**:
   - **Baseline** — original embeddings, original logistic regression.
   - **Ablated** — ablated embeddings, original logistic regression (same
     weights, just fed different inputs).
   - **Retrained** — ablated embeddings, logistic regression retrained from
     scratch on the ablated training-split embeddings.
5. Reports AUROC, TPR, and ROC curves for all three conditions and all four
   sensitive attributes simultaneously.
6. Saves `scores.npz` (y_valid, y_scores) and `attr_arrays.npz` per condition
   so Stage 2 can aggregate them into summary ROC figures without re-running.

**Outputs** (written to `interventions/<attr>_<pathology>_t<threshold>/`):

| Location | File | Description |
|----------|------|-------------|
| `<pair>/` | `ablated_concepts.csv` | Ranked list of ablated latents with η² and specificity |
| | `attr_arrays.npz` | Demographic arrays for the validation split |
| `baseline/` | `linear_separability.csv` | Per-pathology AUROC (baseline condition) |
| | `tpr_by_group.csv` | Per-pathology × attr × group TPR |
| | `scores.npz` | Raw y_valid and y_scores (for post-hoc ROC plots) |
| | `roc_curves/roc_<pathology>.png` | Per-pathology ROC — overall + per-group |
| `ablated/` | *(same structure)* | Ablated condition |
| `retrained/` | *(same structure)* | Retrained condition |
| `vs_baseline/` | `auroc_comparison.png` | AUROC bar comparison: all 3 conditions |
| | `tpr_disparity_comparison.png` | TPR disparity comparison across conditions |
| | `tpr_by_<attr>_comparison.png` | Paired bars: baseline vs ablated/retrained per group |
| | `roc_curves/roc_<pathology>_comparison.png` | 4-quadrant ROC comparison (all attrs) |
| | `roc_curves/roc_<pathology>_focused.png` | Zoomed focused ROC for the target pathology |

**Threshold warnings.** If the `--threshold` value is too high (no concepts
have a specificity score exceeding it), the script logs a warning with the
maximum observed specificity and exits with code 1:

```
WARNING  | No concepts exceed the threshold. Try a lower --threshold value.
         | Max pair specificity in this SAE: 0.0142
```

When Stage 3 is invoked automatically by Stage 2's 56-pair loop, a per-pair
exit code of 1 is tolerated — the runner continues to the next pair — but the
warning appears in the parent process's log.  If you see many such warnings,
lower `--threshold` (the automated loop uses `0.02` by default; try `0.01`).

---

## 3.6 Publication pipelines

```bash
chex-init-paper-config --output configs/publication.yaml
chex-run-core-paper --config configs/default.yaml --publication-config configs/publication.yaml
chex-run-supplement-paper --config configs/default.yaml --publication-config configs/publication.yaml
```

## 4) Methods (Math + ML + Stats)

## 4.1 Notation

- `x`: CheXagent feature vector per image.
- `z`: SAE latent vector.
- `x_hat`: reconstructed feature vector.
- `y`: multi-label pathology target vector.
- `g`: age-group membership.

## 4.2 Data and Labels

- Pathology labels are multi-label binary targets.
- Metadata columns are included for concept analyses.
- Uncertain label policy (`data.uncertain_label_policy`):
  - `zero`: `-1 -> 0`
  - `one`: `-1 -> 1`
  - `ignore`: `-1 -> NaN`, then dropped where required
- Missing pathology NaNs are filled with `0` unless `ignore` policy is used.
- View filtering (`data.allowed_views`): list of view types to retain (e.g. `["frontal"]`).
  View type is inferred from the image file path (`"frontal"` / `"lateral"` / `"unknown"`).
  An empty list (default) keeps all views. The default config restricts to frontal views
  only, consistent with the CheXagent paper's own fairness evaluation methodology.
  Lateral-inclusive runs are reserved for the supplement's `view_sensitivity` analysis.

## 4.3 Embedding Extraction (CheXagent)

The loader uses:

- `AutoProcessor.from_pretrained(..., trust_remote_code=True)`
- `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`

After loading, only the visual encoder is retained: `model = model.vision_model`.

The CheXagent processor's `__call__` stacks all images in a batch and unsqueezes a
leading batch dimension, producing `[1, N, C, H, W]` — treating the entire list as one
multi-image study. For per-image feature extraction we call `processor.image_processor`
directly (the underlying `BlipImageProcessor`), which returns `[B, C, H, W]` with one
row per image, and forward that through `vision_model`.

Feature extraction priority (vision-side only, no text-generation output):

- preferred: `model.get_image_features(**inputs)`
- fallback: vision/image hidden outputs (`last_hidden_state`, `image_embeds`, `vision_embeds`, `hidden_states[-1]`, etc.)

Model/processor artifacts are cached under `features.cache_dir`.

## 4.4 SAE Architectures

Two SAE variants are supported:

1. **L1 SAE**
   - Encoder: `z = ReLU(W_e x + b_e)`
   - Decoder: `x_hat = W_d z + b_d`
   - Loss:
     `L = MSE(x, x_hat) + lambda * mean(|z|)`

2. **Top-k SAE**
   - Same encoder/decoder, but after activation only top-k entries per sample are retained.
   - Loss:
     `L = MSE(x, x_hat)`

## 4.5 Downstream Pathology Probes

Primary classifier is one-vs-rest logistic regression on:

- raw features (`baseline_feature_probe`)
- SAE latents (`sae_concept_probe`)
- debiased SAE latents (`sae_concept_probe_debiased`)

## 4.6 Fairness Intervention: Age Residualization

For each age group `g`:

- `mu_g = mean(z | g)`
- `mu = global mean(z)`

Residualized representation:

- `z' = z - alpha * (mu_g - mu)`

where `alpha = fairness.debias_strength`.

Modes:

- `train_and_test`: residualize both train/test latents
- `test_only`: residualize only test latents
- `train_only`: residualize only train latents

## 4.7 Metrics

## Performance

- `macro_auroc`: mean AUROC across labels with valid class support.
- `macro_accuracy`: per-label thresholded accuracy averaged across labels.
- `micro_accuracy`: global element-wise thresholded accuracy.
- `macro_brier`: per-label Brier score averaged across labels.
- `macro_ece`: per-label ECE averaged across labels (10 bins).

## Group Fairness (age bins)

For each group `g`:

- `AUROC_g`, `ACC_g`, `TPR_g`, `FPR_g`

Aggregates:

- `worst_group_macro_auroc = min_g AUROC_g`
- `worst_group_macro_accuracy = min_g ACC_g`
- `macro_auroc_gap = max_g AUROC_g - min_g AUROC_g`
- `macro_accuracy_gap = max_g ACC_g - min_g ACC_g`
- `equalized_odds_tpr_gap = max_g TPR_g - min_g TPR_g`
- `equalized_odds_fpr_gap = max_g FPR_g - min_g FPR_g`

## 4.8 Disentanglement and Concept Validity

Disentanglement module trains concept-specific sparse predictors:

- pathology concepts: L1 logistic probe (AUROC)
- categorical metadata: logistic probe (macro-F1)
- numeric metadata: Lasso (R²)

Latent-correlation module computes for each concept:

- max absolute correlation with any latent feature
- latent index achieving that max

Additional latent diagnostics:

- `latent_death_rate`: fraction of latent dims with near-zero activation frequency
- `latent_mean_active_per_sample`: average active dimensions per sample
- `latent_reuse_ratio`: proportion of latents reused across multiple concept winners
- `latent_max_concepts_per_feature`: max number of concepts sharing one latent winner

## 4.9 Statistical Inference

## Bootstrap CIs

- Nonparametric bootstrap for headline metrics (95% CI from quantiles).

## Paired Method Tests

- Paired bootstrap deltas between model pairs.
- Reported metrics include:
  - `worst_group_macro_auroc`
  - `macro_auroc_gap`
  - `macro_auroc`
  - `macro_accuracy`
  - `macro_brier`
  - `macro_ece`
- Two-sided p-value from bootstrap delta sign balance.

## Multiple Testing

- Benjamini-Hochberg FDR (`p_adj_bh`)
- Holm-Bonferroni (`p_adj_holm`)

## Permutation Controls

- Concept-permutation tests compare observed concept correlations to null distributions from label permutation.
- Report corrected p-values per concept.

## 4.10 SAE Sweep Selection Objective

Runs are scored via standardized composite metric across selected axes:

- Higher-is-better terms: disentanglement, concept correlation, macro AUROC, worst-group AUROC, latent reuse/activity
- Lower-is-better terms (sign-flipped): reconstruction MSE, fairness gaps, death rate

Best run is the max composite score.

## 4.11 Supplementary Baseline Models

Supplement baseline suite includes:

- `raw`: linear probe on raw embeddings
- `pca`: PCA bottleneck + probe
- `nmf`: NMF bottleneck + probe
- `supervised_bottleneck`: trainable bottleneck predictor
- `group_reweighted`: inverse-frequency group sample weights
- `group_threshold`: post-hoc per-group thresholding
- `equalized_odds`: approximate equalized-odds threshold tuning
- `adversarial_debiasing`: bottleneck predictor with adversarial age-group classifier

## 4.12 Concept Specificity Metrics

The core question in Stage 2 is: *which SAE latents encode demographic
information without encoding pathology?*  Formally, the ideal concept `c` to
ablate for a (demographic attribute `A`, pathology `P`) pair satisfies:

> **c encodes A** (activates differently across demographic groups), and
> **c ⊥ P | A** (provides no additional information about the diagnosis
> beyond what group membership already implies).

Two metrics are provided to approximate this, selectable via
`sae.specificity_mode` in the config or `--specificity-mode` on the CLI.

---

### Mode 1: `eta2` (default)

```
demo_score(c, A)    = η²(c, A)       — fraction of concept variance explained by A
path_score(c, P)    = η²(c, P)       — fraction of concept variance explained by P
specificity(c, A, P) = demo_η²(c, A) − path_η²(c, P)
```

η² is the classical one-way ANOVA effect size: `SS_between / SS_total`.  For a
binary grouping variable it equals the squared point-biserial correlation r².

**Strengths**

- Fast: O(n) per (concept, target) pair.
- Intuitive: fraction of variance explained, bounded in [0, 1].
- Works well when classes are reasonably balanced and activations are
  approximately continuous.

**Known biases**

1. *Class imbalance.*  For a binary target with prevalence p, SS_between is
   proportional to n·p(1−p)·(Δμ)².  A pathology with 2% prevalence
   (p = 0.02) can contribute at most 0.02 × 0.98 ≈ 2% of its variance as
   SS_between, even if the concept is a perfect discriminator.  Multi-class
   demographic attributes (e.g. 8 balanced age bins) face no such ceiling.
   This means `path_η²` is *structurally* lower than `demo_η²` for
   multi-class demographics, and the specificity `demo_η² − path_η²` is
   inflated for those pairs regardless of the concept's true nature.

2. *Binary sex attribute.*  Sex is also a binary variable, so `demo_η²` for
   sex suffers the same suppression as `path_η²`.  Concept specificity for
   (sex, P) pairs is therefore approximately unbiased between the two terms —
   but both are smaller in absolute value than for multi-class demographics,
   making sex pairs harder to distinguish from noise.

3. *Zero-inflation.*  TopK SAE activations are ~95–98% zero for a given
   concept.  All zeros are pulled toward the grand mean, inflating SS_total
   without adding to SS_between.  This depresses η² across the board but
   more severely for concepts that fire rarely (i.e. those that encode rare
   pathologies).

4. *Confounded path score.*  `path_η²(c, P)` is computed unconditionally.
   If age is strongly correlated with Cardiomegaly in the dataset, a concept
   that encodes age (not Cardiomegaly) will still show elevated `path_η²`
   simply because its age-encoding causes it to fire more often for older
   patients, who happen to have higher Cardiomegaly prevalence.  The
   specificity score therefore *underestimates* how demographic the concept
   truly is in highly correlated (A, P) pairs.

**When to use `eta2`**

Use it as a fast baseline or when the demographic attributes are multi-class
and roughly balanced (age_group, race, insurance_type) and the pathologies of
interest are relatively common (>5% prevalence).  The bias in the binary/rare
case is tolerable for a first-pass sweep; the concept rankings are still
informative.  It is also the right choice when you want the grid sweep of 20
SAEs to finish quickly — AUC computation in `conditional_auc` mode is
meaningfully slower.

---

### Mode 2: `conditional_auc`

```
demo_score(c, A)     = macro OVR AUC(c → A)
                       — how well c rank-orders patients by demographic group
path_score(c, A, P)  = mean_g[ AUC(c → P | A = g) ]
                       — mean AUC of c predicting P within each group g of A
specificity(c, A, P) = demo_auc(c, A) − conditional_path_auc(c, A, P)
```

**Demo score.**  For binary attributes (sex), this is the standard binary
AUC.  For multi-class attributes (age_group, race, insurance_type), it is the
macro one-vs-rest AUC: for each group g, compute AUC(c → {patient in g})
and average across all g.

**Conditional path score.**  The key innovation.  Rather than computing
`AUC(c → P)` over the whole dataset, we stratify by demographic group and
compute `AUC(c → P | A = g)` separately within each stratum, then average.
Only strata with at least 5 positive cases contribute to the mean.

**Why this fixes the confounding problem.**  Suppose age is strongly correlated
with Cardiomegaly.  A concept that fires for age (not Cardiomegaly) will show
elevated `AUC(c → Cardiomegaly)` in `eta2` mode — because older patients,
who trigger the concept, also disproportionately have Cardiomegaly.  But
within each age stratum, that concept provides *no useful ranking* of patients
by Cardiomegaly status (it fires indiscriminately for all old patients,
regardless of whether they have the disease).  The conditional path AUC
therefore correctly stays near 0.5, and the concept receives a high
specificity score.

Conversely, a concept that directly encodes Cardiomegaly (e.g. one that
responds to enlarged cardiac silhouette regardless of patient age) will rank
Cardiomegaly-positive patients above Cardiomegaly-negative patients *even
within* a single age stratum.  Its conditional path AUC exceeds 0.5, and its
specificity score is appropriately penalised.

**Why AUC instead of η².**  AUC is the probability that a randomly chosen
positive outranks a randomly chosen negative (Wilcoxon-Mann-Whitney
statistic).  It is:

- *Invariant to class imbalance*: 2% prevalence and 50% prevalence give
  comparably calibrated AUC values.
- *Rank-based*: unaffected by the heavy zero-inflation of TopK activations —
  all zeros are tied at the bottom of the ranking and treated consistently.
- *Comparable across binary and multi-class targets*: both demo and path
  terms live in [0.5, 1], so the subtraction is meaningful and symmetric.

**Known limitations**

1. *Small strata.*  Stratifying by group can leave very few positives per
   stratum for rare pathologies (e.g. Pneumonia with 0.2% prevalence in a
   stratum of 500 patients → ~1 positive).  Strata with fewer than 5
   positives are excluded; if all strata are excluded the value is `NaN` and
   that pair is skipped.  This affects mainly the rarest pathologies and the
   finest-grained demographic attributes.

2. *Computation time.*  With 20 SAEs × n_concepts × 56 (attr, P) pairs ×
   stratified AUC per concept, the scoring step is roughly 5–10× slower than
   `eta2` mode.

3. *Linear confounding only.*  The conditional AUC removes the *rank-based*
   association between demographics and pathology.  Nonlinear confounding
   (e.g. a concept encoding a complex interaction of age × comorbidity) is
   not fully removed.  For most practical purposes this is adequate.

**When to use `conditional_auc`**

Use it when you have reason to believe that the targeted demographic attribute
is epidemiologically correlated with the target pathology (age and Cardiomegaly,
race and Pleural Effusion, insurance type and Support Devices, etc.) and you
want to be confident that the selected concepts are genuinely encoding the
demographic variable rather than being proxies for pathology.  This is the
more defensible choice for any publication-quality analysis, and should be
preferred when the full pipeline is being run to produce final intervention
results.

---

### Choosing between modes: decision guide

| Situation | Recommended mode |
|---|---|
| Quick exploration / first SAE grid sweep | `eta2` |
| Pathology prevalence < 5% | `conditional_auc` |
| Demographic attribute is binary (sex) | `conditional_auc` (η² is equally biased for both terms, but AUC is better calibrated) |
| Demographic and pathology known to be correlated | `conditional_auc` |
| Comparing two runs head-to-head | Run both; compare intervention AUROC/TPR outcomes |
| Publication-quality results | `conditional_auc` |

### Running both modes side by side

Both runs coexist in the same output tree:

```
outputs/default/presentation/
  sae-eval/                       ← eta2 mode
  sae-eval-conditional-auc/       ← conditional_auc mode
  interventions/                  ← interventions from eta2 run
  interventions-conditional-auc/  ← interventions from conditional_auc run
```

```bash
# eta2 (default)
python scripts/concept_analysis_pipeline.py --config configs/default.yaml

# conditional_auc
python scripts/concept_analysis_pipeline.py --config configs/default.yaml \
    --specificity-mode conditional_auc

# Or set in config YAML:
#   sae:
#     specificity_mode: "conditional_auc"
```

## 5) Output Artifacts

## 5.1 Comprehensive study run

`<output_root>/runs/YYYYMMDD_HHMMSS/`

- `run_summary.json`
- `configs/` snapshots
- `sae_sweep/summary.csv`, `sae_sweep/summary.json`
- `workspace/manifest.csv`
- `workspace/features.npz`
- `workspace/sae.pt`
- `workspace/study_metrics.json`
- `workspace/study_predictions.npz`
- `figures/sweep/*.png`
- `figures/best_model/*.png`

## 5.2 Core publication run

`<output_root>/publication/core/YYYYMMDD_HHMMSS/`

- `core_pipeline_summary.json`
- `reproducibility_appendix.json`
- tables:
  - `table1_cohort`
  - `table2_main_results`
  - `table2b_paired_tests`
  - `table2c_group_fairness`
  - `table3_intervention_ablation`
- figures:
  - `figures/sweep/*.png`
  - `figures/best_model/*.png`

## 5.3 Supplement publication run

`<output_root>/publication/supplement/YYYYMMDD_HHMMSS/`

- `supplement_pipeline_summary.json`
- `reproducibility_appendix.json`
- tables include (when available):
  - `stable_seeds`
  - `seed_variance`
  - `uncertain_policy`
  - `debias_ablation`
  - `age_bin_sensitivity`
  - `baseline_comparison`
  - `threshold_sensitivity`
  - `missingness_sensitivity`
  - `permutation_control`
  - `concept_precision_recall`
  - `concept_permutation`
  - `leakage_checks`
  - `view_sensitivity`
  - `external_validation`
  - `human_eval`
- figures: `figures/*.png`

## 6) Figure Catalog (caption-style descriptions)

This section describes every plot produced by the codebase.

## 6.1 Sweep plots from `chex-run-sae-sweep` (`src/chex_sae_fairness/sweep.py`)

- `reconstruction_mse.png`  
  Bar chart of reconstruction MSE by SAE run. Lower is better for fidelity.
- `pathology_correlations.png`  
  Bar chart of mean max absolute pathology-correlation per run. Higher suggests stronger concept alignment.
- `recon_vs_correlation.png`  
  Scatter tradeoff of reconstruction MSE vs pathology correlation. Each point is a run.
- `worst_group_macro_auroc.png`  
  Bar chart of worst-group macro AUROC for concept probes; fairness-sensitive performance lens.

## 6.2 Core sweep figures (`src/chex_sae_fairness/reporting/figures.py`)

- `sae_hyperparam_ranking.png`  
  Composite score ranking across SAE configurations.
- `disentanglement_vs_pathology_correlation.png`  
  Scatter: disentanglement score vs pathology-correlation strength.
- `fairness_performance_tradeoff.png`  
  Scatter: overall macro AUROC vs worst-group AUROC.
- `run_metric_scorecard_heatmap.png`  
  Z-scored metric heatmap across runs (signed so higher-is-better).

## 6.3 Core study figures (`src/chex_sae_fairness/reporting/figures.py`)

- `classifier_performance_summary.png`  
  Baseline vs SAE vs debiased bars for macro metrics and worst-group performance.
- `fairness_gap_summary.png`  
  Group-gap comparisons (`macro_auroc_gap`, `macro_accuracy_gap`, EO TPR/FPR gaps).
- `calibration_summary.png`  
  Macro Brier and ECE by method (lower is better).
- `pareto_fairness_vs_performance.png`  
  Fairness gain vs AUROC drop relative to baseline.
- `group_macro_auroc.png`  
  Per-age-group macro AUROC by method.
- `group_macro_accuracy.png`  
  Per-age-group macro accuracy by method.
- `top_age_associated_latents.png`  
  Highest age-associated latent units (between-group variance style score).
- `sae_training_curve.png`  
  Train/validation SAE loss across epochs.

## 6.4 Supplement figures (`src/chex_sae_fairness/publication/figures.py`)

- `seed_stability_macro_auroc.png`  
  Box/strip plot of macro AUROC across seeds.
- `uncertain_policy_sensitivity.png`  
  Bar plot of uncertain-label policy ablation.
- `debias_mode_strength_sensitivity.png`  
  Line plot: worst-group macro AUROC vs debias strength by mode.
- `age_bin_sensitivity.png`  
  Macro AUROC gap under alternative age-bin schemes.
- `alternative_baselines_macro_auroc.png`  
  Macro AUROC comparison across alternative baseline methods.
- `baseline_pareto_front.png`  
  Fairness improvement vs AUROC drop vs raw baseline across baseline methods.
- `threshold_sensitivity.png`  
  Fairness gap sensitivity to classification threshold.
- `missingness_sensitivity.png`  
  Performance under simulated missing image rows and missing age-group metadata.
- `permutation_control_histogram.png`  
  Null distribution of mean pathology correlation under permutation with observed marker.
- `concept_precision_recall_top_f1.png`  
  Top concept-level F1 scores across pathology/metadata concept probes.
- `concept_permutation_padj_hist.png`  
  Distribution of BH-adjusted permutation p-values across concepts.
- `view_type_sensitivity.png`  
  Performance stratified by view type (frontal/lateral/unknown).

## 7) Table Catalog

## 7.1 Core tables

- `table1_cohort`: split counts, age-group sample sizes, pathology prevalence.
- `table2_main_results`: primary and secondary endpoints with CIs for baseline/SAE/debiased.
- `table2b_paired_tests`: paired bootstrap deltas and corrected p-values.
- `table2c_group_fairness`: per-group AUROC/accuracy/TPR/FPR with CIs.
- `table3_intervention_ablation`: debias mode/strength ablations.

## 7.2 Supplement tables

- `stable_seeds`: per-seed outcomes by method.
- `seed_variance`: mean/std over seeds.
- `uncertain_policy`: uncertain label policy ablations.
- `debias_ablation`: mode/strength intervention grid.
- `age_bin_sensitivity`: alternate age partition robustness.
- `baseline_comparison`: expanded baseline suite comparison.
- `threshold_sensitivity`: threshold robustness.
- `missingness_sensitivity`: missingness robustness.
- `permutation_control`: global permutation null summary.
- `concept_precision_recall`: concept-level precision/recall/F1/AUROC.
- `concept_permutation`: concept-level permutation tests + corrected p-values.
- `leakage_checks`: patient overlap and split integrity checks.
- `view_sensitivity`: frontal/lateral sensitivity.
- `external_validation`: transfer to external configs.
- `human_eval`: human vs auto-label agreement summary.

## 8) Configuration Reference

Primary configs:

- `configs/default.yaml`
- `configs/publication.yaml`
- `configs/sae_sweep.yaml`

Important controls:

- Data: `allowed_views` (e.g. `["frontal"]` to restrict to frontal-only; `[]` for all views), `max_rows_per_split` (integer cap per split for fast pipeline tests; omit or `null` for full dataset)
- SAE: `latent_dim`, `variant`, `topk_k`, `l1_lambda`, optimizer settings
- Fairness: `debias_mode`, `debias_strength`, `threshold`
- Publication supplement: seeds, uncertain policies, baseline methods, threshold grid, age-bin sets, missingness fractions

## 9) Reproducibility Metadata

Each publication run writes `reproducibility_appendix.json` with:

- timestamp
- platform/Python/PyTorch/CUDA details
- configuration seed and output root
- data filtering/split counts
- pipeline-specific metadata

## 10) Sanity Check

```bash
python -m compileall src tests scripts
```
