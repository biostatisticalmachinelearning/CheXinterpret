#!/usr/bin/env python3
"""SAE concept analysis pipeline.

Trains a grid of Top-k Sparse Autoencoders (k × latent_dim) on CheXagent embeddings,
then measures how strongly each SAE latent (concept) correlates with demographic
variables vs. pathology labels using Eta-squared (η²) throughout.

Specificity is computed **per (demographic, pathology) pair**:
    spec(concept, demo_attr, pathology) = demo_η²_attr(concept) − path_η²_pathology(concept)

A concept with high specificity for pair (sex, Cardiomegaly) is strongly correlated
with sex AND weakly correlated with Cardiomegaly — the ideal fairness confound.

Grid: k ∈ {8, 16, 32, 64}  ×  latent_dim ∈ {256, 512, 1024, 2048, 4096}  → 20 SAEs

Output structure (written to <output_root>/presentation/sae-eval/):

  Per-SAE directories  k<k>_d<dim>/
  ├── concept_scores.csv         full η² + per-pair specificity table
  ├── pair_overview.png          (attr × path) heatmap: max single-concept spec per pair
  ├── per_pair/
  │   └── top10_<attr>_<path>.png  top-10 concepts for each (demo, path) pair
  │                                3-bar clusters: demo_η², path_η², pair_specificity
  └── activation_dist_<attr>.png  box plots: top-5 demo concepts by group

  Grid directory  grid/
  ├── grid_summary.csv             one row per SAE, mean spec per pair
  ├── best_sae_per_pair.csv        best (k, latent_dim) for every (attr, path) pair
  ├── best_sae_heatmap.png         (attr × path) heatmap: name of best architecture
  ├── specificity_heatmap.png      k×dim grid coloured by mean pair-specificity
  ├── scatter_demo_vs_path.png     mean demo_η² vs mean path_η² scatter per SAE
  └── pair_heatmaps/
      └── spec_<attr>_<path>.png   k×dim grid heatmap for each (attr, path) pair

Usage:
  python scripts/concept_analysis_pipeline.py --config configs/test.yaml
  python scripts/concept_analysis_pipeline.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# SAE sweep grid
K_VALUES: list[int] = [8, 16, 32, 64]
DIM_VALUES: list[int] = [256, 512, 1024, 2048, 4096]
SAE_EPOCHS: int = 40
SAE_BATCH_SIZE: int = 512
SAE_LR: float = 1e-3
SAE_WEIGHT_DECAY: float = 1e-6

# Top-N concepts shown in per-pair plots
TOP_N: int = 10
# Top concepts shown in activation distribution plots
TOP_DIST: int = 5


# ── Utilities ─────────────────────────────────────────────────────────────────

def _safe(s: str) -> str:
    """Convert a label to a safe filename / column-name fragment."""
    return s.replace(" ", "_").replace("/", "-")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write results. Defaults to <output_root>/presentation/sae-eval/",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=SAE_EPOCHS,
        help=f"SAE training epochs (default: {SAE_EPOCHS})",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--skip-interventions",
        action="store_true",
        default=False,
        help="Skip the 56-intervention runner at the end (default: interventions run)",
    )
    return parser.parse_args()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_bundle(feature_path: Path) -> dict[str, np.ndarray]:
    if not feature_path.exists():
        sys.exit(
            f"\nFeature bundle not found at {feature_path}\n"
            "Wait for `chex-run-study` to finish feature extraction, then re-run.\n"
        )
    logger.info("Loading feature bundle from %s", feature_path)
    with np.load(feature_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _clean_attr(values: np.ndarray) -> np.ndarray:
    out = values.astype(object)
    out[np.isin(values.astype(str), {"nan", "None", "none", ""})] = np.nan
    return out


# ── SAE ────────────────────────────────────────────────────────────────────────

class TopKSAE(torch.nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, k: int) -> None:
        super().__init__()
        self.k = k
        self.latent_dim = latent_dim
        self.encoder = torch.nn.Linear(input_dim, latent_dim)
        self.decoder = torch.nn.Linear(latent_dim, input_dim)
        self.activation = torch.nn.ReLU()
        torch.nn.init.xavier_uniform_(self.encoder.weight)
        torch.nn.init.zeros_(self.encoder.bias)
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        torch.nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.activation(self.encoder(x))
        if self.k >= z.shape[-1]:
            return z
        _, indices = torch.topk(z, k=self.k, dim=-1)
        mask = torch.zeros_like(z)
        mask.scatter_(dim=-1, index=indices, value=1.0)
        return z * mask

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


def train_topk_sae(
    x_train: np.ndarray,
    x_valid: np.ndarray,
    latent_dim: int,
    k: int,
    epochs: int,
    device: str,
) -> TopKSAE:
    model = TopKSAE(x_train.shape[1], latent_dim, k).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SAE_LR, weight_decay=SAE_WEIGHT_DECAY)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32)),
        batch_size=SAE_BATCH_SIZE,
        shuffle=True,
    )
    valid_tensor = torch.tensor(x_valid, dtype=torch.float32)

    best_loss = float("inf")
    best_state: dict | None = None

    for _epoch in range(epochs):
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device)
            x_hat, _ = model(batch)
            loss = torch.mean((batch - x_hat) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            v = valid_tensor.to(device)
            x_hat_v, _ = model(v)
            vloss = float(torch.mean((v - x_hat_v) ** 2).item())

        if vloss < best_loss:
            best_loss = vloss
            best_state = {k_: v_.cpu().clone() for k_, v_ in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def encode_all(model: TopKSAE, x: np.ndarray, device: str) -> np.ndarray:
    model.eval().to(device)
    loader = DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float32)),
        batch_size=SAE_BATCH_SIZE * 4,
        shuffle=False,
    )
    parts: list[np.ndarray] = []
    for (batch,) in loader:
        parts.append(model.encode(batch.to(device)).cpu().numpy().astype(np.float32))
    return np.concatenate(parts, axis=0)


# ── Eta-squared (η²) ──────────────────────────────────────────────────────────

def eta_squared_binary(z_col: np.ndarray, binary_target: np.ndarray) -> float:
    """η² = SS_between / SS_total for a binary grouping variable.

    Equivalent to r² (squared point-biserial correlation).
    """
    valid = np.isfinite(z_col) & np.isfinite(binary_target)
    if valid.sum() < 3:
        return float("nan")
    z = z_col[valid]
    y = binary_target[valid].astype(float)
    grand_mean = z.mean()
    ss_total = float(np.sum((z - grand_mean) ** 2))
    if ss_total < 1e-12:
        return 0.0
    ss_between = float(sum(
        np.sum(y == g) * (z[y == g].mean() - grand_mean) ** 2
        for g in np.unique(y)
    ))
    return float(np.clip(ss_between / ss_total, 0.0, 1.0))


def eta_squared_categorical(z_col: np.ndarray, cat_target: np.ndarray) -> float:
    """η² for a categorical grouping variable (any number of classes)."""
    valid_mask = np.array(
        [v is not np.nan and str(v) not in {"nan", "None", "none", ""} for v in cat_target],
        dtype=bool,
    )
    valid_mask &= np.isfinite(z_col)
    if valid_mask.sum() < 3:
        return float("nan")
    z = z_col[valid_mask]
    cats = cat_target[valid_mask].astype(str)
    grand_mean = z.mean()
    ss_total = float(np.sum((z - grand_mean) ** 2))
    if ss_total < 1e-12:
        return 0.0
    ss_between = float(sum(
        np.sum(cats == g) * (z[cats == g].mean() - grand_mean) ** 2
        for g in np.unique(cats)
    ))
    return float(np.clip(ss_between / ss_total, 0.0, 1.0))


def compute_concept_scores(
    z: np.ndarray,
    y_pathology: np.ndarray,
    pathology_cols: list[str],
    attr_arrays: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compute η² for every concept vs every target and per-pair specificity.

    Per-pair specificity for concept c, demographic attr A, pathology P:
        spec(c, A, P) = demo_η²_A(c) − path_η²_P(c)

    Columns in returned DataFrame:
        concept_idx
        demo_eta2_<attr>           — η² vs each demographic attribute
        path_eta2_<safe_path>      — η² vs each pathology label
        spec_<attr>_<safe_path>    — per-pair specificity (one col per attr × path)
        max_demo_eta2              — max across attributes (summary)
        max_path_eta2              — max across pathologies (summary)
    """
    n_concepts = z.shape[1]
    safe_paths = [_safe(p) for p in pathology_cols]
    rows: list[dict] = []

    for c_idx in range(n_concepts):
        z_col = z[:, c_idx]
        row: dict = {"concept_idx": c_idx}

        # ── Pathology η² (binary) ────────────────────────────────────────────
        path_eta2: dict[str, float] = {}
        for p_idx, (pname, safe_p) in enumerate(zip(pathology_cols, safe_paths)):
            val = eta_squared_binary(z_col, y_pathology[:, p_idx].astype(float))
            row[f"path_eta2_{safe_p}"] = val
            path_eta2[safe_p] = val

        # ── Demographic η² (categorical) ─────────────────────────────────────
        demo_eta2: dict[str, float] = {}
        for attr, attr_vals in attr_arrays.items():
            val = eta_squared_categorical(z_col, attr_vals)
            row[f"demo_eta2_{attr}"] = val
            demo_eta2[attr] = val

        # ── Per-pair specificity ──────────────────────────────────────────────
        for attr, de in demo_eta2.items():
            for safe_p, pe in path_eta2.items():
                if np.isfinite(de) and np.isfinite(pe):
                    row[f"spec_{attr}_{safe_p}"] = de - pe
                else:
                    row[f"spec_{attr}_{safe_p}"] = float("nan")

        # ── Summary statistics (for scatter / grid plots) ────────────────────
        valid_demo = [v for v in demo_eta2.values() if np.isfinite(v)]
        valid_path = [v for v in path_eta2.values() if np.isfinite(v)]
        row["max_demo_eta2"] = float(np.max(valid_demo)) if valid_demo else float("nan")
        row["max_path_eta2"] = float(np.max(valid_path)) if valid_path else float("nan")
        row["mean_demo_eta2"] = float(np.mean(valid_demo)) if valid_demo else float("nan")
        row["mean_path_eta2"] = float(np.mean(valid_path)) if valid_path else float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


# ── Per-SAE: per-pair top-10 plots ────────────────────────────────────────────

def _plot_top10_per_pair(
    scores_df: pd.DataFrame,
    attr: str,
    pathology: str,
    k: int,
    latent_dim: int,
    out_dir: Path,
) -> None:
    """3-bar cluster plot: top-10 concepts ranked by spec(attr, pathology).

    Bars show demo_η²_attr, path_η²_pathology, and pair_specificity for each concept.
    """
    safe_p = _safe(pathology)
    spec_col = f"spec_{attr}_{safe_p}"
    demo_col = f"demo_eta2_{attr}"
    path_col = f"path_eta2_{safe_p}"

    if spec_col not in scores_df.columns:
        return

    sub = scores_df[[spec_col, demo_col, path_col, "concept_idx"]].dropna(subset=[spec_col])
    top = sub.nlargest(TOP_N, spec_col).reset_index(drop=True)
    if top.empty:
        return

    labels = [f"c{int(r.concept_idx)}" for _, r in top.iterrows()]
    demo_vals = top[demo_col].fillna(0).tolist()
    path_vals = top[path_col].fillna(0).tolist()
    spec_vals = top[spec_col].fillna(0).tolist()

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 4.5))
    ax.bar(x - width, demo_vals, width, label=f"demo η² ({attr.replace('_',' ')})", color="#4C78A8", alpha=0.85)
    ax.bar(x,          path_vals, width, label=f"path η² ({pathology})",              color="#F58518", alpha=0.85)
    ax.bar(x + width,  spec_vals, width, label="pair specificity (demo−path)",         color="#54A24B", alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("η²")
    ax.set_title(
        f"Top-{TOP_N} concepts  ·  demo={attr.replace('_',' ')}  ·  path={pathology}\n"
        f"SAE: k={k}, latent_dim={latent_dim}",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    path_out = out_dir / f"top10_{attr}_{safe_p}.png"
    fig.savefig(path_out, dpi=150)
    plt.close(fig)


def _plot_pair_overview_heatmap(
    scores_df: pd.DataFrame,
    attr_names: list[str],
    pathology_cols: list[str],
    k: int,
    latent_dim: int,
    out_dir: Path,
) -> None:
    """(attr × path) heatmap showing max single-concept specificity for this SAE.

    Gives a quick overview of which (demographic, pathology) pairs this SAE is
    most informative about.
    """
    safe_paths = [_safe(p) for p in pathology_cols]
    matrix = np.full((len(attr_names), len(pathology_cols)), fill_value=float("nan"))

    for ai, attr in enumerate(attr_names):
        for pi, safe_p in enumerate(safe_paths):
            col = f"spec_{attr}_{safe_p}"
            if col in scores_df.columns:
                matrix[ai, pi] = float(scores_df[col].dropna().max())

    fig, ax = plt.subplots(figsize=(max(8, len(pathology_cols) * 0.9), max(3, len(attr_names) * 0.8)))
    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=pathology_cols,
        yticklabels=[a.replace("_", " ") for a in attr_names],
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": "Max single-concept pair specificity"},
        linewidths=0.4,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    ax.set_title(
        f"Max Pair Specificity per (demo, pathology)\nSAE: k={k}, latent_dim={latent_dim}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "pair_overview.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved → pair_overview.png")


# ── Per-SAE: activation distribution plots ───────────────────────────────────

def _plot_activation_dist(
    z: np.ndarray,
    scores_df: pd.DataFrame,
    attr: str,
    attr_vals: np.ndarray,
    k: int,
    latent_dim: int,
    out_dir: Path,
) -> None:
    """Box plots of top-5 concepts' activations, split by demographic group.

    Top concepts selected by demo_η²_attr — shows whether the concepts that are
    most correlated with this attribute activate differently across groups.
    """
    demo_col = f"demo_eta2_{attr}"
    if demo_col not in scores_df.columns:
        return

    top5 = (
        scores_df[["concept_idx", demo_col]]
        .dropna(subset=[demo_col])
        .nlargest(TOP_DIST, demo_col)
    )
    if top5.empty:
        return

    groups = sorted(
        np.unique(attr_vals[~pd.isna(attr_vals)].astype(str))
    )
    if len(groups) < 2:
        return

    palette = sns.color_palette("tab10", len(groups))
    fig, axes = plt.subplots(1, len(top5), figsize=(4 * len(top5), 4), squeeze=False)

    for ax_i, (_, row) in enumerate(top5.iterrows()):
        c_idx = int(row["concept_idx"])
        eta2_val = float(row[demo_col])
        ax = axes[0][ax_i]
        data_groups = [z[attr_vals.astype(str) == g, c_idx] for g in groups]
        bp = ax.boxplot(
            data_groups,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        group_labels = [f"{g}\n(n={len(data_groups[gi])})" for gi, g in enumerate(groups)]
        ax.set_xticklabels(group_labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(f"c{c_idx}\nη²={eta2_val:.3f}", fontsize=8)
        ax.set_ylabel("Activation" if ax_i == 0 else "")

    fig.suptitle(
        f"Top-{TOP_DIST} concept activations by {attr.replace('_', ' ')}\n"
        f"SAE: k={k}, latent_dim={latent_dim}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"activation_dist_{attr}.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved → activation_dist_%s.png", attr)


def _plot_activation_dist_pathology(
    z: np.ndarray,
    y_pathology: np.ndarray,
    pathology_cols: list[str],
    scores_df: pd.DataFrame,
    k: int,
    latent_dim: int,
    out_dir: Path,
) -> None:
    """Box plots of top-5 concepts' activations split by positive vs negative for each pathology.

    Top concepts selected by path_eta2_<safe_path> — shows whether concepts most correlated
    with this pathology activate differently between positive and negative cases.
    """
    n_saved = 0
    for pathology in pathology_cols:
        safe_path = _safe(pathology)
        path_col = f"path_eta2_{safe_path}"
        if path_col not in scores_df.columns:
            continue

        p_idx = pathology_cols.index(pathology)
        y_col = y_pathology[:, p_idx]

        top5 = (
            scores_df[["concept_idx", path_col]]
            .dropna(subset=[path_col])
            .nlargest(TOP_DIST, path_col)
        )
        if top5.empty:
            continue

        pos_mask = y_col == 1
        neg_mask = y_col == 0
        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())

        if n_pos < 2 or n_neg < 2:
            continue

        fig, axes = plt.subplots(1, len(top5), figsize=(4 * len(top5), 4), squeeze=False)

        for ax_i, (_, row) in enumerate(top5.iterrows()):
            c_idx = int(row["concept_idx"])
            eta2_val = float(row[path_col])
            ax = axes[0][ax_i]

            pos_data = z[pos_mask, c_idx]
            neg_data = z[neg_mask, c_idx]

            bp = ax.boxplot(
                [neg_data, pos_data],
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.5},
            )
            colors = ["#4C78A8", "#F58518"]
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)

            ax.set_xticks([1, 2])
            ax.set_xticklabels(
                [f"Negative\n(n={n_neg})", f"Positive\n(n={n_pos})"],
                fontsize=7,
            )
            ax.set_title(f"c{c_idx}\nη²={eta2_val:.3f}", fontsize=8)
            ax.set_ylabel("Activation" if ax_i == 0 else "")

        fig.suptitle(
            f"Top-{TOP_DIST} concept activations: {pathology} (pos vs neg)\n"
            f"SAE: k={k}, latent_dim={latent_dim}",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"activation_dist_path_{safe_path}.png", dpi=150)
        plt.close(fig)
        logger.info("  Saved → activation_dist_path_%s.png", safe_path)
        n_saved += 1

    return n_saved


# ── Grid-level plots ───────────────────────────────────────────────────────────

def _plot_pair_grid_heatmap(
    grid_results: list[dict],
    attr: str,
    pathology: str,
    grid_out: Path,
) -> None:
    """k×latent_dim heatmap of best single-concept specificity for one (attr, path) pair."""
    safe_p = _safe(pathology)
    col = f"best_spec_{attr}_{safe_p}"

    k_vals = sorted({r["k"] for r in grid_results})
    d_vals = sorted({r["latent_dim"] for r in grid_results})
    matrix = np.full((len(k_vals), len(d_vals)), fill_value=float("nan"))

    for r in grid_results:
        if col in r:
            matrix[k_vals.index(r["k"]), d_vals.index(r["latent_dim"])] = r[col]

    if np.all(np.isnan(matrix)):
        return

    fig, ax = plt.subplots(figsize=(len(d_vals) * 1.6 + 1.5, len(k_vals) * 1.2 + 1.5))
    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=[str(d) for d in d_vals],
        yticklabels=[str(kv) for kv in k_vals],
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": "Max single-concept pair specificity"},
        linewidths=0.5,
    )
    ax.set_xlabel("Latent dim")
    ax.set_ylabel("k")
    ax.set_title(
        f"Best single-concept specificity\ndemo={attr.replace('_',' ')}  ·  path={pathology}",
        fontsize=9,
    )
    fig.tight_layout()
    safe_attr = _safe(attr)
    fig.savefig(grid_out / f"spec_{safe_attr}_{safe_p}.png", dpi=150)
    plt.close(fig)


def _plot_best_sae_heatmap(
    best_sae_df: pd.DataFrame,
    attr_names: list[str],
    pathology_cols: list[str],
    grid_out: Path,
) -> None:
    """(attr × path) heatmap showing the best architecture label for each pair."""
    safe_paths = [_safe(p) for p in pathology_cols]
    labels = np.full((len(attr_names), len(pathology_cols)), fill_value="", dtype=object)
    values = np.full((len(attr_names), len(pathology_cols)), fill_value=float("nan"))

    for _, row in best_sae_df.iterrows():
        ai = attr_names.index(row["attr"]) if row["attr"] in attr_names else -1
        pi = pathology_cols.index(row["pathology"]) if row["pathology"] in pathology_cols else -1
        if ai >= 0 and pi >= 0:
            labels[ai, pi] = f"k{int(row['best_k'])}\nd{int(row['best_dim'])}"
            values[ai, pi] = float(row["best_spec"])

    fig, axes = plt.subplots(
        1, 2, figsize=(max(10, len(pathology_cols) * 1.0 + 2), max(4, len(attr_names) * 0.9 + 2)),
        gridspec_kw={"width_ratios": [1, 1]},
    )

    # Left: architecture labels
    ax = axes[0]
    im = ax.imshow(values, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="Best pair specificity")
    ax.set_xticks(range(len(pathology_cols)))
    ax.set_xticklabels(pathology_cols, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(attr_names)))
    ax.set_yticklabels([a.replace("_", " ") for a in attr_names], fontsize=8)
    for ai in range(len(attr_names)):
        for pi in range(len(pathology_cols)):
            if labels[ai, pi]:
                ax.text(pi, ai, labels[ai, pi], ha="center", va="center",
                        fontsize=6, color="white", fontweight="bold")
    ax.set_title("Best SAE architecture\nper (demo, path) pair", fontsize=9)

    # Right: seaborn heatmap of the best-spec values
    ax2 = axes[1]
    sns.heatmap(
        values,
        ax=ax2,
        xticklabels=pathology_cols,
        yticklabels=[a.replace("_", " ") for a in attr_names],
        annot=np.round(values, 3),
        fmt="",
        cmap="viridis",
        cbar_kws={"label": "Best pair specificity"},
        linewidths=0.3,
    )
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax2.set_title("Best single-concept specificity\nper (demo, path) pair", fontsize=9)

    fig.suptitle("Best SAE Architecture per (Demographic, Pathology) Pair", fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(grid_out / "best_sae_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → best_sae_heatmap.png")


def _plot_specificity_heatmap(grid_results: list[dict], grid_out: Path) -> None:
    """k×latent_dim heatmap of mean pair-specificity (averaged across all pairs)."""
    k_vals = sorted({r["k"] for r in grid_results})
    d_vals = sorted({r["latent_dim"] for r in grid_results})
    matrix = np.full((len(k_vals), len(d_vals)), fill_value=float("nan"))

    for r in grid_results:
        ki = k_vals.index(r["k"])
        di = d_vals.index(r["latent_dim"])
        matrix[ki, di] = r["mean_pair_specificity"]

    fig, ax = plt.subplots(figsize=(len(d_vals) * 1.8 + 1.5, len(k_vals) * 1.5 + 1.5))
    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=[str(d) for d in d_vals],
        yticklabels=[str(kv) for kv in k_vals],
        annot=True,
        fmt=".4f",
        cmap="viridis",
        cbar_kws={"label": "Mean pair specificity (averaged across all demo×path pairs)"},
        linewidths=0.5,
    )
    ax.set_xlabel("Latent dim")
    ax.set_ylabel("k")
    ax.set_title(
        "Mean Pair Specificity Across SAE Grid\n"
        "(avg of max single-concept spec over all demo×path pairs)"
    )
    fig.tight_layout()
    fig.savefig(grid_out / "specificity_heatmap.png", dpi=150)
    plt.close(fig)
    logger.info("Saved → specificity_heatmap.png")


def _plot_demo_vs_path_scatter(grid_results: list[dict], grid_out: Path) -> None:
    """Per-SAE scatter: mean demo_η² vs mean path_η², coloured by mean pair specificity."""
    if not grid_results:
        return
    df = pd.DataFrame(grid_results)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        df["mean_path_eta2"],
        df["mean_demo_eta2"],
        c=df["mean_pair_specificity"],
        cmap="coolwarm_r",
        s=120,
        edgecolors="k",
        linewidths=0.5,
        zorder=3,
    )
    plt.colorbar(sc, ax=ax, label="mean pair specificity")

    lim = df[["mean_path_eta2", "mean_demo_eta2"]].max().max() * 1.1
    ax.plot([0, lim], [0, lim], "grey", linestyle="--", linewidth=0.8, zorder=2)

    for _, row in df.iterrows():
        ax.annotate(
            f"k={int(row['k'])}\nd={int(row['latent_dim'])}",
            (row["mean_path_eta2"], row["mean_demo_eta2"]),
            fontsize=6,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlabel("Mean path η² (avg concept × pathology)")
    ax.set_ylabel("Mean demo η² (avg concept × demographic)")
    ax.set_title(
        "Demo η² vs Path η² per SAE\n"
        "(above dashed line = more demographic than pathology signal)"
    )
    fig.tight_layout()
    fig.savefig(grid_out / "scatter_demo_vs_path.png", dpi=150)
    plt.close(fig)
    logger.info("Saved → scatter_demo_vs_path.png")


# ── Intervention runner ────────────────────────────────────────────────────────

def _run_all_interventions(cfg, base_out: Path, args: argparse.Namespace) -> None:
    """Run fairness_intervention.py for every (attr, pathology) pair in best_sae_per_pair.csv."""
    import subprocess

    grid_out = base_out / "grid"
    best_csv = grid_out / "best_sae_per_pair.csv"
    if not best_csv.exists():
        logger.warning("best_sae_per_pair.csv not found at %s; skipping interventions.", best_csv)
        return

    best_df = pd.read_csv(best_csv)
    pairs = list(best_df[["attr", "pathology"]].drop_duplicates().itertuples(index=False))
    n_pairs = len(pairs)
    logger.info("Running interventions for all %d (attr, pathology) pairs…", n_pairs)

    intervention_script = Path(__file__).parent / "fairness_intervention.py"
    interventions_out = base_out.parent / "interventions"

    for i, row in enumerate(pairs):
        attr = row.attr
        pathology = row.pathology
        logger.info("  [%d/%d] %s × %s", i + 1, n_pairs, attr, pathology)
        cmd = [
            sys.executable, str(intervention_script),
            "--config", args.config,
            "--attr", attr,
            "--pathology", pathology,
            "--sae-dir", str(base_out),
            "--output-dir", str(interventions_out),
            "--lr-mode", "both",
            "--threshold", "0.02",
        ]
        subprocess.run(cmd, check=False)

    logger.info("All interventions complete.")
    _generate_intervention_summary_roc(interventions_out, pairs, "0.02")


def _generate_intervention_summary_roc(
    interventions_out: Path,
    pairs: list,
    threshold: str,
) -> None:
    """Generate 14 per-pathology summary ROC figures after all 56 interventions finish.

    Each figure has 4 subplots (one per demographic attribute).  Within each
    subplot the per-group ROC curves for baseline (solid) and ablated (dashed)
    are drawn from the specific intervention that targeted that (attr, pathology)
    pair — so every quadrant reflects the right ablation.
    """
    from sklearn.metrics import roc_auc_score, roc_curve  # local import

    summary_dir = interventions_out / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Collect all pathologies and attrs from available pairs
    pathologies = list(dict.fromkeys(p.pathology for p in pairs))
    attrs       = list(dict.fromkeys(p.attr      for p in pairs))
    n_cols = 2
    n_rows = (len(attrs) + 1) // 2

    for pathology in pathologies:
        safe_p = pathology.replace(" ", "_").replace("/", "-")
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 5.5, n_rows * 5),
            squeeze=False,
        )
        fig.suptitle(
            f"ROC Curves — {pathology}\n"
            "Each quadrant: intervention targeting that (demographic, pathology) pair\n"
            "Solid = baseline  ·  Dashed = ablated",
            fontsize=10, fontweight="bold", y=1.02,
        )

        for ax_idx, attr in enumerate(attrs):
            ax = axes[ax_idx // n_cols][ax_idx % n_cols]
            safe_attr = attr.replace(" ", "_").replace("/", "-")
            tag = f"{safe_attr}_{safe_p}_t{threshold}"
            pair_dir = interventions_out / tag

            # Load scores from this (attr, pathology) pair
            baseline_npz  = pair_dir / "baseline" / "scores.npz"
            ablated_npz   = pair_dir / "ablated"  / "scores.npz"
            attr_npz      = pair_dir / "attr_arrays.npz"

            if not (baseline_npz.exists() and ablated_npz.exists() and attr_npz.exists()):
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{attr.replace('_',' ').title()}\n(data missing)", fontsize=9)
                continue

            try:
                b_data  = np.load(baseline_npz)
                a_data  = np.load(ablated_npz)
                at_data = np.load(attr_npz)
            except Exception as exc:
                ax.text(0.5, 0.5, str(exc), ha="center", va="center",
                        fontsize=6, transform=ax.transAxes)
                continue

            # y_valid shape: (n_samples, n_pathologies); column order matches pathology_cols
            y_valid_b  = b_data["y_valid"]
            y_scores_b = b_data["y_scores"]
            y_scores_a = a_data["y_scores"]

            # We need the column index for this pathology.
            # Save pathology_cols in attr_arrays.npz wasn't done, so we derive from
            # CSV if available, otherwise skip.
            lin_csv = pair_dir / "baseline" / "linear_separability.csv"
            if not lin_csv.exists():
                ax.set_title(f"{attr.replace('_',' ').title()}\n(csv missing)", fontsize=9)
                continue
            lin_df = pd.read_csv(lin_csv)
            if pathology not in lin_df["pathology"].values:
                ax.set_title(f"{attr.replace('_',' ').title()}\n({pathology} not in data)", fontsize=9)
                continue
            p_idx = int(lin_df[lin_df["pathology"] == pathology].index[0])
            if p_idx >= y_valid_b.shape[1]:
                continue

            y_true = y_valid_b[:, p_idx]
            if len(np.unique(y_true)) < 2:
                ax.set_title(f"{attr.replace('_',' ').title()}\n(one class only)", fontsize=9)
                continue

            attr_values = at_data.get(attr, at_data.get(attr.replace("_", ""), None))
            if attr_values is None:
                ax.set_title(f"{attr.replace('_',' ').title()}\n(attr not saved)", fontsize=9)
                continue
            attr_values = attr_values.astype(str)
            # Mask out nan strings
            valid_mask_a = ~np.isin(attr_values, {"nan", "None", "none", ""})

            groups = sorted(np.unique(attr_values[valid_mask_a]))
            palette = sns.color_palette("tab10", len(groups))

            CONDITION_COLORS = {"baseline": "#4C78A8", "ablated": "#F58518"}
            for (y_sc, cname, ls, lw) in [
                (y_scores_b, "baseline", "-",  2.2),
                (y_scores_a, "ablated",  "--", 1.8),
            ]:
                y_sc_p = y_sc[:, p_idx]
                if len(np.unique(y_true)) < 2:
                    continue
                fpr_all, tpr_all, _ = roc_curve(y_true, y_sc_p)
                auc_all = float(roc_auc_score(y_true, y_sc_p))
                ax.plot(fpr_all, tpr_all, color=CONDITION_COLORS[cname],
                        linewidth=lw, linestyle=ls,
                        label=f"{cname} overall (AUC={auc_all:.3f})", zorder=5)

                for color, g in zip(palette, groups):
                    mask = attr_values == g
                    y_g = y_true[mask]
                    s_g = y_sc_p[mask]
                    if len(np.unique(y_g)) < 2 or int((y_g == 1).sum()) < 5:
                        continue
                    fpr_g, tpr_g, _ = roc_curve(y_g, s_g)
                    auc_g = float(roc_auc_score(y_g, s_g))
                    ax.plot(fpr_g, tpr_g, color=color, linewidth=1.2, linestyle=ls, alpha=0.75,
                            label=f"{cname} · {g} (AUC={auc_g:.3f})")

            ax.plot([0, 1], [0, 1], color="lightgrey", linewidth=0.8, linestyle=":")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
            ax.set_xlabel("FPR", fontsize=9); ax.set_ylabel("TPR", fontsize=9)
            ax.set_title(
                f"{attr.replace('_', ' ').title()}\n"
                f"(targeted intervention for this pair)",
                fontsize=9,
            )
            ax.legend(fontsize=6, loc="lower right")

        for ax_idx in range(len(attrs), n_rows * n_cols):
            axes[ax_idx // n_cols][ax_idx % n_cols].set_visible(False)

        fig.tight_layout()
        out_path = summary_dir / f"roc_summary_{safe_p}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Summary ROC → %s", out_path.name)

    logger.info("Summary ROC plots saved to %s", summary_dir)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    try:
        from chex_sae_fairness.config import ExperimentConfig
    except ImportError:
        sys.exit("Package not installed. Run:  pip install -e .")

    cfg = ExperimentConfig.from_yaml(args.config)
    bundle = load_bundle(cfg.feature_path)

    base_out = Path(args.output_dir) if args.output_dir else cfg.output_root / "presentation" / "sae-eval"
    base_out.mkdir(parents=True, exist_ok=True)
    grid_out = base_out / "grid"
    (grid_out / "pair_heatmaps").mkdir(parents=True, exist_ok=True)
    logger.info("Outputs → %s", base_out.resolve())

    # ── Unpack bundle ──────────────────────────────────────────────────────────
    splits = bundle["split"].astype(str)
    x = bundle["features"].astype(np.float32)
    y_pathology = bundle["y_pathology"].astype(np.float32)
    pathology_cols = [str(c) for c in bundle["pathology_cols"]]
    metadata_cols = [str(c) for c in bundle["metadata_cols"]]
    metadata = pd.DataFrame(bundle["metadata"], columns=metadata_cols)
    age_group = bundle["age_group"].astype(str)

    train_mask = splits == "train"
    valid_mask = splits == cfg.data.validation_split_name

    logger.info("Split sizes — train: %d  valid: %d", train_mask.sum(), valid_mask.sum())

    # Validation split is NEVER touched here — reserved for final evaluation only.
    x_train = x[train_mask]
    y_train = y_pathology[train_mask]
    meta_train = metadata[train_mask].reset_index(drop=True)
    age_group_train = age_group[train_mask]

    # Internal SAE validation set: a 10% hold-out from training data only.
    # This keeps the true validation split completely uncontaminated.
    rng = np.random.default_rng(cfg.seed)
    n_train = len(x_train)
    n_sae_val = max(1, int(n_train * 0.10))
    sae_val_idx = rng.choice(n_train, size=n_sae_val, replace=False)
    sae_train_idx = np.setdiff1d(np.arange(n_train), sae_val_idx)
    x_sae_train = x_train[sae_train_idx]
    x_sae_val   = x_train[sae_val_idx]
    logger.info(
        "Internal SAE split (from train only): sae_train=%d  sae_val=%d",
        len(x_sae_train), len(x_sae_val),
    )

    # Attribute arrays for η² and activation-distribution plots (training set only).
    attr_arrays_train: dict[str, np.ndarray] = {
        "age_group": _clean_attr(age_group_train),
    }
    for attr in ["sex", "race", "insurance_type"]:
        if attr in meta_train.columns:
            attr_arrays_train[attr] = _clean_attr(meta_train[attr].to_numpy())
        else:
            logger.warning("Attribute '%s' not found in train metadata; skipping.", attr)

    attr_names = list(attr_arrays_train.keys())
    safe_paths = [_safe(p) for p in pathology_cols]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    epochs = args.epochs
    input_dim = x_train.shape[1]

    logger.info(
        "Training %d SAEs (k ∈ %s × latent_dim ∈ %s), %d epochs, input_dim=%d",
        len(K_VALUES) * len(DIM_VALUES), K_VALUES, DIM_VALUES, epochs, input_dim,
    )

    grid_results: list[dict] = []

    for k, latent_dim in product(K_VALUES, DIM_VALUES):
        run_label = f"k{k}_d{latent_dim}"
        run_out = base_out / run_label
        pair_out = run_out / "per_pair"
        pair_out.mkdir(parents=True, exist_ok=True)

        logger.info("─── SAE %s ───", run_label)

        model = train_topk_sae(
            x_train=x_sae_train,
            x_valid=x_sae_val,
            latent_dim=latent_dim,
            k=k,
            epochs=epochs,
            device=device,
        )

        # Save checkpoint so the intervention pipeline can reload this SAE
        torch.save(
            {"state_dict": model.state_dict(), "input_dim": int(x_train.shape[1]),
             "latent_dim": int(latent_dim), "k": int(k)},
            run_out / "sae_checkpoint.pt",
        )
        logger.info("  Saved sae_checkpoint.pt")

        z_train_enc = encode_all(model, x_train, device)

        logger.info(
            "  Computing η² for %d concepts × %d targets (on training set)…",
            latent_dim,
            len(pathology_cols) + len(attr_arrays_train),
        )
        scores_df = compute_concept_scores(z_train_enc, y_train, pathology_cols, attr_arrays_train)
        scores_df.to_csv(run_out / "concept_scores.csv", index=False)
        logger.info("  Saved concept_scores.csv (%d rows, %d cols)", len(scores_df), len(scores_df.columns))

        # ── Per-pair top-10 plots ─────────────────────────────────────────────
        n_pair_plots = 0
        for attr in attr_names:
            for pathology in pathology_cols:
                _plot_top10_per_pair(scores_df, attr, pathology, k, latent_dim, pair_out)
                n_pair_plots += 1
        logger.info("  Saved %d per-pair top-10 plots → per_pair/", n_pair_plots)

        # ── Overview heatmap ──────────────────────────────────────────────────
        _plot_pair_overview_heatmap(scores_df, attr_names, pathology_cols, k, latent_dim, run_out)

        # ── Activation distribution plots ─────────────────────────────────────
        for attr in attr_names:
            _plot_activation_dist(z_train_enc, scores_df, attr, attr_arrays_train[attr], k, latent_dim, run_out)

        n_path_plots = _plot_activation_dist_pathology(
            z_train_enc, y_train, pathology_cols, scores_df, k, latent_dim, run_out
        )
        logger.info("  Saved %d pathology activation distribution plots", n_path_plots)

        # ── Collect grid summary stats ────────────────────────────────────────
        result: dict = {"k": k, "latent_dim": latent_dim}

        pair_best_specs: list[float] = []
        for attr in attr_names:
            for pathology, safe_p in zip(pathology_cols, safe_paths):
                col = f"spec_{attr}_{safe_p}"
                if col in scores_df.columns:
                    best = float(scores_df[col].dropna().max())
                else:
                    best = float("nan")
                result[f"best_spec_{attr}_{safe_p}"] = best
                if np.isfinite(best):
                    pair_best_specs.append(best)

        result["mean_pair_specificity"] = float(np.mean(pair_best_specs)) if pair_best_specs else float("nan")
        result["mean_demo_eta2"] = float(scores_df["mean_demo_eta2"].dropna().mean())
        result["mean_path_eta2"] = float(scores_df["mean_path_eta2"].dropna().mean())

        logger.info(
            "  mean_pair_spec=%.4f  mean_demo_η²=%.4f  mean_path_η²=%.4f",
            result["mean_pair_specificity"],
            result["mean_demo_eta2"],
            result["mean_path_eta2"],
        )
        grid_results.append(result)

    # ── Grid-level outputs ─────────────────────────────────────────────────────
    logger.info("Generating grid-level plots…")

    grid_df = pd.DataFrame(grid_results).sort_values(["k", "latent_dim"])
    grid_df.to_csv(grid_out / "grid_summary.csv", index=False)
    logger.info("Saved → grid_summary.csv")

    # Per-pair grid heatmaps + build best-SAE lookup
    best_sae_rows: list[dict] = []
    pair_heatmap_out = grid_out / "pair_heatmaps"

    for attr in attr_names:
        for pathology, safe_p in zip(pathology_cols, safe_paths):
            _plot_pair_grid_heatmap(grid_results, attr, pathology, pair_heatmap_out)

            # Find best SAE for this pair
            col = f"best_spec_{attr}_{safe_p}"
            best_val = float("-inf")
            best_k, best_dim = -1, -1
            for r in grid_results:
                v = r.get(col, float("nan"))
                if np.isfinite(v) and v > best_val:
                    best_val = v
                    best_k = r["k"]
                    best_dim = r["latent_dim"]
            best_sae_rows.append({
                "attr": attr,
                "pathology": pathology,
                "best_k": best_k,
                "best_dim": best_dim,
                "best_spec": best_val if best_val > float("-inf") else float("nan"),
            })

    logger.info("Saved %d per-pair grid heatmaps → grid/pair_heatmaps/", len(best_sae_rows))

    best_sae_df = pd.DataFrame(best_sae_rows)
    best_sae_df.to_csv(grid_out / "best_sae_per_pair.csv", index=False)
    logger.info("Saved → best_sae_per_pair.csv")

    _plot_best_sae_heatmap(best_sae_df, attr_names, pathology_cols, grid_out)
    _plot_specificity_heatmap(grid_results, grid_out)
    _plot_demo_vs_path_scatter(grid_results, grid_out)

    logger.info("Done. All outputs in %s", base_out.resolve())

    if not args.skip_interventions:
        _run_all_interventions(cfg, base_out, args)


if __name__ == "__main__":
    main()
