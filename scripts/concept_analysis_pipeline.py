#!/usr/bin/env python3
"""SAE concept analysis pipeline.

Trains a grid of Top-k Sparse Autoencoders (k × latent_dim) on CheXagent embeddings,
then measures how strongly each SAE latent (concept) correlates with demographic
variables vs. pathology labels, using Eta-squared (η²) throughout.

Grid: k ∈ {8, 16, 32, 64}  ×  latent_dim ∈ {256, 512, 1024, 2048, 4096}  → 20 SAEs

Outputs (written to <output_root>/presentation/sae-eval/):
  Per-SAE sub-directories (k<k>_d<dim>/):
    concept_scores.csv          — η² columns for every concept × every target
    top10_per_demo_<attr>.png   — top-10 concepts for each demographic attribute
    top10_per_path_<path>.png   — top-10 concepts for each pathology
    activation_dist_<attr>.png  — activation distributions for top-5 demo concepts

  Grid-level plots (grid/):
    specificity_heatmap.png     — mean specificity across k × latent_dim grid
    scatter_demo_vs_path.png    — demo_η² vs path_η² scatter for all SAEs

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

# Sensitive attributes to analyse
SENSITIVE_ATTRS: list[str] = ["sex", "age_group", "race", "insurance_type"]

# Top-N concepts shown in per-SAE plots
TOP_N: int = 10
# Top concepts shown in activation distribution plots
TOP_DIST: int = 5


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


# ── SAE training ───────────────────────────────────────────────────────────────

class TopKSAE(torch.nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, k: int) -> None:
        super().__init__()
        self.k = k
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decoder(z), z


def train_topk_sae(
    x_train: np.ndarray,
    x_valid: np.ndarray,
    latent_dim: int,
    k: int,
    epochs: int,
    device: str,
) -> TopKSAE:
    input_dim = x_train.shape[1]
    model = TopKSAE(input_dim, latent_dim, k).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SAE_LR, weight_decay=SAE_WEIGHT_DECAY)

    train_tensor = torch.tensor(x_train, dtype=torch.float32)
    valid_tensor = torch.tensor(x_valid, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=SAE_BATCH_SIZE, shuffle=True)

    best_valid_loss = float("inf")
    best_state: dict | None = None

    for epoch in range(epochs):
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device)
            x_hat, z = model(batch)
            loss = torch.mean((batch - x_hat) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # Validation loss
        model.eval()
        with torch.no_grad():
            v = valid_tensor.to(device)
            x_hat_v, _ = model(v)
            valid_loss = float(torch.mean((v - x_hat_v) ** 2).item())

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = {k_: v_.cpu().clone() for k_, v_ in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def encode_all(model: TopKSAE, x: np.ndarray, device: str) -> np.ndarray:
    model.eval().to(device)
    tensor = torch.tensor(x, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=SAE_BATCH_SIZE * 4, shuffle=False)
    parts: list[np.ndarray] = []
    for (batch,) in loader:
        z = model.encode(batch.to(device))
        parts.append(z.cpu().numpy().astype(np.float32))
    return np.concatenate(parts, axis=0)


# ── Eta-squared (η²) ──────────────────────────────────────────────────────────

def eta_squared_binary(z_col: np.ndarray, binary_target: np.ndarray) -> float:
    """η² = SS_between / SS_total for a binary group variable.

    Equivalent to r² (squared point-biserial correlation).
    """
    valid = np.isfinite(z_col) & np.isfinite(binary_target)
    if valid.sum() < 3:
        return float("nan")
    z = z_col[valid]
    y = binary_target[valid].astype(float)
    grand_mean = z.mean()
    ss_total = np.sum((z - grand_mean) ** 2)
    if ss_total < 1e-12:
        return 0.0
    groups = np.unique(y)
    ss_between = sum(
        np.sum(y == g) * (z[y == g].mean() - grand_mean) ** 2
        for g in groups
    )
    return float(np.clip(ss_between / ss_total, 0.0, 1.0))


def eta_squared_categorical(z_col: np.ndarray, cat_target: np.ndarray) -> float:
    """η² for a categorical group variable (any number of classes)."""
    valid_mask = np.array([v is not np.nan and str(v) not in {"nan", "None", "none", ""} for v in cat_target])
    valid_mask &= np.isfinite(z_col)
    if valid_mask.sum() < 3:
        return float("nan")
    z = z_col[valid_mask]
    cats = cat_target[valid_mask].astype(str)
    grand_mean = z.mean()
    ss_total = np.sum((z - grand_mean) ** 2)
    if ss_total < 1e-12:
        return 0.0
    groups = np.unique(cats)
    ss_between = sum(
        np.sum(cats == g) * (z[cats == g].mean() - grand_mean) ** 2
        for g in groups
    )
    return float(np.clip(ss_between / ss_total, 0.0, 1.0))


def compute_concept_scores(
    z: np.ndarray,
    y_pathology: np.ndarray,
    pathology_cols: list[str],
    attr_arrays: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compute η² for every concept dimension vs every target.

    Returns a DataFrame with one row per concept and columns:
      concept_idx, path_eta2_<path>, demo_eta2_<attr>, max_demo_eta2,
      max_path_eta2, specificity
    """
    n_concepts = z.shape[1]
    rows: list[dict] = []

    for c_idx in range(n_concepts):
        z_col = z[:, c_idx]
        row: dict = {"concept_idx": c_idx}

        # Pathology η² (binary targets)
        path_eta2_vals: list[float] = []
        for p_idx, pname in enumerate(pathology_cols):
            y_bin = y_pathology[:, p_idx].astype(float)
            val = eta_squared_binary(z_col, y_bin)
            row[f"path_eta2_{pname}"] = val
            if np.isfinite(val):
                path_eta2_vals.append(val)

        # Demographic η²
        demo_eta2_vals: list[float] = []
        for attr, attr_vals in attr_arrays.items():
            val = eta_squared_categorical(z_col, attr_vals)
            row[f"demo_eta2_{attr}"] = val
            if np.isfinite(val):
                demo_eta2_vals.append(val)

        row["max_demo_eta2"] = float(np.max(demo_eta2_vals)) if demo_eta2_vals else float("nan")
        row["max_path_eta2"] = float(np.max(path_eta2_vals)) if path_eta2_vals else float("nan")
        row["mean_demo_eta2"] = float(np.mean(demo_eta2_vals)) if demo_eta2_vals else float("nan")
        row["mean_path_eta2"] = float(np.mean(path_eta2_vals)) if path_eta2_vals else float("nan")
        row["specificity"] = (
            row["max_demo_eta2"] - row["max_path_eta2"]
            if np.isfinite(row["max_demo_eta2"]) and np.isfinite(row["max_path_eta2"])
            else float("nan")
        )
        rows.append(row)

    return pd.DataFrame(rows)


# ── Per-SAE plots ──────────────────────────────────────────────────────────────

def _plot_top10_per_demo(
    scores_df: pd.DataFrame,
    attr: str,
    k: int,
    latent_dim: int,
    out_dir: Path,
) -> None:
    """Horizontal bar chart: top-10 concepts by demo_η² for one attribute.

    Each bar cluster shows demo_η², path_η², and specificity.
    """
    demo_col = f"demo_eta2_{attr}"
    if demo_col not in scores_df.columns:
        return

    sub = scores_df[[demo_col, "max_path_eta2", "specificity", "concept_idx"]].dropna(
        subset=[demo_col]
    )
    top = sub.nlargest(TOP_N, demo_col).reset_index(drop=True)
    if top.empty:
        return

    labels = [f"c{int(r.concept_idx)}" for _, r in top.iterrows()]
    demo_vals = top[demo_col].tolist()
    path_vals = top["max_path_eta2"].tolist()
    spec_vals = top["specificity"].tolist()

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.5)))
    ax.bar(x - width, demo_vals, width, label=f"demo η² ({attr})", color="#4C78A8", alpha=0.85)
    ax.bar(x, path_vals, width, label="max path η²", color="#F58518", alpha=0.85)
    ax.bar(x + width, spec_vals, width, label="specificity (demo−path)", color="#54A24B", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("η²")
    ax.set_ylim(bottom=0)
    ax.set_title(
        f"Top-{TOP_N} concepts by demo η² ({attr.replace('_', ' ')})\n"
        f"SAE: k={k}, latent_dim={latent_dim}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    safe = attr.replace(" ", "_")
    path = out_dir / f"top10_per_demo_{safe}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved → %s", path.name)


def _plot_top10_per_pathology(
    scores_df: pd.DataFrame,
    pathology: str,
    k: int,
    latent_dim: int,
    out_dir: Path,
) -> None:
    """Top-10 concepts with highest demo η² that also have non-trivial path η² for this pathology."""
    path_col = f"path_eta2_{pathology}"
    if path_col not in scores_df.columns:
        return

    # Rank by demo η² (demographic specificity)
    sub = scores_df[["max_demo_eta2", path_col, "specificity", "concept_idx"]].dropna(
        subset=["max_demo_eta2", path_col]
    )
    # Show concepts most associated with this pathology
    top = sub.nlargest(TOP_N, path_col).reset_index(drop=True)
    if top.empty:
        return

    labels = [f"c{int(r.concept_idx)}" for _, r in top.iterrows()]
    demo_vals = top["max_demo_eta2"].tolist()
    path_vals = top[path_col].tolist()
    spec_vals = top["specificity"].tolist()

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.5)))
    ax.bar(x - width, demo_vals, width, label="max demo η²", color="#4C78A8", alpha=0.85)
    ax.bar(x, path_vals, width, label=f"path η² ({pathology})", color="#F58518", alpha=0.85)
    ax.bar(x + width, spec_vals, width, label="specificity (demo−path)", color="#54A24B", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("η²")
    ax.set_ylim(bottom=0)
    safe_path = pathology.replace(" ", "_").replace("/", "-")
    ax.set_title(
        f"Top-{TOP_N} concepts for {pathology}\n"
        f"SAE: k={k}, latent_dim={latent_dim}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / f"top10_per_path_{safe_path}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved → %s", path.name)


def _plot_activation_dist(
    z: np.ndarray,
    scores_df: pd.DataFrame,
    attr: str,
    attr_vals: np.ndarray,
    k: int,
    latent_dim: int,
    out_dir: Path,
) -> None:
    """Box plots of top-5 demo concepts' activation distributions, split by group."""
    demo_col = f"demo_eta2_{attr}"
    if demo_col not in scores_df.columns:
        return

    top5 = (
        scores_df[[demo_col, "concept_idx"]]
        .dropna(subset=[demo_col])
        .nlargest(TOP_DIST, demo_col)
    )
    if top5.empty:
        return

    groups = sorted(np.unique(attr_vals[~pd.isna(attr_vals)].astype(str)))
    palette = sns.color_palette("tab10", len(groups))

    fig, axes = plt.subplots(1, len(top5), figsize=(4 * len(top5), 4), squeeze=False)

    for ax_i, (_, row) in enumerate(top5.iterrows()):
        c_idx = int(row["concept_idx"])
        ax = axes[0][ax_i]
        data_per_group = []
        for g in groups:
            mask = attr_vals.astype(str) == g
            data_per_group.append(z[mask, c_idx])
        bp = ax.boxplot(
            data_per_group,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=7)
        ax.set_title(f"c{c_idx}\nη²={row[demo_col]:.3f}", fontsize=8)
        ax.set_ylabel("Activation" if ax_i == 0 else "")

    safe = attr.replace(" ", "_")
    fig.suptitle(
        f"Top-{TOP_DIST} concept activations by {attr.replace('_', ' ')}\n"
        f"SAE: k={k}, latent_dim={latent_dim}",
        fontsize=10,
    )
    fig.tight_layout()
    path = out_dir / f"activation_dist_{safe}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved → %s", path.name)


# ── Grid-level plots ───────────────────────────────────────────────────────────

def _plot_specificity_heatmap(
    grid_results: list[dict],
    grid_out: Path,
) -> None:
    """Heatmap of mean concept specificity across the k × latent_dim grid."""
    k_vals = sorted({r["k"] for r in grid_results})
    d_vals = sorted({r["latent_dim"] for r in grid_results})

    matrix = np.full((len(k_vals), len(d_vals)), fill_value=float("nan"))
    for r in grid_results:
        ki = k_vals.index(r["k"])
        di = d_vals.index(r["latent_dim"])
        matrix[ki, di] = r["mean_specificity"]

    fig, ax = plt.subplots(figsize=(len(d_vals) * 1.8 + 1.5, len(k_vals) * 1.5 + 1.5))
    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=[str(d) for d in d_vals],
        yticklabels=[str(k) for k in k_vals],
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": "Mean concept specificity (demo η² − path η²)"},
        linewidths=0.5,
    )
    ax.set_xlabel("Latent dim")
    ax.set_ylabel("Top-k (k)")
    ax.set_title("Mean Concept Specificity Across SAE Grid\n(higher = more demographic-specific concepts)")
    fig.tight_layout()
    path = grid_out / "specificity_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved → %s", path)


def _plot_demo_vs_path_scatter(
    grid_results: list[dict],
    grid_out: Path,
) -> None:
    """Per-SAE scatter: mean demo η² vs mean path η², coloured by specificity."""
    if not grid_results:
        return

    df = pd.DataFrame(grid_results)
    palette = sns.color_palette("coolwarm_r", as_cmap=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        df["mean_path_eta2"],
        df["mean_demo_eta2"],
        c=df["mean_specificity"],
        cmap=palette,
        s=120,
        edgecolors="k",
        linewidths=0.5,
        zorder=3,
    )
    plt.colorbar(sc, ax=ax, label="mean specificity")
    ax.plot([0, df[["mean_path_eta2", "mean_demo_eta2"]].max().max()],
            [0, df[["mean_path_eta2", "mean_demo_eta2"]].max().max()],
            "grey", linestyle="--", linewidth=0.8, zorder=2)

    for _, row in df.iterrows():
        ax.annotate(
            f"k={int(row['k'])}\nd={int(row['latent_dim'])}",
            (row["mean_path_eta2"], row["mean_demo_eta2"]),
            fontsize=6,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlabel("Mean path η² (pathology correlation)")
    ax.set_ylabel("Mean demo η² (demographic correlation)")
    ax.set_title("Demo η² vs Path η² per SAE\n(above dashed = more demographic than pathology signal)")
    fig.tight_layout()
    path = grid_out / "scatter_demo_vs_path.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved → %s", path)


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
    grid_out.mkdir(exist_ok=True)
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

    logger.info(
        "Split sizes — train: %d  valid: %d",
        train_mask.sum(), valid_mask.sum(),
    )

    x_train = x[train_mask]
    x_valid = x[valid_mask]

    # Use validation set for η² analysis (held-out)
    y_val = y_pathology[valid_mask]
    meta_val = metadata[valid_mask].reset_index(drop=True)
    age_group_val = age_group[valid_mask]

    # Build attribute arrays for the validation set
    attr_arrays: dict[str, np.ndarray] = {"age_group": _clean_attr(age_group_val)}
    for attr in ["sex", "race", "insurance_type"]:
        if attr in meta_val.columns:
            attr_arrays[attr] = _clean_attr(meta_val[attr].to_numpy())
        else:
            logger.warning("Attribute '%s' not found in metadata; skipping.", attr)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    epochs = args.epochs
    input_dim = x_train.shape[1]
    logger.info(
        "Training %d SAEs (k ∈ %s × latent_dim ∈ %s), %d epochs each, input_dim=%d",
        len(K_VALUES) * len(DIM_VALUES),
        K_VALUES,
        DIM_VALUES,
        epochs,
        input_dim,
    )

    grid_results: list[dict] = []

    for k, latent_dim in product(K_VALUES, DIM_VALUES):
        run_label = f"k{k}_d{latent_dim}"
        run_out = base_out / run_label
        run_out.mkdir(exist_ok=True)

        logger.info("─── SAE %s ───", run_label)

        # Train
        model = train_topk_sae(
            x_train=x_train,
            x_valid=x_valid,
            latent_dim=latent_dim,
            k=k,
            epochs=epochs,
            device=device,
        )

        # Encode validation set
        z_val = encode_all(model, x_valid, device)

        # Compute concept scores
        logger.info("  Computing η² for %d concepts × %d targets…", latent_dim, len(pathology_cols) + len(attr_arrays))
        scores_df = compute_concept_scores(z_val, y_val, pathology_cols, attr_arrays)
        scores_csv = run_out / "concept_scores.csv"
        scores_df.to_csv(scores_csv, index=False)
        logger.info("  Saved concept scores → %s", scores_csv.name)

        mean_spec = float(scores_df["specificity"].dropna().mean())
        mean_demo = float(scores_df["mean_demo_eta2"].dropna().mean())
        mean_path = float(scores_df["mean_path_eta2"].dropna().mean())
        logger.info(
            "  mean_specificity=%.4f  mean_demo_η²=%.4f  mean_path_η²=%.4f",
            mean_spec, mean_demo, mean_path,
        )

        grid_results.append({
            "k": k,
            "latent_dim": latent_dim,
            "mean_specificity": mean_spec,
            "mean_demo_eta2": mean_demo,
            "mean_path_eta2": mean_path,
        })

        # Per-demographic plots
        for attr in list(attr_arrays.keys()):
            _plot_top10_per_demo(scores_df, attr, k, latent_dim, run_out)
            _plot_activation_dist(z_val, scores_df, attr, attr_arrays[attr], k, latent_dim, run_out)

        # Per-pathology plots
        for pathology in pathology_cols:
            _plot_top10_per_pathology(scores_df, pathology, k, latent_dim, run_out)

    # ── Grid-level plots ───────────────────────────────────────────────────────
    logger.info("Generating grid-level plots…")
    _plot_specificity_heatmap(grid_results, grid_out)
    _plot_demo_vs_path_scatter(grid_results, grid_out)

    # Save grid summary CSV
    grid_df = pd.DataFrame(grid_results).sort_values(["k", "latent_dim"])
    grid_csv = grid_out / "grid_summary.csv"
    grid_df.to_csv(grid_csv, index=False)
    logger.info("Saved grid summary → %s", grid_csv)

    logger.info("Done. All outputs in %s", base_out.resolve())


if __name__ == "__main__":
    main()
