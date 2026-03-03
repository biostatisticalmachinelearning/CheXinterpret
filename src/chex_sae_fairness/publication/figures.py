from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from chex_sae_fairness.reporting.figures import generate_study_figures, generate_sweep_figures


def generate_core_publication_figures(
    sweep_summary: pd.DataFrame,
    best_report: dict[str, Any],
    output_dir: str | Path,
) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    generated.extend(generate_sweep_figures(sweep_summary, out / "sweep"))
    generated.extend(generate_study_figures(best_report, out / "best_model"))
    return generated


def generate_supplement_figures(
    output_dir: str | Path,
    seed_stability: pd.DataFrame | None = None,
    uncertain_policy: pd.DataFrame | None = None,
    debias_ablation: pd.DataFrame | None = None,
    age_bin_sensitivity: pd.DataFrame | None = None,
    baseline_comparison: pd.DataFrame | None = None,
    threshold_sensitivity: pd.DataFrame | None = None,
    missingness_sensitivity: pd.DataFrame | None = None,
    permutation_control: pd.DataFrame | None = None,
) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _set_style()
    generated: list[Path] = []

    if seed_stability is not None and not seed_stability.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=seed_stability, x="method", y="macro_auroc", ax=ax, color="#8DA0CB")
        sns.stripplot(data=seed_stability, x="method", y="macro_auroc", ax=ax, color="#1B3C73", alpha=0.75)
        ax.set_title("Seed Stability: Macro AUROC")
        ax.set_xlabel("Method")
        ax.set_ylabel("Macro AUROC")
        generated.append(_save(fig, out / "seed_stability_macro_auroc.png"))

    if uncertain_policy is not None and not uncertain_policy.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=uncertain_policy,
            x="uncertain_policy",
            y="macro_auroc",
            hue="method",
            palette="Set2",
            ax=ax,
        )
        ax.set_title("Uncertain Label Policy Sensitivity")
        ax.set_xlabel("Uncertain label policy")
        ax.set_ylabel("Macro AUROC")
        generated.append(_save(fig, out / "uncertain_policy_sensitivity.png"))

    if debias_ablation is not None and not debias_ablation.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=debias_ablation,
            x="debias_strength",
            y="worst_group_macro_auroc",
            hue="debias_mode",
            marker="o",
            ax=ax,
        )
        ax.set_title("Debias Strength/Mode Sensitivity")
        ax.set_xlabel("Debias strength")
        ax.set_ylabel("Worst-group macro AUROC")
        generated.append(_save(fig, out / "debias_mode_strength_sensitivity.png"))

    if age_bin_sensitivity is not None and not age_bin_sensitivity.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=age_bin_sensitivity,
            x="age_bins",
            y="macro_auroc_gap",
            hue="method",
            palette="Set2",
            ax=ax,
        )
        ax.set_title("Age Bin Sensitivity (Macro AUROC Gap)")
        ax.set_xlabel("Age bin scheme")
        ax.set_ylabel("Macro AUROC gap")
        ax.tick_params(axis="x", rotation=25)
        generated.append(_save(fig, out / "age_bin_sensitivity.png"))

    if baseline_comparison is not None and not baseline_comparison.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=baseline_comparison, x="method", y="macro_auroc", palette="Set2", ax=ax)
        ax.set_title("Alternative Baselines: Macro AUROC")
        ax.set_xlabel("Method")
        ax.set_ylabel("Macro AUROC")
        ax.tick_params(axis="x", rotation=20)
        generated.append(_save(fig, out / "alternative_baselines_macro_auroc.png"))

    if threshold_sensitivity is not None and not threshold_sensitivity.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=threshold_sensitivity,
            x="threshold",
            y="macro_auroc_gap",
            hue="method",
            marker="o",
            ax=ax,
        )
        ax.set_title("Fairness Threshold Sensitivity")
        ax.set_xlabel("Classification threshold")
        ax.set_ylabel("Macro AUROC gap")
        generated.append(_save(fig, out / "threshold_sensitivity.png"))

    if missingness_sensitivity is not None and not missingness_sensitivity.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=missingness_sensitivity,
            x="missing_fraction",
            y="macro_auroc",
            hue="method",
            marker="o",
            ax=ax,
        )
        ax.set_title("Missingness Sensitivity")
        ax.set_xlabel("Simulated missing fraction")
        ax.set_ylabel("Macro AUROC")
        generated.append(_save(fig, out / "missingness_sensitivity.png"))

    if permutation_control is not None and not permutation_control.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(
            permutation_control["null_mean_pathology_corr"],
            bins=20,
            color="#9ECAE1",
            edgecolor="white",
            ax=ax,
        )
        observed = permutation_control["observed_mean_pathology_corr"].iloc[0]
        ax.axvline(observed, color="#C0392B", linestyle="--", linewidth=2)
        ax.set_title("Permutation Control: Pathology Correlation Null")
        ax.set_xlabel("Mean pathology max |corr| under permutation")
        ax.set_ylabel("Count")
        generated.append(_save(fig, out / "permutation_control_histogram.png"))

    return generated


def _set_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
        }
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
