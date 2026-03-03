from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chex_sae_fairness.evaluation.fairness import evaluate_multilabel_predictions
from chex_sae_fairness.publication.statistics import bootstrap_core_metrics


def write_table(df: pd.DataFrame, output_path_no_ext: str | Path) -> dict[str, str]:
    base = Path(output_path_no_ext)
    base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = base.with_suffix(".csv")
    md_path = base.with_suffix(".md")
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
    return {"csv": str(csv_path.resolve()), "markdown": str(md_path.resolve())}


def build_core_table_cohort(
    report: dict[str, Any],
    prediction_bundle: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    counts = report.get("counts", {})
    rows: list[dict[str, Any]] = []
    if isinstance(counts, dict):
        rows.extend(
            [
                {"section": "split_counts", "item": "n_total", "value": counts.get("n_total")},
                {"section": "split_counts", "item": "n_train", "value": counts.get("n_train")},
                {"section": "split_counts", "item": "n_valid", "value": counts.get("n_valid")},
                {"section": "split_counts", "item": "n_test", "value": counts.get("n_test")},
            ]
        )
    if prediction_bundle is not None:
        groups = prediction_bundle.get("age_groups")
        y_true = prediction_bundle.get("y_true")
        if groups is not None:
            unique, freq = np.unique(groups.astype(str), return_counts=True)
            for group, count in zip(unique.tolist(), freq.tolist()):
                rows.append({"section": "test_age_groups", "item": str(group), "value": int(count)})
        if y_true is not None:
            prevalence = np.mean(y_true.astype(float), axis=0)
            labels = prediction_bundle.get("pathology_cols")
            if labels is not None and len(labels) == len(prevalence):
                for label, prev in zip(labels.tolist(), prevalence.tolist()):
                    rows.append(
                        {
                            "section": "test_pathology_prevalence",
                            "item": str(label),
                            "value": float(prev),
                        }
                    )
    return pd.DataFrame(rows)


def build_core_table_main_results(
    report: dict[str, Any],
    prediction_bundle: dict[str, np.ndarray],
    threshold: float,
    bootstrap_samples: int = 300,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = [str(v) for v in prediction_bundle["pathology_cols"].tolist()]
    y_true = prediction_bundle["y_true"].astype(np.float32)
    groups = prediction_bundle["age_groups"].astype(str)
    method_specs = [
        ("baseline_feature_probe", "Baseline", "baseline_scores"),
        ("sae_concept_probe", "SAE Concepts", "concept_scores"),
        ("sae_concept_probe_debiased", "SAE Concepts (Debiased)", "debiased_scores"),
    ]
    for report_key, method_label, score_key in method_specs:
        perf = report.get(report_key, {}).get("performance", {})
        fair = report.get(report_key, {}).get("fairness", {})
        scores = prediction_bundle[score_key].astype(np.float32)
        ci = bootstrap_core_metrics(
            y_true=y_true,
            y_score=scores,
            age_groups=groups,
            label_names=labels,
            threshold=threshold,
            n_bootstrap=bootstrap_samples,
        )
        rows.append(
            {
                "method": method_label,
                "endpoint_primary_worst_group_macro_auroc": _worst_group_value(fair.get("worst_group_macro_auroc")),
                "endpoint_primary_worst_group_macro_auroc_ci_low": ci.get("worst_group_macro_auroc", {}).get("ci_low"),
                "endpoint_primary_worst_group_macro_auroc_ci_high": ci.get("worst_group_macro_auroc", {}).get("ci_high"),
                "endpoint_primary_macro_auroc_gap": fair.get("macro_auroc_gap"),
                "endpoint_primary_macro_auroc_gap_ci_low": ci.get("macro_auroc_gap", {}).get("ci_low"),
                "endpoint_primary_macro_auroc_gap_ci_high": ci.get("macro_auroc_gap", {}).get("ci_high"),
                "endpoint_secondary_macro_auroc": perf.get("macro_auroc"),
                "endpoint_secondary_macro_auroc_ci_low": ci.get("macro_auroc", {}).get("ci_low"),
                "endpoint_secondary_macro_auroc_ci_high": ci.get("macro_auroc", {}).get("ci_high"),
                "endpoint_secondary_macro_accuracy": perf.get("macro_accuracy"),
                "endpoint_secondary_macro_accuracy_ci_low": ci.get("macro_accuracy", {}).get("ci_low"),
                "endpoint_secondary_macro_accuracy_ci_high": ci.get("macro_accuracy", {}).get("ci_high"),
                "endpoint_secondary_macro_brier": perf.get("macro_brier"),
                "endpoint_secondary_macro_ece": perf.get("macro_ece"),
                "macro_accuracy_gap": fair.get("macro_accuracy_gap"),
                "macro_auroc": perf.get("macro_auroc"),
                "macro_auroc_ci_low": ci.get("macro_auroc", {}).get("ci_low"),
                "macro_auroc_ci_high": ci.get("macro_auroc", {}).get("ci_high"),
                "macro_accuracy": perf.get("macro_accuracy"),
                "macro_accuracy_ci_low": ci.get("macro_accuracy", {}).get("ci_low"),
                "macro_accuracy_ci_high": ci.get("macro_accuracy", {}).get("ci_high"),
                "worst_group_macro_auroc": _worst_group_value(fair.get("worst_group_macro_auroc")),
                "worst_group_macro_auroc_ci_low": ci.get("worst_group_macro_auroc", {}).get("ci_low"),
                "worst_group_macro_auroc_ci_high": ci.get("worst_group_macro_auroc", {}).get("ci_high"),
                "macro_auroc_gap": fair.get("macro_auroc_gap"),
                "macro_auroc_gap_ci_low": ci.get("macro_auroc_gap", {}).get("ci_low"),
                "macro_auroc_gap_ci_high": ci.get("macro_auroc_gap", {}).get("ci_high"),
            }
        )
    return pd.DataFrame(rows)


def build_core_table_intervention_ablation(ablations: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in ablations:
        row = dict(result)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        by=["debias_mode", "debias_strength"],
        ascending=[True, True],
    )


def build_supplement_table_seed_stability(seed_runs: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(seed_runs).sort_values(by=["seed"])


def build_supplement_table_ablations(entries: list[dict[str, Any]], sort_by: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(entries)
    if frame.empty:
        return frame
    cols = [col for col in sort_by if col in frame.columns]
    if cols:
        return frame.sort_values(by=cols)
    return frame


def build_core_table_paired_tests(test_rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(test_rows)
    if frame.empty:
        return frame
    sort_cols = [col for col in ["metric", "method_a", "method_b"] if col in frame.columns]
    if sort_cols:
        frame = frame.sort_values(by=sort_cols)
    return frame


def build_core_table_group_fairness(
    report: dict[str, Any],
    prediction_bundle: dict[str, np.ndarray],
    threshold: float,
    bootstrap_samples: int = 300,
) -> pd.DataFrame:
    y_true = prediction_bundle["y_true"].astype(np.float32)
    groups = prediction_bundle["age_groups"].astype(str)
    labels = [str(v) for v in prediction_bundle["pathology_cols"].tolist()]
    method_specs = [
        ("Baseline", "baseline_scores"),
        ("SAE Concepts", "concept_scores"),
        ("SAE Concepts (Debiased)", "debiased_scores"),
    ]

    rows: list[dict[str, Any]] = []
    for method_label, score_key in method_specs:
        scores = prediction_bundle[score_key].astype(np.float32)
        for group in sorted(np.unique(groups).tolist()):
            mask = groups == group
            if int(mask.sum()) == 0:
                continue
            perf = evaluate_multilabel_predictions(
                y_true=y_true[mask],
                y_score=scores[mask],
                label_names=labels,
                threshold=threshold,
            )
            ci = _bootstrap_group_metrics(
                y_true=y_true[mask],
                y_score=scores[mask],
                labels=labels,
                threshold=threshold,
                n_bootstrap=bootstrap_samples,
                seed=13,
            )
            rows.append(
                {
                    "method": method_label,
                    "age_group": str(group),
                    "n_samples": int(mask.sum()),
                    "macro_auroc": perf.get("macro_auroc"),
                    "macro_auroc_ci_low": ci.get("macro_auroc", {}).get("ci_low"),
                    "macro_auroc_ci_high": ci.get("macro_auroc", {}).get("ci_high"),
                    "macro_accuracy": perf.get("macro_accuracy"),
                    "macro_accuracy_ci_low": ci.get("macro_accuracy", {}).get("ci_low"),
                    "macro_accuracy_ci_high": ci.get("macro_accuracy", {}).get("ci_high"),
                    "macro_tpr": report.get(
                        _method_label_to_report_key(method_label),
                        {},
                    ).get("fairness", {}).get("groups", {}).get(str(group), {}).get("macro_tpr"),
                    "macro_tpr_ci_low": ci.get("macro_tpr", {}).get("ci_low"),
                    "macro_tpr_ci_high": ci.get("macro_tpr", {}).get("ci_high"),
                    "macro_fpr": report.get(
                        _method_label_to_report_key(method_label),
                        {},
                    ).get("fairness", {}).get("groups", {}).get(str(group), {}).get("macro_fpr"),
                    "macro_fpr_ci_low": ci.get("macro_fpr", {}).get("ci_low"),
                    "macro_fpr_ci_high": ci.get("macro_fpr", {}).get("ci_high"),
                }
            )
    return pd.DataFrame(rows)


def _worst_group_value(entry: Any) -> float:
    if isinstance(entry, dict):
        value = entry.get("value")
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
    return float("nan")


def _bootstrap_group_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: list[str],
    threshold: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if n_bootstrap <= 0 or len(y_true) == 0:
        return {}
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y_true))
    auroc_values: list[float] = []
    acc_values: list[float] = []
    tpr_values: list[float] = []
    fpr_values: list[float] = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(indices, size=len(indices), replace=True)
        perf = evaluate_multilabel_predictions(
            y_true=y_true[sampled],
            y_score=y_score[sampled],
            label_names=labels,
            threshold=threshold,
        )
        auroc = perf.get("macro_auroc")
        acc = perf.get("macro_accuracy")
        if isinstance(auroc, (int, float, np.integer, np.floating)) and np.isfinite(float(auroc)):
            auroc_values.append(float(auroc))
        if isinstance(acc, (int, float, np.integer, np.floating)) and np.isfinite(float(acc)):
            acc_values.append(float(acc))
        y_pred = (y_score[sampled] >= threshold).astype(int)
        tpr, fpr = _macro_tpr_fpr(y_true[sampled].astype(int), y_pred.astype(int))
        if np.isfinite(tpr):
            tpr_values.append(float(tpr))
        if np.isfinite(fpr):
            fpr_values.append(float(fpr))
    out: dict[str, dict[str, float]] = {}
    if auroc_values:
        arr = np.asarray(auroc_values, dtype=float)
        out["macro_auroc"] = {
            "ci_low": float(np.quantile(arr, 0.025)),
            "ci_high": float(np.quantile(arr, 0.975)),
        }
    if acc_values:
        arr = np.asarray(acc_values, dtype=float)
        out["macro_accuracy"] = {
            "ci_low": float(np.quantile(arr, 0.025)),
            "ci_high": float(np.quantile(arr, 0.975)),
        }
    if tpr_values:
        arr = np.asarray(tpr_values, dtype=float)
        out["macro_tpr"] = {
            "ci_low": float(np.quantile(arr, 0.025)),
            "ci_high": float(np.quantile(arr, 0.975)),
        }
    if fpr_values:
        arr = np.asarray(fpr_values, dtype=float)
        out["macro_fpr"] = {
            "ci_low": float(np.quantile(arr, 0.025)),
            "ci_high": float(np.quantile(arr, 0.975)),
        }
    return out


def _method_label_to_report_key(method_label: str) -> str:
    mapping = {
        "Baseline": "baseline_feature_probe",
        "SAE Concepts": "sae_concept_probe",
        "SAE Concepts (Debiased)": "sae_concept_probe_debiased",
    }
    return mapping.get(method_label, method_label)


def _macro_tpr_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    tpr_values: list[float] = []
    fpr_values: list[float] = []
    for idx in range(y_true.shape[1]):
        yt = y_true[:, idx].astype(int)
        yp = y_pred[:, idx].astype(int)
        tp = float(np.sum((yt == 1) & (yp == 1)))
        fn = float(np.sum((yt == 1) & (yp == 0)))
        fp = float(np.sum((yt == 0) & (yp == 1)))
        tn = float(np.sum((yt == 0) & (yp == 0)))
        if tp + fn > 0:
            tpr_values.append(tp / (tp + fn))
        if fp + tn > 0:
            fpr_values.append(fp / (fp + tn))
    return (
        float(np.mean(tpr_values)) if tpr_values else float("nan"),
        float(np.mean(fpr_values)) if fpr_values else float("nan"),
    )
