from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate_multilabel_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: list[str],
    threshold: float = 0.5,
) -> dict[str, object]:
    # We report both ranking quality (AUROC) and thresholded correctness metrics
    # because fairness interventions can change calibration at a fixed threshold.
    per_label_auroc: dict[str, float] = {}
    per_label_accuracy: dict[str, float] = {}
    per_label_brier: dict[str, float] = {}
    per_label_ece: dict[str, float] = {}

    for idx, label in enumerate(label_names):
        y = y_true[:, idx]
        s = y_score[:, idx]
        if len(np.unique(y)) < 2:
            per_label_auroc[label] = float("nan")
        else:
            per_label_auroc[label] = float(roc_auc_score(y, s))
        per_label_accuracy[label] = float(np.mean((s >= threshold).astype(int) == y.astype(int)))
        per_label_brier[label] = float(np.mean((s - y.astype(float)) ** 2))
        per_label_ece[label] = float(_expected_calibration_error(y.astype(int), s.astype(float)))

    valid_aurocs = [score for score in per_label_auroc.values() if not math.isnan(score)]
    macro_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else float("nan")
    macro_accuracy = float(np.mean(list(per_label_accuracy.values()))) if per_label_accuracy else float("nan")
    micro_accuracy = float(
        np.mean((y_score >= threshold).astype(int) == y_true.astype(int))
    )
    macro_brier = float(np.mean(list(per_label_brier.values()))) if per_label_brier else float("nan")
    macro_ece = float(np.mean(list(per_label_ece.values()))) if per_label_ece else float("nan")
    return {
        "macro_auroc": macro_auroc,
        "macro_accuracy": macro_accuracy,
        "micro_accuracy": micro_accuracy,
        "macro_brier": macro_brier,
        "macro_ece": macro_ece,
        "label_auroc": per_label_auroc,
        "label_accuracy": per_label_accuracy,
        "label_brier": per_label_brier,
        "label_ece": per_label_ece,
    }


def evaluate_group_fairness(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    label_names: list[str],
    threshold: float,
    bootstrap_samples: int,
) -> dict[str, object]:
    unique_groups = sorted(np.unique(groups).tolist())
    group_metrics: dict[str, dict[str, object]] = {}

    for group in unique_groups:
        mask = groups == group
        group_eval = evaluate_multilabel_predictions(
            y_true[mask],
            y_score[mask],
            label_names,
            threshold=threshold,
        )
        odds = _equalized_odds_components(y_true[mask], y_score[mask], threshold)
        group_metrics[str(group)] = {
            **group_eval,
            **odds,
            "n": int(mask.sum()),
        }

    macro_scores = [
        group_metrics[g]["macro_auroc"]
        for g in group_metrics
        if not math.isnan(group_metrics[g]["macro_auroc"])
    ]
    macro_accuracy_scores = [
        group_metrics[g]["macro_accuracy"]
        for g in group_metrics
        if not math.isnan(group_metrics[g]["macro_accuracy"])
    ]

    auroc_gap = float(np.max(macro_scores) - np.min(macro_scores)) if macro_scores else float("nan")
    accuracy_gap = (
        float(np.max(macro_accuracy_scores) - np.min(macro_accuracy_scores))
        if macro_accuracy_scores
        else float("nan")
    )

    tpr_values = [group_metrics[g]["macro_tpr"] for g in group_metrics if not math.isnan(group_metrics[g]["macro_tpr"])]
    fpr_values = [group_metrics[g]["macro_fpr"] for g in group_metrics if not math.isnan(group_metrics[g]["macro_fpr"])]

    eo_tpr_gap = float(np.max(tpr_values) - np.min(tpr_values)) if tpr_values else float("nan")
    eo_fpr_gap = float(np.max(fpr_values) - np.min(fpr_values)) if fpr_values else float("nan")

    bootstrap = _bootstrap_auroc_gap(y_true, y_score, groups, label_names, bootstrap_samples)
    overall = evaluate_multilabel_predictions(y_true, y_score, label_names, threshold=threshold)
    worst_group_auroc = _worst_group_metric(group_metrics, "macro_auroc")
    worst_group_accuracy = _worst_group_metric(group_metrics, "macro_accuracy")

    return {
        "groups": group_metrics,
        "overall": overall,
        "macro_auroc_gap": auroc_gap,
        "macro_accuracy_gap": accuracy_gap,
        "worst_group_macro_auroc": worst_group_auroc,
        "worst_group_macro_accuracy": worst_group_accuracy,
        "equalized_odds_tpr_gap": eo_tpr_gap,
        "equalized_odds_fpr_gap": eo_fpr_gap,
        "bootstrap_macro_auroc_gap": bootstrap,
    }


def _equalized_odds_components(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)

    tprs: list[float] = []
    fprs: list[float] = []

    for idx in range(y_true.shape[1]):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]

        tp = float(np.sum((yt == 1) & (yp == 1)))
        fn = float(np.sum((yt == 1) & (yp == 0)))
        fp = float(np.sum((yt == 0) & (yp == 1)))
        tn = float(np.sum((yt == 0) & (yp == 0)))

        if (tp + fn) > 0:
            tprs.append(tp / (tp + fn))
        if (fp + tn) > 0:
            fprs.append(fp / (fp + tn))

    macro_tpr = float(np.mean(tprs)) if tprs else float("nan")
    macro_fpr = float(np.mean(fprs)) if fprs else float("nan")
    return {
        "macro_tpr": macro_tpr,
        "macro_fpr": macro_fpr,
    }


def _bootstrap_auroc_gap(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    label_names: list[str],
    samples: int,
) -> dict[str, float]:
    if samples <= 0:
        return {}

    rng = np.random.default_rng(13)
    by_group = defaultdict(list)
    for idx, g in enumerate(groups):
        by_group[str(g)].append(idx)

    gaps: list[float] = []
    for _ in range(samples):
        sampled_indices: list[int] = []
        for idxs in by_group.values():
            sampled_indices.extend(rng.choice(idxs, size=len(idxs), replace=True).tolist())

        sampled_indices = sorted(sampled_indices)
        ys = y_true[sampled_indices]
        ps = y_score[sampled_indices]
        gs = groups[sampled_indices]

        group_scores: list[float] = []
        for g in sorted(np.unique(gs).tolist()):
            mask = gs == g
            eval_dict = evaluate_multilabel_predictions(
                ys[mask],
                ps[mask],
                label_names,
            )
            if not math.isnan(eval_dict["macro_auroc"]):
                group_scores.append(eval_dict["macro_auroc"])

        if group_scores:
            gaps.append(float(np.max(group_scores) - np.min(group_scores)))

    if not gaps:
        return {}

    return {
        "mean": float(np.mean(gaps)),
        "p05": float(np.quantile(gaps, 0.05)),
        "p95": float(np.quantile(gaps, 0.95)),
    }


def _worst_group_metric(
    group_metrics: dict[str, dict[str, object]],
    metric: str,
) -> dict[str, object] | None:
    candidates: list[tuple[str, float]] = []
    for group, values in group_metrics.items():
        value = values.get(metric)
        if isinstance(value, (float, np.floating)) and not math.isnan(float(value)):
            candidates.append((group, float(value)))
    if not candidates:
        return None
    group, score = min(candidates, key=lambda x: x[1])
    return {"group": group, "value": score}


def _expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(len(y_true))
    ece = 0.0
    for idx in range(n_bins):
        lo, hi = bins[idx], bins[idx + 1]
        if idx == n_bins - 1:
            mask = (y_score >= lo) & (y_score <= hi)
        else:
            mask = (y_score >= lo) & (y_score < hi)
        if not np.any(mask):
            continue
        frac = float(mask.mean())
        accuracy = float(np.mean(y_true[mask]))
        confidence = float(np.mean(y_score[mask]))
        ece += frac * abs(accuracy - confidence)
    return ece
