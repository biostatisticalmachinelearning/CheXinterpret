from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate_multilabel_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: list[str],
) -> dict[str, object]:
    per_label: dict[str, float] = {}

    for idx, label in enumerate(label_names):
        y = y_true[:, idx]
        s = y_score[:, idx]
        if len(np.unique(y)) < 2:
            continue
        per_label[label] = float(roc_auc_score(y, s))

    macro_auroc = float(np.mean(list(per_label.values()))) if per_label else float("nan")
    return {
        "macro_auroc": macro_auroc,
        "label_auroc": per_label,
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
        group_eval = evaluate_multilabel_predictions(y_true[mask], y_score[mask], label_names)
        odds = _equalized_odds_components(y_true[mask], y_score[mask], threshold)
        group_metrics[str(group)] = {
            **group_eval,
            **odds,
            "n": int(mask.sum()),
        }

    macro_scores = [group_metrics[g]["macro_auroc"] for g in group_metrics if not math.isnan(group_metrics[g]["macro_auroc"])]
    auroc_gap = float(np.max(macro_scores) - np.min(macro_scores)) if macro_scores else float("nan")

    tpr_values = [group_metrics[g]["macro_tpr"] for g in group_metrics if not math.isnan(group_metrics[g]["macro_tpr"])]
    fpr_values = [group_metrics[g]["macro_fpr"] for g in group_metrics if not math.isnan(group_metrics[g]["macro_fpr"])]

    eo_tpr_gap = float(np.max(tpr_values) - np.min(tpr_values)) if tpr_values else float("nan")
    eo_fpr_gap = float(np.max(fpr_values) - np.min(fpr_values)) if fpr_values else float("nan")

    bootstrap = _bootstrap_auroc_gap(y_true, y_score, groups, label_names, bootstrap_samples)

    return {
        "groups": group_metrics,
        "macro_auroc_gap": auroc_gap,
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
            eval_dict = evaluate_multilabel_predictions(ys[mask], ps[mask], label_names)
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
