from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, r2_score
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class ConceptMetric:
    concept: str
    task_type: str
    performance: float
    sparsity_ratio: float
    top_weight_share: float


def evaluate_disentanglement(
    z_train: np.ndarray,
    z_test: np.ndarray,
    y_path_train: np.ndarray,
    y_path_test: np.ndarray,
    pathology_cols: list[str],
    metadata_train: pd.DataFrame,
    metadata_test: pd.DataFrame,
    metadata_cols: list[str],
) -> dict[str, object]:
    metrics: list[ConceptMetric] = []

    for idx, concept in enumerate(pathology_cols):
        y_tr = y_path_train[:, idx].astype(int)
        y_te = y_path_test[:, idx].astype(int)
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        result = _binary_probe(z_train, z_test, y_tr, y_te, concept)
        metrics.append(result)

    for concept in metadata_cols:
        y_tr = metadata_train[concept]
        y_te = metadata_test[concept]

        if _is_numeric_series(y_tr):
            result = _regression_probe(z_train, z_test, y_tr.to_numpy(), y_te.to_numpy(), concept)
            metrics.append(result)
        else:
            result = _categorical_probe(z_train, z_test, y_tr.astype(str).to_numpy(), y_te.astype(str).to_numpy(), concept)
            if result is not None:
                metrics.append(result)

    if not metrics:
        return {
            "num_concepts_evaluated": 0,
            "mean_disentanglement": float("nan"),
            "mean_performance": float("nan"),
            "mean_top_weight_share": float("nan"),
            "concept_metrics": [],
        }

    mean_disentanglement = float(np.mean([1.0 - m.sparsity_ratio for m in metrics]))
    mean_performance = float(np.mean([m.performance for m in metrics]))
    mean_top_weight_share = float(np.mean([m.top_weight_share for m in metrics]))

    return {
        "num_concepts_evaluated": len(metrics),
        "mean_disentanglement": mean_disentanglement,
        "mean_performance": mean_performance,
        "mean_top_weight_share": mean_top_weight_share,
        "concept_metrics": [m.__dict__ for m in metrics],
    }


def reconstruction_metrics(x_true: np.ndarray, x_recon: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((x_true - x_recon) ** 2))
    var = float(np.var(x_true))
    explained_variance = 1.0 - (mse / var if var > 0 else 0.0)
    return {
        "mse": mse,
        "explained_variance": explained_variance,
    }


def _binary_probe(
    z_train: np.ndarray,
    z_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    concept: str,
) -> ConceptMetric:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(z_train)
    x_test = scaler.transform(z_test)

    clf = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=0.2,
        max_iter=2000,
        random_state=13,
    )
    clf.fit(x_train, y_train)
    prob = clf.predict_proba(x_test)[:, 1]
    score = float(roc_auc_score(y_test, prob))

    weights = np.abs(clf.coef_.ravel())
    return ConceptMetric(
        concept=concept,
        task_type="binary",
        performance=score,
        sparsity_ratio=_sparsity_ratio(weights),
        top_weight_share=_top_weight_share(weights),
    )


def _categorical_probe(
    z_train: np.ndarray,
    z_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    concept: str,
) -> ConceptMetric | None:
    uniq_train = np.unique(y_train)
    uniq_test = np.unique(y_test)
    if len(uniq_train) < 2 or len(uniq_test) < 2:
        return None

    scaler = StandardScaler()
    x_train = scaler.fit_transform(z_train)
    x_test = scaler.transform(z_test)

    clf = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=0.2,
        max_iter=3000,
        random_state=13,
        multi_class="auto",
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    score = float(f1_score(y_test, pred, average="macro"))

    weights = np.abs(clf.coef_).sum(axis=0)
    return ConceptMetric(
        concept=concept,
        task_type="categorical",
        performance=score,
        sparsity_ratio=_sparsity_ratio(weights),
        top_weight_share=_top_weight_share(weights),
    )


def _regression_probe(
    z_train: np.ndarray,
    z_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    concept: str,
) -> ConceptMetric:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(z_train)
    x_test = scaler.transform(z_test)

    reg = Lasso(alpha=0.001, max_iter=4000, random_state=13)
    reg.fit(x_train, y_train)
    pred = reg.predict(x_test)
    score = float(r2_score(y_test, pred))

    weights = np.abs(reg.coef_)
    return ConceptMetric(
        concept=concept,
        task_type="continuous",
        performance=score,
        sparsity_ratio=_sparsity_ratio(weights),
        top_weight_share=_top_weight_share(weights),
    )


def _sparsity_ratio(weights: np.ndarray, eps: float = 1e-6) -> float:
    active = float(np.count_nonzero(weights > eps))
    return active / float(weights.size)


def _top_weight_share(weights: np.ndarray, eps: float = 1e-8) -> float:
    total = float(np.sum(weights))
    if total < eps:
        return 0.0
    return float(np.max(weights) / total)


def _is_numeric_series(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna().mean() > 0.95
