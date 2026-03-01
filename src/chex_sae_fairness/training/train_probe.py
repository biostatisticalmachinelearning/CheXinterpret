from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class MultilabelProbe:
    scaler: StandardScaler
    classifier: OneVsRestClassifier

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        transformed = self.scaler.transform(x)
        return self.classifier.predict_proba(transformed)


def fit_multilabel_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = 2000,
    c_value: float = 1.0,
) -> MultilabelProbe:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)

    base = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=c_value,
        max_iter=max_iter,
        class_weight="balanced",
        random_state=13,
    )
    classifier = OneVsRestClassifier(base)
    classifier.fit(x_scaled, y_train)
    return MultilabelProbe(scaler=scaler, classifier=classifier)
