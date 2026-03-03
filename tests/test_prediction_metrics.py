import numpy as np

from chex_sae_fairness.evaluation.fairness import evaluate_multilabel_predictions


def test_evaluate_multilabel_predictions_reports_accuracy_metrics() -> None:
    y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.int64)
    y_score = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.6], [0.1, 0.2]], dtype=np.float32)

    metrics = evaluate_multilabel_predictions(y_true, y_score, label_names=["a", "b"], threshold=0.5)

    assert metrics["macro_accuracy"] == 1.0
    assert metrics["micro_accuracy"] == 1.0
    assert "label_accuracy" in metrics
