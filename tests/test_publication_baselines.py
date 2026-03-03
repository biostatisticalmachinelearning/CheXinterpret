import numpy as np

from chex_sae_fairness.publication.baselines import BaselineSuiteInputs, run_baseline_suite


def test_run_baseline_suite_raw_only() -> None:
    rng = np.random.default_rng(13)
    x_train = rng.normal(size=(40, 8)).astype(np.float32)
    x_test = rng.normal(size=(20, 8)).astype(np.float32)
    y_train = (rng.uniform(size=(40, 3)) > 0.5).astype(np.int64)
    y_test = (rng.uniform(size=(20, 3)) > 0.5).astype(np.int64)
    groups_train = np.array(["young"] * 20 + ["old"] * 20)
    groups_test = np.array(["young"] * 10 + ["old"] * 10)

    inputs = BaselineSuiteInputs(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        age_groups_train=groups_train,
        age_groups_test=groups_test,
        pathology_cols=["a", "b", "c"],
        threshold=0.5,
        bootstrap_samples=0,
        probe_c_value=1.0,
        probe_max_iter=200,
        latent_dim=4,
    )

    result = run_baseline_suite(inputs, methods=["raw"])
    assert "raw" in result
    assert "performance" in result["raw"]
