import numpy as np
import pandas as pd

from chex_sae_fairness.publication.validity import (
    build_concept_permutation_table,
    build_concept_precision_recall_table,
    build_patient_split_leakage_table,
)


def test_build_concept_precision_recall_table_outputs_rows() -> None:
    rng = np.random.default_rng(13)
    z_train = rng.normal(size=(50, 6)).astype(np.float32)
    z_test = rng.normal(size=(20, 6)).astype(np.float32)
    y_train = (rng.uniform(size=(50, 2)) > 0.5).astype(np.int64)
    y_test = (rng.uniform(size=(20, 2)) > 0.5).astype(np.int64)
    meta_train = pd.DataFrame({"age": rng.integers(20, 80, size=50), "sex": rng.choice(["M", "F"], size=50)})
    meta_test = pd.DataFrame({"age": rng.integers(20, 80, size=20), "sex": rng.choice(["M", "F"], size=20)})

    frame = build_concept_precision_recall_table(
        z_train=z_train,
        z_test=z_test,
        y_path_train=y_train,
        y_path_test=y_test,
        pathology_cols=["a", "b"],
        metadata_train=meta_train,
        metadata_test=meta_test,
        metadata_cols=["age", "sex"],
    )
    assert not frame.empty
    assert {"concept", "precision", "recall", "f1"}.issubset(frame.columns)


def test_build_concept_permutation_table_has_adjusted_p_values() -> None:
    rng = np.random.default_rng(7)
    z = rng.normal(size=(40, 4)).astype(np.float32)
    y = (rng.uniform(size=(40, 2)) > 0.5).astype(np.int64)
    meta = pd.DataFrame({"age": rng.integers(20, 90, size=40), "sex": rng.choice(["M", "F"], size=40)})

    frame = build_concept_permutation_table(
        z=z,
        y_pathology=y,
        pathology_cols=["a", "b"],
        metadata=meta,
        metadata_cols=["age", "sex"],
        repeats=5,
        seed=11,
    )
    assert "p_adj_bh" in frame.columns
    assert "p_adj_holm" in frame.columns


def test_build_patient_split_leakage_table_detects_overlap() -> None:
    split = np.array(["train", "train", "test", "test"])
    patient = np.array(["p1", "p2", "p2", "p3"])
    frame = build_patient_split_leakage_table(split=split, patient_id=patient)
    overlap_rows = frame.loc[frame["check"] == "patient_overlap"]
    assert not overlap_rows.empty
    assert overlap_rows["n_overlap"].max() >= 1
