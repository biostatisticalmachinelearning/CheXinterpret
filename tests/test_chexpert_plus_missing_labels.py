import numpy as np
import pandas as pd

from chex_sae_fairness.config import (
    DataConfig,
    ExperimentConfig,
    FairnessConfig,
    FeatureConfig,
    PathsConfig,
    ProbeConfig,
    SAEConfig,
    SchemaConfig,
)
from chex_sae_fairness.data.chexpert_plus import _handle_missing_pathology_labels, build_manifest


def _cfg(uncertain_policy: str) -> ExperimentConfig:
    return ExperimentConfig(
        seed=0,
        paths=PathsConfig(image_root=".", metadata_csv=".", output_root="."),
        schema=SchemaConfig(
            image_path_col="path_to_image",
            split_col="split",
            patient_id_col="patient_id",
            age_col="age",
            sex_col="sex",
            race_col="race",
            pathology_cols=["No Finding", "Cardiomegaly"],
            metadata_cols=[],
        ),
        data=DataConfig(uncertain_label_policy=uncertain_policy),
        features=FeatureConfig(model_name="dummy"),
        sae=SAEConfig(latent_dim=8),
        probes=ProbeConfig(),
        fairness=FairnessConfig(),
    )


def test_missing_pathology_labels_filled_for_non_ignore_policy() -> None:
    frame = pd.DataFrame(
        {
            "No Finding": [1.0, np.nan],
            "Cardiomegaly": [np.nan, 0.0],
        }
    )

    out = _handle_missing_pathology_labels(frame, _cfg("zero"))
    assert float(out.loc[1, "No Finding"]) == 0.0
    assert float(out.loc[0, "Cardiomegaly"]) == 0.0


def test_missing_pathology_labels_left_when_ignore_policy() -> None:
    frame = pd.DataFrame(
        {
            "No Finding": [1.0, np.nan],
            "Cardiomegaly": [np.nan, 0.0],
        }
    )

    out = _handle_missing_pathology_labels(frame, _cfg("ignore"))
    assert np.isnan(float(out.loc[1, "No Finding"]))
    assert np.isnan(float(out.loc[0, "Cardiomegaly"]))


def test_build_manifest_keeps_rows_with_missing_pathology_under_default_policy(tmp_path) -> None:
    image_path = tmp_path / "train" / "patient00001" / "study1" / "view1_frontal.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"x")

    csv_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {
                "path_to_image": "train/patient00001/study1/view1_frontal.png",
                "split": "train",
                "patient_id": "patient00001",
                "age": 55,
                "sex": "F",
                "race": "White",
                "No Finding": np.nan,
                "Cardiomegaly": np.nan,
            }
        ]
    ).to_csv(csv_path, index=False)

    cfg = ExperimentConfig(
        seed=0,
        paths=PathsConfig(
            image_root=str(tmp_path),
            metadata_csv=str(csv_path),
            output_root=str(tmp_path / "outputs"),
        ),
        schema=SchemaConfig(
            image_path_col="path_to_image",
            split_col="split",
            patient_id_col="patient_id",
            age_col="age",
            sex_col="sex",
            race_col="race",
            pathology_cols=["No Finding", "Cardiomegaly"],
            metadata_cols=[],
        ),
        data=DataConfig(uncertain_label_policy="zero", min_age=18, max_age=110, age_bins=[18, 40, 60, 120]),
        features=FeatureConfig(model_name="dummy"),
        sae=SAEConfig(latent_dim=8),
        probes=ProbeConfig(),
        fairness=FairnessConfig(),
    )

    result = build_manifest(cfg)
    assert len(result.manifest) == 1
    assert float(result.manifest.iloc[0]["No Finding"]) == 0.0
    assert float(result.manifest.iloc[0]["Cardiomegaly"]) == 0.0
