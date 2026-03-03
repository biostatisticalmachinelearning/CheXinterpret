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
from chex_sae_fairness.data.chexpert_plus import _apply_common_column_aliases


def _make_cfg(metadata_cols: list[str], age_col: str = "age", sex_col: str = "sex", race_col: str = "race") -> ExperimentConfig:
    return ExperimentConfig(
        seed=0,
        paths=PathsConfig(image_root=".", metadata_csv=".", output_root="."),
        schema=SchemaConfig(
            image_path_col="path_to_image",
            split_col="split",
            patient_id_col="patient_id",
            age_col=age_col,
            sex_col=sex_col,
            race_col=race_col,
            pathology_cols=[],
            metadata_cols=metadata_cols,
        ),
        data=DataConfig(),
        features=FeatureConfig(model_name="dummy"),
        sae=SAEConfig(latent_dim=8),
        probes=ProbeConfig(),
        fairness=FairnessConfig(),
    )


def test_aliases_old_patient_prefixed_columns_to_current_defaults() -> None:
    cfg = _make_cfg(
        metadata_cols=[
            "age",
            "sex",
            "race",
            "ethnicity",
            "interpreter_needed",
            "insurance_type",
            "recent_bmi",
            "deceased",
        ]
    )
    frame = pd.DataFrame(
        {
            "path_to_image": ["train/patient00001/study1/view1_frontal.jpg"],
            "split": ["train"],
            "patient_age": [67],
            "patient_sex": ["M"],
            "patient_race": ["White"],
            "patient_ethnicity": ["Non-Hispanic"],
            "patient_primary_language": [0],
            "patient_insurance_type": ["Medicare"],
            "patient_recent_bmi": [28.1],
            "patient_deceased": [0],
        }
    )

    aliased = _apply_common_column_aliases(frame, cfg)

    for col in cfg.schema.metadata_cols:
        assert col in aliased.columns


def test_aliases_current_columns_to_legacy_patient_prefixed_targets() -> None:
    cfg = _make_cfg(
        metadata_cols=["patient_ethnicity", "patient_insurance_type", "patient_primary_language"],
        age_col="patient_age",
        sex_col="patient_sex",
        race_col="patient_race",
    )
    frame = pd.DataFrame(
        {
            "path_to_image": ["train/patient00001/study1/view1_frontal.jpg"],
            "split": ["train"],
            "age": [67],
            "sex": ["M"],
            "race": ["White"],
            "ethnicity": ["Non-Hispanic"],
            "insurance_type": ["Medicare"],
            "interpreter_needed": [0],
        }
    )

    aliased = _apply_common_column_aliases(frame, cfg)

    assert "patient_age" in aliased.columns
    assert "patient_sex" in aliased.columns
    assert "patient_race" in aliased.columns
    assert "patient_ethnicity" in aliased.columns
    assert "patient_insurance_type" in aliased.columns
    assert "patient_primary_language" in aliased.columns
