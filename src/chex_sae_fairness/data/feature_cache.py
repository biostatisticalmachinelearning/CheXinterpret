from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

from chex_sae_fairness.config import ExperimentConfig
from chex_sae_fairness.data.chexpert_plus import build_manifest, load_manifest, save_manifest
from chex_sae_fairness.models.chexagent_features import (
    CheXagentVisionFeatureExtractor,
    FeatureExtractionConfig,
    load_feature_bundle,
    save_feature_bundle,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FeatureBundleResult:
    bundle: dict[str, np.ndarray]
    used_cache: bool
    manifest_rows_dropped: int


def load_or_create_feature_bundle(
    cfg: ExperimentConfig,
    force_recompute: bool = False,
) -> FeatureBundleResult:
    cfg.ensure_output_dirs()

    if cfg.feature_path.exists() and not force_recompute:
        logger.info("Using cached features at %s", cfg.feature_path)
        bundle = load_feature_bundle(str(cfg.feature_path))
        return FeatureBundleResult(bundle=bundle, used_cache=True, manifest_rows_dropped=0)

    logger.info("Feature cache miss (or force enabled). Preparing manifest/features.")
    if cfg.manifest_path.exists():
        logger.info("Loading existing manifest from %s", cfg.manifest_path)
        manifest = load_manifest(cfg.manifest_path)
        dropped_rows = 0
    else:
        logger.info("Building manifest from metadata at %s", cfg.paths.metadata_csv)
        manifest_result = build_manifest(cfg)
        manifest = manifest_result.manifest
        dropped_rows = manifest_result.dropped_rows
        save_manifest(manifest, cfg.manifest_path)
        logger.info("Saved manifest to %s (rows=%d)", cfg.manifest_path, len(manifest))

    logger.info("Initializing CheXagent feature extractor: %s", cfg.features.model_name)
    extractor = CheXagentVisionFeatureExtractor(
        FeatureExtractionConfig(
            model_name=cfg.features.model_name,
            device=cfg.features.device,
            batch_size=cfg.features.batch_size,
            num_workers=cfg.features.num_workers,
            precision=cfg.features.precision,
            pooling=cfg.features.pooling,
        )
    )
    logger.info("Extracting image features from %d manifest rows", len(manifest))
    features = extractor.extract_from_manifest(manifest)
    logger.info("Feature extraction complete: shape=%s", tuple(features.shape))

    save_feature_bundle(
        output_path=str(cfg.feature_path),
        features=features,
        manifest=manifest,
        split_col=cfg.schema.split_col,
        pathology_cols=cfg.schema.pathology_cols,
        metadata_cols=cfg.schema.metadata_cols,
        age_col=cfg.schema.age_col,
        patient_id_col=cfg.schema.patient_id_col,
    )
    logger.info("Saved feature bundle to %s", cfg.feature_path)

    bundle = load_feature_bundle(str(cfg.feature_path))
    return FeatureBundleResult(bundle=bundle, used_cache=False, manifest_rows_dropped=dropped_rows)
