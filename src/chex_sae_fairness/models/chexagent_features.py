from __future__ import annotations

from dataclasses import dataclass
import logging
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor

from chex_sae_fairness.data.chexpert_plus import CheXImageDataset

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FeatureExtractionConfig:
    model_name: str
    device: str
    batch_size: int
    num_workers: int
    precision: str = "fp16"
    pooling: str = "mean"


class CheXagentVisionFeatureExtractor:
    def __init__(self, cfg: FeatureExtractionConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        logger.info("Loading CheXagent processor for %s", cfg.model_name)
        self.processor = AutoProcessor.from_pretrained(cfg.model_name, trust_remote_code=True)
        logger.info("Loading CheXagent model weights for %s", cfg.model_name)
        self.model = AutoModel.from_pretrained(cfg.model_name, trust_remote_code=True)
        self.model.eval().to(self.device)

        if cfg.precision == "fp16" and self.device.type == "cuda":
            self.model.half()
        logger.info(
            "CheXagent model ready on %s (precision=%s, pooling=%s)",
            self.device,
            cfg.precision,
            cfg.pooling,
        )

    def extract_from_manifest(self, manifest: pd.DataFrame) -> np.ndarray:
        dataset = CheXImageDataset(manifest)
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            pin_memory=self.device.type == "cuda",
            collate_fn=_collate_images,
        )

        logger.info(
            "Starting feature extraction over %d images (%d batches).",
            len(dataset),
            len(loader),
        )
        all_features: list[np.ndarray] = []
        progress = tqdm(
            loader,
            desc="Extracting image features",
            unit="batch",
            disable=not sys.stderr.isatty(),
        )
        with torch.no_grad():
            for batch in progress:
                features = self._encode_images(batch["images"])
                all_features.append(features.detach().cpu().float().numpy())

        if not all_features:
            return np.empty((0, 0), dtype=np.float32)

        output = np.concatenate(all_features, axis=0).astype(np.float32)
        logger.info("Finished feature extraction with output shape=%s", tuple(output.shape))
        return output

    def _encode_images(self, images: list[Any]) -> torch.Tensor:
        model_inputs = self.processor(images=images, return_tensors="pt")
        model_inputs = {
            key: value.to(self.device)
            for key, value in model_inputs.items()
            if isinstance(value, torch.Tensor)
        }

        if hasattr(self.model, "get_image_features"):
            image_features = self.model.get_image_features(**model_inputs)
            return _pool_features(image_features, self.cfg.pooling)

        outputs = self.model(**model_inputs, output_hidden_states=True, return_dict=True)

        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            return _pool_features(outputs.image_embeds, self.cfg.pooling)

        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return _pool_features(outputs.last_hidden_state, self.cfg.pooling)

        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            return _pool_features(outputs.hidden_states[-1], self.cfg.pooling)

        raise RuntimeError(
            "Unable to extract vision features from model outputs. "
            "Inspect the checkpoint's vision API and adapt `_encode_images`."
        )


def _pool_features(tensor: torch.Tensor, mode: str) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim != 3:
        raise ValueError(f"Expected 2D or 3D tensor for pooling, got shape {tuple(tensor.shape)}")

    if mode == "cls":
        return tensor[:, 0, :]
    if mode == "mean":
        return tensor.mean(dim=1)
    raise ValueError(f"Unknown pooling mode: {mode}")


def _collate_images(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [sample["image"] for sample in samples],
        "indices": [sample["index"] for sample in samples],
    }


def save_feature_bundle(
    output_path: str,
    features: np.ndarray,
    manifest: pd.DataFrame,
    split_col: str,
    pathology_cols: list[str],
    metadata_cols: list[str],
    age_col: str,
) -> None:
    split = manifest[split_col].astype(str).to_numpy()
    age = manifest[age_col].astype(float).to_numpy()
    age_group = manifest["age_group"].astype(str).to_numpy()
    y_pathology = manifest[pathology_cols].astype(np.float32).to_numpy()

    # Metadata is kept in the NPZ bundle as object dtype so downstream tasks can encode as needed.
    metadata = manifest[metadata_cols].astype(str).to_numpy(dtype=object)

    np.savez_compressed(
        output_path,
        features=features.astype(np.float32),
        split=split,
        age=age,
        age_group=age_group,
        y_pathology=y_pathology,
        metadata=metadata,
        metadata_cols=np.array(metadata_cols, dtype=object),
        pathology_cols=np.array(pathology_cols, dtype=object),
    )


def load_feature_bundle(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}
