from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers

from chex_sae_fairness.data.chexpert_plus import CheXImageDataset

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FeatureExtractionConfig:
    model_name: str
    cache_dir: str | None
    device: str
    batch_size: int
    num_workers: int
    precision: str = "fp16"
    pooling: str = "mean"


class CheXagentVisionFeatureExtractor:
    def __init__(self, cfg: FeatureExtractionConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model_dtype = _resolve_model_dtype(cfg.precision, self.device)
        cache_dir = _resolve_cache_dir(cfg.cache_dir)
        self.cache_dir = cache_dir

        logger.info("Loading CheXagent processor for %s (cache_dir=%s)", cfg.model_name, cache_dir)
        self.processor = transformers.AutoProcessor.from_pretrained(
            cfg.model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
            device_map="auto",
        )
        logger.info("Loading CheXagent model weights for %s", cfg.model_name)
        self.model = _load_chexagent_model_strict(
            model_name=cfg.model_name,
            torch_dtype=self.model_dtype,
            cache_dir=cache_dir,
        )
        self.model.eval().to(self.device)

        if self.model_dtype == torch.float16 and self.device.type == "cuda":
            self.model.half()
        logger.info(
            "CheXagent model ready on %s (precision=%s, pooling=%s, loader=%s, dtype=%s)",
            self.device,
            cfg.precision,
            cfg.pooling,
            "AutoModelForCausalLM",
            self.model_dtype,
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
        # Mirror the official HF usage: multimodal prompt + image batch.
        # We still extract vision-side embeddings rather than generated text.
        model_inputs = _to_device_tensors(
            self.processor(
                images=images,
                text=_prompt_batch(len(images)),
                return_tensors="pt",
                padding=True,
            ),
            self.device,
            dtype=self.model_dtype,
        )
        image_only_inputs = _to_device_tensors(
            self.processor(images=images, return_tensors="pt"),
            self.device,
            dtype=self.model_dtype,
        )

        if hasattr(self.model, "get_image_features"):
            try:
                image_features = self.model.get_image_features(**image_only_inputs)
                return _pool_features(image_features, self.cfg.pooling)
            except Exception as exc:  # pragma: no cover - backend/model dependent
                logger.debug("get_image_features failed; falling back to model forward: %s", exc)

        outputs = _forward_with_fallback_inputs(
            model=self.model,
            processor=self.processor,
            images=images,
            device=self.device,
            primary_inputs=model_inputs,
            dtype=self.model_dtype,
        )
        return _extract_features_from_outputs(outputs, self.cfg.pooling)


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


def _load_chexagent_model_strict(
    model_name: str,
    torch_dtype: torch.dtype | None,
    cache_dir: str | None,
) -> torch.nn.Module:
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except Exception as exc:  # pragma: no cover - depends on user env/model
        raise RuntimeError(
            "Failed to load CheXagent with AutoModelForCausalLM only. "
            "No fallback model loader is used to avoid silent mis-loading. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    config_name = model.config.__class__.__name__
    model_class_name = model.__class__.__name__
    if "chexagent" not in config_name.lower() and "chexagent" not in model_class_name.lower():
        raise RuntimeError(
            "Loaded model does not appear to be a CheXagent class. "
            f"config_class={config_name}, model_class={model_class_name}. "
            "Aborting to prevent extracting invalid embeddings."
        )
    return model


def _to_device_tensors(
    model_inputs: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in model_inputs.items():
        if not isinstance(value, torch.Tensor):
            continue
        if dtype is not None and torch.is_floating_point(value):
            out[key] = value.to(device=device, dtype=dtype)
        else:
            out[key] = value.to(device=device)
    return out


def _forward_with_fallback_inputs(
    model: torch.nn.Module,
    processor: Any,
    images: list[Any],
    device: torch.device,
    primary_inputs: dict[str, Any],
    dtype: torch.dtype | None,
) -> Any:
    try:
        return model(**primary_inputs, output_hidden_states=True, return_dict=True)
    except Exception as primary_exc:  # pragma: no cover - backend/model dependent
        # Some multimodal causal LMs require text tokens even when extracting image features.
        try:
            text_inputs = _to_device_tensors(
                processor(
                    images=images,
                    text=[""] * len(images),
                    return_tensors="pt",
                    padding=True,
                ),
                device,
                dtype=dtype,
            )
            return model(**text_inputs, output_hidden_states=True, return_dict=True)
        except Exception as secondary_exc:
            raise RuntimeError(
                "Model forward failed for both image-only and image+empty-text inputs. "
                f"image-only error: {type(primary_exc).__name__}: {primary_exc} | "
                f"image+text error: {type(secondary_exc).__name__}: {secondary_exc}"
            ) from secondary_exc


def _extract_features_from_outputs(outputs: Any, pooling: str) -> torch.Tensor:
    if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        return _pool_features(outputs.image_embeds, pooling)
    if hasattr(outputs, "vision_embeds") and outputs.vision_embeds is not None:
        return _pool_features(outputs.vision_embeds, pooling)
    if hasattr(outputs, "image_hidden_states") and outputs.image_hidden_states:
        return _pool_features(outputs.image_hidden_states[-1], pooling)
    if hasattr(outputs, "vision_hidden_states") and outputs.vision_hidden_states:
        return _pool_features(outputs.vision_hidden_states[-1], pooling)
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return _pool_features(outputs.last_hidden_state, pooling)
    if hasattr(outputs, "hidden_states") and outputs.hidden_states:
        return _pool_features(outputs.hidden_states[-1], pooling)
    raise RuntimeError(
        "Unable to extract vision features from model outputs. "
        "Inspect checkpoint API and adapt feature extraction."
    )


def save_feature_bundle(
    output_path: str,
    features: np.ndarray,
    manifest: pd.DataFrame,
    split_col: str,
    pathology_cols: list[str],
    metadata_cols: list[str],
    age_col: str,
    patient_id_col: str,
) -> None:
    split = manifest[split_col].astype(str).to_numpy()
    age = manifest[age_col].astype(float).to_numpy()
    age_group = manifest["age_group"].astype(str).to_numpy()
    y_pathology = manifest[pathology_cols].astype(np.float32).to_numpy()
    image_path = manifest["image_path"].astype(str).to_numpy() if "image_path" in manifest.columns else np.array([], dtype=object)
    patient_id = (
        manifest[patient_id_col].astype(str).to_numpy()
        if patient_id_col in manifest.columns
        else np.array([], dtype=object)
    )
    view_type = np.array([_infer_view_type(path) for path in image_path], dtype=object) if image_path.size else np.array([], dtype=object)

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
        image_path=image_path.astype(object),
        patient_id=patient_id.astype(object),
        view_type=view_type.astype(object),
    )


def load_feature_bundle(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}


def _infer_view_type(path: str) -> str:
    value = str(path).lower()
    if "lateral" in value:
        return "lateral"
    if "frontal" in value:
        return "frontal"
    return "unknown"


def _resolve_model_dtype(precision: str, device: torch.device) -> torch.dtype | None:
    value = str(precision).strip().lower()
    if value == "fp16" and device.type == "cuda":
        return torch.float16
    if value in {"bf16", "bfloat16"} and device.type == "cuda":
        return torch.bfloat16
    return None


def _prompt_batch(batch_size: int) -> list[str]:
    prompt = ' USER: <s>Describe this chest x-ray image. ASSISTANT: <s>'
    return [prompt] * batch_size


def _resolve_cache_dir(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return str(Path(text).expanduser().resolve())
