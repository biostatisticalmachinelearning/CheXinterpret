from __future__ import annotations

from dataclasses import dataclass
import logging
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from chex_sae_fairness.config import SAEConfig
from chex_sae_fairness.models.sae import SparseAutoencoder, sae_loss

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SAETrainingOutput:
    model: SparseAutoencoder
    train_curve: list[dict[str, float]]
    valid_curve: list[dict[str, float]]


@torch.no_grad()
def encode_features(
    model: SparseAutoencoder,
    features: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    tensor = torch.tensor(features, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)

    outputs: list[np.ndarray] = []
    progress = tqdm(
        loader,
        desc="Encoding SAE latents",
        unit="batch",
        disable=not sys.stderr.isatty(),
    )
    for (batch,) in progress:
        z = model.encode(batch.to(device))
        outputs.append(z.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(outputs, axis=0)


def train_sae_model(
    train_features: np.ndarray,
    valid_features: np.ndarray,
    cfg: SAEConfig,
    device: str,
) -> SAETrainingOutput:
    train_tensor = torch.tensor(train_features, dtype=torch.float32)
    valid_tensor = torch.tensor(valid_features, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(valid_tensor), batch_size=cfg.batch_size, shuffle=False)

    model = SparseAutoencoder(
        input_dim=train_features.shape[1],
        latent_dim=cfg.latent_dim,
        variant=cfg.variant,
        topk_k=cfg.topk_k,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    train_curve: list[dict[str, float]] = []
    valid_curve: list[dict[str, float]] = []
    logger.info(
        "SAE training started: train_rows=%d, valid_rows=%d, input_dim=%d, latent_dim=%d, variant=%s",
        train_features.shape[0],
        valid_features.shape[0],
        train_features.shape[1],
        cfg.latent_dim,
        cfg.variant,
    )

    for epoch in range(cfg.epochs):
        model.train()
        train_running = {"total": 0.0, "mse": 0.0, "sparsity": 0.0, "regularization": 0.0, "steps": 0}

        batch_iterator = tqdm(
            train_loader,
            desc=f"SAE train epoch {epoch + 1}/{cfg.epochs}",
            unit="batch",
            leave=False,
            disable=not sys.stderr.isatty(),
        )
        for (batch,) in batch_iterator:
            batch = batch.to(device)
            x_hat, z = model(batch)
            loss = sae_loss(batch, x_hat, z, cfg.variant, cfg.l1_lambda)

            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            optimizer.step()

            train_running["total"] += float(loss.total.item())
            train_running["mse"] += float(loss.mse.item())
            train_running["sparsity"] += float(loss.sparsity.item())
            train_running["regularization"] += float(loss.regularization.item())
            train_running["steps"] += 1
            batch_iterator.set_postfix(
                loss=f"{loss.total.item():.4f}",
                mse=f"{loss.mse.item():.4f}",
            )

        train_epoch_metrics = _normalize_running(epoch, cfg.variant, train_running)
        train_curve.append(train_epoch_metrics)

        valid_epoch_metrics: dict[str, float] | None = None
        if (epoch + 1) % cfg.eval_every_epochs == 0 or epoch == cfg.epochs - 1:
            valid_epoch_metrics = _evaluate_epoch(
                model,
                valid_loader,
                cfg.variant,
                cfg.l1_lambda,
                epoch,
                device,
            )
            valid_curve.append(valid_epoch_metrics)

        if valid_epoch_metrics is not None:
            logger.info(
                "Epoch %d/%d | train_loss=%.6f train_mse=%.6f | valid_loss=%.6f valid_mse=%.6f",
                epoch + 1,
                cfg.epochs,
                train_epoch_metrics["loss"],
                train_epoch_metrics["mse"],
                valid_epoch_metrics["loss"],
                valid_epoch_metrics["mse"],
            )
        else:
            logger.info(
                "Epoch %d/%d | train_loss=%.6f train_mse=%.6f",
                epoch + 1,
                cfg.epochs,
                train_epoch_metrics["loss"],
                train_epoch_metrics["mse"],
            )

    logger.info("SAE training finished.")
    return SAETrainingOutput(model=model, train_curve=train_curve, valid_curve=valid_curve)


def _evaluate_epoch(
    model: SparseAutoencoder,
    loader: DataLoader,
    variant: str,
    l1_lambda: float,
    epoch: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    running = {"total": 0.0, "mse": 0.0, "sparsity": 0.0, "regularization": 0.0, "steps": 0}

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            x_hat, z = model(batch)
            loss = sae_loss(batch, x_hat, z, variant, l1_lambda)

            running["total"] += float(loss.total.item())
            running["mse"] += float(loss.mse.item())
            running["sparsity"] += float(loss.sparsity.item())
            running["regularization"] += float(loss.regularization.item())
            running["steps"] += 1

    return _normalize_running(epoch, variant, running)


def _normalize_running(epoch: int, variant: str, running: dict[str, float]) -> dict[str, float]:
    steps = max(1, int(running["steps"]))
    return {
        "epoch": float(epoch + 1),
        "variant": variant,
        "loss": running["total"] / steps,
        "mse": running["mse"] / steps,
        "sparsity": running["sparsity"] / steps,
        "regularization": running["regularization"] / steps,
    }
