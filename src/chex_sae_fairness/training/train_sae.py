from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from chex_sae_fairness.config import SAEConfig
from chex_sae_fairness.models.sae import SparseAutoencoder, sae_loss


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
    for (batch,) in loader:
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

    for epoch in range(cfg.epochs):
        model.train()
        train_running = {"total": 0.0, "mse": 0.0, "sparsity": 0.0, "regularization": 0.0, "steps": 0}

        for (batch,) in train_loader:
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

        train_curve.append(_normalize_running(epoch, cfg.variant, train_running))

        if (epoch + 1) % cfg.eval_every_epochs == 0 or epoch == cfg.epochs - 1:
            valid_curve.append(_evaluate_epoch(model, valid_loader, cfg.variant, cfg.l1_lambda, epoch, device))

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
