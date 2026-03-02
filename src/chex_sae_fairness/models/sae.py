from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        variant: str = "l1",
        topk_k: int = 64,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.topk_k = topk_k
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.activation = nn.ReLU()

        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.activation(self.encoder(x))
        if self.variant == "l1":
            return z
        if self.variant == "topk":
            return _topk_sparse(z, self.topk_k)
        raise ValueError(f"Unknown SAE variant: {self.variant}")

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


@dataclass(slots=True)
class SAELossBreakdown:
    total: torch.Tensor
    mse: torch.Tensor
    sparsity: torch.Tensor
    regularization: torch.Tensor


def sae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    z: torch.Tensor,
    variant: str,
    l1_lambda: float,
) -> SAELossBreakdown:
    mse = torch.mean((x - x_hat) ** 2)
    sparsity = z.abs().mean()
    if variant == "l1":
        regularization = l1_lambda * sparsity
    elif variant == "topk":
        regularization = torch.zeros((), device=x.device)
    else:
        raise ValueError(f"Unknown SAE variant: {variant}")

    total = mse + regularization
    return SAELossBreakdown(
        total=total,
        mse=mse,
        sparsity=sparsity,
        regularization=regularization,
    )


def _topk_sparse(z: torch.Tensor, topk_k: int) -> torch.Tensor:
    if topk_k <= 0:
        raise ValueError("topk_k must be positive for top-k SAE.")
    if topk_k >= z.shape[-1]:
        return z

    _, indices = torch.topk(z, k=topk_k, dim=-1)
    mask = torch.zeros_like(z)
    mask.scatter_(dim=-1, index=indices, value=1.0)
    return z * mask
