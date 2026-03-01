from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.activation = nn.ReLU()

        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.encoder(x))

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


def sae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    z: torch.Tensor,
    l1_lambda: float,
) -> SAELossBreakdown:
    mse = torch.mean((x - x_hat) ** 2)
    sparsity = z.abs().mean()
    total = mse + l1_lambda * sparsity
    return SAELossBreakdown(total=total, mse=mse, sparsity=sparsity)
