import torch

from chex_sae_fairness.models.sae import SparseAutoencoder, sae_loss


def test_topk_encoder_emits_k_nonzero_latents() -> None:
    model = SparseAutoencoder(input_dim=6, latent_dim=10, variant="topk", topk_k=3)
    x = torch.randn(4, 6)

    with torch.no_grad():
        z = model.encode(x)

    nonzero_per_row = (z > 0).sum(dim=1)
    assert torch.all(nonzero_per_row <= 3)


def test_topk_loss_has_zero_regularization_term() -> None:
    x = torch.tensor([[1.0, 2.0], [0.5, -0.2]])
    x_hat = x.clone()
    z = torch.tensor([[0.1, 0.0], [0.2, 0.0]])

    out = sae_loss(x=x, x_hat=x_hat, z=z, variant="topk", l1_lambda=1.0)

    assert float(out.mse.item()) == 0.0
    assert float(out.regularization.item()) == 0.0
