"""Tests for VAE loss function."""

import pytest
import torch

from dgm.vae.loss import vae_loss


class TestVAELoss:
    """Tests for vae_loss function (xfail until implemented)."""

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_returns_three_tensors(self):
        """Loss returns (total, recon, kl) tuple of scalars."""
        recon_x = torch.sigmoid(torch.randn(8, 784))
        x = torch.rand(8, 784)
        mu = torch.randn(8, 64)
        log_var = torch.randn(8, 64)
        total, recon, kl = vae_loss(recon_x, x, mu, log_var)
        assert total.dim() == 0  # scalar
        assert recon.dim() == 0
        assert kl.dim() == 0

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_kl_zero_at_prior(self):
        """KL should be ~0 when encoder outputs match the prior (mu=0, log_var=0)."""
        recon_x = torch.sigmoid(torch.randn(8, 784))
        x = torch.rand(8, 784)
        mu = torch.zeros(8, 64)
        log_var = torch.zeros(8, 64)
        total, recon, kl = vae_loss(recon_x, x, mu, log_var)
        assert kl.item() == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_total_equals_sum(self):
        """Total loss should equal recon + kl."""
        recon_x = torch.sigmoid(torch.randn(8, 784))
        x = torch.rand(8, 784)
        mu = torch.randn(8, 64)
        log_var = torch.randn(8, 64)
        total, recon, kl = vae_loss(recon_x, x, mu, log_var)
        assert total.item() == pytest.approx((recon + kl).item(), rel=1e-5)

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_loss_nonnegative(self):
        """All loss components should be non-negative."""
        recon_x = torch.sigmoid(torch.randn(8, 784))
        x = torch.rand(8, 784)
        mu = torch.randn(8, 64)
        log_var = torch.randn(8, 64)
        total, recon, kl = vae_loss(recon_x, x, mu, log_var)
        assert recon.item() >= 0
        assert kl.item() >= 0

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_mse_loss_type(self):
        """MSE reconstruction loss type should work."""
        recon_x = torch.sigmoid(torch.randn(8, 784))
        x = torch.rand(8, 784)
        mu = torch.randn(8, 64)
        log_var = torch.randn(8, 64)
        total, recon, kl = vae_loss(recon_x, x, mu, log_var, recon_loss_type="mse")
        assert total.dim() == 0

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_kl_increases_with_deviation(self):
        """KL should increase when mu deviates from 0."""
        recon_x = torch.sigmoid(torch.randn(8, 784))
        x = torch.rand(8, 784)
        log_var = torch.zeros(8, 64)

        _, _, kl_small = vae_loss(recon_x, x, torch.zeros(8, 64), log_var)
        _, _, kl_large = vae_loss(recon_x, x, torch.full((8, 64), 5.0), log_var)
        assert kl_large.item() > kl_small.item()

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_gradients_flow(self):
        """Gradients should flow through the loss."""
        recon_x = torch.sigmoid(torch.randn(8, 784, requires_grad=True))
        x = torch.rand(8, 784)
        mu = torch.randn(8, 64, requires_grad=True)
        log_var = torch.randn(8, 64, requires_grad=True)
        total, _, _ = vae_loss(recon_x, x, mu, log_var)
        total.backward()
        assert mu.grad is not None
        assert log_var.grad is not None

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_all_outputs_finite(self):
        """All loss components should be finite."""
        recon_x = torch.sigmoid(torch.randn(8, 784))
        x = torch.rand(8, 784)
        mu = torch.randn(8, 64)
        log_var = torch.randn(8, 64)
        total, recon, kl = vae_loss(recon_x, x, mu, log_var)
        assert torch.isfinite(total)
        assert torch.isfinite(recon)
        assert torch.isfinite(kl)
