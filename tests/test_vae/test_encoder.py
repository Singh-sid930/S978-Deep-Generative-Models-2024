"""Tests for VAE Encoder module."""

import pytest
import torch

from dgm.vae.encoder import Encoder


class TestEncoderInit:
    """Tests for Encoder initialization (these should pass immediately)."""

    def test_default_construction(self):
        """Encoder can be created with default parameters."""
        encoder = Encoder()
        assert encoder.input_dim == 784
        assert encoder.latent_dim == 64

    def test_custom_dims(self):
        """Encoder accepts custom dimensions."""
        encoder = Encoder(input_dim=256, hidden_dims=[128, 64], latent_dim=32)
        assert encoder.input_dim == 256
        assert encoder.latent_dim == 32

    def test_network_structure(self):
        """Encoder builds correct network layers."""
        encoder = Encoder(input_dim=784, hidden_dims=[512, 256], latent_dim=64)
        assert hasattr(encoder, "network")
        assert hasattr(encoder, "fc_mu")
        assert hasattr(encoder, "fc_log_var")

    def test_default_hidden_dims(self):
        """Encoder uses [512, 256] as default hidden dimensions."""
        encoder = Encoder()
        # Network should have 4 layers: Linear, ReLU, Linear, ReLU
        assert len(encoder.network) == 4

    def test_network_is_sequential(self):
        """Encoder network is a torch.nn.Sequential."""
        encoder = Encoder()
        assert isinstance(encoder.network, torch.nn.Sequential)

    def test_fc_mu_output_size(self):
        """fc_mu projects to latent_dim."""
        encoder = Encoder(latent_dim=32)
        assert encoder.fc_mu.out_features == 32

    def test_fc_log_var_output_size(self):
        """fc_log_var projects to latent_dim."""
        encoder = Encoder(latent_dim=32)
        assert encoder.fc_log_var.out_features == 32


class TestEncoderForward:
    """Tests for Encoder forward pass (xfail until implemented)."""

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_output_shapes(self):
        """Forward returns (mu, log_var) with correct shapes."""
        encoder = Encoder(input_dim=784, latent_dim=64)
        x = torch.randn(8, 784)
        mu, log_var = encoder(x)
        assert mu.shape == (8, 64)
        assert log_var.shape == (8, 64)

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    @pytest.mark.parametrize("latent_dim", [16, 32, 64, 128])
    def test_different_latent_dims(self, latent_dim):
        """Output shapes match specified latent_dim."""
        encoder = Encoder(input_dim=784, latent_dim=latent_dim)
        x = torch.randn(4, 784)
        mu, log_var = encoder(x)
        assert mu.shape == (4, latent_dim)
        assert log_var.shape == (4, latent_dim)

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_deterministic(self):
        """Same input gives same output (no randomness in encoder)."""
        torch.manual_seed(42)
        encoder = Encoder()
        encoder.eval()
        x = torch.randn(4, 784)
        mu1, lv1 = encoder(x)
        mu2, lv2 = encoder(x)
        assert torch.allclose(mu1, mu2)
        assert torch.allclose(lv1, lv2)

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_outputs_are_finite(self):
        """Encoder outputs should not contain NaN or Inf."""
        encoder = Encoder()
        x = torch.randn(8, 784)
        mu, log_var = encoder(x)
        assert torch.isfinite(mu).all()
        assert torch.isfinite(log_var).all()

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_gradients_flow(self):
        """Gradients should flow through encoder."""
        encoder = Encoder(input_dim=784, latent_dim=64)
        x = torch.randn(4, 784)
        mu, log_var = encoder(x)
        loss = mu.sum() + log_var.sum()
        loss.backward()
        for param in encoder.parameters():
            assert param.grad is not None

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_returns_tuple(self):
        """Forward should return a tuple of two tensors."""
        encoder = Encoder()
        x = torch.randn(4, 784)
        result = encoder(x)
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_with_fixture(self, batch_of_vectors):
        """Encoder works with test fixture data."""
        encoder = Encoder(input_dim=784, latent_dim=64)
        mu, log_var = encoder(batch_of_vectors)
        assert mu.shape == (4, 64)
        assert log_var.shape == (4, 64)
