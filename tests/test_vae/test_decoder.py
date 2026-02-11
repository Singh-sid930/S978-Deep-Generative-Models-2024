"""Tests for VAE Decoder module."""

import pytest
import torch

from dgm.vae.decoder import Decoder


class TestDecoderInit:
    """Tests for Decoder initialization (these should pass immediately)."""

    def test_default_construction(self):
        """Decoder can be created with default parameters."""
        decoder = Decoder()
        assert decoder.latent_dim == 64
        assert decoder.output_dim == 784

    def test_custom_dims(self):
        """Decoder accepts custom dimensions."""
        decoder = Decoder(latent_dim=32, hidden_dims=[64, 128], output_dim=256)
        assert decoder.latent_dim == 32
        assert decoder.output_dim == 256

    def test_network_structure(self):
        """Decoder builds network with correct layers."""
        decoder = Decoder(latent_dim=64, hidden_dims=[256, 512], output_dim=784)
        assert hasattr(decoder, "network")
        assert isinstance(decoder.network, torch.nn.Sequential)

    def test_default_hidden_dims(self):
        """Decoder uses [256, 512] as default hidden dimensions."""
        decoder = Decoder()
        # Network: Linear, ReLU, Linear, ReLU, Linear, Sigmoid (6 layers)
        assert len(decoder.network) == 6

    def test_final_sigmoid_present(self):
        """Decoder network ends with Sigmoid activation."""
        decoder = Decoder()
        assert isinstance(decoder.network[-1], torch.nn.Sigmoid)


class TestDecoderForward:
    """Tests for Decoder forward pass (xfail until implemented)."""

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_output_shape(self):
        """Decoder output has correct shape."""
        decoder = Decoder(latent_dim=64, output_dim=784)
        z = torch.randn(8, 64)
        out = decoder(z)
        assert out.shape == (8, 784)

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_output_range(self):
        """Sigmoid output should be in [0, 1]."""
        decoder = Decoder()
        z = torch.randn(8, 64)
        out = decoder(z)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_outputs_are_finite(self):
        """Decoder outputs should not contain NaN or Inf."""
        decoder = Decoder()
        z = torch.randn(8, 64)
        out = decoder(z)
        assert torch.isfinite(out).all()

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_deterministic(self):
        """Same input gives same output (no randomness in decoder)."""
        torch.manual_seed(42)
        decoder = Decoder()
        decoder.eval()
        z = torch.randn(4, 64)
        out1 = decoder(z)
        out2 = decoder(z)
        assert torch.allclose(out1, out2)

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_gradients_flow(self):
        """Gradients should flow through decoder."""
        decoder = Decoder(latent_dim=64, output_dim=784)
        z = torch.randn(4, 64)
        out = decoder(z)
        loss = out.sum()
        loss.backward()
        for param in decoder.parameters():
            assert param.grad is not None

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    @pytest.mark.parametrize("latent_dim", [16, 32, 64, 128])
    def test_different_latent_dims(self, latent_dim):
        """Decoder works with different latent dimensions."""
        decoder = Decoder(latent_dim=latent_dim, output_dim=784)
        z = torch.randn(4, latent_dim)
        out = decoder(z)
        assert out.shape == (4, 784)

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_zero_latent_input(self):
        """Decoder handles zero latent input."""
        decoder = Decoder()
        z = torch.zeros(4, 64)
        out = decoder(z)
        assert out.shape == (4, 784)
        assert torch.isfinite(out).all()
        assert out.min() >= 0.0
        assert out.max() <= 1.0
