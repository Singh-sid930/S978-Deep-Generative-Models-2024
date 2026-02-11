"""Tests for the full VAE model."""

import pytest
import torch

from dgm.vae import VAE


class TestVAEInit:
    """Tests for VAE initialization (these should pass immediately)."""

    def test_default_construction(self):
        """VAE can be created with default parameters."""
        vae = VAE()
        assert vae.latent_dim == 64
        assert hasattr(vae, "encoder")
        assert hasattr(vae, "decoder")

    def test_custom_dims(self):
        """VAE accepts custom dimensions."""
        vae = VAE(input_dim=256, hidden_dims=[128, 64], latent_dim=32)
        assert vae.latent_dim == 32

    def test_encoder_decoder_mirrored(self):
        """Encoder and decoder have mirrored architectures."""
        vae = VAE(input_dim=784, hidden_dims=[512, 256], latent_dim=64)
        # Encoder: 784 -> 512 -> 256 -> (64, 64)
        assert vae.encoder.fc_mu.out_features == 64
        # Decoder: 64 -> 256 -> 512 -> 784
        assert vae.decoder.output_dim == 784


class TestVAEReparameterize:
    """Tests for reparameterization trick (xfail until implemented)."""

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_reparameterize_shape(self):
        """Reparameterize output has correct shape."""
        vae = VAE(latent_dim=64)
        mu = torch.zeros(8, 64)
        log_var = torch.zeros(8, 64)
        z = vae.reparameterize(mu, log_var)
        assert z.shape == (8, 64)

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_reparameterize_stochastic(self):
        """In train mode, reparameterize should produce different samples."""
        vae = VAE(latent_dim=64)
        vae.train()
        mu = torch.zeros(8, 64)
        log_var = torch.ones(8, 64)  # non-zero variance
        z1 = vae.reparameterize(mu, log_var)
        z2 = vae.reparameterize(mu, log_var)
        assert not torch.allclose(z1, z2)

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_reparameterize_mean(self):
        """With zero log_var, samples should be close to mu."""
        vae = VAE(latent_dim=64)
        mu = torch.randn(8, 64)
        log_var = torch.full((8, 64), -20.0)  # very small variance
        z = vae.reparameterize(mu, log_var)
        assert torch.allclose(z, mu, atol=1e-3)

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_reparameterize_gradients(self):
        """Gradients should flow through reparameterization."""
        vae = VAE(latent_dim=64)
        mu = torch.randn(8, 64, requires_grad=True)
        log_var = torch.randn(8, 64, requires_grad=True)
        z = vae.reparameterize(mu, log_var)
        z.sum().backward()
        assert mu.grad is not None
        assert log_var.grad is not None


class TestVAEForward:
    """Tests for VAE forward pass (xfail until implemented)."""

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_forward_shapes(self):
        """Forward returns (recon, mu, log_var) with correct shapes."""
        vae = VAE(input_dim=784, latent_dim=64)
        x = torch.randn(8, 784)
        recon, mu, log_var = vae(x)
        assert recon.shape == (8, 784)
        assert mu.shape == (8, 64)
        assert log_var.shape == (8, 64)

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_forward_returns_tuple_of_three(self):
        """Forward should return a tuple of three tensors."""
        vae = VAE()
        x = torch.randn(4, 784)
        result = vae(x)
        assert isinstance(result, tuple)
        assert len(result) == 3

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_reconstruction_range(self):
        """Reconstruction should be in [0, 1] due to sigmoid."""
        vae = VAE(input_dim=784, latent_dim=64)
        x = torch.randn(8, 784)
        recon, _, _ = vae(x)
        assert recon.min() >= 0.0
        assert recon.max() <= 1.0

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_roundtrip(self, batch_of_vectors):
        """Forward pass on fixture data doesn't error."""
        vae = VAE(input_dim=784, latent_dim=64)
        recon, mu, log_var = vae(batch_of_vectors)
        assert recon.shape == batch_of_vectors.shape


class TestVAESample:
    """Tests for VAE sampling (xfail until implemented)."""

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_sample_shape(self):
        """sample() returns correct shape."""
        vae = VAE(input_dim=784, latent_dim=64)
        samples = vae.sample(16, torch.device("cpu"))
        assert samples.shape == (16, 784)

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_sample_range(self):
        """Samples should be in [0, 1] due to sigmoid decoder."""
        vae = VAE(input_dim=784, latent_dim=64)
        samples = vae.sample(16, torch.device("cpu"))
        assert samples.min() >= 0.0
        assert samples.max() <= 1.0

    @pytest.mark.xfail(raises=NotImplementedError, reason="Stub not yet implemented")
    def test_sample_stochastic(self):
        """Different calls to sample should produce different results."""
        vae = VAE(input_dim=784, latent_dim=64)
        s1 = vae.sample(8, torch.device("cpu"))
        s2 = vae.sample(8, torch.device("cpu"))
        assert not torch.allclose(s1, s2)
