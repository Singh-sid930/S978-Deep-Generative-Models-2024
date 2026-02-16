"""Variational Autoencoder (VAE) model.

Combines encoder and decoder with the reparameterization trick to enable
end-to-end training of a generative model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dgm.vae.encoder import Encoder
from dgm.vae.decoder import Decoder


class VAE(nn.Module):
    """Variational Autoencoder.

    Composes an Encoder and Decoder with the reparameterization trick.
    The encoder maps input x to a distribution q(z|x) = N(mu, sigma^2).
    The reparameterization trick enables backpropagation through sampling.
    The decoder maps sampled z back to reconstruct x.

    Attributes:
        encoder: Encoder network that outputs distribution parameters.
        decoder: Decoder network that reconstructs from latent codes.
        latent_dim: Dimensionality of the latent space.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list[int] | None = None,
        latent_dim: int = 64,
    ) -> None:
        """Initialize the VAE.

        Args:
            input_dim: Dimensionality of input data (default: 784 for MNIST).
            hidden_dims: List of hidden layer dimensions. If None, defaults to [512, 256]
                for encoder and [256, 512] for decoder (mirrored).
            latent_dim: Dimensionality of latent space (default: 64).
        """
        super().__init__()

        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.latent_dim = latent_dim

        # Create encoder and decoder
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
        )

        # Decoder uses reversed hidden dimensions to mirror encoder
        decoder_hidden_dims = list(reversed(hidden_dims))
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=input_dim,
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Apply the reparameterization trick to sample from N(mu, var).

        Instead of sampling z ~ N(mu, var) directly (which is not differentiable),
        we sample epsilon ~ N(0, 1) and compute z = mu + std * epsilon.
        This allows gradients to flow through mu and std.

        Args:
            mu: Mean parameters of shape (batch_size, latent_dim).
            log_var: Log variance parameters of shape (batch_size, latent_dim).

        Returns:
            Sampled latent code z of shape (batch_size, latent_dim).

        Hints:
            - Compute std = exp(0.5 * log_var). We use log_var for numerical stability.
            - Sample epsilon from standard normal: torch.randn_like(std).
            - Compute z = mu + std * epsilon.
            - This maintains the same distribution as sampling from N(mu, var) directly,
              but gradients can flow through mu and log_var.

        Reference:
            Kingma & Welling, "Auto-Encoding Variational Bayes", 2013, Section 2.4.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std*eps
        
        return z

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE.

        Encodes input to distribution parameters, samples latent code using
        reparameterization trick, and decodes to reconstruct input.

        Args:
            x: Input tensor of shape (batch_size, input_dim) or (batch_size, C, H, W).

        Returns:
            Tuple of (reconstruction, mu, log_var):
                reconstruction: Reconstructed input of shape (batch_size, input_dim).
                mu: Mean parameters of shape (batch_size, latent_dim).
                log_var: Log variance parameters of shape (batch_size, latent_dim).
            The mu and log_var are needed for computing the KL divergence loss.

        Hints:
            - Encode x to get mu and log_var using self.encoder.
            - Sample z from the distribution using self.reparameterize.
            - Decode z to get the reconstruction using self.decoder.
            - Return all three: reconstruction, mu, log_var (needed for loss).

        Reference:
            Kingma & Welling, "Auto-Encoding Variational Bayes", 2013, Algorithm 1.
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recons = self.decoder(z)

        return(recons, mu, log_var)

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate samples from the prior p(z) = N(0, I).

        Samples from the prior distribution over latent codes and decodes them
        to generate new data. This is used for generating new samples from the
        trained model.

        Args:
            num_samples: Number of samples to generate.
            device: Device to generate samples on (cpu or cuda).

        Returns:
            Generated samples of shape (num_samples, output_dim).

        Hints:
            - Sample z from standard normal: torch.randn(num_samples, self.latent_dim).
            - Move z to the specified device.
            - Decode z using self.decoder to get generated samples.
            - No need to call reparameterize here since we're sampling from prior.

        Reference:
            Kingma & Welling, "Auto-Encoding Variational Bayes", 2013, Section 2.3.
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(device)
        gen = self.decoder(z)
        return gen

    def __repr__(self) -> str:
        """Return string representation of the VAE."""
        return (
            f"VAE(latent_dim={self.latent_dim}, "
            f"encoder={self.encoder}, "
            f"decoder={self.decoder})"
        )
