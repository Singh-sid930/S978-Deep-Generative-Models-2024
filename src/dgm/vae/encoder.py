"""Encoder module for Variational Autoencoder.

The encoder maps input data to latent distribution parameters (mu, log_var).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """MLP encoder that maps input x to latent distribution parameters (mu, log_var).

    Architecture: input_dim → [hidden_dims] → two parallel heads → latent_dim
    The shared body uses Linear + ReLU layers. Two separate linear heads
    project to mu and log_var respectively.

    Attributes:
        network: Sequential model containing the shared hidden layers.
        fc_mu: Linear layer projecting to mean parameters.
        fc_log_var: Linear layer projecting to log variance parameters.
        input_dim: Dimensionality of input data.
        latent_dim: Dimensionality of latent space.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list[int] | None = None,
        latent_dim: int = 64,
    ) -> None:
        """Initialize the encoder network.

        Args:
            input_dim: Dimensionality of input data (default: 784 for MNIST).
            hidden_dims: List of hidden layer dimensions. If None, defaults to [512, 256].
            latent_dim: Dimensionality of latent space (default: 64).
        """
        super().__init__()

        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Build the shared encoder network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)

        # Separate heads for mu and log_var
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input into latent distribution parameters.

        Args:
            x: Input tensor of shape (batch_size, input_dim) or (batch_size, C, H, W).
               Will be flattened if needed.

        Returns:
            Tuple of (mu, log_var), each of shape (batch_size, latent_dim).
            mu: Mean parameters of the latent distribution.
            log_var: Log variance parameters of the latent distribution.

        Hints:
            - Flatten x to (batch_size, input_dim) if it has more dimensions.
            - Pass flattened x through self.network to get hidden features.
            - Use self.fc_mu and self.fc_log_var to project to latent parameters.
            - Both mu and log_var should have the same shape: (batch_size, latent_dim).

        Reference:
            Kingma & Welling, "Auto-Encoding Variational Bayes", 2013, Eq. 9-10.
        """
        raise NotImplementedError("Implement the encoder forward pass.")

    def __repr__(self) -> str:
        """Return string representation of the encoder."""
        return (
            f"Encoder(input_dim={self.input_dim}, "
            f"latent_dim={self.latent_dim}, "
            f"network={self.network})"
        )
