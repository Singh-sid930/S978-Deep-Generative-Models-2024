"""Decoder module for Variational Autoencoder.

The decoder maps latent codes back to the input space (reconstruction).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """MLP decoder that maps latent z back to input space.

    Architecture: latent_dim → [hidden_dims] → output_dim
    Uses Linear + ReLU for hidden layers, sigmoid on final output
    (appropriate for normalized image data in [0, 1]).

    Attributes:
        network: Sequential model containing all decoder layers.
        latent_dim: Dimensionality of latent space.
        output_dim: Dimensionality of output data.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dims: list[int] | None = None,
        output_dim: int = 784,
    ) -> None:
        """Initialize the decoder network.

        Args:
            latent_dim: Dimensionality of latent space (default: 64).
            hidden_dims: List of hidden layer dimensions. If None, defaults to [256, 512]
                (mirror of encoder architecture).
            output_dim: Dimensionality of output data (default: 784 for MNIST).
        """
        super().__init__()

        # Default hidden dimensions if not provided (mirror of encoder)
        if hidden_dims is None:
            hidden_dims = [256, 512]

        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Build the decoder network
        layers = []
        prev_dim = latent_dim

        # Hidden layers with ReLU activations
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Final output layer with Sigmoid activation
        # Sigmoid ensures output is in [0, 1] range for image reconstruction
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent code to reconstruct input.

        Args:
            z: Latent tensor of shape (batch_size, latent_dim).

        Returns:
            Reconstruction of shape (batch_size, output_dim).
            Values are in [0, 1] due to sigmoid activation.

        Hints:
            - Simply pass z through self.network.
            - The network already includes the sigmoid activation.
            - Output can be reshaped to image dimensions if needed downstream.

        Reference:
            Kingma & Welling, "Auto-Encoding Variational Bayes", 2013, Eq. 11-12.
        """
        raise NotImplementedError("Implement the decoder forward pass.")

    def __repr__(self) -> str:
        """Return string representation of the decoder."""
        return (
            f"Decoder(latent_dim={self.latent_dim}, "
            f"output_dim={self.output_dim}, "
            f"network={self.network})"
        )
