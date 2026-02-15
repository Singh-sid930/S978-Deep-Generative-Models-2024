"""Training loop for Variational Autoencoder (VAE).

Provides a focused training function that tracks per-epoch loss components
and supports progress callbacks for integration with Streamlit or other UIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch.utils.data import DataLoader

from dgm.vae import VAE
from dgm.vae.loss import vae_loss


@dataclass
class TrainingHistory:
    """Loss history from VAE training.

    Tracks the three key loss components across epochs: total loss (ELBO),
    reconstruction loss, and KL divergence. Each is a list where index i
    corresponds to epoch i.

    Attributes:
        total_loss: Per-epoch average total loss (recon + KL).
        recon_loss: Per-epoch average reconstruction loss.
        kl_loss: Per-epoch average KL divergence loss.
    """

    total_loss: list[float] = field(default_factory=list)
    recon_loss: list[float] = field(default_factory=list)
    kl_loss: list[float] = field(default_factory=list)


def train_vae(
    model: VAE,
    train_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: torch.device | None = None,
    recon_loss_type: str = "bce",
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> TrainingHistory:
    """Train a VAE model on the provided data.

    Performs end-to-end training of a VAE, tracking reconstruction loss and
    KL divergence separately. The model is optimized to maximize the ELBO
    (minimize the negative ELBO).

    Args:
        model: The VAE model to train.
        train_loader: DataLoader providing batches of (data, labels).
            Labels are ignored; only data is used.
        epochs: Number of training epochs (default: 10).
        lr: Learning rate for Adam optimizer (default: 1e-3).
        device: Device to train on. If None, auto-detects CUDA availability.
        recon_loss_type: Type of reconstruction loss ("bce" or "mse").
            Default is "bce" for binary cross-entropy, suitable for normalized
            images in [0, 1].
        progress_callback: Optional callback function called after each epoch.
            Signature: callback(current_epoch, total_epochs, avg_total_loss).
            Useful for updating progress bars in UIs like Streamlit.

    Returns:
        TrainingHistory containing per-epoch loss components.

    Example:
        >>> from dgm.vae import VAE
        >>> from torch.utils.data import DataLoader, TensorDataset
        >>> import torch
        >>>
        >>> # Create dummy data
        >>> data = torch.randn(100, 784)
        >>> dataset = TensorDataset(data, torch.zeros(100))
        >>> loader = DataLoader(dataset, batch_size=32)
        >>>
        >>> # Train VAE
        >>> model = VAE(input_dim=784, latent_dim=20)
        >>> history = train_vae(model, loader, epochs=5, lr=1e-3)
        >>>
        >>> # Access loss history
        >>> print(f"Final total loss: {history.total_loss[-1]:.4f}")
        >>> print(f"Final recon loss: {history.recon_loss[-1]:.4f}")
        >>> print(f"Final KL loss: {history.kl_loss[-1]:.4f}")
    """
    # Auto-detect device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device and set to training mode
    model.to(device)
    model.train()

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initialize training history
    history = TrainingHistory()

    # Training loop
    for epoch in range(epochs):
        # Accumulators for epoch-level losses
        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            # Extract data, ignore labels
            x, _ = batch
            x = x.to(device)

            # Forward pass
            recon_x, mu, log_var = model(x)

            # Compute loss
            total, recon, kl = vae_loss(
                recon_x, x, mu, log_var, recon_loss_type=recon_loss_type
            )

            # Backward pass and optimization
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            # Accumulate batch losses
            epoch_total_loss += total.item()
            epoch_recon_loss += recon.item()
            epoch_kl_loss += kl.item()
            num_batches += 1

        # Compute epoch averages
        avg_total = epoch_total_loss / num_batches
        avg_recon = epoch_recon_loss / num_batches
        avg_kl = epoch_kl_loss / num_batches

        # Update history
        history.total_loss.append(avg_total)
        history.recon_loss.append(avg_recon)
        history.kl_loss.append(avg_kl)

        # Call progress callback if provided
        if progress_callback is not None:
            progress_callback(epoch + 1, epochs, avg_total)

    return history
