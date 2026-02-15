"""Visualization utilities for deep generative models.

Pure matplotlib functions that return Figure objects for display in notebooks
or Streamlit apps. All functions handle PyTorch tensors and numpy arrays.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_curves(
    history,
    figsize: tuple[int, int] = (10, 4),
) -> plt.Figure:
    """Plot training loss curves over epochs.

    Args:
        history: Object with attributes `total_loss`, `recon_loss`, `kl_loss`,
            each a list of floats (one per epoch).
        figsize: Figure size in inches (width, height).

    Returns:
        Matplotlib Figure with loss curves.

    Example:
        >>> fig = plot_training_curves(training_history)
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(history.total_loss) + 1)

    ax.plot(epochs, history.total_loss, color="#292524", label="Total Loss", linewidth=2)
    ax.plot(
        epochs, history.recon_loss, color="#6B8EAE", label="Reconstruction Loss", linewidth=2
    )
    ax.plot(epochs, history.kl_loss, color="#A8735A", label="KL Divergence", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_reconstructions(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    n: int = 8,
    figsize: tuple[int, int] = (12, 3),
) -> plt.Figure:
    """Plot original images alongside their reconstructions.

    Args:
        originals: Original images, shape (batch, 784) or (batch, 1, 28, 28).
        reconstructions: Reconstructed images, same shape as originals.
        n: Number of image pairs to display.
        figsize: Figure size in inches (width, height).

    Returns:
        Matplotlib Figure with two rows: originals on top, reconstructions below.

    Example:
        >>> fig = plot_reconstructions(x_test[:8], x_recon[:8])
        >>> plt.show()
    """
    # Convert to numpy and ensure we have enough samples
    originals_np = originals.detach().cpu().numpy()
    reconstructions_np = reconstructions.detach().cpu().numpy()

    n = min(n, originals_np.shape[0])

    # Reshape to (n, 28, 28) if needed
    if originals_np.ndim == 2:
        # Shape is (batch, 784)
        originals_np = originals_np[:n].reshape(n, 28, 28)
        reconstructions_np = reconstructions_np[:n].reshape(n, 28, 28)
    else:
        # Shape is (batch, 1, 28, 28)
        originals_np = originals_np[:n, 0]
        reconstructions_np = reconstructions_np[:n, 0]

    fig, axes = plt.subplots(2, n, figsize=figsize)

    for i in range(n):
        # Top row: originals
        axes[0, i].imshow(originals_np[i], cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")

        # Bottom row: reconstructions
        axes[1, i].imshow(reconstructions_np[i], cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")

    # Add row labels
    axes[0, 0].set_ylabel("Input", rotation=0, labelpad=30, fontsize=10)
    axes[1, 0].set_ylabel("Reconstruction", rotation=0, labelpad=30, fontsize=10)

    plt.tight_layout()
    return fig


def plot_samples(
    samples: torch.Tensor,
    nrow: int = 8,
    ncol: int = 8,
    figsize: tuple[int, int] = (8, 8),
) -> plt.Figure:
    """Plot a grid of generated samples.

    Args:
        samples: Generated samples, shape (N, 784) or (N, 1, 28, 28).
        nrow: Number of rows in the grid.
        ncol: Number of columns in the grid.
        figsize: Figure size in inches (width, height).

    Returns:
        Matplotlib Figure with a grid of sample images.

    Example:
        >>> z = torch.randn(64, latent_dim)
        >>> samples = model.sample(z)
        >>> fig = plot_samples(samples)
        >>> plt.show()
    """
    samples_np = samples.detach().cpu().numpy()

    # Reshape to (N, 28, 28) if needed
    if samples_np.ndim == 2:
        # Shape is (N, 784)
        samples_np = samples_np.reshape(-1, 28, 28)
    else:
        # Shape is (N, 1, 28, 28)
        samples_np = samples_np[:, 0]

    # Limit to nrow * ncol samples
    n_samples = min(nrow * ncol, samples_np.shape[0])
    samples_np = samples_np[:n_samples]

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = axes.flatten()

    for i in range(n_samples):
        axes[i].imshow(samples_np[i], cmap="gray", vmin=0, vmax=1)
        axes[i].axis("off")

    # Turn off any unused subplots
    for i in range(n_samples, nrow * ncol):
        axes[i].axis("off")

    fig.suptitle("Samples from Prior", fontsize=14, y=0.98)
    plt.tight_layout()
    return fig


def plot_latent_space(
    latents: np.ndarray,
    labels: np.ndarray,
    figsize: tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Visualize latent space with dimensionality reduction if needed.

    For 2D latent spaces, plots directly. For higher dimensions, uses t-SNE
    (if sklearn available) or falls back to PCA via SVD.

    Args:
        latents: Latent representations, shape (N, latent_dim).
        labels: Integer class labels, shape (N,).
        figsize: Figure size in inches (width, height).

    Returns:
        Matplotlib Figure with scatter plot of latent space.

    Example:
        >>> z, y = encode_dataset(model, test_loader)
        >>> fig = plot_latent_space(z, y)
        >>> plt.show()
    """
    latent_dim = latents.shape[1]

    # Determine visualization method and reduce dimensionality if needed
    if latent_dim == 2:
        # Direct visualization
        latents_2d = latents
        method = "2D"
    elif latent_dim > 2:
        # Limit to 5000 points for t-SNE performance
        if latents.shape[0] > 5000:
            indices = np.random.choice(latents.shape[0], 5000, replace=False)
            latents_subset = latents[indices]
            labels_subset = labels[indices]
        else:
            latents_subset = latents
            labels_subset = labels

        # Try t-SNE, fall back to PCA
        try:
            from sklearn.manifold import TSNE

            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            latents_2d = tsne.fit_transform(latents_subset)
            labels = labels_subset
            method = "t-SNE"
        except ImportError:
            # Fall back to PCA using numpy SVD
            # Center the data
            mean = np.mean(latents_subset, axis=0)
            centered = latents_subset - mean

            # Compute SVD
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)

            # Project onto first 2 principal components
            latents_2d = U[:, :2] @ np.diag(S[:2])
            labels = labels_subset
            method = "PCA"
    else:
        raise ValueError(f"latent_dim must be >= 2, got {latent_dim}")

    # Create scatter plot
    fig, ax = plt.subplots(figsize=figsize)

    # Use tab10 colormap for digits 0-9
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for label in unique_labels:
        mask = labels == label
        ax.scatter(
            latents_2d[mask, 0],
            latents_2d[mask, 1],
            c=[colors[int(label)]],
            label=f"{int(label)}",
            alpha=0.6,
            s=20,
        )

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(f"Latent Space ({method})")
    ax.legend(title="Digit", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_interpolation(
    model,
    z_start: torch.Tensor,
    z_end: torch.Tensor,
    steps: int = 10,
    device: str | None = None,
    figsize: tuple[int, int] = (15, 2),
) -> plt.Figure:
    """Visualize interpolation between two latent points.

    Args:
        model: Model with a `.decoder` attribute (nn.Module).
        z_start: Starting latent vector, shape (latent_dim,).
        z_end: Ending latent vector, shape (latent_dim,).
        steps: Number of interpolation steps.
        device: Device to run decoding on. If None, uses z_start.device.
        figsize: Figure size in inches (width, height).

    Returns:
        Matplotlib Figure showing interpolated images.

    Example:
        >>> z1 = torch.randn(latent_dim)
        >>> z2 = torch.randn(latent_dim)
        >>> fig = plot_interpolation(model, z1, z2, steps=12)
        >>> plt.show()
    """
    if device is None:
        device = z_start.device

    # Create interpolation points: alpha in [0, 1]
    alphas = torch.linspace(0, 1, steps, device=device)

    # Interpolate: z(alpha) = (1 - alpha) * z_start + alpha * z_end
    z_start = z_start.to(device)
    z_end = z_end.to(device)

    # Shape: (steps, latent_dim)
    z_interp = torch.stack(
        [(1 - alpha) * z_start + alpha * z_end for alpha in alphas]
    )

    # Decode interpolated points
    with torch.no_grad():
        decoded = model.decoder(z_interp)

    # Convert to numpy and reshape
    decoded_np = decoded.detach().cpu().numpy()

    if decoded_np.ndim == 2:
        # Shape is (steps, 784)
        decoded_np = decoded_np.reshape(steps, 28, 28)
    else:
        # Shape is (steps, 1, 28, 28)
        decoded_np = decoded_np[:, 0]

    # Create figure with a single row
    fig, axes = plt.subplots(1, steps, figsize=figsize)

    for i in range(steps):
        axes[i].imshow(decoded_np[i], cmap="gray", vmin=0, vmax=1)
        axes[i].axis("off")

    fig.suptitle("Latent Space Interpolation", fontsize=14, y=1.05)
    plt.tight_layout()
    return fig
