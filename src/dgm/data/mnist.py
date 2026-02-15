"""MNIST dataset wrapper with ready-to-use DataLoaders.

Provides convenience functions for loading MNIST with transforms suitable
for VAE training (normalized to [0,1], optionally flattened to 784-dim vectors).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass
class MNISTLoaders:
    """Container for train and test DataLoaders.

    Attributes:
        train: DataLoader for training set (shuffled).
        test: DataLoader for test set (not shuffled).
    """

    train: DataLoader
    test: DataLoader


def get_mnist_loaders(
    data_dir: str | Path = "./data",
    batch_size: int = 128,
    flatten: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> MNISTLoaders:
    """Create train and test DataLoaders for MNIST dataset.

    Downloads MNIST if not already cached. Applies ToTensor() transform
    (which maps pixel values from [0,255] to [0,1]) and optionally flattens
    images from (1,28,28) to (784,) for MLP-based models.

    Args:
        data_dir: Directory to download/cache MNIST data.
        batch_size: Number of samples per batch.
        flatten: If True, reshape images from (1,28,28) to (784,).
            Set to False for convolutional models.
        num_workers: Number of subprocesses for data loading.
            Set to 0 for single-process loading (useful for debugging).
        pin_memory: If True, use pinned memory for faster GPU transfer.
            Recommended when training on GPU.

    Returns:
        MNISTLoaders containing train and test DataLoaders.

    Example:
        >>> loaders = get_mnist_loaders(batch_size=64, flatten=True)
        >>> for batch in loaders.train:
        >>>     x, y = batch  # x: (64, 784), y: (64,)
        >>>     break
    """
    # Convert to Path for consistent handling
    data_dir = Path(data_dir)

    # Build transform pipeline
    transform_list = [transforms.ToTensor()]
    if flatten:
        # Reshape (1, 28, 28) → (784,)
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))

    transform = transforms.Compose(transform_list)

    # Load train and test datasets
    train_dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        transform=transform,
        download=True,
    )

    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        transform=transform,
        download=True,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return MNISTLoaders(train=train_loader, test=test_loader)
