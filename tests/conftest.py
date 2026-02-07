"""Shared test fixtures for the dgm test suite."""

import pytest
import torch


@pytest.fixture
def device():
    """Use CPU for tests by default. GPU tests use @pytest.mark.gpu."""
    return torch.device("cpu")


@pytest.fixture
def seed():
    """Set reproducible random seed for test determinism."""
    torch.manual_seed(42)
    return 42


@pytest.fixture
def batch_of_images():
    """A small batch of fake single-channel 28x28 images for testing."""
    torch.manual_seed(42)
    return torch.randn(4, 1, 28, 28)


@pytest.fixture
def batch_of_vectors():
    """A small batch of flat vectors (e.g., flattened MNIST) for testing."""
    torch.manual_seed(42)
    return torch.randn(4, 784)
