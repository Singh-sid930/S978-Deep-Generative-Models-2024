"""Training infrastructure.

Shared training loops, callbacks, logging, and evaluation metrics.
"""

from dgm.training.vae_trainer import TrainingHistory, train_vae

__all__ = ["train_vae", "TrainingHistory"]
