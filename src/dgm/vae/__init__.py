"""Variational Autoencoder (VAE) implementations.

Course reference: Week 2 - Lecture: Variational Autoencoder (VAE)
Covers: ELBO, reparameterization trick, encoder-decoder architecture.
"""

from dgm.vae.decoder import Decoder
from dgm.vae.encoder import Encoder
from dgm.vae.loss import vae_loss
from dgm.vae.vae import VAE

__all__ = ["Encoder", "Decoder", "VAE", "vae_loss"]
