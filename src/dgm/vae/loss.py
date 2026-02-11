"""Loss functions for Variational Autoencoder.

Implements the VAE loss (ELBO) combining reconstruction loss and KL divergence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    recon_loss_type: str = "bce",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the VAE loss: reconstruction loss + KL divergence.

    The loss is the negative ELBO (Evidence Lower Bound):
        L = E_q[log p(x|z)] - KL(q(z|x) || p(z))

    We minimize the negative ELBO, which is equivalent to maximizing the ELBO.
    The reconstruction loss approximates E_q[log p(x|z)], and the KL term
    regularizes the latent distribution to be close to the prior p(z) = N(0, I).

    Args:
        recon_x: Reconstructed input, shape (batch_size, input_dim).
        x: Original input, shape (batch_size, input_dim).
        mu: Encoder mean, shape (batch_size, latent_dim).
        log_var: Encoder log variance, shape (batch_size, latent_dim).
        recon_loss_type: Type of reconstruction loss to use.
            - "bce": Binary cross-entropy (for normalized images in [0, 1]).
            - "mse": Mean squared error.

    Returns:
        Tuple of (total_loss, recon_loss, kl_loss), all scalar tensors.
        All losses are averaged over the batch.
        - total_loss: Sum of reconstruction loss and KL loss.
        - recon_loss: Reconstruction loss term.
        - kl_loss: KL divergence term.

    Hints:
        Reconstruction loss:
        - For BCE: F.binary_cross_entropy(recon_x, x, reduction='sum') / batch_size
          Use reduction='sum' to sum over all dimensions, then divide by batch size.
        - For MSE: F.mse_loss(recon_x, x, reduction='sum') / batch_size

        KL divergence for q(z|x) = N(mu, var) and p(z) = N(0, I):
        - Analytical form: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        - This sums over all latent dimensions and batch elements.
        - Then divide by batch_size for per-sample average.
        - Derivation: each dimension contributes -0.5 * (1 + log(var) - mu^2 - var)

        Total loss:
        - total_loss = recon_loss + kl_loss
        - Both terms should be averaged over the batch for stable training.

    Reference:
        Kingma & Welling, "Auto-Encoding Variational Bayes", 2013, Appendix B.
        KL divergence formula: Appendix B, Equation (10).
    """
    raise NotImplementedError("Implement the VAE loss function.")
