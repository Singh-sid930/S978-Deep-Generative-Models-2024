"""Section: Assignment

Renders the VAE implementation assignment section.
"""

import streamlit as st

from webapp.page_utils import render_assignments


def render():
    """Render the assignment section."""
    st.markdown("---")
    render_assignments(["dgm.vae"])

    st.markdown("### Your Implementation")
    st.markdown(
        "Complete these four files:\n\n"
        "1. **`src/dgm/vae/encoder.py`** — `Encoder.forward()`: pass input through the network, "
        "project to $\\mu$ and $\\log \\sigma^2$\n"
        "2. **`src/dgm/vae/decoder.py`** — `Decoder.forward()`: pass latent code through the "
        "network to reconstruct\n"
        "3. **`src/dgm/vae/vae.py`** — `VAE.reparameterize()`, `VAE.forward()`, `VAE.sample()`\n"
        "4. **`src/dgm/vae/loss.py`** — `vae_loss()`: compute reconstruction loss + KL divergence\n\n"
        "Each file has detailed hints in the docstrings. Read them carefully before implementing."
    )
