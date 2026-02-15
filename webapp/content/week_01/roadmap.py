"""Section: Industry Applications and Course Roadmap

Why generative models matter in industry and what you'll build.
"""

import streamlit as st


def render():
    """Render the Industry Applications and Course Roadmap section."""
    st.markdown("---")
    st.markdown("## Why This Matters in Industry")

    st.markdown(
        "- **Content creation** — Text-to-image (DALL-E, Midjourney), video generation (Sora), "
        "3D asset generation. These are the models behind the products.\n"
        "- **Synthetic data & augmentation** — When real data is scarce, expensive, or private. "
        "Medical imaging, autonomous driving, and fraud detection all benefit.\n"
        "- **Representation learning** — Latent spaces from VAEs and diffusion models power "
        "downstream tasks: anomaly detection, semantic search, controllable generation."
    )

    st.markdown("---")
    st.markdown("## What You'll Build")

    st.markdown(
        "Over the next 10 weeks, you'll implement each major model family in the taxonomy above — "
        "from VAEs to flow matching. Each week adds a new tool to your generative modeling toolkit. "
        "By the end, you'll have a working implementation of every major approach, written in clean, "
        "industry-standard PyTorch."
    )

    st.markdown("---")
    st.markdown("## Course Roadmap")

    st.markdown(
        """
| Week | Topic | You'll Build |
|------|-------|-------------|
| 2 | Image Priors & VAE | Encoder, Decoder, VAE, ELBO loss |
| 3 | Normalizing Flows & AR | Coupling layers, autoregressive model |
| 4-5 | Autoregressive Models | PixelCNN, transformers for images |
| 5-6 | GANs | Generator, discriminator, training loop |
| 7-8 | Diffusion Models | DDPM, score matching, sampling |
| 9-10 | Flow Matching | Conditional flow matching, rectified flows |
| 11-13 | Applications | Video, 3D, robotics, molecules, proteins |
"""
    )
