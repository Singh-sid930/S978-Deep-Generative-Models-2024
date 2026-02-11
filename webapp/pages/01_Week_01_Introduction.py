"""Week 1: Introduction."""

import streamlit as st

from webapp.page_utils import setup_page, render_sessions, render_mermaid

setup_page(1)
render_sessions(1)

# --- Educational Content ---

st.markdown("---")
st.markdown("## What Are Generative Models?")
st.markdown(
    "Discriminative models learn decision boundaries — given an input, they predict a label "
    '("is this a cat or a dog?"). Generative models learn the data distribution $p(x)$ itself — '
    "they can create entirely new, realistic data points. If a discriminative model is a critic, "
    "a generative model is an artist. The central question: given a dataset of examples, can we "
    "learn a model that generates new examples indistinguishable from the real ones?"
)

st.markdown("---")
st.markdown("## Taxonomy of Generative Models")
st.markdown(
    "The course covers every major family of generative models. They differ in how they "
    "represent or approximate the data distribution:"
)

render_mermaid(
    """
graph TD
    GM["Generative Models"]
    GM --> ED["Explicit Density"]
    GM --> ID["Implicit Density"]
    GM --> SB["Score-Based"]
    ED --> TR["Tractable"]
    ED --> AP["Approximate"]
    TR --> AR["Autoregressive<br/>Weeks 3-5"]
    TR --> NF["Normalizing Flows<br/>Week 3"]
    AP --> VAE["VAE<br/>Week 2"]
    ID --> GAN["GAN<br/>Weeks 5-6"]
    SB --> DM["Diffusion<br/>Weeks 7-8"]
    SB --> FM["Flow Matching<br/>Weeks 9-10"]
""",
    height=600,
)

st.markdown(
    "**Explicit density** models define $p(x)$ directly — either exactly (autoregressive, flows) "
    "or approximately (VAE). **Implicit density** models (GANs) learn to generate samples without "
    "ever defining $p(x)$. **Score-based** models learn the gradient $\\nabla_x \\log p(x)$ and use "
    "it to iteratively refine samples. Each approach trades off sample quality, training stability, "
    "and inference speed differently."
)

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
