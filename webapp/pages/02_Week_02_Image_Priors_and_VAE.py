"""Week 2: Image Priors & VAE."""

import streamlit as st

from webapp.page_utils import setup_page, render_sessions, render_assignments, render_mermaid

setup_page(2)
render_sessions(2)

# ── Part A: Image Priors ─────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## Understanding Image Priors")

st.markdown(
    "A natural image isn't random noise — it has smooth regions, sharp edges, repetitive "
    "textures, and predictable statistics. An **image prior** $p(x)$ is a model of what "
    '"natural images look like." It\'s the same idea as a Bayesian prior, but operating over '
    "pixel space. A good prior lets you separate signal from noise, fill in missing pixels, "
    "and super-resolve low-resolution images."
)

st.markdown("### Evolution of Image Priors")

render_mermaid(
    """
graph LR
    PS["Patch Statistics<br/><i>Efros & Leung 1999</i>"]
    GMM["Gaussian Mixtures<br/><i>Zoran & Weiss 2011</i>"]
    DIP["Deep Image Prior<br/><i>Ulyanov et al. 2018</i>"]
    LGM["Learned Generative<br/>Models<br/><i>VAEs, Diffusion...</i>"]
    PS --> GMM --> DIP --> LGM
""",
    height=300,
)

st.info(
    '**Key Insight: "Deep Image Prior"** — The architecture of a ConvNet itself is a prior '
    "over natural images. Even with random weights and no training data, optimizing a network "
    "to fit a single corrupted image produces a clean result — because the network structure "
    "resists noise. This is why neural networks are so effective for image generation."
)

st.markdown(
    "**Industry connection:** Image priors are the backbone of computational photography "
    "(phone cameras, Photoshop). Super-resolution, denoising, inpainting, and HDR all rely "
    "on prior knowledge of what images should look like."
)

# ── Part B: Variational Autoencoder ──────────────────────────────────────────

st.markdown("---")
st.markdown("## Variational Autoencoder (VAE)")

st.markdown("### The Core Problem")
st.markdown(
    "We want to learn $p(x)$ — the distribution of natural images. But images live in "
    "high-dimensional space (e.g., 784 dimensions for 28x28 MNIST). Directly modeling this is "
    "intractable. The VAE's key idea: introduce a low-dimensional **latent variable** $z$ that "
    "captures the essential structure, then learn both how to encode data into $z$ and how to "
    "generate data from $z$."
)

st.markdown("### Architecture")

render_mermaid(
    """
graph LR
    X["Input x"] --> ENC["Encoder<br/>q(z|x)"]
    ENC --> MU["μ"]
    ENC --> LV["log σ²"]
    MU --> REP["Reparameterize<br/>z = μ + σ · ε"]
    LV --> REP
    EP["ε ~ N(0,I)"] -.-> REP
    REP --> DEC["Decoder<br/>p(x|z)"]
    DEC --> XR["x̂"]
""",
    height=400,
)

st.markdown("### ELBO in Plain English")
st.markdown(
    "We can't compute $p(x)$ directly, but we can optimize a lower bound called the "
    "**ELBO** (Evidence Lower Bound):"
)
st.markdown(
    "**ELBO = Reconstruction Quality $-$ KL Divergence**\n\n"
    "- **Reconstruction quality**: How well can the decoder reconstruct $x$ from $z$? "
    "(Higher is better)\n"
    "- **KL divergence**: How close is the encoder's distribution $q(z|x)$ to the prior "
    "$p(z) = \\mathcal{N}(0, I)$? (Lower is better)\n\n"
    "Maximizing the ELBO pushes the model to reconstruct well while keeping the latent "
    "space well-organized."
)

st.latex(
    r"\mathcal{L}(\theta, \phi; x) = "
    r"\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - "
    r"D_{\mathrm{KL}}(q_\phi(z|x) \| p(z))"
)

st.markdown("### The Reparameterization Trick")
st.markdown(
    "We need gradients to flow through the sampling step $z \\sim q(z|x)$, but sampling "
    "is not differentiable. The trick: instead of sampling $z$ directly, we sample "
    "$\\varepsilon \\sim \\mathcal{N}(0, I)$ and compute:"
)
st.latex(r"z = \mu + \sigma \cdot \varepsilon")
st.markdown(
    "The randomness is in $\\varepsilon$ (which doesn't depend on parameters), so gradients "
    "flow through $\\mu$ and $\\sigma$ back to the encoder. This simple change is what makes "
    "VAEs trainable end-to-end with standard backpropagation."
)

st.markdown("---")
st.markdown("### Industry Applications")
st.info(
    "**In industry, VAEs are used for:**\n\n"
    "- **Anomaly detection** — high reconstruction error signals the model hasn't seen "
    "data like this before\n"
    "- **Latent space interpolation** — smooth transitions between data points "
    "(face morphing, style transfer)\n"
    "- **Data augmentation** — sample from the latent space to generate synthetic "
    "training data\n"
    "- **Controllable generation** — disentangled latent dimensions map to "
    "interpretable features"
)

# ── Assignment ───────────────────────────────────────────────────────────────

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

from webapp.chat import render_chat

render_chat(2)
