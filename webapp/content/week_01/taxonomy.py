"""Section: Taxonomy of Generative Models

Overview of all major generative model families covered in the course.
"""

import streamlit as st

from webapp.page_utils import render_mermaid


def render():
    """Render the Taxonomy of Generative Models section."""
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

    st.markdown("")
    st.markdown("### Understanding Each Approach")

    with st.expander("**Autoregressive Models** — Sequential Prediction"):
        st.markdown(
            "**What it means:** Predict data one piece at a time, conditioning each step on all previous steps. "
            "Like writing a sentence word by word, where each word depends on everything written before it."
        )
        st.markdown(
            "**Concrete example:** GPT predicts the next token given all previous tokens. PixelCNN generates "
            "an image pixel-by-pixel in raster-scan order (top-left to bottom-right)."
        )
        st.markdown(
            "**Key insight:** The variable ordering is **designed** (e.g., left-to-right, top-to-bottom), "
            "but the conditional distributions $p(x_i | x_{<i})$ are **learned** via neural networks. "
            "This design choice makes the density tractable via the chain rule: "
            "$p(x) = \\prod_{i=1}^{n} p(x_i | x_{<i})$."
        )

    with st.expander("**Normalizing Flows** — Invertible Transformations"):
        st.markdown(
            "**What it means:** Transform a simple distribution (Gaussian) into the data distribution through "
            "a series of invertible, differentiable transformations. Like sculpting clay — each deformation "
            "is reversible."
        )
        st.markdown(
            "**Concrete example:** Glow, RealNVP. Start with noise $z \\sim \\mathcal{N}(0, I)$, apply "
            "invertible coupling layers to produce realistic images $x = f(z)$."
        )
        st.markdown(
            "**Key insight:** Invertibility is the core constraint (unlike AR which imposes ordering). "
            "We can compute exact densities via change-of-variables: "
            "$p(x) = p(z) \\left|\\det \\frac{\\partial f^{-1}}{\\partial x}\\right|$."
        )

    with st.expander("**Variational Autoencoders (VAE)** — Latent Compression"):
        st.markdown(
            "**What it means:** Learn a low-dimensional latent code $z$ that captures the essence of the data. "
            "An encoder compresses data $x \\to z$, a decoder reconstructs $z \\to x$. The variational gap "
            "means we optimize a lower bound (ELBO) instead of the exact likelihood."
        )
        st.markdown(
            "**Concrete example:** A face VAE might learn latent dimensions for 'smiling', 'age', 'lighting'. "
            "Interpolating in latent space produces smooth morphs between faces."
        )
        st.markdown(
            "**Key insight:** Explicit latent variables $z$ with a simple prior $p(z) = \\mathcal{N}(0, I)$. "
            "The approximate posterior $q_\\phi(z|x)$ and likelihood $p_\\theta(x|z)$ are learned. "
            "Trade-off: samples are often blurry but training is stable."
        )

    with st.expander("**Generative Adversarial Networks (GAN)** — Generator vs Discriminator"):
        st.markdown(
            "**What it means:** Train two networks in opposition. A generator creates fake samples, a discriminator "
            "tries to distinguish real from fake. Like a counterfeiter vs a detective locked in an arms race."
        )
        st.markdown(
            "**Concrete example:** StyleGAN generates photorealistic faces. The generator never sees real images — "
            "it only receives gradients from the discriminator's judgment."
        )
        st.markdown(
            "**Key insight:** Implicit density — the generator produces samples from $p(x)$, but we never "
            "explicitly compute $p(x)$ for a given $x$. When training succeeds, sample quality is excellent. "
            "But training is notoriously unstable (mode collapse, oscillation)."
        )

    with st.expander("**Diffusion Models** — Gradual Denoising"):
        st.markdown(
            "**What it means:** Gradually add noise to data over many steps (forward process), then learn to reverse "
            "this process (backward denoising). Like watching ink dissolve in water, then playing the video in reverse."
        )
        st.markdown(
            "**Concrete example:** DDPM, Stable Diffusion. Start with pure noise $x_T \\sim \\mathcal{N}(0, I)$, "
            "iteratively denoise over $T$ steps (often 1000) to produce a realistic image $x_0$."
        )
        st.markdown(
            "**Key insight:** The forward noising process is **designed** (usually Gaussian noise with a fixed schedule). "
            "The reverse denoising $p_\\theta(x_{t-1} | x_t)$ is **learned** via a UNet. "
            "Trade-off: SOTA quality and excellent mode coverage, but inference is slow (many steps)."
        )

    with st.expander("**Flow Matching** — ODE Transport"):
        st.markdown(
            "**What it means:** Learn a vector field that transports samples from a simple distribution (Gaussian) "
            "to the data distribution along continuous paths. Like a wind field moving particles from one configuration "
            "to another."
        )
        st.markdown(
            "**Concrete example:** Stable Diffusion 3. Define a time-dependent vector field $v_\\theta(x, t)$, "
            "solve the ODE $\\frac{dx}{dt} = v_\\theta(x, t)$ to transport $x_0 \\sim \\mathcal{N}(0, I)$ to $x_1 \\sim p_{data}$."
        )
        st.markdown(
            "**Key insight:** Deterministic ODE transport (not stochastic SDE like diffusion). Fewer sampling steps "
            "needed, easier training (continuous-time formulation avoids discretization artifacts). "
            "SOTA quality with better speed than diffusion."
        )

    st.markdown("")
    st.markdown("### Comparison: Quality vs Speed vs Stability")

    comparison_table = """
| Approach | Sample Quality | Training Stability | Inference Speed | Mode Coverage |
|----------|---------------|-------------------|----------------|--------------|
| Autoregressive | Excellent | Excellent | Slow (sequential) | Excellent |
| Normalizing Flows | Good | Excellent | Very Fast | Good |
| VAE | Fair (blurry) | Excellent | Very Fast | Fair |
| GAN | Excellent* | Poor | Very Fast | Poor (mode collapse) |
| Diffusion | SOTA | Excellent | Slow (many steps) | Excellent |
| Flow Matching | SOTA | Excellent | Moderate (fewer steps) | Excellent |
"""

    st.markdown(comparison_table)
    st.markdown('<p class="caption">*GANs produce excellent quality when training succeeds, but instability is common.</p>', unsafe_allow_html=True)
