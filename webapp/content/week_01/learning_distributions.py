"""Section: Learning to Represent Probability Distributions

Understanding the curse of dimensionality and the designed vs learned paradigm.
"""

import streamlit as st


def render():
    """Render the Learning to Represent Probability Distributions section."""
    st.markdown("---")
    st.markdown("## Learning to Represent Probability Distributions")

    st.markdown(
        "High-dimensional data creates a fundamental challenge. How do we represent and learn "
        "probability distributions over images, audio, or text without enumerating every possible configuration?"
    )

    st.markdown("### The Curse of Dimensionality")

    st.markdown(
        "A 256×256 RGB image has $256 \\times 256 \\times 3 = 196{,}608$ dimensions. "
        "A naive tabular representation would require $256^{196{,}608}$ entries — approximately $10^{472{,}000}$ numbers. "
        "For reference, the observable universe contains roughly $10^{80}$ atoms."
    )

    st.markdown(
        "Worse, in high dimensions, almost all volume concentrates in the corners of the hypercube, "
        "and distances become uninformative (all points are roughly equidistant)."
    )

    st.markdown("")
    st.markdown(
        "**The solution:** Exploit structure. Natural images don't uniformly fill 196,608-dimensional space — "
        "they lie on a much lower-dimensional manifold. A generative model's job is to learn this manifold."
    )

    st.markdown("### Designed vs Learned: The Key Insight")

    st.markdown(
        "A profound observation from the course: **not all parts of distribution modeling are learned.** "
        "The most effective models combine careful **design** (human knowledge, inductive bias) with flexible "
        "**learning** (neural networks fit to data)."
    )

    st.markdown("")
    st.markdown("Each model family makes different design choices:")

    with st.expander("**Autoregressive Models**"):
        st.markdown("**Designed:**")
        st.markdown("- Variable ordering (e.g., raster-scan for images: top-left → bottom-right)")
        st.markdown("- Chain rule factorization: $p(x) = \\prod_{i=1}^{n} p(x_i | x_{<i})$")
        st.markdown("")
        st.markdown("**Learned:**")
        st.markdown("- Conditional distributions $p(x_i | x_{<i})$ via neural networks (e.g., transformers, PixelCNN)")
        st.markdown("")
        st.markdown(
            "**Why it matters:** The ordering imposes structure — we don't need to learn which variables "
            "to condition on. The network only learns *how* $x_i$ depends on previous variables."
        )

    with st.expander("**Diffusion Models**"):
        st.markdown("**Designed:**")
        st.markdown("- Forward noising process: $q(x_t | x_{t-1}) = \\mathcal{N}(\\sqrt{1-\\beta_t} x_{t-1}, \\beta_t I)$")
        st.markdown("- Noise schedule $\\{\\beta_t\\}_{t=1}^{T}$ (e.g., linear, cosine)")
        st.markdown("- Number of diffusion steps $T$ (e.g., 1000)")
        st.markdown("")
        st.markdown("**Learned:**")
        st.markdown("- Reverse denoising distribution $p_\\theta(x_{t-1} | x_t)$ via UNet")
        st.markdown("")
        st.markdown(
            "**Why it matters:** The forward process is simple and deterministic — no learning needed. "
            "The model only learns to invert it. This design provides strong inductive bias (smooth denoising) "
            "and theoretical guarantees (convergence to data distribution)."
        )

    with st.expander("**Variational Autoencoders**"):
        st.markdown("**Designed:**")
        st.markdown("- Latent dimensionality $d_z$ (much smaller than data dimension)")
        st.markdown("- Prior distribution: $p(z) = \\mathcal{N}(0, I)$")
        st.markdown("- Mean-field approximate posterior: $q_\\phi(z|x) = \\mathcal{N}(\\mu_\\phi(x), \\text{diag}(\\sigma_\\phi^2(x)))$")
        st.markdown("")
        st.markdown("**Learned:**")
        st.markdown("- Encoder network: $x \\to (\\mu_\\phi(x), \\sigma_\\phi(x))$")
        st.markdown("- Decoder network: $z \\to p_\\theta(x|z)$")
        st.markdown("")
        st.markdown(
            "**Why it matters:** The latent structure (Gaussian prior, factorized posterior) is designed. "
            "The networks learn the nonlinear mapping between data space and latent space. "
            "This compression is what makes VAEs sample efficiently."
        )

    st.markdown("")
    st.markdown("### Design vs Learning: A Summary")

    design_learned_table = """
| Approach | Designed | Learned |
|----------|----------|---------|
| **Autoregressive** | Variable ordering (e.g., raster-scan) | Conditional distributions $p(x_i | x_{<i})$ |
| **Diffusion** | Forward noising, noise schedule, $T$ steps | Reverse denoising $p_\\theta(x_{t-1} | x_t)$ |
| **VAE** | Latent dim $d_z$, prior $p(z)$, factorized posterior | Encoder $q_\\phi(z|x)$, decoder $p_\\theta(x|z)$ |
"""

    st.markdown(design_learned_table)

    st.success(
        "**Key Takeaway:** Pure end-to-end learning in high dimensions is inefficient. "
        "The winning strategy is smart **design** (encode domain knowledge as inductive bias) + "
        "flexible **learning** (neural nets fit remaining structure from data). "
        "Understanding what to design vs what to learn is the art of building generative models."
    )
