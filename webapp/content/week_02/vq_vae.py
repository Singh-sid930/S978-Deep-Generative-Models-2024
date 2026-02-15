"""Section: Vector Quantized VAE (VQ-VAE)

Introduces VQ-VAE, which uses discrete latent codes instead of continuous Gaussians,
and its role as a learned tokenizer for images.
"""

import streamlit as st

from webapp.page_utils import render_mermaid


def render():
    """Render the Vector Quantized VAE section."""
    st.markdown("---")
    st.markdown("## Vector Quantized VAE (VQ-VAE)")

    st.markdown(
        "The standard VAE uses **continuous** latent variables $z \\sim \\mathcal{N}(\\mu, \\sigma^2)$. "
        "But what if we want **discrete** latent codes, like words in a vocabulary? "
        "**VQ-VAE** combines the expressiveness of neural encoders/decoders with the discrete structure "
        "of K-means clustering."
    )

    st.markdown("### Discrete vs Continuous Latents")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Continuous (VAE)")
        st.markdown(
            "- $z \\in \\mathbb{R}^d$ — infinite possible codes\n"
            "- Smooth latent space, good for interpolation\n"
            "- Direct correspondence with pixels\n"
            "- Hard to model with autoregressive models (need discretization)"
        )

    with col2:
        st.markdown("##### Discrete (VQ-VAE)")
        st.markdown(
            "- $z \\in \\{e_1, e_2, \\dots, e_K\\}$ — finite codebook\n"
            "- Natural for sequential modeling (like language)\n"
            "- Each code is a learned 'visual word'\n"
            "- Can be modeled by transformers, GPT, etc."
        )

    st.markdown("### Why Discrete Latents?")
    st.markdown(
        "**Motivation from language:** In NLP, we represent sentences as sequences of discrete tokens. "
        "VQ-VAE applies the same idea to images — represent an image as a sequence of discrete 'visual tokens'. "
        "This enables:\n\n"
        "- **Autoregressive modeling** of images with transformers (DALL-E, Parti)\n"
        "- **Categorical priors** instead of Gaussian (easier to model complex distributions)\n"
        "- **Interpretable codebooks** — each code represents a learned visual concept\n"
        "- **Compression** — map high-res images to compact token sequences"
    )

    st.markdown("### The VQ-VAE Architecture")

    render_mermaid(
        """
graph LR
    X["Input x"] --> ENC["Encoder<br/>(neural network)"]
    ENC --> E["Continuous<br/>embedding<br/>e(x)"]
    E --> VQ["Vector<br/>Quantize"]
    CB["Codebook<br/>{e₁, e₂, ..., eₖ}"] -.lookup.-> VQ
    VQ --> ZQ["Discrete<br/>latent zq = eₖ"]
    ZQ --> DEC["Decoder<br/>(neural network)"]
    DEC --> XR["Reconstruction<br/>x̂"]
""",
        height=350,
    )

    st.markdown("#### The Forward Pass")
    st.markdown("1. **Encoder**: Neural network maps input $x$ to continuous embedding $e(x) \\in \\mathbb{R}^d$")
    st.markdown(
        "2. **Vector Quantization**: Find the nearest codebook vector:\n\n"
        "   $$z_q = e_k \\quad \\text{where} \\quad k = \\arg\\min_j \\|e(x) - e_j\\|^2$$\n\n"
        "   This is like K-means assignment — snap to the closest code."
    )
    st.markdown("3. **Decoder**: Neural network maps $z_q$ back to pixel space → reconstruction $\\hat{x}$")

    st.markdown("#### The Codebook")
    st.markdown(
        "The **codebook** $\\{e_1, e_2, \\dots, e_K\\}$ is a **learned** set of $K$ vectors in $\\mathbb{R}^d$. "
        "Think of it as:\n\n"
        "- A **vocabulary** of visual concepts (like word embeddings in NLP)\n"
        "- **Cluster centers** that are jointly learned with the encoder/decoder\n"
        "- A **lookup table** — the encoder outputs an index $k$, the decoder gets vector $e_k$\n\n"
        "Typical sizes: $K = 512$ to $8192$ codes, $d = 64$ to $512$ dimensions."
    )

    with st.expander("How is the codebook initialized and updated?"):
        st.markdown("#### Initialization")
        st.markdown(
            "The codebook vectors $\\{e_1, \\dots, e_K\\}$ are initialized **randomly** (or with K-means++ "
            "on a batch of encoder outputs). They are treated as **learnable parameters** — part of the "
            "model that gets updated during training."
        )

        st.markdown("#### Updating the Codebook")
        st.markdown(
            "The codebook is updated via the **codebook loss** (see loss section below). Two strategies:\n\n"
            "**1. Gradient-based (VQ-VAE original paper):**\n\n"
            "- Minimize $\\|\\text{sg}[e(x)] - e_k\\|^2$ where sg[] = stop gradient\n"
            "- Move codebook vectors toward encoder outputs (like K-means M-step)\n\n"
            "**2. Exponential moving average (EMA):**\n\n"
            "- Track running average of encoder outputs assigned to each code\n"
            "- Update: $e_k \\leftarrow \\text{EMA}(\\{e(x_i) : z_i = e_k\\})$\n"
            "- More stable, used in many modern implementations"
        )

        st.markdown("#### Dead Codes")
        st.markdown(
            "Some codebook entries may never be used (no encoder outputs map to them). Strategies:\n\n"
            "- **Reset**: Re-initialize unused codes to random encoder outputs\n"
            "- **Ignore**: Let them die (codebook becomes smaller in practice)\n"
            "- **Entropy regularization**: Encourage uniform use of all codes"
        )

    st.markdown("### The Straight-Through Trick")
    st.markdown(
        "**The Problem:** The vector quantization step $z_q = e_{\\arg\\min_k \\|e(x) - e_k\\|}$ uses $\\arg\\min$, "
        "which is **not differentiable**. Gradients can't flow through the discrete selection."
    )

    st.markdown(
        "**The Solution:** Use different paths for forward and backward passes:\n\n"
        "- **Forward pass**: Use the quantized code $z_q = e_k$ (discrete)\n"
        "- **Backward pass**: Copy gradients directly from decoder to encoder, bypassing the quantization\n\n"
        "In code, this is implemented as:"
    )

    st.code(
        """# Straight-through estimator
z_q = e(x) + (z_q - e(x)).detach()

# What this does:
# - Forward: z_q is the quantized code
# - Backward: gradient flows as if z_q = e(x) (identity)""",
        language="python",
    )

    st.markdown(
        "This is called the **straight-through estimator** — pretend the discrete operation is the identity "
        "for gradient purposes. It's a biased gradient estimator, but works well in practice."
    )

    with st.expander("Why does the straight-through trick work?"):
        st.markdown(
            "The straight-through estimator is **not theoretically justified** — it's a heuristic. "
            "But it works because:\n\n"
            "**1. Small quantization error:** If $\\|e(x) - z_q\\|$ is small, then the gradient approximation "
            "$\\frac{\\partial z_q}{\\partial e(x)} \\approx I$ is reasonable.\n\n"
            "**2. Encoder adapts:** The encoder learns to output embeddings close to codebook vectors, "
            "minimizing quantization error.\n\n"
            "**3. Codebook tracks encoder:** The codebook loss pulls codebook vectors toward encoder outputs.\n\n"
            "These two mechanisms (encoder → codebook, codebook → encoder) keep the gap small, making the "
            "straight-through approximation acceptable."
        )

    st.markdown("### VQ-VAE Loss Function")
    st.markdown("The VQ-VAE loss has **three** terms:")

    st.latex(
        r"\mathcal{L}_{\text{VQ-VAE}} = "
        r"\underbrace{\|x - \text{decoder}(z_q)\|^2}_{\text{Reconstruction}} + "
        r"\underbrace{\|\text{sg}[e(x)] - z_q\|^2}_{\text{Codebook}} + "
        r"\underbrace{\beta \|e(x) - \text{sg}[z_q]\|^2}_{\text{Commitment}}"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 1. Reconstruction Loss")
        st.markdown(
            "How well does the decoder reconstruct $x$ from $z_q$?\n\n"
            "Same as VAE reconstruction, but with discrete $z_q$."
        )

    with col2:
        st.markdown("##### 2. Codebook Loss")
        st.markdown(
            "Move codebook entries toward encoder outputs:\n\n"
            "$\\|\\text{sg}[e(x)] - z_q\\|^2$\n\n"
            "sg[] stops gradient to encoder — only updates codebook."
        )

    with col3:
        st.markdown("##### 3. Commitment Loss")
        st.markdown(
            "Encourage encoder to commit to codebook vectors:\n\n"
            "$\\beta \\|e(x) - \\text{sg}[z_q]\\|^2$\n\n"
            "Prevents encoder from drifting away. $\\beta \\approx 0.25$ typical."
        )

    st.code(
        """# PyTorch implementation
def vq_vae_loss(x, x_recon, e_x, z_q, beta=0.25):
    # 1. Reconstruction loss
    recon_loss = F.mse_loss(x_recon, x)

    # 2. Codebook loss (sg on encoder output)
    codebook_loss = F.mse_loss(e_x.detach(), z_q)

    # 3. Commitment loss (sg on quantized code)
    commitment_loss = beta * F.mse_loss(e_x, z_q.detach())

    return recon_loss + codebook_loss + commitment_loss""",
        language="python",
    )

    st.markdown("### VQ-VAE as a Learned Tokenizer")
    st.markdown(
        "The real power of VQ-VAE is that it produces **spatial latent maps**, not single vectors. "
        "For an image, the encoder outputs a grid of embeddings:"
    )

    st.markdown(
        "$$e(x) \\in \\mathbb{R}^{H' \\times W' \\times d} \\quad \\longrightarrow \\quad "
        "z_q \\in \\{e_1, \\dots, e_K\\}^{H' \\times W'}$$"
    )

    st.markdown(
        "Each spatial position gets its own discrete code. For example:\n\n"
        "- Input: $256 \\times 256$ image\n"
        "- Encoder downsamples by 16× → $16 \\times 16$ latent grid\n"
        "- Each of 256 positions assigned a code from $K=512$ vocabulary\n"
        "- Result: $16 \\times 16$ grid of discrete tokens\n\n"
        "This grid of tokens can now be modeled by **autoregressive transformers** (like GPT)."
    )

    st.markdown("#### Industry Use Case: DALL-E, Parti, etc.")
    st.info(
        "**Modern text-to-image models** use VQ-VAE (or similar) as a **two-stage pipeline:**\n\n"
        "**Stage 1 (VQ-VAE):** Train an image tokenizer\n\n"
        "- Encoder: image → discrete token grid\n"
        "- Decoder: token grid → image\n\n"
        "**Stage 2 (Transformer):** Train an autoregressive model\n\n"
        "- Input: text prompt\n"
        "- Output: sequence of image tokens\n"
        "- Model: GPT-like transformer predicting next token\n\n"
        "**Generation:** Text → Transformer → tokens → VQ-VAE decoder → image\n\n"
        "This is how **DALL-E (2021)** and **Parti (2022)** work. The VQ-VAE compresses images into "
        "a discrete space that transformers can model effectively."
    )

    st.markdown("### Comparison: VAE vs VQ-VAE")

    comparison_data = {
        "": ["VAE", "VQ-VAE"],
        "Latent Type": [
            "Continuous $z \\sim \\mathcal{N}(\\mu, \\sigma^2)$",
            "Discrete $z_q \\in \\{e_1, \\dots, e_K\\}$",
        ],
        "Prior": [
            "$p(z) = \\mathcal{N}(0, I)$ (Gaussian)",
            "Categorical or learned prior",
        ],
        "Encoder Output": [
            "$\\mu(x), \\sigma(x)$ (continuous)",
            "Index $k$ or one-hot (discrete)",
        ],
        "Reparameterization": [
            "$z = \\mu + \\sigma \\cdot \\varepsilon$",
            "Straight-through estimator",
        ],
        "Loss Terms": [
            "Reconstruction + KL divergence",
            "Reconstruction + codebook + commitment",
        ],
        "Generation": [
            "Sample $z \\sim \\mathcal{N}(0,I)$, decode",
            "Sample token sequence from prior, decode",
        ],
        "Autoregressive Modeling": [
            "Hard (need to discretize)",
            "Natural (already discrete)",
        ],
        "Use Cases": [
            "Anomaly detection, latent interpolation",
            "Tokenization for transformers, compression",
        ],
    }
    st.table(comparison_data)

    st.markdown("### Implementation Status")
    st.info(
        "**VQ-VAE is covered conceptually here.** Implementation stubs and exercises for VQ-VAE "
        "will be added in a future update. The core VAE implementation (this week's assignment) "
        "provides the foundation — VQ-VAE replaces the reparameterization trick with vector quantization "
        "and changes the loss terms."
    )

    st.success(
        "**Key Takeaway:** VQ-VAE bridges continuous and discrete latent variable models. It combines "
        "the expressiveness of neural encoders/decoders with the discrete structure of clustering, "
        "enabling autoregressive modeling of images with transformers. This is the foundation of modern "
        "text-to-image generation (DALL-E, Parti) and video generation (VideoGPT)."
    )
