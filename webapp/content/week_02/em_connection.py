"""Section: EM, VAE, and K-means

Connects VAE to classical EM algorithm and K-means clustering, showing how they
optimize the same objective with different strategies.
"""

import streamlit as st

from webapp.page_utils import render_mermaid


def render():
    """Render the EM, VAE, and K-means connection section."""
    st.markdown("---")
    st.markdown("## EM, VAE, and K-means: The Unified Picture")

    st.markdown(
        "The VAE, Expectation-Maximization (EM) algorithm, and K-means clustering all optimize "
        "the **same objective** (the ELBO) but use different optimization strategies and "
        "different forms for the latent distribution $q(z)$. Understanding this connection "
        "reveals the broader family of latent variable models."
    )

    st.markdown("### Overall Loss: Expectation Over Data")
    st.markdown(
        "The ELBO we've seen so far is computed **per datapoint**. The overall training objective "
        "averages over the entire dataset:"
    )
    st.latex(
        r"\mathcal{L}_{\text{total}} = \frac{1}{N} \sum_{i=1}^N \text{ELBO}(x_i) = "
        r"\frac{1}{N} \sum_{i=1}^N \left[ \mathbb{E}_{q(z|x_i)}[\log p(x_i|z)] - "
        r"D_{\mathrm{KL}}(q(z|x_i) \| p(z)) \right]"
    )

    st.markdown(
        "In practice, we use **mini-batch stochastic gradient descent (SGD)** to approximate "
        "this expectation:\n\n"
        "1. **Sample a mini-batch** of $B$ datapoints from the full dataset\n"
        "2. **Compute ELBO** for each datapoint in the batch\n"
        "3. **Average** the ELBOs: $\\frac{1}{B} \\sum_{i=1}^B \\text{ELBO}(x_i)$\n"
        "4. **Backpropagate** and update $\\theta$ and $\\phi$\n\n"
        "This gives an **unbiased estimate** of the true gradient over the full dataset."
    )

    with st.expander("Two levels of sampling in VAE training"):
        st.markdown(
            "There are actually **two** sources of randomness in VAE training:\n\n"
            "**Level 1: Sample mini-batch of $x$**\n\n"
            "- Each training step uses a random subset of the dataset\n"
            "- Approximates the expectation $\\mathbb{E}_{x \\sim p_\\text{data}}[\\text{ELBO}(x)]$\n"
            "- Standard practice in deep learning\n\n"
            "**Level 2: Sample $z$ from encoder (reparameterization trick)**\n\n"
            "- For each $x$ in the batch, sample $z \\sim q_\\phi(z|x)$\n"
            "- Approximates $\\mathbb{E}_{z \\sim q(z|x)}[\\log p(x|z)]$\n"
            "- Specific to VAE (and other stochastic latent variable models)\n\n"
            "Both approximations average out over many training iterations."
        )

    st.markdown("### The EM Algorithm")
    st.markdown(
        "**Expectation-Maximization (EM)** is a classical algorithm for learning latent variable models. "
        "It optimizes the ELBO via **coordinate ascent** — alternating between two steps:"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### E-Step (Expectation)")
        st.markdown(
            "- **Fix $\\theta$** (model parameters)\n"
            "- **Optimize $q(z)$** to maximize ELBO\n"
            "- For many models (e.g., Gaussian mixtures), this has a **closed-form solution**\n"
            "- Example: In GMM, compute 'responsibilities' $\\gamma_{ik} = p(z=k|x_i)$"
        )

    with col2:
        st.markdown("##### M-Step (Maximization)")
        st.markdown(
            "- **Fix $q(z)$** (from E-step)\n"
            "- **Optimize $\\theta$** to maximize ELBO\n"
            "- Often has a **closed-form solution**\n"
            "- Example: In GMM, update cluster centers to weighted mean of assigned points"
        )

    st.markdown("#### Gaussian Mixture Model (GMM) Example")
    st.markdown(
        "A GMM models data as a mixture of $K$ Gaussian clusters. The latent variable $z \\in \\{1, \\dots, K\\}$ "
        "indicates which cluster generated each datapoint."
    )

    st.code(
        """# EM for GMM (simplified pseudocode)
for iteration in range(max_iters):
    # E-step: Compute responsibilities (soft assignments)
    for i in range(N):
        for k in range(K):
            gamma[i, k] = prob(x[i] | cluster_k) * prior[k]
        gamma[i] /= gamma[i].sum()  # Normalize

    # M-step: Update cluster parameters
    for k in range(K):
        cluster_mean[k] = (gamma[:, k] @ x) / gamma[:, k].sum()
        cluster_cov[k] = weighted_covariance(x, gamma[:, k])
        prior[k] = gamma[:, k].mean()""",
        language="python",
    )

    st.markdown("### VAE vs EM: Same Objective, Different Optimization")

    comparison_data = {
        "": ["EM", "VAE"],
        "Objective": [
            "Maximize ELBO",
            "Maximize ELBO",
        ],
        "Optimization": [
            "Coordinate ascent (alternating E/M steps)",
            "Simultaneous SGD on θ and φ",
        ],
        "q(z) Form": [
            "Analytical (e.g., categorical for GMM)",
            "Neural network $q_\\phi(z|x)$",
        ],
        "Updates": [
            "Often closed-form",
            "Gradient-based (backprop)",
        ],
        "Scalability": [
            "Limited to tractable posteriors",
            "Scales to complex, high-dim data",
        ],
    }
    st.table(comparison_data)

    st.info(
        "**Key Difference:** EM alternates between optimizing $q$ and $\\theta$ (one at a time). "
        "VAE optimizes both **simultaneously** using gradient descent. VAE trades analytical "
        "tractability for expressive power — the encoder can be arbitrarily complex."
    )

    st.markdown("### K-means: The Simplest Special Case")
    st.markdown(
        "K-means clustering can be viewed as a **hard** (discrete) version of both EM and the autoencoder."
    )

    st.markdown("#### K-means as EM with Hard Assignments")
    st.markdown(
        "EM for GMM uses 'soft' responsibilities $\\gamma_{ik} \\in [0,1]$ — probabilistic assignments. "
        "K-means uses **hard assignments**: $\\gamma_{ik} \\in \\{0, 1\\}$ (each point belongs to exactly one cluster)."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**E-step → Assignment**")
        st.markdown("Assign each point to the **nearest** cluster center:")
        st.latex(r"z_i = \arg\min_k \|x_i - \mu_k\|^2")

    with col2:
        st.markdown("**M-step → Update Centers**")
        st.markdown("Move each center to the **mean** of assigned points:")
        st.latex(r"\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} x_i")

    st.markdown("#### K-means as an Autoencoder")
    st.markdown(
        "K-means is also a special case of an autoencoder with discrete latent codes:"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Encoder**")
        st.markdown(
            "Map $x$ to one-hot vector (nearest center):\n\n"
            "$z = \\text{one-hot}(\\arg\\min_k \\|x - \\mu_k\\|^2)$"
        )

    with col2:
        st.markdown("**Decoder**")
        st.markdown(
            "Map one-hot $z$ to cluster center:\n\n"
            "$\\hat{x} = \\sum_k z_k \\mu_k = \\mu_{z_i}$"
        )

    with col3:
        st.markdown("**Loss**")
        st.markdown(
            "Reconstruction error:\n\n"
            "$\\|x - \\hat{x}\\|^2 = \\|x - \\mu_{z_i}\\|^2$"
        )

    st.markdown("#### Limitations of K-means")
    st.markdown(
        "- **Discrete latent space**: Only $K$ possible codes (Voronoi cells)\n"
        "- **Linear decoder**: Just looks up cluster centers\n"
        "- **Poor generative quality**: Can only generate $K$ distinct outputs\n"
        "- **No probabilistic interpretation**: No way to model uncertainty\n\n"
        "K-means is useful for clustering, but not for high-quality generation."
    )

    st.markdown("### The Progression: From K-means to VQ-VAE")

    render_mermaid(
        """
graph LR
    KM["K-means<br/><i>Hard assignment</i><br/><i>Fixed centers</i>"]
    EM["EM / GMM<br/><i>Soft assignment</i><br/><i>Analytical q(z)</i>"]
    VAE["VAE<br/><i>Continuous latent</i><br/><i>Neural q(z|x)</i>"]
    VQ["VQ-VAE<br/><i>Discrete latent</i><br/><i>Neural encoder/decoder</i>"]
    KM --> EM --> VAE --> VQ
""",
        height=300,
    )

    st.markdown(
        "- **K-means → EM**: Soft assignments, probabilistic interpretation\n"
        "- **EM → VAE**: Neural encoder, continuous latents, scales to high-dim data\n"
        "- **VAE → VQ-VAE**: Back to discrete latents, but with neural encoder/decoder\n\n"
        "Each step adds expressiveness while maintaining the same underlying objective."
    )

    st.markdown("### Comparison Table: The Full Picture")

    full_comparison = {
        "": ["K-means", "EM (GMM)", "VAE", "VQ-VAE"],
        "Latent Type": [
            "Discrete (categorical)",
            "Discrete or continuous",
            "Continuous (Gaussian)",
            "Discrete (codebook)",
        ],
        "q Distribution": [
            "Hard (one-hot)",
            "Analytical (soft)",
            "Neural $q_\\phi(z|x)$",
            "Neural (vector quantize)",
        ],
        "Optimization": [
            "Alternating (coordinate descent)",
            "Alternating (coordinate ascent)",
            "Simultaneous SGD",
            "Simultaneous SGD",
        ],
        "Encoder": [
            "Nearest neighbor search",
            "Closed-form posterior",
            "Neural network",
            "Neural network + VQ",
        ],
        "Decoder": [
            "Lookup table (centers)",
            "Gaussian mixture",
            "Neural network",
            "Neural network",
        ],
        "Expressiveness": [
            "Very limited",
            "Limited to tractable models",
            "Very expressive",
            "Very expressive",
        ],
        "Generative Quality": [
            "Poor (only K outputs)",
            "Good for classical models",
            "State-of-the-art",
            "State-of-the-art",
        ],
    }
    st.table(full_comparison)

    st.success(
        "**The Big Picture:** K-means, EM, VAE, and VQ-VAE all optimize the ELBO for latent variable "
        "models. They differ in:\n\n"
        "1. **Latent representation** (discrete vs continuous)\n"
        "2. **Form of $q(z)$** (hard, analytical, or neural)\n"
        "3. **Optimization strategy** (alternating vs simultaneous)\n\n"
        "Understanding these connections helps you choose the right model for your problem and adapt "
        "techniques across domains."
    )
