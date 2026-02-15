"""Section: Variational Autoencoder (VAE)

Covers the core problem, architecture, ELBO, reparameterization trick,
and industry applications of VAEs.
"""

import streamlit as st

from webapp.page_utils import render_mermaid


def render():
    """Render the Variational Autoencoder (VAE) section."""
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

    with st.expander("Why maximize log-likelihood?"):
        st.markdown("#### The Learning Objective")
        st.markdown(
            "We're building a generative model $p_\\theta(x)$ parameterized by neural network weights "
            "$\\theta$. Our goal: make $p_\\theta(x)$ as close as possible to the true data distribution "
            "$p_\\text{data}(x)$."
        )

        st.markdown("#### Measuring Closeness with KL Divergence")
        st.markdown(
            "The natural way to measure 'closeness' between two distributions is the KL divergence:"
        )
        st.latex(
            r"D_{\mathrm{KL}}(p_{\text{data}} \| p_\theta) = "
            r"\mathbb{E}_{x \sim p_{\text{data}}}[\log p_{\text{data}}(x)] - "
            r"\mathbb{E}_{x \sim p_{\text{data}}}[\log p_\theta(x)]"
        )
        st.markdown(
            "The first term is the entropy of the data distribution — it's a constant we can't control. "
            "So minimizing KL divergence is equivalent to **maximizing** the second term:"
        )
        st.latex(r"\mathbb{E}_{x \sim p_{\text{data}}}[\log p_\theta(x)]")
        st.markdown("In practice, we approximate this expectation with our training dataset:")
        st.latex(r"\frac{1}{N} \sum_{i=1}^N \log p_\theta(x_i)")
        st.markdown(
            "This is **maximum likelihood estimation (MLE)** — find $\\theta$ that maximizes the "
            "probability the model assigns to the observed data."
        )

        st.markdown("#### Intuition: How Surprised Is the Model?")
        st.markdown(
            "Think of $\\log p_\\theta(x)$ as a measure of how unsurprised the model is by data point $x$:\n\n"
            "- **High log-probability** (e.g., $-2$): Model thinks $x$ is very likely → Good fit\n"
            "- **Low log-probability** (e.g., $-1000$): Model thinks $x$ is extremely unlikely → Poor fit\n\n"
            "Training pushes the model to be unsurprised by the training data."
        )

        st.markdown("#### Why the Logarithm?")
        st.markdown("We use log-probability instead of raw probability for three critical reasons:")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**1. Products → Sums**")
            st.markdown(
                "For independent data points:\n\n"
                "$p(x_1, x_2) = p(x_1) \\cdot p(x_2)$\n\n"
                "$\\log p(x_1, x_2) = \\log p(x_1) + \\log p(x_2)$\n\n"
                "Sums are easier to optimize."
            )

        with col2:
            st.markdown("**2. Numerical Stability**")
            st.markdown(
                "Probabilities can be incredibly small (e.g., $10^{-300}$). Computers can't represent "
                "these accurately. Log space keeps numbers in a manageable range."
            )

        with col3:
            st.markdown("**3. Information Theory**")
            st.markdown(
                "$-\\log_2 p(x)$ = number of bits needed to encode $x$ under an optimal code. "
                "Maximizing log-likelihood = minimizing coding length."
            )

        st.info(
            "**Key Takeaway:** Maximizing log-likelihood is the principled way to fit a generative model. "
            "It's equivalent to minimizing KL divergence from data to model, which measures distributional "
            "closeness."
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

    with st.expander("Where does q(z) come from? The ELBO derivation"):
        st.markdown("#### The Intractability Problem")
        st.markdown(
            "We want to maximize $\\log p_\\theta(x)$. Let's write it using the latent variable $z$:"
        )
        st.latex(r"\log p_\theta(x) = \log \int p_\theta(x|z) p(z) \, dz")
        st.markdown(
            "This integral is **intractable** — it sums over all possible latent codes $z$, weighted by "
            "how likely each $z$ is to generate $x$. For high-dimensional $z$, we can't compute this."
        )

        st.markdown("#### The Mathematical Trick: Introduce q(z)")
        st.markdown(
            "Here's the key insight: multiply and divide by ANY distribution $q(z)$ inside the integral. "
            "This is valid for any choice of $q(z)$ — it's pure algebra, not an approximation:"
        )
        st.latex(
            r"\log p_\theta(x) = \log \int \frac{q(z)}{q(z)} p_\theta(x|z) p(z) \, dz = "
            r"\log \mathbb{E}_{z \sim q(z)} \left[ \frac{p_\theta(x|z) p(z)}{q(z)} \right]"
        )

        st.markdown("#### Jensen's Inequality: From Equality to Lower Bound")
        st.markdown(
            "The logarithm is a concave function, so by Jensen's inequality, we can pull the log inside "
            "the expectation at the cost of introducing an inequality:"
        )
        st.latex(
            r"\log \mathbb{E}_{z \sim q(z)} \left[ \frac{p_\theta(x|z) p(z)}{q(z)} \right] \geq "
            r"\mathbb{E}_{z \sim q(z)} \left[ \log \frac{p_\theta(x|z) p(z)}{q(z)} \right]"
        )
        st.markdown("Rearranging the right-hand side gives the **ELBO**:")
        st.latex(
            r"\text{ELBO}(q) = \mathbb{E}_{q(z)} [\log p_\theta(x|z)] - D_{\mathrm{KL}}(q(z) \| p(z))"
        )

        st.markdown("#### What Is q(z)?")
        st.markdown(
            "$q(z)$ is called a **proposal distribution** or **variational distribution**. It's a "
            "mathematical tool we introduced to make the problem tractable:\n\n"
            "- In the abstract derivation, $q(z)$ can be ANY distribution over $z$\n"
            "- In the **VAE**, we make $q$ depend on the data: $q_\\phi(z|x)$ — a neural network encoder\n"
            "- In **Expectation-Maximization (EM)**, $q$ takes an analytical form (e.g., Gaussian mixture "
            "responsibilities)\n\n"
            "The choice of $q$ determines how tight the lower bound is. The best $q$ makes ELBO = $\\log p(x)$."
        )

        st.markdown("#### The Complete Picture")
        st.markdown("Expanding the derivation fully, we get:")
        st.latex(
            r"\log p_\theta(x) = \text{ELBO}(q) + D_{\mathrm{KL}}(q(z) \| p_\theta(z|x))"
        )
        st.markdown(
            "Since $D_\\text{KL} \\geq 0$, the ELBO is always a lower bound. The gap between ELBO and "
            "$\\log p_\\theta(x)$ is exactly the KL divergence from $q(z)$ to the true posterior "
            "$p_\\theta(z|x)$."
        )

        st.success(
            "**Key Insight:** We converted an intractable integral into a tractable optimization problem. "
            "Instead of computing $\\log p_\\theta(x)$ directly, we maximize a lower bound (ELBO) that "
            "we CAN compute."
        )

    with st.expander("What does 'tractable' mean in the ELBO decomposition?"):
        st.markdown("#### Tractable vs. Intractable")
        st.markdown(
            "- **Tractable**: Can compute exactly (closed form) or efficiently approximate (Monte Carlo)\n"
            "- **Intractable**: Cannot compute in reasonable time, even approximately\n\n"
            "The ELBO decomposition separates terms we can compute from terms we can't:"
        )
        st.latex(
            r"\log p_\theta(x) = \underbrace{\text{ELBO}(q_\phi)}_{\text{TRACTABLE}} + "
            r"\underbrace{D_{\mathrm{KL}}(q_\phi(z|x) \| p_\theta(z|x))}_{\text{INTRACTABLE}}"
        )

        st.markdown("#### The ELBO: Two Tractable Terms")
        st.markdown("The ELBO itself has two components:")
        st.latex(
            r"\text{ELBO} = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{Reconstruction}} "
            r"- \underbrace{D_{\mathrm{KL}}(q_\phi(z|x) \| p(z))}_{\text{KL to Prior}}"
        )

        st.markdown("##### 1. Reconstruction Term (Monte Carlo)")
        st.markdown(
            "We approximate the expectation by sampling $z \\sim q_\\phi(z|x)$ and evaluating "
            "$\\log p_\\theta(x|z)$:"
        )
        st.code(
            """# PyTorch pseudocode
mu, log_var = encoder(x)
z = reparameterize(mu, log_var)  # z ~ q(z|x)
x_recon = decoder(z)
recon_loss = -log_prob(x, x_recon)  # Negative log-likelihood""",
            language="python",
        )
        st.markdown(
            "With one sample, this is an unbiased estimator. Averaging over batches and iterations gives "
            "a good approximation."
        )

        st.markdown("##### 2. KL to Prior (Closed Form)")
        st.markdown(
            "For Gaussian $q_\\phi(z|x) = \\mathcal{N}(\\mu(x), \\text{diag}(\\sigma^2(x)))$ and prior "
            "$p(z) = \\mathcal{N}(0, I)$, the KL divergence has a closed-form formula:"
        )
        st.latex(
            r"D_{\mathrm{KL}}(q_\phi(z|x) \| p(z)) = "
            r"\frac{1}{2} \sum_{j=1}^{d} \left( \sigma_j^2 + \mu_j^2 - 1 - \log \sigma_j^2 \right)"
        )
        st.code(
            """# PyTorch implementation
kl_div = 0.5 * torch.sum(sigma**2 + mu**2 - 1 - torch.log(sigma**2))""",
            language="python",
        )
        st.markdown("No sampling needed — direct computation from encoder outputs.")

        st.markdown("#### The Intractable Term: KL to True Posterior")
        st.markdown(
            "The gap between ELBO and $\\log p_\\theta(x)$ is $D_\\text{KL}(q_\\phi(z|x) \\| p_\\theta(z|x))$. "
            "This is **intractable** because:"
        )
        st.latex(r"p_\theta(z|x) = \frac{p_\theta(x|z) p(z)}{p_\theta(x)}")
        st.markdown(
            "Computing $p_\\theta(z|x)$ requires $p_\\theta(x) = \\int p_\\theta(x|z) p(z) \\, dz$ — "
            "the very integral we couldn't compute in the first place! It's a circular dependency."
        )

        st.warning(
            "**Why we can't compute the true posterior:** The denominator $p_\\theta(x)$ is the intractable "
            "marginal likelihood. If we could compute it, we wouldn't need the ELBO."
        )

        st.markdown("#### Summary Table")
        summary_data = {
            "Term": [
                "Reconstruction",
                "KL to Prior",
                "KL to True Posterior",
            ],
            "Formula": [
                "$\\mathbb{E}_{q(z|x)}[\\log p(x|z)]$",
                "$D_\\text{KL}(q(z|x) \\| p(z))$",
                "$D_\\text{KL}(q(z|x) \\| p(z|x))$",
            ],
            "Tractability": [
                "Tractable (Monte Carlo)",
                "Tractable (Closed form for Gaussians)",
                "Intractable (requires $p(x)$)",
            ],
            "Used in Training?": [
                "Yes — maximize",
                "Yes — minimize",
                "No — we ignore it",
            ],
        }
        st.table(summary_data)

        st.success(
            "**The ELBO is a lower bound because we ignore a positive KL term.** By maximizing the ELBO, "
            "we maximize a tractable approximation to $\\log p_\\theta(x)$."
        )

    st.markdown("### Understanding the Latent Space Distributions")
    st.markdown(
        "The VAE involves three key distributions. Understanding the difference between what's "
        "chosen vs. what's learned is critical:"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**$p(z)$ — The Prior**")
        st.markdown("*(CHOSEN)*")
        st.markdown(
            "Fixed as $\\mathcal{N}(0, I)$ before training. Never updated during training. "
            "Think of it like a filing cabinet — the organization system you set up in advance."
        )

    with col2:
        st.markdown("**$q(z|x)$ — The Encoder**")
        st.markdown("*(LEARNED)*")
        st.markdown(
            "Neural network mapping each input $x$ to a distribution over $z$. Parameterized by "
            "$\\mu(x)$ and $\\sigma(x)$. The 'filing clerk' that learns where to file each document."
        )

    with col3:
        st.markdown("**$p(x|z)$ — The Decoder**")
        st.markdown("*(LEARNED)*")
        st.markdown(
            "Maps latent code $z$ back to data space. The 'retrieval system' that reconstructs "
            "the original data from its compressed representation."
        )

    st.markdown("#### Why $p(z) = \\mathcal{N}(0, I)$?")
    st.markdown(
        "We choose the standard normal distribution for four practical reasons:\n\n"
        "1. **Closed-form KL divergence** — KL between two Gaussians has an analytic formula, "
        "no Monte Carlo approximation needed\n"
        "2. **Standardized coordinate system** — prevents the encoder from 'cheating' by spreading "
        "codes far apart\n"
        "3. **Trivial sampling** — at generation time, just sample from $\\mathcal{N}(0, I)$ and decode\n"
        "4. **Enforces regularity** — the KL term forces all $q(z|x)$ to cluster around the origin"
    )

    st.markdown("#### What Happens at Generation Time")
    st.markdown(
        "To generate a new image:\n\n"
        "1. Sample $z \\sim \\mathcal{N}(0, I)$ (using the fixed prior)\n"
        "2. Pass $z$ through the decoder to get $\\hat{x}$\n\n"
        "This works because the KL term forced all encoder distributions to 'pile up' around "
        "$\\mathcal{N}(0, I)$ during training. Without the KL regularization, the encoder could "
        "scatter codes anywhere in latent space, and random sampling would land in empty regions "
        "→ garbage output."
    )

    st.warning(
        "**Common misconception**: $p(z)$ is NOT learned. It's a fixed $\\mathcal{N}(0, I)$ that "
        "you choose before training. What IS learned is $q(z|x)$ — the encoder that maps your data "
        "INTO this fixed prior space."
    )

    with st.expander("Understanding the notation: q_φ(z|x) vs q_φ(z)"):
        st.markdown("#### Two Different Distributions")
        st.markdown("The notation can be confusing. There are actually two distinct distributions:")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### $q_\\phi(z|x)$ — The Encoder")
            st.markdown(
                "- A **conditional distribution**: different for every input $x$\n"
                "- Parameterized by neural network weights $\\phi$\n"
                "- Maps $x \\to \\mathcal{N}(\\mu_\\phi(x), \\text{diag}(\\sigma^2_\\phi(x)))$\n"
                "- Used at **training time** (and for encoding new data)\n"
                "- **Not used at generation time**"
            )

        with col2:
            st.markdown("##### $q_\\phi(z)$ — The Aggregated Posterior")
            st.markdown(
                "- A **marginal distribution**: single distribution over entire latent space\n"
                "- Average over all data: $\\frac{1}{N} \\sum_i q_\\phi(z|x_i)$\n"
                "- It's a **mixture of Gaussians** (one per datapoint)\n"
                "- **Implicit** — never computed explicitly\n"
                "- Goal: $q_\\phi(z) \\approx p(z) = \\mathcal{N}(0, I)$"
            )

        st.markdown("#### Concrete Example")
        st.markdown(
            "Suppose you have 3 datapoints in your training set: $x_1$, $x_2$, $x_3$. "
            "The encoder maps each to a Gaussian:"
        )
        st.latex(
            r"\begin{align}"
            r"q_\phi(z|x_1) &= \mathcal{N}(\mu_1, \sigma_1^2) \\"
            r"q_\phi(z|x_2) &= \mathcal{N}(\mu_2, \sigma_2^2) \\"
            r"q_\phi(z|x_3) &= \mathcal{N}(\mu_3, \sigma_3^2)"
            r"\end{align}"
        )
        st.markdown("The aggregated posterior is:")
        st.latex(
            r"q_\phi(z) = \frac{1}{3} \left[ \mathcal{N}(\mu_1, \sigma_1^2) + "
            r"\mathcal{N}(\mu_2, \sigma_2^2) + \mathcal{N}(\mu_3, \sigma_3^2) \right]"
        )
        st.markdown(
            "This is a mixture of 3 Gaussians. For MNIST (60k images), it's a mixture of 60,000 Gaussians!"
        )

        st.markdown("#### How Does Training Push q_φ(z) → p(z)?")
        st.markdown(
            "The KL term in the ELBO is $D_\\text{KL}(q_\\phi(z|x) \\| p(z))$ — computed **per datapoint**. "
            "Minimizing this for ALL $x$ pushes every $q_\\phi(z|x)$ toward $\\mathcal{N}(0, I)$. "
            "Since the average of Gaussians centered near the origin is also centered near the origin, "
            "the aggregated $q_\\phi(z)$ ends up close to $p(z)$."
        )

        st.markdown("#### Code Example: Encoder Forward Pass")
        st.code(
            """# This is q_φ(z|x) — different Gaussian for each input x
class Encoder(nn.Module):
    def forward(self, x):
        h = self.network(x)
        mu = self.fc_mu(h)       # μ_φ(x)
        log_var = self.fc_logvar(h)  # log σ²_φ(x)
        return mu, log_var  # Parameters of q_φ(z|x) = N(μ, σ²)

# During training, for a batch of inputs:
mu, log_var = encoder(x_batch)  # Shape: (batch_size, latent_dim)
# Each row is a different Gaussian q_φ(z|x_i)""",
            language="python",
        )

        st.markdown("#### Connection to the ELBO Derivation")
        st.markdown(
            "In the abstract derivation (Q2), we introduced $q(z)$ as a proposal distribution. "
            "In the VAE:\n\n"
            "- $q(z)$ from the derivation → $q_\\phi(z|x)$ — the encoder\n"
            "- In EM algorithms → $q(z)$ stays as an analytical form\n\n"
            "The key difference: VAE uses a neural network to parameterize the proposal distribution "
            "**conditioned on the data**."
        )

        st.markdown("#### Summary Table")
        summary_data = {
            "Notation": ["$q_\\phi(z|x)$", "$q_\\phi(z)$"],
            "What It Is": [
                "Encoder distribution (conditional)",
                "Aggregated posterior (marginal)",
            ],
            "Form": [
                "$\\mathcal{N}(\\mu_\\phi(x), \\text{diag}(\\sigma^2_\\phi(x)))$",
                "$\\frac{1}{N} \\sum_i q_\\phi(z|x_i)$ (mixture)",
            ],
            "When Used?": [
                "Training and encoding",
                "Implicit (never computed)",
            ],
            "How Computed?": [
                "Encoder forward pass",
                "Theoretical construct only",
            ],
        }
        st.table(summary_data)

        st.info(
            "**Key Takeaway:** $q_\\phi(z|x)$ is the encoder network you implement. $q_\\phi(z)$ is "
            "the average latent distribution across your dataset — it's what you're regularizing toward "
            "$p(z) = \\mathcal{N}(0, I)$ via the KL term."
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

    with st.expander("How exactly is epsilon chosen?"):
        st.markdown("#### What is $\\varepsilon$?")
        st.markdown(
            "$\\varepsilon$ is a random vector sampled from $\\mathcal{N}(0, I)$ — the standard "
            "normal distribution. In PyTorch:\n\n"
            "```python\n"
            "eps = torch.randn_like(sigma)  # Fresh sample every forward pass\n"
            "```\n\n"
            "A new $\\varepsilon$ is sampled for every forward pass, every datapoint."
        )

        st.markdown("#### Why $\\mathcal{N}(0, I)$?")
        st.markdown(
            "The encoder outputs $\\mu$ and $\\sigma$ that SHIFT and SCALE this base noise. "
            "The formula $z = \\mu + \\sigma \\cdot \\varepsilon$ gives $z \\sim \\mathcal{N}(\\mu, \\sigma^2)$. "
            "The noise provides the 'base randomness' that the encoder parameters reshape."
        )

        st.markdown("#### Numerical Example")
        st.markdown(
            "Suppose the encoder outputs:\n"
            "- $\\mu = [2.0, -1.0]$\n"
            "- $\\sigma = [0.5, 0.3]$\n\n"
            "And we sample:\n"
            "- $\\varepsilon = [0.7, -0.4]$\n\n"
            "Then:\n\n"
            "$$z = [2.0, -1.0] + [0.5, 0.3] \\cdot [0.7, -0.4] = [2.35, -1.12]$$\n\n"
            "The gradients are:\n"
            "- $\\frac{\\partial z}{\\partial \\mu} = [1, 1]$ (clean gradient path)\n"
            "- $\\frac{\\partial z}{\\partial \\sigma} = \\varepsilon = [0.7, -0.4]$ (depends on random sample)\n\n"
            "These well-defined gradients flow back through the encoder via standard backpropagation."
        )

        st.markdown("#### Different Forward Passes")
        st.markdown(
            "Same input $x$ → different $\\varepsilon$ → different $z$ values each time. "
            "This Monte Carlo approximation of $\\mathbb{E}_{q(z|x)}[\\log p(x|z)]$ averages out "
            "over the course of training (many batches, many samples)."
        )

        st.markdown("#### At Generation Time")
        st.markdown(
            "No $\\varepsilon$ needed. Sample $z \\sim \\mathcal{N}(0, I)$ directly and decode. "
            "The encoder is not used during generation."
        )

        st.markdown("#### Summary Table")
        summary_data = {
            "Quantity": ["$p(z)$", "$q(z|x)$", "$\\varepsilon$", "$z$", "$\\mu, \\sigma$"],
            "What is it?": [
                "Prior distribution",
                "Encoder distribution",
                "Standard noise",
                "Latent code",
                "Encoder outputs",
            ],
            "Chosen or Learned?": [
                "Chosen ($\\mathcal{N}(0,I)$)",
                "Learned ($\\mu$, $\\sigma$ networks)",
                "Sampled from $\\mathcal{N}(0,I)$",
                "Computed: $\\mu + \\sigma \\cdot \\varepsilon$",
                "Learned (network weights)",
            ],
            "When?": [
                "Generation time",
                "Training (via $\\varepsilon$)",
                "Every forward pass",
                "Every training step",
                "Computed each forward pass",
            ],
        }

        st.table(summary_data)

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
