"""Section: Understanding Image Priors

Covers the evolution of image priors from patch statistics to learned generative models,
and the Deep Image Prior insight.
"""

import streamlit as st

from webapp.page_utils import render_mermaid


def render():
    """Render the Understanding Image Priors section."""
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

    with st.expander("What makes a ConvNet architecture a prior?"):
        st.markdown(
            "A ConvNet's architecture encodes assumptions about natural images through four key features:\n\n"
            "**Local connectivity** — Convolutional filters assume nearby pixels are correlated. "
            "A 3×3 filter encodes the belief that 'neighbors matter.'\n\n"
            "**Weight sharing** — The same filter applied everywhere assumes that patterns repeat "
            "at different positions (edges, textures can appear anywhere in an image).\n\n"
            "**Multi-scale hierarchy** — Downsampling layers mean the network sees structure at "
            "multiple resolutions, from fine details to global composition.\n\n"
            "**Smoothness bias** — Deep compositions of smooth operations (conv, ReLU, upsample) "
            "favor smooth outputs over random noise."
        )

        st.markdown("#### Concrete Example: Denoising with a Random ConvNet")
        st.markdown(
            "Imagine denoising a single image of a cat corrupted by Gaussian noise, using a random "
            "ConvNet with NO training data:\n\n"
            "1. **Start**: Corrupted image of a cat + Gaussian noise. Random ConvNet initialized.\n"
            "2. **Iterations 1-100**: The network first recovers large-scale structure — cat silhouette, "
            "major color regions emerge.\n"
            "3. **Iterations 100-1000**: Edges sharpen, fur textures emerge. The noise pattern "
            "lags behind.\n"
            "4. **Sweet spot (~1000 iterations)**: Stop here and you get a clean image. Signal is "
            '"easy" for the ConvNet to represent; noise is "hard."\n'
            "5. **Keep going (5000+ iterations)**: The network starts fitting the noise. "
            "The gap between signal convergence and noise convergence IS the Deep Image Prior."
        )

        st.markdown("#### Why the ConvNet Resists Noise")
        st.markdown(
            "To fit a smooth edge: one filter, weight sharing works everywhere. "
            "**To fit noise**: every pixel is independent, weight sharing doesn't help, requires "
            "memorizing each pixel separately. The architecture is like a compression scheme "
            "biased toward natural images."
        )

        st.markdown("#### Industry Analogy")
        st.markdown(
            "Think of the ConvNet's architecture as a language with limited vocabulary. "
            "Natural images are well-formed English sentences. Noise is random character sequences. "
            'The network "writes in English" first (fits signal), then has to memorize random '
            "characters (fits noise)."
        )

    st.markdown(
        "**Industry connection:** Image priors are the backbone of computational photography "
        "(phone cameras, Photoshop). Super-resolution, denoising, inpainting, and HDR all rely "
        "on prior knowledge of what images should look like."
    )
