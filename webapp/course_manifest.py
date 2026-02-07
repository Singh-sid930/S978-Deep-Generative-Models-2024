"""Course manifest: maps MIT 6.S978 weekly schedule to dgm package modules.

This is the single source of truth for the week-to-code relationship.
The webapp reads this to generate navigation and content.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Reading:
    """A paper or resource from the reading list."""
    title: str
    authors: str
    venue: str
    year: int
    url: str
    is_optional: bool = False


@dataclass(frozen=True)
class Session:
    """A single class session (lecture or reading discussion)."""
    date: str
    session_type: str  # "lecture" | "reading" | "guest_lecture" | "holiday"
    title: str
    description: str
    dgm_modules: list[str] = field(default_factory=list)
    readings: list[Reading] = field(default_factory=list)
    slides_url: str | None = None
    assignment_stubs: list[str] = field(default_factory=list)


COURSE_SCHEDULE: dict[int, list[Session]] = {
    1: [
        Session(
            date="2024-09-05",
            session_type="lecture",
            title="Introduction",
            description="Course overview, taxonomy of generative models, and motivation.",
            slides_url="https://mit-6s978.github.io/assets/pdfs/lec1_intro.pdf",
        ),
    ],
    2: [
        Session(
            date="2024-09-10",
            session_type="reading",
            title="Modeling Image Prior",
            description="Statistical models of natural images, from patches to deep priors.",
            dgm_modules=["dgm.data"],
            readings=[
                Reading("From Learning Models of Natural Image Patches to Whole Image Restoration", "D. Zoran and Y. Weiss", "ICCV", 2011, "https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6126278"),
                Reading("Natural Images, Gaussian Mixtures and Dead Leaves", "D. Zoran and Y. Weiss", "NeurIPS", 2012, "https://proceedings.neurips.cc/paper_files/paper/2012/file/e97ee2054defb209c35fe4dc94"),
                Reading("Deep Image Prior", "D. Ulyanov et al.", "CVPR", 2018, "https://openaccess.thecvf.com/content_cvpr_2018/papers/Ulyanov_Deep_Image_Prior_C"),
                Reading("Texture Synthesis by Non-parametric Sampling", "A. A. Efros and T. K. Leung", "ICCV", 1999, "https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=790383", is_optional=True),
                Reading("PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing", "C. Barnes et al.", "SIGGRAPH", 2009, "https://3dvar.com/Barnes2009PatchMatch.pdf", is_optional=True),
                Reading("Example-Based Super-Resolution", "W. T. Freeman et al.", "IEEE CGA", 2002, "https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=988747", is_optional=True),
            ],
        ),
        Session(
            date="2024-09-12",
            session_type="lecture",
            title="Variational Autoencoder (VAE)",
            description="ELBO derivation, reparameterization trick, encoder-decoder architecture.",
            dgm_modules=["dgm.vae"],
            slides_url="https://mit-6s978.github.io/assets/pdfs/lec2_vae.pdf",
        ),
    ],
    3: [
        Session(
            date="2024-09-17",
            session_type="reading",
            title="Normalizing Flows",
            description="Change of variables formula, coupling layers, invertible networks.",
            dgm_modules=["dgm.flows"],
            readings=[
                Reading("Variational Inference with Normalizing Flows", "D. Rezende and S. Mohamed", "ICML", 2015, "https://proceedings.mlr.press/v37/rezende15"),
                Reading("Glow: Generative Flow with Invertible 1x1 Convolutions", "D. P. Kingma and P. Dhariwal", "NeurIPS", 2018, "https://proceedings.neurips.cc/paper_files/paper/2018/hash/d139db6a236200b21cc7f752"),
                Reading("Invertible Residual Networks", "J. Behrmann et al.", "ICML", 2019, "https://proceedings.mlr.press/v97/behrmann19a.html"),
                Reading("Flow-based Deep Generative Models (blog post)", "Lilian Weng", "Blog", 2018, "https://lilianweng.github.io/posts/2018-10-13-flow-models/", is_optional=True),
            ],
        ),
        Session(
            date="2024-09-19",
            session_type="lecture",
            title="Autoregressive (AR) Models",
            description="Factorization of joint distributions, causal masking, likelihood-based training.",
            dgm_modules=["dgm.autoregressive"],
            slides_url="https://mit-6s978.github.io/assets/pdfs/lec3_ar.pdf",
        ),
    ],
    4: [
        Session(
            date="2024-09-24",
            session_type="reading",
            title="Autoregressive (AR) Models",
            description="Foundational AR architectures: neural autoregressive models, PixelCNN, inverse autoregressive flows.",
            dgm_modules=["dgm.autoregressive"],
            readings=[
                Reading("Modeling High-Dimensional Discrete Data with Multi-Layer Neural Networks", "Y. Bengio and S. Bengio", "NeurIPS", 1999, "https://proceedings.neurips.cc/paper_files/paper/1999/file/e63847114913"),
                Reading("Pixel Recurrent Neural Networks", "A. Van Den Oord et al.", "ICML", 2016, "https://proceedings.mlr.press/v48/oord16.pdf"),
                Reading("Improved Variational Inference with Inverse Autoregressive Flow", "D. P. Kingma et al.", "NeurIPS", 2016, "https://proceedings.neurips.cc/paper_files/paper/2016/file/ddeebdeefdb7e7e7a697e1c3e"),
            ],
        ),
        Session(
            date="2024-09-26",
            session_type="reading",
            title="AR and Tokenizers",
            description="Visual tokenizers, VQ-VAE, scaling autoregressive models for images.",
            dgm_modules=["dgm.autoregressive"],
            readings=[
                Reading("Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation", "L. Yu et al.", "ICLR", 2024, "https://arxiv.org/abs/2310.05737"),
                Reading("Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction", "K. Tian et al.", "arXiv", 2024, "https://arxiv.org/abs/2404.02905"),
                Reading("An Image is Worth 32 Tokens for Reconstruction and Generation", "Q. Yu et al.", "arXiv", 2024, "https://arxiv.org/abs/2406.07550"),
                Reading("Generative Pretraining from Pixels", "M. Chen et al.", "ICML", 2020, "https://proceedings.mlr.press/v119/chen20s/chen20s.pdf", is_optional=True),
                Reading("Zero-Shot Text-to-Image Generation (DALLE1)", "A. Ramesh et al.", "ICML", 2021, "https://arxiv.org/pdf/2102.12092", is_optional=True),
                Reading("Finite Scalar Quantization: VQ-VAE Made Simple", "F. Mentzer et al.", "arXiv", 2023, "https://arxiv.org/abs/2309.15505", is_optional=True),
            ],
        ),
    ],
    5: [
        Session(
            date="2024-10-01",
            session_type="reading",
            title="AR and Diffusion",
            description="Combining autoregressive and diffusion approaches for image generation.",
            dgm_modules=["dgm.autoregressive", "dgm.diffusion"],
            readings=[
                Reading("Autoregressive Image Generation without Vector Quantization", "T. Li et al.", "arXiv", 2024, "https://arxiv.org/abs/2406.11838"),
                Reading("Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model", "C. Zhou et al.", "arXiv", 2024, "https://arxiv.org/abs/2408.11039"),
                Reading("Show-o: One Single Transformer to Unify Multimodal Understanding and Generation", "J. Xie et al.", "arXiv", 2024, "https://arxiv.org/abs/2408.12528"),
                Reading("Autoregressive Diffusion Models", "E. Hoogeboom et al.", "ICLR", 2022, "https://arxiv.org/abs/2110.02037", is_optional=True),
                Reading("Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion", "B. Chen et al.", "arXiv", 2024, "https://arxiv.org/abs/2407.01392", is_optional=True),
            ],
        ),
        Session(
            date="2024-10-03",
            session_type="lecture",
            title="Generative Adversarial Network (GAN)",
            description="Minimax formulation, training dynamics, mode collapse, modern architectures.",
            dgm_modules=["dgm.gan"],
            slides_url="https://mit-6s978.github.io/assets/pdfs/lec4_gan.pdf",
        ),
    ],
    6: [
        Session(
            date="2024-10-08",
            session_type="reading",
            title="GAN 1",
            description="Modern GAN architectures: StyleGAN-T, scaling GANs, distilling diffusion into GANs.",
            dgm_modules=["dgm.gan"],
            readings=[
                Reading("StyleGAN-T: Unlocking the Power of GANs for Fast Large-Scale Text-to-Image Synthesis", "A. Sauer et al.", "arXiv", 2023, "https://arxiv.org/abs/2301.09515"),
                Reading("Scaling up GANs for Text-to-Image Synthesis", "M. Kang et al.", "CVPR", 2023, "https://arxiv.org/abs/2303.05511"),
                Reading("Distilling Diffusion Models into Conditional GANs", "M. Kang et al.", "ECCV", 2024, "https://arxiv.org/abs/2405.05967"),
                Reading("Learning Generative Models via Discriminative Approaches", "Z. Tu", "CVPR", 2007, "https://ieeexplore.ieee.org/abstract/document/4270060", is_optional=True),
            ],
        ),
        Session(
            date="2024-10-10",
            session_type="reading",
            title="GAN 2",
            description="Modern GAN baselines, training GANs with diffusion, GAN evaluation.",
            dgm_modules=["dgm.gan"],
            readings=[
                Reading("The GAN is dead; long live the GAN! A Modern GAN Baseline", "N. Huang et al.", "ICML", 2024, "https://openreview.net/pdf/9197a4074900f763675c3c896d7c4ca3402a3c55.pdf"),
                Reading("Diffusion-GAN: Training GANs with Diffusion", "Z. Wang et al.", "ICLR", 2023, "https://arxiv.org/abs/2206.02262"),
                Reading("GANs Settle Scores!", "S. Asokan et al.", "arXiv", 2023, "https://arxiv.org/abs/2306.01654"),
            ],
        ),
    ],
    7: [
        Session(
            date="2024-10-15",
            session_type="holiday",
            title="No class (student holiday)",
            description="",
        ),
        Session(
            date="2024-10-17",
            session_type="lecture",
            title="Energy-based Models, Score Matching, Diffusion Models",
            description="EBMs, Boltzmann distributions, score functions, Langevin dynamics, denoising score matching, DDPM.",
            dgm_modules=["dgm.diffusion"],
            slides_url="https://mit-6s978.github.io/assets/pdfs/lec5_diffusion.pdf",
        ),
    ],
    8: [
        Session(
            date="2024-10-22",
            session_type="reading",
            title="Diffusion Models",
            description="Classifier-free guidance, progressive distillation, simple diffusion.",
            dgm_modules=["dgm.diffusion"],
            readings=[
                Reading("Classifier-Free Diffusion Guidance", "J. Ho and T. Salimans", "NeurIPS", 2021, "https://arxiv.org/abs/2207.12598"),
                Reading("Progressive Distillation for Fast Sampling of Diffusion Models", "T. Salimans and J. Ho", "ICLR", 2022, "https://arxiv.org/abs/2202.00512"),
                Reading("Simple diffusion: End-to-end diffusion for high resolution images", "E. Hoogeboom et al.", "ICML", 2023, "https://arxiv.org/abs/2301.11093"),
                Reading("High-Resolution Image Synthesis with Latent Diffusion Models", "R. Rombach et al.", "CVPR", 2022, "https://arxiv.org/abs/2112.10752", is_optional=True),
                Reading("Elucidating the Design Space of Diffusion-Based Generative Models", "T. Karras et al.", "NeurIPS", 2022, "https://arxiv.org/abs/2206.00364", is_optional=True),
                Reading("Hierarchical Text-Conditional Image Generation with CLIP Latents", "A. Ramesh et al.", "arXiv", 2022, "https://arxiv.org/abs/2204.06125", is_optional=True),
                Reading("On the Importance of Noise Scheduling for Diffusion Models", "T. Chen", "arXiv", 2023, "https://arxiv.org/pdf/2301.10972", is_optional=True),
                Reading("Diffusion is spectral autoregression (blog post)", "Sander Dieleman", "Blog", 2024, "https://sander.ai/2024/09/02/spectral-autoregression.html", is_optional=True),
            ],
        ),
        Session(
            date="2024-10-24",
            session_type="reading",
            title="Diffusion beyond Denoising",
            description="Cold diffusion, inverse heat dissipation, direct iteration inversion.",
            dgm_modules=["dgm.diffusion"],
            readings=[
                Reading("Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise", "A. Bansal et al.", "NeurIPS", 2022, "https://arxiv.org/abs/2208.09392"),
                Reading("Generative Modelling With Inverse Heat Dissipation", "S. Rissanen et al.", "ICLR", 2023, "https://arxiv.org/abs/2206.13397"),
                Reading("Inversion by Direct Iteration: An Alternative to Denoising Diffusion for Image Restoration", "M. Delbracio et al.", "TMLR", 2023, "https://arxiv.org/abs/2303.11435"),
                Reading("Soft Diffusion: Score Matching for General Corruptions", "G. Daras et al.", "TMLR", 2023, "https://arxiv.org/abs/2209.05442", is_optional=True),
            ],
        ),
    ],
    9: [
        Session(
            date="2024-10-29",
            session_type="reading",
            title="Discrete Diffusion",
            description="Diffusion models for discrete state spaces, sequence generation.",
            dgm_modules=["dgm.discrete_diffusion"],
            readings=[
                Reading("Structured Denoising Diffusion Models in Discrete State-Spaces", "J. Austin et al.", "NeurIPS", 2021, "https://arxiv.org/abs/2107.03006"),
                Reading("DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models", "S. Gong et al.", "ICLR", 2023, "https://arxiv.org/abs/2210.08933"),
                Reading("Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution", "A. Lou et al.", "arXiv", 2023, "https://arxiv.org/abs/2310.16834"),
            ],
        ),
        Session(
            date="2024-10-31",
            session_type="reading",
            title="Flow Matching 1",
            description="Foundations of flow matching: conditional flow matching, stochastic interpolants, rectified flows.",
            dgm_modules=["dgm.flow_matching"],
            readings=[
                Reading("Flow Matching for Generative Modeling", "Y. Lipman et al.", "ICLR", 2023, "https://arxiv.org/abs/2210.02747"),
                Reading("Building Normalizing Flows with Stochastic Interpolants", "M. S. Albergo et al.", "ICLR", 2023, "https://arxiv.org/abs/2209.15571"),
                Reading("Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flows", "X. Liu et al.", "ICLR", 2023, "https://arxiv.org/abs/2209.03003"),
                Reading("An Introduction to Flow Matching (blog post)", "T. Fjelde et al.", "Blog", 2024, "https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html", is_optional=True),
            ],
        ),
    ],
    10: [
        Session(
            date="2024-11-05",
            session_type="reading",
            title="Flow Matching 2",
            description="Scaling flow matching: rectified flow transformers, SiT, discrete flow matching.",
            dgm_modules=["dgm.flow_matching"],
            readings=[
                Reading("Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", "P. Esser et al.", "arXiv", 2024, "https://arxiv.org/abs/2403.03206"),
                Reading("SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers", "N. Ma et al.", "ECCV", 2024, "https://arxiv.org/abs/2401.08740"),
                Reading("Discrete Flow Matching", "I. Gat et al.", "arXiv", 2024, "https://arxiv.org/abs/2407.15595"),
            ],
        ),
        Session(
            date="2024-11-07",
            session_type="guest_lecture",
            title="Guest Lecture: Jun-Yan Zhu - Ensuring Data Ownership in Generative Models",
            description="Data ownership, attribution, and ethical considerations in generative AI.",
            slides_url="https://mit-6s978.github.io/assets/pdfs/data_ownership_MIT_guest_lecture.pdf",
        ),
    ],
    11: [
        Session(
            date="2024-11-12",
            session_type="reading",
            title="Application - Videos",
            description="Spatiotemporal diffusion models for video generation.",
            dgm_modules=["dgm.diffusion"],
            readings=[
                Reading("Lumiere: A Space-Time Diffusion Model for Video Generation", "O. Bar-Tal et al.", "arXiv", 2024, "https://arxiv.org/abs/2401.12945"),
                Reading("Genie: Generative Interactive Environments", "J. Bruce et al.", "arXiv", 2024, "https://arxiv.org/abs/2402.15391"),
                Reading("Movie Gen: A Cast of Media Foundation Models", "The Movie Gen team @ Meta", "arXiv", 2024, "https://ai.meta.com/static-resource/movie-gen-research-paper"),
            ],
        ),
        Session(
            date="2024-11-14",
            session_type="reading",
            title="Application - 3D and Geometry",
            description="Text-to-3D, large reconstruction models, mesh generation.",
            dgm_modules=["dgm.diffusion"],
            readings=[
                Reading("DreamFusion: Text-to-3D using 2D Diffusion", "B. Poole et al.", "ICLR", 2023, "https://arxiv.org/abs/2209.14988"),
                Reading("LRM: Large Reconstruction Model for Single Image to 3D", "Y. Hong et al.", "ICLR", 2024, "https://arxiv.org/abs/2311.04400"),
                Reading("MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers", "Y. Siddiqui et al.", "arXiv", 2023, "https://nihalsid.github.io/mesh-gpt/"),
                Reading("CLAY: A Controllable Large-scale Generative Model for Creating High-quality 3D Assets", "L. Zhang et al.", "SIGGRAPH", 2024, "https://arxiv.org/abs/2406.13897", is_optional=True),
                Reading("MeshLRM: Large Reconstruction Model for High-Quality Meshes", "X. Wei et al.", "arXiv", 2024, "https://sarahweiii.github.io/meshlrm/", is_optional=True),
                Reading("Flexible Isosurface Extraction for Gradient-Based Mesh Optimization", "T. Shen et al.", "SIGGRAPH", 2023, "https://research.nvidia.com/labs/toronto-ai/flexicubes/", is_optional=True),
            ],
        ),
    ],
    12: [
        Session(
            date="2024-11-19",
            session_type="reading",
            title="Application - Robotics",
            description="Diffusion models for planning, policy learning, and simulation.",
            dgm_modules=["dgm.diffusion"],
            readings=[
                Reading("Planning with Diffusion for Flexible Behavior Synthesis", "M. Janner et al.", "ICML", 2022, "https://arxiv.org/abs/2205.09991"),
                Reading("Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", "C. Chi et al.", "RSS", 2023, "https://arxiv.org/abs/2303.04137"),
                Reading("UniSim: Learning Interactive Real-World Simulators", "S. Yang et al.", "ICLR", 2024, "https://arxiv.org/abs/2310.06114"),
                Reading("PaLM-E: An Embodied Multimodal Language Model", "D. Driess et al.", "ICML", 2023, "https://arxiv.org/abs/2303.03378", is_optional=True),
            ],
        ),
        Session(
            date="2024-11-21",
            session_type="guest_lecture",
            title="Guest Lecture: Yang Song - Consistency Models",
            description="Consistency models for fast sampling from diffusion models.",
            slides_url="https://mit-6s978.github.io/assets/pdfs/CM_lecture.pdf",
        ),
    ],
    13: [
        Session(
            date="2024-11-26",
            session_type="reading",
            title="Application - Material Science",
            description="Molecular graph generation, equivariant diffusion, 3D molecule design.",
            dgm_modules=["dgm.diffusion", "dgm.flow_matching"],
            readings=[
                Reading("Junction Tree Variational Autoencoder for Molecular Graph Generation", "W. Jin et al.", "ICML", 2018, "https://arxiv.org/pdf/1802.04364"),
                Reading("Equivariant Diffusion for Molecule Generation in 3D", "E. Hoogeboom et al.", "ICML", 2022, "https://proceedings.mlr.press/v162/hoogeboom22a"),
                Reading("Uni-Mol: A Universal 3D Molecular Representation Learning Framework", "G. Zhou et al.", "ICLR", 2023, "https://openreview.net/forum?id=6K2RM6wVqKu"),
                Reading("Geometric Latent Diffusion Models for 3D Molecule Generation", "M. Xu et al.", "ICML", 2023, "https://arxiv.org/abs/2305.01140", is_optional=True),
                Reading("Two for One: Diffusion Models and Force Fields for Coarse-Grained Molecular Dynamics", "M. Arts et al.", "arXiv", 2023, "https://arxiv.org/abs/2302.00600", is_optional=True),
            ],
        ),
        Session(
            date="2024-11-28",
            session_type="holiday",
            title="No class (Thanksgiving)",
            description="",
        ),
        Session(
            date="2024-12-03",
            session_type="reading",
            title="Application - Protein and Biology",
            description="Protein structure prediction, design, and programmable generation.",
            dgm_modules=["dgm.diffusion", "dgm.flow_matching"],
            readings=[
                Reading("De novo design of protein structure and function with RFdiffusion", "J. L. Watson et al.", "Nature", 2023, "https://www.nature.com/articles/s41586-023-06415-8"),
                Reading("Accurate structure prediction of biomolecular interactions with AlphaFold 3", "J. Abramson et al.", "Nature", 2024, "https://www.nature.com/articles/s41586-024-07487-w"),
                Reading("Illuminating protein space with a programmable generative model", "J. B. Ingraham et al.", "Nature", 2023, "https://www.nature.com/articles/s41586-023-06728-8"),
            ],
        ),
    ],
}
