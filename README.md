# MIT 6.S978: Deep Generative Models

An industry-grade learning repository for [MIT 6.S978: Deep Generative Models (Fall 2024)](https://mit-6s978.github.io/schedule.html).

This repository bridges academia and industry: academic research concepts implemented with clean, idiomatic Python that industry professionals can immediately apply, and academic researchers can see how their work translates to production-quality code.

## Quick Start

```bash
# Clone and set up environment
git clone https://github.com/yourusername/S978-Deep-Generative-Models-2024.git
cd S978-Deep-Generative-Models-2024
conda env create -f environment.yml
conda activate dgm

# Launch the interactive tutorial
streamlit run webapp/app.py

# Run tests
pytest tests/
```

## Three Ways to Learn

### 1. Interactive WebUI
A Streamlit app walks you through each week of the course — concepts, readings, and code assignments.
```bash
streamlit run webapp/app.py
```

### 2. Python Package
A clean, installable PyTorch package organized by model type. Fork the repo, fill in the stub functions, and build your understanding through implementation.
```python
from dgm.vae import VAE
from dgm.diffusion import DDPM
from dgm.flow_matching import FlowMatcher
```

### 3. READMEs
Every directory has a README documenting what it contains, what concepts it covers, and how to use it. Navigate the repo in your editor or on GitHub.

## Repository Structure

```
src/dgm/                  # Installable Python package
  vae/                     # Variational Autoencoders (Week 2)
  flows/                   # Normalizing Flows (Week 3)
  autoregressive/          # Autoregressive Models (Weeks 3-5)
  gan/                     # GANs (Weeks 5-6)
  diffusion/               # Diffusion Models (Weeks 7-8)
  discrete_diffusion/      # Discrete Diffusion (Week 9)
  flow_matching/           # Flow Matching (Weeks 9-10)
  nn/                      # Shared building blocks
  training/                # Training infrastructure
  data/                    # Dataset utilities
  utils/                   # Visualization, helpers

webapp/                    # Streamlit tutorial app
  course_manifest.py       # Week-to-code mapping (single source of truth)
  pages/                   # One page per week

tests/                     # pytest test suite (mirrors src/)
notebooks/                 # Exploratory notebooks
```

## Course-to-Code Mapping

| Week | Topic | Package Module |
|------|-------|---------------|
| 1 | Introduction | — |
| 2 | Image Priors, VAE | `dgm.vae` |
| 3 | Normalizing Flows, AR Models | `dgm.flows`, `dgm.autoregressive` |
| 4 | AR Models, Tokenizers | `dgm.autoregressive` |
| 5 | AR + Diffusion, GAN | `dgm.autoregressive`, `dgm.gan` |
| 6 | GAN Deep Dive | `dgm.gan` |
| 7 | EBM, Score Matching, Diffusion | `dgm.diffusion` |
| 8 | Diffusion Models | `dgm.diffusion` |
| 9 | Discrete Diffusion, Flow Matching | `dgm.discrete_diffusion`, `dgm.flow_matching` |
| 10 | Flow Matching | `dgm.flow_matching` |
| 11-13 | Applications (Video, 3D, Robotics, Materials, Biology) | Reuses existing modules |

## Assignment Workflow

This repo uses a stub-based assignment pattern. Implementation functions have full signatures, type hints, docstrings with hints, and `raise NotImplementedError`:

```python
def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode input into latent distribution parameters.

    Args:
        x: Input tensor of shape (batch_size, input_dim).

    Returns:
        Tuple of (mu, log_var), each of shape (batch_size, latent_dim).

    Hints:
        - Pass x through self.network to get hidden features.
        - Use self.fc_mu and self.fc_log_var to project to latent params.
    """
    raise NotImplementedError("Implement the encoder forward pass.")
```

### Branching Strategy

- **`main`** — Assignment stubs. Fork from here.
- **`solutions/week-XX`** — Reference implementations per week.

## Installation

**Recommended — Conda** (creates `dgm` environment with all dependencies):
```bash
conda env create -f environment.yml
conda activate dgm
```

**Alternative — pip only**:
```bash
pip install -e .          # Core (PyTorch + basics)
pip install -e ".[webapp]"  # + Streamlit
pip install -e ".[all]"     # + pytest, ruff
```

Requires Python 3.10+.

## License

MIT
