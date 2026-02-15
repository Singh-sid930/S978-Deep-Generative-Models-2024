# MIT 6.S978: Deep Generative Models

A learning repository for [MIT 6.S978: Deep Generative Models (Fall 2024)](https://mit-6s978.github.io/schedule.html). Academic research implemented with clean, idiomatic Python/PyTorch.

## Quick Start

```bash
git clone https://github.com/yourusername/S978-Deep-Generative-Models-2024.git
cd S978-Deep-Generative-Models-2024
conda env create -f environment.yml
conda activate dgm
streamlit run webapp/app.py
```

That's it. The conda environment installs Python 3.12, PyTorch, Streamlit, and all dependencies. Open the browser link and start learning.

## Three Ways to Learn

### 1. Interactive WebUI
A Streamlit app walks you through each week — theory, readings, code assignments, and visual validation.
```bash
streamlit run webapp/app.py
```

### 2. Python Package
An installable PyTorch package organized by model type. Fill in the stub functions, run the unit tests, and validate visually in the webapp.
```python
from dgm.vae import VAE
from dgm.diffusion import DDPM
from dgm.flow_matching import FlowMatcher
```

### 3. READMEs
Every directory has a README for browsing in your editor or on GitHub.

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
  training/                # Training loops, metrics
  data/                    # Dataset loaders (MNIST, etc.)
  utils/                   # Visualization helpers

webapp/                    # Streamlit tutorial app
  course_manifest.py       # Week-to-code mapping (single source of truth)
  pages/                   # One page per week
  content/                 # Rich content modules per week

tests/                     # pytest test suite (mirrors src/)
```

## Course-to-Code Mapping

| Week | Topic | Package Module |
|------|-------|---------------|
| 1 | Introduction | -- |
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

Implementation functions have full signatures, type hints, docstrings with hints, and `raise NotImplementedError`:

```python
def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode input into latent distribution parameters.

    Hints:
        - Pass x through self.network to get hidden features.
        - Use self.fc_mu and self.fc_log_var to project to latent params.
    """
    raise NotImplementedError("Implement the encoder forward pass.")
```

Validate your implementation:
1. **Unit tests**: `pytest tests/test_vae/` -- checks shapes, gradients, ranges
2. **Visual validation**: The webapp has a "Validate Your Implementation" section that trains on MNIST and shows reconstructions, samples, latent space, and interpolations

### Branching Strategy

- **`main`** -- Assignment stubs. Fork from here.
- **`solutions/week-XX`** -- Reference implementations per week.

## Installation

**Recommended -- Conda** (creates `dgm` environment with everything):
```bash
conda env create -f environment.yml
conda activate dgm
```

**Alternative -- pip only** (Python 3.10+ required):
```bash
pip install -e ".[all]"    # Everything: PyTorch, Streamlit, scikit-learn, pytest, ruff
```

Or install selectively:
```bash
pip install -e .            # Core only (PyTorch + basics)
pip install -e ".[webapp]"  # + Streamlit
pip install -e ".[viz]"     # + scikit-learn (for t-SNE latent space plots)
pip install -e ".[dev]"     # + pytest, ruff
```

## AI Chat (Optional)

Each week's page includes a chat assistant that can answer questions using course slides, code, and readings as context. To enable it:

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

## License

MIT
