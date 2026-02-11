# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a learning repository for MIT 6.S978: Deep Generative Models (Fall 2024). It bridges academia and industry — academic research implemented with clean, idiomatic Python that industry professionals can immediately apply.

Three components:
1. **Streamlit WebUI** (`webapp/`) — week-by-week course walkthrough
2. **Python package** (`src/dgm/`) — installable via `pip install -e .`, organized by model type
3. **READMEs** — at every subfolder level for old-school navigation

## Course Website

https://mit-6s978.github.io/schedule.html

## Package Structure

The Python package is organized by **model type** (not by week):

```
src/dgm/
  vae/                 # Week 2: Variational Autoencoders
  flows/               # Week 3: Normalizing Flows
  autoregressive/      # Weeks 3-5: Autoregressive Models
  gan/                 # Weeks 5-6: GANs
  diffusion/           # Weeks 7-8: Diffusion Models (incl. EBM, score matching)
  discrete_diffusion/  # Week 9: Discrete Diffusion
  flow_matching/       # Weeks 9-10: Flow Matching
  nn/                  # Shared neural net building blocks
  training/            # Training loops, callbacks, metrics
  data/                # Dataset wrappers, transforms
  utils/               # Visualization, device, seed helpers
```

The webapp navigates by **week**. `webapp/course_manifest.py` is the bridge — single source of truth for the week-to-code mapping.

## Development Commands

```bash
pip install -e ".[all]"          # Install with all dependencies
pytest tests/                     # Run tests
streamlit run webapp/app.py       # Launch tutorial webapp
ruff check src/                   # Lint
```

## Coding Conventions

- **PyTorch** for all deep learning code
- **Type hints** on all function signatures
- **Google-style docstrings** for all public classes and methods
- **`src/` layout** with hatchling build backend
- **pytest** for testing; test files mirror `src/dgm/` structure in `tests/`
- **Ruff** for linting (PEP 8 + import sorting)
- **dataclasses** for configuration objects (not raw dicts)

## Stub Pattern

Assignment functions use this pattern — signatures, type hints, docstrings with hints, and `raise NotImplementedError`:

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

    Reference:
        Kingma & Welling, "Auto-Encoding Variational Bayes", 2013, Eq. 9-10.
    """
    raise NotImplementedError("Implement the encoder forward pass.")
```

## Branching Strategy

- **`main`** — assignment stubs (users fork from here)
- **`solutions/week-XX`** — completed reference implementations per week

## Agent Workflow

When building weekly content, the typical workflow is:
1. **dgm-research-scientist** — distills papers into core concepts and informs what to implement
2. **python-implementer** — creates stub files with signatures, types, docstrings
3. **dgm-test-writer** — writes pytest tests for the new module
4. **streamlit-web-builder** — builds the week's webapp page
5. **readme-generator** — creates/updates READMEs for affected directories
6. **product-strategist-advisor** — validates architectural decisions when needed
