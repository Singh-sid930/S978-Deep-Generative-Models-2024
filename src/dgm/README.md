# dgm -- Deep Generative Models Package

PyTorch implementations of generative model families covered in MIT 6.S978 (Fall 2024).

## Subpackages

| Subpackage | Description | Course Week |
|---|---|---|
| `dgm.vae` | Variational Autoencoders | Week 2 |
| `dgm.flows` | Normalizing Flows | Week 3 |
| `dgm.autoregressive` | Autoregressive Models | Weeks 3-5 |
| `dgm.gan` | Generative Adversarial Networks | Weeks 5-6 |
| `dgm.diffusion` | Diffusion Models | Weeks 7-8 |
| `dgm.discrete_diffusion` | Discrete Diffusion | Week 9 |
| `dgm.flow_matching` | Flow Matching | Weeks 9-10 |
| `dgm.nn` | Shared neural network building blocks | -- |
| `dgm.training` | Training loops and evaluation | -- |
| `dgm.data` | Dataset utilities | -- |
| `dgm.utils` | Visualization and helpers | -- |

## Installation

```bash
pip install -e .
```

## Usage

```python
from dgm.vae import VAE
from dgm.diffusion import DDPM
from dgm.flow_matching import FlowMatcher
```

Implementations will be added week by week as the course progresses.
