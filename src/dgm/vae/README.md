# Variational Autoencoders (VAE)

ELBO derivation, reparameterization trick, and encoder-decoder architectures. Course reference: **Week 2**.

## Key Papers

- "Auto-Encoding Variational Bayes" -- Kingma & Welling, 2013
- "Deep Image Prior" -- Ulyanov et al., CVPR 2018

## Module Structure

| File | Contents |
|------|----------|
| `encoder.py` | `Encoder` — MLP encoder mapping input to (mu, log_var) |
| `decoder.py` | `Decoder` — MLP decoder mapping latent z to reconstruction |
| `vae.py` | `VAE` — Full model: encoder + reparameterization + decoder |
| `loss.py` | `vae_loss()` — Reconstruction loss + KL divergence |

## Architecture

Default configuration (MNIST):
- **Input**: 784 (flattened 28x28)
- **Encoder**: 784 → 512 → 256 → (mu: 64, log_var: 64)
- **Decoder**: 64 → 256 → 512 → 784 (sigmoid output)

## Assignment

The `__init__` methods are implemented. You need to fill in:
- `Encoder.forward()` — pass input through network, project to mu and log_var
- `Decoder.forward()` — pass latent code through network
- `VAE.reparameterize()` — the reparameterization trick (z = mu + sigma * epsilon)
- `VAE.forward()` — encode, reparameterize, decode
- `VAE.sample()` — generate from prior
- `vae_loss()` — reconstruction loss + KL divergence

## Running Tests

```bash
pytest tests/test_vae/ -v
```
