"""Section: Validate Your VAE Implementation

Interactive training and visualization to verify the user's VAE works correctly.
"""

import streamlit as st
import torch
import matplotlib.pyplot as plt


def _check_implementation():
    """Test if the VAE stubs are implemented by running a quick forward pass."""
    try:
        from dgm.vae import VAE
        from dgm.vae.loss import vae_loss

        model = VAE(input_dim=784, latent_dim=2)
        x = torch.randn(2, 784).clamp(0, 1)  # Clamp to [0,1] for BCE
        recon, mu, log_var = model(x)
        total, recon_l, kl_l = vae_loss(recon, x, mu, log_var)
        st.success("✓ All VAE components implemented. Ready to train!")
        return True
    except NotImplementedError as e:
        st.warning(f"Implementation incomplete: `{e}`\n\nComplete all four stub files before training.")
        return False
    except Exception as e:
        st.error(f"Error in your implementation: `{e}`")
        return False


def render():
    """Render the VAE validation section."""
    st.markdown("---")
    st.markdown("## Validate Your Implementation")

    st.markdown("""
Once you've implemented the four stub files, use this section to train your VAE on MNIST
and verify it produces meaningful results.
    """)

    # Check implementation first
    if not _check_implementation():
        st.stop()

    # Sidebar hyperparameters
    st.sidebar.markdown("---")
    st.sidebar.markdown("### VAE Training Settings")

    latent_dim = st.sidebar.slider("Latent Dimension", 2, 128, 64)
    st.sidebar.caption("Use 2 for direct latent space visualization")

    epochs = st.sidebar.slider("Epochs", 1, 50, 10)
    batch_size = st.sidebar.selectbox("Batch Size", [32, 64, 128, 256], index=2)
    learning_rate = st.sidebar.selectbox("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3], index=2)

    # Training button
    if st.button("Train VAE", type="primary"):
        from dgm.vae import VAE
        from dgm.data import get_mnist_loaders
        from dgm.training import train_vae

        # Create model with selected hyperparams
        model = VAE(input_dim=784, latent_dim=latent_dim)

        # Load data (show spinner during potential download)
        with st.spinner("Loading MNIST..."):
            loaders = get_mnist_loaders(batch_size=batch_size)

        # Train with progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(epoch, total, loss):
            progress_bar.progress(epoch / total)
            status_text.text(f"Epoch {epoch}/{total} — Loss: {loss:.4f}")

        history = train_vae(
            model, loaders.train,
            epochs=epochs, lr=learning_rate,
            recon_loss_type="bce",
            progress_callback=update_progress,
        )

        progress_bar.empty()
        status_text.empty()
        st.success("Training complete!")

        # Store in session state
        st.session_state.vae_model = model
        st.session_state.vae_history = history
        st.session_state.vae_loaders = loaders
        st.session_state.vae_device = next(model.parameters()).device
        st.session_state.vae_latent_dim = latent_dim

    # Visualization tabs (only shown if model exists in session_state)
    if "vae_model" in st.session_state and st.session_state.vae_model is not None:
        from dgm.utils import (
            plot_training_curves,
            plot_reconstructions,
            plot_samples,
            plot_latent_space,
            plot_interpolation,
        )

        # Retrieve from session state
        model = st.session_state.vae_model
        history = st.session_state.vae_history
        loaders = st.session_state.vae_loaders
        device = st.session_state.vae_device
        latent_dim = st.session_state.vae_latent_dim

        st.markdown("---")
        st.markdown("### Validation Results")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Training Curves",
            "Reconstructions",
            "Samples",
            "Latent Space",
            "Interpolation"
        ])

        with tab1:
            st.markdown("""
Loss should decrease over epochs. If KL collapses to 0 early, the model may ignore
the latent space.
            """)
            fig = plot_training_curves(history)
            st.pyplot(fig)
            plt.close(fig)

        with tab2:
            st.markdown("""
Reconstructed digits should be recognizable. Some blurriness is normal for VAEs.
            """)
            model.eval()
            with torch.no_grad():
                x_test, _ = next(iter(loaders.test))
                x_test = x_test.to(device)
                recon, _, _ = model(x_test)
            fig = plot_reconstructions(x_test, recon)
            st.pyplot(fig)
            plt.close(fig)

        with tab3:
            st.markdown("""
Generated digits should look like handwritten numbers. Some may be ambiguous — that's expected.
            """)
            if st.button("Regenerate Samples", key="resample"):
                pass  # Button press triggers re-run

            model.eval()
            samples = model.sample(64, device)
            fig = plot_samples(samples)
            st.pyplot(fig)
            plt.close(fig)

        with tab4:
            if latent_dim == 2:
                st.markdown("""
Direct 2D visualization. Digits of the same class should cluster together.
                """)
            else:
                st.markdown(f"""
Using dimensionality reduction on {latent_dim}-dim latent space. Clusters indicate
structured latent space.
                """)

            model.eval()
            all_latents = []
            all_labels = []
            with torch.no_grad():
                for x_batch, y_batch in loaders.test:
                    x_batch = x_batch.to(device)
                    mu, _ = model.encoder(x_batch)
                    all_latents.append(mu.cpu().numpy())
                    all_labels.append(y_batch.numpy())

            import numpy as np
            latents = np.concatenate(all_latents)
            labels = np.concatenate(all_labels)

            with st.spinner("Computing latent space visualization..."):
                fig = plot_latent_space(latents, labels)
            st.pyplot(fig)
            plt.close(fig)

        with tab5:
            st.markdown("""
Smooth transitions between digits indicate a continuous, well-structured latent space.
            """)
            if st.button("New Pair", key="new_interp"):
                pass  # triggers re-run

            model.eval()
            # Get two random test images, encode them
            x_test, y_test = next(iter(loaders.test))
            x_test = x_test.to(device)
            with torch.no_grad():
                mu, _ = model.encoder(x_test)

            # Pick two random indices
            import random
            idx1, idx2 = random.sample(range(x_test.shape[0]), 2)

            st.markdown(f"Interpolating between two encoded test images:")
            fig = plot_interpolation(model, mu[idx1], mu[idx2], steps=12, device=device)
            st.pyplot(fig)
            plt.close(fig)
