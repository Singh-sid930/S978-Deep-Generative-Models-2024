"""Week 2: Image Priors & VAE."""

from webapp.page_utils import setup_page, render_sessions
from webapp.content.week_02 import (
    image_priors,
    vae_theory,
    em_connection,
    vq_vae,
    vae_assignment,
    vae_validation,
)

setup_page(2)
render_sessions(2)

image_priors.render()
vae_theory.render()
em_connection.render()
vq_vae.render()
vae_assignment.render()
vae_validation.render()

from webapp.chat import render_chat

render_chat(2)
