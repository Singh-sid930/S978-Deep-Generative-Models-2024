"""Week 1: Introduction."""

import streamlit as st

from webapp.page_utils import setup_page, render_sessions
from webapp.content.week_01 import taxonomy, duality, learning_distributions, roadmap

setup_page(1)
render_sessions(1)

# --- Educational Content ---

st.markdown("---")
st.markdown("## What Are Generative Models?")
st.markdown(
    "Discriminative models learn decision boundaries — given an input, they predict a label "
    '("is this a cat or a dog?"). Generative models learn the data distribution $p(x)$ itself — '
    "they can create entirely new, realistic data points. If a discriminative model is a critic, "
    "a generative model is an artist. The central question: given a dataset of examples, can we "
    "learn a model that generates new examples indistinguishable from the real ones?"
)

taxonomy.render()
duality.render()
learning_distributions.render()
roadmap.render()

from webapp.chat import render_chat

render_chat(1)
