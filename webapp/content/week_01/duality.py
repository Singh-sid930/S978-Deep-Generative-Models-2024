"""Section: Generative ↔ Discriminative Duality

Connection between generative and discriminative models via Bayes' rule.
"""

import streamlit as st


def render():
    """Render the Generative ↔ Discriminative Duality section."""
    st.markdown("---")
    st.markdown("## Generative ↔ Discriminative Duality")

    st.markdown(
        "One of the most elegant insights in machine learning is the deep connection between "
        "generative and discriminative models. They're not just different approaches — they're "
        "two sides of the same coin, related by Bayes' rule."
    )

    st.markdown("### The Fundamental Difference")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Discriminative: $p(y|x)$**")
        st.markdown("Given data, predict the label.")
        st.markdown("*'Is this email spam?'*")
        st.markdown("Memorize the decision boundary.")
    with col2:
        st.markdown("**Generative: $p(x|y)$ and $p(y)$**")
        st.markdown("Given a label, model what data looks like.")
        st.markdown("*'What do spam emails look like?'*")
        st.markdown("Understand the underlying mechanism.")

    st.markdown("")
    st.markdown(
        "**Analogy:** Diagnosing disease. A discriminative model memorizes symptom patterns → disease mappings. "
        "A generative model understands the disease mechanism and how it produces symptoms."
    )

    st.markdown("### From Generative to Discriminative: Bayes' Rule")

    st.markdown(
        "Given a generative model $p(x|y)$ and prior $p(y)$, we can compute the discriminative "
        "posterior via Bayes' rule:"
    )

    st.markdown(
        r"""
$$
p(y|x) = \frac{p(x|y) \, p(y)}{p(x)} = \frac{p(x|y) \, p(y)}{\sum_{y'} p(x|y') \, p(y')}
$$
"""
    )

    with st.expander("Worked Example: Spam Classification"):
        st.markdown(
            "**Prior $p(y)$:** Before reading any email, what fraction of all emails are spam? "
            "If historically 30% of emails on your server are spam, then $p(\\text{spam}) = 0.3$ and "
            "$p(\\text{ham}) = 0.7$. This is your baseline belief before seeing any content."
        )

        st.markdown(
            "**Likelihood $p(x|y)$:** If you looked at all spam emails ever, what fraction would contain "
            "the phrase 'Buy cheap watches now'? About 0.1% — spammy language is common in spam but not "
            "every spam says this exact thing, so $p(\\text{email}|\\text{spam}) = 0.001$. For legitimate "
            "emails, that fraction is vanishingly small: $p(\\text{email}|\\text{ham}) = 0.000001$. "
            "The key is the **ratio** — this phrase is 1000x more likely under spam than ham."
        )

        st.markdown(
            "**Posterior $p(y|x)$:** After reading the email, Bayes' rule updates your belief. "
            "The prior said 30% chance of spam, but the content is 1000x more consistent with spam than ham, "
            "so the posterior jumps to 99.8%. The evidence overwhelms the prior."
        )

        st.markdown("**Setup:**")
        st.markdown("- $p(\\text{spam}) = 0.3$, $p(\\text{ham}) = 0.7$")
        st.markdown("- Email text: *'Buy cheap watches now!'*")
        st.markdown("- $p(\\text{email} | \\text{spam}) = 0.001$")
        st.markdown("- $p(\\text{email} | \\text{ham}) = 0.000001$")
        st.markdown("")
        st.markdown("**Question:** Is this email spam?")
        st.markdown("")
        st.markdown("**Compute $p(\\text{spam} | \\text{email})$ via Bayes:**")
        st.markdown(
            r"""
$$
\begin{align}
p(\text{spam} | \text{email}) &= \frac{p(\text{email}|\text{spam}) \, p(\text{spam})}{p(\text{email}|\text{spam}) \, p(\text{spam}) + p(\text{email}|\text{ham}) \, p(\text{ham})} \\
&= \frac{0.001 \times 0.3}{0.001 \times 0.3 + 0.000001 \times 0.7} \\
&= \frac{0.0003}{0.0003 + 0.0000007} \\
&\approx 0.998
\end{align}
$$
"""
        )
        st.markdown("")
        st.success("**Answer:** 99.8% probability this is spam. The generative model's understanding of 'what spam looks like' gives us strong discriminative power.")

    st.markdown("### From Discriminative to Generative: Classifier Guidance")

    st.markdown(
        "Can we go the other direction? Given a discriminative classifier $p(y|x)$, can we generate samples "
        "from a class? This requires $p(x)$ (the prior over inputs), which a pure discriminative model doesn't provide."
    )

    st.markdown(
        "**Modern solution:** In diffusion models, we combine an unconditional score with a classifier gradient:"
    )

    st.markdown(
        r"""
$$
\nabla_x \log p(x|y) = \nabla_x \log p(x) + \nabla_x \log p(y|x)
$$
"""
    )

    st.markdown(
        "- $\\nabla_x \\log p(x)$ comes from the unconditional diffusion model (score network)"
    )
    st.markdown(
        "- $\\nabla_x \\log p(y|x)$ comes from the classifier gradient"
    )
    st.markdown(
        "Combine them during sampling to guide generation toward a specific class."
    )

    st.markdown("")
    st.markdown(
        "**Classifier-free guidance:** Train a single model both conditionally $p(x|y)$ and unconditionally $p(x)$. "
        "At sampling time, interpolate: $\\nabla_x \\log p(x|y) \\approx w \\cdot \\nabla_x \\log p(x|y) + (1-w) \\cdot \\nabla_x \\log p(x)$. "
        "Used in DALL-E, Stable Diffusion, and most modern text-to-image models."
    )

    st.info(
        "**Why This Duality Matters in Practice:**\n\n"
        "1. **Semi-supervised learning** — Use generative model $p(x|y)$ trained on limited labeled data + "
        "lots of unlabeled data (via $p(x)$) to improve discriminative performance.\n\n"
        "2. **Controllable generation** — Classifier-free guidance in DALL-E and Stable Diffusion. "
        "Combine discriminative control ('make it look like a sunset') with generative modeling.\n\n"
        "3. **Anomaly detection** — Model $p(x)$ for normal data (generative), flag low-likelihood samples, "
        "use discriminative classifier for refinement."
    )
