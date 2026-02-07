"""Shared styling for the MIT 6.S978 webapp.

Anthropic-inspired minimal aesthetic: muted palette, generous whitespace,
clean typography, and subtle dividers.
"""

import streamlit as st


def apply_style():
    """Apply custom CSS to the Streamlit app."""
    st.markdown("""
    <style>
        /* Base layout */
        .stApp {
            background-color: #FAFAF9;
        }

        .block-container {
            max-width: 56rem;
            padding-top: 2.5rem;
            padding-bottom: 4rem;
        }

        /* Typography */
        h1 {
            color: #292524 !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em !important;
            margin-bottom: 0.25rem !important;
        }

        h2 {
            color: #44403C !important;
            font-weight: 500 !important;
            letter-spacing: -0.01em !important;
            margin-top: 2.5rem !important;
        }

        h3 {
            color: #57534E !important;
            font-weight: 500 !important;
            margin-top: 1.5rem !important;
        }

        p, li {
            color: #57534E !important;
            line-height: 1.7 !important;
        }

        a {
            color: #6B8EAE !important;
            text-decoration: none !important;
        }

        a:hover {
            color: #57534E !important;
            text-decoration: underline !important;
        }

        /* Subtle dividers */
        hr {
            border: none !important;
            border-top: 1px solid #E7E5E4 !important;
            margin: 2rem 0 !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #F5F5F4;
            border-right: 1px solid #E7E5E4;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #44403C !important;
        }

        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] li {
            color: #78716C !important;
        }

        /* Tables */
        table {
            border-collapse: collapse;
            width: 100%;
        }

        th {
            background-color: #F5F5F4 !important;
            color: #44403C !important;
            font-weight: 500 !important;
            text-align: left !important;
            padding: 0.75rem 1rem !important;
            border-bottom: 1px solid #D6D3D1 !important;
            font-size: 0.85rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.04em !important;
        }

        td {
            color: #57534E !important;
            padding: 0.625rem 1rem !important;
            border-bottom: 1px solid #E7E5E4 !important;
            font-size: 0.925rem !important;
        }

        tr:hover td {
            background-color: #F5F5F4 !important;
        }

        /* Code blocks */
        code {
            color: #78716C !important;
            background-color: #F5F5F4 !important;
            padding: 0.15rem 0.4rem !important;
            border-radius: 3px !important;
            font-size: 0.85em !important;
        }

        /* Expanders */
        [data-testid="stExpander"] {
            border: 1px solid #E7E5E4 !important;
            border-radius: 4px !important;
            background-color: #FFFFFF !important;
        }

        /* Metric labels */
        [data-testid="stMetricLabel"] {
            color: #78716C !important;
        }

        /* Caption text */
        .caption {
            color: #A8A29E;
            font-size: 0.8rem;
            line-height: 1.5;
        }

        /* Session type badges */
        .badge {
            display: inline-block;
            padding: 0.15rem 0.5rem;
            border-radius: 3px;
            font-size: 0.75rem;
            font-weight: 500;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }

        .badge-lecture {
            background-color: #E0E7FF;
            color: #4338CA;
        }

        .badge-reading {
            background-color: #FEF3C7;
            color: #92400E;
        }

        .badge-guest {
            background-color: #D1FAE5;
            color: #065F46;
        }

        .badge-holiday {
            background-color: #F5F5F4;
            color: #A8A29E;
        }
    </style>
    """, unsafe_allow_html=True)


# Week titles for navigation and page headers
WEEK_TITLES: dict[int, str] = {
    1: "Introduction",
    2: "Image Priors & VAE",
    3: "Normalizing Flows & AR Models",
    4: "Autoregressive Models",
    5: "AR + Diffusion & GAN",
    6: "GAN Deep Dive",
    7: "EBM, Score Matching & Diffusion",
    8: "Diffusion Models",
    9: "Discrete Diffusion & Flow Matching",
    10: "Flow Matching",
    11: "Applications: Video & 3D",
    12: "Applications: Robotics",
    13: "Applications: Materials & Biology",
}


def session_type_badge(session_type: str) -> str:
    """Return an HTML badge for a session type."""
    labels = {
        "lecture": ("Lecture", "badge-lecture"),
        "reading": ("Reading", "badge-reading"),
        "guest_lecture": ("Guest Lecture", "badge-guest"),
        "holiday": ("Holiday", "badge-holiday"),
    }
    label, css_class = labels.get(session_type, (session_type, "badge-reading"))
    return f'<span class="badge {css_class}">{label}</span>'
