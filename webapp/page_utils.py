"""Shared rendering utilities for weekly pages.

Each page imports render_week() to display its content,
keeping individual page files minimal.
"""

import streamlit as st

from webapp.course_manifest import COURSE_SCHEDULE, Session
from webapp.style import apply_style, WEEK_TITLES, session_type_badge


def render_week(week_number: int):
    """Render a complete week page from the course manifest."""
    week_title = WEEK_TITLES.get(week_number, f"Week {week_number}")
    st.set_page_config(
        page_title=f"Week {week_number} - {week_title}",
        page_icon="~",
        layout="wide",
    )
    apply_style()

    sessions = COURSE_SCHEDULE.get(week_number, [])

    # Page header
    st.markdown(f"##### Week {week_number}")
    st.markdown(f"# {week_title}")
    st.markdown("---")

    if not sessions:
        st.markdown("*No sessions scheduled for this week.*")
        return

    for i, session in enumerate(sessions):
        _render_session(session)
        if i < len(sessions) - 1:
            st.markdown("---")

    # Assignment placeholder
    st.markdown("---")
    st.markdown("## Assignments")
    st.markdown(
        '<p class="caption">Coming soon -- assignments and exercises '
        "will appear here as they are developed.</p>",
        unsafe_allow_html=True,
    )


def _render_session(session: Session):
    """Render a single session block."""
    # Session header with badge
    badge = session_type_badge(session.session_type)
    st.markdown(
        f'{badge} &nbsp; <span class="caption">{session.date}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f"### {session.title}")

    if session.description:
        st.markdown(session.description)

    # Slides link
    if session.slides_url:
        st.markdown(f"[View lecture slides]({session.slides_url})")

    # Package modules
    if session.dgm_modules:
        modules_str = ", ".join(f"`{m}`" for m in session.dgm_modules)
        st.markdown(f"**Package modules:** {modules_str}")

    # Readings
    if session.readings:
        required = [r for r in session.readings if not r.is_optional]
        optional = [r for r in session.readings if r.is_optional]

        st.markdown("#### Readings")

        if required:
            for r in required:
                st.markdown(
                    f"- [{r.title}]({r.url})  \n"
                    f'  <span class="caption">{r.authors} -- {r.venue} {r.year}</span>',
                    unsafe_allow_html=True,
                )

        if optional:
            with st.expander("Optional readings"):
                for r in optional:
                    st.markdown(
                        f"- [{r.title}]({r.url})  \n"
                        f'  <span class="caption">{r.authors} -- {r.venue} {r.year}</span>',
                        unsafe_allow_html=True,
                    )
