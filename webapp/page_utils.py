"""Shared rendering utilities for weekly pages.

Each page imports render_week() to display its content,
keeping individual page files minimal.
"""

import streamlit as st

from webapp.course_manifest import COURSE_SCHEDULE, Session
from webapp.style import apply_style, WEEK_TITLES, session_type_badge


def setup_page(week_number: int):
    """Call first in any page. Sets page config + applies CSS + renders page header (week number + title)."""
    week_title = WEEK_TITLES.get(week_number, f"Week {week_number}")
    st.set_page_config(
        page_title=f"Week {week_number} - {week_title}",
        page_icon="~",
        layout="wide",
    )
    apply_style()

    # Page header
    st.markdown(f"##### Week {week_number}")
    st.markdown(f"# {week_title}")
    st.markdown("---")


def render_sessions(week_number: int):
    """Render the session blocks from the course manifest for a given week."""
    sessions = COURSE_SCHEDULE.get(week_number, [])

    if not sessions:
        st.markdown("*No sessions scheduled for this week.*")
        return

    for i, session in enumerate(sessions):
        _render_session(session)
        if i < len(sessions) - 1:
            st.markdown("---")


def render_assignments(modules: list[str] | None = None):
    """Render the assignment section. If modules provided, show which dgm.* modules to implement."""
    st.markdown("---")
    st.markdown("## Assignments")

    if modules is None:
        st.markdown(
            '<p class="caption">Coming soon -- assignments and exercises '
            "will appear here as they are developed.</p>",
            unsafe_allow_html=True,
        )
    else:
        # Show module badges
        modules_str = " &nbsp; ".join(f"`{m}`" for m in modules)
        st.markdown(f"**Implement these modules:** {modules_str}", unsafe_allow_html=True)

        # Show test command for the first module (simplified)
        if modules:
            test_path = modules[0].replace("dgm.", "test_").replace(".", "/")
            st.markdown(f"**Run tests:** `pytest tests/{test_path}/ -v`")


def render_mermaid(chart: str, height: int = 500):
    """Render a Mermaid diagram in the Streamlit page.

    Args:
        chart: Mermaid chart definition string (e.g., "graph TD; A-->B")
        height: Height of the rendered diagram container in pixels
    """
    import streamlit.components.v1 as components

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'neutral',
                themeVariables: {{
                    primaryColor: '#F5F5F4',
                    primaryTextColor: '#292524',
                    primaryBorderColor: '#D6D3D1',
                    lineColor: '#78716C',
                    secondaryColor: '#FAFAF9',
                    tertiaryColor: '#E7E5E4',
                    fontSize: '18px',
                    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
                }}
            }});
        </script>
        <style>
            .mermaid svg {{
                max-width: 100%;
                height: auto;
            }}
            .node rect, .node circle, .node polygon {{
                stroke-width: 2px !important;
            }}
        </style>
    </head>
    <body style="margin:0; padding:1rem 0; display:flex; justify-content:center; background:transparent;">
        <div class="mermaid" style="width:100%;">
{chart}
        </div>
    </body>
    </html>
    """
    components.html(html, height=height)


def render_week(week_number: int):
    """Render a complete week page (backward compatible)."""
    setup_page(week_number)
    render_sessions(week_number)
    render_assignments()
    from webapp.chat import render_chat
    render_chat(week_number)


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
