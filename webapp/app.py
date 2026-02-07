"""MIT 6.S978 Deep Generative Models -- Course Tutorial

Entry point: streamlit run webapp/app.py
"""

import streamlit as st

from webapp.course_manifest import COURSE_SCHEDULE
from webapp.style import apply_style, WEEK_TITLES, session_type_badge

st.set_page_config(
    page_title="MIT 6.S978 Deep Generative Models",
    page_icon="~",
    layout="wide",
)
apply_style()

# -- Header -------------------------------------------------------------------

st.markdown("# MIT 6.S978")
st.markdown("### Deep Generative Models, Fall 2024")

st.markdown(
    "A hands-on learning companion for the MIT graduate course on deep "
    "generative modeling. Each week pairs lecture material and readings with "
    "runnable PyTorch implementations in the `dgm` package -- bridging the "
    "gap between academic theory and industry practice."
)

st.markdown("---")

# -- Course overview table -----------------------------------------------------

st.markdown("## Weekly Schedule")

# Build the overview table
rows: list[str] = []
for week_num in sorted(COURSE_SCHEDULE.keys()):
    sessions = COURSE_SCHEDULE[week_num]
    title = WEEK_TITLES.get(week_num, f"Week {week_num}")

    # Collect session types as badges
    badges = " ".join(
        session_type_badge(s.session_type) for s in sessions
    )

    # Collect unique dgm modules across all sessions
    modules = sorted(
        {m for s in sessions for m in s.dgm_modules}
    )
    modules_str = ", ".join(f"<code>{m}</code>" for m in modules) if modules else ""

    # Date range
    dates = [s.date for s in sessions]
    if len(dates) == 1:
        date_str = dates[0]
    else:
        date_str = f"{dates[0]} / {dates[-1]}"

    rows.append(
        f"<tr>"
        f"<td style='white-space:nowrap; font-weight:500;'>Week {week_num}</td>"
        f"<td>{title}</td>"
        f"<td>{badges}</td>"
        f"<td>{modules_str}</td>"
        f"<td class='caption'>{date_str}</td>"
        f"</tr>"
    )

table_html = f"""
<table>
    <thead>
        <tr>
            <th>Week</th>
            <th>Topic</th>
            <th>Sessions</th>
            <th>Package Modules</th>
            <th>Dates</th>
        </tr>
    </thead>
    <tbody>
        {"".join(rows)}
    </tbody>
</table>
"""
st.markdown(table_html, unsafe_allow_html=True)

st.markdown("---")

# -- Getting started -----------------------------------------------------------

st.markdown("## Getting Started")

st.code(
    """# Clone the repository
git clone <repo-url>
cd S978-Deep-Generative-Models-2024

# Install the dgm package (editable)
pip install -e .

# Launch the tutorial webapp
streamlit run webapp/app.py""",
    language="bash",
)

st.markdown("---")

# -- Links ---------------------------------------------------------------------

st.markdown("## Links")

st.markdown(
    "- [Course website & schedule](https://mit-6s978.github.io/schedule.html)\n"
    "- [MIT 6.S978 homepage](https://mit-6s978.github.io/)"
)

# -- Footer -------------------------------------------------------------------

st.markdown("---")
st.markdown(
    '<p class="caption">'
    "This is a personal learning project, not an official MIT resource. "
    "All course materials belong to their respective authors."
    "</p>",
    unsafe_allow_html=True,
)
