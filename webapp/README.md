# Webapp -- Course Tutorial Interface

A Streamlit webapp that serves as an interactive companion for MIT 6.S978. It maps weekly lectures and readings to the `dgm` package modules.

## Running

```bash
streamlit run webapp/app.py
```

## Structure

| File | Description |
|---|---|
| `app.py` | Entry point -- renders the course overview and weekly schedule table |
| `course_manifest.py` | Single source of truth for the week-to-code mapping, readings, and session metadata |
| `style.py` | Anthropic-inspired CSS styling, week titles, and badge helpers |
| `page_utils.py` | `render_week()` function that generates weekly pages from the manifest |
| `pages/` | One page per week (13 pages), each calling `render_week(n)` |
| `components/` | Reusable Streamlit components (placeholder) |
