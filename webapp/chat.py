"""RAG chat assistant for weekly course pages.

Chunks course content (slides, code, readings, session overviews),
embeds with sentence-transformers, stores in ChromaDB, and retrieves
relevant context for Claude-powered Q&A.
"""

from __future__ import annotations

import os
import pathlib
import tempfile
from dataclasses import dataclass, field

import streamlit as st

# ---------------------------------------------------------------------------
# .env loading — best-effort, no crash if python-dotenv is missing
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    _repo_root = pathlib.Path(__file__).resolve().parent.parent
    load_dotenv(_repo_root / ".env")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    """A single piece of indexed content."""

    id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Content extractors
# ---------------------------------------------------------------------------
def _extract_slides_chunks(session_title: str, slides_url: str) -> list[Chunk]:
    """Download a PDF and return one chunk per page via PyMuPDF."""
    try:
        import urllib.request

        import fitz  # PyMuPDF
    except ImportError:
        return []

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            urllib.request.urlretrieve(slides_url, tmp.name)
            tmp_path = tmp.name

        doc = fitz.open(tmp_path)
        chunks: list[Chunk] = []
        for page_num in range(len(doc)):
            text = doc[page_num].get_text().strip()
            if len(text) < 20:
                continue
            chunks.append(
                Chunk(
                    id=f"slides:{session_title}:p{page_num + 1}",
                    text=f"[Slide page {page_num + 1} — {session_title}]\n{text}",
                    metadata={
                        "source": "slides",
                        "session": session_title,
                        "page": str(page_num + 1),
                        "url": slides_url,
                    },
                )
            )
        doc.close()
        return chunks
    except Exception:
        return []
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _extract_code_chunks(dgm_modules: list[str]) -> list[Chunk]:
    """Return one chunk per .py file for each dgm module (skip __init__.py)."""
    src_root = pathlib.Path(__file__).resolve().parent.parent / "src"
    chunks: list[Chunk] = []
    for module in dgm_modules:
        # e.g. "dgm.vae" -> src/dgm/vae
        parts = module.split(".")
        module_dir = src_root / pathlib.Path(*parts)
        if not module_dir.is_dir():
            continue
        for py_file in sorted(module_dir.glob("*.py")):
            if py_file.name == "__init__.py":
                continue
            text = py_file.read_text(encoding="utf-8")
            if not text.strip():
                continue
            rel = py_file.relative_to(src_root)
            chunks.append(
                Chunk(
                    id=f"code:{rel}",
                    text=f"[Source file: {rel}]\n{text}",
                    metadata={
                        "source": "code",
                        "module": module,
                        "file": str(rel),
                    },
                )
            )
    return chunks


def _extract_reading_chunks(session_title: str, readings) -> list[Chunk]:
    """One chunk per paper: title + authors + venue + year + URL."""
    chunks: list[Chunk] = []
    for r in readings:
        text = (
            f"[Reading — {session_title}]\n"
            f"Title: {r.title}\n"
            f"Authors: {r.authors}\n"
            f"Venue: {r.venue} {r.year}\n"
            f"URL: {r.url}\n"
            f"Optional: {'yes' if r.is_optional else 'no'}"
        )
        chunks.append(
            Chunk(
                id=f"reading:{r.title[:60]}",
                text=text,
                metadata={
                    "source": "reading",
                    "session": session_title,
                    "title": r.title,
                    "url": r.url,
                },
            )
        )
    return chunks


def _extract_session_overview_chunk(session) -> Chunk:
    """One chunk with session title, description, and module list."""
    modules_str = ", ".join(session.dgm_modules) if session.dgm_modules else "none"
    text = (
        f"[Session overview]\n"
        f"Title: {session.title}\n"
        f"Date: {session.date}\n"
        f"Type: {session.session_type}\n"
        f"Description: {session.description}\n"
        f"Package modules: {modules_str}"
    )
    return Chunk(
        id=f"overview:{session.title}",
        text=text,
        metadata={"source": "overview", "session": session.title},
    )


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------
def _chroma_dir(week_number: int) -> pathlib.Path:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    return repo_root / "data" / "chroma" / f"week_{week_number:02d}"


def _get_collection(week_number: int):
    """Return a ChromaDB collection, creating it if needed."""
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    client = chromadb.PersistentClient(path=str(_chroma_dir(week_number)))
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return client.get_or_create_collection(
        name=f"week_{week_number:02d}",
        embedding_function=ef,
    )


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------
def build_index(week_number: int, force_rebuild: bool = False) -> int:
    """Build (or rebuild) the ChromaDB index for a given week.

    Returns the number of chunks indexed.
    """
    from webapp.course_manifest import COURSE_SCHEDULE

    collection = _get_collection(week_number)

    if collection.count() > 0 and not force_rebuild:
        return collection.count()

    # Clear existing data on rebuild
    if collection.count() > 0:
        collection.delete(where={"source": {"$ne": ""}})

    sessions = COURSE_SCHEDULE.get(week_number, [])
    all_chunks: list[Chunk] = []

    for session in sessions:
        # Session overview
        if session.description:
            all_chunks.append(_extract_session_overview_chunk(session))

        # Slides
        if session.slides_url:
            all_chunks.extend(
                _extract_slides_chunks(session.title, session.slides_url)
            )

        # Code
        if session.dgm_modules:
            all_chunks.extend(_extract_code_chunks(session.dgm_modules))

        # Readings
        if session.readings:
            all_chunks.extend(
                _extract_reading_chunks(session.title, session.readings)
            )

    if not all_chunks:
        return 0

    # Upsert in batches (ChromaDB limit)
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        collection.upsert(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
        )

    return len(all_chunks)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve_context(
    week_number: int, query: str, top_k: int = 6
) -> list[dict]:
    """Query the week's ChromaDB collection and return top-k results.

    Returns list of dicts with keys: id, document, metadata, distance.
    """
    collection = _get_collection(week_number)
    if collection.count() == 0:
        return []

    n = min(top_k, collection.count())
    results = collection.query(query_texts=[query], n_results=n)

    items: list[dict] = []
    for i in range(len(results["ids"][0])):
        items.append(
            {
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )
    return items


# ---------------------------------------------------------------------------
# Claude call
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are a knowledgeable teaching assistant for MIT 6.S978: Deep Generative Models.

Rules:
- Answer questions using ONLY the provided context from the course materials.
- If the context does not contain enough information, say so honestly.
- Cite sources (e.g. "According to slide page 5..." or "In the paper by X et al.").
- Use LaTeX notation ($...$) for mathematical expressions.
- Keep answers clear and concise, aimed at someone learning generative models.
- When discussing code, reference the specific file paths.
"""


def _call_claude(
    query: str,
    context: list[dict],
    week_number: int,
    history: list[dict],
) -> str:
    """Call Claude with retrieved context and conversation history."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "Error: ANTHROPIC_API_KEY not set."

    # Build context string
    context_parts = []
    for item in context:
        source = item["metadata"].get("source", "unknown")
        context_parts.append(f"[{source}] {item['document']}")
    context_str = "\n\n---\n\n".join(context_parts)

    # Build messages — last 6 turns of history + current query
    messages: list[dict] = []
    recent_history = history[-6:] if len(history) > 6 else history
    for msg in recent_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Current user message with context
    user_message = (
        f"Context from Week {week_number} materials:\n\n"
        f"{context_str}\n\n"
        f"---\n\n"
        f"Student question: {query}"
    )
    messages.append({"role": "user", "content": user_message})

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def render_chat(week_number: int) -> None:
    """Render the RAG chat assistant widget for a week page."""
    st.markdown("---")
    st.markdown("## Ask the Course Assistant")

    history_key = f"chat_history_week_{week_number}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    # Check if dependencies are available
    try:
        import chromadb  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError:
        st.info(
            'Chat dependencies not installed. Run `pip install -e ".[chat]"` '
            "to enable the course assistant."
        )
        return

    # Check if index exists
    collection = _get_collection(week_number)
    index_exists = collection.count() > 0

    # --- State: no index ---
    if not index_exists:
        st.info("The search index for this week has not been built yet.")
        if st.button("Build Index", key=f"build_idx_{week_number}"):
            with st.spinner("Building index (downloading slides, indexing content)..."):
                count = build_index(week_number, force_rebuild=True)
            st.success(f"Index built with {count} chunks.")
            st.rerun()
        return

    # --- State: no API key ---
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.warning(
            "Set your `ANTHROPIC_API_KEY` environment variable or add it to a "
            "`.env` file in the repository root to enable the chat assistant."
        )
        return

    # --- State: ready ---
    # Sidebar: rebuild button
    with st.sidebar:
        if st.button("Rebuild Index", key=f"rebuild_idx_{week_number}"):
            with st.spinner("Rebuilding..."):
                count = build_index(week_number, force_rebuild=True)
            st.success(f"Rebuilt with {count} chunks.")
            st.rerun()

    # Display chat history
    for msg in st.session_state[history_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input(f"Ask about Week {week_number} materials..."):
        # Show user message
        st.session_state[history_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve context
        with st.spinner("Searching course materials..."):
            results = retrieve_context(week_number, prompt)

        # Show sources in expander
        if results:
            with st.expander(f"Sources ({len(results)} chunks retrieved)"):
                for item in results:
                    src = item["metadata"].get("source", "unknown")
                    label = item["id"]
                    distance = item["distance"]
                    st.markdown(
                        f"- **[{src}]** `{label}` (distance: {distance:.3f})"
                    )

        # Call Claude
        with st.spinner("Thinking..."):
            response = _call_claude(
                prompt,
                results,
                week_number,
                st.session_state[history_key][:-1],  # exclude current user msg
            )

        # Display and store assistant response
        st.session_state[history_key].append(
            {"role": "assistant", "content": response}
        )
        with st.chat_message("assistant"):
            st.markdown(response)
