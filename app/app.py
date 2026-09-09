"""
app.py -: Streamlit frontend.

Run with:  streamlit run app.py

WHAT IT DOES:
  - Clean chat interface for fitness/nutrition Q&A
  - Shows retrieved citations in expandable sections below each answer
  - Sidebar: upload new documents, rebuild index, show RAGAS scores
  - Session state preserves conversation history
  - Displays fallback warning when retrieval confidence is low

INTERVIEW DEFENCE:
  "Why Streamlit over Flask/FastAPI?"
  → Streamlit lets you build a working demo in one file with no HTML.
    For a portfolio RAG project, the goal is to demo the ML pipeline,
    not showcase web dev. FastAPI would be the right choice for a
    production API with proper auth and horizontal scaling.

  "How does session state work here?"
  → st.session_state persists data across Streamlit reruns (which happen
    on every user interaction). We store chat_history as a list of dicts
    and the Retriever object (so it's not re-initialised on every query).
"""

import os
import shutil
import json
import streamlit as st


from ingest import load_documents
from chunk import chunk_documents, save_chunks
from embed import build_and_save_index
from retrieve import Retriever
from generate import generate_answer

from embed import BASE_DIR
from ingest import DATA_RAW
from generate import FALLBACK_MESSAGE
from eval import LOG_DIR
LOG_DIR    = BASE_DIR / "logs"

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FitRAG — Fitness & Nutrition Assistant",
    page_icon="💪",
    layout="wide",
)


# ── Session state init ─────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None


# ── Helper: load or reload retriever ──────────────────────────────────────
@st.cache_resource(show_spinner="Loading retrieval models…")
def get_retriever() -> Retriever:
    """
    Load once, cache across all sessions.
    st.cache_resource keeps this in memory — we don't reload the
    embedding model + FAISS index on every query.
    """
    return Retriever()


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    # Document upload
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload fitness/nutrition docs (.txt)",
        type=["txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for f in uploaded_files:
            dest = DATA_RAW / f.name
            dest.write_bytes(f.read())
        st.success(f"Uploaded {len(uploaded_files)} file(s) to data/raw/")

    # Rebuild index
    if st.button("🔄 Rebuild Index", help="Re-ingest, chunk, embed, and index all documents"):
        with st.spinner("Rebuilding index… this may take a minute."):
            try:
                pages  = load_documents()
                chunks = chunk_documents(pages)
                save_chunks(chunks)
                build_and_save_index()
                # Clear the cached retriever so it reloads fresh
                st.cache_resource.clear()
                st.session_state.retriever = None
                st.success("✓ Index rebuilt successfully!")
            except Exception as e:
                st.error(f"Index rebuild failed: {e}")

    st.markdown("---")

    # RAGAS scores
    st.subheader("📊 Eval Scores")
    ragas_file = LOG_DIR / "ragas_results.json"
    if ragas_file.exists():
        with open(ragas_file) as f:
            results = json.load(f)
        # scores = results.get("scores", {})
        scores = results.get("summary", {})
        for metric, score in scores.items():
            label = metric.replace("_", " ").title()
            st.metric(label=label, value=f"{score:.3f}")
        st.caption(f"Based on {results.get('num_pairs', '?')} eval pairs")
    else:
        st.info("No eval results yet. Run `python eval.py` to generate.")

    st.markdown("---")
    st.caption("FitRAG · Portfolio RAG Project · Built with FAISS + BM25 + OpenAI")


# ── Main UI ────────────────────────────────────────────────────────────────
st.title("💪 FitRAG — Fitness & Nutrition Assistant")
st.caption(
    "Ask questions about exercise, nutrition, meal planning, and healthy living. "
    "Answers are grounded in uploaded documents with source citations."
)

# Load retriever (from cache or fresh)
try:
    retriever = get_retriever()
except FileNotFoundError:
    st.warning(
        "⚠️ No index found. Upload documents and click **Rebuild Index** in the sidebar, "
        "or run `python chunk.py && python embed.py` to build from the command line."
    )
    st.stop()

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander(f"📚 {len(msg['citations'])} Source(s) Used"):
                for c in msg["citations"]:
                    st.markdown(
                        f"**{c['source']}** — Page {c['page']}  "
                        f"*(relevance: {c['rrf_score']:.3f})*"
                    )
                    st.caption(c["excerpt"])
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask about fitness or nutrition…"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer…"):
            chunks   = retriever.query(prompt)
            response = generate_answer(prompt, chunks)

        if response.used_fallback:
            st.warning("⚠️ Low retrieval confidence — showing fallback response.")

        st.markdown(response.answer)

        citations_dicts = []
        if response.citations:
            with st.expander(f"📚 {len(response.citations)} Source(s) Used"):
                for c in response.citations:
                    st.markdown(
                        f"**{c.source}** — Page {c.page}  "
                        f"*(relevance: {c.rrf_score:.3f})*"
                    )
                    st.caption(c.excerpt)
                    st.divider()
                    citations_dicts.append({
                        "source"   : c.source,
                        "page"     : c.page,
                        "excerpt"  : c.excerpt,
                        "rrf_score": c.rrf_score,
                    })

    # Save to history
    st.session_state.chat_history.append({
        "role"     : "assistant",
        "content"  : response.answer,
        "citations": citations_dicts,
    })
