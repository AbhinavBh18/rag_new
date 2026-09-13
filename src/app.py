"""
Phase 6: User Interface (Streamlit)

Chat interface over the LangChain RAG pipeline. Handles session state, queries
the chain, and renders answers alongside their citations.

WHAT CHANGED
------------
* The chain is now conversational. Each browser session gets a UUID; that UUID
  keys a ChatSession in src/memory.py. Follow-up questions ("what about the
  other one?") now resolve correctly because the chain rewrites them using the
  history before retrieving.
* Two separate stores, deliberately:
    - st.session_state.messages  -> what we RENDER (includes error bubbles,
                                    sources, scores)
    - ChatSession                -> what we SEND to the LLM (clean Q/A pairs,
                                    trimmed to a sliding window)
  Keeping them apart means a rendering change can never corrupt the prompt.
* Sidebar exposes the retrieval mode so you can demo dense vs sparse vs hybrid
  side by side — useful in an interview, and it costs three lines.
* Citations now show which retriever surfaced each chunk (dense rank / sparse
  rank), which makes the hybrid search visible rather than a claim in a README.
"""

import os
import sys

import streamlit as st

# Must be the very first Streamlit command
st.set_page_config(
    page_title="PEFT Research Assistant",
    page_icon="📚",
    layout="centered",
)

# Add the project root directory to Python's path so 'src' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import MAX_HISTORY_TURNS, RETRIEVAL_MODE, TOP_K  # noqa: E402
from src.memory import clear_session, get_session, new_session_id  # noqa: E402
from src.pipeline import answer_question, get_query_engine  # noqa: E402


def render_sources(sources: list[dict]) -> None:
    """Render the citation expander for one assistant message."""
    if not sources:
        return
    with st.expander(f"View Citations ({len(sources)})"):
        for s in sources:
            provenance = []
            if s.get("rank_dense"):
                provenance.append(f"dense #{s['rank_dense']}")
            if s.get("rank_sparse"):
                provenance.append(f"BM25 #{s['rank_sparse']}")
            tag = f" — _via {', '.join(provenance)}_" if provenance else ""
            st.markdown(
                f"- **{s['title']}** (Page {s['page']}) "
                f"_[RRF: {s['score']:.4f}]_{tag}"
            )


# ──────────────────────────────────────────────
# Session bootstrap
# ──────────────────────────────────────────────
# One UUID per browser session. It dies with the session — no persistence
# across restarts, which is exactly the intended scope.
if "session_id" not in st.session_state:
    st.session_state.session_id = new_session_id()

if "retrieval_mode" not in st.session_state:
    st.session_state.retrieval_mode = RETRIEVAL_MODE

# The engine is expensive to build (loads the embedding model + BM25 index),
# so it is cached per retrieval mode and reused across reruns.


@st.cache_resource(show_spinner=False)
def load_engine(mode: str):
    return get_query_engine(mode=mode, top_k=TOP_K)


with st.spinner("Initializing AI, vector database, and BM25 index..."):
    try:
        engine = load_engine(st.session_state.retrieval_mode)
    except Exception as e:
        st.error(
            "Failed to load the pipeline. Check that GEMINI_API_KEY and "
            f"PINECONE_API_KEY are set, and that you have run "
            f"`python -m src.indexing`.\n\nError: {e}"
        )
        st.stop()

# Initialize chat history (display layer)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am your AI Research Assistant. Ask me anything about "
                "PEFT techniques like LoRA, QLoRA, Prefix-Tuning, or Adapters, "
                "and I will cite my sources. I remember this conversation, so "
                "follow-up questions work."
            ),
            "sources": [],
        }
    ]

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.subheader("Retrieval settings")
    mode = st.radio(
        "Search strategy",
        options=["hybrid", "dense", "sparse"],
        index=["hybrid", "dense", "sparse"].index(st.session_state.retrieval_mode),
        help=(
            "hybrid = dense embeddings + BM25 keywords, fused with Reciprocal "
            "Rank Fusion. dense = semantic only. sparse = keyword only."
        ),
    )
    if mode != st.session_state.retrieval_mode:
        st.session_state.retrieval_mode = mode
        st.rerun()

    st.caption(f"Top-K chunks per query: **{TOP_K}**")

    st.divider()
    st.subheader("Conversation")
    session = get_session(st.session_state.session_id)
    st.caption(
        f"Turns remembered: **{len(session)}** / {MAX_HISTORY_TURNS} "
        "(sliding window)"
    )
    st.caption(f"Session ID: `{st.session_state.session_id[:8]}…`")

    if st.button("Clear conversation", use_container_width=True):
        clear_session(st.session_state.session_id)
        st.session_state.session_id = new_session_id()
        st.session_state.messages = st.session_state.messages[:1]
        st.rerun()

    st.caption(
        "History lives in memory for this session only — it is not saved to "
        "disk and does not carry over to a new session."
    )

# ──────────────────────────────────────────────
# UI Layout
# ──────────────────────────────────────────────
st.title("📚 PEFT Research Assistant")
st.markdown(
    "Ask questions based on the downloaded LLM fine-tuning papers. "
    f"Currently searching in **{st.session_state.retrieval_mode}** mode."
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        render_sources(msg.get("sources", []))

# ──────────────────────────────────────────────
# Chat Input
# ──────────────────────────────────────────────
if prompt := st.chat_input("E.g., What is the difference between LoRA and Adapter modules?"):
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching papers and reading context..."):
            try:
                # Passing session_id is what makes this conversational: the
                # chain reads the history to rewrite the query, then appends
                # this turn to it.
                result = answer_question(
                    prompt,
                    session_id=st.session_state.session_id,
                    engine=engine,
                )
                ans_text = result["answer"]
                sources_list = result["sources"]

                st.write(ans_text)
                render_sources(sources_list)

                st.session_state.messages.append(
                    {"role": "assistant", "content": ans_text, "sources": sources_list}
                )

            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {e}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg, "sources": []}
                )
