"""
Phase 4: Retrieval and Generation Pipeline  (LangChain)

Connects the hybrid retriever to Google Gemini and produces the conversational
Query Engine:
  1. Take a user question + the session's chat history.
  2. Rewrite the question into a standalone query (so follow-ups work).
  3. Retrieve top-K chunks via hybrid dense+BM25 search.
  4. Inject those chunks — WITH their citation metadata — into the prompt.
  5. Ask Gemini to answer strictly from that context.

MIGRATION NOTES (LlamaIndex -> LangChain)
-----------------------------------------
* GoogleGenAI(llm)            -> ChatGoogleGenerativeAI
* PromptTemplate + text_qa_template
                              -> ChatPromptTemplate with a system message and
                                 a MessagesPlaceholder for chat history.
* RetrieverQueryEngine        -> create_retrieval_chain(retriever, doc_chain),
                                 an LCEL Runnable. Invoke with a dict, get a
                                 dict back: {input, chat_history, context, answer}.
* Settings.llm global         -> gone. The LLM is passed explicitly, which
                                 removes the load-order hazard the old code
                                 flagged ("call build_or_load_index FIRST").
* response.source_nodes       -> result["context"] (a list[Document]).
                                 Scores now live in doc.metadata, since
                                 LangChain Documents have no `.score` field.
* NEW: create_history_aware_retriever handles conversational follow-ups.

CITATION FIX
------------
The old code told Gemini to cite "[Paper Title, Page X]" but explicitly
excluded that metadata from what the LLM saw (excluded_llm_metadata_keys), so
the model had to guess. Here, DOCUMENT_PROMPT stamps the title and page onto
every chunk as it is formatted into the context block — the model can only cite
what it can see.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

# Compat shim: these three helpers live in `langchain_classic` on LangChain 1.x
# and in `langchain` on 0.3.x. Try the new home first.
try:  # LangChain >= 1.0
    from langchain_classic.chains import (
        create_history_aware_retriever,
        create_retrieval_chain,
    )
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
except ImportError:  # LangChain 0.3.x
    from langchain.chains import (
        create_history_aware_retriever,
        create_retrieval_chain,
    )
    from langchain.chains.combine_documents import create_stuff_documents_chain

from src.config import (
    GEMINI_API_KEY,
    JUDGE_MODEL_NAME,
    JUDGE_TEMPERATURE,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    RETRIEVAL_MODE,
    TOP_K,
)
from src.memory import get_session
from src.retrieval import get_retriever

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────

# 1. Question rewriting. Runs BEFORE retrieval. Turns "and how does it compare
#    to adapters?" into "How does QLoRA compare to adapter modules?" so the
#    retriever gets something with actual semantic content.
CONTEXTUALIZE_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question which might reference "
    "context in the chat history, formulate a standalone question which can be "
    "understood without the chat history. Do NOT answer the question — just "
    "reformulate it if needed, and otherwise return it as is. Preserve any "
    "technical terms, model names, and acronyms exactly as written, since they "
    "are used for keyword search."
)

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 2. Answer generation. Same strict-researcher persona as the original prompt.
QA_SYSTEM_PROMPT = (
    "You are an expert AI research assistant specializing in LLM Fine-tuning.\n"
    "You have been provided with context from various research papers.\n"
    "Your goal is to answer the user's question clearly and accurately, "
    "strictly based on the provided context.\n"
    "\n"
    "CRITICAL RULES:\n"
    "1. Do not use outside knowledge. If the context does not contain the "
    "answer, say 'I cannot find the answer in the provided papers.'\n"
    "2. You MUST cite your sources for every major claim.\n"
    "3. Use the format [Paper Title, Page X] for citations, taking the title "
    "and page directly from the source header above each context passage.\n"
    "4. If the chat history is relevant, stay consistent with what you said "
    "earlier, but still ground every new claim in the context below.\n"
    "\n"
    "Context Information:\n"
    "---------------------\n"
    "{context}\n"
    "---------------------"
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 3. How each retrieved chunk is rendered inside {context}.
#    This is what makes citation possible — see CITATION FIX above.
DOCUMENT_PROMPT = PromptTemplate.from_template(
    "[Source: {paper_title} | Page {page_number}]\n{page_content}"
)


# ──────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────
def get_llm(model_name: str | None = None, temperature: float | None = ...) -> ChatGoogleGenerativeAI:
    """
    Initialize the Google Gemini chat model using the key from .env.

    `temperature` is passed only when set, because Gemini 3.x deprecated the
    sampling parameters and newer endpoints reject them.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Please add it to your .env file.")

    model_name = model_name or LLM_MODEL_NAME
    temp = LLM_TEMPERATURE if temperature is ... else temperature

    logger.info(f"Initializing LLM: {model_name}")
    kwargs: dict[str, Any] = {"model": model_name, "google_api_key": GEMINI_API_KEY}
    if temp is not None:
        kwargs["temperature"] = temp
    return ChatGoogleGenerativeAI(**kwargs)


def get_judge_llm() -> ChatGoogleGenerativeAI:
    """Deterministic model used by the evaluation harness."""
    return get_llm(model_name=JUDGE_MODEL_NAME, temperature=JUDGE_TEMPERATURE)


# ──────────────────────────────────────────────
# The chain
# ──────────────────────────────────────────────
def get_query_engine(
    mode: str = RETRIEVAL_MODE,
    top_k: int = TOP_K,
    llm: ChatGoogleGenerativeAI | None = None,
) -> Runnable:
    """
    Build the full conversational RAG chain.

    Returns an LCEL Runnable. Invoke it with:
        {"input": "...", "chat_history": [BaseMessage, ...]}
    and it returns:
        {"input", "chat_history", "context": list[Document], "answer": str}
    """
    llm = llm or get_llm()
    retriever = get_retriever(mode=mode, top_k=top_k)

    # Step 1: history-aware retrieval.
    # If chat_history is empty this passes `input` straight through to the
    # retriever (no wasted LLM call). If it isn't, it rewrites first.
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, CONTEXTUALIZE_PROMPT
    )

    # Step 2: "stuff" every retrieved doc into one prompt and generate.
    # ("Stuff" = concatenate. With TOP_K=5 chunks of 256 tokens we are nowhere
    # near Gemini's context limit, so map-reduce/refine would only add latency.)
    question_answer_chain = create_stuff_documents_chain(
        llm, QA_PROMPT, document_prompt=DOCUMENT_PROMPT
    )

    # Step 3: glue them together.
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


def format_sources(docs: list[Document]) -> list[dict[str, Any]]:
    """Flatten retrieved Documents into the shape the UI and eval both want."""
    sources = []
    for doc in docs:
        sources.append(
            {
                "title": doc.metadata.get("paper_title", "Unknown Paper"),
                "file_name": doc.metadata.get("file_name", "unknown.pdf"),
                "page": doc.metadata.get("page_number", "?"),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                # rrf_score for hybrid; dense/sparse modes have no fused score.
                "score": float(doc.metadata.get("rrf_score", 0.0)),
                "rank_dense": doc.metadata.get("rank_dense"),
                "rank_sparse": doc.metadata.get("rank_sparse"),
                "snippet": doc.page_content[:300],
            }
        )
    return sources


def answer_question(
    question: str,
    session_id: str | None = None,
    engine: Runnable | None = None,
) -> dict[str, Any]:
    """
    Convenience wrapper: run one turn, using and updating session memory.

    Passing session_id=None gives you a stateless, single-shot query — which is
    what the evaluation harness wants, so that each test question is scored
    independently.
    """
    engine = engine or get_query_engine()
    session = get_session(session_id) if session_id else None
    history = session.messages if session else []

    result = engine.invoke({"input": question, "chat_history": history})
    answer = result.get("answer", "")
    docs: list[Document] = result.get("context", [])

    if session:
        session.add_turn(question, answer)

    return {
        "question": question,
        "answer": answer,
        "documents": docs,
        "sources": format_sources(docs),
    }


# ──────────────────────────────────────────────
# Standalone execution for Phase 4 Verification
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("\nPhase 4: Retrieval and Generation Pipeline (LangChain + Hybrid)")
    print("=" * 60)

    try:
        engine = get_query_engine()
        session_id = "cli-demo"

        # Two turns, to prove conversation memory works: the second question
        # is meaningless without the first.
        for test_query in [
            "What is the core difference between LoRA and QLoRA?",
            "And which of the two uses less GPU memory?",
        ]:
            print(f"\n[Query]: {test_query}")
            print("Querying Gemini (this takes a few seconds)...\n")

            out = answer_question(test_query, session_id=session_id, engine=engine)

            print("=" * 60)
            print("[Gemini Answer]:")
            print("=" * 60)
            # Safe print for Windows consoles
            print(out["answer"].encode("ascii", "replace").decode("ascii"))

            print(f"\n[Sources Retrieved]: {len(out['sources'])}")
            for i, s in enumerate(out["sources"], 1):
                title = s["title"].encode("ascii", "replace").decode("ascii")
                print(
                    f"  {i}. {title} (Page {s['page']}) "
                    f"- RRF: {s['score']:.4f} "
                    f"[dense#{s['rank_dense']} sparse#{s['rank_sparse']}]"
                )

    except Exception as e:
        logger.error(f"Error during RAG pipeline execution: {e}")
