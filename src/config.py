"""
Centralized configuration for the RAG pipeline.

All tunable parameters live here so experiments (chunk sizes, top-k, hybrid
weights) only require edits in one file.

CHANGED IN THIS REFACTOR
------------------------
* Added Qdrant settings (replaces ChromaDB).
* Added hybrid-retrieval settings (dense k, sparse k, RRF constant, weights).
* Added conversation-memory settings.
* Added evaluation settings (judge model, k-cutoffs).
* CHUNK_SIZE is now measured with the *embedding model's own tokenizer*
  instead of LlamaIndex's default token counter — see note below.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
# The chunk store is a plain JSONL file holding every chunk we ever upserted.
# WHY: Qdrant is a *vector* database — you can scroll points out of it, but
# document back out of it, but BM25 needs the full corpus in memory to build
# its term-frequency statistics. So the chunk store is the shared source of
# truth for the sparse half of hybrid retrieval. (ChromaDB used to double as
# this local docstore; a vector DB should not be relied on for it.)
CHUNK_STORE_PATH = PROCESSED_DIR / "chunks.jsonl"

EVAL_DIR = PROJECT_ROOT / "eval"
EVAL_RESULTS_DIR = EVAL_DIR / "results"

# ──────────────────────────────────────────────
# Gemini API
# ──────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6JcgLBT-EJu0P8KsKWccYqoDt0oC4Rmj3nIL4cDvCh8gg")
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gemini-3.5-flash")

# NOTE: Google deprecated `temperature` / `top_p` / `top_k` on the newest
# Gemini 3.x endpoints. Set LLM_TEMPERATURE to an empty string in .env to omit
# the parameter entirely if your model rejects it.
_raw_temp = os.getenv("LLM_TEMPERATURE", "0.1")
LLM_TEMPERATURE: float | None = float(_raw_temp) if _raw_temp.strip() else None

# A separate (cheap, deterministic) model used as the LLM-as-judge in Phase 5.
# Keeping the judge distinct from the generator avoids self-preference bias.
JUDGE_MODEL_NAME: str = os.getenv("JUDGE_MODEL_NAME", "gemini-3.5-flash")
JUDGE_TEMPERATURE: float | None = 0.0

# ──────────────────────────────────────────────
# Embedding Model (local — sentence-transformers)
# ──────────────────────────────────────────────
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384  # Output dimension for all-MiniLM-L6-v2

# ──────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────
# IMPORTANT FIX: all-MiniLM-L6-v2 has a hard max sequence length of 256 tokens.
# The old config used 512-token chunks, which meant roughly the second half of
# every chunk was silently truncated before embedding — the text was retrievable
# only by whatever happened to be in its first 256 tokens. We now chunk at 256
# tokens using the model's own tokenizer so nothing is dropped.
CHUNK_SIZE: int = 256   # tokens per chunk (measured with the MiniLM tokenizer)
CHUNK_OVERLAP: int = 40  # token overlap between consecutive chunks

# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────
TOP_K: int = 5           # final number of chunks handed to the LLM
DENSE_TOP_K: int = 10    # candidates pulled from Qdrant before fusion
SPARSE_TOP_K: int = 10   # candidates pulled from BM25 before fusion

# Reciprocal Rank Fusion constant. Standard value from the original RRF paper
# (Cormack et al., 2009). Larger k => flatter curve => deep ranks matter more.
RRF_K: int = 60

# Relative trust in each retriever during fusion. Dense is weighted slightly
# higher because paraphrased questions ("how does it save memory?") do not
# share vocabulary with the papers; BM25 earns its keep on exact identifiers
# ("NF4", "rank r=8", "IA3").
HYBRID_DENSE_WEIGHT: float = 0.6
HYBRID_SPARSE_WEIGHT: float = 0.4

RETRIEVAL_MODE: str = os.getenv("RETRIEVAL_MODE", "hybrid")  # hybrid | dense | sparse

# ──────────────────────────────────────────────
# Qdrant (replaces ChromaDB; chosen over Pinecone for Python 3.14 support)
# ──────────────────────────────────────────────
# Qdrant runs in two modes and the code supports both:
#   * EMBEDDED  (QDRANT_URL empty) — qdrant-client writes to a local folder.
#     No server, no Docker, no API key. This is the default.
#   * SERVER    (QDRANT_URL set)   — a real Qdrant instance, either
#     `docker run -p 6333:6333 qdrant/qdrant` or Qdrant Cloud.
# The only reason to move to server mode is concurrency — see the note in
# indexing.py about the embedded-mode directory lock.
QDRANT_URL: str = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")  # Qdrant Cloud only
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant_db"      # embedded-mode storage
QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "peft_papers")
QDRANT_DISTANCE: str = "COSINE"
UPSERT_BATCH_SIZE: int = 100

# ──────────────────────────────────────────────
# Conversation memory (session-scoped only)
# ──────────────────────────────────────────────
# Number of past *turns* (1 turn = user msg + assistant msg) kept in context.
# Bounded so the prompt cannot grow without limit over a long chat.
MAX_HISTORY_TURNS: int = 6

# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
EVAL_K_CUTOFFS: tuple[int, ...] = (1, 3, 5, 10)
EVAL_RETRIEVAL_CANDIDATES: int = 10  # depth to score retrieval metrics at
EVAL_MODES: tuple[str, ...] = ("dense", "sparse", "hybrid")
