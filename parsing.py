"""
Phase 2: PDF Parsing and Chunking  (LangChain)

Loads PDFs from data/papers/, extracts text page-by-page, and splits it into
overlapping chunks. Every chunk carries the metadata needed for citations
(file_name, paper_title, page_number) plus a deterministic `chunk_id`.

MIGRATION NOTES (LlamaIndex -> LangChain)
-----------------------------------------
* SimpleDirectoryReader   -> PyPDFLoader (one loader per file, so we control
                             the metadata that lands on each page).
* Document / TextNode     -> langchain_core.documents.Document (one type for
                             both loaded pages and split chunks — LangChain has
                             no separate "node" concept).
* SentenceSplitter        -> RecursiveCharacterTextSplitter built from the
                             embedding model's HuggingFace tokenizer, so
                             `chunk_size` really means MiniLM tokens.
* excluded_embed_metadata_keys / excluded_llm_metadata_keys
                          -> not needed. LangChain never auto-prepends metadata
                             to the embedded text; metadata reaches the LLM only
                             via the explicit `document_prompt` in pipeline.py.
                             This gives us the same token savings by default.
* NEW: chunks are persisted to a JSONL chunk store so the BM25 (sparse) half of
  hybrid retrieval can be rebuilt without re-parsing the PDFs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNK_STORE_PATH,
    EMBEDDING_MODEL_NAME,
    PAPERS_DIR,
)

logger = logging.getLogger(__name__)

# Helper to map filename to clean title
PAPER_TITLES: dict[str, str] = {
    "lora.pdf": "LoRA: Low-Rank Adaptation of Large Language Models",
    "qlora.pdf": "QLoRA: Efficient Finetuning of Quantized Language Models",
    "prefix_tuning.pdf": "Prefix-Tuning: Optimizing Continuous Prompts for Generation",
    "adapters_houlsby.pdf": "Parameter-Efficient Transfer Learning for NLP (Adapters)",
    "p_tuning.pdf": "GPT Understands, Too (P-Tuning)",
    "p_tuning_v2.pdf": "P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning",
    "ia3_t_few.pdf": "Few-Shot Parameter-Efficient Fine-Tuning (IA3)",
    "prompt_tuning_lester.pdf": "The Power of Scale for Parameter-Efficient Prompt Tuning",
}


def _build_splitter() -> RecursiveCharacterTextSplitter:
    """
    Build a token-aware recursive splitter.

    RecursiveCharacterTextSplitter counts *characters* by default. We swap in
    the MiniLM tokenizer so CHUNK_SIZE is expressed in the same units the
    embedding model uses, guaranteeing no chunk exceeds its 256-token window.

    Falls back to a character-based approximation (~4 chars/token) if
    `transformers` is unavailable, so the pipeline still runs.
    """
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            # Recursive separators: try paragraph, then line, then sentence,
            # then word. This is LangChain's equivalent of LlamaIndex's
            # SentenceSplitter "keep sentences intact" behaviour.
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning(
            "Could not load HF tokenizer (%s). Falling back to character-based "
            "splitting at ~4 chars/token.", exc
        )
        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE * 4,
            chunk_overlap=CHUNK_OVERLAP * 4,
            separators=["\n\n", "\n", ". ", " ", ""],
        )


def clean_metadata(doc: Document, file_name: str) -> None:
    """
    Normalise metadata on a freshly loaded page.

    PyPDFLoader gives us `source` (full path) and `page` (0-indexed int).
    We convert those into the citation-friendly fields the prompt expects and
    drop everything noisy, so the metadata payload stays small — Pinecone
    charges by stored metadata size, and small payloads keep upserts fast.
    """
    page_index = doc.metadata.get("page", 0)
    try:
        page_number = int(page_index) + 1  # PyPDF is 0-indexed; humans are not
    except (TypeError, ValueError):
        page_number = 1

    doc.metadata = {
        "file_name": file_name,
        "paper_title": PAPER_TITLES.get(file_name, file_name),
        "page_number": page_number,
    }


def load_and_chunk_pdfs(data_dir: Path = PAPERS_DIR) -> list[Document]:
    """
    Load all PDFs in the directory and split them into overlapping chunks.

    Returns:
        A list of LangChain Document objects, each with a unique `chunk_id`.
    """
    if not data_dir.exists() or not list(data_dir.glob("*.pdf")):
        raise FileNotFoundError(f"No PDFs found in {data_dir}. Run Phase 1 first.")

    logger.info(f"Loading PDFs from {data_dir}...")

    # 1. Read documents, one file at a time so metadata is unambiguous.
    pages: list[Document] = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf_path))
        file_pages = loader.load()
        for page in file_pages:
            clean_metadata(page, pdf_path.name)
        pages.extend(file_pages)
        logger.info(f"  {pdf_path.name}: {len(file_pages)} pages")

    logger.info(f"Loaded {len(pages)} total pages across all PDFs.")

    # 2. Split into chunks.
    splitter = _build_splitter()
    logger.info(f"Chunking with size={CHUNK_SIZE} tokens, overlap={CHUNK_OVERLAP}...")
    chunks = splitter.split_documents(pages)

    # 3. Assign deterministic IDs.
    # WHY deterministic: Pinecone upserts are keyed by ID, so re-running the
    # build overwrites the same vectors instead of creating duplicates. It also
    # gives the hybrid fusion step a stable key to dedupe on, and lets the eval
    # harness talk about "chunk X" reproducibly across runs.
    per_page_counter: dict[tuple[str, int], int] = {}
    for chunk in chunks:
        key = (chunk.metadata["file_name"], chunk.metadata["page_number"])
        idx = per_page_counter.get(key, 0)
        per_page_counter[key] = idx + 1
        chunk.metadata["chunk_id"] = f"{key[0]}::p{key[1]:03d}::c{idx:02d}"

    logger.info(f"Created {len(chunks)} chunks from the source PDFs.")
    return chunks


# ──────────────────────────────────────────────
# Chunk store (needed by BM25 — see config.py)
# ──────────────────────────────────────────────
def save_chunk_store(chunks: list[Document], path: Path = CHUNK_STORE_PATH) -> Path:
    """Persist chunks as JSONL so BM25 can be rebuilt without re-parsing PDFs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(
                json.dumps(
                    {"page_content": chunk.page_content, "metadata": chunk.metadata},
                    ensure_ascii=False,
                )
                + "\n"
            )
    logger.info(f"Wrote {len(chunks)} chunks to chunk store: {path}")
    return path


def load_chunk_store(path: Path = CHUNK_STORE_PATH) -> list[Document]:
    """Read the JSONL chunk store back into Documents."""
    if not path.exists():
        raise FileNotFoundError(
            f"Chunk store missing at {path}. Run `python -m src.indexing` to build it."
        )
    docs: list[Document] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                docs.append(
                    Document(page_content=row["page_content"], metadata=row["metadata"])
                )
    logger.info(f"Loaded {len(docs)} chunks from chunk store.")
    return docs


# ──────────────────────────────────────────────
# Standalone execution for Phase 2 Verification
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("\nPhase 2: PDF Parsing and Chunking (LangChain)")
    print("=" * 60)

    try:
        chunks = load_and_chunk_pdfs()
        print(f"\nSuccessfully created {len(chunks)} chunks.")

        if chunks:
            sample = chunks[5]  # Pick a chunk slightly into the paper
            print("\nSample Chunk Metadata:")
            for k, v in sample.metadata.items():
                print(f"  {k}: {v}")

            text = sample.page_content[:300]
            # Safe print for Windows consoles
            print(f"  {text.encode('ascii', 'replace').decode('ascii')}...\n")

    except Exception as e:
        logger.error(f"Error during parsing: {e}")
