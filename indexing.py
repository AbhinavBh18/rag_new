"""
Phase 3: Embedding and Vector Storage  (LangChain + Pinecone)

Takes the chunks from Phase 2, embeds them with a local sentence-transformers
model, and upserts them into a Pinecone serverless index.

MIGRATION NOTES (ChromaDB -> Pinecone)
--------------------------------------
* chromadb.PersistentClient(path=...)  -> pinecone.Pinecone(api_key=...)
  Chroma persisted to a local folder; Pinecone is a hosted service, so the
  "does the index exist / is it populated" check is now a network call
  (`describe_index_stats`) rather than a directory read.
* db.get_or_create_collection(...)     -> pc.create_index(..., ServerlessSpec)
  Pinecone requires the vector `dimension` and `metric` UP FRONT and they are
  immutable. That is why EMBEDDING_DIMENSION (384) now genuinely matters — if
  you swap the embedding model you must also delete and recreate the index.
* Index creation is asynchronous, so we poll until the index reports ready.
* Chroma stored the chunk text alongside the vector and could stream it all
  back; Pinecone does store text in metadata but is not designed for bulk
  export, so we ALSO write the chunk store JSONL (see parsing.py) for BM25.
* Upserts are batched (UPSERT_BATCH_SIZE) because Pinecone caps request size
  at ~2 MB / 1000 vectors.

MIGRATION NOTES (LlamaIndex -> LangChain)
-----------------------------------------
* Settings.embed_model (global singleton) -> the embedding object is passed
  explicitly into PineconeVectorStore. LangChain has no global Settings, which
  also removes the ordering hazard the old pipeline.py comment warned about
  ("this might overwrite Settings.llm to None").
* VectorStoreIndex / StorageContext -> PineconeVectorStore (a single object
  that is both the writer and the retriever factory).
"""

from __future__ import annotations

import logging
import time

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_METRIC,
    PINECONE_NAMESPACE,
    PINECONE_REGION,
    UPSERT_BATCH_SIZE,
)
from src.parsing import load_and_chunk_pdfs, save_chunk_store

logger = logging.getLogger(__name__)

# Cached so Streamlit reruns / eval loops don't reload the model from disk.
_embedding_singleton: HuggingFaceEmbeddings | None = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initialize the local HuggingFace embedding model.
    all-MiniLM-L6-v2 gives fast, free, local embeddings.

    normalize_embeddings=True is important: with L2-normalised vectors,
    Pinecone's cosine metric and dot product agree, and similarity scores land
    in a predictable [0, 1]-ish range that we can display in the UI.
    """
    global _embedding_singleton
    if _embedding_singleton is None:
        logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
        _embedding_singleton = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_singleton


def get_pinecone_client() -> Pinecone:
    """Create the Pinecone control-plane client."""
    if not PINECONE_API_KEY:
        raise ValueError(
            "PINECONE_API_KEY is not set. Add it to your .env file "
            "(free tier: https://app.pinecone.io)."
        )
    return Pinecone(api_key=PINECONE_API_KEY)


def ensure_index(pc: Pinecone | None = None) -> Pinecone:
    """
    Create the serverless index if it does not already exist, then wait for it
    to become ready. Idempotent — safe to call on every startup.
    """
    pc = pc or get_pinecone_client()

    existing = {idx["name"] for idx in pc.list_indexes()}
    if PINECONE_INDEX_NAME not in existing:
        logger.info(
            f"Creating Pinecone index '{PINECONE_INDEX_NAME}' "
            f"(dim={EMBEDDING_DIMENSION}, metric={PINECONE_METRIC})..."
        )
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        # Index creation is async — poll until ready (usually <60s).
        for _ in range(60):
            if pc.describe_index(PINECONE_INDEX_NAME).status.get("ready"):
                break
            time.sleep(2)
        logger.info("Index is ready.")
    else:
        logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")

    return pc


def get_vector_count(pc: Pinecone | None = None) -> int:
    """How many vectors are already in our namespace? (Chroma's collection.count())"""
    pc = pc or get_pinecone_client()
    try:
        stats = pc.Index(PINECONE_INDEX_NAME).describe_index_stats()
    except Exception as exc:
        logger.warning(f"Could not read index stats: {exc}")
        return 0
    namespaces = stats.get("namespaces", {}) or {}
    ns = namespaces.get(PINECONE_NAMESPACE, {})
    return int(ns.get("vector_count", 0))


def get_vector_store() -> PineconeVectorStore:
    """Return the LangChain wrapper around our Pinecone index."""
    ensure_index()
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=get_embedding_model(),
        namespace=PINECONE_NAMESPACE,
        # Pinecone metadata key under which the chunk text is stored.
        text_key="text",
    )


def build_or_load_index(force_rebuild: bool = False) -> PineconeVectorStore:
    """
    Loads the vector store if the index is already populated.
    If not (or if force_rebuild=True), parses the PDFs, embeds them, and
    upserts them to Pinecone. Also refreshes the local chunk store used by BM25.
    """
    pc = ensure_index()
    vector_store = get_vector_store()

    count = get_vector_count(pc)
    if count > 0 and not force_rebuild:
        logger.info(
            f"Found {count} existing vectors in Pinecone namespace "
            f"'{PINECONE_NAMESPACE}'. Loading existing index..."
        )
        return vector_store

    if force_rebuild and count > 0:
        # Clear the namespace so a re-chunk doesn't leave orphaned vectors from
        # the previous chunking strategy behind.
        logger.info(f"force_rebuild=True — clearing namespace '{PINECONE_NAMESPACE}'.")
        try:
            pc.Index(PINECONE_INDEX_NAME).delete(
                delete_all=True, namespace=PINECONE_NAMESPACE
            )
        except Exception as exc:
            logger.warning(f"Namespace delete failed (may be empty already): {exc}")

    logger.info("Building new index from PDFs. This may take a moment to embed...")
    chunks: list[Document] = load_and_chunk_pdfs()

    # Persist locally FIRST so BM25 works even if the upsert is interrupted.
    save_chunk_store(chunks)

    ids = [c.metadata["chunk_id"] for c in chunks]
    total = len(chunks)
    for start in range(0, total, UPSERT_BATCH_SIZE):
        batch = chunks[start : start + UPSERT_BATCH_SIZE]
        batch_ids = ids[start : start + UPSERT_BATCH_SIZE]
        vector_store.add_documents(documents=batch, ids=batch_ids)
        logger.info(f"  Upserted {min(start + UPSERT_BATCH_SIZE, total)}/{total} chunks")

    logger.info("Indexing and embedding completed successfully!")
    return vector_store


# ──────────────────────────────────────────────
# Standalone execution for Phase 3 Verification
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("\nPhase 3: Embedding and Vector Storage (Pinecone)")
    print("=" * 60)

    try:
        build_or_load_index(force_rebuild=True)

        # Pinecone's index is eventually consistent; give it a beat before we
        # read stats back, otherwise the count can lag behind the upsert.
        time.sleep(5)
        count = get_vector_count()

        print("\nVerification Success:")
        print(f"  - Pinecone Index   : {PINECONE_INDEX_NAME}")
        print(f"  - Namespace        : {PINECONE_NAMESPACE}")
        print(f"  - Cloud / Region   : {PINECONE_CLOUD} / {PINECONE_REGION}")
        print(f"  - Embedding Dim    : {EMBEDDING_DIMENSION}")
        print(f"  - Vectors Stored   : {count}")

    except Exception as e:
        logger.error(f"Error during indexing: {e}")
