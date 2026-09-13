"""
Phase 3: Embedding and Vector Storage  (LangChain + Qdrant)

Takes the chunks from Phase 2, embeds them with a local sentence-transformers
model, and upserts them into a Qdrant collection.

WHY QDRANT INSTEAD OF PINECONE
------------------------------
The Pinecone SDK has no Python 3.14 wheel. qdrant-client ships a pure-Python
`py3-none-any` wheel with 3.14 classified and effectively no compiled
dependencies, so it installs cleanly on new interpreters. It is also a
genuine production vector database (same class of tool as Pinecone: HNSW
index, payload filtering, hybrid support, hosted cloud tier), so nothing about
the project's story downgrades — and it gains an embedded mode that removes an
API key and a network dependency from local development.

TWO MODES, ONE CODE PATH
------------------------
  * Embedded  (QDRANT_URL empty): QdrantClient(path=...) writes to a local
    folder. No server, no Docker, no key. This is the default.
  * Server    (QDRANT_URL set):   docker run -p 6333:6333 qdrant/qdrant,
    or Qdrant Cloud with QDRANT_API_KEY.
Switching is one environment variable; no code changes.

MIGRATION NOTES (Pinecone -> Qdrant)
------------------------------------
* pc.create_index(dimension=, metric=)  -> client.create_collection(
      vectors_config=VectorParams(size=384, distance=Distance.COSINE))
  Same constraint: size and distance are fixed at creation. Change the
  embedding model and you must recreate the collection.
* Creation is SYNCHRONOUS — the ready-polling loop Pinecone needed is gone,
  and so is Pinecone's eventual-consistency lag on the vector count.
* Pinecone "namespaces" -> Qdrant "collections". One collection per corpus.
* POINT IDs MUST BE UUIDs OR UNSIGNED INTS. Our human-readable chunk_id
  ("lora.pdf::p004::c02") is not a legal Qdrant id, so we hash it into a
  deterministic UUID5. Determinism is the point: re-running the build
  overwrites the same points instead of duplicating them.

MIGRATION NOTES (LlamaIndex -> LangChain)
-----------------------------------------
* Settings.embed_model (global singleton) -> the embedding object is passed
  explicitly into QdrantVectorStore. LangChain has no global Settings, which
  also removes the ordering hazard the old pipeline.py warned about
  ("this might overwrite Settings.llm to None").
* VectorStoreIndex / StorageContext -> QdrantVectorStore.
"""

from __future__ import annotations

import logging
import uuid

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_DISTANCE,
    QDRANT_PATH,
    QDRANT_URL,
    UPSERT_BATCH_SIZE,
)
from src.parsing import load_and_chunk_pdfs, save_chunk_store

logger = logging.getLogger(__name__)

# Fixed namespace so chunk_id -> UUID is stable across machines and runs.
_ID_NAMESPACE = uuid.UUID("6f0d5c8e-4a2b-5f31-9c77-2f1b8a0e4d33")

# Cached so Streamlit reruns / eval loops don't reload the model from disk.
_embedding_singleton: HuggingFaceEmbeddings | None = None
_client_singleton: QdrantClient | None = None


def chunk_uuid(chunk_id: str) -> str:
    """
    Map a readable chunk_id to a deterministic UUID.

    Qdrant rejects arbitrary string point ids, but we still want idempotent
    upserts. UUID5 is a pure function of the chunk_id, so the same chunk always
    lands on the same point. The readable id stays in the payload metadata, so
    citations and the eval harness are unaffected.
    """
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initialize the local HuggingFace embedding model.
    all-MiniLM-L6-v2 gives fast, free, local embeddings.

    normalize_embeddings=True is important: with L2-normalised vectors, cosine
    and dot product agree, and similarity scores land in a predictable range
    that we can display in the UI.
    """
    global _embedding_singleton
    if _embedding_singleton is None:
        logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
        _embedding_singleton = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_singleton


def get_qdrant_client() -> QdrantClient:
    """
    Create the Qdrant client — embedded or server, depending on config.

    IMPORTANT (embedded mode): QdrantClient(path=...) takes an exclusive lock
    on the storage directory. Only ONE process may hold it at a time, so you
    cannot run `streamlit run app/app.py` and `python -m src.evaluate`
    simultaneously — the second will raise "already accessed by another
    instance". Stop one, or switch to server mode:
        docker run -p 6333:6333 qdrant/qdrant
        export QDRANT_URL=http://localhost:6333
    """
    global _client_singleton
    if _client_singleton is None:
        if QDRANT_URL:
            logger.info(f"Connecting to Qdrant server at {QDRANT_URL}")
            _client_singleton = QdrantClient(
                url=QDRANT_URL, api_key=QDRANT_API_KEY or None
            )
        else:
            QDRANT_PATH.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using embedded Qdrant at {QDRANT_PATH}")
            _client_singleton = QdrantClient(path=str(QDRANT_PATH))
    return _client_singleton


def ensure_collection(client: QdrantClient | None = None) -> QdrantClient:
    """
    Create the collection if it does not exist. Idempotent — safe on startup.
    Unlike Pinecone index creation, this returns ready; no polling needed.
    """
    client = client or get_qdrant_client()

    if not client.collection_exists(QDRANT_COLLECTION_NAME):
        logger.info(
            f"Creating Qdrant collection '{QDRANT_COLLECTION_NAME}' "
            f"(size={EMBEDDING_DIMENSION}, distance={QDRANT_DISTANCE})..."
        )
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance[QDRANT_DISTANCE],
            ),
        )
    else:
        logger.info(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' already exists.")

    return client


def get_vector_count(client: QdrantClient | None = None) -> int:
    """How many points are in the collection? (Chroma's collection.count())"""
    client = client or get_qdrant_client()
    try:
        if not client.collection_exists(QDRANT_COLLECTION_NAME):
            return 0
        return int(client.count(QDRANT_COLLECTION_NAME, exact=True).count)
    except Exception as exc:
        logger.warning(f"Could not read collection stats: {exc}")
        return 0


def get_vector_store() -> QdrantVectorStore:
    """Return the LangChain wrapper around our Qdrant collection."""
    client = ensure_collection()
    return QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=get_embedding_model(),
    )


def build_or_load_index(force_rebuild: bool = False) -> QdrantVectorStore:
    """
    Loads the vector store if the collection is already populated.
    If not (or if force_rebuild=True), parses the PDFs, embeds them, and
    upserts them to Qdrant. Also refreshes the local chunk store used by BM25.
    """
    client = ensure_collection()

    count = get_vector_count(client)
    if count > 0 and not force_rebuild:
        logger.info(
            f"Found {count} existing points in collection "
            f"'{QDRANT_COLLECTION_NAME}'. Loading existing index..."
        )
        return get_vector_store()

    if force_rebuild and client.collection_exists(QDRANT_COLLECTION_NAME):
        # Drop and recreate so a re-chunk doesn't leave orphaned points from
        # the previous chunking strategy behind.
        logger.info(f"force_rebuild=True — dropping '{QDRANT_COLLECTION_NAME}'.")
        client.delete_collection(QDRANT_COLLECTION_NAME)
        ensure_collection(client)

    vector_store = get_vector_store()

    logger.info("Building new index from PDFs. This may take a moment to embed...")
    chunks: list[Document] = load_and_chunk_pdfs()

    # Persist locally FIRST so BM25 works even if the upsert is interrupted.
    save_chunk_store(chunks)

    ids = [chunk_uuid(c.metadata["chunk_id"]) for c in chunks]
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

    print("\nPhase 3: Embedding and Vector Storage (Qdrant)")
    print("=" * 60)

    try:
        build_or_load_index(force_rebuild=True)
        count = get_vector_count()

        print("\nVerification Success:")
        print(f"  - Mode           : {'server' if QDRANT_URL else 'embedded'}")
        print(f"  - Location       : {QDRANT_URL or QDRANT_PATH}")
        print(f"  - Collection     : {QDRANT_COLLECTION_NAME}")
        print(f"  - Embedding Dim  : {EMBEDDING_DIMENSION}")
        print(f"  - Points Stored  : {count}")

    except Exception as e:
        logger.error(f"Error during indexing: {e}")
