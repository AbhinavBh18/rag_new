"""
Phase 3b (NEW): Hybrid Retrieval

The old pipeline used dense vector search only. Dense embeddings are great at
paraphrase ("how does it cut memory use?" -> a passage about GPU footprint) but
weak at rare literal tokens — model names, hyperparameters, acronyms like
"NF4", "IA3", "r=8". BM25 is the mirror image: excellent on exact terms, blind
to paraphrase. Running both and fusing the rankings gets the strengths of each.

FUSION STRATEGY: Weighted Reciprocal Rank Fusion (RRF)
------------------------------------------------------
    score(doc) = Σ_retriever  weight_r * 1 / (RRF_K + rank_r(doc))

Why RRF instead of just adding the raw scores together?
  * Cosine similarity (0-1, tightly clustered around 0.3-0.7) and BM25 scores
    (unbounded, corpus-dependent, often 0-30) live on completely different
    scales. Summing them means BM25 silently dominates.
  * Min-max normalising per query is unstable — one outlier rescales everything.
  * RRF throws away the magnitudes and keeps only the ordering, which is the
    part both retrievers actually agree on the meaning of. It needs no tuning
    and no training data.
  * A document ranked highly by BOTH retrievers accumulates from both terms, so
    consensus results naturally float to the top.

RRF_K (default 60) damps the head of the curve: with K=60, rank 1 scores
1/61 and rank 2 scores 1/62 — close together — so a document only wins by
appearing in several lists, not by narrowly topping one.

IMPLEMENTATION NOTE
-------------------
LangChain ships `EnsembleRetriever`, which does essentially this. We implement
it explicitly here for two reasons: (1) it moved packages between LangChain
0.3 and 1.x (`langchain.retrievers` -> `langchain_classic.retrievers`), and
(2) writing it out lets us attach per-retriever ranks to each document's
metadata, which the evaluation harness uses to attribute wins to dense vs
sparse. It is ~40 lines and removes a moving dependency.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.config import (
    DENSE_TOP_K,
    HYBRID_DENSE_WEIGHT,
    HYBRID_SPARSE_WEIGHT,
    RRF_K,
    SPARSE_TOP_K,
    TOP_K,
)
from src.indexing import get_vector_store
from src.parsing import load_chunk_store

logger = logging.getLogger(__name__)

# BM25 is built in-process from the chunk store; cache it so we don't rebuild
# the term statistics on every Streamlit rerun.
_bm25_singleton: BaseRetriever | None = None


def _doc_key(doc: Document) -> str:
    """Stable identity for a chunk, used to dedupe across the two retrievers."""
    return doc.metadata.get("chunk_id") or f"hash::{hash(doc.page_content)}"


# ──────────────────────────────────────────────
# The fusion retriever
# ──────────────────────────────────────────────
class HybridRRFRetriever(BaseRetriever):
    """Fuses N retrievers' ranked lists with weighted Reciprocal Rank Fusion."""

    retrievers: list[BaseRetriever]
    names: list[str]
    weights: list[float]
    top_k: int = TOP_K
    rrf_k: int = RRF_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        fused_scores: dict[str, float] = {}
        best_doc: dict[str, Document] = {}
        rank_info: dict[str, dict[str, int]] = {}

        for retriever, name, weight in zip(self.retrievers, self.names, self.weights):
            try:
                docs = retriever.invoke(query)
            except Exception as exc:
                # Graceful degradation: if Pinecone is down we still answer
                # from BM25 (and vice versa) rather than failing the request.
                logger.warning(f"Retriever '{name}' failed: {exc}")
                continue

            for rank, doc in enumerate(docs, start=1):
                key = _doc_key(doc)
                fused_scores[key] = fused_scores.get(key, 0.0) + weight / (
                    self.rrf_k + rank
                )
                rank_info.setdefault(key, {})[f"rank_{name}"] = rank
                if key not in best_doc:
                    best_doc[key] = Document(
                        page_content=doc.page_content, metadata=dict(doc.metadata)
                    )

        ordered = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)

        results: list[Document] = []
        for position, (key, score) in enumerate(ordered[: self.top_k], start=1):
            doc = best_doc[key]
            doc.metadata["rrf_score"] = round(score, 6)
            doc.metadata["final_rank"] = position
            # e.g. rank_dense=2, rank_sparse=7 -> we can see WHY it surfaced.
            doc.metadata.update(rank_info.get(key, {}))
            results.append(doc)

        return results


# ──────────────────────────────────────────────
# Component builders
# ──────────────────────────────────────────────
def build_dense_retriever(k: int = DENSE_TOP_K) -> BaseRetriever:
    """Pinecone similarity search. (Replaces `index.as_retriever(similarity_top_k=)`.)"""
    return get_vector_store().as_retriever(search_kwargs={"k": k})


def build_sparse_retriever(
    k: int = SPARSE_TOP_K, docs: Iterable[Document] | None = None
) -> BaseRetriever:
    """
    BM25 keyword retriever, built in memory from the local chunk store.

    Scale note: BM25Retriever holds the whole corpus in RAM. Fine here (~8
    papers, a few thousand chunks). For a corpus in the millions you would
    instead use Pinecone's native sparse-dense hybrid vectors or an
    Elasticsearch/OpenSearch BM25 backend — see MIGRATION_NOTES.md.
    """
    global _bm25_singleton
    if _bm25_singleton is not None and docs is None:
        _bm25_singleton.k = k
        return _bm25_singleton

    from langchain_community.retrievers import BM25Retriever  # needs `rank_bm25`

    corpus = list(docs) if docs is not None else load_chunk_store()
    retriever = BM25Retriever.from_documents(corpus)
    retriever.k = k
    logger.info(f"Built BM25 index over {len(corpus)} chunks.")

    if docs is None:
        _bm25_singleton = retriever
    return retriever


def get_retriever(
    mode: str = "hybrid",
    top_k: int = TOP_K,
    dense_k: int = DENSE_TOP_K,
    sparse_k: int = SPARSE_TOP_K,
) -> BaseRetriever:
    """
    Factory for the three retrieval strategies. Having all three behind one
    interface is what makes the ablation study in evaluate.py a one-liner.

    Note we over-fetch (dense_k/sparse_k = 10) and then cut to top_k = 5 after
    fusion. Fusion can only reorder what it is given, so the candidate pool has
    to be deeper than the final slice.
    """
    mode = mode.lower()

    if mode == "dense":
        return build_dense_retriever(k=top_k)
    if mode == "sparse":
        return build_sparse_retriever(k=top_k)
    if mode == "hybrid":
        return HybridRRFRetriever(
            retrievers=[build_dense_retriever(dense_k), build_sparse_retriever(sparse_k)],
            names=["dense", "sparse"],
            weights=[HYBRID_DENSE_WEIGHT, HYBRID_SPARSE_WEIGHT],
            top_k=top_k,
        )

    raise ValueError(f"Unknown retrieval mode '{mode}'. Use dense | sparse | hybrid.")


# ──────────────────────────────────────────────
# Standalone verification
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("\nHybrid Retrieval Check")
    print("=" * 60)

    query = "What quantization data type does QLoRA introduce?"
    print(f"Query: {query}\n")

    for mode in ("dense", "sparse", "hybrid"):
        print(f"--- {mode.upper()} ---")
        try:
            docs = get_retriever(mode=mode).invoke(query)
            for i, d in enumerate(docs, 1):
                extra: dict[str, Any] = {
                    k: v for k, v in d.metadata.items() if k.startswith(("rank_", "rrf_"))
                }
                print(
                    f"  {i}. {d.metadata.get('file_name')} "
                    f"p{d.metadata.get('page_number')} {extra}"
                )
        except Exception as e:
            print(f"  Failed: {e}")
        print()
