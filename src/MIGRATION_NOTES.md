# Refactor Notes — LlamaIndex → LangChain, Chroma → Qdrant, Hybrid Retrieval, Memory, Evaluation

## File map

| Old | New | Status |
|---|---|---|
| `src/config.py` | `src/config.py` | Extended |
| `src/download_papers.py` | `src/download_papers.py` | Unchanged + one bugfix |
| `src/parsing.py` | `src/parsing.py` | Rewritten (LangChain loaders/splitters) |
| `src/indexing.py` | `src/indexing.py` | Rewritten (Qdrant) |
| — | `src/retrieval.py` | **New** — hybrid dense + BM25 with RRF |
| — | `src/memory.py` | **New** — session-scoped conversation history |
| `src/pipeline.py` | `src/pipeline.py` | Rewritten (LCEL, history-aware) |
| `src/evaluate.py` | `src/evaluate.py` | Rewritten (3-stage framework) |
| `app.py` | `app/app.py` | Updated |
| — | `eval/test_questions.json` | **New** — gold set with negative controls |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in GEMINI_API_KEY (Qdrant needs no key)
python -m src.download_papers # Phase 1 (unchanged)
python -m src.indexing        # Phase 3 — builds Qdrant collection + chunk store
python -m src.retrieval       # sanity-check hybrid search
python -m src.pipeline        # sanity-check RAG + memory
streamlit run app/app.py
python -m src.evaluate        # full evaluation
```

---

## 1. LlamaIndex → LangChain

| LlamaIndex | LangChain | Note |
|---|---|---|
| `SimpleDirectoryReader` | `PyPDFLoader` per file | One loader per PDF so metadata is unambiguous |
| `TextNode` / `Document` | `Document` | LangChain has no separate node type |
| `SentenceSplitter` | `RecursiveCharacterTextSplitter.from_huggingface_tokenizer` | See §5 |
| `Settings.embed_model` / `Settings.llm` | explicit arguments | Global singleton removed |
| `VectorStoreIndex` + `StorageContext` | `QdrantVectorStore` | One object |
| `RetrieverQueryEngine` | `create_retrieval_chain(...)` (LCEL) | Dict in, dict out |
| `response.source_nodes`, `node.score` | `result["context"]`, `doc.metadata[...]` | LangChain `Document` has no `.score` |
| `excluded_llm_metadata_keys` | `document_prompt` | See §6 |

**Import-path gotcha.** `create_retrieval_chain`, `create_history_aware_retriever` and `create_stuff_documents_chain` live in `langchain.chains` on LangChain 0.3.x but moved to `langchain_classic.chains` in LangChain 1.x. `src/pipeline.py` tries `langchain_classic` first and falls back, so both work. If you pin LangChain 1.x you must also `pip install langchain-classic`.

**Removed hazard.** The old `get_query_engine()` carried a comment warning that `build_or_load_index` had to be called *first* because `indexing.py` could clobber `Settings.llm`. That entire class of bug is gone — LangChain passes the LLM explicitly.

---

## 2. ChromaDB → Qdrant

The original plan was Pinecone, but **the Pinecone SDK has no Python 3.14 wheel**. Qdrant was chosen instead: `qdrant-client` ships a pure-Python `py3-none-any` wheel with 3.14 classified and effectively no compiled dependencies, so it installs cleanly on new interpreters.

This is not a downgrade in the story. Qdrant is the same class of tool as Pinecone — HNSW index, payload filtering, native sparse-dense hybrid, hosted cloud tier — and it adds an **embedded mode** that removes an API key and a network dependency from local development.

### Two modes, one code path

| | Config | Use when |
|---|---|---|
| **Embedded** *(default)* | `QDRANT_URL=` empty | Local dev. Writes to `data/qdrant_db/`. No server, no Docker, no key. |
| **Server** | `QDRANT_URL=http://localhost:6333` | Concurrency, or deploying. `docker run -p 6333:6333 qdrant/qdrant`. Qdrant Cloud also needs `QDRANT_API_KEY`. |

Switching is one environment variable. No code changes.

### Differences that forced code changes

1. **Vector size and distance are immutable.** `create_collection(vectors_config=VectorParams(size=384, distance=COSINE))` is set once. Change the embedding model and you must recreate the collection. `EMBEDDING_DIMENSION` in config is load-bearing, not decorative.
2. **Point IDs must be UUIDs or unsigned ints.** Our readable `chunk_id` (`lora.pdf::p004::c02`) is not a legal Qdrant id, so `chunk_uuid()` hashes it into a deterministic UUID5. Determinism is the point — re-running the build overwrites the same points instead of duplicating them. The readable id stays in the payload, so citations and the eval harness are unaffected.
3. **Collection creation is synchronous** and the count is immediately consistent — both the ready-polling loop and the post-upsert sleep that Pinecone needed are gone.
4. **Pinecone "namespaces" → Qdrant "collections."** One collection per corpus.
5. **No bulk export in the design.** Chroma doubled as a local docstore. BM25 needs the full corpus in RAM, so `parsing.py` writes `data/processed/chunks.jsonl` and that file is the source of truth for sparse retrieval. This remains the most important structural consequence of leaving Chroma.

### Gotcha: the embedded-mode directory lock

`QdrantClient(path=...)` takes an **exclusive lock** on the storage folder. You cannot run the Streamlit app and the evaluation script at the same time — the second process raises *"already accessed by another instance."* Either stop one, or switch to server mode:

```bash
docker run -p 6333:6333 qdrant/qdrant
export QDRANT_URL=http://localhost:6333
```

This is worth mentioning unprompted in an interview: it is the concrete trade-off embedded mode buys you, and knowing it signals you actually ran the thing.

### If you ever want to go back to Pinecone

Everything downstream depends only on the LangChain `VectorStore` interface, so `retrieval.py`, `pipeline.py`, `app.py`, and `evaluate.py` need no edits. Only `get_vector_store()` and `build_or_load_index()` in `indexing.py` change. That portability is itself the argument for the abstraction.

---

## 3. Hybrid retrieval

`src/retrieval.py`. Dense (Qdrant) and sparse (BM25) each return 10 candidates; they are fused by **weighted Reciprocal Rank Fusion** and cut to the top 5.

```
score(doc) = Σ_r  weight_r / (60 + rank_r(doc))
```

Why RRF rather than summing scores: cosine similarity sits in a narrow ~0.3–0.7 band while BM25 is unbounded and corpus-dependent, so a raw sum lets BM25 dominate silently, and per-query min-max normalisation is unstable when one outlier rescales the list. RRF discards magnitudes and keeps only rank order — the one thing both retrievers mean the same way. It needs no tuning or training data, and a chunk that both retrievers like accumulates from both terms, so consensus results rise.

Implemented by hand (~40 lines) instead of using `EnsembleRetriever` for two reasons: that class also moved packages between LangChain versions, and writing it out lets us stamp `rank_dense` / `rank_sparse` onto each returned document. The UI shows that provenance in the citation panel and the evaluator uses it to attribute wins — the hybrid search is visible rather than asserted.

Failure isolation: if either retriever raises, the fusion logs it and continues with the other, so a Qdrant failure degrades to keyword search instead of a 500.

### Scale caveat
`BM25Retriever` holds the whole corpus in memory and rebuilds statistics at startup. Fine for 8 papers / a few thousand chunks. At millions of chunks you would move to Qdrant's native sparse-vector hybrid search (server mode) or an OpenSearch BM25 backend. Worth saying out loud in an interview — it shows you know where the design stops working.

---

## 4. Conversation history

`src/memory.py` — a `dict[session_id -> list[BaseMessage]]` in process memory with a sliding window of the last 6 turns. No disk, no Redis, no DB. Restart the app and it is gone, which is the requested scope.

The important part is *why* memory matters here, and it is two separate jobs:

1. **Query rewriting (before retrieval).** "How does it compare to LoRA?" has no retrievable content — "it" has no useful embedding and no useful keyword. `create_history_aware_retriever` uses the history to rewrite it into a standalone question *before* searching. Without this step, follow-up questions retrieve noise no matter how good the retriever is. This is the non-obvious point most people miss.
2. **Answer conditioning (after retrieval).** The QA prompt also receives the history so the model stays consistent with what it already said.

Two stores are kept deliberately separate: `st.session_state.messages` is the render layer (includes error bubbles and source metadata), `ChatSession` is the prompt layer (clean Q/A pairs, trimmed). A rendering change can therefore never corrupt the prompt.

`RunnableWithMessageHistory` would do the same job, but it hides the wiring behind a callback and a config dict and its import path has churned. Passing `chat_history` explicitly means you can print the exact message list that entered the prompt.

Cost note: when history is non-empty, every turn costs one extra LLM call for the rewrite. When it is empty, the chain skips that call.

---

## 5. Two real bugs found in the original code

**MiniLM truncation (silent, and it mattered).** `all-MiniLM-L6-v2` has a hard 256-token input limit, but `CHUNK_SIZE` was 512 tokens. Roughly the back half of every chunk was truncated before embedding — that text was retrievable only via whatever happened to be in its first 256 tokens. Chunks are now 256 tokens, measured with the model's own tokenizer via `from_huggingface_tokenizer` (note that `RecursiveCharacterTextSplitter` counts *characters* by default; passing 256 without the tokenizer would have chunked at 256 characters).

**Citations the model could not see.** `parsing.py` set `excluded_llm_metadata_keys = ["paper_title", "page_number", ...]` while the prompt instructed Gemini to cite `[Paper Title, Page X]`. The model was being asked to cite metadata that was stripped before it ever reached the prompt, so any correct citation was luck. Fixed with an explicit `document_prompt` that stamps `[Source: {paper_title} | Page {page_number}]` onto each chunk as it is formatted into the context block. Stage 3 of the evaluation now measures this directly.

Minor: `download_papers.py` built a `Request` with a User-Agent header and then called `urlretrieve(url, ...)`, which ignores it. Switched to `urlopen(request)`.

---

## 6. Evaluation framework

The old harness measured "did we retrieve any chunks" (always yes — a vector search always returns k results, so the metric is a constant) and substring keyword matching (rewards keyword stuffing, punishes correct paraphrase).

The new framework separates the two failure modes, because they have different fixes: **retrieval failure** = the answer was never in the context; **generation failure** = it was there and the model still got it wrong.

**Stage 1 — Retrieval (deterministic, no LLM, free).** Hit@k, Precision@k, Recall@k, MRR, nDCG@10 against hand-labelled gold `(paper, page)` units, run as an **ablation across dense / sparse / hybrid**. This is what justifies the hybrid retriever with a number instead of a claim. Recall is the ceiling on the whole system — the generator cannot beat it. Precision is proportional to wasted context tokens and therefore cost. MRR matters because LLMs attend unevenly across long contexts.

**Stage 2 — Generation (LLM-as-judge).** Faithfulness (the hallucination metric), relevance, and completeness scored 1–5 with a written rationale by a separate deterministic (temperature 0) model instance, so scores are stable and not self-congratulatory.

**Stage 3 — Behavioural (deterministic).**
- *Citation validity*: parse `[Title, Page N]` out of the answer and verify each against what was actually retrieved. Distinguishes valid / right-paper-wrong-page / fabricated. A fabricated citation is the worst failure a research assistant can have.
- *Abstention*: the gold set includes three questions the corpus **cannot** answer. The system must say so. This is a negative control — without it, a model that answers confidently no matter what scores well on everything else.
- *Memory probe*: turn 1 asks about QLoRA, turn 2 says "how much memory does **it** save?" and the check asserts that `qlora.pdf` was still retrieved. Only possible if history-aware rewriting works.
- Keyword coverage retained as a cheap regression tripwire; latency p50/p95.

Outputs a timestamped JSON blob and a readable Markdown table.

```bash
python -m src.evaluate --no-judge          # Stages 1 + 3 only, zero judge cost
python -m src.evaluate --modes hybrid      # skip the ablation
python -m src.evaluate --dump-retrievals   # labelling aid, see below
```

### Known limitation, stated honestly
The gold set currently labels relevance at **paper level** (`"pages": []` means any page of that file counts). That makes Recall@k optimistic. Page-level labels are supported by the same code — fill in the `pages` arrays — and `--dump-retrievals` prints the top chunks per question to make that labelling fast. Retrieval evaluation is bounded by labelling effort, not by code, and it is better to say so than to quietly report a flattering number.

Also worth knowing: LLM-as-judge correlates with human judgement but is not a substitute for it, and it inherits the judge model's blind spots. Using a different model family as judge (e.g. Claude or GPT judging Gemini) would remove the remaining shared-blind-spot risk.

---

## 7. Model-name note

`gemini-3.5-flash` in your config is a valid current model (May 2026). `gemini-3.6-flash` and `gemini-3.7-flash` are newer and cheaper on the introductory pricing. One caveat that will bite you: Google has deprecated `temperature`, `top_p` and `top_k` on the newest Gemini 3.x endpoints. `get_llm()` therefore only passes `temperature` when it is set — leave `LLM_TEMPERATURE` empty in `.env` to omit it entirely if your endpoint rejects it.
