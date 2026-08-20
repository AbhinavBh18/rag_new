"""
Phase 5: Evaluation Framework

The old harness measured two things: "did we retrieve *any* chunks?" (which is
always yes — a vector search always returns k results) and "did the answer
contain some expected substrings?" (which rewards keyword stuffing and punishes
correct paraphrase). Neither can tell you whether the system is actually good.

This framework separates the two failure modes a RAG system has, because they
need different fixes:

  RETRIEVAL FAILURE   — the answer was never in the context. Fix the retriever.
  GENERATION FAILURE  — the answer was in the context and the model still got
                        it wrong (or made something up). Fix the prompt/model.

┌─ STAGE 1: Retrieval quality (deterministic, no LLM, cheap, reproducible) ────┐
│  Against a hand-labelled gold set of relevant (paper, page) units:           │
│    * Hit@k        — did at least one relevant chunk make the cut?            │
│    * Precision@k  — how much of the context window was actually useful?      │
│                     Directly proportional to wasted tokens and cost.         │
│    * Recall@k     — of the pages that contain the answer, how many did we    │
│                     find? This is the metric that caps the whole system:     │
│                     the generator cannot beat its recall.                    │
│    * MRR          — how high up the first relevant hit lands. Matters        │
│                     because LLMs attend unevenly across a long context.      │
│    * nDCG@k       — rank-discounted gain; the standard IR ranking metric.    │
│  Run as an ABLATION across dense / sparse / hybrid, which is what actually   │
│  justifies the hybrid retriever with a number instead of a claim.            │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ STAGE 2: Generation quality (LLM-as-judge, on the winning retriever) ───────┐
│    * Faithfulness   — is every claim supported by the retrieved context?     │
│                       This is the hallucination metric.                      │
│    * Relevance      — does it answer the question that was asked?            │
│    * Completeness   — does it use everything useful the context offered?     │
│  Judged 1-5 with a written rationale, by a separate deterministic model      │
│  instance (temperature 0) so scores are stable and not self-congratulatory.  │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ STAGE 3: Behavioural checks (deterministic, no LLM) ────────────────────────┐
│    * Citation validity — parse [Title, Page N] out of the answer and verify  │
│                          each one against what was actually retrieved. A     │
│                          citation to a page we never saw is a fabricated     │
│                          citation, which is the most dangerous failure a     │
│                          research assistant can have.                        │
│    * Abstention        — the gold set contains questions the corpus CANNOT   │
│                          answer. The system must say so. This is a negative  │
│                          control: without it, a model that answers           │
│                          confidently no matter what scores well everywhere.  │
│    * Keyword coverage  — retained from v1 as a cheap regression tripwire.    │
│    * Latency          — p50/p95 per query.                                   │
└──────────────────────────────────────────────────────────────────────────────┘

Usage:
    python -m src.evaluate                      # full run
    python -m src.evaluate --no-judge           # Stage 1 + 3 only (no API cost)
    python -m src.evaluate --modes hybrid       # skip the ablation
    python -m src.evaluate --dump-retrievals    # helper for labelling gold pages
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.config import (
    EVAL_DIR,
    EVAL_K_CUTOFFS,
    EVAL_MODES,
    EVAL_RESULTS_DIR,
    EVAL_RETRIEVAL_CANDIDATES,
    TOP_K,
)

logger = logging.getLogger(__name__)

ABSTENTION_MARKERS = (
    "cannot find the answer",
    "not in the provided papers",
    "does not contain",
    "no information",
    "cannot answer",
)


# ══════════════════════════════════════════════════════════════════
# Gold-set matching
# ══════════════════════════════════════════════════════════════════
def _gold_units(item: dict[str, Any]) -> set[tuple[str, int | None]]:
    """
    Expand a question's `relevant_sources` into a set of (file, page) units.

    A source with an empty `pages` list means "any page of this paper counts",
    represented as (file, None). This lets you start evaluating immediately
    with file-level labels and tighten to page-level later without changing
    any code — labelling effort is the real bottleneck on retrieval eval.
    """
    units: set[tuple[str, int | None]] = set()
    for src in item.get("relevant_sources", []):
        file_name = src["file"]
        pages = src.get("pages") or []
        if not pages:
            units.add((file_name, None))
        else:
            for p in pages:
                units.add((file_name, int(p)))
    return units


def _is_relevant(doc: Document, gold: set[tuple[str, int | None]]) -> bool:
    """Is this retrieved chunk one of the gold units?"""
    file_name = doc.metadata.get("file_name")
    page = doc.metadata.get("page_number")
    return (file_name, page) in gold or (file_name, None) in gold


def _matched_units(
    docs: list[Document], gold: set[tuple[str, int | None]]
) -> set[tuple[str, int | None]]:
    """Which gold units did we actually cover? (Used for recall.)"""
    matched = set()
    for doc in docs:
        file_name = doc.metadata.get("file_name")
        page = doc.metadata.get("page_number")
        if (file_name, page) in gold:
            matched.add((file_name, page))
        elif (file_name, None) in gold:
            matched.add((file_name, None))
    return matched


# ══════════════════════════════════════════════════════════════════
# STAGE 1: Retrieval metrics
# ══════════════════════════════════════════════════════════════════
def retrieval_metrics(
    docs: list[Document],
    gold: set[tuple[str, int | None]],
    cutoffs: tuple[int, ...] = EVAL_K_CUTOFFS,
) -> dict[str, float]:
    """Compute the standard IR metric suite for one query's ranked results."""
    if not gold:
        return {}

    rel_flags = [1 if _is_relevant(d, gold) else 0 for d in docs]
    out: dict[str, float] = {}

    for k in cutoffs:
        top = docs[:k]
        flags = rel_flags[:k]
        out[f"hit@{k}"] = 1.0 if any(flags) else 0.0
        # Precision@k: signal-to-noise of the context window we pay for.
        out[f"precision@{k}"] = sum(flags) / k if k else 0.0
        # Recall@k over gold UNITS (pages), not chunks — a page usually maps to
        # several chunks, so chunk-level recall would need chunk-level labels.
        out[f"recall@{k}"] = len(_matched_units(top, gold)) / len(gold)

    # MRR: 1 / rank of the first relevant document. 0 if none found.
    out["mrr"] = 0.0
    for i, flag in enumerate(rel_flags, start=1):
        if flag:
            out["mrr"] = 1.0 / i
            break

    # nDCG with binary gains. IDCG assumes a perfect ranking could fill every
    # slot with a relevant chunk — reasonable here because a gold page yields
    # multiple chunks. Documented because the assumption changes the number.
    k = max(cutoffs)
    dcg = sum(flag / math.log2(i + 1) for i, flag in enumerate(rel_flags[:k], start=1))
    idcg = sum(1 / math.log2(i + 1) for i in range(1, k + 1))
    out[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0

    return out


# ══════════════════════════════════════════════════════════════════
# STAGE 3: Deterministic behavioural checks
# ══════════════════════════════════════════════════════════════════
CITATION_RE = re.compile(r"\[([^\[\]]+?)[,|]\s*[Pp]ages?\s*(\d+)\s*\]")


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def citation_metrics(answer: str, docs: list[Document]) -> dict[str, Any]:
    """
    Parse citations out of the answer and verify each against the context.

    A citation is *valid* if a chunk with that paper title AND page number was
    genuinely in the retrieved context. Anything else is fabricated — the model
    inventing a plausible-looking reference, which is exactly the failure a
    research assistant must not have.
    """
    retrieved = {
        (_normalise(str(d.metadata.get("paper_title", ""))), str(d.metadata.get("page_number")))
        for d in docs
    }
    # Also allow a title-only match, for when the model cites the right paper
    # but drifts a page (a softer failure worth distinguishing).
    retrieved_titles = {t for t, _ in retrieved}

    found = CITATION_RE.findall(answer)
    valid = 0
    title_only = 0
    for raw_title, page in found:
        norm = _normalise(raw_title)
        # Titles get truncated by the model; accept substring matches both ways.
        exact = any(
            (norm in t or t in norm) and p == page for t, p in retrieved
        )
        loose = any(norm in t or t in norm for t in retrieved_titles)
        if exact:
            valid += 1
        elif loose:
            title_only += 1

    total = len(found)
    return {
        "citations_found": total,
        "citations_valid": valid,
        "citations_wrong_page": title_only,
        "citations_fabricated": total - valid - title_only,
        "citation_precision": (valid / total) if total else None,
        "has_citations": total > 0,
    }


def abstained(answer: str) -> bool:
    low = answer.lower()
    return any(marker in low for marker in ABSTENTION_MARKERS)


def keyword_coverage(answer: str, keywords: list[str]) -> dict[str, Any]:
    """Legacy v1 metric, kept as a cheap deterministic regression tripwire."""
    low = answer.lower()
    hits = [kw for kw in keywords if kw.lower() in low]
    return {
        "matched_keywords": hits,
        "keyword_coverage": len(hits) / len(keywords) if keywords else None,
    }


# ══════════════════════════════════════════════════════════════════
# STAGE 2: LLM-as-judge
# ══════════════════════════════════════════════════════════════════
JUDGE_PROMPT = """You are a strict evaluator of a retrieval-augmented question \
answering system for machine learning research papers. Score the ANSWER using \
ONLY the CONTEXT — you must not use outside knowledge, and you must not reward \
a claim just because you believe it is true.

QUESTION:
{question}

CONTEXT GIVEN TO THE SYSTEM:
{context}

ANSWER PRODUCED BY THE SYSTEM:
{answer}

Score each dimension from 1 to 5:

faithfulness: 5 = every factual claim is directly supported by the context.
  3 = mostly supported but contains at least one claim the context does not
  state. 1 = substantially fabricated or contradicts the context.
relevance: 5 = fully answers exactly what was asked. 3 = partially answers or
  drifts to an adjacent topic. 1 = does not address the question.
completeness: 5 = uses all the answer-relevant material available in the
  context. 3 = misses some available material. 1 = ignores most of it.

Special case: if the context genuinely does not contain the answer and the \
system correctly says so, score faithfulness 5 and relevance 5.

Respond with ONLY a JSON object, no markdown fences, no commentary:
{{"faithfulness": <int>, "relevance": <int>, "completeness": <int>, \
"unsupported_claims": ["..."], "rationale": "<one sentence>"}}"""


def _parse_judge_json(raw: str) -> dict[str, Any]:
    """Judges love markdown fences. Strip them, then parse."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def judge_answer(judge_llm, question: str, answer: str, docs: list[Document]) -> dict[str, Any]:
    """Score one answer. Retries once, then degrades to a null score."""
    context = "\n\n".join(
        f"[{d.metadata.get('paper_title')} | Page {d.metadata.get('page_number')}]\n"
        f"{d.page_content}"
        for d in docs
    )
    prompt = JUDGE_PROMPT.format(question=question, context=context, answer=answer)

    for attempt in range(2):
        try:
            raw = judge_llm.invoke(prompt).content
            parsed = _parse_judge_json(raw if isinstance(raw, str) else str(raw))
            return {
                "faithfulness": int(parsed.get("faithfulness", 0)),
                "relevance": int(parsed.get("relevance", 0)),
                "completeness": int(parsed.get("completeness", 0)),
                "unsupported_claims": parsed.get("unsupported_claims", []),
                "rationale": parsed.get("rationale", ""),
            }
        except Exception as exc:
            logger.warning(f"Judge parse failed (attempt {attempt + 1}): {exc}")
            time.sleep(1)

    return {"faithfulness": None, "relevance": None, "completeness": None,
            "unsupported_claims": [], "rationale": "judge_failed"}


# ══════════════════════════════════════════════════════════════════
# Aggregation helpers
# ══════════════════════════════════════════════════════════════════
def _mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(statistics.fmean(clean), 4) if clean else None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(pct * (len(ordered) - 1))))
    return round(ordered[idx], 3)


# ══════════════════════════════════════════════════════════════════
# Stage 1 runner: retrieval-only ablation
# ══════════════════════════════════════════════════════════════════
def run_retrieval_ablation(
    test_suite: list[dict[str, Any]], modes: tuple[str, ...]
) -> dict[str, Any]:
    """
    Score dense vs sparse vs hybrid on retrieval alone.

    Retrieval is evaluated WITHOUT calling the generator: it's free, it's
    deterministic, and it isolates the variable. Only answerable questions
    have gold labels, so unanswerable ones are excluded here.
    """
    from src.retrieval import get_retriever

    labelled = [q for q in test_suite if q.get("relevant_sources")]
    logger.info(f"Retrieval ablation over {len(labelled)} labelled questions.")

    results: dict[str, Any] = {}
    for mode in modes:
        logger.info(f"  mode = {mode}")
        retriever = get_retriever(
            mode=mode,
            top_k=EVAL_RETRIEVAL_CANDIDATES,
            dense_k=EVAL_RETRIEVAL_CANDIDATES,
            sparse_k=EVAL_RETRIEVAL_CANDIDATES,
        )

        per_question: list[dict[str, Any]] = []
        for item in labelled:
            gold = _gold_units(item)
            t0 = time.perf_counter()
            docs = retriever.invoke(item["question"])
            latency = time.perf_counter() - t0
            metrics = retrieval_metrics(docs, gold)
            metrics["latency_s"] = round(latency, 3)
            per_question.append({"id": item["id"], **metrics})

        # Macro-average: every question weighted equally.
        keys = [k for k in per_question[0] if k != "id"] if per_question else []
        results[mode] = {
            "aggregate": {k: _mean([q[k] for q in per_question]) for k in keys},
            "per_question": per_question,
        }

    return results


# ══════════════════════════════════════════════════════════════════
# Stage 2+3 runner: end-to-end
# ══════════════════════════════════════════════════════════════════
def run_end_to_end(
    test_suite: list[dict[str, Any]], mode: str, use_judge: bool
) -> dict[str, Any]:
    """Run the full chain per question and score generation + behaviour."""
    from src.pipeline import answer_question, get_judge_llm, get_query_engine

    engine = get_query_engine(mode=mode, top_k=TOP_K)
    judge_llm = get_judge_llm() if use_judge else None

    details: list[dict[str, Any]] = []
    latencies: list[float] = []

    for item in test_suite:
        q_id, question = item["id"], item["question"]
        q_type = item.get("type", "answerable")
        print(f"\n[Evaluating {q_id} ({q_type})]: {question}")

        try:
            t0 = time.perf_counter()
            # session_id=None -> stateless, so questions don't contaminate
            # each other. Conversation memory gets its own targeted test below.
            out = answer_question(question, session_id=None, engine=engine)
            latency = time.perf_counter() - t0
            latencies.append(latency)

            answer, docs = out["answer"], out["documents"]
            gold = _gold_units(item)

            record: dict[str, Any] = {
                "id": q_id,
                "question": question,
                "type": q_type,
                "answer": answer,
                "latency_s": round(latency, 3),
                "retrieved": [
                    f"{d.metadata.get('file_name')} p{d.metadata.get('page_number')}"
                    for d in docs
                ],
                **retrieval_metrics(docs, gold, cutoffs=(TOP_K,)),
                **citation_metrics(answer, docs),
                **keyword_coverage(answer, item.get("expected_keywords", [])),
            }

            # Negative control: unanswerable questions MUST be refused.
            record["abstained"] = abstained(answer)
            if q_type == "unanswerable":
                record["abstention_correct"] = record["abstained"]
            else:
                # Abstaining on an answerable question is a false negative.
                record["abstention_correct"] = not record["abstained"]

            if judge_llm is not None:
                record["judge"] = judge_answer(judge_llm, question, answer, docs)

            details.append(record)

            print(f"  -> retrieved {len(docs)} chunks | "
                  f"recall@{TOP_K}={record.get(f'recall@{TOP_K}', 'n/a')} | "
                  f"citations {record['citations_valid']}/{record['citations_found']} valid")
            if judge_llm is not None:
                j = record["judge"]
                print(f"  -> judge: faith={j['faithfulness']} "
                      f"rel={j['relevance']} comp={j['completeness']}")

        except Exception as exc:
            logger.error(f"  -> Failed to evaluate {q_id}: {exc}")
            details.append({"id": q_id, "question": question, "error": str(exc)})

    ok = [d for d in details if "error" not in d]
    aggregate: dict[str, Any] = {
        "questions": len(test_suite),
        "evaluated": len(ok),
        f"recall@{TOP_K}": _mean([d.get(f"recall@{TOP_K}") for d in ok]),
        f"precision@{TOP_K}": _mean([d.get(f"precision@{TOP_K}") for d in ok]),
        "mrr": _mean([d.get("mrr") for d in ok]),
        "citation_precision": _mean([d.get("citation_precision") for d in ok]),
        "answers_with_citations": _mean([1.0 if d.get("has_citations") else 0.0 for d in ok]),
        "fabricated_citations_total": sum(d.get("citations_fabricated", 0) for d in ok),
        "keyword_coverage": _mean([d.get("keyword_coverage") for d in ok]),
        "abstention_accuracy": _mean([1.0 if d.get("abstention_correct") else 0.0 for d in ok]),
        "latency_p50_s": _percentile(latencies, 0.50),
        "latency_p95_s": _percentile(latencies, 0.95),
    }

    if use_judge:
        aggregate["faithfulness"] = _mean([d.get("judge", {}).get("faithfulness") for d in ok])
        aggregate["relevance"] = _mean([d.get("judge", {}).get("relevance") for d in ok])
        aggregate["completeness"] = _mean([d.get("judge", {}).get("completeness") for d in ok])

    return {"mode": mode, "aggregate": aggregate, "details": details}


# ══════════════════════════════════════════════════════════════════
# Targeted test: does conversation memory actually work?
# ══════════════════════════════════════════════════════════════════
def run_memory_check(mode: str) -> dict[str, Any]:
    """
    A follow-up question with a dangling pronoun is un-retrievable on its own.
    If the history-aware rewriting works, turn 2 retrieves QLoRA content even
    though the word "QLoRA" never appears in turn 2.
    """
    from src.pipeline import answer_question, get_query_engine

    engine = get_query_engine(mode=mode, top_k=TOP_K)
    session_id = "eval-memory-probe"

    turn1 = answer_question(
        "What is QLoRA and what problem does it solve?", session_id=session_id, engine=engine
    )
    turn2 = answer_question(
        "How much memory does it save compared to standard finetuning?",
        session_id=session_id,
        engine=engine,
    )

    files = {d.metadata.get("file_name") for d in turn2["documents"]}
    return {
        "turn2_question": turn2["question"],
        "turn2_retrieved_files": sorted(files),
        # The actual assertion: the follow-up pulled the right paper without
        # naming it, which is only possible if history was used to rewrite it.
        "resolved_pronoun_correctly": "qlora.pdf" in files,
        "turn2_answer": turn2["answer"][:400],
    }


# ══════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════
def write_markdown_report(summary: dict[str, Any], path: Path) -> None:
    lines = ["# RAG Evaluation Report", "", f"_Generated {summary['timestamp']}_", ""]

    abl = summary.get("retrieval_ablation", {})
    if abl:
        lines += ["## Stage 1 — Retrieval ablation", ""]
        metric_keys = sorted(next(iter(abl.values()))["aggregate"].keys())
        lines.append("| metric | " + " | ".join(abl.keys()) + " |")
        lines.append("|---|" + "---|" * len(abl))
        for m in metric_keys:
            row = [f"{abl[mode]['aggregate'].get(m)}" for mode in abl]
            lines.append(f"| {m} | " + " | ".join(row) + " |")
        lines.append("")

    e2e = summary.get("end_to_end", {})
    if e2e:
        lines += [f"## Stage 2/3 — End-to-end (mode: {e2e['mode']})", ""]
        lines += ["| metric | value |", "|---|---|"]
        for k, v in e2e["aggregate"].items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    mem = summary.get("memory_check")
    if mem:
        lines += ["## Conversation memory probe", "",
                  f"- Follow-up resolved correctly: **{mem['resolved_pronoun_correctly']}**",
                  f"- Files retrieved on turn 2: {', '.join(mem['turn2_retrieved_files'])}", ""]

    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    abl = summary.get("retrieval_ablation", {})
    if abl:
        print("\n-- Stage 1: retrieval ablation (macro-averaged) --")
        headline = [f"recall@{TOP_K}", f"precision@{TOP_K}", "mrr", f"ndcg@{max(EVAL_K_CUTOFFS)}"]
        print(f"{'metric':<16}" + "".join(f"{m:>12}" for m in abl))
        for m in headline:
            row = "".join(f"{str(abl[mode]['aggregate'].get(m)):>12}" for mode in abl)
            print(f"{m:<16}{row}")

    e2e = summary.get("end_to_end", {})
    if e2e:
        print(f"\n-- Stage 2/3: end-to-end ({e2e['mode']}) --")
        for k, v in e2e["aggregate"].items():
            print(f"  {k:<28}: {v}")

    mem = summary.get("memory_check")
    if mem:
        print("\n-- Conversation memory probe --")
        print(f"  follow-up resolved correctly : {mem['resolved_pronoun_correctly']}")


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════
def run_evaluation(
    modes: tuple[str, ...] = EVAL_MODES,
    use_judge: bool = True,
    primary_mode: str = "hybrid",
    limit: int | None = None,
) -> dict[str, Any]:
    questions_file = EVAL_DIR / "test_questions.json"
    if not questions_file.exists():
        logger.error(f"Cannot find eval file at {questions_file}")
        return {}

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(questions_file, "r", encoding="utf-8") as f:
        test_suite = json.load(f)
    if limit:
        test_suite = test_suite[:limit]

    logger.info(f"Loaded {len(test_suite)} questions for evaluation.")
    print("\n" + "=" * 70)
    print("STARTING EVALUATION SUITE")
    print("=" * 70)

    summary: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "top_k": TOP_K,
            "retrieval_candidates": EVAL_RETRIEVAL_CANDIDATES,
            "modes": list(modes),
            "judge_enabled": use_judge,
        },
    }

    if len(modes) > 0:
        summary["retrieval_ablation"] = run_retrieval_ablation(test_suite, modes)

    summary["end_to_end"] = run_end_to_end(test_suite, primary_mode, use_judge)

    try:
        summary["memory_check"] = run_memory_check(primary_mode)
    except Exception as exc:
        logger.warning(f"Memory probe failed: {exc}")

    print_summary(summary)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = EVAL_RESULTS_DIR / f"report_{timestamp}.json"
    md_path = EVAL_RESULTS_DIR / f"report_{timestamp}.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(summary, md_path)

    print(f"\nDetailed report saved to: {json_path}")
    print(f"Readable report saved to: {md_path}")
    return summary


def dump_retrievals(limit: int | None = None) -> None:
    """
    Labelling aid: print the top chunks per question so you can fill in the
    `pages` field of `relevant_sources` by eye. Good retrieval eval is bounded
    by labelling effort, so it is worth making labelling fast.
    """
    from src.retrieval import get_retriever

    with open(EVAL_DIR / "test_questions.json", "r", encoding="utf-8") as f:
        suite = json.load(f)
    retriever = get_retriever(mode="hybrid", top_k=EVAL_RETRIEVAL_CANDIDATES)

    for item in suite[:limit] if limit else suite:
        print(f"\n### {item['id']}: {item['question']}")
        for i, d in enumerate(retriever.invoke(item["question"]), 1):
            snippet = d.page_content[:160].replace("\n", " ")
            print(f"  {i:>2}. {d.metadata.get('file_name')} p{d.metadata.get('page_number')} "
                  f"| {snippet.encode('ascii', 'replace').decode('ascii')}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="RAG evaluation harness")
    parser.add_argument("--modes", nargs="*", default=list(EVAL_MODES),
                        help="Retrieval modes for the Stage 1 ablation.")
    parser.add_argument("--primary-mode", default="hybrid",
                        help="Mode used for the end-to-end run.")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip the LLM-as-judge stage (no extra API cost).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dump-retrievals", action="store_true",
                        help="Print retrieved chunks to help label gold pages.")
    args = parser.parse_args()

    if args.dump_retrievals:
        dump_retrievals(args.limit)
    else:
        run_evaluation(
            modes=tuple(args.modes),
            use_judge=not args.no_judge,
            primary_mode=args.primary_mode,
            limit=args.limit,
        )
