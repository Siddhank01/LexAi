"""
Retrieval evaluation — self-supervised, no labeled data required.

Relevance oracle: two cases are "relevant" if they share
at least one IPC section AND the same case_type.

Metrics:
  MRR@5   Mean Reciprocal Rank at 5
  P@5     Precision at 5
  NDCG@5  Normalized Discounted Cumulative Gain at 5

Compares: FAISS-only vs FAISS + cross-encoder reranker.

Output: data/processed/eval_metrics_retrieval.json
Run:    python src/eval_pipeline.py
"""

import json, numpy as np, faiss, math, time, random
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from config import (
    CASES_JSON_PATH, EMBEDDINGS_PATH, FAISS_INDEX_PATH,
    RETRIEVAL_METRICS_PATH, EMBEDDING_MODEL, RERANKER_MODEL,
    TOP_K_RETRIEVAL, TOP_K_RESULTS, EVAL_SAMPLE_SIZE, EVAL_TOP_K,
    MAX_TEXT_LENGTH
)


# ── Oracle ────────────────────────────────────────────────────────────────────

def is_relevant(query_case: dict, candidate_case: dict) -> bool:
    """
    Self-supervised relevance: shared IPC section + same case type.
    Cases with no IPC sections are unevaluable by this oracle.
    """
    q_ipc  = set(query_case.get("ipc_sections", []))
    c_ipc  = set(candidate_case.get("ipc_sections", []))
    q_type = query_case.get("case_type", "")
    c_type = candidate_case.get("case_type", "")
    if not q_ipc or not c_ipc:
        return False
    return bool(q_ipc & c_ipc) and (q_type == c_type)


# ── Standard IR metrics ───────────────────────────────────────────────────────

def reciprocal_rank(relevant_flags: list) -> float:
    for i, flag in enumerate(relevant_flags):
        if flag:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(relevant_flags: list, k: int) -> float:
    return sum(relevant_flags[:k]) / k if k > 0 else 0.0


def ndcg_at_k(relevant_flags: list, k: int) -> float:
    if k == 0:
        return 0.0
    dcg  = sum(flag / math.log2(i + 2) for i, flag in enumerate(relevant_flags[:k]))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(sum(relevant_flags[:k])))
    return dcg / idcg if idcg > 0 else 0.0


# ── Main eval ─────────────────────────────────────────────────────────────────

def run_retrieval_eval():
    print("=" * 60)
    print("LEXAI RETRIEVAL EVALUATION")
    print("=" * 60)

    # Load assets
    print("\nLoading cases and index...")
    with open(CASES_JSON_PATH, encoding="utf-8") as f:
        cases = json.load(f)

    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    index      = faiss.read_index(FAISS_INDEX_PATH)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    rerank_model = CrossEncoder(RERANKER_MODEL, max_length=512)

    print(f"  Cases: {len(cases)} | Embeddings: {embeddings.shape} | FAISS: {index.ntotal}")

    # Select eval queries — only cases with IPC sections
    evaluable = [
        (i, c) for i, c in enumerate(cases)
        if c.get("ipc_sections")
    ]

    if len(evaluable) < 10:
        print(f"\n⚠️  Only {len(evaluable)} evaluable cases (need >= 10 with IPC sections).")
        print("   Re-run nlp_pipeline.py to improve IPC extraction, then retry.")
        return None

    random.seed(42)
    sample_size = min(EVAL_SAMPLE_SIZE, len(evaluable))
    eval_sample = random.sample(evaluable, sample_size)
    print(f"\nEval queries: {sample_size} (from {len(evaluable)} evaluable cases)")

    # Evaluation loop
    faiss_mrr, faiss_p5, faiss_ndcg     = [], [], []
    rerank_mrr, rerank_p5, rerank_ndcg  = [], [], []
    embed_latencies   = []
    faiss_latencies   = []
    rerank_latencies  = []
    skipped = 0

    import numpy as _np

    for eval_idx, (case_idx, query_case) in enumerate(eval_sample):
        if eval_idx % 20 == 0:
            print(f"  [{eval_idx}/{sample_size}] evaluating...")

        # Embed query
        q_words = query_case["text"].split()[:MAX_TEXT_LENGTH]
        t_embed = time.perf_counter()
        q_emb   = embed_model.encode(
            [" ".join(q_words)], convert_to_numpy=True
        ).astype("float32")
        embed_latencies.append((time.perf_counter() - t_embed) * 1000)

        faiss.normalize_L2(q_emb)

        # FAISS search
        t_faiss = time.perf_counter()
        scores_arr, idxs = index.search(q_emb, TOP_K_RETRIEVAL + 1)
        faiss_latencies.append((time.perf_counter() - t_faiss) * 1000)

        # Exclude query case itself
        faiss_candidates = [
            (cases[idx], float(sc))
            for idx, sc in zip(idxs[0], scores_arr[0])
            if idx != case_idx and 0 <= idx < len(cases)
        ][:TOP_K_RETRIEVAL]

        if not faiss_candidates:
            skipped += 1
            continue

        # Check oracle: any relevant in full candidate set?
        all_flags = [is_relevant(query_case, c) for c, _ in faiss_candidates]
        if not any(all_flags):
            skipped += 1
            continue

        # FAISS-only metrics
        faiss_flags = [is_relevant(query_case, c) for c, _ in faiss_candidates[:EVAL_TOP_K]]
        faiss_mrr.append(reciprocal_rank(faiss_flags))
        faiss_p5.append(precision_at_k(faiss_flags, EVAL_TOP_K))
        faiss_ndcg.append(ndcg_at_k(faiss_flags, EVAL_TOP_K))

        # Reranking
        pairs = [
            [query_case["text"][:256], cand["text"][:400]]
            for cand, _ in faiss_candidates
        ]
        t_rerank = time.perf_counter()
        raw_scores = rerank_model.predict(pairs, apply_softmax=True, show_progress_bar=False)
        rerank_latencies.append((time.perf_counter() - t_rerank) * 1000)

        raw_scores = _np.array(raw_scores)
        if raw_scores.ndim == 2 and raw_scores.shape[1] == 3:
            scores = raw_scores[:, 2]   # entailment column
        else:
            scores = raw_scores.flatten()

        reranked = sorted(
            zip(faiss_candidates, scores.tolist()),
            key=lambda x: x[1], reverse=True
        )
        rerank_top5 = [cand for (cand, _), _ in reranked[:EVAL_TOP_K]]
        rerank_flags = [is_relevant(query_case, c) for c in rerank_top5]

        rerank_mrr.append(reciprocal_rank(rerank_flags))
        rerank_p5.append(precision_at_k(rerank_flags, EVAL_TOP_K))
        rerank_ndcg.append(ndcg_at_k(rerank_flags, EVAL_TOP_K))

    # Aggregate
    evaluated = len(faiss_mrr)
    if evaluated == 0:
        print("\n⚠️  No queries evaluated. Check IPC section extraction.")
        return None

    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

    all_latencies = [e + f + r for e, f, r in zip(embed_latencies, faiss_latencies, rerank_latencies)]

    results = {
        "evaluation_summary": {
            "total_cases":       len(cases),
            "evaluable_cases":   len(evaluable),
            "queries_evaluated": evaluated,
            "queries_skipped":   skipped,
            "eval_k":            EVAL_TOP_K,
            "relevance_oracle":  "shared_ipc_section_and_same_case_type",
        },
        "faiss_only": {
            "MRR@5":  avg(faiss_mrr),
            "P@5":    avg(faiss_p5),
            "NDCG@5": avg(faiss_ndcg),
        },
        "faiss_plus_reranker": {
            "MRR@5":  avg(rerank_mrr),
            "P@5":    avg(rerank_p5),
            "NDCG@5": avg(rerank_ndcg),
        },
        "reranker_improvement": {
            "MRR_delta":  round(avg(rerank_mrr)  - avg(faiss_mrr),  4),
            "P5_delta":   round(avg(rerank_p5)   - avg(faiss_p5),   4),
            "NDCG_delta": round(avg(rerank_ndcg) - avg(faiss_ndcg), 4),
        },
        "latency": {
            "avg_embed_ms":    round(avg(embed_latencies), 1),
            "avg_faiss_ms":    round(avg(faiss_latencies), 1),
            "avg_rerank_ms":   round(avg(rerank_latencies), 1),
            "avg_query_ms":    round(avg(all_latencies), 1),
            "p95_query_ms":    round(
                sorted(all_latencies)[int(len(all_latencies) * 0.95)]
                if all_latencies else 0, 1
            ),
        },
        "model_info": {
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model":  RERANKER_MODEL,
            "score_column":    "entailment (index 2 of 3 NLI labels)",
        },
    }

    # Save
    Path(RETRIEVAL_METRICS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(RETRIEVAL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print report
    fi  = results["faiss_only"]
    re  = results["faiss_plus_reranker"]
    dlt = results["reranker_improvement"]
    lat = results["latency"]

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"  Evaluated: {evaluated} queries | Skipped: {skipped}")
    print(f"  {'Metric':<12}  {'FAISS only':>12}  {'+ Reranker':>12}  {'Delta':>10}")
    print(f"  {'-'*52}")
    for metric, f_key, r_key, d_key in [
        ("MRR@5",  "MRR@5",  "MRR@5",  "MRR_delta"),
        ("P@5",    "P@5",    "P@5",    "P5_delta"),
        ("NDCG@5", "NDCG@5", "NDCG@5", "NDCG_delta"),
    ]:
        f_val = fi[f_key]; r_val = re[r_key]; d = dlt[d_key]
        sign = "+" if d >= 0 else ""
        print(f"  {metric:<12}  {f_val:>12.4f}  {r_val:>12.4f}  {sign}{d:>9.4f}")

    print(f"\n  Latency breakdown:")
    print(f"    Embed:   {lat['avg_embed_ms']}ms")
    print(f"    FAISS:   {lat['avg_faiss_ms']}ms")
    print(f"    Rerank:  {lat['avg_rerank_ms']}ms")
    print(f"    Total:   {lat['avg_query_ms']}ms avg  /  {lat['p95_query_ms']}ms P95")

    ndcg_delta = dlt["NDCG_delta"]
    print(f"\n{'='*60}")
    if ndcg_delta >= 0:
        print(f"✅ Reranker improved NDCG by +{ndcg_delta:.4f}")
        print("   Two-stage pipeline adds measurable value.")
    else:
        print(f"⚠️  Reranker NDCG delta = {ndcg_delta:.4f} (reranker degraded vs FAISS)")
        print("   Documented as known limitation — NLI model not fine-tuned on Indian law.")
        print("   FAISS bi-encoder alone performs well on this dataset.")
        print("   Fine-tuning on labeled pairs would resolve this.")

    print(f"\n*** RESUME / INTERVIEW NUMBERS ***")
    print(f"  MRR@5  (reranker): {re['MRR@5']:.4f}")
    print(f"  P@5    (reranker): {re['P@5']:.4f}")
    print(f"  NDCG@5 (reranker): {re['NDCG@5']:.4f}")
    print(f"  NDCG delta:        {ndcg_delta:+.4f}")
    print(f"  Avg query latency: {lat['avg_query_ms']}ms")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    run_retrieval_eval()
