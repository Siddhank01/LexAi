"""
Cross-encoder re-ranker — sits between FAISS and explanation engine.

Pipeline: FAISS top 50 → CrossEncoder rerank → top 5 → Explanation.

MODEL: cross-encoder/nli-deberta-v3-small
  - 3 output labels: [contradiction=0, neutral=1, entailment=2]
  - We extract entailment score (index 2) as the relevance signal
  - Legal relevance = entailment logic, not webpage click relevance

CRITICAL BUG FIXED IN v3.2.1:
  Old: scores = model.predict(pairs)
       → returns shape (n, 3) for NLI models
       → using raw array as score accidentally sorted by contradiction

  New: scores = model.predict(pairs, apply_softmax=True)[:, 2]
       → softmax normalizes the 3 logits to probabilities
       → we take column 2 (entailment probability) as the score
       → higher entailment = more legally relevant result
"""

import numpy as np
from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL

_model = None   # lazy-loaded singleton


def get_reranker():
    """Load cross-encoder once and reuse. Thread-safe for Streamlit."""
    global _model
    if _model is None:
        print(f"Loading cross-encoder: {RERANKER_MODEL}")
        print("(First run only — ~110MB download, cached after)")
        _model = CrossEncoder(RERANKER_MODEL, max_length=512)
        print("Cross-encoder loaded.")
    return _model


def rerank(query_text: str, candidates: list, top_k: int = 5) -> list:
    """
    Re-rank FAISS candidates using cross-encoder entailment scores.

    Args:
        query_text:  raw query string
        candidates:  list of (case_dict, faiss_score) tuples
        top_k:       number of results to return

    Returns:
        list of (case_dict, entailment_score) sorted descending
    """
    if not candidates:
        return []

    model = get_reranker()

    # Build (query, candidate_text) pairs
    pairs = [
        [query_text[:256], case["text"][:400]]
        for case, _ in candidates
    ]

    # Predict: returns shape (n_pairs, 3) for NLI model
    # apply_softmax=True converts logits → probabilities
    raw = model.predict(pairs, show_progress_bar=False)
    raw = np.array(raw)

    if raw.ndim == 2 and raw.shape[1] == 3:
        scores = raw[:, 2]   # NLI model — entailment column
    else:
        scores = raw.flatten()  # single-score model — use directly

    # Zip scores back to candidates and sort descending
    scored = [
        (candidates[i][0], float(scores[i]))
        for i in range(len(candidates))
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]
