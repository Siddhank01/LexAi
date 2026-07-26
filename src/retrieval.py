"""FAISS retrieval wrapper — handles index loading and similarity search."""
import numpy as np, faiss, json
from config import FAISS_INDEX_PATH, CASES_JSON_PATH, EMBEDDINGS_PATH, TOP_K_RETRIEVAL

_index = None
_cases = None


def _load():
    global _index, _cases
    if _index is None:
        _index = faiss.read_index(FAISS_INDEX_PATH)
    if _cases is None:
        with open(CASES_JSON_PATH, encoding="utf-8") as f:
            _cases = json.load(f)


def search_faiss(query_vec: np.ndarray, top_k: int = TOP_K_RETRIEVAL) -> list:
    """
    Search FAISS index with a normalized query vector.

    Args:
        query_vec: (1, dim) float32 numpy array, L2-normalized
        top_k: number of candidates to retrieve

    Returns:
        List of dicts: [{case data + 'faiss_score'}]
    """
    _load()
    distances, indices = _index.search(query_vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(_cases):
            continue
        case = _cases[int(idx)].copy()
        case["faiss_score"] = float(dist)
        case["result_index"] = int(idx)
        results.append(case)

    return results


def get_case_by_index(idx: int) -> dict:
    """Get a single case by its index."""
    _load()
    if 0 <= idx < len(_cases):
        return _cases[idx].copy()
    return {}


def get_total_cases() -> int:
    """Return the total number of indexed cases."""
    _load()
    return len(_cases)
