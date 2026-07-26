"""
Search pipeline orchestrator — single entry point for all search.

Usage in app.py:
    from src.search_pipeline import SearchPipeline
    pipeline = SearchPipeline()
    ok, msg = pipeline.health_check()
    response = pipeline.search("your query", top_k=5)

Architecture:
    validate → NLP → embed → FAISS → rerank → explain
    All ML code lives HERE. app.py is UI only.

Runtime safety:
    Missing files → clear error message, not traceback.
    health_check() lets app.py show a banner before first search.
"""

import json, numpy as np, faiss, logging, time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from config import (
    CASES_JSON_PATH, EMBEDDINGS_PATH, FAISS_INDEX_PATH,
    EMBEDDING_MODEL, TOP_K_RETRIEVAL, TOP_K_RESULTS, MAX_TEXT_LENGTH
)
from src.query_validator import validate_query
from src.nlp_pipeline import (
    clean_text, extract_verdict, extract_ipc_sections,
    extract_case_type, extract_entities, extract_evidence_types
)
from src.reranker import rerank
from src.explanation_engine import explain_results

log = logging.getLogger(__name__)

# ── Required files + error messages ─────────────────────────────────────────

REQUIRED_FILES = {
    "cases":       CASES_JSON_PATH,
    "embeddings":  EMBEDDINGS_PATH,
    "faiss_index": FAISS_INDEX_PATH,
}

SETUP_INSTRUCTIONS = {
    "cases": (
        "data/processed/cases.json not found.\n"
        "Run: python src/fetcher.py  then  python src/nlp_pipeline.py"
    ),
    "embeddings": (
        "data/processed/embeddings.npy not found.\n"
        "Run the Colab notebook (Phase 4) and download all outputs."
    ),
    "faiss_index": (
        "data/processed/faiss.index not found.\n"
        "Run the Colab notebook (Phase 4) and download all outputs."
    ),
}


def check_required_files() -> tuple:
    """
    Check all required files exist before loading any model.
    Returns (ok: bool, error_message: str).
    """
    for name, path in REQUIRED_FILES.items():
        if not Path(path).exists():
            return False, SETUP_INSTRUCTIONS[name]
    return True, ""


# ── Result dataclasses ───────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """One result from the search pipeline."""
    case:        dict
    score:       float
    explanation: dict
    rank:        int


@dataclass
class PipelineResponse:
    """Full response from pipeline.search()."""
    success:    bool
    query_case: Optional[dict]  = None
    results:    list            = field(default_factory=list)
    error:      Optional[str]   = None
    error_type: Optional[str]   = None
    latency_ms: Optional[float] = None


# ── Pipeline class ───────────────────────────────────────────────────────────

class SearchPipeline:
    """
    Full LexAI search pipeline with lazy-loaded singleton resources.

    Create once per Streamlit session (@st.cache_resource).
    """

    def __init__(self):
        self._cases       = None
        self._index       = None
        self._embed_model = None
        self._ready       = False
        self._init_error  = None

        files_ok, file_error = check_required_files()
        if not files_ok:
            self._init_error = file_error
            log.error(f"SearchPipeline init: {file_error}")
        else:
            self._ready = True

    def health_check(self) -> tuple:
        """
        Returns (ok: bool, message: str).
        Call from app.py startup to show warning banner if setup incomplete.
        """
        if self._init_error:
            return False, self._init_error
        files_ok, file_error = check_required_files()
        if not files_ok:
            return False, file_error
        return True, "Pipeline ready."

    def _load_assets(self):
        """Lazy-load heavy assets on first search call."""
        if self._cases is not None:
            return

        log.info("Loading search assets (first call)...")

        with open(CASES_JSON_PATH, encoding="utf-8") as f:
            self._cases = json.load(f)

        self._index = faiss.read_index(FAISS_INDEX_PATH)

        from sentence_transformers import SentenceTransformer
        self._embed_model = SentenceTransformer(EMBEDDING_MODEL)

        log.info(
            f"Assets loaded: {len(self._cases)} cases, "
            f"{self._index.ntotal} FAISS vectors."
        )

    def _build_query_case(self, query_text: str) -> dict:
        """Run NLP extraction on the raw query text."""
        clean = clean_text(query_text)
        return {
            "text":           clean,
            "verdict":        extract_verdict(clean),
            "ipc_sections":   extract_ipc_sections(clean),
            "case_type":      extract_case_type(clean),
            "entities":       extract_entities(clean),
            "evidence_types": extract_evidence_types(clean),
            "court":          "query",
            "date":           "",
        }

    def search(
        self,
        query_text:     str,
        top_k:          int = TOP_K_RESULTS,
        verdict_filter: str = "All"
    ) -> PipelineResponse:
        """
        Full pipeline: validate → NLP → embed → FAISS → rerank → explain.

        Args:
            query_text:     raw text from the user
            top_k:          number of results to return
            verdict_filter: "All" or a specific verdict label

        Returns:
            PipelineResponse
        """
        t_start = time.time()

        # Step 1: Runtime safety
        if not self._ready:
            return PipelineResponse(
                success=False,
                error=self._init_error,
                error_type="setup_error"
            )

        # Step 2: Validate query
        is_valid, validation_error = validate_query(query_text)
        if not is_valid:
            return PipelineResponse(
                success=False,
                error=validation_error,
                error_type="validation_error"
            )

        # Step 3: Load assets (lazy)
        try:
            self._load_assets()
        except Exception as e:
            log.error(f"Asset load failed: {e}")
            return PipelineResponse(
                success=False,
                error=f"Failed to load search index: {e}",
                error_type="load_error"
            )

        # Step 4: NLP extraction
        query_case = self._build_query_case(query_text)

        # Step 5: Embed + FAISS search
        q_words = query_case["text"].split()[:MAX_TEXT_LENGTH]
        q_emb   = self._embed_model.encode(
            [" ".join(q_words)]
        ).astype("float32")
        faiss.normalize_L2(q_emb)
        scores_arr, idxs = self._index.search(q_emb, TOP_K_RETRIEVAL)

        candidates = []
        for idx, score in zip(idxs[0], scores_arr[0]):
            if 0 <= idx < len(self._cases):
                case = self._cases[idx]
                if verdict_filter == "All" or case.get("verdict") == verdict_filter:
                    candidates.append((case, float(score)))

        if not candidates:
            return PipelineResponse(
                success=True,
                query_case=query_case,
                results=[],
                error=None,
                latency_ms=round((time.time() - t_start) * 1000, 1)
            )

        # Step 6: Reranking
        from config import USE_RERANKER
        if USE_RERANKER:
            reranked = rerank(query_text, candidates, top_k=top_k)
        else:
            reranked = candidates[:top_k]

        # Step 7: Explanation engine
        explanations = explain_results(query_case, reranked)

        results = [
            SearchResult(
                case=case,
                score=score,
                explanation=exp,
                rank=i + 1
            )
            for i, ((case, score), exp) in enumerate(zip(reranked, explanations))
        ]

        latency = round((time.time() - t_start) * 1000, 1)
        log.info(f"Search: {len(results)} results in {latency}ms")

        return PipelineResponse(
            success=True,
            query_case=query_case,
            results=results,
            latency_ms=latency
        )
