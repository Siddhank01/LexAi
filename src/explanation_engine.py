"""
Explanation engine — deterministic, template-based structured diff.

NO LLM dependency. No API calls. Always fast. Always consistent.

Exported functions (both required by search_pipeline.py):
  explain_similarity(query_case, retrieved_case, similarity_score) -> dict
  explain_results(query_case, results) -> list[dict]
"""


def explain_similarity(
    query_case: dict,
    retrieved_case: dict,
    similarity_score: float
) -> dict:
    """
    Generate structured explanation comparing query to one retrieved case.

    Args:
        query_case:       NLP-processed dict for the user's query text
        retrieved_case:   NLP-processed dict for a retrieved result
        similarity_score: reranker relevance score (float)

    Returns:
        dict with 10 keys
    """
    # IPC comparison
    q_ipc = set(query_case.get("ipc_sections", []))
    r_ipc = set(retrieved_case.get("ipc_sections", []))
    shared_ipc   = sorted(q_ipc & r_ipc)
    q_only_ipc   = sorted(q_ipc - r_ipc)
    r_only_ipc   = sorted(r_ipc - q_ipc)

    # Evidence comparison
    q_evidence = set(query_case.get("evidence_types", []))
    r_evidence = set(retrieved_case.get("evidence_types", []))
    shared_evidence = sorted(q_evidence & r_evidence)
    q_only_evidence = sorted(q_evidence - r_evidence)
    r_only_evidence = sorted(r_evidence - q_evidence)

    # Case type
    q_type = query_case.get("case_type", "unknown")
    r_type = retrieved_case.get("case_type", "unknown")
    shared_case_type = (q_type == r_type)

    # Court and verdict
    q_court   = query_case.get("court", "unknown")
    r_court   = retrieved_case.get("court", "unknown")
    q_verdict = query_case.get("verdict", "unknown")
    r_verdict = retrieved_case.get("verdict", "unknown")

    # ── Similarity reason ──────────────────────────────────────────────────
    reasons = []
    if shared_ipc:
        reasons.append(f"both cite IPC {', '.join(shared_ipc)}")
    if shared_case_type:
        reasons.append(f"both are {q_type} cases")
    if shared_evidence:
        reasons.append(f"both involve {', '.join(shared_evidence)} evidence")
    if not reasons:
        reasons.append(
            "high semantic similarity in legal language and factual context"
        )
    similarity_reason = "Similarity: " + "; ".join(reasons).capitalize() + "."

    # ── Key differences ────────────────────────────────────────────────────
    diffs = []
    if q_only_ipc:
        diffs.append(f"your case cites IPC {', '.join(q_only_ipc)} (absent here)")
    if r_only_ipc:
        diffs.append(f"this case additionally cites IPC {', '.join(r_only_ipc)}")
    if q_only_evidence:
        diffs.append(
            f"your case has {', '.join(q_only_evidence)} evidence (absent here)"
        )
    if r_only_evidence:
        diffs.append(
            f"this case has {', '.join(r_only_evidence)} evidence (absent in yours)"
        )
    if not shared_case_type:
        diffs.append(
            f"case type differs: yours is {q_type}, this is {r_type}"
        )
    if q_court != r_court and r_court != "unknown":
        diffs.append(f"decided by {r_court}")
    if not diffs:
        diffs.append("no major structural differences detected")
    key_differences = "Differences: " + "; ".join(diffs).capitalize() + "."

    # ── Verdict analysis ───────────────────────────────────────────────────
    if q_verdict == "unknown" or r_verdict == "unknown":
        verdict_analysis = (
            "Verdict comparison: unable to extract verdicts reliably "
            "from one or both cases."
        )
    elif q_verdict == r_verdict:
        verdict_analysis = (
            f"Verdict alignment: both cases resulted in {r_verdict}."
        )
    else:
        verdict_factors = []
        if "forensic" in r_evidence and "forensic" not in q_evidence:
            verdict_factors.append("this case had forensic evidence")
        if "eyewitness" in r_evidence and "eyewitness" not in q_evidence:
            verdict_factors.append("this case had eyewitness testimony")
        if "confession" in r_evidence and "confession" not in q_evidence:
            verdict_factors.append("this case included a confession")
        if "forensic" in q_evidence and "forensic" not in r_evidence:
            verdict_factors.append(
                "your case has forensic evidence this one lacked"
            )
        if verdict_factors:
            verdict_analysis = (
                f"Verdict divergence: your case trends {q_verdict}, "
                f"this case was {r_verdict}. "
                f"Possible factor: {'; '.join(verdict_factors)}."
            )
        else:
            verdict_analysis = (
                f"Verdict divergence: your case trends {q_verdict}, "
                f"this case was {r_verdict}. "
                f"Similar charges led to opposite outcomes — review carefully."
            )

    return {
        "similarity_score":  round(similarity_score, 3),
        "similarity_reason": similarity_reason,
        "key_differences":   key_differences,
        "verdict_analysis":  verdict_analysis,
        "shared_ipc":        shared_ipc,
        "shared_evidence":   shared_evidence,
        "shared_case_type":  shared_case_type,
        "retrieved_verdict": r_verdict,
        "retrieved_court":   r_court,
        "retrieved_date":    retrieved_case.get("date", ""),
    }


def explain_results(query_case: dict, results: list) -> list:
    """
    Run explanation for all reranked results.

    Args:
        query_case: NLP-processed dict for the query
        results:    list of (case_dict, score) tuples from reranker

    Returns:
        list of explanation dicts, one per result
    """
    return [
        explain_similarity(query_case, case, score)
        for case, score in results
    ]
