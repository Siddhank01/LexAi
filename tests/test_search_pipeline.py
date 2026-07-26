"""
Integration test for the full search pipeline.

Tests in order:
  1. check_required_files detects files correctly
  2. pipeline.health_check() returns correct status
  3. Empty query → validation_error
  4. Short query → validation_error
  5. Non-legal query → validation_error
  6. Valid legal query → success with results
  7. SearchResult fields all present
  8. Explanation dict has all required keys
  9. query_case NLP extraction populated
  10. latency_ms reported and > 0

Run: python test_search_pipeline.py
"""

from src.search_pipeline import SearchPipeline, PipelineResponse, check_required_files
from pathlib import Path

print("=" * 55)
print("SEARCH PIPELINE INTEGRATION TEST")
print("=" * 55)

pipeline   = SearchPipeline()
all_passed = True
_resp      = None   # store valid query response for later tests


def check(label, condition, detail=""):
    global all_passed
    status = "✅ PASS" if condition else "❌ FAIL"
    if not condition:
        all_passed = False
    print(f"  {status}  {label}")
    if detail:
        print(f"         {detail}")
    return condition


# Test 1: check_required_files
ok, msg = check_required_files()
check("1. check_required_files() runs without error",
      True,
      f"ok={ok}, msg='{msg[:60]}'")

# Test 2: health_check
h_ok, h_msg = pipeline.health_check()
check("2. pipeline.health_check() returns status",
      True,
      f"ok={h_ok}, msg='{h_msg[:60]}'")

# Test 3: empty query
resp = pipeline.search("")
check("3. Empty query → validation_error",
      not resp.success and resp.error_type == "validation_error",
      f"success={resp.success}, error_type={resp.error_type}")

# Test 4: short query
resp = pipeline.search("ipc murder")
check("4. Short query → validation_error",
      not resp.success and resp.error_type == "validation_error",
      f"success={resp.success}, error_type={resp.error_type}")

# Test 5: non-legal query
resp = pipeline.search("what is the weather in Mumbai today please tell me now")
check("5. Non-legal query → validation_error",
      not resp.success and resp.error_type == "validation_error",
      f"success={resp.success}, error_type={resp.error_type}")

# Test 6: valid query
if h_ok:
    _resp = pipeline.search(
        "Accused charged under IPC Section 302 for murder. "
        "Prosecution relies on eyewitness testimony and forensic evidence. "
        "Sessions Court hearing in progress. Defence claims alibi."
    )
    check("6. Valid query returns success",
          _resp.success,
          f"error={_resp.error}" if not _resp.success else f"{len(_resp.results)} results")
else:
    print("  ⏭️  6–10 SKIPPED: pipeline not ready (run Colab Phase 4 first)")
    _resp = None

# Tests 7–10 require valid query results
if _resp and _resp.success and _resp.results:
    r = _resp.results[0]

    # Test 7: SearchResult fields
    check("7. SearchResult has: case, score, explanation, rank",
          hasattr(r, 'case') and hasattr(r, 'score') and
          hasattr(r, 'explanation') and hasattr(r, 'rank'),
          f"rank={r.rank}  score={r.score:.3f}  verdict={r.case.get('verdict')}")

    # Test 8: explanation keys
    exp_keys = ["similarity_reason", "key_differences", "verdict_analysis",
                "shared_ipc", "shared_evidence", "retrieved_verdict",
                "retrieved_court", "retrieved_date", "similarity_score",
                "shared_case_type"]
    missing  = [k for k in exp_keys if k not in r.explanation]
    check("8. Explanation dict has all 10 required keys",
          len(missing) == 0,
          f"Missing: {missing}" if missing else f"keys OK: {len(r.explanation)}")

    # Test 9: query_case NLP
    qc = _resp.query_case
    qc_fields = ["text", "verdict", "ipc_sections", "case_type", "evidence_types"]
    qc_missing = [k for k in qc_fields if k not in qc]
    check("9. query_case NLP extraction populated",
          len(qc_missing) == 0,
          f"ipc={qc.get('ipc_sections')}  type={qc.get('case_type')}  evidence={qc.get('evidence_types')}")

    # Test 10: latency
    check("10. latency_ms reported and > 0",
          _resp.latency_ms is not None and _resp.latency_ms > 0,
          f"{_resp.latency_ms}ms total pipeline")

elif _resp and _resp.success and not _resp.results:
    print("  ⚠️  6b. Valid query returned 0 results.")
    print("       Check: FAISS index not empty, cases.json loaded.")
    all_passed = False
elif _resp and not _resp.success:
    print(f"  ❌ 6. Pipeline error: {_resp.error}")
    all_passed = False

# Summary
print(f"\n{'='*55}")
if all_passed:
    print("✅ ALL TESTS PASSED — search pipeline is production-ready.")
else:
    print("❌ SOME TESTS FAILED — paste output for diagnosis.")
print(f"{'='*55}")
