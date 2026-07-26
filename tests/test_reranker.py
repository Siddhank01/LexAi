"""
Test: cross-encoder correctly reorders FAISS results.

Wrong case (civil property, FAISS=0.82) must end up BELOW
right case (IPC 302 criminal, FAISS=0.79) after reranking.

Run: python test_reranker.py
"""
from src.reranker import rerank

query = "accused charged under IPC 302 murder with eyewitness and forensic evidence"

fake_candidates = [
    (
        {
            "text": "property dispute partition civil court decree injunction damages",
            "verdict": "acquitted", "ipc_sections": [],
            "case_type": "civil", "evidence_types": [],
            "court": "Delhi HC", "date": "2020"
        },
        0.82   # high FAISS score — wrong case
    ),
    (
        {
            "text": "accused convicted under IPC 302 murder eyewitness testimony "
                    "forensic report confirmed sessions court",
            "verdict": "convicted", "ipc_sections": ["302"],
            "case_type": "criminal", "evidence_types": ["eyewitness", "forensic"],
            "court": "Supreme Court", "date": "2019"
        },
        0.79   # lower FAISS score — correct case
    ),
]

print("Testing cross-encoder reranker (nli-deberta entailment score)...")
print()

results = rerank(query, fake_candidates, top_k=2)

print("Re-ranked order:")
for i, (case, score) in enumerate(results, 1):
    print(f"  #{i}  entailment={score:.4f}  "
          f"case_type={case['case_type']}  "
          f"verdict={case['verdict']}")
    print(f"       {case['text'][:65]}...")

print()
if results[0][0]["case_type"] == "criminal":
    print("✅ PASS: IPC 302 criminal case ranked #1 despite lower FAISS score.")
    print("   Entailment score fix is working correctly.")
else:
    print("❌ FAIL: Civil case ranked #1 — entailment extraction not working.")
    print("   Check: reranker.py uses raw_scores[:, 2] (entailment column)")
    print("   Debug: add print(raw_scores.shape) inside reranker.py predict()")
