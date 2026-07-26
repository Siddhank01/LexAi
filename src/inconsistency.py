"""Gap detection — identifies verdict inconsistencies within clusters."""
import json, numpy as np
from collections import defaultdict
from config import CASES_JSON_PATH, LABELS_PATH, GAPS_PATH


def detect_gaps(cases=None, labels=None):
    """
    Find clusters where similar cases have opposing verdicts.

    Returns list of gap dicts sorted by inconsistency_score (desc).
    """
    if cases is None:
        with open(CASES_JSON_PATH, encoding="utf-8") as f:
            cases = json.load(f)
    if labels is None:
        labels = np.load(LABELS_PATH)

    groups = defaultdict(list)
    for case, label in zip(cases, labels):
        lid = int(label)
        if lid != -1:
            groups[lid].append(case)

    gaps = []
    for cid, members in groups.items():
        verdict_counts = defaultdict(int)
        for m in members:
            verdict_counts[m["verdict"]] += 1

        granted = verdict_counts.get("bail_granted", 0) + verdict_counts.get("acquitted", 0)
        rejected = verdict_counts.get("bail_rejected", 0) + verdict_counts.get("convicted", 0)

        if not granted or not rejected:
            continue

        total = len(members)
        score = round(min(granted, rejected) / total, 3)

        all_sections = []
        for m in members:
            all_sections.extend(m["ipc_sections"])
        common_sections = sorted(
            set(all_sections), key=lambda s: -all_sections.count(s)
        )[:5]

        gaps.append({
            "cluster_id": cid,
            "total_cases": total,
            "positive_outcome_count": granted,
            "negative_outcome_count": rejected,
            "convicted_count": verdict_counts.get("convicted", 0),
            "acquitted_count": verdict_counts.get("acquitted", 0),
            "bail_granted_count": verdict_counts.get("bail_granted", 0),
            "bail_rejected_count": verdict_counts.get("bail_rejected", 0),
            "inconsistency_score": score,
            "common_ipc_sections": common_sections,
            "dominant_case_type": max(
                set(m["case_type"] for m in members),
                key=lambda t: sum(1 for m in members if m["case_type"] == t)
            ),
            "courts_involved": list(set(m["court"] for m in members))[:5],
        })

    return sorted(gaps, key=lambda x: -x["inconsistency_score"])


if __name__ == "__main__":
    gaps = detect_gaps()
    with open(GAPS_PATH, "w", encoding="utf-8") as f:
        json.dump(gaps, f, indent=2)
    print(f"Detected {len(gaps)} inconsistent clusters.")
    for g in gaps:
        print(f"  Cluster {g['cluster_id']}: {g['inconsistency_score']:.0%} inconsistency "
              f"({g['total_cases']} cases)")
