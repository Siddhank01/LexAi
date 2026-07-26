"""Generate meaningful cluster topic labels using TF-IDF.

Replaces failed BERTopic labels with reliable keyword extraction.
"""
import json, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from config import CASES_JSON_PATH, LABELS_PATH, TOPICS_PATH


def generate_cluster_topics():
    with open(CASES_JSON_PATH) as f:
        cases = json.load(f)
    labels = np.load(LABELS_PATH)

    cluster_texts = defaultdict(list)
    cluster_meta = defaultdict(lambda: defaultdict(int))

    for case, label in zip(cases, labels):
        lid = int(label)
        if lid == -1:
            continue
        cluster_texts[lid].append(case["text"][:1000])
        # Track common IPC sections and crime types
        for s in case.get("ipc_sections", []):
            cluster_meta[lid][f"IPC {s}"] += 1
        crime = case.get("crime_type", "").strip()
        if crime:
            cluster_meta[lid][crime] += 1

    cluster_topics = {}
    for cid, texts in cluster_texts.items():
        # TF-IDF keywords
        tfidf = TfidfVectorizer(
            stop_words="english", max_features=200,
            ngram_range=(1, 2), min_df=2
        )
        try:
            matrix = tfidf.fit_transform(texts)
            mean_scores = matrix.mean(axis=0).A1
            feature_names = tfidf.get_feature_names_out()
            top_indices = mean_scores.argsort()[-5:][::-1]
            keywords = [feature_names[i] for i in top_indices]
        except Exception:
            keywords = []

        # Top IPC sections and crime types
        meta_sorted = sorted(
            cluster_meta[cid].items(), key=lambda x: -x[1]
        )[:3]
        meta_labels = [f"{k}" for k, v in meta_sorted if v >= 2]

        # Combine: IPC/crime context + TF-IDF keywords
        parts = []
        if meta_labels:
            parts.append(" | ".join(meta_labels))
        if keywords:
            parts.append(" · ".join(keywords[:3]))

        cluster_topics[str(cid)] = " — ".join(parts) if parts else f"Cluster {cid}"

    with open(TOPICS_PATH, "w", encoding="utf-8") as f:
        json.dump(cluster_topics, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(cluster_topics)} cluster labels:")
    for cid, label in sorted(cluster_topics.items(), key=lambda x: int(x[0])):
        print(f"  Cluster {cid}: {label}")

    return cluster_topics


if __name__ == "__main__":
    generate_cluster_topics()
