---
title: LexAI — Indian Court Judgment AI
emoji: ⚖️
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
short_description: Semantic search + legal gap detection for Indian court judgments
---

# ⚖️ LexAI — Indian Court Judgment AI

Semantic legal research powered by **LegalBERT + FAISS**.  
Find similar past Indian court judgments, understand why they match, and detect verdict inconsistencies.

- **5,007** Indian court judgments indexed  
- **~270ms** search latency (FAISS direct retrieval)  
- **MRR@5: 0.5269** · **NDCG@5: 0.5746**  
- Zero LLM inference — fully deterministic explanations

> ⚠️ First startup may take 2–3 minutes while data files are downloaded from HuggingFace Datasets.
