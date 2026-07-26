<div align="center">

<img src="https://img.shields.io/badge/Version-v3.4-blue?style=for-the-badge&logo=semantic-release&logoColor=white" alt="Version"/>
<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
<img src="https://img.shields.io/badge/LegalBERT-nlpaueb-8A2BE2?style=for-the-badge&logo=huggingface&logoColor=white" alt="LegalBERT"/>
<img src="https://img.shields.io/badge/FAISS-Meta_AI-00599C?style=for-the-badge&logo=meta&logoColor=white" alt="FAISS"/>
<img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge" alt="License"/>

<br/><br/>

# ⚖️ LexAI — Indian Court Judgment AI

### *Semantic legal research powered by LegalBERT + FAISS — no LLM hallucinations*

**Find similar past judgments · Understand why they match · Detect legal inconsistencies**

<br/>

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-HuggingFace_Spaces-FF4B4B?style=for-the-badge)](https://huggingface.co/spaces/Satyam810/lexai)
[![HuggingFace](https://img.shields.io/badge/🤗_Dataset-HuggingFace-FFD21E?style=for-the-badge)](https://huggingface.co/datasets/Satyam810/lexai-data)
[![GitHub Stars](https://img.shields.io/github/stars/Satyam810/lexai?style=for-the-badge&logo=github&color=gold)](https://github.com/Satyam810/lexai/stargazers)

</div>

---

## 📊 Performance at a Glance

<div align="center">

| Metric | Value | Notes |
|:---|:---:|:---|
| 🗃️ Cases Indexed | **5,007** | Indian High Court + Supreme Court judgments |
| ⚡ Search Latency | **~270ms** | FAISS ANN retrieval (CPU) |
| 🎯 MRR@5 | **0.5269** | Evaluated on 96 real queries |
| 📈 NDCG@5 | **0.5746** | Cross-IPC relevance oracle |
| 🔵 Silhouette Score | **0.834** | HDBSCAN cluster quality (GOOD) |
| 📦 Embedding Dim | **768** | LegalBERT base uncased |
| 🔍 Queries Evaluated | **96** | From 500-case held-out pool |

</div>

---

## ✨ What LexAI Does

<table>
<tr>
<td width="50%">

### 🔍 Semantic Search
Describe your case in plain English. LexAI embeds it with **LegalBERT** and searches 5,007 Indian court judgments using **FAISS ANN** — returning the 5 most similar past cases in under 300ms.

</td>
<td width="50%">

### ⚖️ Explainable Results
Not just "here are similar cases" — every result includes:
- Why each case matches (shared IPC sections, evidence)
- What structurally differs
- Verdict divergence analysis

</td>
</tr>
<tr>
<td>

### 🚨 Legal Gap Detection
Automatically clusters cases and surfaces verdict **inconsistencies** — clusters where identical charges led to opposite outcomes across different courts. Exposes how Indian law is applied unevenly.

</td>
<td>

### 📊 Analytics Dashboard
Interactive **UMAP cluster map**, verdict distribution charts, per-cluster summaries with IPC section breakdown, and retrieval evaluation metrics — all rendered with Plotly.

</td>
</tr>
</table>

> **Zero LLM inference** — All explanation logic is deterministic rule-based. No OpenAI API, no hallucinations, no variable latency.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      USER QUERY                             │
│    "bail application IPC 302 murder, no prior record"       │
└─────────────────────┬───────────────────────────────────────┘
                      │
              ┌───────▼────────┐
              │ Query Validator │  length · language · legal signal
              └───────┬────────┘
                      │
              ┌───────▼──────────┐
              │  NLP Extraction   │  spaCy + regex
              │ IPC · verdict ·   │  section numbers · evidence
              │ case_type ·       │  bail/murder/fraud/NDPS…
              │ evidence_types    │
              └───────┬──────────┘
                      │
              ┌───────▼──────────────────────┐
              │   LegalBERT Embedding         │
              │   nlpaueb/legal-bert-base-    │
              │   uncased  →  768-dim vector  │
              └───────┬──────────────────────┘
                      │
              ┌───────▼──────────────────────┐
              │   FAISS ANN Search            │
              │   top-50 candidates  <0.1ms  │
              │   cosine similarity           │
              └───────┬──────────────────────┘
                      │
              ┌───────▼──────────────────────┐
              │   Explanation Engine          │
              │   deterministic · no LLM      │
              │   IPC match · court level ·   │
              │   evidence delta analysis     │
              └───────┬──────────────────────┘
                      │
              ┌───────▼───────┐
              │  Response      │  top-5 results · <300ms total
              └───────────────┘

─────────────── Background Pipeline (Colab GPU, runs once) ───────────────

  5,007 cases  →  batch LegalBERT embed  →  FAISS index
                →  HDBSCAN clustering    →  UMAP 2D projection
                →  BERTopic labeling     →  Gap detection
```

---

## 📦 Data Sources

| Source | Volume | Content |
|:---|:---:|:---|
| **Indian Kanoon API** | Live | Real current Indian court judgments |
| **pile-of-law/indiankanoon** | HuggingFace corpus | Pre-scraped legal corpus |

**IPC Sections Covered:** 302 (murder), 376 (rape), 420 (fraud), 304B (dowry death), 307 (attempt to murder), 498A (cruelty), 120B (conspiracy), NDPS Act, POCSO Act, and more.

**Courts:** Supreme Court of India, all major High Courts (Delhi, Bombay, Madras, Calcutta, Karnataka, Allahabad, Rajasthan, etc.)

---

## 🛠️ Run Locally

### Prerequisites

| Requirement | Version |
|:---|:---:|
| Python | **3.11** (required for HDBSCAN wheels) |
| RAM | **8 GB+** recommended |
| GPU | Optional (CPU inference works) |

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Satyam810/lexai.git
cd lexai

# 2. Create virtual environment (Python 3.11 required)
py -3.11 -m venv venv311

# Windows
venv311\Scripts\activate

# Mac/Linux
source venv311/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model
python -m spacy download en_core_web_lg

# 5. Configure environment
cp .env.example .env
# Edit .env and add your Indian Kanoon API key
```

### Download Pre-built Data

> The processed data files are hosted on Hugging Face due to GitHub's 100MB file size limit.

```bash
# Option A — Hugging Face CLI
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
# Download cases.json and FAISS index from HF dataset
"

# Option B — Rebuild from scratch (requires API key + Colab GPU for embeddings)
python src/fetcher.py          # fetch raw judgments
python src/nlp_pipeline.py     # extract IPC, verdict, evidence
# Run notebooks/colab_phase4.ipynb on Colab for GPU embeddings
# Download outputs back to data/processed/
```

### Launch

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` 🎉

---

## ☁️ Deploy — Free Options

### Option 1: Streamlit Community Cloud *(Recommended)*

1. **Push to GitHub** — data files are already excluded via `.gitignore`
2. **Host data on Hugging Face Datasets** — upload `cases.json`, `embeddings.npy`, `faiss.index`
3. **Deploy at [share.streamlit.io](https://share.streamlit.io)**
   - Connect your GitHub repo
   - Set `app.py` as the main file
4. **Add secrets** — in App Settings → Secrets:
   ```toml
   INDIAN_KANOON_API_KEY = "your_key_here"
   ```

### Option 2: Hugging Face Spaces (Current Deployment)

LexAI is currently deployed on Hugging Face Spaces.
**[View Live App](https://huggingface.co/spaces/Satyam810/lexai)**

Our architecture for free-tier deployment:
1. **Source Code**: Hosted in the HF Space repo.
2. **ML Artifacts**: Large files (`cases.json`, `faiss.index`, cluster data) are decoupled and hosted in a linked dataset (`Satyam810/lexai-data`).
3. **Cold Start Strategy**: A custom `download_data.py` script fetches the ML artifacts automatically on container boot.
4. **Secrets**: `INDIAN_KANOON_API_KEY` is securely injected via Space Secrets.

> **Free tier:** 2 vCPU + **16 GB RAM** — handles FAISS + Streamlit comfortably.

---

## 📁 Project Structure

```
lexai/
│
├── app.py                      # Streamlit UI (display-only, no ML logic)
├── config.py                   # All paths, model names, hyperparameters
├── requirements.txt            # Python dependencies
├── packages.txt                # System packages (for Streamlit Cloud)
├── .env.example                # Template for environment variables
├── .streamlit/
│   └── config.toml             # Dark theme, primary color config
│
├── src/                        # Core ML pipeline modules
│   ├── search_pipeline.py      # Main search orchestrator
│   ├── embedder.py             # LegalBERT embedding wrapper
│   ├── retrieval.py            # FAISS index load + search
│   ├── query_validator.py      # Input validation (length, language, legal signal)
│   ├── nlp_pipeline.py         # IPC section + verdict + evidence extraction
│   ├── explanation_engine.py   # Deterministic result explanation (no LLM)
│   ├── inconsistency.py        # Legal gap / verdict inconsistency detection
│   ├── eval_pipeline.py        # MRR@5, P@5, NDCG@5 evaluation harness
│   ├── fetcher.py              # Indian Kanoon API + HuggingFace data fetch
│   └── reranker.py             # Cross-encoder (evaluated, disabled in prod)
│
├── data/
│   ├── processed/              # Pipeline outputs (cases.json, FAISS, clusters)
│   │   ├── cases.json          # 5,007 processed judgments [hosted on HF]
│   │   ├── faiss.index         # FAISS ANN index [hosted on HF]
│   │   ├── embeddings.npy      # LegalBERT embeddings [hosted on HF]
│   │   ├── cluster_labels.npy  # HDBSCAN cluster assignments
│   │   ├── cluster_topics.json # BERTopic cluster labels
│   │   ├── coords_2d.npy       # UMAP 2D coordinates for scatter plot
│   │   ├── gaps.json           # Detected verdict inconsistency clusters
│   │   ├── eval_metrics.json   # Clustering evaluation results
│   │   └── eval_metrics_retrieval.json  # MRR/NDCG retrieval results
│   └── judgments.db            # Raw SQLite database [not in git, ~143MB]
│
├── assets/
│   └── styles.css              # Custom dark theme CSS
│
├── notebooks/
│   └── colab_phase4.ipynb      # GPU embedding + clustering pipeline
│
├── scripts/
│   ├── fetch_full_text.py      # Full-text fetcher
│   └── generate_topics.py      # BERTopic topic generation
│
├── tests/                      # pytest test suite
└── docs/                       # Additional documentation
```

---

## 🔬 Evaluation Results

### Retrieval Quality (FAISS-only, 96 queries)

| Metric | Score | Benchmark |
|:---|:---:|:---|
| MRR@5 | **0.5269** | Mean Reciprocal Rank |
| P@5 | **0.3250** | Precision at 5 |
| NDCG@5 | **0.5746** | Normalized DCG |

> **Relevance oracle:** Shared IPC section + same case type (conservative, explainable).

### Clustering Quality (HDBSCAN, 5,007 cases)

| Metric | HDBSCAN | K-Means (k=5) |
|:---|:---:|:---:|
| Silhouette | **0.834** | 0.614 |
| Davies-Bouldin | **0.736** | — |
| Clusters Found | **3** | 5 |

### Cross-Encoder Evaluation

> We tested two cross-encoder rerankers and **disabled both** in production:

| Model | MRR@5 | NDCG@5 | Avg Latency |
|:---|:---:|:---:|:---:|
| FAISS only ✅ | 0.5269 | 0.5746 | **126ms** |
| + ms-marco-MiniLM | 0.5269 | 0.5746 | 807ms |

The cross-encoder added **680ms latency with zero improvement** — ms-marco is trained on web search clicks, not legal text. Disabled until domain fine-tuned.

---

## ⚠️ Known Limitations

| Limitation | Cause | Mitigation |
|:---|:---|:---|
| 47.5% unknown verdicts | Indian Kanoon requires paid API for premium docs (403) | Upgrade to paid tier or expand HF corpus |
| Cross-encoder disabled | ms-marco trained on web queries, not legal text | Fine-tune on Indian legal corpus |
| 3 HDBSCAN clusters | Conservative min_cluster_size=10 on 5k cases | Grows with dataset size |
| Gap detection limited | Low verdict coverage in current dataset | Improves as verdict % increases |

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes with tests
4. Run the test suite: `pytest tests/`
5. Submit a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements.txt
pytest  # run tests

# Lint
flake8 src/ app.py --max-line-length=100
```

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and deploy.  
See [LICENSE](LICENSE) for details.

---

## 👤 Author

<div align="center">

**Satyam Kumar** — AI/ML Engineer

[![GitHub](https://img.shields.io/badge/GitHub-Satyam810-181717?style=for-the-badge&logo=github)](https://github.com/Satyam810)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-satyamlpu-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/satyamlpu/)

</div>

---

<div align="center">

**Built with ❤️ for the Indian legal research community**

*LexAI uses AI to surface patterns — it does not provide legal advice.*  
*Always consult a qualified lawyer for legal decisions.*

<br/>

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![FAISS](https://img.shields.io/badge/FAISS-Meta-0467DF?style=flat-square&logo=meta&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)

</div>
