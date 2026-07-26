import os
from dotenv import load_dotenv
load_dotenv()

INDIAN_KANOON_API_KEY = os.getenv("INDIAN_KANOON_API_KEY", "")
INDIAN_KANOON_BASE_URL = "https://api.indiankanoon.org"

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
DB_PATH = "data/judgments.db"
CASES_JSON_PATH = "data/processed/cases.json"
EMBEDDINGS_PATH = "data/processed/embeddings.npy"
FAISS_INDEX_PATH = "data/processed/faiss.index"
LABELS_PATH = "data/processed/cluster_labels.npy"
TOPICS_PATH = "data/processed/cluster_topics.json"
COORDS_PATH = "data/processed/coords_2d.npy"
GAPS_PATH = "data/processed/gaps.json"
METRICS_PATH = "data/processed/eval_metrics.json"
RETRIEVAL_METRICS_PATH = "data/processed/eval_metrics_retrieval.json"  # NEW v3.2

EMBEDDING_MODEL = "nlpaueb/legal-bert-base-uncased"
SPACY_MODEL = "en_core_web_lg"

# v3.1: nli-deberta instead of ms-marco
# ms-marco trained on Bing web search clicks — wrong domain for law
# nli-deberta trained on NLI entailment — maps to legal reasoning
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# v3.2 Production config: Disable reranker for latency improvements
USE_RERANKER = False
START_WITH_N_CASES = 5000
UMAP_SUBSAMPLE_LIMIT = 2000
MAX_TEXT_LENGTH = 512
BATCH_SIZE = 32
HDBSCAN_MIN_CLUSTER_SIZE = 10
MIN_K = 5
MAX_K = 20
RANDOM_STATE = 42
TOP_K_RETRIEVAL = 50
TOP_K_RESULTS = 5

# v3.1: query validation thresholds
QUERY_MIN_CHARS = 20
QUERY_MAX_CHARS = 5000
QUERY_MIN_WORDS = 4

# v3.2: retrieval evaluation settings
EVAL_SAMPLE_SIZE = 100      # how many cases to use as eval queries
EVAL_TOP_K = 5              # k for MRR@k, P@k, NDCG@k

APP_TITLE = "LexAI — Court Judgment Analyzer"
GDRIVE_OUTPUT_PATH = os.getenv(
    "GDRIVE_OUTPUT_PATH",
    "/content/drive/MyDrive/lexai_outputs"
)
