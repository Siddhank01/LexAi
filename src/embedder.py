"""Embedding engine — wraps SentenceTransformer for local inference."""
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, MAX_TEXT_LENGTH

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list, batch_size: int = 32) -> np.ndarray:
    """Encode a list of texts into embeddings."""
    model = get_model()
    truncated = [" ".join(t.split()[:MAX_TEXT_LENGTH]) for t in texts]
    return model.encode(
        truncated, batch_size=batch_size,
        show_progress_bar=len(texts) > 50,
        convert_to_numpy=True
    )


def embed_query(query: str) -> np.ndarray:
    """Encode a single query string."""
    model = get_model()
    truncated = " ".join(query.split()[:MAX_TEXT_LENGTH])
    vec = model.encode([truncated], convert_to_numpy=True).astype("float32")
    import faiss
    faiss.normalize_L2(vec)
    return vec
