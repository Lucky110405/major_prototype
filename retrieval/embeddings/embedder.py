# ingestion/retrieval/embeddings/embedder.py

from typing import List
import numpy as np
import hashlib
from config import EMBEDDER_PROVIDER, LOCAL_EMBED_MODEL, OPENAI_API_KEY, OPENAI_EMBED_MODEL, EMBED_BATCH_SIZE

# Lazy imports for speed
_st_model = None
_openai = None

def _init_st():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(LOCAL_EMBED_MODEL)
    return _st_model

def _init_openai():
    global _openai
    if _openai is None:
        import openai
        openai.api_key = OPENAI_API_KEY
        _openai = openai
    return _openai

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return list of float vectors"""
    if EMBEDDER_PROVIDER == "sentence_transformers":
        model = _init_st()
        arr = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [a.astype('float32').tolist() for a in arr]
    elif EMBEDDER_PROVIDER == "openai":
        openai = _init_openai()
        vectors = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i:i+EMBED_BATCH_SIZE]
            resp = openai.Embedding.create(model=OPENAI_EMBED_MODEL, input=batch)
            vectors.extend([item["embedding"] for item in resp["data"]])
        return vectors
    else:
        raise ValueError(f"Unknown provider {EMBEDDER_PROVIDER}")

def chunk_id(text: str, metadata: dict) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    if metadata:
        for k in sorted(metadata.keys()):
            h.update(f"{k}={metadata[k]}".encode("utf-8"))
    return h.hexdigest()
