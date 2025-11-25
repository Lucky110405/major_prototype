# ingestion/retrieval/embeddings/dedupe.py

from typing import List
from .embedder import embed_texts
import numpy as np

def simple_dedupe(chunks: List[dict], threshold: float = 0.95):
    """
    Given chunks [{text, metadata}], return filtered chunks removing near-duplicates by cosine on embeddings.
    Uses embed_texts (may be expensive).
    """
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)
    arr = np.vstack([np.array(v, dtype=np.float32) for v in vectors])
    # normalize
    norms = np.linalg.norm(arr, axis=1, keepdims=True); norms[norms==0]=1.0
    arrn = arr / norms
    keep = []
    for i in range(len(arrn)):
        vi = arrn[i]
        if any(np.dot(vi, arrn[j]) > threshold for j in keep):
            continue
        keep.append(i)
    return [chunks[i] for i in keep]
