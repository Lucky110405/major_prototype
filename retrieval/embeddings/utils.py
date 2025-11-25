# ingestion/retrieval/embeddings/utils.py

import numpy as np

def to_numpy(vec):
    if isinstance(vec, list):
        return np.array(vec, dtype=np.float32)
    if isinstance(vec, np.ndarray):
        return vec.astype(np.float32)
    raise TypeError("Unsupported vector type")

def normalize(vec):
    a = to_numpy(vec)
    n = np.linalg.norm(a)
    if n == 0:
        return a
    return a / n
