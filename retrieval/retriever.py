# retrieval/retriever.py
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Expect a VectorStore adapter implementing .search(query_vector, top_k, filter=None) -> List[(id,score)]
# and a MetadataStore adapter implementing .get(id) -> record with text/metadata

class DenseRetriever:
    def __init__(self, vectorstore, metadata_store, embed_fn):
        """
        vectorstore: adapter (Faiss/Qdrant/Chroma) with .search(query_vector, top_k, filter=None)
        metadata_store: adapter with .get(id) to fetch text + metadata
        embed_fn: function that converts list[str] -> list[vector]  (embedding provider)
        """
        self.vs = vectorstore
        self.meta = metadata_store
        self.embed_fn = embed_fn

    def retrieve(self, query: str, top_k: int = 10, metadata_filter: Optional[Dict]=None) -> List[Dict[str,Any]]:
        """
        Returns top_k context dicts: {id, score, text, metadata}
        metadata_filter: optional dict to filter by payload (e.g., {"source":"sample.pdf", "page": 3})
        """
        q_vec = np.array(self.embed_fn([query])[0], dtype=np.float32)
        hits = self.vs.search(q_vec, top_k=top_k, filter=metadata_filter)

        results = []
        for hid, score in hits:
            rec = self.meta.get(hid)
            if rec is None:
                continue
            results.append({
                "id": hid,
                "score": score,
                "text": rec.get("text"),
                "metadata": rec.get("metadata", {})
            })
        return results
