# retrieval/hybrid_retriever.py
from typing import List, Dict, Any, Optional
import numpy as np

class HybridRetriever:
    def __init__(self, dense_retriever, bm25_index=None, fusion_weights=(0.6, 0.4)):
        """
        fusion_weights: (dense_weight, bm25_weight)
        """
        self.dense = dense_retriever
        self.bm25 = bm25_index
        self.dw, self.bw = fusion_weights

    def search(self, query: str, top_k: int = 10, metadata_filter: Optional[Dict]=None) -> List[Dict[str,Any]]:
        dense_results = self.dense.retrieve(query, top_k=top_k*2, metadata_filter=metadata_filter)
        dense_map = {r["id"]: r for r in dense_results}
        combined = []

        if self.bm25:
            bm25_results = self.bm25.query(query, top_k=top_k*2)
            for r in bm25_results:
                rid = r["id"]
                bm_score = r["score"]
                dens = dense_map.get(rid)
                dens_score = dens["score"] if dens else 0.0
                fused = self.dw * dens_score + self.bw * bm_score
                combined.append({
                    "id": rid,
                    "score": fused,
                    "text": dens["text"] if dens else r["text"],
                    "metadata": dens["metadata"] if dens else {}
                })
            # also include dense-only ids not in BM25
            for rid, dens in dense_map.items():
                if rid in [c["id"] for c in combined]:
                    continue
                combined.append({
                    "id": rid,
                    "score": self.dw * dens["score"],
                    "text": dens["text"],
                    "metadata": dens["metadata"]
                })
        else:
            # only dense
            combined = dense_results

        # sort and trim
        combined_sorted = sorted(combined, key=lambda x: x["score"] or 0.0, reverse=True)
        return combined_sorted[:top_k]
