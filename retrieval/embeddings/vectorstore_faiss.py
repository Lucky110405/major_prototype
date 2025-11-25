# ingestion/retrieval/embeddings/vectorstore_faiss.py

import os
import faiss
import numpy as np
from typing import List, Tuple
from .vectorstore_interface import VectorStoreInterface

class FaissVectorStore(VectorStoreInterface):
    def __init__(self, dim: int, index_path: str):
        self.dim = dim
        self.index_path = index_path
        self.index = None
        self.id_map = []
        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            map_path = self.index_path + ".ids.npy"
            if os.path.exists(map_path):
                self.id_map = list(np.load(map_path, allow_pickle=True))
            else:
                self.id_map = []
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.id_map = []

    def _save(self):
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        np.save(self.index_path + ".ids.npy", np.array(self.id_map, dtype=object), allow_pickle=True)

    def upsert(self, items: List[dict]):
        # normalize embeddings and append new ones; if updates exist, mark and rebuild
        embeddings = []
        ids_to_append = []
        existing = {cid: idx for idx, cid in enumerate(self.id_map)}
        rebuild_needed = False

        for it in items:
            emb = np.array(it["embedding"], dtype=np.float32)
            # normalize
            norm = np.linalg.norm(emb)
            if norm == 0:
                emb = emb
            else:
                emb = emb / norm
            cid = it["id"]
            if cid in existing:
                # mark rebuild
                rebuild_needed = True
                # update metadata should be handled in MetadataStore; we rebuild full index later
            else:
                embeddings.append(emb)
                ids_to_append.append(cid)

        if embeddings:
            arr = np.vstack(embeddings)
            self.index.add(arr)
            self.id_map.extend(ids_to_append)

        if rebuild_needed:
            # caller should rebuild index from metadata store (we can't fetch metadata here)
            # For simplicity, we raise a flag (or you can implement a rebuild() method)
            # here we just save current state; user can call rebuild_index_from_store later if needed.
            pass

        self._save()

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
        q = query_vector.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-10)
        D, I = self.index.search(np.expand_dims(q, axis=0), top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.id_map):
                continue
            cid = self.id_map[idx]
            if cid is None:
                continue
            results.append((cid, float(score)))
        return results

    def delete(self, ids: List[str]):
        # mark removals and require rebuild
        self.id_map = [cid for cid in self.id_map if cid not in set(ids)]
        # caller should rebuild index from metadata store
        self._save()
