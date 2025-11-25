# ingestion/retrieval/embeddings/vectorstore_interface.py

from typing import List, Tuple
import numpy as np
from abc import ABC, abstractmethod

class VectorStoreInterface(ABC):
    @abstractmethod
    def upsert(self, items: List[dict]):
        """
        items: [{id, embedding (np.array or list), metadata (dict)}...]
        """
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        returns list of (id, score)
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]):
        pass
