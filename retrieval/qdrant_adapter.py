# qdrant_adapter.py
"""
Qdrant Vector Store Adapter for OSURA
------------------------------------
This module provides a clean wrapper around Qdrant for:
- Creating a collection
- Upserting vectors
- Searching vectors
- Deleting and updating points

Dependencies:
    pip install qdrant-client

Usage:
    from qdrant_adapter import QdrantAdapter

    qa = QdrantAdapter(collection_name="osura_docs")
    qa.connect_local()  # or connect_remote(url, api_key)
    qa.create_collection(vector_size=768)

    qa.upsert(points=[
        {
            "id": 1,
            "embedding": [...],
            "payload": {"text": "Hello World", "source": "pdf1.pdf"}
        }
    ])

    results = qa.search(query_vector=[...], top_k=5)
"""

from typing import List, Dict, Optional, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


class QdrantAdapter:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.client: Optional[QdrantClient] = None

    # ------------------------------------------------
    # CONNECTION METHODS
    # ------------------------------------------------
    def connect_local(self, path: str = "./qdrant_data"):
        """Connect to local Qdrant (docker or local folder)."""
        self.client = QdrantClient(path=path)
        print(f"Connected to local Qdrant @ {path}")

    def connect_remote(self, url: str, api_key: str):
        """Connect to remote Qdrant Cloud or hosted instance."""
        self.client = QdrantClient(url=url, api_key=api_key)
        print(f"Connected to remote Qdrant @ {url}")

    # ------------------------------------------------
    # COLLECTION MANAGEMENT
    # ------------------------------------------------
    def create_collection(self, vector_size: int, distance: str = "Cosine"):
        """Creates a vector collection if not exists."""
        assert self.client is not None, "Call connect_local() or connect_remote() first"

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=distance)
        )
        print(f"Collection '{self.collection_name}' created with vector size {vector_size}")

    def delete_collection(self):
        """Deletes the entire collection."""
        assert self.client is not None
        self.client.delete_collection(self.collection_name)
        print(f"Collection '{self.collection_name}' deleted")

    # ------------------------------------------------
    # UPSERT DOCUMENTS
    # ------------------------------------------------
    def upsert(self, points: List[Dict[str, Any]]):
        """Upserts points into Qdrant.
        points = [{"id": int/str, "embedding": [...], "payload": {...}}]
        """
        assert self.client is not None

        qdrant_points = [
            qmodels.PointStruct(
                id=p["id"], vector=p["embedding"], payload=p.get("payload", {})
            )
            for p in points
        ]

        self.client.upsert(collection_name=self.collection_name, points=qdrant_points)
        print(f"Upserted {len(points)} points into '{self.collection_name}'")

    # ------------------------------------------------
    # SEARCH API
    # ------------------------------------------------
    def search(self, query_vector: List[float], top_k: int = 5, filters: Dict = None):
        """Semantic vector search with optional metadata filters."""
        assert self.client is not None

        q_filter = self._build_filter(filters) if filters else None

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=q_filter,
        )

        return [
            {
                "id": r.id,
                "score": r.score,
                "payload": r.payload
            }
            for r in results
        ]

    # ------------------------------------------------
    # DELETE POINTS
    # ------------------------------------------------
    def delete_points(self, ids: List[int]):
        assert self.client is not None
        self.client.delete(collection_name=self.collection_name, points_selector=qmodels.PointIdsList(points=ids))
        print(f"Deleted points: {ids}")

    # ------------------------------------------------
    # FILTER HELPERS
    # ------------------------------------------------
    def _build_filter(self, filters: Dict) -> qmodels.Filter:
        """
        Convert Python dict → Qdrant filter object.
        Example:
            filters = {"source": "chapter1.pdf"}
        """
        conditions = []
        for key, value in filters.items():
            conditions.append(
                qmodels.FieldCondition(
                    key=key, match=qmodels.MatchValue(value=value)
                )
            )

        return qmodels.Filter(must=conditions)
