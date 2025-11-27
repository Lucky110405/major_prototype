# # qdrant_adapter.py
# """
# Qdrant Vector Store Adapter for OSURA
# ------------------------------------
# This module provides a clean wrapper around Qdrant for:
# - Creating a collection
# - Upserting vectors
# - Searching vectors
# - Deleting and updating points

# Dependencies:
#     pip install qdrant-client

# Usage:
#     from qdrant_adapter import QdrantAdapter

#     qa = QdrantAdapter(collection_name="osura_docs")
#     qa.connect_local()  # or connect_remote(url, api_key)
#     qa.create_collection(vector_size=768)

#     qa.upsert(points=[
#         {
#             "id": 1,
#             "embedding": [...],
#             "payload": {"text": "Hello World", "source": "pdf1.pdf"}
#         }
#     ])

#     results = qa.search(query_vector=[...], top_k=5)
# """

# from typing import List, Dict, Optional, Any
# from qdrant_client import QdrantClient
# from qdrant_client.http import models as qmodels


# class QdrantAdapter:
#     def __init__(self, collection_name: str):
#         self.collection_name = collection_name
#         self.client: Optional[QdrantClient] = None

#     # ------------------------------------------------
#     # CONNECTION METHODS
#     # ------------------------------------------------
#     def connect_local(self, path: str = "./qdrant_data"):
#         """Connect to local Qdrant (docker or local folder)."""
#         self.client = QdrantClient(path=path)
#         print(f"Connected to local Qdrant @ {path}")

#     def connect_remote(self, url: str, api_key: str):
#         """Connect to remote Qdrant Cloud or hosted instance."""
#         self.client = QdrantClient(url=url, api_key=api_key)
#         print(f"Connected to remote Qdrant @ {url}")

#     # ------------------------------------------------
#     # COLLECTION MANAGEMENT
#     # ------------------------------------------------
#     def create_collection(self, vector_size: int, distance: str = "Cosine"):
#         """Creates a vector collection if not exists."""
#         assert self.client is not None, "Call connect_local() or connect_remote() first"

#         self.client.recreate_collection(
#             collection_name=self.collection_name,
#             vectors_config=qmodels.VectorParams(size=vector_size, distance=distance)
#         )
#         print(f"Collection '{self.collection_name}' created with vector size {vector_size}")

#     def delete_collection(self):
#         """Deletes the entire collection."""
#         assert self.client is not None
#         self.client.delete_collection(self.collection_name)
#         print(f"Collection '{self.collection_name}' deleted")

#     # ------------------------------------------------
#     # UPSERT DOCUMENTS
#     # ------------------------------------------------
#     def upsert(self, points: List[Dict[str, Any]]):
#         """Upserts points into Qdrant.
#         points = [{"id": int/str, "embedding": [...], "payload": {...}}]
#         """
#         assert self.client is not None

#         qdrant_points = [
#             qmodels.PointStruct(
#                 id=p["id"], vector=p["embedding"], payload=p.get("payload", {})
#             )
#             for p in points
#         ]

#         self.client.upsert(collection_name=self.collection_name, points=qdrant_points)
#         print(f"Upserted {len(points)} points into '{self.collection_name}'")

#     # ------------------------------------------------
#     # SEARCH API
#     # ------------------------------------------------
#     def search(self, query_vector: List[float], top_k: int = 5, filters: Dict = None):
#         """Semantic vector search with optional metadata filters."""
#         assert self.client is not None

#         q_filter = self._build_filter(filters) if filters else None

#         results = self.client.search(
#             collection_name=self.collection_name,
#             query_vector=query_vector,
#             limit=top_k,
#             query_filter=q_filter,
#         )

#         return [
#             {
#                 "id": r.id,
#                 "score": r.score,
#                 "payload": r.payload
#             }
#             for r in results
#         ]

#     # ------------------------------------------------
#     # DELETE POINTS
#     # ------------------------------------------------
#     def delete_points(self, ids: List[int]):
#         assert self.client is not None
#         self.client.delete(collection_name=self.collection_name, points_selector=qmodels.PointIdsList(points=ids))
#         print(f"Deleted points: {ids}")

#     # ------------------------------------------------
#     # FILTER HELPERS
#     # ------------------------------------------------
#     def _build_filter(self, filters: Dict) -> qmodels.Filter:
#         """
#         Convert Python dict → Qdrant filter object.
#         Example:
#             filters = {"source": "chapter1.pdf"}
#         """
#         conditions = []
#         for key, value in filters.items():
#             conditions.append(
#                 qmodels.FieldCondition(
#                     key=key, match=qmodels.MatchValue(value=value)
#                 )
#             )

#         return qmodels.Filter(must=conditions)


import logging
from typing import List, Dict, Any, Optional

# qdrant-client has changed APIs between versions; try imports defensively
try:
    from qdrant_client import QdrantClient
except Exception:
    QdrantClient = None

try:
    # newer versions expose models at this path
    from qdrant_client.models import Distance, VectorParams, PointStruct
except Exception:
    try:
        # older layout
        from qdrant_client.http.models import PointStruct
        from qdrant_client.models import Distance, VectorParams
    except Exception:
        PointStruct = None
        Distance = None
        VectorParams = None

logger = logging.getLogger(__name__)


class QdrantAdapter:
    def __init__(self, host: str = "localhost", port: int = 6333):
        """HTTP-based Qdrant adapter (robust across qdrant-client versions).

        Uses plain REST calls to Qdrant so we don't depend on qdrant-client internals.
        """
        import requests

        self.requests = requests
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.text_collection = "text_docs"
        self.image_collection = "image_docs"
        self._ensure_collections()

    def _ensure_collections(self):
        """Create collections if they don't exist."""
        # Use REST API to create/recreate collections
        url_text = f"{self.base_url}/collections/{self.text_collection}"
        body_text = {"vectors": {"size": 768, "distance": "Cosine"}}
        try:
            resp = self.requests.put(url_text, json={"vectors": body_text["vectors"]})
            if resp.status_code in (200, 201):
                logger.info(f"Ensured collection: {self.text_collection}")
            else:
                logger.debug(f"Create collection response: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.error(f"Error creating text collection via REST: {e}")

        url_image = f"{self.base_url}/collections/{self.image_collection}"
        body_image = {"vectors": {"size": 512, "distance": "Cosine"}}
        try:
            resp = self.requests.put(url_image, json={"vectors": body_image["vectors"]})
            if resp.status_code in (200, 201):
                logger.info(f"Ensured collection: {self.image_collection}")
            else:
                logger.debug(f"Create image collection response: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.error(f"Error creating image collection via REST: {e}")

    def create_collection_if_not_exists(self, collection: str, vector_size: int = 768, distance: str = "Cosine"):
        """Create a collection via REST if it does not already exist."""
        url = f"{self.base_url}/collections/{collection}"
        body = {"vectors": {"size": vector_size, "distance": distance}}
        try:
            resp = self.requests.put(url, json={"vectors": body["vectors"]})
            if resp.status_code in (200, 201):
                logger.info(f"Ensured collection: {collection} (size={vector_size})")
            else:
                # If already exists, Qdrant may return 400/409 depending on config; just log debug
                logger.debug(f"Create collection response for {collection}: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.error(f"Error creating collection {collection} via REST: {e}")

    def upsert_vectors(self, collection: str, vectors, metadata: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None):
        """Upsert vectors with metadata.

        Supports two shapes for backward compatibility:
        - upsert_vectors(collection, embeddings_list, metadata_list, ids_list)
        - upsert_vectors(collection, points_list) where points_list = [(id, embedding, metadata), ...]
        """
        # Normalize inputs
        points = []
        # If caller passed a single list of tuples (id, embedding, meta)
        if metadata is None and ids is None and isinstance(vectors, list) and vectors and isinstance(vectors[0], (list, tuple)) and len(vectors[0]) >= 3:
            for id_, vec, meta in vectors:
                points.append({"id": id_, "vector": (vec.tolist() if hasattr(vec, 'tolist') else vec), "payload": meta})
        else:
            # Expect vectors:list, metadata:list, ids:list
            if not isinstance(vectors, list) or metadata is None or ids is None:
                raise ValueError("Invalid arguments for upsert_vectors. Expected (collection, vectors, metadata, ids) or (collection, points_list).")
            for id_, vec, meta in zip(ids, vectors, metadata):
                payload = meta or {}
                embedding = vec.tolist() if hasattr(vec, 'tolist') else vec
                points.append({"id": id_, "vector": embedding, "payload": payload})

        # Use Qdrant HTTP API to upsert points
        url = f"{self.base_url}/collections/{collection}/points?wait=true"
        try:
            resp = self.requests.put(url, json={"points": points})
            if resp.status_code not in (200, 201):
                raise RuntimeError(f"Unexpected Response: {resp.status_code} (Bad Request)\nRaw response content:\n{resp.content}")
        except Exception as e:
            logger.error(f"Error upserting points via REST: {e}")
            raise

    def search(self, collection: str, query_vector: List[float], top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """Search vectors with optional filters."""
        # Use REST search endpoint
        url = f"{self.base_url}/collections/{collection}/points/search"
        body = {"vector": query_vector, "limit": top_k, "with_payload": True}
        if filters:
            body["filter"] = filters
        try:
            resp = self.requests.post(url, json=body)
            if resp.status_code != 200:
                raise RuntimeError(f"Unexpected Response: {resp.status_code} (Bad Request)\nRaw response content:\n{resp.content}")
            data = resp.json()
        except Exception as e:
            logger.error(f"Error searching via REST: {e}")
            raise

        # Parse known response shapes
        hits = []
        if isinstance(data, dict):
            # Qdrant may return {'result': {'points': [...]}} or {'result': [{'id':..., 'score':..., 'payload':...}, ...]}
            if 'result' in data and isinstance(data['result'], dict):
                hits = data['result'].get('points') or data['result'].get('hits') or []
            elif 'result' in data and isinstance(data['result'], list):
                hits = data['result']
            elif 'hits' in data:
                hits = data['hits']
            elif 'points' in data:
                hits = data['points']
        elif isinstance(data, list):
            hits = data

        parsed_results: List[Dict[str, Any]] = []
        for hit in hits:
            if isinstance(hit, dict):
                id_ = hit.get('id')
                score = hit.get('score') or hit.get('payload', {}).get('score')
                metadata = hit.get('payload', {})
                parsed_results.append({"id": id_, "score": score, "metadata": metadata})
            elif isinstance(hit, (list, tuple)) and len(hit) >= 2:
                parsed_results.append({"id": hit[0], "score": hit[1], "metadata": hit[2] if len(hit) > 2 else {}})

        return parsed_results

    def delete_collection(self, collection: str):
        """Delete a collection."""
        try:
            url = f"{self.base_url}/collections/{collection}"
            resp = self.requests.delete(url)
            if resp.status_code not in (200, 202):
                logger.error(f"Failed to delete collection {collection}: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.error(f"Error deleting collection via REST: {e}")