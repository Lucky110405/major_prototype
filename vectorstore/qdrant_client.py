from qdrant_client import QdrantClient
from qdrant_client.http import models
from .schema import COLLECTION_NAME, get_vector_params

class QdrantDB:
    def __init__(self, host="localhost", port=6333):
        self.client = QdrantClient(host=host, port=port)

    def create_collection(self, dim: int):
        if not self.client.collection_exists(COLLECTION_NAME):
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=get_vector_params(dim)
            )

    def add_embeddings(self, vectors, metadata):
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=models.Batch(
                vectors=vectors,
                payloads=metadata,
                ids=list(range(100000, 100000 + len(vectors)))
            )
        )

    def search(self, query_vector, limit=5):
        return self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit
        )
