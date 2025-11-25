from qdrant_client.models import VectorParams, Distance

COLLECTION_NAME = "business_intel_docs"

def get_vector_params(dim: int):
    return VectorParams(
        size=dim,
        distance=Distance.COSINE
    )
