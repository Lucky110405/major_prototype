from fastapi import APIRouter, Query, Depends, Request
from typing import Optional, List

router = APIRouter()

def get_multimodal_retriever(request: Request):
    return request.app.state.multimodal_retriever

def get_text_embedder(request: Request):
    return request.app.state.text_embedder

def get_hybrid_retriever(request: Request):
    return request.app.state.hybrid_retriever

@router.get("/")
def query_endpoint(q: str = Query(...), top_k: Optional[int] = 10, 
                   text_weight: Optional[float] = 0.5, image_weight: Optional[float] = 0.5,
                   multimodal_retriever=Depends(get_multimodal_retriever),
                   text_embedder=Depends(get_text_embedder)):
    """
    Query the system for retrieval.
    """
    # Generate query vector
    query_vector = text_embedder.embed([q])[0] if text_embedder else []
    try:
        results = multimodal_retriever.retrieve(query_vector, top_k=top_k, 
                                                text_weight=text_weight, image_weight=image_weight,
                                                query_text=q)
        return {"results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/hybrid")
def hybrid_query_endpoint(q: str = Query(...), top_k: Optional[int] = 10,
                          hybrid_retriever=Depends(get_hybrid_retriever)):
    """
    Query using hybrid retrieval (dense + BM25).
    """
    try:
        results = hybrid_retriever.retrieve(q, top_k=top_k)
        return {"results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/bm25")
def bm25_query_endpoint(q: str = Query(...), top_k: Optional[int] = 10):
    """
    Query using BM25 retrieval.
    """
    # Note: BM25 index not set up in current setup
    return {"status": "error", "message": "BM25 index not initialized"}