from fastapi import APIRouter, Query, Depends
from typing import Optional, List

router = APIRouter()

def get_multimodal_retriever(request):
    return request.app.state.multimodal_retriever

def get_text_embedder(request):
    return request.app.state.text_embedder

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
                                                text_weight=text_weight, image_weight=image_weight)
        return {"results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}