from fastapi import APIRouter, Query, Depends, Request
from typing import Optional, List
from models.embeddings.metadata_store import MetadataStore
import os

router = APIRouter()

def get_multimodal_retriever(request: Request):
    return request.app.state.multimodal_retriever

def get_text_embedder(request: Request):
    return request.app.state.text_embedder

def get_hybrid_retriever(request: Request):
    return request.app.state.hybrid_retriever

def get_metadata_store(request: Request):
    return request.app.state.metadata_store

@router.get("")
@router.get("/")
def query_endpoint(q: str = Query(...), top_k: Optional[int] = 10, 
                   text_weight: Optional[float] = 0.5, image_weight: Optional[float] = 0.5,
                   multimodal_retriever=Depends(get_multimodal_retriever),
                   text_embedder=Depends(get_text_embedder),
                   metadata_store=Depends(get_metadata_store)):
    """
    Query the system for retrieval.
    """
    # Generate query vector
    query_vector = text_embedder.embed([q])[0] if text_embedder else []
    try:
        results = multimodal_retriever.retrieve(query_vector, top_k=top_k, 
                                                text_weight=text_weight, image_weight=image_weight,
                                                query_text=q)
        # Build sources list by fetching metadata for each result if available
        sources = []
        for r in results:
            meta = None
            try:
                if metadata_store:
                    meta = metadata_store.get_metadata(r.get('id'))
            except Exception:
                meta = None
            if meta:
                sources.append({
                    'id': r.get('id'),
                    'file_name': meta.get('file_name') or meta.get('filename') or os.path.basename(meta.get('stored_path', '')),
                    'file_type': meta.get('file_type') or meta.get('type') or '',
                    'file_size': meta.get('file_size') or 0,
                    'uploaded_at': meta.get('uploaded_at') or meta.get('created_at') or '',
                    'user_id': meta.get('user_id') or None,
                    'metadata': meta
                })
            else:
                sources.append({
                    'id': r.get('id'),
                    'file_name': r.get('metadata', {}).get('filename') or r.get('id'),
                    'file_type': r.get('metadata', {}).get('type') or '',
                    'file_size': r.get('metadata', {}).get('file_size') or 0,
                    'uploaded_at': '',
                    'user_id': None,
                    'metadata': r.get('metadata', {})
                })

        # Create a simple answer summary from the top results
        if results:
            top = results[0]
            top_meta = top.get('metadata', {})
            score = top.get('score')
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            answer = f"Found {len(results)} relevant result(s). Top item: {top_meta.get('filename') or top_meta.get('file_name') or top.get('id')} - score {score_str}" if top else "No results"
        else:
            answer = 'No relevant documents found.'

        return {"query": {"query_text": q, "answer": answer}, "sources": sources}
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