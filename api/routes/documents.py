from fastapi import APIRouter, Depends, Request, Query
from typing import List, Optional
import os
import hashlib
from datetime import datetime

router = APIRouter()

def get_metadata_store(request: Request):
    return request.app.state.metadata_store

def get_qdrant_adapter(request: Request):
    return request.app.state.qdrant_adapter

@router.get("/documents")
def list_documents(source: Optional[str] = Query("disk", description="Source to list documents from: 'disk' or 'supabase'"),
                   metadata_store=Depends(get_metadata_store)):
    """Return all documents stored in the metadata store or from local data dir.
    By default returns documents from the local repository `data` folder (source='disk').
    Use source='supabase' to query the metadata store instead.
    """
    try:
        if source == "supabase":
            docs = metadata_store.get_all_documents()
            return {"documents": docs}

        # Default: read from disk (repo data/ folder)
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
        if not os.path.exists(base_path):
            return {"documents": []}

        entries = []
        allowed_exts = ['.pdf', '.csv', '.txt', '.xlsx', '.xls', '.png', '.jpg', '.jpeg', '.bmp']
        for root, dirs, files in os.walk(base_path):
            for fname in files:
                fpath = os.path.join(root, fname)
                ext = os.path.splitext(fname)[1].lower()
                if ext not in allowed_exts:
                    continue
                stat = os.stat(fpath)
                # create deterministic id based on path and mtime
                hash_src = f"{os.path.relpath(fpath, base_path)}|{stat.st_mtime}".encode('utf-8')
                id_ = hashlib.md5(hash_src).hexdigest()
                uploaded_at = datetime.utcfromtimestamp(stat.st_mtime).isoformat() + 'Z'
                doc = {
                    'id': id_,
                    'file_name': fname,
                    'file_type': ext.replace('.', ''),
                    'file_size': stat.st_size,
                    'content': None,
                    'uploaded_at': uploaded_at,
                    'user_id': None,
                }
                entries.append(doc)

        # sort by uploaded_at desc
        entries.sort(key=lambda d: d.get('uploaded_at', ''), reverse=True)
        return {"documents": entries}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str, metadata_store=Depends(get_metadata_store), qdrant_adapter=Depends(get_qdrant_adapter)):
    """Delete a document and associated vectors.

    Deletes the metadata entry and corresponding vector in Qdrant.
    """
    try:
        # Fetch metadata to determine collection
        metadata = metadata_store.get_metadata(doc_id)
        collection = "text_docs"
        if metadata and metadata.get("type") == "image":
            collection = "image_docs"

        # Delete from qdrant if adapter provided
        if qdrant_adapter:
            try:
                qdrant_adapter.delete_points(collection, [doc_id])
            except Exception as e:
                # Log but continue to delete metadata
                print(f"Warning: failed to delete qdrant point: {e}")

        # Delete metadata
        metadata_store.delete_document(doc_id)
        return {"status": "success", "id": doc_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}
