from fastapi import APIRouter, UploadFile, File, Form, Depends
from typing import Optional
import os
from ingestion.pipeline.pdf_ingest_pipeline import ingest_pdf
from ingestion.image.image_ingestor import ImageIngestor

router = APIRouter()

def get_metadata_store(request):
    return request.app.state.metadata_store

def get_qdrant_adapter(request):
    return request.app.state.qdrant_adapter

def get_text_embedder(request):
    return request.app.state.text_embedder

@router.post("/pdf")
def ingest_pdf_endpoint(file: UploadFile = File(...), index_path: Optional[str] = Form(None), 
                        meta_db_path: Optional[str] = Form(None), limit_chunks: Optional[int] = Form(None)):
    """
    Ingest a PDF file.
    """
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    try:
        result = ingest_pdf(file_path, index_path, meta_db_path, limit_chunks)
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        os.remove(file_path)

@router.post("/image")
def ingest_image_endpoint(file: UploadFile = File(...), source: Optional[str] = Form("api"),
                          metadata_store=Depends(get_metadata_store),
                          qdrant_adapter=Depends(get_qdrant_adapter),
                          text_embedder=Depends(get_text_embedder)):
    """
    Ingest an image file.
    """
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    try:
        ingestor = ImageIngestor(metadata_store, qdrant_adapter, text_embedder)
        success = ingestor.process_image(file_path, source)
        return {"status": "success" if success else "failed"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        os.remove(file_path)