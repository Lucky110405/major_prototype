from fastapi import APIRouter, UploadFile, File, Form, Depends, Request
from typing import Optional
import os
import tempfile
from ingestion.pipeline.pdf_ingest_pipeline import ingest_pdf
from ingestion.image.image_ingestor import ImageIngestor
from ingestion.etl_structured_data.csv_loader import CSVLoader
from ingestion.etl_structured_data.excel_loader import ExcelLoader
from ingestion.multimodal_unstructured_data.audio_transcriber import AudioTranscriber
from ingestion.multimodal_unstructured_data.chart_ocr import ChartOCR
from ingestion.multimodal_unstructured_data.table_extract import TableExtractor
from models.embeddings.embedder import TextEmbedder
from models.embeddings.metadata_store import MetadataStore
from retrieval.qdrant_adapter import QdrantAdapter

router = APIRouter()

def get_text_embedder(request: Request):
    return request.app.state.text_embedder

def get_metadata_store(request: Request):
    return request.app.state.metadata_store

def get_qdrant_adapter(request: Request):
    return request.app.state.qdrant_adapter

def get_ingestion_agent(request: Request):
    return request.app.state.ingestion_agent

@router.post("/pdf")
def ingest_pdf_endpoint(file: UploadFile = File(...), index_path: Optional[str] = Form(None), 
                        meta_db_path: Optional[str] = Form(None), limit_chunks: Optional[int] = Form(None)):
    """
    Ingest a PDF file.
    """
    file_path = os.path.join(tempfile.gettempdir(), file.filename)
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
    file_path = os.path.join(tempfile.gettempdir(), file.filename)
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

@router.post("/csv")
def ingest_csv_endpoint(file: UploadFile = File(...), source: Optional[str] = Form("api"),
                        metadata_store=Depends(get_metadata_store),
                        qdrant_adapter=Depends(get_qdrant_adapter),
                        text_embedder=Depends(get_text_embedder)):
    """
    Ingest a CSV file.
    """
    file_path = os.path.join(tempfile.gettempdir(), file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    try:
        df = CSVLoader.load(file_path)
        # Convert to text representation
        text_content = df.to_string()
        # Embed and store
        embedding = text_embedder.embed([text_content])[0]
        metadata = {"source": source, "type": "csv", "filename": file.filename}
        qdrant_adapter.upsert_vectors("text_docs", [embedding], [metadata], [file.filename])
        metadata_store.store_metadata(file.filename, metadata)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        os.remove(file_path)

@router.post("/excel")
def ingest_excel_endpoint(file: UploadFile = File(...), sheet_name: Optional[str] = Form(0), source: Optional[str] = Form("api"),
                          metadata_store=Depends(get_metadata_store),
                          qdrant_adapter=Depends(get_qdrant_adapter),
                          text_embedder=Depends(get_text_embedder)):
    """
    Ingest an Excel file.
    """
    file_path = os.path.join(tempfile.gettempdir(), file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    try:
        df = ExcelLoader.load(file_path, sheet_name)
        # Convert to text representation
        text_content = df.to_string()
        # Embed and store
        embedding = text_embedder.embed([text_content])[0]
        metadata = {"source": source, "type": "excel", "filename": file.filename, "sheet": sheet_name}
        qdrant_adapter.upsert_vectors("text_docs", [embedding], [metadata], [file.filename])
        metadata_store.store_metadata(file.filename, metadata)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        os.remove(file_path)

@router.post("/audio")
def ingest_audio_endpoint(file: UploadFile = File(...), source: Optional[str] = Form("api"),
                          metadata_store=Depends(get_metadata_store),
                          qdrant_adapter=Depends(get_qdrant_adapter),
                          text_embedder=Depends(get_text_embedder)):
    """
    Ingest an audio file and transcribe to text.
    """
    file_path = os.path.join(tempfile.gettempdir(), file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    try:
        transcriber = AudioTranscriber()
        text_content = transcriber.transcribe(file_path)
        # Embed and store
        embedding = text_embedder.embed([text_content])[0]
        metadata = {"source": source, "type": "audio", "filename": file.filename}
        qdrant_adapter.upsert_vectors("text_docs", [embedding], [metadata], [file.filename])
        metadata_store.store_metadata(file.filename, metadata)
        return {"status": "success", "transcription": text_content}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        os.remove(file_path)

@router.post("/chart")
def ingest_chart_endpoint(file: UploadFile = File(...), source: Optional[str] = Form("api"),
                          metadata_store=Depends(get_metadata_store),
                          qdrant_adapter=Depends(get_qdrant_adapter),
                          text_embedder=Depends(get_text_embedder)):
    """
    Ingest a chart image and extract insights.
    """
    file_path = os.path.join(tempfile.gettempdir(), file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    try:
        # Assuming api_key is set in environment or app state
        api_key = os.getenv("GOOGLE_API_KEY")  # or from app.state
        ocr = ChartOCR(api_key)
        insights = ocr.extract_insights(file_path)
        # Embed and store
        embedding = text_embedder.embed([insights])[0]
        metadata = {"source": source, "type": "chart", "filename": file.filename}
        qdrant_adapter.upsert_vectors("text_docs", [embedding], [metadata], [file.filename])
        metadata_store.store_metadata(file.filename, metadata)
        return {"status": "success", "insights": insights}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        os.remove(file_path)

@router.post("/table")
def ingest_table_endpoint(file: UploadFile = File(...), source: Optional[str] = Form("api"),
                          metadata_store=Depends(get_metadata_store),
                          qdrant_adapter=Depends(get_qdrant_adapter),
                          text_embedder=Depends(get_text_embedder)):
    """
    Ingest a file with tables (PDF or image) and extract tables.
    """
    file_path = os.path.join(tempfile.gettempdir(), file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        extractor = TableExtractor(api_key)
        if file.filename and file.filename.endswith('.pdf'):
            tables = extractor.extract_from_pdf(file_path)
        else:
            from PIL import Image
            img = Image.open(file_path)
            tables = extractor.extract_from_image(img)
        # Convert tables to text
        text_content = str(tables)
        # Embed and store
        embedding = text_embedder.embed([text_content])[0]
        metadata = {"source": source, "type": "table", "filename": file.filename}
        qdrant_adapter.upsert_vectors("text_docs", [embedding], [metadata], [file.filename])
        metadata_store.store_metadata(file.filename, metadata)
        return {"status": "success", "tables": tables}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        os.remove(file_path)

@router.post("/auto")
def ingest_auto_endpoint(file: UploadFile = File(...), source: Optional[str] = Form("auto"),
                         ingestion_agent=Depends(get_ingestion_agent)):
    """
    Automatically ingest a file based on detected modality.
    """
    file_path = os.path.join(tempfile.gettempdir(), file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    try:
        result = ingestion_agent.ingest_file(file_path, source)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}