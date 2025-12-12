from fastapi import APIRouter, UploadFile, File, Form, Depends, Request
from typing import Optional
import os
import shutil
import uuid
from datetime import datetime
import tempfile
import uuid
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
    
    # Save copy to data/raw for archival
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    raw_dir = os.path.join(data_root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    unique_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}_{os.path.basename(file.filename)}"
    stored_path = os.path.join(raw_dir, unique_name)
    shutil.copy2(file_path, stored_path)

    try:
        result = ingest_pdf(file_path, index_path, limit_chunks, stored_path=stored_path)
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
    
    # Save copy to data/raw
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    raw_dir = os.path.join(data_root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    unique_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}_{os.path.basename(file.filename)}"
    stored_path = os.path.join(raw_dir, unique_name)
    shutil.copy2(file_path, stored_path)

    try:
        ingestor = ImageIngestor(metadata_store, qdrant_adapter, text_embedder)
        ids = ingestor.process_image(file_path, source, stored_path=stored_path)
        return {"status": "success" if ids else "failed", "ids": ids}
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
    
    # Save copy to data/raw
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    raw_dir = os.path.join(data_root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    unique_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}_{os.path.basename(file.filename)}"
    stored_path = os.path.join(raw_dir, unique_name)
    shutil.copy2(file_path, stored_path)

    try:
        df = CSVLoader.load(file_path)
        # Convert to text representation
        text_content = df.to_string()
        # Embed and store
        embedding = text_embedder.embed([text_content])[0]
        id_ = str(uuid.uuid4())
        metadata = {"source": source, "type": "csv", "filename": file.filename, "id": id_, "stored_path": stored_path}
        qdrant_adapter.upsert_vectors("text_docs", [embedding], [metadata], [id_])
        metadata_store.store_metadata(id_, metadata)
        return {"status": "success", "id": id_}
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
    
    # Save copy to data/raw
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    raw_dir = os.path.join(data_root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    unique_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}_{os.path.basename(file.filename)}"
    stored_path = os.path.join(raw_dir, unique_name)
    shutil.copy2(file_path, stored_path)

    try:
        df = ExcelLoader.load(file_path, sheet_name)
        # Convert to text representation
        text_content = df.to_string()
        # Embed and store
        embedding = text_embedder.embed([text_content])[0]
        id_ = str(uuid.uuid4())
        metadata = {"source": source, "type": "excel", "filename": file.filename, "sheet": sheet_name, "id": id_, "stored_path": stored_path}
        qdrant_adapter.upsert_vectors("text_docs", [embedding], [metadata], [id_])
        metadata_store.store_metadata(id_, metadata)
        return {"status": "success", "id": id_}
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
    
    # Save copy to data/raw
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    raw_dir = os.path.join(data_root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    unique_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}_{os.path.basename(file.filename)}"
    stored_path = os.path.join(raw_dir, unique_name)
    shutil.copy2(file_path, stored_path)

    try:
        transcriber = AudioTranscriber()
        text_content = transcriber.transcribe(file_path)
        # Embed and store
        embedding = text_embedder.embed([text_content])[0]
        id_ = str(uuid.uuid4())
        metadata = {"source": source, "type": "audio", "filename": file.filename, "id": id_, "transcription": text_content, "stored_path": stored_path}
        qdrant_adapter.upsert_vectors("text_docs", [embedding], [metadata], [id_])
        metadata_store.store_metadata(id_, metadata)
        return {"status": "success", "id": id_, "transcription": text_content}
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
    
    # Save copy to data/raw
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    raw_dir = os.path.join(data_root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    unique_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}_{os.path.basename(file.filename)}"
    stored_path = os.path.join(raw_dir, unique_name)
    shutil.copy2(file_path, stored_path)

    try:
        # Assuming api_key is set in environment or app state
        api_key = os.getenv("GOOGLE_API_KEY")  # or from app.state
        ocr = ChartOCR(api_key)
        insights = ocr.extract_insights(file_path)
        # Embed and store
        embedding = text_embedder.embed([insights])[0]
        id_ = str(uuid.uuid4())
        metadata = {"source": source, "type": "chart", "filename": file.filename, "id": id_, "insights": insights, "stored_path": stored_path}
        qdrant_adapter.upsert_vectors("text_docs", [embedding], [metadata], [id_])
        metadata_store.store_metadata(id_, metadata)
        return {"status": "success", "id": id_, "insights": insights}
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
    
    # Save copy to data/raw
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    raw_dir = os.path.join(data_root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    unique_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}_{os.path.basename(file.filename)}"
    stored_path = os.path.join(raw_dir, unique_name)
    shutil.copy2(file_path, stored_path)

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
        id_ = str(uuid.uuid4())
        metadata = {"source": source, "type": "table", "filename": file.filename, "id": id_, "tables_summary": tables, "stored_path": stored_path}
        qdrant_adapter.upsert_vectors("text_docs", [embedding], [metadata], [id_])
        metadata_store.store_metadata(id_, metadata)
        return {"status": "success", "id": id_, "tables": tables}
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
    
    # Save a copy, pass its path to the ingestion agent; ingestion agent will process and delete temp file
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    raw_dir = os.path.join(data_root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    unique_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}_{os.path.basename(file.filename)}"
    stored_path = os.path.join(raw_dir, unique_name)
    shutil.copy2(file_path, stored_path)

    try:
        result = ingestion_agent.ingest_file(file_path, source, stored_path=stored_path)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/meta/{doc_id}")
def get_metadata_endpoint(doc_id: str, metadata_store=Depends(get_metadata_store)):
    """Fetch stored metadata by document ID."""
    try:
        data = metadata_store.get_metadata(doc_id)
        if not data:
            return {"status": "not_found", "id": doc_id}
        return {"status": "success", "id": doc_id, "metadata": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}