import logging
from typing import Dict, Any, Optional
import os
import uuid
from ingestion.pipeline.pdf_ingest_pipeline import ingest_pdf
from ingestion.image.image_ingestor import ImageIngestor
from ingestion.etl_structured_data.csv_loader import CSVLoader
from ingestion.etl_structured_data.excel_loader import ExcelLoader
from ingestion.multimodal_unstructured_data.audio_transcriber import AudioTranscriber
from ingestion.multimodal_unstructured_data.chart_ocr import ChartOCR
from ingestion.multimodal_unstructured_data.table_extract import TableExtractor
from models.embeddings.embedder import TextEmbedder, create_chunks_with_embeddings
from models.embeddings.metadata_store import MetadataStore
from retrieval.qdrant_adapter import QdrantAdapter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IngestionAgent:
    def __init__(self, metadata_store: MetadataStore, qdrant_adapter: QdrantAdapter, text_embedder: TextEmbedder):
        """
        Initialize the Ingestion Agent with dependencies.
        
        Args:
            metadata_store: MetadataStore instance.
            qdrant_adapter: QdrantAdapter instance.
            text_embedder: TextEmbedder instance.
        """
        self.metadata_store = metadata_store
        self.qdrant_adapter = qdrant_adapter
        self.text_embedder = text_embedder

    def detect_modality(self, file_path: str) -> str:
        """
        Detect the modality of the file based on extension and content.
        
        Args:
            file_path: Path to the file.
        
        Returns:
            Modality string: 'pdf', 'image', 'csv', 'excel', 'audio', 'chart', 'table', 'unknown'.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return 'pdf'
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            # For chart/table, might need content check, but for simplicity, assume image
            # Could enhance with ML detection
            return 'image'  # or 'chart'/'table' if detected
        elif ext == '.csv':
            return 'csv'
        elif ext in ['.xlsx', '.xls']:
            return 'excel'
        elif ext in ['.wav', '.mp3', '.flac']:
            return 'audio'
        else:
            # Fallback: try to detect content
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(10)
                    if header.startswith(b'%PDF'):
                        return 'pdf'
                    # Add more checks if needed
            except:
                pass
            return 'unknown'

    def ingest_file(self, file_path: str, source: str = "auto", stored_path: str = None) -> Dict[str, Any]:
        """
        Automatically ingest a file based on its detected modality.
        
        Args:
            file_path: Path to the file.
            source: Source identifier.
        
        Returns:
            Dict with status and result.
        """
        modality = self.detect_modality(file_path)
        logger.info(f"Detected modality: {modality} for {file_path}")
        
        try:
            if modality == 'pdf':
                result = ingest_pdf(file_path, stored_path=stored_path)
                return {"status": "success", "modality": modality, "result": result}
            elif modality == 'image':
                ingestor = ImageIngestor(self.metadata_store, self.qdrant_adapter, self.text_embedder)
                success = ingestor.process_image(file_path, source)
                return {"status": "success" if success else "failed", "modality": modality}
            elif modality == 'csv':
                df = CSVLoader.load(file_path)
                chunks = []
                max_chunk_chars = 4000
                # If an 'asset' column exists, group by asset and create a chunk per asset
                if 'asset' in df.columns:
                    grouped = df.groupby('asset')
                    for asset, group in grouped:
                        text = group.to_string(index=False)
                        text = text[:max_chunk_chars]
                        meta = {"source": source, "type": "csv", "filename": os.path.basename(file_path), "asset": str(asset)}
                        if stored_path:
                            meta["stored_path"] = stored_path
                        chunks.append({"text": text, "metadata": meta})
                else:
                    # Chunk by fixed number of rows to keep chunks reasonably sized
                    chunk_rows = 50
                    num_rows = len(df)
                    for start in range(0, num_rows, chunk_rows):
                        sub = df.iloc[start:start+chunk_rows]
                        text = sub.to_string(index=False)
                        text = text[:max_chunk_chars]
                        meta = {"source": source, "type": "csv", "filename": os.path.basename(file_path), "row_range": f"{start}-{start+len(sub)-1}"}
                        if stored_path:
                            meta["stored_path"] = stored_path
                        chunks.append({"text": text, "metadata": meta})

                # Create embeddings for chunks (uses batching inside helper)
                items = create_chunks_with_embeddings(chunks)

                # Prepare upsert lists
                vectors = [item['embedding'].tolist() if hasattr(item['embedding'], 'tolist') else item['embedding'] for item in items]
                metas = []
                ids = []
                for item in items:
                    meta = item.get('metadata', {})
                    # include a short text excerpt to help downstream analyzer and retrieval displays
                    meta['text_excerpt'] = item.get('text', '')[:1000]
                    meta['filename'] = os.path.basename(file_path)
                    if stored_path:
                        meta['stored_path'] = stored_path
                    metas.append(meta)
                    ids.append(item['id'])

                if vectors:
                    self.qdrant_adapter.upsert_vectors("text_docs", vectors, metas, ids)
                    for i, id_ in enumerate(ids):
                        self.metadata_store.store_metadata(id_, metas[i])

                return {"status": "success", "modality": modality, "chunks_stored": len(ids)}
            elif modality == 'excel':
                df = ExcelLoader.load(file_path)
                text_content = df.to_string()
                embedding = self.text_embedder.embed([text_content])[0]
                id_ = str(uuid.uuid4())
                metadata = {"source": source, "type": "excel", "filename": os.path.basename(file_path), "id": id_}
                if stored_path:
                    metadata['stored_path'] = stored_path
                self.qdrant_adapter.upsert_vectors("text_docs", [embedding], [metadata], [id_])
                self.metadata_store.store_metadata(id_, metadata)
                return {"status": "success", "modality": modality}
            elif modality == 'audio':
                transcriber = AudioTranscriber()
                text_content = transcriber.transcribe(file_path)
                embedding = self.text_embedder.embed([text_content])[0]
                id_ = str(uuid.uuid4())
                metadata = {"source": source, "type": "audio", "filename": os.path.basename(file_path), "id": id_}
                if stored_path:
                    metadata['stored_path'] = stored_path
                self.qdrant_adapter.upsert_vectors("text_docs", [embedding], [metadata], [id_])
                self.metadata_store.store_metadata(id_, metadata)
                return {"status": "success", "modality": modality, "transcription": text_content}
            elif modality == 'chart':
                api_key = os.getenv("GOOGLE_API_KEY")
                ocr = ChartOCR(api_key)
                insights = ocr.extract_insights(file_path)
                embedding = self.text_embedder.embed([insights])[0]
                id_ = str(uuid.uuid4())
                metadata = {"source": source, "type": "chart", "filename": os.path.basename(file_path), "id": id_}
                if stored_path:
                    metadata['stored_path'] = stored_path
                self.qdrant_adapter.upsert_vectors("text_docs", [embedding], [metadata], [id_])
                self.metadata_store.store_metadata(id_, metadata)
                return {"status": "success", "modality": modality, "insights": insights}
            elif modality == 'table':
                api_key = os.getenv("GOOGLE_API_KEY")
                extractor = TableExtractor(api_key)
                if file_path.endswith('.pdf'):
                    tables = extractor.extract_from_pdf(file_path)
                else:
                    from PIL import Image
                    img = Image.open(file_path)
                    tables = extractor.extract_from_image(img)
                text_content = str(tables)
                embedding = self.text_embedder.embed([text_content])[0]
                id_ = str(uuid.uuid4())
                metadata = {"source": source, "type": "table", "filename": os.path.basename(file_path), "id": id_}
                if stored_path:
                    metadata['stored_path'] = stored_path
                self.qdrant_adapter.upsert_vectors("text_docs", [embedding], [metadata], [id_])
                self.metadata_store.store_metadata(id_, metadata)
                return {"status": "success", "modality": modality, "tables": tables}
            else:
                return {"status": "error", "message": f"Unsupported modality: {modality}"}
        except Exception as e:
            logger.error(f"Error ingesting {file_path}: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)