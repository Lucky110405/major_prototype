import os
import logging
from typing import List, Dict, Any
from PIL import Image
import pytesseract
from transformers import CLIPProcessor, CLIPModel
import torch

# Assuming these are in the project
from models.embeddings.embedder import TextEmbedder
from models.embeddings.metadata_store import MetadataStore
from retrieval.qdrant_adapter import QdrantAdapter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardIngestor:
    def __init__(self, metadata_store: MetadataStore, qdrant_adapter: QdrantAdapter, text_embedder: TextEmbedder):
        """
        Initialize the Dashboard Ingestor (treats dashboards as images).
        
        Args:
            metadata_store: Instance of MetadataStore.
            qdrant_adapter: Instance of QdrantAdapter.
            text_embedder: Instance of TextEmbedder.
        """
        self.metadata_store = metadata_store
        self.qdrant_adapter = qdrant_adapter
        self.text_embedder = text_embedder
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()
        self.qdrant_adapter.create_collection_if_not_exists("image_docs", vector_size=512)

    def extract_text_ocr(self, image_path: str) -> str:
        """Extract text from dashboard image using OCR."""
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from {image_path}: {e}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Chunk long OCR text."""
        words = text.split()
        chunks = []
        current_chunk = ""
        for word in words:
            if len(current_chunk) + len(word) + 1 > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                current_chunk += " " + word
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def embed_image_clip(self, image_path: str) -> List[float]:
        """Generate CLIP embedding for dashboard image."""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.clip_model.get_image_features(**inputs)
            return outputs.squeeze().tolist()
        except Exception as e:
            logger.error(f"Error embedding image {image_path}: {e}")
            return []

    def process_dashboard(self, image_path: str, source: str = "dashboard", stored_path: str = None) -> bool:
        """
        Process a dashboard image: OCR, chunk, embed, store.
        
        Args:
            image_path: Path to the dashboard image.
            source: Source identifier.
        
        Returns:
            True if successful.
        """
        if not os.path.exists(image_path):
            logger.error(f"Dashboard image does not exist: {image_path}")
            return False
        
        ocr_text = self.extract_text_ocr(image_path)
        chunks = self.chunk_text(ocr_text) if ocr_text else []
        image_embedding = self.embed_image_clip(image_path)
        
        vectors = []
        if image_embedding:
            doc_id = f"{source}_dashboard_{os.path.basename(image_path)}"
            metadata = {
                "source": source,
                "file_path": image_path,
                "stored_path": stored_path,
                "type": "dashboard",
                "ocr_text": ocr_text,
                "chunk_count": len(chunks)
            }
            self.metadata_store.store_metadata(doc_id, metadata)
            vectors.append((doc_id, image_embedding, metadata))
        
        if self.text_embedder and chunks:
            for i, chunk in enumerate(chunks):
                chunk_id = f"{source}_dashboard_{os.path.basename(image_path)}_chunk_{i}"
                chunk_metadata = {
                    "source": source,
                    "file_path": image_path,
                    "stored_path": stored_path,
                    "type": "dashboard_chunk",
                    "chunk_index": i,
                    "text": chunk
                }
                self.metadata_store.store_metadata(chunk_id, chunk_metadata)
                chunk_embedding = self.text_embedder.embed([chunk])[0]
                vectors.append((chunk_id, chunk_embedding, metadata))
        
        if vectors:
            self.qdrant_adapter.upsert_vectors("image_docs", vectors)
            logger.info(f"Processed dashboard {image_path}")
            return True
        return False