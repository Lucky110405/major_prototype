import os
import logging
from typing import List, Dict, Any, Optional
from PIL import Image
import pytesseract
from transformers import CLIPProcessor, CLIPModel
import torch
import uuid

# Assuming these are in the project
from models.embeddings.embedder import TextEmbedder  # For text chunks if needed
from models.embeddings.metadata_store import MetadataStore
from retrieval.qdrant_adapter import QdrantAdapter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageIngestor:
    def __init__(self, 
                 metadata_store: MetadataStore, 
                 qdrant_adapter: QdrantAdapter,
                 text_embedder: Optional[TextEmbedder] = None,
                 use_clip: bool = True,
                 clip_model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the Image Ingestor.
        
        Args:
            metadata_store: Instance of MetadataStore for storing metadata.
            qdrant_adapter: Instance of QdrantAdapter for vector storage.
            text_embedder: Optional TextEmbedder for embedding OCR text chunks.
            use_clip: Whether to use CLIP for image embeddings.
            clip_model_name: CLIP model name if using CLIP.
        """
        self.metadata_store = metadata_store
        self.qdrant_adapter = qdrant_adapter
        self.text_embedder = text_embedder
        self.use_clip = use_clip
        
        if self.use_clip:
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            self.clip_model = CLIPModel.from_pretrained(clip_model_name)
            self.clip_model.eval()
        
        # Ensure Qdrant has the image_docs collection
        self.qdrant_adapter.create_collection_if_not_exists("image_docs", vector_size=512 if use_clip else 768)  # Adjust size based on model

    def extract_text_ocr(self, image_path: str) -> str:
        """
        Extract text from image using OCR.
        
        Args:
            image_path: Path to the image file.
        
        Returns:
            Extracted text.
        """
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from {image_path}: {e}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Chunk long OCR text into smaller pieces.
        
        Args:
            text: Full text to chunk.
            chunk_size: Maximum characters per chunk.
        
        Returns:
            List of text chunks.
        """
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
        """
        Generate CLIP embedding for the image.
        
        Args:
            image_path: Path to the image file.
        
        Returns:
            Embedding vector.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.clip_model.get_image_features(**inputs)
            return outputs.squeeze().tolist()
        except Exception as e:
            logger.error(f"Error embedding image {image_path}: {e}")
            return []

    def process_image(self, image_path: str, source: str = "unknown", stored_path: str = None) -> bool:
        """
        Process a single image: OCR, chunk, embed, store metadata, upsert to Qdrant.
        
        Args:
            image_path: Path to the image file.
            source: Source identifier (e.g., document name).
        
        Returns:
            True if successful, False otherwise.
        """
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            return False
        
        # Extract text via OCR
        ocr_text = self.extract_text_ocr(image_path)
        if not ocr_text:
            logger.warning(f"No text extracted from {image_path}")
        
        # Chunk the text
        chunks = self.chunk_text(ocr_text) if ocr_text else []
        
        # Generate image embedding
        image_embedding = self.embed_image_clip(image_path) if self.use_clip else []
        
        # Prepare metadata and vectors
        vectors = []
        metadata_list = []
        
        # For image embedding
        if image_embedding:
            id_ = str(uuid.uuid4())
            metadata = {
                "source": source,
                "file_path": image_path,
                "stored_path": stored_path,
                "type": "image",
                "ocr_text": ocr_text,
                "chunk_count": len(chunks),
                "id": id_
            }
            self.metadata_store.store_metadata(id_, metadata)
            vectors.append((id_, image_embedding, metadata))
        
        # For text chunks (if embedder available)
        if self.text_embedder and chunks:
            for i, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                chunk_metadata = {
                    "source": source,
                    "file_path": image_path,
                    "stored_path": stored_path,
                    "type": "image_chunk",
                    "chunk_index": i,
                    "text": chunk,
                    "id": chunk_id
                }
                self.metadata_store.store_metadata(chunk_id, chunk_metadata)
                chunk_embedding = self.text_embedder.embed([chunk])[0]  # Assuming embed returns list
                vectors.append((chunk_id, chunk_embedding, chunk_metadata))
        
        # Upsert to Qdrant and return stored IDs
        if vectors:
            ids = [v[0] for v in vectors]
            embeddings = [v[1] for v in vectors]
            metas = [v[2] for v in vectors]
            # Use adapter's flexible upsert (supports points_list or separate lists)
            self.qdrant_adapter.upsert_vectors("image_docs", embeddings, metas, ids)
            logger.info(f"Processed and stored {len(vectors)} vectors for {image_path}")
            return ids
        else:
            logger.warning(f"No vectors generated for {image_path}")
            return []

    def process_batch(self, image_paths: List[str], source: str = "batch") -> int:
        """
        Process a batch of images.
        
        Args:
            image_paths: List of image file paths.
            source: Source identifier.
        
        Returns:
            Number of successfully processed images.
        """
        success_count = 0
        for path in image_paths:
            if self.process_image(path, source):
                success_count += 1
        return success_count