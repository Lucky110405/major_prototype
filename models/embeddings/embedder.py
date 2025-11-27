from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import hashlib
import uuid
from typing import List, Dict, Any

class TextEmbedder:
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def embed(self, texts: list) -> list:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        return embeddings.tolist()

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return list of float vectors for texts"""
    embedder = TextEmbedder()
    return embedder.embed(texts)

def create_chunks_with_embeddings(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create embeddings for text chunks and add IDs, processing in batches to save memory"""
    items = []
    batch_size = 5  # Process in small batches to avoid memory issues
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [chunk["text"] for chunk in batch]
        embeddings = embed_texts(texts)
        for chunk, embedding in zip(batch, embeddings):
            unique_id = str(uuid.uuid4())
            items.append({
                "id": unique_id,
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {}),
                "embedding": np.array(embedding, dtype=np.float32)
            })
    return items