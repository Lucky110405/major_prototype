from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import hashlib
from typing import List, Dict, Any

class TextEmbedder:
    def __init__(self, model_name: str = "prajjwal1/bert-tiny"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def embed(self, texts: list) -> list:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        return embeddings.tolist()

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return list of float vectors for texts"""
    embedder = TextEmbedder()
    return embedder.embed(texts)

def create_chunks_with_embeddings(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create embeddings for text chunks and add IDs"""
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    items = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # Create unique ID for the chunk
        chunk_id = hashlib.sha256()
        chunk_id.update(chunk["text"].encode("utf-8"))
        chunk_id.update(str(chunk.get("metadata", {})).encode("utf-8"))
        unique_id = chunk_id.hexdigest()

        items.append({
            "id": unique_id,
            "text": chunk["text"],
            "metadata": chunk.get("metadata", {}),
            "embedding": np.array(embedding, dtype=np.float32)
        })

    return items