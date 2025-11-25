# ingestion/retrieval/embeddings/multimodal_embedder.py

from typing import List
import numpy as np
from config import IMAGE_EMBEDDER_PROVIDER, CLIP_MODEL, OPENAI_API_KEY, EMBED_BATCH_SIZE

# This module provides:
# - embed_images(image_paths: List[str]) -> List[List[float]]
# - embed_audio_transcripts(texts: List[str]) -> List[List[float]]  (audio embeddings via text embedder)

# Image embedding options:
# - local CLIP (transformers + torchvision)
# - OpenAI image embeddings (if available) by converting image->base64 and calling embed API (provider-specific)

def embed_images(image_paths: List[str]) -> List[List[float]]:
    if IMAGE_EMBEDDER_PROVIDER == "clip":
        # use huggingface CLIP (transformers + torch)
        from PIL import Image
        import torch
        from transformers import CLIPProcessor, CLIPModel

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        vectors = []
        batch = []
        paths = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            batch.append(img)
            paths.append(p)
            if len(batch) >= EMBED_BATCH_SIZE:
                inputs = processor(images=batch, return_tensors="pt")
                with torch.no_grad():
                    embs = model.get_image_features(**inputs)
                embs = embs / embs.norm(p=2, dim=-1, keepdim=True)
                for e in embs:
                    vectors.append(e.cpu().numpy().astype('float32').tolist())
                batch = []
        if batch:
            inputs = processor(images=batch, return_tensors="pt")
            with torch.no_grad():
                embs = model.get_image_features(**inputs)
            embs = embs / embs.norm(p=2, dim=-1, keepdim=True)
            for e in embs:
                vectors.append(e.cpu().numpy().astype('float32').tolist())
        return vectors
    elif IMAGE_EMBEDDER_PROVIDER == "openai_clip":
        # Placeholder - depends on OpenAI offering; fallback to text embedder on captions
        raise NotImplementedError("openai_clip provider not implemented. Use clip or add provider code.")
    else:
        raise ValueError("Unknown IMAGE_EMBEDDER_PROVIDER")

def embed_audio_transcripts(transcripts: List[str], text_embed_fn) -> List[List[float]]:
    """
    For audio, we assume a transcriber already gave text. Use text embedder to embed transcripts.
    text_embed_fn: function(texts: List[str]) -> List[List[float]]
    """
    return text_embed_fn(transcripts)
