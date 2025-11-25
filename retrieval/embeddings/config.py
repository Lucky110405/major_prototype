# ingestion/retrieval/embeddings/config.py
import os

EMBEDDER_PROVIDER = os.getenv("EMBEDDER_PROVIDER", "sentence_transformers")
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Image embedder (CLIP-like) provider choice ("openai_clip" or "clip")
IMAGE_EMBEDDER_PROVIDER = os.getenv("IMAGE_EMBEDDER_PROVIDER", "clip")
CLIP_MODEL = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")  # if using huggingface clip

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss.index")
METADATA_DB_PATH = os.getenv("METADATA_DB_PATH", "data/metadata.db")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))

# Metadata fields persisted with each chunk
DEFAULT_PROVENANCE = {
    "source_system": "pdf_parser",
    "parser_version": "v1",
}
