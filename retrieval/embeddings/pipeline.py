# ingestion/retrieval/embeddings/pipeline.py

from typing import List
from .embedder import embed_texts, chunk_id
from .multimodal_embedder import embed_images, embed_audio_transcripts
from .metadata_store import MetadataStore
from .vectorstore_faiss import FaissVectorStore
from .utils import to_numpy
from config import METADATA_DB_PATH, FAISS_INDEX_PATH, DEFAULT_PROVENANCE

def run_embedding_pipeline(chunks: List[dict], meta_db_path: str = METADATA_DB_PATH, index_path: str = FAISS_INDEX_PATH):
    """
    chunks: list of dicts, each dict:
      {
        "modality": "text"|"table"|"image"|"audio_transcript",
        "text": "...",                # for text or table or transcript
        "image_path": "/tmp/abc.png", # for images
        "metadata": {...}
      }
    """
    # normalize metadata
    for c in chunks:
        c.setdefault("metadata", {})
        # merge default provenance if not present
        for k,v in DEFAULT_PROVENANCE.items():
            c["metadata"].setdefault(k, v)

    # split by modality
    text_chunks = [c for c in chunks if c.get("modality", "text") in ("text","table","audio_transcript")]
    image_chunks = [c for c in chunks if c.get("modality") == "image"]

    # embed text-like items
    texts = [c["text"] for c in text_chunks]
    if texts:
        text_embeddings = embed_texts(texts)
    else:
        text_embeddings = []

    # embed images
    image_embeddings = []
    if image_chunks:
        image_paths = [c["image_path"] for c in image_chunks]
        image_embeddings = embed_images(image_paths)

    # create items with deterministic ids
    items = []
    # text items
    for c, emb in zip(text_chunks, text_embeddings):
        cid = chunk_id(c["text"], c["metadata"])
        items.append({"id": cid, "embedding": emb, "metadata": c["metadata"], "text": c["text"]})
    # image items
    for c, emb in zip(image_chunks, image_embeddings):
        # use file path + metadata to form id
        cid = chunk_id(c.get("image_path","") + (c.get("text","") or ""), c["metadata"])
        items.append({"id": cid, "embedding": emb, "metadata": c["metadata"], "text": c.get("text","")})

    if not items:
        return {"upserted": 0, "error": "no items"}

    # init stores
    dim = len(items[0]["embedding"])
    meta = MetadataStore(meta_db_path)
    vs = FaissVectorStore(dim, index_path)

    # persist into metadata store with embedding
    for it in items:
        meta.upsert(it["id"], it["metadata"], it.get("text"), it["embedding"])

    # upsert into FAISS
    vs.upsert(items)
    return {"upserted": len(items)}
