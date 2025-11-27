"""
ingestion/pipelines/pdf_ingest_pipeline.py

Usage:
    python pdf_ingest_pipeline.py /path/to/file.pdf --index-path data/faiss.index --meta-db data/metadata.db

This script:
1. Parses the PDF using the parser pipeline (text, tables, images, chunks).
2. Converts chunks -> embeddings (sentence-transformers or OpenAI depending on config).
3. Upserts embeddings + metadata into MetadataStore (sqlite) and FaissVectorStore.
4. Prints a small search demo.
"""

import argparse
import os
import logging
from pathlib import Path
import numpy as np
from typing import Optional

# Adjust imports according to your package layout; if running as script, make sure PYTHONPATH includes repo root.
from ingestion.multimodal_unstructured_data.pdf_parser import parse_pdf            # your parser pipeline
from models.embeddings.embedder import create_chunks_with_embeddings, embed_texts
from models.embeddings.metadata_store import MetadataStore
from retrieval.qdrant_adapter import QdrantAdapter
from retrieval.embeddings.config import METADATA_DB_PATH, FAISS_INDEX_PATH

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf-ingest")

def ensure_dirs_for(path_str):
    p = Path(path_str)
    if not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)

def ingest_pdf(file_path: str, meta_db_path: str = METADATA_DB_PATH, limit_chunks: Optional[int] = None):
    logger.info("Parsing PDF: %s", file_path)
    parsed = parse_pdf(file_path)
    # parsed expected to contain: metadata, raw_text (pages), tables, images, chunks (list of {"page","text","metadata"})
    chunks = parsed.get("chunks", [])
    logger.info("Parsed %d chunks", len(chunks))

    if limit_chunks:
        chunks = chunks[:limit_chunks]
        logger.info("Limiting to %d chunks for this run", len(chunks))

    if not chunks:
        logger.warning("No chunks produced from parser; aborting ingest.")
        return

    # create dense vectors
    logger.info("Creating embeddings for chunks (this may take a while)...")
    items = create_chunks_with_embeddings([{"text": c["text"], "metadata": {**c.get("metadata", {}), "source_file": os.path.basename(file_path), "page": c.get("page")}} for c in chunks])

    # Prepare stores
    ensure_dirs_for(meta_db_path)
    meta_store = MetadataStore(meta_db_path)
    qdrant_adapter = QdrantAdapter(host="localhost", port=6333)  # Assuming Qdrant is running

    # Upsert each item: store embedding in metadata_store (so rebuilds possible)
    logger.info("Upserting %d items into metadata store and Qdrant", len(items))
    embeddings = []
    metadatas = []
    ids = []
    for it in items:
        # store embedding as list for sqlite (metadata_store handles serialization)
        emb_list = it["embedding"].tolist() if hasattr(it["embedding"], "tolist") else list(map(float, it["embedding"]))
        meta_store.upsert(it["id"], it["metadata"], it.get("text"), emb_list)
        embeddings.append(emb_list)
        metadatas.append(it["metadata"])
        ids.append(it["id"])

    # Upsert vectors into Qdrant
    qdrant_adapter.upsert_vectors("text_docs", embeddings, metadatas, ids)
    logger.info("Upsert complete.")

def main():
    parser = argparse.ArgumentParser(description="PDF ingest -> embeddings -> Qdrant")
    parser.add_argument("pdf", help="Path to PDF file to ingest")
    parser.add_argument("--meta-db", default=METADATA_DB_PATH, help="Metadata sqlite path")
    parser.add_argument("--limit-chunks", type=int, default=None, help="Only ingest first N chunks (for testing)")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        logger.error("File does not exist: %s", args.pdf)
        return

    ingest_pdf(args.pdf, meta_db_path=args.meta_db, limit_chunks=args.limit_chunks)

if __name__ == "__main__":
    main()
