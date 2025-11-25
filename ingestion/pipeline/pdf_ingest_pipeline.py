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
from models.embeddings.vectorstore_faiss import FaissVectorStore
from models.embeddings.config import METADATA_DB_PATH, FAISS_INDEX_PATH

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf-ingest")

def ensure_dirs_for(path_str):
    p = Path(path_str)
    if not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)

def ingest_pdf(file_path: str, index_path: str = FAISS_INDEX_PATH, meta_db_path: str = METADATA_DB_PATH, limit_chunks: Optional[int] = None):
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
    dim = len(items[0]["embedding"])
    ensure_dirs_for(index_path)
    ensure_dirs_for(meta_db_path)
    meta_store = MetadataStore(meta_db_path)
    vs = FaissVectorStore(dim, index_path)

    # Upsert each item: store embedding in metadata_store (so rebuilds possible)
    logger.info("Upserting %d items into metadata store and faiss index", len(items))
    for it in items:
        # store embedding as list for sqlite (metadata_store handles serialization)
        emb_list = it["embedding"].tolist() if hasattr(it["embedding"], "tolist") else list(map(float, it["embedding"]))
        meta_store.upsert(it["id"], it["metadata"], it.get("text"), emb_list)

    # Upsert vectors into faiss (FaissVectorStore.upsert expects embedding in items)
    vs.upsert(items, meta_store)
    logger.info("Upsert complete. Index now has %d vectors (approx).", vs.index.ntotal if vs.index else 0)

    # Demonstration query
    sample_query = "Why did sales increase in region A?"  # change as you like
    logger.info("Running demo query: %s", sample_query)
    q_vec = np.array(embed_texts([sample_query])[0], dtype=np.float32)
    results = vs.search(q_vec, top_k=5)
    if not results:
        logger.info("No results returned.")
        return

    logger.info("Top results:")
    for cid, score in results:
        rec = meta_store.get(cid)
        text = rec.get("text") if rec else "<no text>"
        meta = rec.get("metadata") if rec else {}
        logger.info("score=%.4f id=%s meta=%s text(leading)=%s", score, cid, meta, (text[:200].replace("\n"," ") if text else ""))

def main():
    parser = argparse.ArgumentParser(description="PDF ingest -> embeddings -> faiss index")
    parser.add_argument("pdf", help="Path to PDF file to ingest")
    parser.add_argument("--index-path", default=FAISS_INDEX_PATH, help="FAISS index path")
    parser.add_argument("--meta-db", default=METADATA_DB_PATH, help="Metadata sqlite path")
    parser.add_argument("--limit-chunks", type=int, default=None, help="Only ingest first N chunks (for testing)")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        logger.error("File does not exist: %s", args.pdf)
        return

    ingest_pdf(args.pdf, index_path=args.index_path, meta_db_path=args.meta_db, limit_chunks=args.limit_chunks)

if __name__ == "__main__":
    main()
