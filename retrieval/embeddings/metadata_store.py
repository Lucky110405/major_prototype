# ingestion/retrieval/embeddings/metadata_store.py

import sqlite3
import json
import os
from typing import Optional, Any

class MetadataStore:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._ensure_table()

    def _ensure_table(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            text TEXT,
            metadata TEXT,
            embedding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        self.conn.commit()

    def upsert(self, chunk_id: str, metadata: dict, text: Optional[str] = None, embedding: Optional[list] = None):
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO chunks (id, text, metadata, embedding) VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET text=excluded.text, metadata=excluded.metadata, embedding=excluded.embedding
        """, (chunk_id, text or "", json.dumps(metadata), sqlite3.Binary(json.dumps(embedding).encode('utf-8')) if embedding is not None else None))
        self.conn.commit()

    def get(self, chunk_id: str) -> Optional[dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT id, text, metadata, embedding, created_at FROM chunks WHERE id = ?", (chunk_id,))
        row = cur.fetchone()
        if not row:
            return None
        _id, text, metadata_json, embedding_blob, created_at = row
        metadata = json.loads(metadata_json) if metadata_json else {}
        embedding = None
        if embedding_blob:
            embedding = json.loads(embedding_blob.decode('utf-8'))
        return {"id": _id, "text": text, "metadata": metadata, "embedding": embedding, "created_at": created_at}

    def delete(self, chunk_id: str):
        cur = self.conn.cursor()
        cur.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
        self.conn.commit()

    def list_ids(self):
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM chunks")
        return [r[0] for r in cur.fetchall()]
