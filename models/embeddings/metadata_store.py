import sqlite3
import json
from typing import Dict, Any

class MetadataStore:
    def __init__(self, db_path: str = "metadata.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    id TEXT PRIMARY KEY,
                    data TEXT
                )
            """)

    def store_metadata(self, doc_id: str, metadata: Dict[str, Any]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO metadata (id, data) VALUES (?, ?)",
                         (doc_id, json.dumps(metadata)))

    def get_metadata(self, doc_id: str) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT data FROM metadata WHERE id = ?", (doc_id,)).fetchone()
            return json.loads(row[0]) if row else {}