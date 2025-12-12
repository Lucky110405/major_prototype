import os
import json
from typing import Dict, Any
from supabase import create_client, Client

class MetadataStore:
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set")
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)

    def store_metadata(self, doc_id: str, metadata: Dict[str, Any]):
        data = json.dumps(metadata)
        self.supabase.table('backend_metadata').upsert({'id': doc_id, 'data': data}).execute()

    def upsert(self, doc_id: str, metadata: Dict[str, Any], text: str = None, embedding: list = None):
        full_metadata = metadata.copy()
        if text:
            full_metadata["text"] = text
        if embedding:
            full_metadata["embedding"] = embedding
        self.store_metadata(doc_id, full_metadata)

    def get_metadata(self, doc_id: str) -> Dict[str, Any]:
        response = self.supabase.table('backend_metadata').select('data').eq('id', doc_id).execute()
        if response.data:
            return json.loads(response.data[0]['data'])
        return {}

    def get_all_documents(self):
        response = self.supabase.table('backend_metadata').select('*').execute()
        documents = []
        for item in response.data:
            raw = item.get('data')
            # parse data which may already be JSON string or a dict
            if isinstance(raw, str):
                doc_data = json.loads(raw)
            elif isinstance(raw, dict):
                doc_data = raw
            else:
                # fallback: treat as string
                try:
                    doc_data = json.loads(str(raw))
                except Exception:
                    doc_data = {}

            doc = {
                'id': item.get('id') or doc_data.get('id'),
                'file_name': doc_data.get('file_name') or doc_data.get('filename') or doc_data.get('source_file') or doc_data.get('filename', ''),
                'file_type': doc_data.get('file_type') or doc_data.get('type') or '',
                'file_size': doc_data.get('file_size') or doc_data.get('size') or 0,
                'content': doc_data.get('content') or doc_data.get('text') or doc_data.get('text_excerpt') or None,
                'uploaded_at': doc_data.get('uploaded_at') or doc_data.get('created_at') or '',
                'user_id': doc_data.get('user_id') or None,
            }
            documents.append(doc)
        return documents

    def delete_document(self, doc_id: str):
        self.supabase.table('backend_metadata').delete().eq('id', doc_id).execute()