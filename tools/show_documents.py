#!/usr/bin/env python3
"""
Script to fetch and display all documents from Supabase backend_metadata table.
"""

import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_all_documents(use_backend: bool = True):
    """Fetch all documents: by default via backend API; optional direct Supabase access (legacy)."""
    if use_backend:
        api_url = os.getenv("BACKEND_URL") or "http://localhost:8000"
        try:
            resp = requests.get(f"{api_url}/documents")
            if resp.status_code != 200:
                print(f"Error fetching via backend: {resp.status_code} {resp.text}")
                return
            data = resp.json()
            docs = data.get("documents", [])
            if docs:
                print(f"Found {len(docs)} documents via backend:")
                print("-" * 50)
                for d in docs:
                    print(f"Document ID: {d.get('id')}")
                    print(f"File Name: {d.get('file_name', 'N/A')}")
                    print(f"File Type: {d.get('file_type', 'N/A')}")
                    print(f"File Size: {d.get('file_size', 'N/A')} bytes")
                    print(f"Uploaded At: {d.get('uploaded_at', 'N/A')}")
                    preview = d.get('content') or d.get('text') or d.get('text_excerpt')
                    print(f"Content Preview: {str(preview)[:100]}...")
                    print("-" * 50)
            else:
                print("No documents found via backend API.")
        except Exception as e:
            print(f"Error fetching documents via backend: {e}")
        return

    # Legacy: direct Supabase access (not recommended)
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    if not supabase_url or not supabase_key:
        print("No Supabase keys found in .env; cannot fetch directly.")
        return
    try:
        from supabase import create_client
        supabase = create_client(supabase_url, supabase_key)
        response = supabase.table('backend_metadata').select('*').execute()
        if response.data:
            print(f"Found {len(response.data)} documents (direct Supabase):")
            print("-" * 50)
            for item in response.data:
                doc_id = item.get('id')
                raw = item.get('data')
                doc_data = json.loads(raw) if isinstance(raw, str) else raw
                print(f"Document ID: {doc_id}")
                print(f"File Name: {doc_data.get('file_name', 'N/A')}\n")
                print(f"File Type: {doc_data.get('file_type', 'N/A')}\n")
                print(f"File Size: {doc_data.get('file_size', 'N/A')} bytes\n")
                print(f"Uploaded At: {doc_data.get('uploaded_at', 'N/A')}\n")
                print(f"Content Preview: {doc_data.get('content', 'N/A')[:100]}...\n")
                print("-" * 50)
        else:
            print("No documents found in the database (direct Supabase).")
    except Exception as e:
        print(f"Error fetching documents directly from Supabase: {e}")

if __name__ == "__main__":
    get_all_documents()