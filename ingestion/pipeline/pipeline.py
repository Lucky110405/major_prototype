from models.embeddings.embedder import Embedder
from models.embeddings.chunker import TextChunker
from models.embeddings.metadata import generate_metadata
from vectorstore.qdrant_client import QdrantDB

class IngestionPipeline:
    def __init__(self):
        self.chunker = TextChunker()
        self.embedder = Embedder()
        self.vdb = QdrantDB()

    def ingest_pdf(self, document_path):
        # Extract text from PDF (already done previously)
        from ingest.pdf_loader import load_pdf_text
        text = load_pdf_text(document_path)

        chunks = self.chunker.split(text)
        embeddings = self.embedder.get_embeddings(chunks)

        self.vdb.create_collection(dim=len(embeddings[0]))

        metadata = generate_metadata(chunks, source=document_path)

        self.vdb.add_embeddings(embeddings, metadata)
