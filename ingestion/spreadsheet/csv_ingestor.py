import os
import logging
from typing import List, Dict, Any
import pandas as pd
from models.embeddings.embedder import TextEmbedder
from models.embeddings.metadata_store import MetadataStore
from retrieval.qdrant_adapter import QdrantAdapter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVIngestor:
    def __init__(self, metadata_store: MetadataStore, qdrant_adapter: QdrantAdapter, text_embedder: TextEmbedder):
        """
        Initialize the CSV Ingestor.
        
        Args:
            metadata_store: Instance of MetadataStore.
            qdrant_adapter: Instance of QdrantAdapter.
            text_embedder: Instance of TextEmbedder.
        """
        self.metadata_store = metadata_store
        self.qdrant_adapter = qdrant_adapter
        self.text_embedder = text_embedder
        self.qdrant_adapter.create_collection_if_not_exists("text_docs", vector_size=768)  # Adjust size

    def extract_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract insights from the DataFrame.
        
        Args:
            df: Pandas DataFrame.
        
        Returns:
            Dict with column names, types, sample rows, etc.
        """
        insights = {
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "num_rows": len(df),
            "num_cols": len(df.columns),
            "sample_rows": df.head(5).to_dict(orient="records")
        }
        return insights

    def chunk_table(self, df: pd.DataFrame, chunk_size: int = 100) -> List[str]:
        """
        Chunk the table into text representations.
        
        Args:
            df: Pandas DataFrame.
            chunk_size: Number of rows per chunk.
        
        Returns:
            List of text chunks.
        """
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i+chunk_size]
            text = chunk_df.to_string(index=False)
            chunks.append(text)
        return chunks

    def process_csv(self, csv_path: str, source: str = "unknown", stored_path: str = None) -> bool:
        """
        Process a CSV file: load, extract insights, chunk, embed, store.
        
        Args:
            csv_path: Path to the CSV file.
            source: Source identifier.
        
        Returns:
            True if successful.
        """
        if not os.path.exists(csv_path):
            logger.error(f"CSV file does not exist: {csv_path}")
            return False
        
        try:
            df = pd.read_csv(csv_path)
            insights = self.extract_insights(df)
            chunks = self.chunk_table(df)
            
            vectors = []
            for i, chunk in enumerate(chunks):
                doc_id = f"{source}_csv_{os.path.basename(csv_path)}_chunk_{i}"
                metadata = {
                    "source": source,
                    "file_path": csv_path,
                    "type": "spreadsheet",
                    "chunk_index": i,
                    "insights": insights,
                    "text": chunk
                }
                if stored_path:
                    metadata['stored_path'] = stored_path
                self.metadata_store.store_metadata(doc_id, metadata)
                embedding = self.text_embedder.embed([chunk])[0]
                vectors.append((doc_id, embedding, metadata))
            
            if vectors:
                self.qdrant_adapter.upsert_vectors("text_docs", vectors)
                logger.info(f"Processed CSV {csv_path} into {len(vectors)} chunks")
                return True
        except Exception as e:
            logger.error(f"Error processing CSV {csv_path}: {e}")
            return False