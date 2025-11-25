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

class ExcelIngestor:
    def __init__(self, metadata_store: MetadataStore, qdrant_adapter: QdrantAdapter, text_embedder: TextEmbedder):
        """
        Initialize the Excel Ingestor.
        
        Args:
            metadata_store: Instance of MetadataStore.
            qdrant_adapter: Instance of QdrantAdapter.
            text_embedder: Instance of TextEmbedder.
        """
        self.metadata_store = metadata_store
        self.qdrant_adapter = qdrant_adapter
        self.text_embedder = text_embedder
        self.qdrant_adapter.create_collection_if_not_exists("text_docs", vector_size=768)

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

    def process_excel(self, excel_path: str, sheet_name: str = 0, source: str = "unknown") -> bool:
        """
        Process an Excel file: load sheet, extract insights, chunk, embed, store.
        
        Args:
            excel_path: Path to the Excel file.
            sheet_name: Sheet name or index.
            source: Source identifier.
        
        Returns:
            True if successful.
        """
        if not os.path.exists(excel_path):
            logger.error(f"Excel file does not exist: {excel_path}")
            return False
        
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            insights = self.extract_insights(df)
            chunks = self.chunk_table(df)
            
            vectors = []
            for i, chunk in enumerate(chunks):
                doc_id = f"{source}_excel_{os.path.basename(excel_path)}_sheet_{sheet_name}_chunk_{i}"
                metadata = {
                    "source": source,
                    "file_path": excel_path,
                    "sheet": str(sheet_name),
                    "type": "spreadsheet",
                    "chunk_index": i,
                    "insights": insights,
                    "text": chunk
                }
                self.metadata_store.store_metadata(doc_id, metadata)
                embedding = self.text_embedder.embed([chunk])[0]
                vectors.append((doc_id, embedding, metadata))
            
            if vectors:
                self.qdrant_adapter.upsert_vectors("text_docs", vectors)
                logger.info(f"Processed Excel {excel_path} sheet {sheet_name} into {len(vectors)} chunks")
                return True
        except Exception as e:
            logger.error(f"Error processing Excel {excel_path}: {e}")
            return False