import logging
from typing import Dict, Any, List
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.multimodal_retriever import MultimodalRetriever

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrieverAgent:
    def __init__(self, hybrid_retriever: HybridRetriever, multimodal_retriever: MultimodalRetriever):
        """
        Initialize the Retriever Agent.
        
        Args:
            hybrid_retriever: Instance of HybridRetriever for text.
            multimodal_retriever: Instance of MultimodalRetriever for images.
        """
        self.hybrid_retriever = hybrid_retriever
        self.multimodal_retriever = multimodal_retriever

    def run(self, query: str, query_vector: List[float], top_k: int = 10, intent: str = "descriptive") -> Dict[str, Any]:
        """
        Retrieve relevant chunks based on query and intent.
        
        Args:
            query: The user query string.
            query_vector: Embedding vector for the query.
            top_k: Number of top results.
            intent: Classified intent (affects retrieval strategy).
        
        Returns:
            Dict with retrieved chunks and metadata.
        """
        try:
            # Use multimodal retriever for combined text and image retrieval
            results = self.multimodal_retriever.retrieve(query_vector, top_k=top_k)
            
            # Optionally, use hybrid for text if needed
            if intent in ["diagnostic", "predictive"]:
                # For analytical queries, prioritize text
                text_results = self.hybrid_retriever.retrieve(query, top_k=top_k)
                # Merge or prioritize
                results.extend(text_results)
                results = sorted(results, key=lambda x: x.get("fused_score", x.get("score", 0)), reverse=True)[:top_k]
            
            selected_chunks = [res for res in results if res.get("score", 0) > 0.5]  # Threshold for relevance
            
            logger.info(f"Retrieved {len(selected_chunks)} relevant chunks")
            
            return {
                "chunks": selected_chunks,
                "total_retrieved": len(results),
                "strategy": "multimodal" if intent == "descriptive" else "hybrid"
            }
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return {
                "chunks": [],
                "total_retrieved": 0,
                "strategy": "failed"
            }