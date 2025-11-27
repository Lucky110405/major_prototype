import logging
from typing import List, Dict, Any, Optional
from retrieval.qdrant_adapter import QdrantAdapter
from retrieval.reranker import CrossEncoderReranker

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalRetriever:
    def __init__(self, qdrant_adapter: QdrantAdapter, text_collection: str = "text_docs", image_collection: str = "image_docs", reranker: Optional[CrossEncoderReranker] = None):
        """
        Initialize the Multimodal Retriever.
        
        Args:
            qdrant_adapter: Instance of QdrantAdapter.
            text_collection: Name of the text documents collection.
            image_collection: Name of the image documents collection.
        """
        self.qdrant_adapter = qdrant_adapter
        self.text_collection = text_collection
        self.image_collection = image_collection
        self.reranker = reranker

    def retrieve_text(self, query_vector: List[float], top_k: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Retrieve from text collection.
        
        Args:
            query_vector: Embedding vector for the query.
            top_k: Number of top results.
            filters: Optional metadata filters.
        
        Returns:
            List of results with id, score, metadata, type.
        """
        results = self.qdrant_adapter.search(self.text_collection, query_vector, top_k=top_k, filters=filters)
        for res in results:
            res["type"] = "text"
        return results

    def retrieve_images(self, query_vector: List[float], top_k: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Retrieve from image collection.
        
        Args:
            query_vector: Embedding vector for the query.
            top_k: Number of top results.
            filters: Optional metadata filters.
        
        Returns:
            List of results with id, score, metadata, type.
        """
        results = self.qdrant_adapter.search(self.image_collection, query_vector, top_k=top_k, filters=filters)
        for res in results:
            res["type"] = "image"
        return results

    def normalize_scores(self, results: List[Dict]) -> List[Dict]:
        """
        Normalize scores to [0, 1] range.
        
        Args:
            results: List of result dicts with 'score'.
        
        Returns:
            Results with normalized scores.
        """
        if not results:
            return results
        scores = [r["score"] for r in results]
        min_score, max_score = min(scores), max(scores)
        if max_score == min_score:
            for r in results:
                r["normalized_score"] = 1.0
        else:
            for r in results:
                r["normalized_score"] = (r["score"] - min_score) / (max_score - min_score)
        return results

    def fuse_results(self, text_results: List[Dict], image_results: List[Dict], 
                     text_weight: float = 0.5, image_weight: float = 0.5, 
                     fusion_method: str = "weighted") -> List[Dict]:
        """
        Fuse text and image results.
        
        Args:
            text_results: Results from text retrieval.
            image_results: Results from image retrieval.
            text_weight: Weight for text scores.
            image_weight: Weight for image scores.
            fusion_method: "weighted" or "rrf" (reciprocal rank fusion).
        
        Returns:
            Fused and sorted results.
        """
        # Normalize scores
        text_results = self.normalize_scores(text_results)
        image_results = self.normalize_scores(image_results)
        
        if fusion_method == "weighted":
            # Combine scores with weights
            all_results = text_results + image_results
            for res in all_results:
                if res["type"] == "text":
                    res["fused_score"] = res["normalized_score"] * text_weight
                else:
                    res["fused_score"] = res["normalized_score"] * image_weight
        elif fusion_method == "rrf":
            # Reciprocal Rank Fusion
            all_results = []
            id_to_result = {}
            
            for i, res in enumerate(text_results):
                rank = i + 1
                res["rrf_score"] = 1.0 / (60 + rank)  # Standard RRF constant
                res["fused_score"] = res["rrf_score"]
                id_to_result[res["id"]] = res
            
            for i, res in enumerate(image_results):
                rank = i + 1
                rrf = 1.0 / (60 + rank)
                if res["id"] in id_to_result:
                    id_to_result[res["id"]]["fused_score"] += rrf
                else:
                    res["rrf_score"] = rrf
                    res["fused_score"] = rrf
                    id_to_result[res["id"]] = res
            
            all_results = list(id_to_result.values())
        else:
            raise ValueError("Unsupported fusion method")
        
        # Sort by fused score descending
        all_results.sort(key=lambda x: x["fused_score"], reverse=True)
        return all_results

    def retrieve(self, query_vector: List[float], top_k: int = 10, 
                 text_filters: Optional[Dict] = None, image_filters: Optional[Dict] = None,
                 text_weight: float = 0.5, image_weight: float = 0.5, 
                 fusion_method: str = "weighted", query_text: Optional[str] = None) -> List[Dict]:
        """
        Perform multimodal retrieval.
        
        Args:
            query_vector: Embedding vector for the query.
            top_k: Total number of top results after fusion.
            text_filters: Filters for text collection.
            image_filters: Filters for image collection.
            text_weight: Weight for text scores.
            image_weight: Weight for image scores.
            fusion_method: Fusion method ("weighted" or "rrf").
        
        Returns:
            Unified list of top results.
        """
        # Retrieve from both collections
        text_results = self.retrieve_text(query_vector, top_k=top_k, filters=text_filters)
        # If a reranker is provided and we have the original query text, apply reranking
        if self.reranker and query_text and text_results:
            # Prepare candidate dicts expected by reranker
            candidates = []
            for r in text_results:
                candidates.append({
                    "id": r.get("id"),
                    "score": r.get("score"),
                    "metadata": r.get("metadata", {})
                })
            reranked = self.reranker.rerank(query_text, candidates, top_k=top_k)
            # map reranked entries back to retrieval format
            text_results = []
            for rr in reranked:
                text_results.append({"id": rr.get("id"), "score": rr.get("rerank_score"), "metadata": rr.get("metadata", {}), "type": "text"})
        # image_results = self.retrieve_images(query_vector, top_k=top_k, filters=image_filters)
        
        # Fuse results
        # fused_results = self.fuse_results(text_results, image_results, text_weight, image_weight, fusion_method)
        
        # Return top_k
        return text_results[:top_k]