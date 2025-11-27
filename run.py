#!/usr/bin/env python3
"""
Run script for the Collaborative Multi-Modal Agentic BI Framework.
Sets up components and starts the FastAPI server.
"""

import logging
import os
from models.embeddings.embedder import TextEmbedder
from models.embeddings.metadata_store import MetadataStore
from retrieval.qdrant_adapter import QdrantAdapter
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.multimodal_retriever import MultimodalRetriever
from retrieval.reranker import CrossEncoderReranker
from agents.intent_agent import IntentAgent
from agents.retriever_agent import RetrieverAgent
from agents.analyzer_agent import AnalyzerAgent
from agents.visual_agent import VisualAgent
from agents.ingestion_agent import IngestionAgent
from agents.modality_agent import ModalityAgent
from agents.orchestrator import Orchestrator
from api.main import app
import uvicorn
from dotenv import load_dotenv
load_dotenv()


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_components():
    """Initialize all components."""
    logger.info("Setting up components...")

    # Metadata Store
    metadata_store = MetadataStore(db_path="metadata.db")

    # Qdrant Adapter
    qdrant_adapter = QdrantAdapter(host="localhost", port=6333)

    # Text Embedder
    text_embedder = TextEmbedder()

    # Retrievers
    hybrid_retriever = HybridRetriever(qdrant_adapter, bm25_index=None)  # Assume BM25 is set up
    # Optionally initialize a cross-encoder reranker. Set environment var `SKIP_RERANKER=1` to skip loading.
    reranker = None
    skip_reranker = os.getenv("SKIP_RERANKER", "0")
    if skip_reranker == "1":
        logger.info("SKIP_RERANKER=1 set; skipping reranker initialization.")
    else:
        try:
            reranker = CrossEncoderReranker()
            logger.info("Cross-encoder reranker loaded successfully.")
        except Exception as e:
            logger.warning(f"Reranker not available: {e}")

    multimodal_retriever = MultimodalRetriever(qdrant_adapter, reranker=reranker)

    # Agents
    intent_agent = IntentAgent()
    retriever_agent = RetrieverAgent(hybrid_retriever, multimodal_retriever)
    analyzer_agent = AnalyzerAgent()
    visual_agent = VisualAgent()
    ingestion_agent = IngestionAgent(metadata_store, qdrant_adapter, text_embedder)
    modality_agent = ModalityAgent()
    orchestrator = Orchestrator(intent_agent, retriever_agent, analyzer_agent, visual_agent)

    # Store in app state for dependency injection
    app.state.metadata_store = metadata_store
    app.state.qdrant_adapter = qdrant_adapter
    app.state.text_embedder = text_embedder
    app.state.multimodal_retriever = multimodal_retriever
    app.state.hybrid_retriever = hybrid_retriever
    app.state.ingestion_agent = ingestion_agent
    app.state.modality_agent = modality_agent
    app.state.orchestrator = orchestrator

    logger.info("Components set up successfully.")

if __name__ == "__main__":
    setup_components()
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)