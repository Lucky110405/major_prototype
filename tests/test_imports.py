import importlib


def test_import_core_modules():
    """Smoke test: import core modules to detect syntax/import-time errors without instantiating heavy objects.

    This avoids creating network connections or downloading models by not calling constructors.
    """
    modules = [
        "models.embeddings.embedder",
        "models.embeddings.metadata_store",
        "retrieval.qdrant_adapter",
        "retrieval.multimodal_retriever",
        "retrieval.reranker",
        "agents.ingestion_agent",
    ]

    for name in modules:
        mod = importlib.import_module(name)
        assert mod is not None, f"Failed to import {name}"
