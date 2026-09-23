"""Factory for creating vector database adapters."""

from self_rag.core.config import get_settings
from self_rag.vectordb.base import AbstractVectorDB


def get_vectordb() -> AbstractVectorDB:
    """
    Factory function to get the configured vector database adapter.
    Selection via VECTORDB_PROVIDER env var.

    Returns:
        AbstractVectorDB: Concrete adapter instance (Qdrant, Pinecone, etc.)

    Raises:
        ValueError: If VECTORDB_PROVIDER is unknown
    """
    settings = get_settings()
    provider = settings.vectordb_provider.lower()

    if provider == "qdrant":
        from self_rag.vectordb.adapters.qdrant_adapter import get_qdrant_adapter

        return get_qdrant_adapter()

    elif provider == "pinecone":
        from self_rag.vectordb.adapters.pinecone_adapter import get_pinecone_adapter

        return get_pinecone_adapter()

    elif provider == "weaviate":
        from self_rag.vectordb.adapters.weaviate_adapter import get_weaviate_adapter

        return get_weaviate_adapter()

    elif provider == "chroma":
        from self_rag.vectordb.adapters.chroma_adapter import get_chroma_adapter

        return get_chroma_adapter()

    else:
        raise ValueError(
            f"Unknown VECTORDB_PROVIDER: {provider}. "
            f"Supported: qdrant, pinecone, weaviate, chroma"
        )
