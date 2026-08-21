from functools import lru_cache

from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode

from self_rag.clients.llm import get_embedding_model
from self_rag.clients.qdrant import get_qdrant_client
from self_rag.core.config import get_settings


@lru_cache(maxsize=1)
def get_sparse_embedding() -> FastEmbedSparse:
    return FastEmbedSparse(model_name="Qdrant/bm25")


@lru_cache
def get_vector_store() -> QdrantVectorStore:
    """
    Qdrant Hybrid Vector Store.

    Combines:
    - Dense embeddings (OpenAI) for semantic matching
    - Sparse embeddings (FastEmbed BM25) for keyword/phrase matching
    - Server-side Hybrid Fusion (RRF) in Qdrant
    """

    settings = get_settings()

    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=settings.qdrant_collection,
        embedding=get_embedding_model(),
        sparse_embedding=get_sparse_embedding(),
        retrieval_mode=RetrievalMode.HYBRID,
    )


@lru_cache
def get_parent_vector_store() -> QdrantVectorStore:
    """Store complete parent chunks for expansion after child-level retrieval."""
    settings = get_settings()
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=settings.qdrant_parent_collection,
        embedding=get_embedding_model(),
        retrieval_mode=RetrievalMode.DENSE,
    )
