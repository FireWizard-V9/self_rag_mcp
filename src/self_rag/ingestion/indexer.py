from qdrant_client.models import (
    Distance,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from self_rag.clients.qdrant import get_qdrant_client
from self_rag.core.config import get_settings


def _create_collection_if_missing(collection_name: str, *, sparse: bool) -> None:
    """Create a Qdrant collection used for either child search or parent storage."""
    settings = get_settings()
    client = get_qdrant_client()

    if client.collection_exists(collection_name):
        return

    sparse_vectors_config = None
    if sparse:
        sparse_vectors_config = {
            "langchain-sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        }

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=settings.embedding_dim,
            distance=Distance.COSINE,
        ),
        sparse_vectors_config=sparse_vectors_config,
    )


def ensure_collections_exist() -> None:
    """
    Creates and configures the Qdrant collection for Hybrid Search:
    - Dense vectors (semantic) using settings.embedding_dim
    - Sparse vectors (BM25 keyword) using 'langchain-sparse'
    """
    settings = get_settings()

    _create_collection_if_missing(settings.qdrant_collection, sparse=True)
    _create_collection_if_missing(settings.qdrant_parent_collection, sparse=False)


def reset_collections() -> None:
    """Delete this application's child and parent collections before a full reindex."""
    settings = get_settings()
    client = get_qdrant_client()

    for collection_name in (
        settings.qdrant_collection,
        settings.qdrant_parent_collection,
    ):
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
