from functools import lru_cache

from qdrant_client import QdrantClient

from self_rag.core.config import get_settings


@lru_cache
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()

    return QdrantClient(
        url=settings.qdrant_url,
    )
