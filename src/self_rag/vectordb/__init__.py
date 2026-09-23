"""Vector database abstraction layer."""

from self_rag.vectordb.base import AbstractVectorDB, VectorSearchResult
from self_rag.vectordb.factory import get_vectordb

__all__ = ["AbstractVectorDB", "VectorSearchResult", "get_vectordb"]
