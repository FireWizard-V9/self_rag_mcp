"""Abstract Vector Database interface for DB-agnostic retrieval."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document


@dataclass
class VectorSearchResult:
    """Result from vector/sparse search."""

    document: Document
    score: float
    source: str  # "dense" | "sparse" | "hybrid"


class AbstractVectorDB(ABC):
    """
    Universal interface for vector database backends.

    Implementations support: Qdrant, Pinecone, Weaviate, Chroma, Milvus, etc.
    Adapters handle DB-specific APIs and return unified results.
    """

    @abstractmethod
    def search_dense(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """
        Search by dense vector embedding (semantic similarity).

        Args:
            query_embedding: Dense embedding vector (e.g., from OpenAI)
            top_k: Number of results to return
            filters: Optional metadata filters
            collection_name: Collection to search (if DB has multiple)

        Returns:
            List of documents with scores, sorted by relevance
        """
        pass

    @abstractmethod
    def search_sparse(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """
        Search by sparse vectors (BM25/keyword matching).
        Falls back to dense search if DB doesn't support sparse.

        Args:
            query: Text query
            top_k: Number of results to return
            filters: Optional metadata filters
            collection_name: Collection to search

        Returns:
            List of documents with scores
        """
        pass

    @abstractmethod
    def retrieve_by_ids(
        self,
        doc_ids: List[str],
        collection_name: Optional[str] = None,
    ) -> List[Document]:
        """
        Fetch specific documents by their IDs.
        Used for parent document expansion.

        Args:
            doc_ids: List of document IDs to retrieve
            collection_name: Collection to retrieve from

        Returns:
            List of documents (may be subset if some IDs missing)
        """
        pass

    @abstractmethod
    def upsert_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        doc_ids: List[str],
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Insert or update documents with their embeddings.

        Args:
            documents: List of Document objects with page_content + metadata
            embeddings: Dense embeddings (one per document)
            doc_ids: Unique IDs for each document (idempotent)
            collection_name: Collection to insert into
        """
        pass

    @abstractmethod
    def delete_documents(
        self,
        doc_ids: List[str],
        collection_name: Optional[str] = None,
    ) -> None:
        """Delete documents by ID."""
        pass

    @abstractmethod
    def ensure_collection(
        self,
        name: str,
        config: Dict[str, Any],
    ) -> bool:
        """
        Create collection if it doesn't exist.

        Args:
            name: Collection name
            config: Provider-specific config (dimension, distance metric, sparse support, etc.)

        Returns:
            True if created/exists, False if error
        """
        pass

    @abstractmethod
    def reset_collection(self, name: str) -> None:
        """Delete all documents in a collection (used for re-ingestion)."""
        pass

    @abstractmethod
    def collection_exists(self, name: str) -> bool:
        """Check if collection exists."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Verify database connectivity."""
        pass
