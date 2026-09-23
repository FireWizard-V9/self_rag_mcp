"""Abstract search strategy interface."""

from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document

from self_rag.vectordb.base import AbstractVectorDB


class SearchStrategy(ABC):
    """Abstract base for hybrid search strategies."""

    @abstractmethod
    def execute(
        self,
        query: str,
        vectordb: AbstractVectorDB,
        query_embedding: List[float],
        top_k: int,
        collection_name: str,
    ) -> List[Document]:
        """
        Execute hybrid search strategy.

        Args:
            query: Text query for sparse search
            vectordb: Vector database adapter
            query_embedding: Dense embedding vector
            top_k: Number of results to return
            collection_name: Collection to search

        Returns:
            List of documents sorted by relevance score
        """
        pass
