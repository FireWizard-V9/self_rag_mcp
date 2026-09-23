"""Semantic-only search strategy (fallback for databases without sparse support)."""

from typing import List

from langchain_core.documents import Document

from self_rag.search.base import SearchStrategy
from self_rag.vectordb.base import AbstractVectorDB


class SemanticStrategy(SearchStrategy):
    """
    Dense vectors only + reranking fallback.

    Used for databases that don't support sparse/keyword search (Pinecone, simple vector DBs).
    Quality is maintained via FlashRank reranking in the retriever pipeline.

    This is a fallback strategy when hybrid search is not available.
    """

    def execute(
        self,
        query: str,
        vectordb: AbstractVectorDB,
        query_embedding: List[float],
        top_k: int,
        collection_name: str,
    ) -> List[Document]:
        """
        Execute dense semantic search only.

        Args:
            query: Text query (unused, for interface compatibility)
            vectordb: Vector database
            query_embedding: Dense vector
            top_k: Result count
            collection_name: Collection to search

        Returns:
            Top-k documents by semantic similarity
        """
        results = vectordb.search_dense(
            query_embedding=query_embedding,
            top_k=top_k,
            collection_name=collection_name,
        )

        # Attach strategy info
        for result in results:
            result.document.metadata["search_strategy"] = "semantic"

        return [result.document for result in results]
