"""Reciprocal Rank Fusion (RRF) search strategy."""

from typing import Dict, List

from langchain_core.documents import Document

from self_rag.search.base import SearchStrategy
from self_rag.vectordb.base import AbstractVectorDB


class RRFStrategy(SearchStrategy):
    """
    Reciprocal Rank Fusion combines dense + sparse results.

    Uses RRF formula: score = sum(1 / (k + rank))
    where k is a constant (typically 60) to avoid division by zero.

    Works best with databases that have native hybrid support (Qdrant, Weaviate).
    For other DBs, falls back to manual fusion in Python.
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
        Execute RRF by combining dense + sparse results.

        Args:
            query: Text query
            vectordb: Vector database
            query_embedding: Dense vector
            top_k: Final result count
            collection_name: Collection to search

        Returns:
            Top-k documents by RRF score
        """
        k = 60  # RRF constant

        # Get dense results
        dense_results = vectordb.search_dense(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Over-fetch for fusion
            collection_name=collection_name,
        )

        # Get sparse results
        sparse_results = vectordb.search_sparse(
            query=query,
            top_k=top_k * 2,
            collection_name=collection_name,
        )

        # Manual RRF fusion
        docs_by_id: Dict[str, dict] = {}

        # Process dense results
        for rank, result in enumerate(dense_results):
            doc_id = result.document.metadata.get("child_id", str(id(result.document)))
            rrf_score = 1.0 / (k + rank)

            if doc_id not in docs_by_id:
                docs_by_id[doc_id] = {
                    "document": result.document,
                    "rrf_score": 0.0,
                    "dense_rank": rank,
                    "sparse_rank": None,
                }
            docs_by_id[doc_id]["rrf_score"] += rrf_score
            docs_by_id[doc_id]["dense_rank"] = rank

        # Process sparse results
        for rank, result in enumerate(sparse_results):
            doc_id = result.document.metadata.get("child_id", str(id(result.document)))
            rrf_score = 1.0 / (k + rank)

            if doc_id not in docs_by_id:
                docs_by_id[doc_id] = {
                    "document": result.document,
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "sparse_rank": rank,
                }
            docs_by_id[doc_id]["rrf_score"] += rrf_score
            docs_by_id[doc_id]["sparse_rank"] = rank

        # Sort by RRF score, return top-k
        sorted_docs = sorted(
            docs_by_id.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        # Attach ranking metadata for debugging
        for item in sorted_docs[:top_k]:
            item["document"].metadata["rrf_score"] = item["rrf_score"]
            item["document"].metadata["dense_rank"] = item["dense_rank"]
            item["document"].metadata["sparse_rank"] = item["sparse_rank"]

        return [item["document"] for item in sorted_docs[:top_k]]
