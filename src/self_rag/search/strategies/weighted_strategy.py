"""Weighted fusion search strategy."""

from typing import Dict, List

from langchain_core.documents import Document

from self_rag.search.base import SearchStrategy
from self_rag.vectordb.base import AbstractVectorDB


class WeightedStrategy(SearchStrategy):
    """
    Normalize dense + sparse scores, combine with configurable weight.

    Formula: final_score = alpha * normalized_dense + (1-alpha) * normalized_sparse
    Default alpha=0.5 means equal weight to both.

    Works with any vector database that supports both dense and sparse search.
    """

    def __init__(self, alpha: float = 0.5):
        """
        Initialize weighted strategy.

        Args:
            alpha: Weight for dense scores (0.0-1.0).
                   1.0 = dense only, 0.0 = sparse only, 0.5 = equal weight
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")
        self.alpha = alpha

    def execute(
        self,
        query: str,
        vectordb: AbstractVectorDB,
        query_embedding: List[float],
        top_k: int,
        collection_name: str,
    ) -> List[Document]:
        """
        Execute weighted fusion by normalizing and combining scores.

        Args:
            query: Text query
            vectordb: Vector database
            query_embedding: Dense vector
            top_k: Final result count
            collection_name: Collection to search

        Returns:
            Top-k documents by weighted score
        """
        # Over-fetch for deduplication
        num_candidates = top_k * 2

        dense_results = vectordb.search_dense(
            query_embedding=query_embedding,
            top_k=num_candidates,
            collection_name=collection_name,
        )

        sparse_results = vectordb.search_sparse(
            query=query,
            top_k=num_candidates,
            collection_name=collection_name,
        )

        # Normalize scores
        docs_by_id: Dict[str, dict] = {}

        # Process dense results
        if dense_results:
            dense_scores = [r.score for r in dense_results]
            max_dense = max(dense_scores)
            min_dense = min(dense_scores)
            dense_range = max_dense - min_dense

            for result in dense_results:
                doc_id = result.document.metadata.get(
                    "child_id", str(id(result.document))
                )
                normalized = (result.score - min_dense) / (dense_range or 1.0)

                if doc_id not in docs_by_id:
                    docs_by_id[doc_id] = {
                        "document": result.document,
                        "dense_score": 0.0,
                        "sparse_score": 0.0,
                    }
                docs_by_id[doc_id]["dense_score"] = normalized

        # Process sparse results
        if sparse_results:
            sparse_scores = [r.score for r in sparse_results]
            max_sparse = max(sparse_scores)
            min_sparse = min(sparse_scores)
            sparse_range = max_sparse - min_sparse

            for result in sparse_results:
                doc_id = result.document.metadata.get(
                    "child_id", str(id(result.document))
                )
                normalized = (result.score - min_sparse) / (sparse_range or 1.0)

                if doc_id not in docs_by_id:
                    docs_by_id[doc_id] = {
                        "document": result.document,
                        "dense_score": 0.0,
                        "sparse_score": 0.0,
                    }
                docs_by_id[doc_id]["sparse_score"] = normalized

        # Combine with weight
        for item in docs_by_id.values():
            item["final_score"] = (
                self.alpha * item["dense_score"]
                + (1 - self.alpha) * item["sparse_score"]
            )

        # Sort by final score
        sorted_docs = sorted(
            docs_by_id.values(),
            key=lambda x: x["final_score"],
            reverse=True,
        )

        # Attach scoring metadata
        for item in sorted_docs[:top_k]:
            item["document"].metadata["weighted_score"] = item["final_score"]
            item["document"].metadata["dense_normalized"] = item["dense_score"]
            item["document"].metadata["sparse_normalized"] = item["sparse_score"]

        return [item["document"] for item in sorted_docs[:top_k]]
