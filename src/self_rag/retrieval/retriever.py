from functools import lru_cache

from langchain_core.documents import Document

from self_rag.clients.llm import get_embedding_model
from self_rag.core.config import get_settings
from self_rag.retrieval.mmr import apply_mmr
from self_rag.retrieval.reranker import rerank_documents
from self_rag.search.factory import get_search_strategy
from self_rag.vectordb.factory import get_vectordb


class HybridRetriever:
    """
    DB-agnostic hybrid retrieval pipeline:

    1. Hybrid search (Dense + Sparse via SearchStrategy)
    2. MMR (diversity reranking)
    3. Cross-Encoder Reranking (FlashRank)
    4. Parent document expansion
    """

    def __init__(self):
        self.vectordb = get_vectordb()
        self.search_strategy = get_search_strategy()
        self.embedding_model = get_embedding_model()
        self.settings = get_settings()

    def invoke(self, query: str, top_k: int | None = None, expand_parents: bool = True) -> list[Document]:
        """
        Run full retrieval pipeline.

        Args:
            query: Search query
            top_k: Number of documents to return
            expand_parents: If True, fetch parent documents (if configured).
                           If False, skip expansion and return ranked docs directly.
        """

        # Get query embedding
        query_embedding = self.embedding_model.embed_query(query)

        # Step 1: Hybrid search (dense + sparse fusion)
        candidate_count = max(
            self.settings.retrieval_k_initial,
            self.settings.retrieval_k_mmr,
            top_k or self.settings.retrieval_k_rerank,
        )

        docs = self.search_strategy.execute(
            query=query,
            vectordb=self.vectordb,
            query_embedding=query_embedding,
            top_k=candidate_count,
            collection_name=self.settings.retrieval_collection,
        )

        # Step 2: MMR diversity reranking
        docs = apply_mmr(
            query,
            docs,
            k=self.settings.retrieval_k_mmr,
            lambda_param=self.settings.retrieval_lambda,
        )

        # Step 3: Cross-encoder reranking (FlashRank)
        docs = rerank_documents(
            query,
            docs,
            top_k=top_k or self.settings.retrieval_k_rerank,
        )

        # Step 4: Parent document expansion (optional)
        if expand_parents and self.settings.parent_expansion_collection:
            return self._expand_to_parent_documents(docs)

        return docs

    def _expand_to_parent_documents(
        self, ranked_children: list[Document]
    ) -> list[Document]:
        """
        Fetch each unique full parent chunk for the ranked child results.
        Gracefully handles flat (no-hierarchy) ingestion by returning children as-is.
        """
        parent_ids: list[str] = []
        child_metadata_by_parent: dict[str, list[dict]] = {}

        for child in ranked_children:
            parent_id = child.metadata.get("parent_id")
            if not parent_id:
                continue
            if parent_id not in child_metadata_by_parent:
                parent_ids.append(parent_id)
                child_metadata_by_parent[parent_id] = []
            child_metadata_by_parent[parent_id].append(dict(child.metadata))

        # No parent_ids found → flat ingestion, return ranked children as-is
        if not parent_ids:
            return ranked_children

        try:
            # Use VectorDB adapter to fetch parents (DB-agnostic)
            parents_docs = self.vectordb.retrieve_by_ids(
                doc_ids=parent_ids,
                collection_name=self.settings.parent_expansion_collection,
            )
        except Exception as e:
            # Parent collection doesn't exist or fetch failed → return children
            import logging
            logger = logging.getLogger("self_rag")
            logger.debug(
                f"Parent expansion failed (collection may not exist): {e}. "
                f"Returning ranked children instead."
            )
            return ranked_children

        parents: list[Document] = []
        for parent in parents_docs:
            parent_id = parent.metadata.get("parent_id")
            if parent_id not in child_metadata_by_parent:
                continue

            matches = child_metadata_by_parent[parent_id]
            parent.metadata["matched_child_ids"] = [
                match.get("child_id") for match in matches if match.get("child_id")
            ]
            parent.metadata["rerank_scores"] = [
                match.get("rerank_score")
                for match in matches
                if match.get("rerank_score") is not None
            ]
            # Keep singular fields for existing context builders and clients
            if parent.metadata["matched_child_ids"]:
                parent.metadata["matched_child_id"] = parent.metadata["matched_child_ids"][0]
            if parent.metadata["rerank_scores"]:
                parent.metadata["rerank_score"] = parent.metadata["rerank_scores"][0]

            parents.append(parent)

        # If parent fetch returned nothing, gracefully fall back to children
        return parents if parents else ranked_children


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()
