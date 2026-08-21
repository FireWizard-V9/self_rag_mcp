from functools import lru_cache

from langchain_core.documents import Document

from self_rag.clients.qdrant import get_qdrant_client
from self_rag.core.config import get_settings
from self_rag.retrieval.mmr import apply_mmr
from self_rag.retrieval.reranker import rerank_documents
from self_rag.retrieval.vector_store import get_vector_store


class HybridRetriever:
    """
    FINAL  PIPELINE:

    1. Qdrant Hybrid (Dense + BM25 + RRF)
    2. MMR (diversity)
    3. Cross-Encoder Reranker (final)
    """

    def __init__(self):
        self.store = get_vector_store()
        self.settings = get_settings()

    def invoke(self, query: str, top_k: int | None = None) -> list[Document]:

        candidate_count = max(
            self.settings.retrieval_k_initial,
            self.settings.retrieval_k_mmr,
            top_k or self.settings.retrieval_k_rerank,
        )
        docs = self.store.similarity_search(query, k=candidate_count)

        docs = apply_mmr(
            query,
            docs,
            k=self.settings.retrieval_k_mmr,
            lambda_param=self.settings.retrieval_lambda,
        )

        docs = rerank_documents(
            query,
            docs,
            top_k=top_k or self.settings.retrieval_k_rerank,
        )

        return self._expand_to_parent_documents(docs)

    def _expand_to_parent_documents(
        self, ranked_children: list[Document]
    ) -> list[Document]:
        """Fetch each unique full parent chunk for the ranked child results."""
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

        if not parent_ids:
            return ranked_children

        points = get_qdrant_client().retrieve(
            collection_name=self.settings.qdrant_parent_collection,
            ids=parent_ids,
            with_payload=True,
            with_vectors=False,
        )
        points_by_id = {str(point.id): point for point in points}
        parents: list[Document] = []
        for parent_id in parent_ids:
            point = points_by_id.get(parent_id)
            if point is None or not point.payload:
                continue
            metadata = dict(point.payload.get("metadata", {}))
            matches = child_metadata_by_parent[parent_id]
            metadata["matched_child_ids"] = [
                match.get("child_id") for match in matches if match.get("child_id")
            ]
            metadata["rerank_scores"] = [
                match.get("rerank_score")
                for match in matches
                if match.get("rerank_score") is not None
            ]
            # Keep singular fields for existing context builders and clients.
            metadata["matched_child_id"] = metadata["matched_child_ids"][0]
            metadata["rerank_score"] = metadata["rerank_scores"][0]
            parents.append(
                Document(
                    page_content=point.payload["page_content"],
                    metadata=metadata,
                )
            )

        return parents or ranked_children


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()
