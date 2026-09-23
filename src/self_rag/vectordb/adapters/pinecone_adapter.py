"""Pinecone vector database adapter using LangChain integration."""

from functools import lru_cache
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from self_rag.clients.llm import get_embedding_model
from self_rag.core.config import get_settings
from self_rag.vectordb.base import AbstractVectorDB, VectorSearchResult


class PineconeAdapter(AbstractVectorDB):
    """
    Pinecone vector database adapter using LangChain's PineconeVectorStore.

    Features:
    - Dense vector search (semantic similarity)
    - Supports namespace isolation
    - Sparse search falls back to semantic with FlashRank reranking
    """

    def __init__(self, api_key: str, index_name: str, namespace: str = ""):
        """Initialize Pinecone adapter with API key and index name."""
        self.api_key = api_key
        self.index_name = index_name
        self.namespace = namespace
        self.embedding_model = get_embedding_model()
        self._store: Optional[PineconeVectorStore] = None

    def _get_vectorstore(self) -> PineconeVectorStore:
        """Get or create Pinecone VectorStore instance."""
        if self._store is None:
            self._store = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embedding_model,
                namespace=self.namespace,
            )
        return self._store

    def search_dense(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Dense semantic search via Pinecone."""
        try:
            from pinecone import Pinecone

            pc = Pinecone(api_key=self.api_key)
            index = pc.Index(self.index_name)

            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=self.namespace,
                include_metadata=True,
            )

            docs = []
            for match in results.get("matches", []):
                if not match.get("metadata"):
                    continue
                doc = Document(
                    page_content=match["metadata"].get("page_content", ""),
                    metadata=match["metadata"].get("metadata", {}),
                )
                docs.append(
                    VectorSearchResult(
                        document=doc,
                        score=match.get("score", 0.0),
                        source="dense",
                    )
                )

            return docs
        except Exception as e:
            print(f"Error in dense search: {e}")
            return []

    def search_sparse(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """
        Sparse search - Pinecone doesn't support BM25.
        Returns empty list; hybrid behavior via semantic + FlashRank reranking.
        """
        return []

    def retrieve_by_ids(
        self,
        doc_ids: List[str],
        collection_name: Optional[str] = None,
    ) -> List[Document]:
        """Fetch documents by ID (metadata stored in Pinecone)."""
        try:
            from pinecone import Pinecone

            pc = Pinecone(api_key=self.api_key)
            index = pc.Index(self.index_name)

            response = index.fetch(
                ids=doc_ids,
                namespace=self.namespace,
            )

            docs = []
            for vector_id, vector_data in response.get("vectors", {}).items():
                if not vector_data.get("metadata"):
                    continue
                doc = Document(
                    page_content=vector_data["metadata"].get("page_content", ""),
                    metadata=vector_data["metadata"].get("metadata", {}),
                )
                docs.append(doc)

            return docs
        except Exception as e:
            print(f"Error retrieving by IDs: {e}")
            return []

    def upsert_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        doc_ids: List[str],
        collection_name: Optional[str] = None,
    ) -> None:
        """Insert/update documents with embeddings via LangChain."""
        try:
            store = self._get_vectorstore()
            store.add_documents(documents, ids=doc_ids)
        except Exception as e:
            print(f"Error upserting documents: {e}")

    def delete_documents(
        self,
        doc_ids: List[str],
        collection_name: Optional[str] = None,
    ) -> None:
        """Delete documents by ID."""
        try:
            from pinecone import Pinecone

            pc = Pinecone(api_key=self.api_key)
            index = pc.Index(self.index_name)
            index.delete(ids=doc_ids, namespace=self.namespace)
        except Exception as e:
            print(f"Error deleting documents: {e}")

    def ensure_collection(
        self,
        name: str,
        config: Dict[str, Any],
    ) -> bool:
        """Create index if needed (Pinecone indexes are pre-created)."""
        try:
            from pinecone import Pinecone

            pc = Pinecone(api_key=self.api_key)
            indexes = [idx.name for idx in pc.list_indexes()]

            if self.index_name not in indexes:
                print(f"Index {self.index_name} not found. Please create it in Pinecone console.")
                return False

            return True
        except Exception as e:
            print(f"Error checking index: {e}")
            return False

    def reset_collection(self, name: str) -> None:
        """Clear all documents from namespace."""
        try:
            from pinecone import Pinecone

            pc = Pinecone(api_key=self.api_key)
            index = pc.Index(self.index_name)
            index.delete(delete_all=True, namespace=self.namespace)
        except Exception as e:
            print(f"Error resetting collection: {e}")

    def collection_exists(self, name: str) -> bool:
        """Check if index exists."""
        try:
            from pinecone import Pinecone

            pc = Pinecone(api_key=self.api_key)
            indexes = [idx.name for idx in pc.list_indexes()]
            return self.index_name in indexes
        except Exception:
            return False

    def health_check(self) -> bool:
        """Verify Pinecone connectivity."""
        try:
            from pinecone import Pinecone

            pc = Pinecone(api_key=self.api_key)
            pc.list_indexes()
            return True
        except Exception:
            return False


@lru_cache(maxsize=1)
def get_pinecone_adapter() -> PineconeAdapter:
    """Get Pinecone adapter singleton."""
    settings = get_settings()
    return PineconeAdapter(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        namespace=settings.pinecone_namespace,
    )
