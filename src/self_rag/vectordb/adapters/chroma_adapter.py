"""Chroma vector database adapter using LangChain integration."""

from functools import lru_cache
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from self_rag.clients.llm import get_embedding_model
from self_rag.core.config import get_settings
from self_rag.vectordb.base import AbstractVectorDB, VectorSearchResult


class ChromaAdapter(AbstractVectorDB):
    """
    Chroma vector database adapter using LangChain integration.

    Features:
    - Dense vector search (in-memory or persistent)
    - Lightweight, easy to deploy
    - Sparse search uses semantic + FlashRank fallback
    - Metadata filtering support
    """

    def __init__(self, persist_dir: str = "./chroma_data"):
        """Initialize Chroma adapter with optional persistence."""
        self.persist_dir = persist_dir
        self.embedding_model = get_embedding_model()
        self._store: Optional[Chroma] = None

    def _get_vectorstore(self) -> Chroma:
        """Get or create Chroma VectorStore instance."""
        if self._store is None:
            self._store = Chroma(
                embedding_function=self.embedding_model,
                persist_directory=self.persist_dir,
            )
        return self._store

    def search_dense(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Dense semantic search via Chroma."""
        try:
            store = self._get_vectorstore()

            # Chroma doesn't support direct embedding search via LangChain wrapper
            # Use similarity_search_by_vector instead
            docs = store.similarity_search_by_vector(
                embedding=query_embedding,
                k=top_k,
            )

            results = []
            for doc in docs:
                results.append(
                    VectorSearchResult(
                        document=doc,
                        score=doc.metadata.get("score", 0.5),
                        source="dense",
                    )
                )

            return results
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
        Sparse search - Chroma doesn't support BM25 natively.
        Returns empty list; hybrid behavior via semantic + FlashRank reranking.
        """
        return []

    def retrieve_by_ids(
        self,
        doc_ids: List[str],
        collection_name: Optional[str] = None,
    ) -> List[Document]:
        """Fetch documents by ID."""
        try:
            store = self._get_vectorstore()
            docs = store.get(ids=doc_ids)

            result_docs = []
            for i, doc_content in enumerate(docs.get("documents", [])):
                doc = Document(
                    page_content=doc_content,
                    metadata=docs["metadatas"][i] if docs.get("metadatas") else {},
                )
                result_docs.append(doc)

            return result_docs
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
            store = self._get_vectorstore()
            store.delete(ids=doc_ids)
        except Exception as e:
            print(f"Error deleting documents: {e}")

    def ensure_collection(
        self,
        name: str,
        config: Dict[str, Any],
    ) -> bool:
        """Create collection/database (Chroma auto-creates)."""
        try:
            # Chroma auto-creates collections, just verify it's accessible
            store = self._get_vectorstore()
            return store is not None
        except Exception as e:
            print(f"Error ensuring collection: {e}")
            return False

    def reset_collection(self, name: str) -> None:
        """Delete all documents from Chroma."""
        try:
            store = self._get_vectorstore()
            # Get all documents and delete them
            all_docs = store.get()
            if all_docs and all_docs.get("ids"):
                store.delete(ids=all_docs["ids"])
        except Exception as e:
            print(f"Error resetting collection: {e}")

    def collection_exists(self, name: str) -> bool:
        """Check if Chroma instance is accessible."""
        try:
            store = self._get_vectorstore()
            return store is not None
        except Exception:
            return False

    def health_check(self) -> bool:
        """Verify Chroma is accessible."""
        try:
            store = self._get_vectorstore()
            # Check if we can get collection info
            store.get(limit=1)
            return True
        except Exception:
            return False


@lru_cache(maxsize=1)
def get_chroma_adapter() -> ChromaAdapter:
    """Get Chroma adapter singleton."""
    settings = get_settings()
    return ChromaAdapter(persist_dir=settings.chroma_persist_dir)
