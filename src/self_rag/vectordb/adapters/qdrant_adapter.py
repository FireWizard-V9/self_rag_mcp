"""Qdrant vector database adapter using LangChain integration."""

from functools import lru_cache
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, SparseIndexParams, SparseVectorParams, VectorParams

from self_rag.clients.llm import get_embedding_model
from self_rag.core.config import get_settings
from self_rag.vectordb.base import AbstractVectorDB, VectorSearchResult


class QdrantAdapter(AbstractVectorDB):
    """
    Qdrant vector database adapter using LangChain's QdrantVectorStore.

    Supports:
    - Dense vectors (semantic similarity)
    - Sparse vectors (BM25 keyword search)
    - Native hybrid search with RRF fusion
    - Parent-child document expansion
    """

    def __init__(self, url: str, api_key: Optional[str] = None):
        """Initialize Qdrant adapter with client and embedding model."""
        self.url = url
        self.api_key = api_key
        self.client = QdrantClient(url=url, api_key=api_key)
        self.embedding_model = get_embedding_model()
        self.sparse_embedding = FastEmbedSparse(model_name="Qdrant/bm25")
        self._stores: Dict[str, QdrantVectorStore] = {}

    def _get_or_create_store(
        self,
        collection_name: str,
        has_sparse: bool = True,
    ) -> QdrantVectorStore:
        """Get cached vectorstore or create new one."""
        key = f"{collection_name}_{has_sparse}"
        if key in self._stores:
            return self._stores[key]

        if has_sparse:
            # Child documents with hybrid search
            store = QdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
                embedding=self.embedding_model,
                sparse_embedding=self.sparse_embedding,
                retrieval_mode=RetrievalMode.HYBRID,
            )
        else:
            # Parent documents with dense search only
            store = QdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
                embedding=self.embedding_model,
                retrieval_mode=RetrievalMode.DENSE,
            )

        self._stores[key] = store
        return store

    def search_dense(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Dense semantic search via Qdrant."""
        settings = get_settings()
        collection = collection_name or settings.qdrant_collection

        try:
            results = self.client.search(
                collection_name=collection,
                query_vector=query_embedding,
                limit=top_k,
            )

            docs = []
            for scored_point in results:
                if not scored_point.payload:
                    continue
                doc = Document(
                    page_content=scored_point.payload.get("page_content", ""),
                    metadata=scored_point.payload.get("metadata", {}),
                )
                docs.append(
                    VectorSearchResult(
                        document=doc,
                        score=scored_point.score,
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
        """Sparse BM25 search via Qdrant (FastEmbedSparse)."""
        settings = get_settings()
        collection = collection_name or settings.qdrant_collection

        try:
            store = self._get_or_create_store(collection, has_sparse=True)
            docs = store.similarity_search(query, k=top_k)

            results = []
            for doc in docs:
                results.append(
                    VectorSearchResult(
                        document=doc,
                        score=doc.metadata.get("score", 0.0),
                        source="sparse",
                    )
                )

            return results
        except Exception as e:
            print(f"Error in sparse search: {e}")
            return []

    def retrieve_by_ids(
        self,
        doc_ids: List[str],
        collection_name: Optional[str] = None,
    ) -> List[Document]:
        """Fetch documents by ID (used for parent expansion)."""
        settings = get_settings()
        collection = collection_name or settings.qdrant_parent_collection

        try:
            points = self.client.retrieve(
                collection_name=collection,
                ids=doc_ids,
                with_payload=True,
                with_vectors=False,
            )

            docs = []
            for point in points:
                if not point.payload:
                    continue
                doc = Document(
                    page_content=point.payload.get("page_content", ""),
                    metadata=point.payload.get("metadata", {}),
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
        """Insert/update documents with embeddings via LangChain VectorStore."""
        settings = get_settings()
        collection = collection_name or settings.qdrant_collection

        try:
            # Determine if sparse vectors needed
            has_sparse = collection == settings.qdrant_collection
            store = self._get_or_create_store(collection, has_sparse=has_sparse)
            store.add_documents(documents, ids=doc_ids)
        except Exception as e:
            print(f"Error upserting documents: {e}")

    def delete_documents(
        self,
        doc_ids: List[str],
        collection_name: Optional[str] = None,
    ) -> None:
        """Delete documents."""
        settings = get_settings()
        collection = collection_name or settings.qdrant_collection

        try:
            self.client.delete(
                collection_name=collection,
                points_selector=doc_ids,
            )
        except Exception as e:
            print(f"Error deleting documents: {e}")

    def ensure_collection(
        self,
        name: str,
        config: Dict[str, Any],
    ) -> bool:
        """Create Qdrant collection with proper config."""
        try:
            if self.client.collection_exists(name):
                return True

            embedding_dim = config.get("embedding_dim", 1536)
            has_sparse = config.get("has_sparse", False)

            sparse_vectors_config = None
            if has_sparse:
                sparse_vectors_config = {
                    "langchain-sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                }

            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE,
                ),
                sparse_vectors_config=sparse_vectors_config,
            )

            return True
        except Exception as e:
            print(f"Error ensuring collection: {e}")
            return False

    def reset_collection(self, name: str) -> None:
        """Delete and recreate collection."""
        try:
            if self.client.collection_exists(name):
                self.client.delete_collection(name)
            # Clear cached stores
            self._stores.clear()
        except Exception as e:
            print(f"Error resetting collection: {e}")

    def collection_exists(self, name: str) -> bool:
        """Check if collection exists."""
        try:
            return self.client.collection_exists(name)
        except Exception:
            return False

    def health_check(self) -> bool:
        """Verify Qdrant is running."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False


@lru_cache(maxsize=1)
def get_qdrant_adapter() -> QdrantAdapter:
    """Cached Qdrant adapter singleton."""
    settings = get_settings()
    return QdrantAdapter(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
