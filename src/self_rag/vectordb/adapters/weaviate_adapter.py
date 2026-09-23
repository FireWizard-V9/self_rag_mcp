"""Weaviate vector database adapter using LangChain integration."""

from functools import lru_cache
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Weaviate

from self_rag.clients.llm import get_embedding_model
from self_rag.core.config import get_settings
from self_rag.vectordb.base import AbstractVectorDB, VectorSearchResult


class WeaviateAdapter(AbstractVectorDB):
    """
    Weaviate vector database adapter using LangChain integration.

    Features:
    - Dense vector search (semantic similarity)
    - Native BM25 keyword search
    - Native hybrid search support
    - GraphQL-based queries
    """

    def __init__(self, url: str, class_name: str = "SelfRAGDocument"):
        """Initialize Weaviate adapter with URL and class name."""
        self.url = url
        self.class_name = class_name
        self.embedding_model = get_embedding_model()
        self._store: Optional[Weaviate] = None

    def _get_vectorstore(self) -> Weaviate:
        """Get or create Weaviate VectorStore instance."""
        if self._store is None:
            import weaviate

            client = weaviate.connect_to_local(url=self.url)
            self._store = Weaviate(
                client=client,
                index_name=self.class_name,
                text_key="page_content",
                embedding=self.embedding_model,
                by_text=False,
            )
        return self._store

    def search_dense(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Dense semantic search via nearVector."""
        try:
            import weaviate

            client = weaviate.connect_to_local(url=self.url)

            response = (
                client.graphql_raw_query(
                    gql_query=f"""
                    {{
                        Get {{
                            {self.class_name}(
                                nearVector: {{vector: {query_embedding}}}
                                limit: {top_k}
                            ) {{
                                page_content
                                _additional {{
                                    distance
                                    metadata
                                }}
                            }}
                        }}
                    }}
                    """
                )
            )

            docs = []
            for item in response.get("data", {}).get("Get", {}).get(self.class_name, []):
                doc = Document(
                    page_content=item.get("page_content", ""),
                    metadata=item.get("_additional", {}).get("metadata", {}),
                )
                docs.append(
                    VectorSearchResult(
                        document=doc,
                        score=1.0 - item.get("_additional", {}).get("distance", 0.5),
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
        """BM25 sparse search via Weaviate."""
        try:
            import weaviate

            client = weaviate.connect_to_local(url=self.url)

            response = (
                client.graphql_raw_query(
                    gql_query=f"""
                    {{
                        Get {{
                            {self.class_name}(
                                bm25: {{query: "{query}"}}
                                limit: {top_k}
                            ) {{
                                page_content
                                _additional {{
                                    score
                                    metadata
                                }}
                            }}
                        }}
                    }}
                    """
                )
            )

            docs = []
            for item in response.get("data", {}).get("Get", {}).get(self.class_name, []):
                doc = Document(
                    page_content=item.get("page_content", ""),
                    metadata=item.get("_additional", {}).get("metadata", {}),
                )
                docs.append(
                    VectorSearchResult(
                        document=doc,
                        score=item.get("_additional", {}).get("score", 0.0),
                        source="sparse",
                    )
                )

            return docs
        except Exception as e:
            print(f"Error in sparse search: {e}")
            return []

    def retrieve_by_ids(
        self,
        doc_ids: List[str],
        collection_name: Optional[str] = None,
    ) -> List[Document]:
        """Fetch documents by ID."""
        try:
            import weaviate

            client = weaviate.connect_to_local(url=self.url)

            docs = []
            for doc_id in doc_ids:
                obj = client.data_object.get_by_id(
                    uuid=doc_id,
                    class_name=self.class_name,
                    with_vector=False,
                )

                if obj and "properties" in obj:
                    doc = Document(
                        page_content=obj["properties"].get("page_content", ""),
                        metadata=obj["properties"].get("metadata", {}),
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
            import weaviate

            client = weaviate.connect_to_local(url=self.url)
            for doc_id in doc_ids:
                client.data_object.delete(uuid=doc_id, class_name=self.class_name)
        except Exception as e:
            print(f"Error deleting documents: {e}")

    def ensure_collection(
        self,
        name: str,
        config: Dict[str, Any],
    ) -> bool:
        """Create class if needed."""
        try:
            import weaviate
            from weaviate.classes.config import Property, DataType

            client = weaviate.connect_to_local(url=self.url)

            if client.collections.exists(self.class_name):
                return True

            # Create class with properties
            client.collections.create(
                name=self.class_name,
                properties=[
                    Property(name="page_content", data_type=DataType.TEXT),
                    Property(name="metadata", data_type=DataType.OBJECT),
                ],
            )

            return True
        except Exception as e:
            print(f"Error ensuring collection: {e}")
            return False

    def reset_collection(self, name: str) -> None:
        """Delete all documents in class."""
        try:
            import weaviate

            client = weaviate.connect_to_local(url=self.url)
            client.collections.delete(self.class_name)
        except Exception as e:
            print(f"Error resetting collection: {e}")

    def collection_exists(self, name: str) -> bool:
        """Check if class exists."""
        try:
            import weaviate

            client = weaviate.connect_to_local(url=self.url)
            return client.collections.exists(self.class_name)
        except Exception:
            return False

    def health_check(self) -> bool:
        """Verify Weaviate connectivity."""
        try:
            import weaviate

            client = weaviate.connect_to_local(url=self.url)
            return client.is_ready()
        except Exception:
            return False


@lru_cache(maxsize=1)
def get_weaviate_adapter() -> WeaviateAdapter:
    """Get Weaviate adapter singleton."""
    settings = get_settings()
    return WeaviateAdapter(
        url=settings.weaviate_url,
        class_name=settings.weaviate_class_name,
    )
