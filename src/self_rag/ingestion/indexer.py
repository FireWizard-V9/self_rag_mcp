from typing import List

from langchain_core.documents import Document

from self_rag.clients.llm import get_embedding_model
from self_rag.core.config import get_settings
from self_rag.vectordb.factory import get_vectordb


def ingest_documents(
    parent_chunks: List[Document],
    child_chunks: List[Document],
    reset: bool = False,
) -> None:
    """
    DB-agnostic document ingestion supporting both hierarchical and flat structures.

    - If child_chunks is empty: flat ingestion (index parent_chunks as-is, no hierarchy)
    - If child_chunks exists: hierarchical ingestion (parents + children in two collections)
    """

    vectordb = get_vectordb()
    embedding_model = get_embedding_model()
    settings = get_settings()

    # Determine ingestion mode
    use_hierarchy = len(child_chunks) > 0

    # Reset if requested
    if reset:
        vectordb.reset_collection(settings.qdrant_collection)
        if use_hierarchy:
            vectordb.reset_collection(settings.qdrant_parent_collection)

    # Ensure main collection exists with proper config
    vectordb.ensure_collection(
        name=settings.qdrant_collection,
        config={
            "embedding_dim": settings.embedding_dim,
            "has_sparse": True,  # Support hybrid search
        },
    )

    if use_hierarchy:
        # Hierarchical ingestion: both collections
        print("Ingestion mode: HIERARCHICAL (parent-child chunks)")

        vectordb.ensure_collection(
            name=settings.qdrant_parent_collection,
            config={
                "embedding_dim": settings.embedding_dim,
                "has_sparse": False,  # Parents don't need sparse
            },
        )

        # Embed and ingest child chunks
        print(f"Embedding {len(child_chunks)} child chunks...")
        child_embeddings = embedding_model.embed_documents(
            [doc.page_content for doc in child_chunks]
        )
        child_ids = [doc.metadata["child_id"] for doc in child_chunks]

        print(f"Ingesting {len(child_chunks)} child chunks into {settings.vectordb_provider}...")
        vectordb.upsert_documents(
            documents=child_chunks,
            embeddings=child_embeddings,
            doc_ids=child_ids,
            collection_name=settings.qdrant_collection,
        )

        # Embed and ingest parent chunks
        print(f"Embedding {len(parent_chunks)} parent chunks...")
        parent_embeddings = embedding_model.embed_documents(
            [doc.page_content for doc in parent_chunks]
        )
        parent_ids = [doc.metadata["parent_id"] for doc in parent_chunks]

        print(f"Ingesting {len(parent_chunks)} parent chunks...")
        vectordb.upsert_documents(
            documents=parent_chunks,
            embeddings=parent_embeddings,
            doc_ids=parent_ids,
            collection_name=settings.qdrant_parent_collection,
        )
    else:
        # Flat ingestion: single collection, no hierarchy
        print("Ingestion mode: FLAT (no parent-child hierarchy)")

        # Embed and ingest documents as-is
        print(f"Embedding {len(parent_chunks)} documents...")
        embeddings = embedding_model.embed_documents(
            [doc.page_content for doc in parent_chunks]
        )
        doc_ids = [doc.metadata.get("doc_id", doc.metadata.get("parent_id")) for doc in parent_chunks]

        print(f"Ingesting {len(parent_chunks)} documents into {settings.vectordb_provider}...")
        vectordb.upsert_documents(
            documents=parent_chunks,
            embeddings=embeddings,
            doc_ids=doc_ids,
            collection_name=settings.qdrant_collection,
        )

    print("Ingestion complete!")
