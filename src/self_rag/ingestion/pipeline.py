from self_rag.clients.llm import get_embedding_model
from self_rag.core.config import get_settings
from self_rag.ingestion.chunker import split_documents
from self_rag.ingestion.indexer import ingest_documents
from self_rag.ingestion.loaders import load_documents


def run_ingestion(*, reset: bool = False, use_hierarchy: bool = True) -> None:
    """
    DB-agnostic ingestion pipeline with flexible chunking.

    Args:
        reset: If True, clear collections before ingesting
        use_hierarchy: If True, create parent-child chunks. If False, index documents flat.
    """

    # Load raw PDF documents (page-level)
    documents = load_documents()

    # Split documents (hierarchical or flat)
    parent_chunks, child_chunks = split_documents(documents, use_hierarchy=use_hierarchy)

    # Ingest using VectorDB adapter (handles reset, collection setup, etc.)
    ingest_documents(
        parent_chunks=parent_chunks,
        child_chunks=child_chunks,
        reset=reset,
    )

    if use_hierarchy:
        print(
            f"Ingestion complete (HIERARCHICAL): "
            f"{len(documents)} raw documents → "
            f"{len(parent_chunks)} parent chunks → "
            f"{len(child_chunks)} child chunks indexed in "
            f"{get_settings().vectordb_provider.upper()} database."
        )
    else:
        print(
            f"Ingestion complete (FLAT): "
            f"{len(documents)} raw documents → "
            f"{len(parent_chunks)} documents indexed in "
            f"{get_settings().vectordb_provider.upper()} database."
        )
