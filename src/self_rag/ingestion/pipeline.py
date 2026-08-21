from self_rag.ingestion.chunker import split_documents
from self_rag.ingestion.indexer import ensure_collections_exist, reset_collections
from self_rag.ingestion.loaders import load_documents
from self_rag.retrieval.vector_store import get_parent_vector_store, get_vector_store


def run_ingestion(*, reset: bool = False) -> None:
    #  Load raw PDF documents (page-level)
    documents = load_documents()

    # Split into parent and child chunk hierarchies
    parent_chunks, child_chunks = split_documents(documents)

    if reset:
        reset_collections()
    ensure_collections_exist()

    parent_store = get_parent_vector_store()
    child_store = get_vector_store()
    parent_store.add_documents(
        parent_chunks,
        ids=[parent.metadata["parent_id"] for parent in parent_chunks],
    )
    child_store.add_documents(
        child_chunks,
        ids=[child.metadata["child_id"] for child in child_chunks],
    )

    print(
        f"Ingestion complete: "
        f"{len(documents)} raw documents -> "
        f"{len(parent_chunks)} parent chunks -> "
        f"{len(child_chunks)} child chunks indexed and "
        f"{len(parent_chunks)} parent chunks stored in Qdrant."
    )
