import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from self_rag.core.config import get_settings


def split_documents(
    documents: list[Document],
) -> tuple[list[Document], list[Document]]:
    """
    Splits documents into a hierarchy of Parent and Child chunks.

    Returns:
        tuple[list[Document], list[Document]]: (parent_chunks, child_chunks)
    """
    settings = get_settings()
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.parent_chunk_size,
        chunk_overlap=settings.parent_chunk_overlap,
        add_start_index=True,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )

    parent_chunks: list[Document] = []
    child_chunks: list[Document] = []

    raw_parents = parent_splitter.split_documents(documents)

    for parent_idx, parent in enumerate(raw_parents):
        source = parent.metadata.get("source", "unknown-source")
        page = parent.metadata.get("page", "unknown-page")
        parent_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{source}:{page}:{parent_idx}:{parent.page_content}",
            )
        )
        parent.metadata["parent_id"] = parent_id
        parent.metadata["is_parent"] = True
        parent_chunks.append(parent)

        children = child_splitter.split_documents([parent])
        for child_idx, child in enumerate(children):
            # Qdrant point IDs must be integers or UUIDs.  A UUID5 keeps IDs
            # stable across clean re-ingestions while preserving that contract.
            child.metadata["child_id"] = str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"{parent_id}:child:{child_idx}")
            )
            child.metadata["parent_id"] = parent_id
            child.metadata["is_parent"] = False
            child_chunks.append(child)

    return parent_chunks, child_chunks
