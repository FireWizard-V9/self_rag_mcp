from self_rag.models.graph_state import GraphState


def context_builder(state: GraphState) -> dict:
    """Builds a structured, XML-encapsulated context block optimized for LLM attention and citation."""

    docs = (
        state.get("relevant_documents")
        or state.get("retrieved_documents")
        or state.get("documents", [])
    )

    if not docs:
        return {"context": "No relevant documents found."}

    formatted_chunks = []

    for idx, doc in enumerate(docs, start=1):
        # Extract metadata dynamically
        source = doc.metadata.get("source") or doc.metadata.get(
            "file_name", "Internal Document"
        )
        score = doc.metadata.get("rerank_score")

        # Build optional metadata string
        meta_str = f"Source: {source}"
        if score is not None:
            meta_str += f" | Relevance Score: {score:.4f}"

        # Format individual chunk with explicit XML tags and headers
        chunk_block = (
            f'<document index="{idx}">\n'
            f"  <metadata>{meta_str}</metadata>\n"
            f"  <content>\n{doc.page_content.strip()}\n  </content>\n"
            f"</document>"
        )
        formatted_chunks.append(chunk_block)

    # Wrap the entire list inside a root <context> tag
    full_context = "<context>\n" + "\n\n".join(formatted_chunks) + "\n</context>"

    return {"context": full_context}
