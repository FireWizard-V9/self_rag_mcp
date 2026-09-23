import logging
from typing import Any

from self_rag.retrieval.retriever import get_retriever
from self_rag.services.rag_service import RAGService

logger = logging.getLogger("self_rag_mcp")
_service: RAGService | None = None


def get_rag_service() -> RAGService:
    global _service
    if _service is None:
        _service = RAGService()
    return _service


def answer_question(question: str, max_retries: int = 2) -> dict[str, Any]:
    """Execute the full Self-RAG state graph"""

    if not question.strip():
        return {
            "status": "error",
            "message": "Question cannot be empty.",
            "answer": None,
        }

    try:
        service = get_rag_service()
        result = service.answer(
            question=question.strip(), max_retries=max(1, min(max_retries, 5))
        )
        return {
            "status": "success",
            "result": result,
        }
    except Exception as e:
        logger.exception(f"Error executing Self-RAG graph for question: {question}")
        return {
            "status": "error",
            "message": f"Failed to generate answer: {e!s}",
            "answer": None,
        }


def retrieve_documents(query: str, top_k: int = 10, expand_parents: bool = True) -> dict[str, Any]:
    """
    Retrieve documents from the vector database.

    Args:
        query: Search query
        top_k: Number of documents to return (clamped to 1-50)
        expand_parents: If True, expand to parent documents (if configured).
                       Set to False for plug-and-play with external indexes.
    """
    if not query.strip():
        return {"status": "error", "message": "Query cannot be empty.", "documents": []}

    try:
        retriever = get_retriever()
        bounded_top_k = max(1, min(top_k, 50))

        docs = retriever.invoke(query=query.strip(), top_k=bounded_top_k, expand_parents=expand_parents)

        formatted_docs = [
            {
                "content": getattr(doc, "page_content", str(doc)),
                "metadata": getattr(doc, "metadata", {}),
            }
            for doc in docs
        ]

        return {
            "status": "success",
            "count": len(formatted_docs),
            "documents": formatted_docs,
        }
    except Exception as e:
        logger.exception(f"Retrieval failed for query: {query}")
        return {
            "status": "error",
            "message": f"Document retrieval failed: {e!s}",
            "documents": [],
        }


def health() -> dict[str, Any]:
    """Check service dependencies and return operational status."""
    from self_rag.core.config import get_settings
    from self_rag.vectordb.factory import get_vectordb

    settings = get_settings()
    vectordb_status = "unknown"
    retrieval_status = "unknown"

    try:
        # Check VectorDB connectivity
        vectordb = get_vectordb()
        if vectordb.health_check():
            vectordb_status = "operational"
        else:
            vectordb_status = "unhealthy"
    except Exception as e:  # noqa: BLE001
        vectordb_status = f"unavailable: {e!s}"

    try:
        # Check if retriever can be initialized
        _ = get_retriever()
        retrieval_status = "operational"
    except Exception as e:  # noqa: BLE001
        retrieval_status = f"unhealthy: {e!s}"

    # Determine overall status: retriever failure = unhealthy, vectordb failure = degraded
    overall_status = "healthy"
    if retrieval_status != "operational":
        overall_status = "unhealthy"
    elif vectordb_status != "operational":
        overall_status = "degraded"

    return {
        "status": overall_status,
        "components": {
            "vectordb_provider": settings.vectordb_provider,
            "vectordb_status": vectordb_status,
            "search_strategy": settings.vectordb_hybrid_strategy,
            "retrieval": "Hybrid Search (Dense + Sparse + MMR + FlashRank)",
            "reranker": "CrossEncoder (FlashRank)",
            "retriever_status": retrieval_status,
        },
    }
