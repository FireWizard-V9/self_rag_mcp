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


def retrieve_documents(query: str, top_k: int = 10) -> dict[str, Any]:
    """Retrieve raw document chunks with metadata."""
    if not query.strip():
        return {"status": "error", "message": "Query cannot be empty.", "documents": []}

    try:
        retriever = get_retriever()
        bounded_top_k = max(1, min(top_k, 50))

        docs = retriever.invoke(query=query.strip(), top_k=bounded_top_k)

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
    try:
        # Check if retriever can be initialized
        _ = get_retriever()
        retrieval_status = "operational"
    except Exception as e:  # noqa: BLE001 - health checks must report any initialization failure.
        retrieval_status = f"unhealthy: {e!s}"

    return {
        "status": "healthy" if retrieval_status == "operational" else "degraded",
        "components": {
            "retrieval": "Hybrid Search",
            "reranker": "CrossEncoder",
            "retriever_status": retrieval_status,
        },
    }
