"""Tests for MCP tool functions (answer_question, retrieve_documents, health)."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

import self_rag.mcp.tools as tools_module
from self_rag.mcp.tools import answer_question, health, retrieve_documents


# --- answer_question ---

def test_answer_question_empty_returns_error():
    result = answer_question("   ")
    assert result["status"] == "error"
    assert "empty" in result["message"].lower()


def test_answer_question_success():
    mock_service = MagicMock()
    mock_service.answer.return_value = {
        "question": "What is RAG?",
        "answer": "Retrieval-Augmented Generation.",
        "support_grade": "fully_supported",
        "usefulness_grade": "useful",
        "sources": [],
    }
    with patch.object(tools_module, "get_rag_service", return_value=mock_service):
        result = answer_question("What is RAG?", max_retries=2)

    assert result["status"] == "success"
    assert "answer" in result["result"]
    mock_service.answer.assert_called_once_with(question="What is RAG?", max_retries=2)


def test_answer_question_clamps_max_retries():
    mock_service = MagicMock()
    mock_service.answer.return_value = {"answer": "", "support_grade": "fully_supported", "usefulness_grade": "useful", "sources": []}
    with patch.object(tools_module, "get_rag_service", return_value=mock_service):
        answer_question("q", max_retries=0)
    # max_retries=0 should be clamped to 1
    mock_service.answer.assert_called_once_with(question="q", max_retries=1)


def test_answer_question_service_exception_returns_error():
    mock_service = MagicMock()
    mock_service.answer.side_effect = RuntimeError("LLM unavailable")
    with patch.object(tools_module, "get_rag_service", return_value=mock_service):
        result = answer_question("What is RAG?")
    assert result["status"] == "error"
    assert "LLM unavailable" in result["message"]


# --- retrieve_documents ---

def test_retrieve_documents_empty_query_returns_error():
    result = retrieve_documents("  ")
    assert result["status"] == "error"
    assert result["documents"] == []


def test_retrieve_documents_success():
    doc = Document(page_content="RAG combines retrieval with generation.", metadata={"source": "doc1"})
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [doc]

    with patch("self_rag.mcp.tools.get_retriever", return_value=mock_retriever):
        result = retrieve_documents("What is RAG?", top_k=5)

    assert result["status"] == "success"
    assert result["count"] == 1
    assert result["documents"][0]["content"] == "RAG combines retrieval with generation."
    assert result["documents"][0]["metadata"] == {"source": "doc1"}
    mock_retriever.invoke.assert_called_once_with(query="What is RAG?", top_k=5, expand_parents=True)


def test_retrieve_documents_clamps_top_k():
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    with patch("self_rag.mcp.tools.get_retriever", return_value=mock_retriever):
        retrieve_documents("query", top_k=999)
    mock_retriever.invoke.assert_called_once_with(query="query", top_k=50, expand_parents=True)


def test_retrieve_documents_exception_returns_error():
    mock_retriever = MagicMock()
    mock_retriever.invoke.side_effect = ConnectionError("Qdrant down")
    with patch("self_rag.mcp.tools.get_retriever", return_value=mock_retriever):
        result = retrieve_documents("query")
    assert result["status"] == "error"
    assert "Qdrant down" in result["message"]


# --- health ---

def test_health_operational():
    mock_vectordb = MagicMock()
    mock_vectordb.health_check.return_value = True
    with patch("self_rag.mcp.tools.get_retriever", return_value=MagicMock()), \
         patch("self_rag.vectordb.factory.get_vectordb", return_value=mock_vectordb):
        result = health()
    assert result["status"] == "healthy"
    assert result["components"]["retriever_status"] == "operational"


def test_health_degraded():
    mock_vectordb = MagicMock()
    mock_vectordb.health_check.side_effect = RuntimeError("Qdrant connection failed")
    with patch("self_rag.mcp.tools.get_retriever", return_value=MagicMock()), \
         patch("self_rag.vectordb.factory.get_vectordb", return_value=mock_vectordb):
        result = health()
    assert result["status"] == "degraded"
    assert "Qdrant connection failed" in result["components"]["vectordb_status"]
