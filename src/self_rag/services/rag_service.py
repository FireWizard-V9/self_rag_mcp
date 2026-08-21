"""Business layer for Self-RAG."""

from typing import Any

from self_rag.graph.engine import get_graph


class RAGService:
    def __init__(self):
        self.graph = get_graph()

    def answer(
        self,
        question: str,
        max_retries: int = 2,
    ) -> dict[str, Any]:

        state = {
            "question": question,
            "rewritten_question": None,
            "should_retrieve": True,
            "retrieved_documents": [],
            "relevant_documents": [],
            "context": "",
            "answer": "",
            "support_grade": "not_supported",
            "usefulness_grade": "not_useful",
            "retry_count": 0,
            "max_retries": max_retries,
        }

        result = self.graph.invoke(state)

        return {
            "question": question,
            "answer": result["answer"],
            "support_grade": result["support_grade"],
            "usefulness_grade": result["usefulness_grade"],
            "sources": [
                doc.metadata
                for doc in result.get(
                    "relevant_documents",
                    [],
                )
            ],
        }
