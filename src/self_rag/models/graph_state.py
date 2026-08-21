from typing import Literal, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    question: str
    # control
    need_retrieval: bool

    # retrieval
    documents: list[Document]
    should_retrieve: bool
    retrieved_documents: list[Document]
    relevant_documents: list[Document]

    # generation
    context: str
    answer: str

    # grading
    usefulness_grade: Literal[
        "useful",
        "not_useful",
    ]

    support_grade: Literal[
        "fully_supported",
        "partially_supported",
        "not_supported",
    ]

    retry_count: int
    max_retries: int
