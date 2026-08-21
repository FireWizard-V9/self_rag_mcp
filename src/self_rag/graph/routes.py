"""Conditional routing for the Self-RAG LangGraph."""

from typing import Literal

from self_rag.models.graph_state import GraphState


def route_after_retrieval_decision(
    state: GraphState,
) -> Literal["retrieve", "generator"]:
    """Skip retrieval when unnecessary."""

    if state["should_retrieve"]:
        return "retrieve"

    return "generator"


def route_after_support(
    state: GraphState,
) -> Literal["usefulness", "generator"]:
    """
    If the answer is unsupported, retry generation until retry budget is exhausted.
    """

    if state["support_grade"] in (
        "fully_supported",
        "partially_supported",
    ):
        return "usefulness"

    if state["retry_count"] < state["max_retries"]:
        return "generator"

    return "usefulness"


def route_after_usefulness(
    state: GraphState,
) -> Literal["increment_retry_for_retrieval", "__end__"]:
    """
    If answer isn't useful, retrieve again (optionally after query rewrite).
    """

    if state["usefulness_grade"] == "useful":
        return "__end__"

    if state["retry_count"] >= state["max_retries"]:
        return "__end__"

    return "increment_retry_for_retrieval"
