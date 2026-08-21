"""LangGraph workflow."""

from langgraph.graph import END, StateGraph

from self_rag.graph.routes import (
    route_after_retrieval_decision,
    route_after_support,
    route_after_usefulness,
)
from self_rag.models.graph_state import GraphState
from self_rag.nodes.context_builder import context_builder
from self_rag.nodes.generator import generator
from self_rag.nodes.relevance_grader import relevance_grader
from self_rag.nodes.retrieval_decision import retrieval_decision
from self_rag.nodes.retrieve import retrieve_node
from self_rag.nodes.support_grader import support_grader
from self_rag.nodes.usefulness_grader import usefulness_grader


def increment_retry(state: GraphState) -> dict:
    return {
        "retry_count": state["retry_count"] + 1,
    }


def increment_retry_for_retrieval(state: GraphState) -> dict:
    return {"retry_count": state["retry_count"] + 1}


def build_graph():

    graph = StateGraph(GraphState)

    graph.add_node("retrieval_decision", retrieval_decision)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("relevance", relevance_grader)
    graph.add_node("context", context_builder)
    graph.add_node("generator", generator)
    graph.add_node("support", support_grader)
    graph.add_node("usefulness", usefulness_grader)
    graph.add_node("increment_retry", increment_retry)
    graph.add_node("increment_retry_for_retrieval", increment_retry_for_retrieval)

    graph.set_entry_point("retrieval_decision")
    graph.add_conditional_edges(
        "retrieval_decision",
        route_after_retrieval_decision,
        {
            "retrieve": "retrieve",
            "generator": "generator",
        },
    )

    graph.add_edge("retrieve", "relevance")
    graph.add_edge("relevance", "context")
    graph.add_edge("context", "generator")
    graph.add_edge("generator", "support")

    graph.add_conditional_edges(
        "support",
        route_after_support,
        {
            "generator": "increment_retry",
            "usefulness": "usefulness",
        },
    )

    graph.add_edge("increment_retry", "generator")

    graph.add_conditional_edges(
        "usefulness",
        route_after_usefulness,
        {
            "increment_retry_for_retrieval": "increment_retry_for_retrieval",
            "__end__": END,
        },
    )
    graph.add_edge("increment_retry_for_retrieval", "retrieve")

    return graph.compile()
