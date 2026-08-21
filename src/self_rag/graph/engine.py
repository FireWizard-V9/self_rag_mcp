"""Singleton graph instance."""

from functools import lru_cache

from langgraph.graph.state import CompiledStateGraph

from self_rag.graph.workflow import build_graph


@lru_cache(maxsize=1)
def get_graph() -> CompiledStateGraph:
    """Get the graph instance."""
    return build_graph()
