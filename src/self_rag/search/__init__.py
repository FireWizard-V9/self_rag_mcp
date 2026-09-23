"""Search strategy abstraction layer."""

from self_rag.search.base import SearchStrategy
from self_rag.search.factory import get_search_strategy
from self_rag.search.strategies.rrf_strategy import RRFStrategy
from self_rag.search.strategies.semantic_strategy import SemanticStrategy
from self_rag.search.strategies.weighted_strategy import WeightedStrategy

__all__ = [
    "SearchStrategy",
    "get_search_strategy",
    "RRFStrategy",
    "WeightedStrategy",
    "SemanticStrategy",
]
