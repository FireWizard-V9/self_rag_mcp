"""Factory for creating search strategies."""

from self_rag.core.config import get_settings
from self_rag.search.base import SearchStrategy
from self_rag.search.strategies.rrf_strategy import RRFStrategy
from self_rag.search.strategies.semantic_strategy import SemanticStrategy
from self_rag.search.strategies.weighted_strategy import WeightedStrategy


def get_search_strategy() -> SearchStrategy:
    """
    Factory function to get the configured search strategy.
    Selection via VECTORDB_HYBRID_STRATEGY env var.

    Returns:
        SearchStrategy: Concrete strategy instance (RRF, Weighted, Semantic, etc.)

    Raises:
        ValueError: If VECTORDB_HYBRID_STRATEGY is unknown
    """
    settings = get_settings()
    strategy_name = settings.vectordb_hybrid_strategy.lower()

    if strategy_name == "rrf":
        return RRFStrategy()

    elif strategy_name == "weighted":
        return WeightedStrategy(alpha=0.5)

    elif strategy_name == "semantic":
        return SemanticStrategy()

    elif strategy_name == "two_pass":
        # Two-pass: dense → reranking (handled by HybridRetriever's reranker)
        return SemanticStrategy()

    else:
        raise ValueError(
            f"Unknown search strategy: {strategy_name}. "
            f"Supported: rrf, weighted, semantic, two_pass"
        )
