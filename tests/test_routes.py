"""Tests for graph routing logic."""

import pytest
from self_rag.graph.routes import (
    route_after_retrieval_decision,
    route_after_support,
    route_after_usefulness,
)


def base_state(**overrides):
    state = {
        "question": "test",
        "need_retrieval": True,
        "should_retrieve": True,
        "documents": [],
        "retrieved_documents": [],
        "relevant_documents": [],
        "context": "",
        "answer": "",
        "support_grade": "fully_supported",
        "usefulness_grade": "useful",
        "retry_count": 0,
        "max_retries": 2,
    }
    state.update(overrides)
    return state


# --- route_after_retrieval_decision ---

def test_retrieval_decision_retrieve():
    assert route_after_retrieval_decision(base_state(should_retrieve=True)) == "retrieve"


def test_retrieval_decision_skip():
    assert route_after_retrieval_decision(base_state(should_retrieve=False)) == "generator"


# --- route_after_support ---

def test_support_fully_supported_goes_to_usefulness():
    assert route_after_support(base_state(support_grade="fully_supported")) == "usefulness"


def test_support_partially_supported_goes_to_usefulness():
    assert route_after_support(base_state(support_grade="partially_supported")) == "usefulness"


def test_support_not_supported_retries_when_budget_remains():
    assert route_after_support(base_state(support_grade="not_supported", retry_count=0, max_retries=2)) == "generator"


def test_support_not_supported_goes_to_usefulness_when_budget_exhausted():
    assert route_after_support(base_state(support_grade="not_supported", retry_count=2, max_retries=2)) == "usefulness"


# --- route_after_usefulness ---

def test_usefulness_useful_ends():
    assert route_after_usefulness(base_state(usefulness_grade="useful")) == "__end__"


def test_usefulness_not_useful_retries_when_budget_remains():
    assert route_after_usefulness(base_state(usefulness_grade="not_useful", retry_count=0, max_retries=2)) == "increment_retry_for_retrieval"


def test_usefulness_not_useful_ends_when_budget_exhausted():
    assert route_after_usefulness(base_state(usefulness_grade="not_useful", retry_count=2, max_retries=2)) == "__end__"
