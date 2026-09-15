"""Tests for src.retrieval.context — Phase 4 composition.

Uses fakes only; no network, API keys, ChromaDB.
Covers 9 required assertions.
"""

import json

import pytest

from src.retrieval.context import RetrievalContext
from src.retrieval.formatter import format_context
from src.retrieval.reranker import LLMReranker, RankedCandidate
from src.retrieval.retriever import CoarseRetriever
from src.retrieval.token_budget import estimate_tokens


def _sample_items():
    return [
        {"id": "mem_001", "text": "User prefers dark mode.", "timestamp": "2026-01-15T10:00:00Z", "source": "session_1_turn_2", "embedding": [0.1]},
        {"id": "mem_002", "text": "User lives in Berlin.", "timestamp": "2026-01-16T12:30:00Z", "source": "session_1_turn_5", "embedding": [0.4]},
        {"id": "mem_003", "text": "User has a cat named Momo.", "timestamp": "2026-01-17T09:00:00Z", "source": "session_2_turn_1", "embedding": None},
    ]


class FakeStore:
    def __init__(self, items):
        self._items = items
        self.calls = []

    def query(self, text: str, k: int, tier: str):
        self.calls.append({"text": text, "k": k, "tier": tier})
        return self._items[:k]


def _fake_llm(return_value: str):
    def fake(prompt: str, system=None, **kwargs):
        fake.last_prompt = prompt
        fake.last_system = system
        return return_value
    return fake


# 1. CoarseRetriever output reaches LLMReranker.
def test_coarse_output_reaches_reranker():
    items = _sample_items()
    store = FakeStore(items)
    retriever = CoarseRetriever(store)
    # Track reranker input
    received = {}

    class TrackingReranker:
        def rerank(self, query, candidates):
            received["query"] = query
            received["candidates"] = candidates
            # Return simple ranked wrappers
            return [RankedCandidate(candidate=c, score=0.9 - i * 0.1) for i, c in enumerate(candidates)]

    reranker = TrackingReranker()
    ctx = RetrievalContext(retriever, reranker, k=2, tier="all")
    result = ctx.retrieve_context("hello world", token_budget=1000)
    # Store was called
    assert len(store.calls) == 1
    assert store.calls[0]["text"] == "hello world"
    assert store.calls[0]["k"] == 2
    # Reranker received exactly the candidates from store
    assert received["candidates"] == items[:2]
    assert received["query"] == "hello world"
    assert isinstance(result, str)


# 2. LLMReranker output reaches token-budget selection.
def test_reranker_output_reaches_budget_selection():
    items = _sample_items()
    store = FakeStore(items)
    retriever = CoarseRetriever(store)
    # Use real LLMReranker with fake LLM that returns valid ranking
    llm_response = json.dumps({"rankings": [{"id": "mem_002", "score": 0.9}, {"id": "mem_001", "score": 0.8}]})
    reranker = LLMReranker(_fake_llm(llm_response))

    selected_ids = {}

    def fake_selector(ranked, budget):
        selected_ids["ranked_ids"] = [r.candidate["id"] for r in ranked]
        selected_ids["budget"] = budget
        # Return all ranked for simplicity
        return ranked

    ctx = RetrievalContext(retriever, reranker, k=3, tier="all", selector=fake_selector, formatter_fn=lambda x: "formatted")
    ctx.retrieve_context("query", token_budget=100)
    # Selector received reranked order (mem_002 first per LLM)
    assert selected_ids["ranked_ids"][0] == "mem_002"
    assert selected_ids["budget"] == 100


# 3. selected candidates reach formatter.
def test_selected_candidates_reach_formatter():
    items = _sample_items()
    store = FakeStore(items)
    retriever = CoarseRetriever(store)
    reranker = LLMReranker(_fake_llm(json.dumps({"rankings": [{"id": "mem_001", "score": 0.9}]})))

    formatter_received = {}

    def fake_selector(ranked, budget):
        # Select only first ranked
        return ranked[:1]

    def fake_formatter(selected):
        formatter_received["selected"] = selected
        return "FAKE_CONTEXT"

    ctx = RetrievalContext(retriever, reranker, k=2, selector=fake_selector, formatter_fn=fake_formatter)
    result = ctx.retrieve_context("q", token_budget=100)
    assert result == "FAKE_CONTEXT"
    assert len(formatter_received["selected"]) == 1
    assert formatter_received["selected"][0].candidate["id"] == "mem_001"


# 4. final formatted context is returned.
def test_final_formatted_context_returned():
    items = _sample_items()
    store = FakeStore(items)
    retriever = CoarseRetriever(store)
    reranker = LLMReranker(_fake_llm(json.dumps({"rankings": [{"id": "mem_001", "score": 0.9}]})))
    ctx = RetrievalContext(retriever, reranker, k=1, tier="all")
    result = ctx.retrieve_context("dark mode", token_budget=1000)
    assert isinstance(result, str)
    assert "[Memory 1]" in result
    assert "User prefers dark mode." in result
    assert "Timestamp:" in result
    assert "Source:" in result


# 5. final context stays within token_budget according to same estimator.
def test_final_context_within_budget():
    # Huge candidate that would exceed small budget alone
    huge = {"id": "huge", "text": "word " * 200, "timestamp": "2026-01-01T00:00:00Z", "source": "s1", "embedding": None}
    small = {"id": "small", "text": "hi", "timestamp": "2026-01-02T00:00:00Z", "source": "s2", "embedding": None}
    items = [huge, small]
    store = FakeStore(items)
    retriever = CoarseRetriever(store)
    # LLM ranks huge first
    llm_response = json.dumps({"rankings": [{"id": "huge", "score": 0.9}, {"id": "small", "score": 0.5}]})
    reranker = LLMReranker(_fake_llm(llm_response))
    ctx = RetrievalContext(retriever, reranker, k=2, tier="all")
    result = ctx.retrieve_context("query", token_budget=30)
    # Should skip huge and include small, staying within budget
    assert estimate_tokens(result) <= 30
    assert "huge" not in result or "word word" not in result  # huge not included
    # With large budget, both fit
    result2 = ctx.retrieve_context("query", token_budget=1000)
    assert estimate_tokens(result2) <= 1000
    assert isinstance(result2, str)


# 6. empty retrieval produces valid empty context.
def test_empty_retrieval_produces_empty_context():
    store = FakeStore([])
    retriever = CoarseRetriever(store)
    reranker = LLMReranker(_fake_llm(json.dumps({"rankings": []})))
    ctx = RetrievalContext(retriever, reranker, k=5)
    result = ctx.retrieve_context("anything", token_budget=100)
    assert result == ""
    assert estimate_tokens(result) == 0
    # Also with real store empty but reranker fallback path
    assert isinstance(result, str)


# 7. injected fake LLM can operate through existing LLMReranker.
def test_injected_fake_llm_through_reranker():
    items = _sample_items()
    store = FakeStore(items)
    retriever = CoarseRetriever(store)
    # Fake LLM that ranks mem_003 highest
    llm_response = json.dumps({"rankings": [{"id": "mem_003", "score": 0.99}, {"id": "mem_001", "score": 0.5}]})
    fake = _fake_llm(llm_response)
    reranker = LLMReranker(fake)
    ctx = RetrievalContext(retriever, reranker, k=3)
    result = ctx.retrieve_context("cat Momo", token_budget=1000)
    # mem_003 should be first in formatted context
    assert result.index("Momo") < result.index("dark mode")
    assert "[Memory 1]" in result
    # Verify LLM was actually called with query
    assert "cat Momo" in fake.last_prompt


# 8. Phase-2 CoarseRetriever is used unchanged.
def test_phase2_coarse_retriever_unchanged():
    import pathlib
    import re
    retriever_path = pathlib.Path(__file__).parent.parent / "src" / "retrieval" / "retriever.py"
    source = retriever_path.read_text(encoding="utf-8")
    source_lower = source.lower()
    # Must not import LLM or ChromaDB
    assert not re.search(r"^\s*import\s+chromadb", source_lower, re.MULTILINE)
    assert not re.search(r"^\s*from\s+chromadb", source_lower, re.MULTILINE)
    assert "call_llm(" not in source
    # Verify RetrievalContext actually uses retriever.retrieve
    items = _sample_items()
    store = FakeStore(items)
    retriever = CoarseRetriever(store)
    assert hasattr(retriever, "retrieve")
    reranker = LLMReranker(_fake_llm(json.dumps({"rankings": []})))
    ctx = RetrievalContext(retriever, reranker, k=1)
    ctx.retrieve_context("q", token_budget=100)
    assert len(store.calls) == 1


# 9. Phase-3 LLMReranker is used unchanged.
def test_phase3_reranker_unchanged():
    import pathlib
    import re
    reranker_path = pathlib.Path(__file__).parent.parent / "src" / "retrieval" / "reranker.py"
    source = reranker_path.read_text(encoding="utf-8")
    source_lower = source.lower()
    assert not re.search(r"^\s*import\s+chromadb", source_lower, re.MULTILINE)
    assert not re.search(r"^\s*from\s+chromadb", source_lower, re.MULTILINE)
    assert not re.search(r"^\s*import\s+openai", source_lower, re.MULTILINE)
    assert not re.search(r"from\s+src\.llm", source)
    # Verify composition uses reranker.rerank
    called = {}

    class SpyReranker:
        def rerank(self, query, candidates):
            called["ran"] = True
            called["query"] = query
            return [RankedCandidate(candidate=c, score=0.5) for c in candidates]

    store = FakeStore(_sample_items())
    retriever = CoarseRetriever(store)
    reranker = SpyReranker()
    ctx = RetrievalContext(retriever, reranker, k=2)
    ctx.retrieve_context("hello", token_budget=1000)
    assert called["ran"] is True
    assert called["query"] == "hello"


# Additional edge cases for context
def test_token_budget_validation_in_context():
    store = FakeStore(_sample_items())
    retriever = CoarseRetriever(store)
    reranker = LLMReranker(_fake_llm(json.dumps({"rankings": []})))
    ctx = RetrievalContext(retriever, reranker)
    with pytest.raises(ValueError, match="token_budget"):
        ctx.retrieve_context("q", token_budget=0)
    with pytest.raises(ValueError, match="token_budget"):
        ctx.retrieve_context("q", token_budget=-5)
    # Budget too small for any memory -> returns empty string, not error
    huge = {"id": "h", "text": "word " * 500, "timestamp": "2026-01-01T00:00:00Z", "source": "s", "embedding": None}
    store2 = FakeStore([huge])
    retriever2 = CoarseRetriever(store2)
    ctx2 = RetrievalContext(retriever2, reranker, k=1)
    result = ctx2.retrieve_context("q", token_budget=5)
    assert result == ""
    assert estimate_tokens(result) <= 5


def test_deterministic_repeated_calls():
    store = FakeStore(_sample_items())
    retriever = CoarseRetriever(store)
    reranker = LLMReranker(_fake_llm(json.dumps({"rankings": [{"id": "mem_002", "score": 0.9}]})))
    ctx = RetrievalContext(retriever, reranker, k=3)
    first = ctx.retrieve_context("query", token_budget=100)
    second = ctx.retrieve_context("query", token_budget=100)
    assert first == second
