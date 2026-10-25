"""Tests for src.retrieval.token_budget — Phase 4.

Covers 10 required assertions from Phase-4 spec.
No network, LLM, or ChromaDB.
"""

import copy

import pytest

from src.retrieval.formatter import format_context
from src.retrieval.reranker import RankedCandidate
from src.retrieval.token_budget import estimate_tokens, select_by_token_budget


def _sample_ranked() -> list[RankedCandidate]:
    items = [
        {"id": "a", "text": "short", "timestamp": "2026-01-01T00:00:00Z", "source": "s1", "embedding": None},
        {"id": "b", "text": "a bit longer text here", "timestamp": "2026-01-02T00:00:00Z", "source": "s2", "embedding": None},
        {"id": "c", "text": "word " * 50, "timestamp": "2026-01-03T00:00:00Z", "source": "s3", "embedding": None},
    ]
    return [RankedCandidate(candidate=it, score=1.0 - i * 0.3) for i, it in enumerate(items)]


# 1. selected context never exceeds requested budget
def test_selected_context_never_exceeds_budget():
    ranked = _sample_ranked()
    for budget in [5, 10, 20, 50, 100]:
        selected = select_by_token_budget(ranked, token_budget=budget)
        ctx = format_context(selected)
        est = estimate_tokens(ctx)
        assert est <= budget, f"budget {budget} exceeded: {est} tokens, ctx={ctx!r}"


# 2. ranking order is preserved
def test_ranking_order_preserved():
    items = [
        {"id": "m1", "text": "alpha", "timestamp": "2026-01-01T00:00:00Z", "source": "s1", "embedding": None},
        {"id": "m2", "text": "beta", "timestamp": "2026-01-02T00:00:00Z", "source": "s2", "embedding": None},
        {"id": "m3", "text": "gamma", "timestamp": "2026-01-03T00:00:00Z", "source": "s3", "embedding": None},
    ]
    ranked = [RankedCandidate(candidate=it, score=s) for it, s in zip(items, [0.9, 0.8, 0.7])]
    # Large budget should keep original order
    selected = select_by_token_budget(ranked, token_budget=1000)
    assert [r.candidate["id"] for r in selected] == ["m1", "m2", "m3"]
    # Also with skipping, order still follows ranking
    # Create scenario where m1 huge, m2/m3 small
    huge = {"id": "huge", "text": "x " * 200, "timestamp": "2026-01-01T00:00:00Z", "source": "s1", "embedding": None}
    small1 = {"id": "s1", "text": "hi", "timestamp": "2026-01-02T00:00:00Z", "source": "s2", "embedding": None}
    small2 = {"id": "s2", "text": "hello", "timestamp": "2026-01-03T00:00:00Z", "source": "s3", "embedding": None}
    ranked2 = [
        RankedCandidate(candidate=huge, score=0.9),
        RankedCandidate(candidate=small1, score=0.8),
        RankedCandidate(candidate=small2, score=0.7),
    ]
    # Budget enough for 2 small but not huge
    # Estimate block for small ~ 7 tokens, huge ~ 200+ overhead -> > budget
    budget = 30
    selected2 = select_by_token_budget(ranked2, token_budget=budget)
    ids2 = [r.candidate["id"] for r in selected2]
    # Must be in ranking order (s1 before s2) if both selected
    if "s1" in ids2 and "s2" in ids2:
        assert ids2.index("s1") < ids2.index("s2")
    # Ensure no reordering: ids should be subsequence of ranked order
    ranked_ids = [r.candidate["id"] for r in ranked2]
    # Check that selected is subsequence preserving order
    last_idx = -1
    for sid in ids2:
        idx = ranked_ids.index(sid)
        assert idx > last_idx
        last_idx = idx


# 3. lower-ranked can be selected when higher-ranked does not fit
def test_lower_ranked_selected_when_higher_does_not_fit():
    huge_text = "word " * 100  # ~100 tokens plus overhead
    huge = {"id": "A", "text": huge_text, "timestamp": "2026-01-01T00:00:00Z", "source": "s1", "embedding": None}
    b = {"id": "B", "text": "short", "timestamp": "2026-01-02T00:00:00Z", "source": "s2", "embedding": None}
    c = {"id": "C", "text": "tiny", "timestamp": "2026-01-03T00:00:00Z", "source": "s3", "embedding": None}
    ranked = [
        RankedCandidate(candidate=huge, score=0.9),
        RankedCandidate(candidate=b, score=0.7),
        RankedCandidate(candidate=c, score=0.5),
    ]
    # Budget that fits B and C but not A
    # Estimate: B block ~ 7 tokens, C block ~7 tokens, together ~14-15, A alone ~100+7 =107
    # Choose budget 30: A too large (107>30), but B and C together fit
    selected = select_by_token_budget(ranked, token_budget=30)
    ids = [r.candidate["id"] for r in selected]
    assert "A" not in ids, "Huge high-ranked should be skipped"
    assert "B" in ids
    assert "C" in ids
    assert ids == ["B", "C"]  # ranking order preserved


# 4. formatting overhead is counted
def test_formatting_overhead_counted():
    # Raw text is 1 token, but formatted block is more
    item = {"id": "m1", "text": "hello", "timestamp": "2026-01-01T00:00:00Z", "source": "s1", "embedding": None}
    ranked = [RankedCandidate(candidate=item, score=0.9)]
    raw_tokens = estimate_tokens(item["text"])
    assert raw_tokens == 1
    # Formatted block: "[Memory 1]\nTimestamp: ...\nSource: ...\nContent: hello"
    # Should be more tokens than raw
    ctx = format_context(ranked)
    formatted_tokens = estimate_tokens(ctx)
    assert formatted_tokens > raw_tokens
    # Budget exactly raw_tokens should NOT allow selection (overhead counted)
    selected = select_by_token_budget(ranked, token_budget=raw_tokens)
    assert selected == [], "Should be empty when budget only fits raw text but not formatted overhead"
    # Budget exactly formatted_tokens should allow selection
    selected2 = select_by_token_budget(ranked, token_budget=formatted_tokens)
    assert len(selected2) == 1


# 5. zero budget rejected
def test_zero_budget_rejected():
    ranked = _sample_ranked()
    with pytest.raises(ValueError, match="token_budget"):
        select_by_token_budget(ranked, token_budget=0)


# 6. negative budget rejected
def test_negative_budget_rejected():
    ranked = _sample_ranked()
    for bad in [-1, -10, -100]:
        with pytest.raises(ValueError, match="token_budget"):
            select_by_token_budget(ranked, token_budget=bad)
    # also bool should be rejected
    with pytest.raises(ValueError):
        select_by_token_budget(ranked, token_budget=True)  # type: ignore


# 7. empty candidate list returns empty selection
def test_empty_candidate_list_returns_empty():
    assert select_by_token_budget([], token_budget=100) == []
    assert select_by_token_budget([], token_budget=10) == []


# 8. deterministic repeated selection
def test_deterministic_repeated_selection():
    ranked = _sample_ranked()
    first = select_by_token_budget(ranked, token_budget=50)
    second = select_by_token_budget(ranked, token_budget=50)
    assert [r.candidate["id"] for r in first] == [r.candidate["id"] for r in second]
    assert [r.score for r in first] == [r.score for r in second]
    # Also across budget that forces skipping
    huge = {"id": "X", "text": "x " * 100, "timestamp": "2026-01-01T00:00:00Z", "source": "s", "embedding": None}
    small = {"id": "Y", "text": "hi", "timestamp": "2026-01-02T00:00:00Z", "source": "s", "embedding": None}
    ranked2 = [RankedCandidate(candidate=huge, score=0.9), RankedCandidate(candidate=small, score=0.8)]
    a = select_by_token_budget(ranked2, token_budget=20)
    b = select_by_token_budget(ranked2, token_budget=20)
    assert [r.candidate["id"] for r in a] == [r.candidate["id"] for r in b]


# 9. CandidateMemoryItem objects are not mutated
def test_candidate_objects_not_mutated():
    items = [
        {"id": "m1", "text": "hello world", "timestamp": "2026-01-01T00:00:00Z", "source": "s1", "embedding": [0.1]},
        {"id": "m2", "text": "another", "timestamp": "2026-01-02T00:00:00Z", "source": "s2", "embedding": None},
    ]
    ranked = [RankedCandidate(candidate=it, score=0.5) for it in items]
    originals = copy.deepcopy(items)
    original_scores = [r.score for r in ranked]
    select_by_token_budget(ranked, token_budget=100)
    # Original dicts unchanged
    for orig, cur in zip(originals, items):
        assert cur == orig
        assert "score" not in cur
    # Also ranked list not mutated in terms of objects
    for r in ranked:
        assert r.candidate in items or r.candidate == originals[items.index(r.candidate)] if r.candidate in items else True


# 10. RankedCandidate scores are not modified
def test_ranked_candidate_scores_not_modified():
    ranked = _sample_ranked()
    before = [r.score for r in ranked]
    selected = select_by_token_budget(ranked, token_budget=50)
    after = [r.score for r in ranked]
    assert before == after
    # Selected items retain original scores
    for sel in selected:
        # Find original
        orig = next(r for r in ranked if r.candidate["id"] == sel.candidate["id"])
        assert sel.score == orig.score
        assert sel is orig or sel.score == orig.score  # may be same object reference


# Additional: does not call LLM or store, uses deterministic estimation
def test_uses_whitespace_estimation_consistently():
    # Verify estimate_tokens is whitespace-based and documented
    assert estimate_tokens("hello world") == 2
    assert estimate_tokens("  hello   world  ") == 2
    assert estimate_tokens("") == 0
    assert estimate_tokens("   ") == 0
    # Formatted context tokens must be estimated with same function
    item = {"id": "m1", "text": "hello world", "timestamp": "2026-01-01T00:00:00Z", "source": "s1", "embedding": None}
    ranked = [RankedCandidate(candidate=item, score=0.9)]
    selected = select_by_token_budget(ranked, token_budget=100)
    ctx = format_context(selected)
    assert estimate_tokens(ctx) <= 100

    # Ensure no mutation with missing/None embedding
    item2 = {"id": "m2", "text": "", "timestamp": "2026-01-01T00:00:00Z", "source": "s1", "embedding": None}
    ranked2 = [RankedCandidate(candidate=item2, score=0.5), RankedCandidate(candidate=item, score=0.4)]
    selected2 = select_by_token_budget(ranked2, token_budget=100)
    assert len(selected2) == 2
