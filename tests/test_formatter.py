"""Tests for src.retrieval.formatter — Phase 4.

Covers 10 required assertions from Phase-4 spec.
"""

import copy

from src.retrieval.formatter import format_context, format_memory
from src.retrieval.reranker import RankedCandidate


def _sample_items():
    return [
        {"id": "mem_001", "text": "User prefers dark mode.", "timestamp": "2026-01-15T10:00:00Z", "source": "session_1_turn_2", "embedding": [0.1]},
        {"id": "mem_002", "text": "User lives in Berlin.", "timestamp": "2026-01-16T12:30:00Z", "source": "session_1_turn_5", "embedding": None},
    ]


# 1. single-memory formatting
def test_single_memory_formatting():
    item = {"id": "m1", "text": "hello world", "timestamp": "2026-01-01T00:00:00Z", "source": "s1", "embedding": None}
    out = format_context([item])
    assert "[Memory 1]" in out
    assert "Timestamp: 2026-01-01T00:00:00Z" in out
    assert "Source: s1" in out
    assert "Content: hello world" in out
    # Exactly one memory block
    assert out.count("[Memory") == 1
    # Check format_memory directly
    single = format_memory(item, 1)
    assert single == out


# 2. multiple-memory formatting
def test_multiple_memory_formatting():
    items = _sample_items()
    out = format_context(items)
    assert "[Memory 1]" in out
    assert "[Memory 2]" in out
    assert out.count("[Memory") == 2
    # Separator is double newline
    assert "\n\n[Memory 2]" in out
    # Both texts present
    assert "User prefers dark mode." in out
    assert "User lives in Berlin." in out


# 3. ordering is preserved
def test_ordering_preserved():
    items = [
        {"id": "a", "text": "first", "timestamp": "2026-01-01T00:00:00Z", "source": "s1", "embedding": None},
        {"id": "b", "text": "second", "timestamp": "2026-01-02T00:00:00Z", "source": "s2", "embedding": None},
        {"id": "c", "text": "third", "timestamp": "2026-01-03T00:00:00Z", "source": "s3", "embedding": None},
    ]
    out = format_context(items)
    # Check order via positions
    assert out.index("first") < out.index("second") < out.index("third")
    assert out.index("[Memory 1]") < out.index("[Memory 2]") < out.index("[Memory 3]")
    # Also with RankedCandidate wrappers
    ranked = [RankedCandidate(candidate=it, score=0.9 - i * 0.1) for i, it in enumerate(items)]
    out2 = format_context(ranked)
    assert out2.index("first") < out2.index("second")


# 4. timestamp is preserved
def test_timestamp_preserved():
    items = [
        {"id": "m1", "text": "t", "timestamp": "2026-01-15T10:00:00Z", "source": "s1", "embedding": None},
        {"id": "m2", "text": "t", "timestamp": "2026-12-31T23:59:59Z", "source": "s2", "embedding": None},
    ]
    out = format_context(items)
    assert "2026-01-15T10:00:00Z" in out
    assert "2026-12-31T23:59:59Z" in out
    # Exact preservation, no modification
    for item in items:
        assert f"Timestamp: {item['timestamp']}" in out


# 5. source is preserved
def test_source_preserved():
    items = [
        {"id": "m1", "text": "t", "timestamp": "2026-01-01T00:00:00Z", "source": "session_1_turn_2", "embedding": None},
        {"id": "m2", "text": "t", "timestamp": "2026-01-01T00:00:00Z", "source": "session_12_turn_4", "embedding": None},
    ]
    out = format_context(items)
    assert "Source: session_1_turn_2" in out
    assert "Source: session_12_turn_4" in out


# 6. memory text is preserved exactly
def test_memory_text_preserved_exactly():
    # Include punctuation, casing, whitespace
    text = "User prefers dark mode.  Do not modify THIS text!  "
    item = {"id": "m1", "text": text, "timestamp": "2026-01-01T00:00:00Z", "source": "s1", "embedding": None}
    out = format_context([item])
    # Content line must contain exact text
    assert f"Content: {text}" in out
    # Ensure no lowercasing or trimming
    assert text in out
    # Check with RankedCandidate
    ranked = RankedCandidate(candidate=item, score=0.9)
    out2 = format_context([ranked])
    assert f"Content: {text}" in out2

    # Empty text
    empty_item = {"id": "m2", "text": "", "timestamp": "2026-01-01T00:00:00Z", "source": "s1", "embedding": None}
    out3 = format_context([empty_item])
    assert "Content: " in out3
    assert "Content: \n" in out3 or out3.endswith("Content: ")


# 7. empty input
def test_empty_input():
    assert format_context([]) == ""
    assert format_context([]) == ""
    # No header for empty
    assert "[Memory" not in format_context([])


# 8. deterministic repeated formatting
def test_deterministic_repeated_formatting():
    items = _sample_items()
    first = format_context(items)
    second = format_context(items)
    assert first == second
    # With RankedCandidate wrappers
    ranked = [RankedCandidate(candidate=it, score=0.8) for it in items]
    assert format_context(ranked) == format_context(ranked)
    # Same output regardless of calling twice
    assert format_memory(items[0], 1) == format_memory(items[0], 1)


# 9. candidates are not mutated
def test_candidates_not_mutated():
    items = _sample_items()
    originals = copy.deepcopy(items)
    ranked = [RankedCandidate(candidate=it, score=0.7) for it in items]
    format_context(items)
    format_context(ranked)
    format_memory(items[0], 1)
    for orig, cur in zip(originals, items):
        assert cur == orig
        assert "score" not in cur
    # RankedCandidate not mutated
    for r in ranked:
        assert "score" not in r.candidate


# 10. formatting does not add invented metadata
def test_formatting_does_not_add_invented_metadata():
    item = {"id": "m1", "text": "hello", "timestamp": "2026-01-01T00:00:00Z", "source": "s1", "embedding": None}
    out = format_context([item])
    # Should not contain score
    assert "score" not in out.lower()
    # Should not contain embedding
    assert "embedding" not in out.lower()
    # Should not contain invented IDs beyond header
    assert "mem_001" not in out or item["id"] == "mem_001"  # only if input id matches
    # For this specific item, id is m1, should not appear as header id content unless it's the header index
    # Check that only expected fields appear
    assert out.count("Timestamp:") == 1
    assert out.count("Source:") == 1
    assert out.count("Content:") == 1
    # No extra labels
    assert "Ranking" not in out
    assert "Relevance" not in out
    # Check with RankedCandidate with high score — still no score leakage
    ranked = RankedCandidate(candidate=item, score=0.99)
    out2 = format_context([ranked])
    assert "0.99" not in out2
    assert "score" not in out2.lower()
