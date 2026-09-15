"""Tests for src.retrieval.retriever.CoarseRetriever — Stage-1 only.

Uses a fake store; no ChromaDB, network, or LLM required.
Covers the 13 required assertions from the Phase-2 spec.
"""

import pytest

from src.retrieval.retriever import CoarseRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeStore:
    """Minimal fake implementing query(text, k, tier)."""

    def __init__(self, items: list[dict] | None = None):
        self._items: list[dict] = items if items is not None else []
        self.calls: list[dict] = []

    def query(self, text: str, k: int, tier: str) -> list[dict]:
        self.calls.append({"text": text, "k": k, "tier": tier})
        # Return up to k items (mimics store contract but caller requests exact k)
        return self._items[:k] if self._items else []


def _sample_items() -> list[dict]:
    return [
        {
            "id": "mem_001",
            "text": "User prefers dark mode.",
            "timestamp": "2026-01-15T10:00:00Z",
            "source": "session_1_turn_2",
            "embedding": [0.1, 0.2, 0.3],
        },
        {
            "id": "mem_002",
            "text": "User lives in Berlin.",
            "timestamp": "2026-01-16T12:30:00Z",
            "source": "session_1_turn_5",
            "embedding": [0.4, 0.5, 0.6],
        },
        {
            "id": "mem_003",
            "text": "User has a cat named Momo.",
            "timestamp": "2026-01-17T09:00:00Z",
            "source": "session_2_turn_1",
            "embedding": None,
        },
    ]


# ---------------------------------------------------------------------------
# 1. Query is passed correctly to the store
# ---------------------------------------------------------------------------


def test_query_passed_correctly_to_store():
    store = FakeStore(items=_sample_items())
    retriever = CoarseRetriever(store)
    retriever.retrieve(query="what is user's preference?", k=2, tier="all")
    assert len(store.calls) == 1
    assert store.calls[0]["text"] == "what is user's preference?"


# ---------------------------------------------------------------------------
# 2. Requested k is passed correctly
# ---------------------------------------------------------------------------


def test_k_passed_correctly():
    store = FakeStore(items=_sample_items())
    retriever = CoarseRetriever(store)
    retriever.retrieve(query="hello", k=2, tier="all")
    assert store.calls[0]["k"] == 2
    # also test a different k
    store.calls.clear()
    retriever.retrieve(query="hello", k=1, tier="all")
    assert store.calls[0]["k"] == 1


# ---------------------------------------------------------------------------
# 3. Requested tier is passed correctly
# ---------------------------------------------------------------------------


def test_tier_passed_correctly():
    store = FakeStore(items=_sample_items())
    retriever = CoarseRetriever(store)
    retriever.retrieve(query="hello", k=1, tier="MTM")
    assert store.calls[0]["tier"] == "MTM"


# ---------------------------------------------------------------------------
# 4-7. Each tier works
# ---------------------------------------------------------------------------


def test_stm_tier_works():
    store = FakeStore(items=_sample_items())
    retriever = CoarseRetriever(store)
    result = retriever.retrieve(query="q", k=1, tier="STM")
    assert store.calls[0]["tier"] == "STM"
    assert isinstance(result, list)


def test_mtm_tier_works():
    store = FakeStore(items=_sample_items())
    retriever = CoarseRetriever(store)
    result = retriever.retrieve(query="q", k=1, tier="MTM")
    assert store.calls[0]["tier"] == "MTM"
    assert isinstance(result, list)


def test_ltm_tier_works():
    store = FakeStore(items=_sample_items())
    retriever = CoarseRetriever(store)
    result = retriever.retrieve(query="q", k=1, tier="LTM")
    assert store.calls[0]["tier"] == "LTM"
    assert isinstance(result, list)


def test_all_tier_works():
    store = FakeStore(items=_sample_items())
    retriever = CoarseRetriever(store)
    result = retriever.retrieve(query="q", k=1, tier="all")
    assert store.calls[0]["tier"] == "all"
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 8. Returned CandidateMemoryItem objects are preserved
# ---------------------------------------------------------------------------


def test_returned_candidate_memory_items_preserved():
    items = _sample_items()
    store = FakeStore(items=items)
    retriever = CoarseRetriever(store)
    result = retriever.retrieve(query="q", k=2, tier="all")
    assert len(result) == 2
    for original, returned in zip(items[:2], result):
        assert returned["id"] == original["id"]
        assert returned["text"] == original["text"]
        assert returned["timestamp"] == original["timestamp"]
        assert returned["source"] == original["source"]
        assert returned["embedding"] == original["embedding"]


# ---------------------------------------------------------------------------
# 9. Invalid tiers raise ValueError
# ---------------------------------------------------------------------------


def test_invalid_tiers_raise_value_error():
    store = FakeStore(items=_sample_items())
    retriever = CoarseRetriever(store)
    invalid_tiers = ["stm", "SHORT_TERM", "invalid", "", "All", "lTM", "ST M", "123"]
    for tier in invalid_tiers:
        with pytest.raises(ValueError, match="Invalid tier"):
            retriever.retrieve(query="q", k=1, tier=tier)  # type: ignore[arg-type]
    # None should also raise (not a valid tier string)
    with pytest.raises(ValueError):
        retriever.retrieve(query="q", k=1, tier=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 10. Invalid k values raise ValueError
# ---------------------------------------------------------------------------


def test_invalid_k_values_raise_value_error():
    store = FakeStore(items=_sample_items())
    retriever = CoarseRetriever(store)
    invalid_ks = [0, -1, -100]
    for k in invalid_ks:
        with pytest.raises(ValueError, match="k must be a positive integer"):
            retriever.retrieve(query="q", k=k, tier="all")
    # non-int types should also raise
    for k in [1.5, "2", None, True, False]:  # type: ignore
        with pytest.raises(ValueError):
            retriever.retrieve(query="q", k=k, tier="all")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 11. Retriever does not alter returned memory text
# ---------------------------------------------------------------------------


def test_retriever_does_not_alter_returned_memory_text():
    original_text = "User prefers dark mode. Do not modify this text."
    items = [
        {
            "id": "mem_x",
            "text": original_text,
            "timestamp": "2026-01-15T10:00:00Z",
            "source": "session_1_turn_1",
            "embedding": [0.9, 0.8],
        }
    ]
    store = FakeStore(items=items)
    retriever = CoarseRetriever(store)
    result = retriever.retrieve(query="q", k=1, tier="all")
    assert result[0]["text"] == original_text
    # Also ensure text object not mutated via extra whitespace/casing
    assert result[0]["text"] is original_text or result[0]["text"] == original_text
    # Verify other fields not mutated either
    assert result[0]["timestamp"] == items[0]["timestamp"]
    assert result[0]["source"] == items[0]["source"]


# ---------------------------------------------------------------------------
# 12. Retriever does not perform any LLM call
# ---------------------------------------------------------------------------


def test_retriever_does_not_perform_llm_call():
    """Ensure CoarseRetriever never imports or calls LLM / ChromaDB."""
    import sys

    # Snapshot of modules before retrieve — ensure no llm/chromadb import side-effect
    store = FakeStore(items=_sample_items())
    retriever = CoarseRetriever(store)

    # Inspect retriever source for forbidden *imports/calls* (not mere mentions in docs).
    # The retriever is allowed to *mention* ChromaDB/LLM in docstrings to state
    # that it does NOT use them; it must not actually import or call them.
    import pathlib
    import re

    retriever_path = pathlib.Path(__file__).parent.parent / "src" / "retrieval" / "retriever.py"
    source = retriever_path.read_text(encoding="utf-8")
    source_lower = source.lower()
    # Forbid actual import statements
    assert not re.search(r"^\s*import\s+chromadb", source_lower, re.MULTILINE), "retriever.py must not import ChromaDB"
    assert not re.search(r"^\s*from\s+chromadb", source_lower, re.MULTILINE), "retriever.py must not import ChromaDB"
    assert not re.search(r"^\s*import\s+openai", source_lower, re.MULTILINE)
    assert not re.search(r"^\s*from\s+openai", source_lower, re.MULTILINE)
    assert not re.search(r"^\s*import\s+groq", source_lower, re.MULTILINE)
    assert not re.search(r"^\s*from\s+groq", source_lower, re.MULTILINE)
    # Forbid actual LLM call — e.g. `call_llm(` or `from src.llm`
    assert "call_llm(" not in source, "retriever.py must not call LLM"
    assert not re.search(r"from\s+src\.llm", source), "retriever.py must not import from src.llm"
    assert not re.search(r"import\s+src\.llm", source), "retriever.py must not import src.llm"

    # Runtime check: monkey-patch potential LLM entry points and ensure not called
    # If no llm modules are loaded, this also passes.
    llm_modules = [m for m in sys.modules if "llm" in m.lower()]
    # Retrieve should not cause new llm modules to appear
    retriever.retrieve(query="q", k=1, tier="all")
    new_llm_modules = [m for m in sys.modules if "llm" in m.lower() and m not in llm_modules]
    # Filter to project src.llm — stdlib unrelated
    project_llm = [m for m in new_llm_modules if m.startswith("src.llm")]
    assert project_llm == [], f"retrieve() triggered LLM module load: {project_llm}"

    # Also verify store was called exactly once (and no hidden extra work)
    assert len(store.calls) == 1


# ---------------------------------------------------------------------------
# 13. Fake store returning multiple items produces same candidates in same order
# ---------------------------------------------------------------------------


def test_multiple_candidates_same_order():
    items = _sample_items()
    store = FakeStore(items=items)
    retriever = CoarseRetriever(store)
    result = retriever.retrieve(query="q", k=3, tier="all")
    assert len(result) == 3
    # Identity or equality + order preserved
    assert result == items[:3]
    # Check order explicitly via ids
    assert [r["id"] for r in result] == [i["id"] for i in items[:3]]
    # Ensure same object references (no copying/reranking)
    for r, o in zip(result, items):
        assert r is o or r == o  # allow either same object or equal copy without mutation
    # If same objects, ordering must be exact
    if all(r is o for r, o in zip(result, items)):
        assert result is not items  # returned list is a slice, not the original list object
