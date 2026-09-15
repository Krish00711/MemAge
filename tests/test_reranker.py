"""Tests for src.retrieval.reranker.LLMReranker — Stage-2 only.

Uses fake LLM callables; no network, API key, or ChromaDB required.
Covers the 20 required assertions from the Phase-3 spec.
"""

import copy
import json
import pathlib
import re

import pytest

from src.retrieval.reranker import LLMReranker, RankedCandidate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
            "text": "User lives in Berlin and loves cats.",
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


def _capture_llm(return_value: str | None = None, side_effect: Exception | None = None):
    """Create a fake LLM that captures the last prompt/system."""
    capture: dict = {}

    def fake(prompt: str, system: str | None = None, **kwargs):
        capture["prompt"] = prompt
        capture["system"] = system
        capture["kwargs"] = kwargs
        if side_effect is not None:
            raise side_effect
        return return_value

    fake.capture = capture  # type: ignore[attr-defined]
    return fake


# ---------------------------------------------------------------------------
# 1. Valid LLM JSON produces the requested ranking.
# ---------------------------------------------------------------------------


def test_valid_llm_json_produces_requested_ranking():
    items = _sample_items()
    # LLM wants order 002 > 003 > 001
    llm_response = json.dumps(
        {
            "rankings": [
                {"id": "mem_002", "score": 0.95},
                {"id": "mem_003", "score": 0.80},
                {"id": "mem_001", "score": 0.10},
            ]
        }
    )
    fake = _capture_llm(return_value=llm_response)
    reranker = LLMReranker(fake)
    ranked = reranker.rerank("query about Berlin cats", items)
    assert [r.candidate["id"] for r in ranked] == ["mem_002", "mem_003", "mem_001"]
    assert [r.score for r in ranked] == [0.95, 0.80, 0.10]


# ---------------------------------------------------------------------------
# 2. Candidate objects remain unchanged.
# ---------------------------------------------------------------------------


def test_candidate_objects_remain_unchanged():
    items = _sample_items()
    originals = copy.deepcopy(items)
    llm_response = json.dumps(
        {"rankings": [{"id": "mem_002", "score": 0.9}, {"id": "mem_001", "score": 0.5}]}
    )
    fake = _capture_llm(return_value=llm_response)
    reranker = LLMReranker(fake)
    ranked = reranker.rerank("dark mode", items)
    # Original dicts unchanged field-by-field
    for orig, current in zip(originals, items):
        assert current == orig
        assert current["id"] == orig["id"]
        assert current["text"] == orig["text"]
        assert current["timestamp"] == orig["timestamp"]
        assert current["source"] == orig["source"]
        assert current["embedding"] == orig["embedding"]
        assert "score" not in current
    # Ranked candidates reference same objects (where practical)
    # At least the objects with IDs should be identical references
    id_to_original = {it["id"]: it for it in items}
    for r in ranked:
        assert r.candidate is id_to_original[r.candidate["id"]]


# ---------------------------------------------------------------------------
# 3. Scores are returned separately from CandidateMemoryItem.
# ---------------------------------------------------------------------------


def test_scores_returned_separately():
    items = _sample_items()
    llm_response = json.dumps({"rankings": [{"id": "mem_001", "score": 0.99}]})
    fake = _capture_llm(return_value=llm_response)
    reranker = LLMReranker(fake)
    ranked = reranker.rerank("dark mode", items)
    assert len(ranked) == len(items)
    for rc in ranked:
        assert isinstance(rc, RankedCandidate)
        assert isinstance(rc.score, float)
        assert isinstance(rc.candidate, dict)
        # score lives on RankedCandidate, not inside candidate dict
        assert "score" not in rc.candidate
    # The ranked candidate's score is accessible separately
    assert ranked[0].candidate["id"] == "mem_001"
    assert ranked[0].score == 0.99


# ---------------------------------------------------------------------------
# 4. Unknown IDs are ignored.
# ---------------------------------------------------------------------------


def test_unknown_ids_are_ignored():
    items = _sample_items()
    llm_response = json.dumps(
        {
            "rankings": [
                {"id": "mem_999", "score": 0.99},  # unknown
                {"id": "mem_002", "score": 0.8},
                {"id": "unknown_id", "score": 0.7},
                {"id": "mem_001", "score": 0.6},
            ]
        }
    )
    fake = _capture_llm(return_value=llm_response)
    reranker = LLMReranker(fake)
    ranked = reranker.rerank("test query", items)
    ids = [r.candidate["id"] for r in ranked]
    assert "mem_999" not in ids
    assert "unknown_id" not in ids
    assert set(ids) == {"mem_001", "mem_002", "mem_003"}
    # Valid known IDs respected in order
    assert ids[0] == "mem_002"
    assert ids[1] == "mem_001"
    # No duplicate, no invented candidate
    assert len(ids) == 3


# ---------------------------------------------------------------------------
# 5. Duplicate IDs do not duplicate candidates.
# ---------------------------------------------------------------------------


def test_duplicate_ids_do_not_duplicate_candidates():
    items = _sample_items()
    llm_response = json.dumps(
        {
            "rankings": [
                {"id": "mem_002", "score": 0.9},
                {"id": "mem_002", "score": 0.8},  # duplicate
                {"id": "mem_001", "score": 0.7},
                {"id": "mem_001", "score": 0.6},  # duplicate
            ]
        }
    )
    fake = _capture_llm(return_value=llm_response)
    reranker = LLMReranker(fake)
    ranked = reranker.rerank("query", items)
    ids = [r.candidate["id"] for r in ranked]
    assert ids.count("mem_002") == 1
    assert ids.count("mem_001") == 1
    assert len(ids) == len(set(ids))
    assert len(ids) == 3  # all candidates present exactly once
    # First occurrence wins
    assert ids[0] == "mem_002"
    assert ids[1] == "mem_001"


# ---------------------------------------------------------------------------
# 6. Missing candidates are handled safely (partial ranking preserved).
# ---------------------------------------------------------------------------


def test_missing_candidates_are_handled_safely():
    items = _sample_items()
    # LLM only ranks mem_003, missing mem_001 and mem_002
    llm_response = json.dumps({"rankings": [{"id": "mem_003", "score": 0.95}]})
    fake = _capture_llm(return_value=llm_response)
    reranker = LLMReranker(fake)
    ranked = reranker.rerank("cat Momo", items)
    ids = [r.candidate["id"] for r in ranked]
    # mem_003 must be first (LLM-ranked), remaining via fallback must still appear
    assert ids[0] == "mem_003"
    assert set(ids) == {"mem_001", "mem_002", "mem_003"}
    assert len(ids) == 3
    # Remaining two should not disappear
    assert "mem_001" in ids
    assert "mem_002" in ids


# ---------------------------------------------------------------------------
# 7. Invalid JSON triggers deterministic fallback.
# ---------------------------------------------------------------------------


def test_invalid_json_triggers_fallback():
    items = _sample_items()
    fake = _capture_llm(return_value="this is not json { broken")
    reranker = LLMReranker(fake)
    ranked = reranker.rerank("dark mode", items)
    # Fallback uses lexical overlap: "dark mode" overlaps only mem_001
    # So mem_001 should be first
    assert ranked[0].candidate["id"] == "mem_001"
    assert ranked[0].score > 0
    # All candidates still present
    assert len(ranked) == 3
    assert set(r.candidate["id"] for r in ranked) == {"mem_001", "mem_002", "mem_003"}
    # Deterministic: second call same result
    ranked2 = reranker.rerank("dark mode", items)
    assert [r.candidate["id"] for r in ranked] == [r.candidate["id"] for r in ranked2]
    assert [r.score for r in ranked] == [r.score for r in ranked2]


# ---------------------------------------------------------------------------
# 8. Markdown-fenced JSON is handled.
# ---------------------------------------------------------------------------


def test_markdown_fenced_json_is_handled():
    items = _sample_items()
    llm_response = "```json\n" + json.dumps(
        {
            "rankings": [
                {"id": "mem_003", "score": 0.91},
                {"id": "mem_001", "score": 0.22},
            ]
        }
    ) + "\n```"
    fake = _capture_llm(return_value=llm_response)
    reranker = LLMReranker(fake)
    ranked = reranker.rerank("cat query", items)
    ids = [r.candidate["id"] for r in ranked]
    assert ids[0] == "mem_003"
    assert ids[1] == "mem_001"
    # mem_002 not ranked -> preserved via fallback
    assert "mem_002" in ids
    assert len(ids) == 3

    # Also test fence without json tag
    llm_response2 = "```\n" + json.dumps(
        {"rankings": [{"id": "mem_002", "score": 0.88}]}
    ) + "\n```"
    fake2 = _capture_llm(return_value=llm_response2)
    reranker2 = LLMReranker(fake2)
    ranked2 = reranker2.rerank("query", items)
    assert ranked2[0].candidate["id"] == "mem_002"


# ---------------------------------------------------------------------------
# 9. LLM exception triggers fallback.
# ---------------------------------------------------------------------------


def test_llm_exception_triggers_fallback():
    items = _sample_items()
    fake = _capture_llm(side_effect=RuntimeError("LLM down"))
    reranker = LLMReranker(fake)
    ranked = reranker.rerank("dark mode", items)
    # Should not crash, should fallback
    assert len(ranked) == 3
    assert ranked[0].candidate["id"] == "mem_001"  # lexical best match
    # Deterministic fallback still works
    ranked2 = reranker.rerank("dark mode", items)
    assert [r.candidate["id"] for r in ranked] == [r.candidate["id"] for r in ranked2]


# ---------------------------------------------------------------------------
# 10. Empty LLM response triggers fallback.
# ---------------------------------------------------------------------------


def test_empty_llm_response_triggers_fallback():
    items = _sample_items()
    for empty_val in ["", "   ", "\n\t  \n"]:
        fake = _capture_llm(return_value=empty_val)
        reranker = LLMReranker(fake)
        ranked = reranker.rerank("Berlin", items)
        assert len(ranked) == 3
        # "Berlin" overlaps mem_002
        assert ranked[0].candidate["id"] == "mem_002"

    # None return should also fallback
    def fake_none(prompt, system=None, **kwargs):
        return None  # type: ignore[return-value]

    reranker_none = LLMReranker(fake_none)  # type: ignore[arg-type]
    ranked_none = reranker_none.rerank("Berlin", items)
    assert len(ranked_none) == 3
    assert ranked_none[0].candidate["id"] == "mem_002"


# ---------------------------------------------------------------------------
# 11. Invalid scores trigger safe handling.
# ---------------------------------------------------------------------------


def test_invalid_scores_trigger_safe_handling():
    items = _sample_items()
    llm_response = json.dumps(
        {
            "rankings": [
                {"id": "mem_001", "score": "not_a_number"},
                {"id": "mem_002", "score": None},
                {"id": "mem_003", "score": 0.85},  # only valid one
                {"id": "mem_001", "score": "also_bad"},  # duplicate but invalid
            ]
        }
    )
    fake = _capture_llm(return_value=llm_response)
    reranker = LLMReranker(fake)
    ranked = reranker.rerank("query", items)
    ids = [r.candidate["id"] for r in ranked]
    # mem_003 should be first (only valid score), others fallback
    assert ids[0] == "mem_003"
    assert ids[0] != "mem_001"  # invalid score not accepted as first
    assert set(ids) == {"mem_001", "mem_002", "mem_003"}
    assert len(ids) == 3
    # No crash, scores are floats
    for r in ranked:
        assert isinstance(r.score, float)

    # Test NaN / Inf handling
    llm_response2 = json.dumps(
        {
            "rankings": [
                {"id": "mem_001", "score": float("inf")},
                {"id": "mem_002", "score": float("nan")},
                {"id": "mem_003", "score": 0.5},
            ]
        }
    )
    # JSON dumps with inf/nan produces invalid JSON by default? Python json allows it but
    # we test via raw string that includes those tokens
    # Simulate via direct string to avoid json serialization issues
    raw_inf = '{"rankings": [{"id": "mem_001", "score": Infinity}, {"id": "mem_002", "score": NaN}, {"id": "mem_003", "score": 0.5}]}'
    fake2 = _capture_llm(return_value=raw_inf)
    reranker2 = LLMReranker(fake2)
    ranked2 = reranker2.rerank("query", items)
    # Should fallback or at least handle without crash; mem_003 may be first if parsing succeeds
    assert len(ranked2) == 3
    for r in ranked2:
        assert isinstance(r.score, float)
        assert r.score != float("inf") and str(r.score) != "nan"

    # Test bool score should be rejected
    llm_response3 = json.dumps(
        {"rankings": [{"id": "mem_001", "score": True}, {"id": "mem_002", "score": 0.7}]}
    )
    fake3 = _capture_llm(return_value=llm_response3)
    reranker3 = LLMReranker(fake3)
    ranked3 = reranker3.rerank("query", items)
    ids3 = [r.candidate["id"] for r in ranked3]
    # mem_001 with bool score should not be accepted as ranked first
    assert ids3[0] == "mem_002"


# ---------------------------------------------------------------------------
# 12. Fallback lexical ranking is deterministic.
# ---------------------------------------------------------------------------


def test_fallback_lexical_ranking_is_deterministic():
    items = _sample_items()
    fake = _capture_llm(return_value="invalid json triggers fallback")
    reranker = LLMReranker(fake)
    first = reranker.rerank("dark mode Berlin", items)
    second = reranker.rerank("dark mode Berlin", items)
    assert [r.candidate["id"] for r in first] == [r.candidate["id"] for r in second]
    assert [r.score for r in first] == [r.score for r in second]
    # Also deterministic across different reranker instances
    reranker2 = LLMReranker(_capture_llm(return_value="bad"))
    third = reranker2.rerank("dark mode Berlin", items)
    assert [r.candidate["id"] for r in first] == [r.candidate["id"] for r in third]


# ---------------------------------------------------------------------------
# 13. Fallback ranking is descending by lexical relevance.
# ---------------------------------------------------------------------------


def test_fallback_ranking_is_descending_by_lexical_relevance():
    items = [
        {"id": "a", "text": "User prefers dark mode", "timestamp": "2026-01-01T00:00:00Z", "source": "s1", "embedding": None},
        {"id": "b", "text": "User lives in Berlin", "timestamp": "2026-01-02T00:00:00Z", "source": "s2", "embedding": None},
        {"id": "c", "text": "dark mode Berlin cat", "timestamp": "2026-01-03T00:00:00Z", "source": "s3", "embedding": None},
    ]
    # Query shares tokens with a (2 tokens) and b (1 token) and c (2 tokens)
    # But Jaccard: query tokens = {dark, mode, berlin}
    # a tokens = {user, prefers, dark, mode} -> intersection 2, union 5 -> 0.4
    # b tokens = {user, lives, in, berlin} -> intersection 1, union 6 -> 0.166
    # c tokens = {dark, mode, berlin, cat} -> intersection 3, union 4 -> 0.75  highest
    fake = _capture_llm(return_value="invalid json")
    reranker = LLMReranker(fake)
    ranked = reranker.rerank("dark mode Berlin", items)
    ids = [r.candidate["id"] for r in ranked]
    scores = [r.score for r in ranked]
    assert ids[0] == "c"  # highest overlap
    assert ids[1] == "a"
    assert ids[2] == "b"
    # Scores strictly descending
    assert scores[0] > scores[1] > scores[2]
    # Scores are floats in [0,1]
    for s in scores:
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# 14. Equal fallback scores preserve original candidate order.
# ---------------------------------------------------------------------------


def test_equal_fallback_scores_preserve_original_order():
    items = [
        {"id": "x", "text": "completely unrelated apple", "timestamp": "2026-01-01T00:00:00Z", "source": "s1", "embedding": None},
        {"id": "y", "text": "totally different banana", "timestamp": "2026-01-02T00:00:00Z", "source": "s2", "embedding": None},
        {"id": "z", "text": "nothing overlapping cherry", "timestamp": "2026-01-03T00:00:00Z", "source": "s3", "embedding": None},
    ]
    fake = _capture_llm(return_value="bad json")
    reranker = LLMReranker(fake)
    ranked = reranker.rerank("query with no overlap whatsoever", items)
    # All scores zero, so original order preserved
    assert [r.candidate["id"] for r in ranked] == ["x", "y", "z"]
    assert all(r.score == 0.0 for r in ranked)

    # Also test case where two have equal non-zero overlap
    items2 = [
        {"id": "p", "text": "hello world", "timestamp": "2026-01-01T00:00:00Z", "source": "s1", "embedding": None},
        {"id": "q", "text": "hello world", "timestamp": "2026-01-02T00:00:00Z", "source": "s2", "embedding": None},
        {"id": "r", "text": "hello", "timestamp": "2026-01-03T00:00:00Z", "source": "s3", "embedding": None},
    ]
    ranked2 = reranker.rerank("hello world", items2)
    # p and q have identical score (1.0), should preserve p before q
    assert ranked2[0].candidate["id"] == "p"
    assert ranked2[1].candidate["id"] == "q"
    assert ranked2[0].score == ranked2[1].score


# ---------------------------------------------------------------------------
# 15. Empty candidate list is handled safely.
# ---------------------------------------------------------------------------


def test_empty_candidate_list_handled_safely():
    fake = _capture_llm(return_value=json.dumps({"rankings": [{"id": "any", "score": 0.9}]}))
    reranker = LLMReranker(fake)
    assert reranker.rerank("any query", []) == []
    # Also with fallback path
    fake2 = _capture_llm(return_value="bad")
    reranker2 = LLMReranker(fake2)
    assert reranker2.rerank("any query", []) == []
    # LLM should not be called for empty candidates? Current impl returns early without calling LLM.
    # But even if it were called, empty result is correct. We verify no crash and empty list.
    assert isinstance(reranker.rerank("q", []), list)


# ---------------------------------------------------------------------------
# 16. The LLM receives the original query.
# ---------------------------------------------------------------------------


def test_llm_receives_original_query():
    items = _sample_items()
    query = "What is the user's cat name?"
    llm_response = json.dumps({"rankings": [{"id": "mem_003", "score": 0.9}]})
    fake = _capture_llm(return_value=llm_response)
    reranker = LLMReranker(fake)
    reranker.rerank(query, items)
    assert "prompt" in fake.capture
    assert query in fake.capture["prompt"]
    # Also test with special characters
    query2 = "User's preference: dark-mode? (Berlin)!"
    fake2 = _capture_llm(return_value=llm_response)
    reranker2 = LLMReranker(fake2)
    reranker2.rerank(query2, items)
    assert query2 in fake2.capture["prompt"]


# ---------------------------------------------------------------------------
# 17. The LLM prompt contains candidate IDs and candidate text.
# ---------------------------------------------------------------------------


def test_llm_prompt_contains_candidate_ids_and_text():
    items = _sample_items()
    fake = _capture_llm(return_value=json.dumps({"rankings": []}))
    reranker = LLMReranker(fake)
    reranker.rerank("test query", items)
    prompt = fake.capture["prompt"]
    for item in items:
        assert item["id"] in prompt, f"candidate id {item['id']} missing from prompt"
        assert item["text"] in prompt, f"candidate text {item['text']!r} missing from prompt"
        assert item["timestamp"] in prompt
        assert item["source"] in prompt


# ---------------------------------------------------------------------------
# 18. The reranker does not import or instantiate a concrete LLM provider.
# ---------------------------------------------------------------------------


def test_reranker_does_not_import_concrete_llm_provider():
    reranker_path = pathlib.Path(__file__).parent.parent / "src" / "retrieval" / "reranker.py"
    source = reranker_path.read_text(encoding="utf-8")
    source_lower = source.lower()
    # Forbid actual import statements for concrete providers
    assert not re.search(r"^\s*import\s+openai", source_lower, re.MULTILINE), "reranker.py must not import openai"
    assert not re.search(r"^\s*from\s+openai", source_lower, re.MULTILINE)
    assert not re.search(r"^\s*import\s+groq", source_lower, re.MULTILINE)
    assert not re.search(r"^\s*from\s+groq", source_lower, re.MULTILINE)
    assert not re.search(r"^\s*import\s+google\.generativeai", source_lower, re.MULTILINE)
    assert not re.search(r"^\s*from\s+google", source_lower, re.MULTILINE)
    # Forbid importing src.llm client concretely
    assert not re.search(r"from\s+src\.llm", source), "reranker.py must not import from src.llm"
    assert not re.search(r"import\s+src\.llm", source), "reranker.py must not import src.llm"
    # Forbid instantiating known client classes
    assert "Groq(" not in source
    assert "OpenAI(" not in source
    assert "GenerativeModel" not in source


# ---------------------------------------------------------------------------
# 19. The reranker does not import ChromaDB.
# ---------------------------------------------------------------------------


def test_reranker_does_not_import_chromadb():
    reranker_path = pathlib.Path(__file__).parent.parent / "src" / "retrieval" / "reranker.py"
    source = reranker_path.read_text(encoding="utf-8")
    source_lower = source.lower()
    assert not re.search(r"^\s*import\s+chromadb", source_lower, re.MULTILINE), "reranker.py must not import ChromaDB"
    assert not re.search(r"^\s*from\s+chromadb", source_lower, re.MULTILINE)
    # Also forbid dynamic imports
    assert "__import__('chromadb" not in source_lower
    assert '__import__("chromadb' not in source_lower
    assert "importlib" not in source_lower or "chromadb" not in source_lower


# ---------------------------------------------------------------------------
# 20. All supplied candidates appear at most once in the final output.
# ---------------------------------------------------------------------------


def test_all_supplied_candidates_appear_at_most_once():
    items = _sample_items()
    scenarios = [
        json.dumps({"rankings": [{"id": "mem_001", "score": 0.9}]}),  # partial
        json.dumps({"rankings": [{"id": "mem_002", "score": 0.9}, {"id": "mem_002", "score": 0.8}]}),  # duplicate
        json.dumps({"rankings": [{"id": "unknown", "score": 0.9}, {"id": "mem_001", "score": 0.5}]}),  # unknown
        "not json at all",  # fallback
        "",  # empty -> fallback
        json.dumps({"rankings": [{"id": "mem_001", "score": "bad"}, {"id": "mem_002", "score": 0.7}]}),  # invalid score
    ]
    for resp in scenarios:
        # Need fresh fake for each scenario
        if resp == "":
            def fake_empty(prompt, system=None, **kwargs):
                return ""
            reranker = LLMReranker(fake_empty)
        elif resp == "not json at all":
            fake = _capture_llm(return_value=resp)
            reranker = LLMReranker(fake)
        else:
            fake = _capture_llm(return_value=resp)
            reranker = LLMReranker(fake)
        ranked = reranker.rerank("test query with cat dark mode", items)
        ids = [r.candidate["id"] for r in ranked]
        # At most once
        assert len(ids) == len(set(ids)), f"duplicate ids in scenario {resp!r}: {ids}"
        # Never contains non-supplied
        for cid in ids:
            assert cid in {it["id"] for it in items}, f"invented id {cid} in {resp!r}"
        # All scenarios should either contain all 3 (via fallback preservation) or subset if fallback not triggered?
        # Spec says candidates not returned should NOT disappear (partial ranking) -> should contain all 3
        # For valid or partial valid, we preserve remainder, so should be 3. For fallback, also 3.
        assert len(ids) == 3, f"expected 3 candidates in scenario {resp!r}, got {ids}"

    # Also test with valid full ranking -> still at most once and exactly 3
    fake_full = _capture_llm(
        return_value=json.dumps(
            {
                "rankings": [
                    {"id": "mem_003", "score": 0.9},
                    {"id": "mem_002", "score": 0.8},
                    {"id": "mem_001", "score": 0.7},
                ]
            }
        )
    )
    reranker_full = LLMReranker(fake_full)
    ranked_full = reranker_full.rerank("query", items)
    assert len(ranked_full) == 3
    assert len(set(r.candidate["id"] for r in ranked_full)) == 3


# ---------------------------------------------------------------------------
# Additional integration: reranker accepts CoarseRetriever output directly
# ---------------------------------------------------------------------------


def test_reranker_accepts_coarse_retriever_output():
    """Ensure composition: CoarseRetriever.retrieve -> LLMReranker.rerank works."""

    from src.retrieval.retriever import CoarseRetriever

    class FakeStore:
        def __init__(self, items):
            self._items = items

        def query(self, text: str, k: int, tier: str):
            return self._items[:k]

    items = _sample_items()
    store = FakeStore(items)
    coarse = CoarseRetriever(store)
    candidates = coarse.retrieve("hello", k=2, tier="all")
    assert len(candidates) == 2

    fake = _capture_llm(
        return_value=json.dumps({"rankings": [{"id": "mem_002", "score": 0.95}, {"id": "mem_001", "score": 0.5}]})
    )
    reranker = LLMReranker(fake)
    ranked = reranker.rerank("hello", candidates)
    assert len(ranked) == 2
    assert ranked[0].candidate["id"] == "mem_002"
