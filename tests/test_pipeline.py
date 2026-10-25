"""Integration tests for src/pipeline.py.

All tests use a fake store and a fake LLM — no API keys or ChromaDB needed.
Covers:
  - A new fact is added.
  - A changed fact updates the existing memory.
  - Small talk creates no memory.
  - Empty existing matches are handled correctly.
  - A failed store operation is reported accurately and candidate is preserved.
  - A repeated fact does not cause an unnecessary duplicate (NOOP via policy).

Run: python -m pytest tests/test_pipeline.py -q
"""
from __future__ import annotations

import pytest
from src.pipeline import run_turn, TurnResult, CandidateResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fake_llm(extraction_facts: list[str], importance: float = 0.9,
              contradiction: float = 0.0):
    """Return a deterministic fake LLM.

    - Extraction calls (system contains "memories from conversations") get a
      JSON array of the supplied facts.
    - Importance scoring calls return ``{"importance": <importance>}``.
    - Contradiction calls return ``{"contradiction": <contradiction>}``.
    """
    import json

    def _llm(prompt: str, system: str | None = None, **kw) -> str:
        sys = (system or "").lower()
        if "memories from conversations" in sys:
            return json.dumps(extraction_facts)
        if "importance" in sys:
            return f'{{"importance": {importance}}}'
        return f'{{"contradiction": {contradiction}}}'

    return _llm


class _FakeStore:
    """Minimal injectable store that records all operations."""

    def __init__(self, items: dict | None = None,
                 add_raises: Exception | None = None):
        self._items: dict = dict(items or {})
        self._add_raises = add_raises
        self.added: list[str] = []
        self.updated: list[str] = []
        self.deleted: list[str] = []

    # --- read ---
    def query(self, text: str, k: int = 5, tier: str = "all") -> list[dict]:
        return list(self._items.values())

    def get(self, item_id: str) -> dict | None:
        return self._items.get(item_id)

    def count(self) -> dict:
        return {"STM": len(self._items), "MTM": 0, "LTM": 0}

    # --- write ---
    def add(self, item: dict, tier: str = "STM") -> dict:
        if self._add_raises is not None:
            raise self._add_raises
        self._items[item["id"]] = dict(item)
        self.added.append(item["id"])
        return dict(item)

    def update(self, item_id: str, new_text: str, timestamp: str | None = None) -> bool:
        if item_id in self._items:
            self._items[item_id]["text"] = new_text
        self.updated.append(item_id)
        return item_id in self._items

    def delete(self, item_id: str) -> bool:
        existed = item_id in self._items
        self._items.pop(item_id, None)
        self.deleted.append(item_id)
        return existed

    stm_capacity = 20
    mtm_capacity = 100


META = {"speaker": "Maya", "timestamp": "2026-01-01T00:00:00", "source": "D1:3"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunTurnNewFact:
    """A completely new fact should be added to the store."""

    def test_returns_turn_result(self):
        store = _FakeStore()
        llm = _fake_llm(["Maya lives in Chennai."], importance=0.9, contradiction=0.0)
        result = run_turn("I live in Chennai.", META, store=store, llm=llm)
        assert isinstance(result, TurnResult)

    def test_new_fact_is_added(self):
        store = _FakeStore()
        llm = _fake_llm(["Maya lives in Chennai."], importance=0.9, contradiction=0.0)
        result = run_turn("I live in Chennai.", META, store=store, llm=llm)
        assert len(result.candidates) == 1
        r = result.candidates[0]
        assert r.decision["action"] == "ADD"
        assert r.ok is True
        assert r.error is None
        assert "D1:3__0" in store.added

    def test_added_ids_property(self):
        store = _FakeStore()
        llm = _fake_llm(["Maya lives in Chennai."], importance=0.9, contradiction=0.0)
        result = run_turn("I live in Chennai.", META, store=store, llm=llm)
        assert "D1:3__0" in result.added

    def test_candidate_dict_matches_interface(self):
        """Candidate must contain the five fields from docs/interfaces.md."""
        store = _FakeStore()
        llm = _fake_llm(["Maya lives in Chennai."])
        result = run_turn("I live in Chennai.", META, store=store, llm=llm)
        c = result.candidates[0].candidate
        for key in ("id", "text", "timestamp", "source", "embedding"):
            assert key in c, f"missing key: {key}"
        assert c["embedding"] is None  # store fills this in, not extraction


class TestRunTurnUpdate:
    """A contradicting fact should update the closest existing memory."""

    def test_changed_fact_updates_store(self):
        existing = {
            "id": "D1:3__0", "text": "Maya lives in Mumbai",
            "timestamp": "2025-01-01T00:00:00", "source": "D1:3",
            "embedding": None, "tier": "STM", "distance": 0.05, "access_count": 0,
        }
        store = _FakeStore(items={"D1:3__0": existing})
        # Moderate contradiction (0.6) -> UPDATE
        llm = _fake_llm(["Maya lives in Bangalore."], importance=0.9, contradiction=0.6)
        result = run_turn("I moved to Bangalore.", META, store=store, llm=llm)
        assert len(result.candidates) == 1
        r = result.candidates[0]
        assert r.decision["action"] == "UPDATE"
        assert r.ok is True
        assert "D1:3__0" in store.updated

    def test_updated_ids_property(self):
        existing = {
            "id": "D1:3__0", "text": "Maya lives in Mumbai",
            "timestamp": "2025-01-01T00:00:00", "source": "D1:3",
            "embedding": None, "tier": "STM", "distance": 0.05, "access_count": 0,
        }
        store = _FakeStore(items={"D1:3__0": existing})
        llm = _fake_llm(["Maya lives in Bangalore."], importance=0.9, contradiction=0.6)
        result = run_turn("I moved to Bangalore.", META, store=store, llm=llm)
        assert "D1:3__0" in result.updated


class TestRunTurnSmallTalk:
    """Small talk or empty turns should produce no candidates and no store calls."""

    def test_small_talk_returns_empty_candidates(self):
        store = _FakeStore()
        # Extractor returns [] for small talk (fake LLM returns empty array)
        llm = _fake_llm([])  # empty extraction
        result = run_turn("Hey, how are you?", META, store=store, llm=llm)
        assert result.candidates == []
        assert store.added == []
        assert store.updated == []
        assert store.deleted == []

    def test_empty_turn_returns_empty_candidates(self):
        store = _FakeStore()
        llm = _fake_llm([])
        result = run_turn("   ", META, store=store, llm=llm)
        assert result.candidates == []

    def test_noops_property_for_empty(self):
        store = _FakeStore()
        llm = _fake_llm([])
        result = run_turn("hi!", META, store=store, llm=llm)
        assert result.noops == []
        assert result.added == []


class TestRunTurnEmptyMatches:
    """With an empty store the pipeline should ADD without issues."""

    def test_empty_store_always_adds(self):
        store = _FakeStore()  # no existing items
        llm = _fake_llm(["Maya adopted a cat."], importance=0.85, contradiction=0.0)
        result = run_turn("I adopted a cat.", META, store=store, llm=llm)
        assert result.candidates[0].decision["action"] == "ADD"
        assert result.candidates[0].ok is True

    def test_all_four_scores_present(self):
        store = _FakeStore()
        llm = _fake_llm(["Maya adopted a cat."], importance=0.85)
        result = run_turn("I adopted a cat.", META, store=store, llm=llm)
        scores = result.candidates[0].decision["scores"]
        assert set(scores.keys()) == {
            "importance", "temporal_validity", "contradiction", "storage_pressure"
        }


class TestRunTurnFailedStore:
    """A store that raises must be reported accurately; candidate is preserved."""

    def test_add_failure_reported_as_error(self):
        store = _FakeStore(add_raises=RuntimeError("ChromaDB unavailable"))
        llm = _fake_llm(["Maya likes hiking."], importance=0.9, contradiction=0.0)
        result = run_turn("I love hiking.", META, store=store, llm=llm)
        assert len(result.candidates) == 1
        r = result.candidates[0]
        assert r.ok is False
        assert "ChromaDB unavailable" in (r.error or "")

    def test_errors_property_populated(self):
        store = _FakeStore(add_raises=RuntimeError("disk full"))
        llm = _fake_llm(["Maya likes hiking."], importance=0.9)
        result = run_turn("I love hiking.", META, store=store, llm=llm)
        assert len(result.errors) == 1
        assert "disk full" in result.errors[0]

    def test_candidate_preserved_on_failure(self):
        """Even when the store fails the candidate dict must appear in results."""
        store = _FakeStore(add_raises=RuntimeError("oops"))
        llm = _fake_llm(["Maya likes hiking."])
        result = run_turn("I love hiking.", META, store=store, llm=llm)
        assert result.candidates[0].candidate["text"] == "Maya likes hiking."

    def test_decision_schema_complete_on_failure(self):
        """Fallback decision on failure must still be a valid Decision dict."""
        store = _FakeStore(add_raises=RuntimeError("oops"))
        llm = _fake_llm(["Maya likes hiking."])
        result = run_turn("I love hiking.", META, store=store, llm=llm)
        d = result.candidates[0].decision
        assert "action" in d
        assert "target_id" in d
        assert set(d["scores"].keys()) == {
            "importance", "temporal_validity", "contradiction", "storage_pressure"
        }


class TestRunTurnNoopDeduplication:
    """A repeated identical fact should not cause a duplicate ADD.

    The policy scores it: temporal_validity=0.5 (same timestamp), contradiction=0.0
    -> no conflict -> ADD is still the policy outcome unless importance is suppressed.
    We test that the pipeline correctly reports what the policy decided.
    """

    def test_low_importance_same_text_noop(self):
        """LLM scores it as unimportant (< importance_min 0.3) -> NOOP."""
        existing = {
            "id": "D1:3__0", "text": "Maya greets the world",
            "timestamp": "2026-01-01T00:00:00", "source": "D1:3",
            "embedding": None, "tier": "STM", "distance": 0.01, "access_count": 0,
        }
        store = _FakeStore(items={"D1:3__0": existing})
        # importance below threshold -> NOOP (no duplicate added)
        llm = _fake_llm(["Maya greets the world"], importance=0.1, contradiction=0.0)
        result = run_turn("Hello world.", META, store=store, llm=llm)
        r = result.candidates[0]
        assert r.decision["action"] == "NOOP"
        assert store.added == []

    def test_noop_ids_reported(self):
        existing = {
            "id": "D1:3__0", "text": "Maya greets the world",
            "timestamp": "2026-01-01T00:00:00", "source": "D1:3",
            "embedding": None, "tier": "STM", "distance": 0.01, "access_count": 0,
        }
        store = _FakeStore(items={"D1:3__0": existing})
        llm = _fake_llm(["Maya greets the world"], importance=0.1, contradiction=0.0)
        result = run_turn("Hello world.", META, store=store, llm=llm)
        assert "D1:3__0" in result.noops


class TestRunTurnMissingStore:
    """Passing store=None must raise ValueError immediately."""

    def test_no_store_raises(self):
        with pytest.raises(ValueError, match="store must be provided"):
            run_turn("hello", META, store=None)


class TestRunTurnMultipleFacts:
    """Multiple extracted facts from one turn are all processed."""

    def test_two_facts_both_added(self):
        store = _FakeStore()
        llm = _fake_llm(
            ["Maya lives in Chennai.", "Maya works as an engineer."],
            importance=0.9, contradiction=0.0,
        )
        result = run_turn(
            "I live in Chennai and work as an engineer.",
            META, store=store, llm=llm,
        )
        assert len(result.candidates) == 2
        ids = {r.candidate["id"] for r in result.candidates}
        assert ids == {"D1:3__0", "D1:3__1"}
        assert all(r.ok for r in result.candidates)
        assert all(r.decision["action"] == "ADD" for r in result.candidates)
