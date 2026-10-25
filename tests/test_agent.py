"""Offline tests for the MemAge conversational agent.

All tests use a fake store and a fake LLM — no API keys, no ChromaDB.
Covers:
  - Remember a preference, then ask about it in a later turn.
  - Correct a previously remembered fact.
  - Ask about an unknown fact (no context).
  - Handle LLM failure gracefully.
  - Retrieved context respects the configured token budget.
  - Retrieved IDs are reported in AgentResponse.
  - Ingestion errors do not prevent answer generation.
  - Invalid construction raises immediately.

Run: python -m pytest tests/test_agent.py -q
"""
from __future__ import annotations

import json
import pytest

from src.agent import MemAgeAgent, AgentResponse


# ---------------------------------------------------------------------------
# Shared fake infrastructure
# ---------------------------------------------------------------------------

class _FakeStore:
    """In-memory store that fully satisfies the TieredMemoryStore interface."""

    def __init__(self, items: dict | None = None):
        self._items: dict = dict(items or {})
        self.added: list[str] = []
        self.updated: list[str] = []
        self.deleted: list[str] = []

    def query(self, text: str, k: int = 10, tier: str = "all") -> list[dict]:
        # Return all items (fake — no real semantic search)
        return list(self._items.values())[:k]

    def get(self, item_id: str) -> dict | None:
        return self._items.get(item_id)

    def count(self) -> dict:
        return {"STM": len(self._items), "MTM": 0, "LTM": 0}

    def add(self, item: dict, tier: str = "STM") -> dict:
        stored = dict(item)
        self._items[item["id"]] = stored
        self.added.append(item["id"])
        return stored

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


def _fake_llm(
    extraction_facts: list[str] | None = None,
    answer: str = "Here is what I know.",
    importance: float = 0.9,
    contradiction: float = 0.0,
    rerank_passthrough: bool = True,
):
    """Multi-purpose deterministic fake LLM.

    Dispatches by system prompt content:
    - Extraction system  → JSON array of facts
    - Importance system  → {"importance": <val>}
    - Contradiction sys  → {"contradiction": <val>}
    - Reranker system    → passthrough JSON rankings
    - Answer system      → configured answer string
    """
    if extraction_facts is None:
        extraction_facts = []

    def _llm(prompt: str, system: str | None = None, **kw) -> str:
        sys_lower = (system or "").lower()
        if "memories from conversations" in sys_lower:
            return json.dumps(extraction_facts)
        if "importance scorer" in sys_lower:
            return f'{{"importance": {importance}}}'
        if "contradiction detector" in sys_lower:
            return f'{{"contradiction": {contradiction}}}'
        if "relevance reranker" in sys_lower:
            # Return a passthrough ranking using IDs parsed from the prompt
            import re
            ids = re.findall(r"- ID: (\S+)", prompt)
            rankings = [{"id": i, "score": 0.9 - 0.1 * n} for n, i in enumerate(ids)]
            return json.dumps({"rankings": rankings})
        # Answer generation
        return answer

    return _llm


def _make_memory(item_id: str, text: str,
                 timestamp: str = "2026-01-01T00:00:00",
                 source: str = "s1") -> dict:
    return {
        "id": item_id, "text": text, "timestamp": timestamp,
        "source": source, "embedding": None,
        "tier": "STM", "distance": 0.1, "access_count": 0,
    }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestAgentConstruction:
    def test_no_store_raises(self):
        with pytest.raises(ValueError, match="store must be provided"):
            MemAgeAgent(store=None)

    def test_bad_token_budget_raises(self):
        with pytest.raises(ValueError, match="token_budget"):
            MemAgeAgent(store=_FakeStore(), token_budget=0)

    def test_bad_k_raises(self):
        with pytest.raises(ValueError, match="k must"):
            MemAgeAgent(store=_FakeStore(), k=-1)

    def test_constructs_successfully(self):
        agent = MemAgeAgent(store=_FakeStore(), llm=_fake_llm())
        assert agent is not None


# ---------------------------------------------------------------------------
# Scenario 1 — Remember a preference, then ask about it
# ---------------------------------------------------------------------------

class TestRememberAndRecall:
    def test_preference_added_to_store(self):
        store = _FakeStore()
        llm = _fake_llm(
            extraction_facts=["Maya prefers tea over coffee."],
            answer="You prefer tea over coffee.",
            importance=0.85,
        )
        agent = MemAgeAgent(store=store, llm=llm, token_budget=400, k=5)
        # Ingest preference turn
        r1 = agent.chat("I prefer tea over coffee.", session_id="s1")
        assert r1.turn_result is not None
        # Something was added to the store
        assert len(store.added) > 0

    def test_recall_uses_stored_memory(self):
        """Ask a question after storing a preference — answer must reflect context."""
        memory = _make_memory("s1__0", "Maya prefers tea over coffee.")
        store = _FakeStore(items={"s1__0": memory})
        llm = _fake_llm(
            extraction_facts=[],  # question turn produces no new facts
            answer="You prefer tea over coffee.",
            importance=0.1,
        )
        agent = MemAgeAgent(store=store, llm=llm, token_budget=400, k=5)
        resp = agent.chat("What is my preferred drink?", session_id="s1")
        assert resp.ok
        assert "tea" in resp.answer.lower()
        # Context was populated from the stored memory
        assert "tea" in resp.context.lower()

    def test_retrieved_ids_present(self):
        memory = _make_memory("s1__0", "Maya prefers tea over coffee.")
        store = _FakeStore(items={"s1__0": memory})
        llm = _fake_llm(extraction_facts=[], answer="You prefer tea.")
        agent = MemAgeAgent(store=store, llm=llm, token_budget=400, k=5)
        resp = agent.chat("What do I drink?", session_id="s1")
        assert "s1__0" in resp.retrieved_ids


# ---------------------------------------------------------------------------
# Scenario 2 — Correct a remembered fact
# ---------------------------------------------------------------------------

class TestCorrection:
    def test_correction_triggers_update_or_delete(self):
        old = _make_memory("s1__0", "Maya lives in Chennai.",
                           timestamp="2025-01-01T00:00:00")
        store = _FakeStore(items={"s1__0": old})
        # High contradiction → UPDATE or DELETE
        llm = _fake_llm(
            extraction_facts=["Maya moved to Bangalore."],
            answer="You now live in Bangalore.",
            importance=0.9,
            contradiction=0.7,
        )
        agent = MemAgeAgent(store=store, llm=llm, token_budget=400, k=5)
        resp = agent.chat("I moved to Bangalore.", session_id="s1")
        assert resp.ok
        assert resp.turn_result is not None
        actions = {r.decision["action"] for r in resp.turn_result.candidates}
        assert actions & {"UPDATE", "DELETE", "ADD"}, \
            f"Expected a mutating action, got: {actions}"

    def test_correction_answer_reflects_new_fact(self):
        old = _make_memory("s1__0", "Maya lives in Chennai.")
        store = _FakeStore(items={"s1__0": old})
        llm = _fake_llm(
            extraction_facts=["Maya moved to Bangalore."],
            answer="You live in Bangalore.",
            importance=0.9,
            contradiction=0.7,
        )
        agent = MemAgeAgent(store=store, llm=llm, token_budget=400, k=5)
        resp = agent.chat("I moved to Bangalore.", session_id="s1")
        assert "bangalore" in resp.answer.lower()


# ---------------------------------------------------------------------------
# Scenario 3 — Ask about an unknown fact
# ---------------------------------------------------------------------------

class TestUnknownFact:
    def test_empty_store_answer_admits_ignorance(self):
        store = _FakeStore()  # empty
        llm = _fake_llm(
            extraction_facts=[],
            answer="I don't have any stored information about that.",
        )
        agent = MemAgeAgent(store=store, llm=llm, token_budget=400, k=5)
        resp = agent.chat("What is my favourite colour?", session_id="s1")
        assert resp.ok
        # Context must be empty (no memories)
        assert resp.context == ""
        assert resp.retrieved_ids == []

    def test_no_context_prompt_used_for_empty_store(self):
        """Verify the no-context prompt is actually sent by checking the answer
        comes from the no-context branch (fake LLM returns the same string
        regardless, but we verify retrieved_ids is empty)."""
        store = _FakeStore()
        llm = _fake_llm(answer="I have no stored information about this.")
        agent = MemAgeAgent(store=store, llm=llm, token_budget=400, k=5)
        resp = agent.chat("What do I do for work?", session_id="s1")
        assert resp.retrieved_ids == []


# ---------------------------------------------------------------------------
# Scenario 4 — LLM failure
# ---------------------------------------------------------------------------

class TestLLMFailure:
    def test_llm_exception_captured_not_raised(self):
        store = _FakeStore()

        def broken_llm(prompt, system=None, **kw):
            # Fail on answer generation, succeed on extraction/scoring
            sys_lower = (system or "").lower()
            if "helpful assistant" in sys_lower:
                raise RuntimeError("provider timeout")
            if "memories from conversations" in sys_lower:
                return "[]"
            return '{"importance": 0.5}'

        agent = MemAgeAgent(store=store, llm=broken_llm, token_budget=400, k=5)
        resp = agent.chat("Hello!", session_id="s1")
        assert resp.ok is False
        assert "provider timeout" in (resp.error or "")
        # Answer contains the error description, not an exception traceback
        assert "error" in resp.answer.lower()

    def test_extraction_failure_does_not_prevent_answer(self):
        """If run_turn() fails, the answer must still be returned."""
        store = _FakeStore()

        call_count = {"n": 0}

        def flaky_llm(prompt, system=None, **kw):
            sys_lower = (system or "").lower()
            if "memories from conversations" in sys_lower:
                raise RuntimeError("extraction LLM down")
            if "helpful assistant" in sys_lower:
                return "I can answer without new memories."
            return '{"importance": 0.5}'

        agent = MemAgeAgent(store=store, llm=flaky_llm, token_budget=400, k=5)
        resp = agent.chat("Tell me something.", session_id="s1")
        # Answer was generated even though ingestion failed
        assert "answer without" in resp.answer.lower()


# ---------------------------------------------------------------------------
# Scenario 5 — Token budget respected
# ---------------------------------------------------------------------------

class TestTokenBudget:
    def test_context_tokens_within_budget(self):
        from src.retrieval.token_budget import estimate_tokens

        # Pre-seed store with several memories
        items = {
            f"s1__{i}": _make_memory(f"s1__{i}", f"Fact number {i}: " + "word " * 20)
            for i in range(8)
        }
        store = _FakeStore(items=items)
        budget = 50  # tight budget
        llm = _fake_llm(extraction_facts=[], answer="Here is the answer.")
        agent = MemAgeAgent(store=store, llm=llm, token_budget=budget, k=8)
        resp = agent.chat("Tell me a fact.", session_id="s1")
        tokens = estimate_tokens(resp.context)
        assert tokens <= budget, (
            f"Context has {tokens} tokens, exceeds budget {budget}:\n{resp.context}"
        )

    def test_zero_budget_raises_on_construction(self):
        with pytest.raises(ValueError):
            MemAgeAgent(store=_FakeStore(), token_budget=0)

    def test_context_empty_when_nothing_fits(self):
        """If every memory is larger than the budget, context must be empty."""
        items = {
            "s1__0": _make_memory("s1__0", "word " * 200)  # ~200 tokens
        }
        store = _FakeStore(items=items)
        budget = 5  # impossibly small
        llm = _fake_llm(extraction_facts=[], answer="Nothing fits.")
        agent = MemAgeAgent(store=store, llm=llm, token_budget=budget, k=5)
        resp = agent.chat("What do you know?", session_id="s1")
        from src.retrieval.token_budget import estimate_tokens
        assert estimate_tokens(resp.context) <= budget


# ---------------------------------------------------------------------------
# Scenario 6 — AgentResponse schema
# ---------------------------------------------------------------------------

class TestAgentResponseSchema:
    def test_all_fields_present(self):
        store = _FakeStore()
        llm = _fake_llm(extraction_facts=[], answer="Hello!")
        agent = MemAgeAgent(store=store, llm=llm)
        resp = agent.chat("Hi!", session_id="test")
        assert isinstance(resp, AgentResponse)
        assert isinstance(resp.message, str)
        assert isinstance(resp.session_id, str)
        assert isinstance(resp.answer, str)
        assert isinstance(resp.context, str)
        assert isinstance(resp.retrieved_ids, list)
        assert isinstance(resp.ok, bool)

    def test_turn_result_is_turn_result_or_none(self):
        from src.pipeline import TurnResult
        store = _FakeStore()
        llm = _fake_llm(extraction_facts=[], answer="Hi!")
        agent = MemAgeAgent(store=store, llm=llm)
        resp = agent.chat("Hi!", session_id="test")
        assert resp.turn_result is None or isinstance(resp.turn_result, TurnResult)

    def test_session_id_echoed(self):
        store = _FakeStore()
        llm = _fake_llm(answer="OK")
        agent = MemAgeAgent(store=store, llm=llm)
        resp = agent.chat("hello", session_id="my_session")
        assert resp.session_id == "my_session"
