"""Unit tests for the Selective Memory Controller.

All LLM calls are replaced with deterministic fakes — no API key required.
Tests cover: ADD, UPDATE, DELETE, NOOP, malformed LLM scores, empty matches,
storage-pressure suppression, temporal-validity gating.

Run: python -m pytest tests/test_controller.py -q
"""
from __future__ import annotations

import pytest

from src.controller.scoring import (
    score_importance,
    score_temporal_validity,
    score_contradiction,
    score_storage_pressure,
    _parse_score,
)
from src.controller.policy import decide as policy_decide, DEFAULT_THRESHOLDS
from src.controller.controller import decide


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _item(item_id="D1:3__0", text="User lives in Chennai",
          timestamp="2026-01-01T00:00:00", source="D1:3"):
    return {"id": item_id, "text": text, "timestamp": timestamp,
            "source": source, "embedding": None}


def _match(item_id="existing__0", text="User lives in Mumbai",
           timestamp="2025-06-01T00:00:00", tier="STM"):
    return {"id": item_id, "text": text, "timestamp": timestamp,
            "source": "D1:1", "embedding": None, "tier": tier,
            "distance": 0.1, "access_count": 0}


def _llm_importance(val: float):
    return lambda prompt, system=None, **kw: f'{{"importance": {val}}}'


def _llm_contradiction(val: float):
    return lambda prompt, system=None, **kw: f'{{"contradiction": {val}}}'


def _combined_llm(importance: float, contradiction: float):
    """Return a fake LLM that answers both importance and contradiction calls."""
    def _fake(prompt, system=None, **kw):
        if "importance" in prompt.lower() or "importance" in (system or "").lower():
            return f'{{"importance": {importance}}}'
        return f'{{"contradiction": {contradiction}}}'
    return _fake


# ---------------------------------------------------------------------------
# scoring.py  — unit tests
# ---------------------------------------------------------------------------

class TestParseScore:
    def test_valid_json(self):
        assert _parse_score('{"importance": 0.8}', "importance") == pytest.approx(0.8)

    def test_extra_whitespace_and_text(self):
        assert _parse_score('Sure thing: {"importance": 0.3} done', "importance") == pytest.approx(0.3)

    def test_missing_key_returns_none(self):
        assert _parse_score('{"other": 0.5}', "importance") is None

    def test_malformed_json_returns_none(self):
        assert _parse_score("not json at all", "importance") is None

    def test_value_clamped_above_1(self):
        assert _parse_score('{"importance": 1.5}', "importance") == 1.0

    def test_value_clamped_below_0(self):
        assert _parse_score('{"importance": -0.2}', "importance") == 0.0


class TestScoreImportance:
    def test_llm_returns_high_score(self):
        assert score_importance(_item(), _llm_importance(0.9)) == pytest.approx(0.9)

    def test_llm_returns_low_score(self):
        assert score_importance(_item(), _llm_importance(0.1)) == pytest.approx(0.1)

    def test_malformed_reply_uses_fallback(self):
        bad_llm = lambda p, system=None, **kw: "I cannot score this"
        assert score_importance(_item(), bad_llm, fallback=0.5) == 0.5

    def test_llm_exception_uses_fallback(self):
        def boom(p, system=None, **kw): raise RuntimeError("boom")
        assert score_importance(_item(), boom, fallback=0.42) == pytest.approx(0.42)


class TestScoreTemporalValidity:
    def test_no_matches_returns_1(self):
        assert score_temporal_validity(_item(), []) == 1.0

    def test_newer_candidate_returns_1(self):
        candidate = _item(timestamp="2026-06-01T00:00:00")
        match = _match(timestamp="2025-01-01T00:00:00")
        assert score_temporal_validity(candidate, [match]) == 1.0

    def test_same_timestamp_returns_half(self):
        ts = "2026-01-01T00:00:00"
        assert score_temporal_validity(_item(timestamp=ts), [_match(timestamp=ts)]) == pytest.approx(0.5)

    def test_older_candidate_returns_0(self):
        candidate = _item(timestamp="2024-01-01T00:00:00")
        match = _match(timestamp="2026-01-01T00:00:00")
        assert score_temporal_validity(candidate, [match]) == 0.0

    def test_missing_candidate_timestamp_returns_fallback(self):
        candidate = {**_item(), "timestamp": None}
        val = score_temporal_validity(candidate, [_match()], fallback=0.7)
        assert val == pytest.approx(0.7)

    def test_missing_match_timestamp_returns_fallback(self):
        match = {**_match(), "timestamp": None}
        val = score_temporal_validity(_item(), [match], fallback=0.6)
        assert val == pytest.approx(0.6)


class TestScoreContradiction:
    def test_no_matches_returns_0(self):
        llm = _llm_contradiction(0.9)
        assert score_contradiction(_item(), [], llm) == 0.0

    def test_high_contradiction(self):
        llm = _llm_contradiction(0.9)
        assert score_contradiction(_item(), [_match()], llm) == pytest.approx(0.9)

    def test_no_contradiction(self):
        llm = _llm_contradiction(0.0)
        assert score_contradiction(_item(), [_match()], llm) == pytest.approx(0.0)

    def test_malformed_reply_uses_fallback(self):
        bad_llm = lambda p, system=None, **kw: "gibberish"
        assert score_contradiction(_item(), [_match()], bad_llm, fallback=0.1) == pytest.approx(0.1)


class TestScoreStoragePressure:
    def test_empty_store(self):
        assert score_storage_pressure({"STM": 0, "MTM": 0, "LTM": 0}) == 0.0

    def test_half_full(self):
        # STM cap=20, MTM cap=100 → total=120; 60 used → 0.5
        assert score_storage_pressure({"STM": 10, "MTM": 50, "LTM": 0}) == pytest.approx(0.5)

    def test_over_capacity_clamps_to_1(self):
        assert score_storage_pressure({"STM": 30, "MTM": 200, "LTM": 50}) == 1.0

    def test_ltm_not_counted(self):
        counts = {"STM": 0, "MTM": 0, "LTM": 9999}
        assert score_storage_pressure(counts) == 0.0


# ---------------------------------------------------------------------------
# policy.py  — unit tests
# ---------------------------------------------------------------------------

def _scores(importance=0.8, temporal=1.0, contradiction=0.0, storage=0.0):
    return {
        "importance": importance,
        "temporal_validity": temporal,
        "contradiction": contradiction,
        "storage_pressure": storage,
    }


class TestPolicyDecide:
    def test_add_no_matches(self):
        d = policy_decide(_item(), [], _scores())
        assert d["action"] == "ADD"
        assert d["target_id"] is None

    def test_noop_low_importance(self):
        d = policy_decide(_item(), [], _scores(importance=0.1))
        assert d["action"] == "NOOP"
        assert d["target_id"] is None

    def test_update_on_moderate_contradiction(self):
        matches = [_match()]
        d = policy_decide(_item(), matches, _scores(contradiction=0.6))
        assert d["action"] == "UPDATE"
        assert d["target_id"] == "existing__0"

    def test_delete_on_strong_contradiction(self):
        matches = [_match()]
        d = policy_decide(_item(), matches, _scores(contradiction=0.85))
        assert d["action"] == "DELETE"
        assert d["target_id"] == "existing__0"

    def test_noop_on_high_storage_pressure_low_importance(self):
        d = policy_decide(_item(), [], _scores(importance=0.4, storage=0.95))
        assert d["action"] == "NOOP"

    def test_add_on_high_storage_pressure_high_importance(self):
        d = policy_decide(_item(), [], _scores(importance=0.9, storage=0.95))
        assert d["action"] == "ADD"

    def test_noop_temporal_gate(self):
        thr = {**DEFAULT_THRESHOLDS, "temporal_min": 0.5}
        d = policy_decide(_item(), [], _scores(temporal=0.0), thresholds=thr)
        assert d["action"] == "NOOP"

    def test_decision_contains_all_four_scores(self):
        d = policy_decide(_item(), [], _scores())
        assert set(d["scores"].keys()) == {
            "importance", "temporal_validity", "contradiction", "storage_pressure"
        }

    def test_scores_are_passed_through_unchanged(self):
        s = _scores(importance=0.77, temporal=0.33, contradiction=0.11, storage=0.22)
        d = policy_decide(_item(), [], s)
        assert d["scores"]["importance"] == pytest.approx(0.77)
        assert d["scores"]["temporal_validity"] == pytest.approx(0.33)
        assert d["scores"]["contradiction"] == pytest.approx(0.11)
        assert d["scores"]["storage_pressure"] == pytest.approx(0.22)

    def test_delete_target_is_best_match_id(self):
        match_a = _match("existing__A")
        match_b = _match("existing__B")
        d = policy_decide(_item(), [match_a, match_b], _scores(contradiction=0.9))
        # Policy picks the first (best) match
        assert d["target_id"] == "existing__A"

    def test_update_target_is_best_match_id(self):
        match_a = _match("existing__A")
        match_b = _match("existing__B")
        d = policy_decide(_item(), [match_a, match_b], _scores(contradiction=0.6))
        assert d["target_id"] == "existing__A"

    def test_add_no_contradiction_with_matches(self):
        """Candidate with matching but non-contradicting existing memories → ADD."""
        matches = [_match()]
        d = policy_decide(_item(), matches, _scores(contradiction=0.1))
        assert d["action"] == "ADD"

    def test_thresholds_configurable(self):
        """Lowering importance_min should flip a borderline NOOP to ADD."""
        s = _scores(importance=0.25)
        d_default = policy_decide(_item(), [], s)
        assert d_default["action"] == "NOOP"

        d_custom = policy_decide(_item(), [], s, thresholds={"importance_min": 0.2})
        assert d_custom["action"] == "ADD"


# ---------------------------------------------------------------------------
# controller.py  — integration tests (no real LLM, no real store)
# ---------------------------------------------------------------------------

class TestDecidePublicAPI:
    """Tests for the high-level decide() function in controller.py."""

    def test_add_empty_store(self):
        llm = _combined_llm(importance=0.9, contradiction=0.0)
        d = decide(_item(), [], llm=llm)
        assert d["action"] == "ADD"

    def test_noop_low_importance(self):
        llm = _combined_llm(importance=0.1, contradiction=0.0)
        d = decide(_item(), [], llm=llm)
        assert d["action"] == "NOOP"

    def test_update_moderate_contradiction(self):
        llm = _combined_llm(importance=0.9, contradiction=0.6)
        d = decide(_item(), [_match()], llm=llm)
        assert d["action"] == "UPDATE"
        assert d["target_id"] == "existing__0"

    def test_delete_strong_contradiction(self):
        llm = _combined_llm(importance=0.9, contradiction=0.9)
        d = decide(_item(), [_match()], llm=llm)
        assert d["action"] == "DELETE"
        assert d["target_id"] == "existing__0"

    def test_noop_storage_pressure(self):
        llm = _combined_llm(importance=0.4, contradiction=0.0)
        counts = {"STM": 20, "MTM": 90, "LTM": 0}   # 110/120 ≈ 0.92 pressure
        d = decide(_item(), [], llm=llm, counts=counts)
        assert d["action"] == "NOOP"

    def test_decision_schema_complete(self):
        """Decision always has action, target_id, and all four scores."""
        llm = _combined_llm(importance=0.8, contradiction=0.0)
        d = decide(_item(), [], llm=llm)
        assert "action" in d
        assert "target_id" in d
        assert set(d["scores"].keys()) == {
            "importance", "temporal_validity", "contradiction", "storage_pressure"
        }

    def test_malformed_llm_scores_do_not_raise(self):
        """Malformed LLM output must never crash the controller."""
        def broken_llm(prompt, system=None, **kw):
            return "I am totally broken LLM output with no JSON whatsoever!"

        d = decide(_item(), [], llm=broken_llm)
        assert d["action"] in {"ADD", "NOOP", "UPDATE", "DELETE"}

    def test_empty_existing_matches(self):
        llm = _combined_llm(importance=0.8, contradiction=0.0)
        d = decide(_item(), [], llm=llm)
        assert d["action"] in {"ADD", "NOOP"}

    def test_custom_thresholds_respected(self):
        """importance_min=0 should make everything at least ADD-eligible."""
        llm = _combined_llm(importance=0.05, contradiction=0.0)
        d = decide(_item(), [], llm=llm,
                   thresholds={"importance_min": 0.0})
        assert d["action"] == "ADD"


class TestProcessFunction:
    """Tests for process() using a minimal fake store."""

    class _FakeStore:
        def __init__(self):
            self.added = []
            self.updated = []
            self.deleted = []
            self._items = {}

        def query(self, text, k=5, tier="all"):
            return list(self._items.values())

        def count(self):
            return {"STM": len(self._items), "MTM": 0, "LTM": 0}

        def add(self, item):
            self._items[item["id"]] = item
            self.added.append(item["id"])
            return item

        def update(self, item_id, new_text, timestamp=None):
            if item_id in self._items:
                self._items[item_id]["text"] = new_text
            self.updated.append(item_id)
            return True

        def delete(self, item_id):
            self._items.pop(item_id, None)
            self.deleted.append(item_id)
            return True

        stm_capacity = 20
        mtm_capacity = 100

    def test_add_is_applied_to_store(self):
        from src.controller.controller import process
        store = self._FakeStore()
        llm = _combined_llm(importance=0.9, contradiction=0.0)
        d = process(_item(), store, llm=llm)
        assert d["action"] == "ADD"
        assert "D1:3__0" in store.added

    def test_update_is_applied_to_store(self):
        from src.controller.controller import process
        store = self._FakeStore()
        store._items["existing__0"] = _match()
        llm = _combined_llm(importance=0.9, contradiction=0.65)
        d = process(_item(), store, llm=llm)
        assert d["action"] == "UPDATE"
        assert "existing__0" in store.updated

    def test_delete_removes_existing_and_adds_candidate(self):
        from src.controller.controller import process
        store = self._FakeStore()
        store._items["existing__0"] = _match()
        llm = _combined_llm(importance=0.9, contradiction=0.9)
        d = process(_item(), store, llm=llm)
        assert d["action"] == "DELETE"
        assert "existing__0" in store.deleted
        # Candidate must be preserved (added after deletion)
        assert "D1:3__0" in store.added

    def test_noop_does_not_touch_store(self):
        from src.controller.controller import process
        store = self._FakeStore()
        llm = _combined_llm(importance=0.05, contradiction=0.0)
        d = process(_item(), store, llm=llm)
        assert d["action"] == "NOOP"
        assert store.added == []
        assert store.updated == []
        assert store.deleted == []

# ---------------------------------------------------------------------------
# process() DELETE edge-cases — Bug 3 regression tests
# ---------------------------------------------------------------------------

class TestProcessDeleteEdgeCases:
    """Ensure the candidate is never silently discarded on DELETE paths."""

    class _FakeStore:
        """Minimal store where items can be pre-seeded."""
        def __init__(self, items=None):
            self._items = dict(items or {})
            self.added = []
            self.updated = []
            self.deleted = []

        def query(self, text, k=5, tier="all"):
            return list(self._items.values())

        def get(self, item_id):
            return self._items.get(item_id)

        def count(self):
            return {"STM": len(self._items), "MTM": 0, "LTM": 0}

        def add(self, item):
            self._items[item["id"]] = item
            self.added.append(item["id"])
            return item

        def update(self, item_id, new_text, timestamp=None):
            if item_id in self._items:
                self._items[item_id]["text"] = new_text
            self.updated.append(item_id)
            return True

        def delete(self, item_id):
            existed = item_id in self._items
            self._items.pop(item_id, None)
            self.deleted.append(item_id)
            return existed

        stm_capacity = 20
        mtm_capacity = 100

    def test_delete_with_valid_target_id_adds_candidate(self):
        """Normal DELETE: existing item removed, candidate always added."""
        from src.controller.controller import process
        store = self._FakeStore({"existing__0": _match()})
        llm = _combined_llm(importance=0.9, contradiction=0.9)
        d = process(_item(), store, llm=llm)
        assert d["action"] == "DELETE"
        assert d["target_id"] == "existing__0"
        assert "existing__0" in store.deleted
        # Candidate preserved
        assert "D1:3__0" in store.added

    def test_delete_with_missing_target_id_still_adds_candidate(self):
        """DELETE where policy returns target_id=None must still ADD candidate."""
        from src.controller.controller import process
        from src.controller.policy import _decision

        # Build a match that has no "id" field so policy returns target_id=None
        match_no_id = {
            "text": "User lives in Mumbai", "timestamp": "2025-06-01T00:00:00",
            "source": "D1:1", "embedding": None, "tier": "STM",
            "distance": 0.1, "access_count": 0,
        }
        store = self._FakeStore({"__no_id__": match_no_id})
        llm = _combined_llm(importance=0.9, contradiction=0.9)
        d = process(_item(), store, llm=llm)
        # action is DELETE; target_id is None because the match had no "id"
        assert d["action"] == "DELETE"
        assert d["target_id"] is None
        # Nothing deleted (no valid target) but candidate must be preserved
        assert store.deleted == []
        assert "D1:3__0" in store.added

    def test_delete_target_not_in_store_still_adds_candidate(self):
        """DELETE where target no longer exists in store must still ADD candidate."""
        from src.controller.controller import process

        # Match has an ID but that ID is not in the store (already evicted)
        match_gone = _match("gone__0")
        store = self._FakeStore()  # empty store — "gone__0" not present

        # Force the store to return the stale match from query
        store._items["__placeholder__"] = match_gone

        llm = _combined_llm(importance=0.9, contradiction=0.9)
        # Remove placeholder after query is set up — override query to return stale match
        original_query = store.query
        store.query = lambda text, k=5, tier="all": [match_gone]

        d = process(_item(), store, llm=llm)
        assert d["action"] == "DELETE"
        # get("gone__0") returns None -> store.delete not called
        assert "gone__0" not in store.deleted
        # Candidate always added
        assert "D1:3__0" in store.added


# ---------------------------------------------------------------------------
# src/llm/client.py — unit tests (no real network calls)
# ---------------------------------------------------------------------------

class TestCallLlm:
    """Tests for src/llm/client.py that never make real API calls."""

    def test_module_imports_without_sdk(self):
        """client.py must import cleanly even when provider SDKs are absent."""
        import importlib
        mod = importlib.import_module("src.llm.client")
        assert callable(getattr(mod, "call_llm"))

    def test_unknown_provider_raises_value_error(self, monkeypatch):
        """An unrecognised LLM_PROVIDER value raises ValueError before any SDK call."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        from src.llm import client
        import importlib; importlib.reload(client)
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
            client.call_llm("hello")

    def test_missing_gemini_key_raises_environment_error(self, monkeypatch):
        """Missing GEMINI_API_KEY raises EnvironmentError, not ImportError."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        # Stub out the google.generativeai import so the test is SDK-agnostic
        import sys, types
        fake_genai = types.ModuleType("google.generativeai")
        fake_google = types.ModuleType("google")
        fake_google.generativeai = fake_genai  # type: ignore[attr-defined]
        sys.modules.setdefault("google", fake_google)
        sys.modules["google.generativeai"] = fake_genai

        from src.llm import client
        import importlib; importlib.reload(client)
        with pytest.raises(EnvironmentError, match="GEMINI_API_KEY"):
            client.call_llm("hello")

    def test_missing_groq_key_raises_environment_error(self, monkeypatch):
        """Missing GROQ_API_KEY raises EnvironmentError, not ImportError."""
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        import sys, types
        fake_groq_mod = types.ModuleType("groq")
        fake_groq_mod.Groq = object  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "groq", fake_groq_mod)

        from src.llm import client
        import importlib; importlib.reload(client)
        # call_llm() invokes _load_env() which would re-import the REAL
        # .env (dotenv override=False repopulates the deleted key) and this
        # test would stop testing a missing key at all.  Disable it so the
        # deletion above is what the code actually sees.
        monkeypatch.setattr(client, "_load_env", lambda: None)
        with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
            client.call_llm("hello")

    def test_gemini_sdk_missing_raises_import_error(self, monkeypatch):
        """If google-generativeai is not installed, ImportError with install hint."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        import sys
        # Remove the stub if present so the import fails
        for mod in list(sys.modules.keys()):
            if "google" in mod:
                del sys.modules[mod]

        from src.llm import client
        import importlib; importlib.reload(client)
        with pytest.raises(ImportError, match="google-generativeai"):
            client.call_llm("hello")

    def test_groq_sdk_missing_raises_import_error(self, monkeypatch):
        """If groq is not installed, ImportError with install hint."""
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "fake-key")

        import sys
        # Setting the entry to None makes `import groq` raise ImportError
        # ("... halted; None in sys.modules") — the canonical way to
        # simulate a missing package.  Popping the key would NOT work: the
        # real SDK is installed, so the import would succeed and this test
        # would make a real network call with the fake key (401).
        monkeypatch.setitem(sys.modules, "groq", None)

        from src.llm import client
        import importlib; importlib.reload(client)
        with pytest.raises(ImportError, match="groq"):
            client.call_llm("hello")

    def test_gemini_call_uses_injected_stub(self, monkeypatch):
        """Gemini path returns the stub model's text without touching network."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        import sys, types

        # Build a minimal stub that mimics genai.GenerativeModel
        class _FakeResponse:
            text = '["stubbed fact"]'

        class _FakeModel:
            def __init__(self, name): pass
            def generate_content(self, prompt, **kw): return _FakeResponse()

        fake_genai = types.ModuleType("google.generativeai")
        fake_genai.configure = lambda **kw: None  # type: ignore[attr-defined]
        fake_genai.GenerativeModel = _FakeModel  # type: ignore[attr-defined]
        fake_google = types.ModuleType("google")
        fake_google.generativeai = fake_genai  # type: ignore[attr-defined]
        sys.modules["google"] = fake_google
        sys.modules["google.generativeai"] = fake_genai

        from src.llm import client
        import importlib; importlib.reload(client)

        result = client.call_llm("extract facts", system="be brief")
        assert result == '["stubbed fact"]'

    def test_groq_call_uses_injected_stub(self, monkeypatch):
        """Groq path returns the stub completion's content without touching network."""
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "fake-key")

        import sys, types

        class _FakeMessage:
            content = '{"importance": 0.9}'

        class _FakeChoice:
            message = _FakeMessage()

        class _FakeCompletion:
            choices = [_FakeChoice()]

        class _FakeGroq:
            def __init__(self, api_key): pass
            class chat:
                class completions:
                    @staticmethod
                    def create(model, messages, **kw):
                        return _FakeCompletion()

        fake_groq_mod = types.ModuleType("groq")
        fake_groq_mod.Groq = _FakeGroq  # type: ignore[attr-defined]
        sys.modules["groq"] = fake_groq_mod

        from src.llm import client
        import importlib; importlib.reload(client)

        result = client.call_llm("rate this", system="be a scorer")
        assert result == '{"importance": 0.9}'

    def test_extraction_can_import_client(self):
        """extract_candidates can be imported without crashing (llm= path unused)."""
        from src.extraction.extractor import extract_candidates
        fake = lambda p, system=None, **kw: '["Test fact."]'
        items = extract_candidates("I went jogging today.",
                                   {"speaker": "Ali", "timestamp": "2026-01-01T00:00:00",
                                    "source": "D2:1"},
                                   llm=fake)
        assert items[0]["text"] == "Test fact."

    def test_controller_can_import_client(self):
        """decide() can be called with an explicit llm= without importing the real client."""
        from src.controller.controller import decide
        llm = _combined_llm(importance=0.8, contradiction=0.0)
        d = decide(_item(), [], llm=llm)
        assert d["action"] in {"ADD", "NOOP"}

# ---------------------------------------------------------------------------
# Mistral provider tests — extend TestCallLlm without a real API call
# ---------------------------------------------------------------------------

class TestMistralProvider:
    """Offline tests for the Mistral provider in src/llm/client.py."""

    def _fake_mistral_module(self, response_text: str = '{"importance": 0.8}'):
        """Build a minimal mistralai stub that matches the SDK 3.x API."""
        import sys, types

        class _FakeMessage:
            content = response_text

        class _FakeChoice:
            message = _FakeMessage()

        class _FakeResponse:
            choices = [_FakeChoice()]

        class _FakeChat:
            @staticmethod
            def complete(model, messages, **kw):
                return _FakeResponse()

        class _FakeMistral:
            def __init__(self, api_key):
                self.chat = _FakeChat()

        # SDK 3.x: import is from mistralai.client
        client_mod = types.ModuleType("mistralai.client")
        client_mod.Mistral = _FakeMistral  # type: ignore[attr-defined]
        top_mod = types.ModuleType("mistralai")
        top_mod.Mistral = _FakeMistral  # type: ignore[attr-defined]
        sys.modules["mistralai.client"] = client_mod
        return top_mod

    # -- Provider selection --

    def test_mistral_selected_by_env(self, monkeypatch):
        """Setting LLM_PROVIDER=mistral routes to _call_mistral."""
        import sys, importlib
        monkeypatch.setenv("LLM_PROVIDER", "mistral")
        monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")
        sys.modules["mistralai"] = self._fake_mistral_module()
        from src.llm import client
        importlib.reload(client)
        result = client.call_llm("hello", system="be terse")
        assert result == '{"importance": 0.8}'

    def test_unknown_provider_still_raises(self, monkeypatch):
        """Adding Mistral must not break the unknown-provider guard."""
        import importlib
        monkeypatch.setenv("LLM_PROVIDER", "cohere")
        from src.llm import client
        importlib.reload(client)
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
            client.call_llm("hello")

    # -- API key checks --

    def test_missing_mistral_key_raises_environment_error(self, monkeypatch):
        """Missing MISTRAL_API_KEY raises EnvironmentError before any network call."""
        import sys, importlib
        monkeypatch.setenv("LLM_PROVIDER", "mistral")
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        sys.modules["mistralai"] = self._fake_mistral_module()
        from src.llm import client
        importlib.reload(client)
        monkeypatch.setattr(client, "_load_env", lambda: None)  # prevent .env from restoring key
        with pytest.raises(EnvironmentError, match="MISTRAL_API_KEY"):
            client.call_llm("hello")

    def test_missing_mistral_key_error_mentions_env_file(self, monkeypatch):
        import sys, importlib
        monkeypatch.setenv("LLM_PROVIDER", "mistral")
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        sys.modules["mistralai"] = self._fake_mistral_module()
        from src.llm import client
        importlib.reload(client)
        monkeypatch.setattr(client, "_load_env", lambda: None)
        with pytest.raises(EnvironmentError, match=".env"):
            client.call_llm("hello")

    # -- SDK availability --

    def test_mistral_sdk_missing_raises_import_error(self, monkeypatch):
        """If mistralai is not installed, ImportError with install hint."""
        import sys, importlib, builtins
        monkeypatch.setenv("LLM_PROVIDER", "mistral")
        monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")
        sys.modules.pop("mistralai", None)
        sys.modules.pop("mistralai.client", None)
        real_import = builtins.__import__
        def _block(name, *args, **kwargs):
            if name in ("mistralai", "mistralai.client"):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", _block)
        from src.llm import client
        importlib.reload(client)
        with pytest.raises(ImportError, match="mistralai"):
            client.call_llm("hello")

    # -- Successful response --

    def test_response_text_returned_stripped(self, monkeypatch):
        """The response text is stripped of whitespace."""
        import sys, types, importlib
        monkeypatch.setenv("LLM_PROVIDER", "mistral")
        monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")

        class _Msg:
            content = "  answer with spaces  "
        class _Choice:
            message = _Msg()
        class _Resp:
            choices = [_Choice()]
        class _Chat:
            @staticmethod
            def complete(model, messages, **kw): return _Resp()
        class _Client:
            def __init__(self, api_key): self.chat = _Chat()

        mod = types.ModuleType("mistralai")
        mod.Mistral = _Client  # type: ignore[attr-defined]
        sys.modules["mistralai"] = mod
        sys.modules["mistralai.client"] = mod
        from src.llm import client
        importlib.reload(client)
        result = client.call_llm("q")
        assert result == "answer with spaces"

    def test_system_message_included(self, monkeypatch):
        """System prompt is passed as a system role message to Mistral."""
        import sys, types, importlib
        monkeypatch.setenv("LLM_PROVIDER", "mistral")
        monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")

        captured: list[list[dict]] = []

        class _Msg:
            content = "ok"
        class _Choice:
            message = _Msg()
        class _Resp:
            choices = [_Choice()]
        class _Chat:
            @staticmethod
            def complete(model, messages, **kw):
                captured.append(messages)
                return _Resp()
        class _Client:
            def __init__(self, api_key): self.chat = _Chat()

        mod = types.ModuleType("mistralai")
        mod.Mistral = _Client  # type: ignore[attr-defined]
        sys.modules["mistralai"] = mod
        sys.modules["mistralai.client"] = mod
        from src.llm import client
        importlib.reload(client)
        client.call_llm("user prompt", system="system instruction")
        assert len(captured) == 1
        roles = [m["role"] for m in captured[0]]
        assert roles[0] == "system"
        assert captured[0][0]["content"] == "system instruction"
        assert roles[-1] == "user"

    def test_no_system_omits_system_message(self, monkeypatch):
        """When system=None only a user message is sent."""
        import sys, types, importlib
        monkeypatch.setenv("LLM_PROVIDER", "mistral")
        monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")

        captured: list[list[dict]] = []

        class _Msg:
            content = "ok"
        class _Choice:
            message = _Msg()
        class _Resp:
            choices = [_Choice()]
        class _Chat:
            @staticmethod
            def complete(model, messages, **kw):
                captured.append(messages)
                return _Resp()
        class _Client:
            def __init__(self, api_key): self.chat = _Chat()

        mod = types.ModuleType("mistralai")
        mod.Mistral = _Client  # type: ignore[attr-defined]
        sys.modules["mistralai"] = mod
        sys.modules["mistralai.client"] = mod
        from src.llm import client
        importlib.reload(client)
        client.call_llm("user only")
        assert len(captured[0]) == 1
        assert captured[0][0]["role"] == "user"

    # -- Model configuration --

    def test_default_model_is_mistral_small_latest(self, monkeypatch):
        """Default model is mistral-small-latest when MISTRAL_MODEL is unset."""
        import sys, types, importlib
        monkeypatch.setenv("LLM_PROVIDER", "mistral")
        monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")
        monkeypatch.delenv("MISTRAL_MODEL", raising=False)

        used_models: list[str] = []

        class _Msg:
            content = "ok"
        class _Choice:
            message = _Msg()
        class _Resp:
            choices = [_Choice()]
        class _Chat:
            @staticmethod
            def complete(model, messages, **kw):
                used_models.append(model)
                return _Resp()
        class _Client:
            def __init__(self, api_key): self.chat = _Chat()

        mod = types.ModuleType("mistralai")
        mod.Mistral = _Client  # type: ignore[attr-defined]
        sys.modules["mistralai"] = mod
        sys.modules["mistralai.client"] = mod
        from src.llm import client
        importlib.reload(client)
        client.call_llm("q")
        assert used_models[0] == "mistral-small-latest"

    def test_custom_model_via_env(self, monkeypatch):
        import sys, types, importlib
        monkeypatch.setenv("LLM_PROVIDER", "mistral")
        monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")
        monkeypatch.setenv("MISTRAL_MODEL", "mistral-large-latest")

        used_models: list[str] = []

        class _Msg:
            content = "ok"
        class _Choice:
            message = _Msg()
        class _Resp:
            choices = [_Choice()]
        class _Chat:
            @staticmethod
            def complete(model, messages, **kw):
                used_models.append(model)
                return _Resp()
        class _Client:
            def __init__(self, api_key): self.chat = _Chat()

        mod = types.ModuleType("mistralai")
        mod.Mistral = _Client  # type: ignore[attr-defined]
        sys.modules["mistralai"] = mod
        sys.modules["mistralai.client"] = mod
        from src.llm import client
        importlib.reload(client)
        client.call_llm("q")
        assert used_models[0] == "mistral-large-latest"

    # -- Error handling --

    def test_api_error_raises_runtime_error(self, monkeypatch):
        """SDK exceptions are wrapped in RuntimeError."""
        import sys, types, importlib
        monkeypatch.setenv("LLM_PROVIDER", "mistral")
        monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")

        class _Chat:
            @staticmethod
            def complete(model, messages, **kw):
                raise ConnectionError("network timeout")
        class _Client:
            def __init__(self, api_key): self.chat = _Chat()

        mod = types.ModuleType("mistralai")
        mod.Mistral = _Client  # type: ignore[attr-defined]
        sys.modules["mistralai"] = mod
        sys.modules["mistralai.client"] = mod
        from src.llm import client
        importlib.reload(client)
        with pytest.raises(RuntimeError, match="Mistral API call failed"):
            client.call_llm("q")

    def test_empty_response_raises_runtime_error(self, monkeypatch):
        """An empty string response raises RuntimeError."""
        import sys, types, importlib
        monkeypatch.setenv("LLM_PROVIDER", "mistral")
        monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")

        class _Msg:
            content = "   "
        class _Choice:
            message = _Msg()
        class _Resp:
            choices = [_Choice()]
        class _Chat:
            @staticmethod
            def complete(model, messages, **kw): return _Resp()
        class _Client:
            def __init__(self, api_key): self.chat = _Chat()

        mod = types.ModuleType("mistralai")
        mod.Mistral = _Client  # type: ignore[attr-defined]
        sys.modules["mistralai"] = mod
        sys.modules["mistralai.client"] = mod
        from src.llm import client
        importlib.reload(client)
        with pytest.raises(RuntimeError, match="empty"):
            client.call_llm("q")

    # -- Demo mode independence --

    def test_demo_mode_unaffected_by_missing_mistral(self, monkeypatch):
        """Demo mode (_demo_llm) works without mistralai installed."""
        import sys
        sys.modules.pop("mistralai", None)
        from src.server import _demo_llm
        result = _demo_llm(
            "Speaker: Maya\nDate: 2026-01-01\nMessage: I love hiking.\n",
            system="You extract long-term memories from conversations. "
                   "Reply with ONLY a JSON array of strings, nothing else.",
        )
        import json
        facts = json.loads(result)
        assert isinstance(facts, list)
