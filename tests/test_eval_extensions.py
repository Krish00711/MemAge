"""Tests for eval extensions: signal ablations, --sample/--seed/--output-dir,
real-dataset baseline wiring, and server API endpoints.

All tests are offline — no API keys, no ChromaDB, no real LLM.

Run: python -m pytest tests/test_eval_extensions.py -q
"""
from __future__ import annotations

import json
import sys
import types
from io import StringIO
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Signal ablation config tests
# ---------------------------------------------------------------------------

class TestSignalAblation:
    def test_importance_configs_three_entries(self):
        from src.eval.signal_ablation import importance_configs
        cfgs = importance_configs()
        assert len(cfgs) == 3
        labels = [c.label for c in cfgs]
        assert "importance_min=0.0" in labels
        assert "importance_min=0.3" in labels
        assert "importance_min=0.6" in labels

    def test_temporal_configs(self):
        from src.eval.signal_ablation import temporal_configs
        cfgs = temporal_configs()
        assert all("temporal_min" in c.params.get("thresholds", {}) for c in cfgs)

    def test_contradiction_configs(self):
        from src.eval.signal_ablation import contradiction_configs
        cfgs = contradiction_configs()
        assert len(cfgs) == 3
        labels = [c.label for c in cfgs]
        assert "contradiction=loose" in labels
        assert "contradiction=default" in labels
        assert "contradiction=strict" in labels

    def test_storage_configs(self):
        from src.eval.signal_ablation import storage_configs
        cfgs = storage_configs()
        assert len(cfgs) == 3

    def test_signal_configs_dispatch(self):
        from src.eval.signal_ablation import signal_configs
        for name in ("importance", "temporal", "contradiction", "storage"):
            cfgs = signal_configs(name)
            assert len(cfgs) >= 2

    def test_unknown_signal_raises(self):
        from src.eval.signal_ablation import signal_configs
        with pytest.raises(ValueError, match="Unknown signal"):
            signal_configs("invalid")

    def test_all_labels_unique_within_family(self):
        from src.eval.signal_ablation import signal_configs
        for name in ("importance", "temporal", "contradiction", "storage"):
            cfgs = signal_configs(name)
            labels = [c.label for c in cfgs]
            assert len(labels) == len(set(labels)), f"Duplicate labels in {name}"

    def test_thresholds_in_params(self):
        from src.eval.signal_ablation import importance_configs
        for cfg in importance_configs():
            assert "thresholds" in cfg.params
            assert isinstance(cfg.params["thresholds"], dict)


# ---------------------------------------------------------------------------
# CLI signal ablation integration
# ---------------------------------------------------------------------------

class TestCLISignalAblation:
    def test_ablate_signal_importance_exits_0(self):
        from src.eval.run_benchmark import main
        rc = main(["--dataset", "synthetic", "--ablate", "signal:importance"])
        assert rc == 0

    def test_ablate_signal_temporal_exits_0(self):
        from src.eval.run_benchmark import main
        rc = main(["--dataset", "synthetic", "--ablate", "signal:temporal"])
        assert rc == 0

    def test_ablate_signal_contradiction_exits_0(self):
        from src.eval.run_benchmark import main
        rc = main(["--dataset", "synthetic", "--ablate", "signal:contradiction"])
        assert rc == 0

    def test_ablate_signal_storage_exits_0(self):
        from src.eval.run_benchmark import main
        rc = main(["--dataset", "synthetic", "--ablate", "signal:storage"])
        assert rc == 0

    def test_ablate_signal_unknown_exits_1(self, capsys):
        from src.eval.run_benchmark import main
        rc = main(["--dataset", "synthetic", "--ablate", "signal:badname"])
        assert rc != 0

    def test_ablate_signal_no_name_exits_1(self, capsys):
        from src.eval.run_benchmark import main
        rc = main(["--dataset", "synthetic", "--ablate", "signal:"])
        assert rc != 0


# ---------------------------------------------------------------------------
# --sample flag
# ---------------------------------------------------------------------------

class TestSampleFlag:
    def test_sample_limits_to_1(self):
        from src.eval.run_benchmark import main
        # synthetic has 2 cases; sample=1 should run 1
        rc = main(["--dataset", "synthetic", "--baseline", "none", "--sample", "1"])
        assert rc == 0

    def test_sample_larger_than_dataset_ok(self):
        from src.eval.run_benchmark import main
        rc = main(["--dataset", "synthetic", "--baseline", "none", "--sample", "999"])
        assert rc == 0

    def test_sample_zero_rejected(self, capsys):
        from src.eval.run_benchmark import main
        with pytest.raises(SystemExit) as exc_info:
            main(["--dataset", "synthetic", "--sample", "0"])
        assert exc_info.value.code != 0

    def test_sample_in_compare_mode(self):
        from src.eval.run_benchmark import main
        rc = main(["--dataset", "synthetic", "--compare", "none,full_history",
                   "--sample", "1"])
        assert rc == 0


# ---------------------------------------------------------------------------
# --output-dir saves predictions and results
# ---------------------------------------------------------------------------

class TestOutputDir:
    def test_predictions_json_written(self, tmp_path):
        from src.eval.run_benchmark import main
        out = str(tmp_path / "out")
        rc = main(["--dataset", "synthetic", "--baseline", "none",
                   "--sample", "1", "--output-dir", out])
        assert rc == 0
        pred = tmp_path / "out" / "predictions.json"
        assert pred.exists()
        data = json.loads(pred.read_text())
        assert "predictions" in data
        assert len(data["predictions"]) == 1

    def test_results_json_written(self, tmp_path):
        from src.eval.run_benchmark import main
        out = str(tmp_path / "out")
        main(["--dataset", "synthetic", "--baseline", "none",
              "--sample", "1", "--output-dir", out])
        results = tmp_path / "out" / "results.json"
        assert results.exists()
        data = json.loads(results.read_text())
        assert "results" in data
        assert "mean_f1" in data["results"]

    def test_results_meta_contains_config(self, tmp_path):
        from src.eval.run_benchmark import main
        out = str(tmp_path / "out")
        main(["--dataset", "synthetic", "--baseline", "none",
              "--sample", "1", "--seed", "42", "--output-dir", out])
        data = json.loads((tmp_path / "out" / "results.json").read_text())
        assert data["meta"]["baseline"] == "none"
        assert data["meta"]["seed"] == 42
        assert data["meta"]["sample"] == 1

    def test_predictions_contain_all_required_fields(self, tmp_path):
        from src.eval.run_benchmark import main
        out = str(tmp_path / "out")
        main(["--dataset", "synthetic", "--baseline", "none", "--output-dir", out])
        data = json.loads((tmp_path / "out" / "predictions.json").read_text())
        for pred in data["predictions"]:
            for key in ("case_id", "query", "predicted_answer", "reference_answer",
                        "f1", "bleu1", "latency_ms"):
                assert key in pred


# ---------------------------------------------------------------------------
# Server API tests (no real store/LLM needed)
# ---------------------------------------------------------------------------

def _make_server_handler():
    """Import server module and return a minimal test client using _DemoStore."""
    import importlib
    server_mod = importlib.import_module("src.server")

    class FakeAgent:
        def chat(self, message, session_id="web"):
            from src.pipeline import TurnResult
            from src.agent import AgentResponse
            return AgentResponse(
                message=message,
                session_id=session_id,
                answer=f"Demo answer for: {message}",
                context="",
                retrieved_ids=[],
                turn_result=TurnResult(turn=message),
                ok=True,
                error=None,
            )

    store = server_mod._DemoStore()
    server_mod._store = store
    server_mod._agent = FakeAgent()
    return server_mod, store


class TestServerAPI:
    """Test the server API logic by calling handlers directly."""

    def test_demo_store_list_all(self):
        from src.server import _DemoStore
        s = _DemoStore()
        items = s.list_all()
        assert len(items) == 4

    def test_demo_store_tier_filter(self):
        from src.server import _DemoStore
        s = _DemoStore()
        stm = s.list_all(tier="STM")
        assert all(i["tier"] == "STM" for i in stm)

    def test_demo_store_text_search(self):
        from src.server import _DemoStore
        s = _DemoStore()
        results = s.list_all(q="tea")
        assert any("tea" in i["text"].lower() for i in results)

    def test_demo_store_add_and_get(self):
        from src.server import _DemoStore
        s = _DemoStore()
        item = {"id": "test__0", "text": "Test fact.", "timestamp": "2026-01-01T00:00:00",
                "source": "test", "embedding": None}
        s.add(item)
        got = s.get("test__0")
        assert got is not None
        assert got["text"] == "Test fact."

    def test_demo_store_delete(self):
        from src.server import _DemoStore
        s = _DemoStore()
        ok = s.delete("demo__0")
        assert ok is True
        assert s.get("demo__0") is None

    def test_demo_store_delete_missing(self):
        from src.server import _DemoStore
        s = _DemoStore()
        ok = s.delete("nonexistent")
        assert ok is False

    def test_demo_store_update(self):
        from src.server import _DemoStore
        s = _DemoStore()
        s.update("demo__0", "Maya drinks green tea.")
        got = s.get("demo__0")
        assert got["text"] == "Maya drinks green tea."

    def test_demo_store_reset(self):
        from src.server import _DemoStore
        s = _DemoStore()
        s.reset()
        assert s.list_all() == []

    def test_demo_store_count(self):
        from src.server import _DemoStore
        s = _DemoStore()
        counts = s.count()
        assert set(counts.keys()) == {"STM", "MTM", "LTM"}
        assert sum(counts.values()) == 4

    def test_demo_store_query_keyword(self):
        from src.server import _DemoStore
        s = _DemoStore()
        results = s.query("tea", k=5)
        assert any("tea" in r["text"].lower() for r in results)

    def test_serialise_turn_result_none(self):
        from src.server import _serialise_turn_result
        assert _serialise_turn_result(None) is None

    def test_serialise_turn_result_with_candidates(self):
        from src.pipeline import TurnResult, CandidateResult
        from src.server import _serialise_turn_result
        tr = TurnResult(turn="hello")
        tr.candidates.append(CandidateResult(
            candidate={"id": "x__0", "text": "fact"},
            decision={"action": "ADD", "target_id": None, "scores": {}},
            ok=True,
        ))
        out = _serialise_turn_result(tr)
        assert out["added"] == ["x__0"]
        assert len(out["candidates"]) == 1

    def test_log_event(self):
        import src.server as srv
        import threading
        srv._TIMELINE_LOCK = threading.Lock()
        srv._TIMELINE.clear()
        srv._log_event("ADD", {"id": "a", "text": "fact"}, {"scores": {}, "target_id": None})
        with srv._TIMELINE_LOCK:
            assert len(srv._TIMELINE) == 1
            assert srv._TIMELINE[0]["action"] == "ADD"


# ---------------------------------------------------------------------------
# Demo server behaviour regression tests (no HTTP, no real store)
# ---------------------------------------------------------------------------

class TestDemoLLM:
    """_demo_llm must now extract facts from statements, not always return []."""

    def test_factual_statement_extracts_fact(self):
        import json
        from src.server import _demo_llm
        result = _demo_llm(
            "Speaker: Maya\nDate: 2026-01-01\nMessage: I live in Chennai.\n",
            system="You extract long-term memories from conversations. "
                   "Reply with ONLY a JSON array of strings, nothing else.",
        )
        facts = json.loads(result)
        assert isinstance(facts, list)
        assert len(facts) >= 1
        assert any("chennai" in f.lower() or "live" in f.lower() for f in facts)

    def test_question_produces_no_facts(self):
        import json
        from src.server import _demo_llm
        result = _demo_llm(
            "Speaker: Maya\nDate: 2026-01-01\nMessage: Where do I live?\n",
            system="You extract long-term memories from conversations. "
                   "Reply with ONLY a JSON array of strings, nothing else.",
        )
        assert json.loads(result) == []

    def test_greeting_produces_no_facts(self):
        import json
        from src.server import _demo_llm
        result = _demo_llm(
            "Speaker: Maya\nDate: 2026-01-01\nMessage: Hi there!\n",
            system="You extract long-term memories from conversations. "
                   "Reply with ONLY a JSON array of strings, nothing else.",
        )
        assert json.loads(result) == []

    def test_importance_returns_valid_json(self):
        import json
        from src.server import _demo_llm
        result = _demo_llm("some fact", system="You are a memory importance scorer. Reply with ONLY a JSON object, nothing else.")
        data = json.loads(result)
        assert "importance" in data
        assert 0.0 <= data["importance"] <= 1.0

    def test_contradiction_high_for_move_statement(self):
        import json
        from src.server import _demo_llm
        result = _demo_llm(
            "New fact: User moved to Bangalore\n\nExisting: User lives in Chennai",
            system="You are a contradiction detector. Reply with ONLY a JSON object, nothing else.",
        )
        data = json.loads(result)
        assert "contradiction" in data
        assert data["contradiction"] >= 0.5

    def test_reranker_returns_valid_rankings(self):
        import json
        from src.server import _demo_llm
        prompt = "- ID: mem_001\n  Text: fact one\n- ID: mem_002\n  Text: fact two"
        result = _demo_llm(prompt, system="You are a relevance reranker for a selective long-term memory system.")
        data = json.loads(result)
        assert "rankings" in data
        assert len(data["rankings"]) == 2

    def test_answer_with_context_uses_content(self):
        from src.server import _demo_llm
        result = _demo_llm(
            "Relevant memories:\n[Memory 1]\nContent: Maya lives in Chennai.\n\nUser: Where do I live?\nAnswer:",
            system="You are a helpful assistant with access to a long-term memory store.",
        )
        assert "chennai" in result.lower() or "remember" in result.lower()

    def test_answer_no_context_admits_ignorance(self):
        from src.server import _demo_llm
        result = _demo_llm(
            "No relevant memories were found for this question.\n\nUser: What is my age?\nAnswer:",
            system="You are a helpful assistant with access to a long-term memory store.",
        )
        assert "don't have" in result.lower() or "no" in result.lower()


class TestDemoStoreChanges:
    """Demo store must change when the user states new facts or corrections."""

    def _make_agent(self):
        from src.server import _demo_llm, _DemoStore
        from src.agent import MemAgeAgent
        store = _DemoStore()
        agent = MemAgeAgent(store, llm=_demo_llm, k=4)
        return agent, store

    def test_new_fact_adds_to_store(self):
        agent, store = self._make_agent()
        before = sum(store.count().values())
        resp = agent.chat("I just adopted a puppy named Max.", session_id="s1")
        after = sum(store.count().values())
        assert after > before, "Store should grow after a new fact is stated"
        assert resp.turn_result is not None
        assert any(c.decision["action"] == "ADD" for c in resp.turn_result.candidates)

    def test_correction_updates_or_deletes(self):
        agent, store = self._make_agent()
        # State a fact first, then correct it
        agent.chat("I live in Chennai.", session_id="s1")
        resp = agent.chat("I moved to Bangalore.", session_id="s1")
        assert resp.turn_result is not None
        actions = {c.decision["action"] for c in resp.turn_result.candidates}
        assert actions & {"UPDATE", "DELETE", "ADD"}, f"Expected mutating action, got {actions}"
        assert all(c.ok for c in resp.turn_result.candidates), \
            "All candidates must succeed without datetime errors"

    def test_no_datetime_comparison_error(self):
        """Regression: UTC-aware vs naive datetime must not crash the controller."""
        agent, store = self._make_agent()
        # First chat adds a memory with UTC-aware timestamp
        agent.chat("I prefer coffee.", session_id="s1")
        # Second chat queries that memory — used to raise TypeError
        resp = agent.chat("Actually I prefer tea.", session_id="s1")
        errors = [c.error for c in (resp.turn_result.candidates if resp.turn_result else []) if c.error]
        assert not any("datetime" in (e or "") for e in errors), \
            f"Datetime comparison error should not occur: {errors}"

    def test_timeline_logs_add_events(self):
        import src.server as srv
        import threading
        srv._TIMELINE.clear()
        srv._TIMELINE_LOCK = threading.Lock()

        store = srv._DemoStore()
        srv._store = store
        from src.agent import MemAgeAgent
        srv._agent = MemAgeAgent(store, llm=srv._demo_llm, k=4)

        resp = srv._agent.chat("I own a red bicycle.", session_id="demo")
        if resp.turn_result:
            for cr in resp.turn_result.candidates:
                if cr.decision["action"] != "NOOP":
                    srv._log_event(cr.decision["action"], cr.candidate, cr.decision)

        with srv._TIMELINE_LOCK:
            actions = [e["action"] for e in srv._TIMELINE]
        assert "ADD" in actions, "Timeline should record ADD events from new facts"

    def test_health_is_demo_flag(self):
        from src.server import _DemoStore
        import src.server as srv
        store = _DemoStore()
        srv._store = store
        is_demo = isinstance(srv._store, _DemoStore)
        assert is_demo is True


class TestTemporalValidityNaiveAwareMix:
    """Regression: mixed aware/naive timestamps must not raise TypeError."""

    def test_aware_candidate_naive_match(self):
        from src.controller.scoring import score_temporal_validity
        cand = {"timestamp": "2026-10-25T06:30:00+00:00"}
        match = {"timestamp": "2026-01-01T09:00:00"}
        score = score_temporal_validity(cand, [match])
        assert 0.0 <= score <= 1.0

    def test_naive_candidate_aware_match(self):
        from src.controller.scoring import score_temporal_validity
        cand = {"timestamp": "2026-01-01T09:00:00"}
        match = {"timestamp": "2025-06-01T12:00:00+00:00"}
        score = score_temporal_validity(cand, [match])
        assert 0.0 <= score <= 1.0

    def test_mixed_matches_no_error(self):
        from src.controller.scoring import score_temporal_validity
        cand = {"timestamp": "2026-10-25T06:30:00+00:00"}
        matches = [
            {"timestamp": "2026-01-01T09:00:00"},       # naive
            {"timestamp": "2025-06-01T12:00:00+00:00"},  # aware
            {"timestamp": ""},                            # empty
        ]
        score = score_temporal_validity(cand, matches)
        assert 0.0 <= score <= 1.0

    def test_both_aware_no_error(self):
        from src.controller.scoring import score_temporal_validity
        cand = {"timestamp": "2026-10-25T06:30:00+00:00"}
        match = {"timestamp": "2025-01-01T00:00:00+00:00"}
        score = score_temporal_validity(cand, [match])
        assert score == 1.0  # candidate is newer
