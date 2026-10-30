"""Tests for the presentation/observability extensions.

All offline — fake stores, fake LLMs, temp directories. No API keys, no
network, no ChromaDB required.

Covers the additions made for the presentation UI:
  * ``TieredMemoryStore.list_all`` — full listings for the Memory view.
  * ``RetrievalContext.retrieve_with_trace`` — retrieval trace consistency.
  * ``AgentResponse.retrieval_trace`` — populated by ``MemAgeAgent.chat``.
  * ``src.server._build_benchmark_argv`` — run-request validation (API).
  * ``src.server._load_experiments`` — artifact parsing (API).

Run: python -m pytest tests/test_presentation_backend.py -q
"""
from __future__ import annotations

import json

import pytest

from src.store.memory_store import TieredMemoryStore
from src.retrieval.context import RetrievalContext
from src.retrieval.reranker import RankedCandidate
from tests.test_agent import _FakeStore, _fake_llm, _make_memory


def _item(i, text, ts):
    return {"id": i, "text": text, "timestamp": ts, "source": "s", "embedding": None}


# ---------------------------------------------------------------------------
# TieredMemoryStore.list_all
# ---------------------------------------------------------------------------

def test_list_all_returns_every_memory_across_tiers(tmp_path):
    store = TieredMemoryStore(persist_dir=str(tmp_path), stm_capacity=2)
    store.add(_item("a", "User lives in Chennai", "2026-01-01T00:00:00"))
    store.add(_item("b", "User likes cricket", "2026-01-02T00:00:00"))
    store.add(_item("c", "User owns a dog", "2026-01-03T00:00:00"))
    # capacity 2 pushed the oldest into MTM
    assert store.count() == {"STM": 2, "MTM": 1, "LTM": 0}

    items = store.list_all()
    assert {i["id"] for i in items} == {"a", "b", "c"}
    # newest first
    assert [i["id"] for i in items] == ["c", "b", "a"]
    # tiers reported per item
    assert {i["id"]: i["tier"] for i in items}["a"] == "MTM"


def test_list_all_tier_and_text_filters(tmp_path):
    store = TieredMemoryStore(persist_dir=str(tmp_path))
    store.add(_item("a", "User lives in Chennai", "2026-01-01T00:00:00"))
    store.add(_item("b", "User likes cricket", "2026-01-02T00:00:00"))

    assert [i["id"] for i in store.list_all(tier="STM")] == ["b", "a"]
    assert store.list_all(tier="MTM") == []
    assert [i["id"] for i in store.list_all(q="cricket")] == ["b"]
    with pytest.raises(ValueError):
        store.list_all(tier="nope")


def test_list_all_is_read_only(tmp_path):
    """Listing must not bump access counts (unlike query)."""
    store = TieredMemoryStore(persist_dir=str(tmp_path))
    store.add(_item("a", "User lives in Chennai", "2026-01-01T00:00:00"))
    before = store.get("a")["access_count"]
    store.list_all()
    store.list_all()
    assert store.get("a")["access_count"] == before


# ---------------------------------------------------------------------------
# RetrievalContext.retrieve_with_trace
# ---------------------------------------------------------------------------

class _StaticRetriever:
    def __init__(self, items):
        self._items = items

    def retrieve(self, query, k, tier):
        return self._items[:k]


class _ScoredReranker:
    def rerank(self, query, candidates):
        return [RankedCandidate(candidate=c, score=1.0 - 0.1 * n)
                for n, c in enumerate(candidates)]


def _ctx(items, k=3):
    return RetrievalContext(_StaticRetriever(items), _ScoredReranker(), k=k)


def test_retrieve_with_trace_matches_retrieve_ranked():
    items = [_make_memory(f"m{i}", f"memory text {i}") for i in range(3)]
    context, ids = _ctx(items).retrieve_ranked("q", token_budget=200)
    context2, ids2, trace = _ctx(items).retrieve_with_trace("q", token_budget=200)

    assert context == context2
    assert ids == ids2
    assert [t["id"] for t in trace] == ids2
    assert trace[0]["score"] == pytest.approx(1.0)
    assert trace[1]["score"] == pytest.approx(0.9)
    assert trace[0]["stage1_rank"] == 1
    assert trace[0]["tier"] == "STM"
    assert trace[0]["text"]


def test_retrieve_with_trace_rejects_bad_budget():
    with pytest.raises(ValueError):
        _ctx([]).retrieve_with_trace("q", token_budget=0)


# ---------------------------------------------------------------------------
# AgentResponse.retrieval_trace
# ---------------------------------------------------------------------------

def test_agent_chat_populates_retrieval_trace():
    from src.agent import MemAgeAgent

    store = _FakeStore({"s1__0": _make_memory("s1__0", "User lives in Chennai")})
    agent = MemAgeAgent(store, llm=_fake_llm(answer="You live in Chennai."))
    resp = agent.chat("Where do I live?", session_id="s1")

    assert resp.ok
    assert resp.retrieved_ids == ["s1__0"]
    assert isinstance(resp.retrieval_trace, list)
    assert [t["id"] for t in resp.retrieval_trace] == resp.retrieved_ids
    assert resp.retrieval_trace[0]["tier"] == "STM"
    assert resp.retrieval_trace[0]["score"] is not None


def test_agent_chat_retrieval_failure_yields_empty_trace():
    from src.agent import MemAgeAgent

    class _BoomStore(_FakeStore):
        def query(self, text, k=10, tier="all"):
            raise RuntimeError("store offline")

    agent = MemAgeAgent(_BoomStore(), llm=_fake_llm(answer="no idea"))
    resp = agent.chat("anything", session_id="s1")
    assert resp.retrieved_ids == []
    assert resp.retrieval_trace == []


# ---------------------------------------------------------------------------
# Server: run-request validation
# ---------------------------------------------------------------------------

def test_benchmark_argv_valid_compare():
    from src.server import _build_benchmark_argv

    err, argv, name = _build_benchmark_argv(
        {"dataset": "locomo", "kind": "compare",
         "strategies": ["none", "full_history"], "sample": 2}
    )
    assert err is None
    assert argv[:4] == ["--dataset", "locomo", "--compare", "none,full_history"]
    assert ["--sample", "2"] == argv[4:6]
    assert "locomo_compare_none-full_history_n2" == name


def test_benchmark_argv_rejects_unknown_strategy():
    from src.server import _build_benchmark_argv

    err, argv, _ = _build_benchmark_argv(
        {"dataset": "locomo", "kind": "compare", "strategies": ["gpt5"]}
    )
    assert err and "unknown strategies" in err
    assert argv is None


@pytest.mark.parametrize("spec,expected", [
    ({"kind": "stage"}, "stage"),
    ({"kind": "k", "values": [1, 3, 5]}, "k:1,3,5"),
    ({"kind": "budget", "values": [128, 512]}, "budget:128,512"),
    ({"kind": "signal", "signal": "importance"}, "signal:importance"),
])
def test_benchmark_argv_ablation_kinds(spec, expected):
    from src.server import _build_benchmark_argv

    err, argv, _ = _build_benchmark_argv(
        {"dataset": "synthetic", "kind": "ablate", "ablation": spec}
    )
    assert err is None
    assert argv[argv.index("--ablate") + 1] == expected


def test_benchmark_argv_validation_errors():
    from src.server import _build_benchmark_argv

    # unknown dataset
    err, _, _ = _build_benchmark_argv({"dataset": "hotpot", "kind": "compare",
                                       "strategies": ["none"]})
    assert err and "dataset" in err
    # sample cap (quota safety)
    err, _, _ = _build_benchmark_argv({"dataset": "locomo", "kind": "compare",
                                       "strategies": ["none"], "sample": 500})
    assert err and "sample" in err
    # bad signal name
    err, _, _ = _build_benchmark_argv({"dataset": "synthetic", "kind": "ablate",
                                       "ablation": {"kind": "signal", "signal": "mood"}})
    assert err and "signal" in err
    # name sanitisation
    err, _, name = _build_benchmark_argv({"dataset": "synthetic", "kind": "compare",
                                          "strategies": ["none"],
                                          "name": "my run/../../etc"})
    assert err is None
    assert "/" not in name and " " not in name


# ---------------------------------------------------------------------------
# Server: artifact parsing
# ---------------------------------------------------------------------------

def test_load_experiments_parses_comparison_and_ablation(tmp_path, monkeypatch):
    import src.server as server

    monkeypatch.setattr(server, "_RESULTS_DIR", tmp_path)

    comparison = {
        "dataset_name": "locomo", "k": 5, "num_cases": 2,
        "results": [{"name": "none", "report": {"mean_f1": 0.4,
                                                 "mean_bleu1": 0.3,
                                                 "num_queries": 2,
                                                 "mean_latency_ms": 12.5}}],
    }
    ablation = {
        "ablation": "retrieval_stage",
        "configs": [{"kind": "retrieval_stage", "label": "stage=coarse_only",
                     "params": {"stage": "coarse_only"}}],
        "comparison": {"dataset_name": "locomo", "k": 5, "num_cases": 2,
                       "results": [{"name": "stage=coarse_only",
                                    "report": {"mean_f1": 0.3}}]},
    }
    (tmp_path / "locomo_compare.json").write_text(json.dumps(comparison), encoding="utf-8")
    (tmp_path / "locomo_stage.json").write_text(json.dumps(ablation), encoding="utf-8")
    (tmp_path / "junk.json").write_text('{"not": "a result"}', encoding="utf-8")

    exps = server._load_experiments()
    files = {e["file"] for e in exps}
    assert files == {"locomo_compare.json", "locomo_stage.json"}  # junk skipped

    comp = next(e for e in exps if e["type"] == "comparison")
    assert comp["dataset_name"] == "locomo"
    assert comp["rows"][0]["name"] == "none"
    assert comp["rows"][0]["mean_f1"] == 0.4

    abl = next(e for e in exps if e["type"] == "ablation")
    assert abl["ablation"] == "retrieval_stage"
    assert abl["rows"][0]["kind"] == "retrieval_stage"
    assert abl["rows"][0]["mean_f1"] == 0.3
