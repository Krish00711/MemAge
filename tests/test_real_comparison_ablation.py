"""H-5 tests: real-dataset ``--compare`` / ``--ablate`` (LoCoMo & LongMemEval).

Covers the real-data wiring added on top of the EXISTING abstractions —
nothing here reimplements comparison/ablation/metrics:

* ``StrategySpec.memory_prepare_fn`` -> ``BenchmarkRunner`` (per-case isolation)
* ``build_real_comparison_specs`` / ``run_dataset_comparison`` (reuse of the
  H-4 builders for all five strategies, shared answer function)
* ``build_real_ablation_spec`` / ``run_dataset_ablation`` (one factor varied
  per config over the same real pipeline)
* genuinely functional signal ablation (controller thresholds applied to
  ingestion — store contents and metrics actually change)

Everything is offline: tiny local fixtures (H-4 module), the hash embedder,
and fake LLMs (rerank + controller scoring + answers) — no Groq calls, no
model downloads, no dataset downloads.
"""

from __future__ import annotations

import json
import re

import pytest

from src.eval.run_benchmark import main
from tests.test_real_none_baseline import (
    _FakeLLM,
    _HashEmbedder,
    _patch_llm,
    _patch_store_embedder,
    _write_longmemeval_fixture,
    _write_locomo_fixture,
)


# ---------------------------------------------------------------------------
# Fakes for controller scoring / rerank ordering
# ---------------------------------------------------------------------------


class _SignalLLM(_FakeLLM):
    """Fake LLM that also answers the controller's scoring prompts.

    Deterministic bands chosen so thresholds can genuinely fire:
    importance 0.5 (kept by 0.0/0.3, suppressed by 0.6), contradiction
    configurable (0.1 = below every update threshold; 0.55 = UPDATE for
    loose/default but not strict).
    """

    def __init__(
        self,
        answer: str = "OK",
        importance: float = 0.5,
        contradiction: float = 0.1,
    ) -> None:
        super().__init__(answer=answer)
        self.importance = importance
        self.contradiction = contradiction

    def __call__(self, prompt: str, system: str | None = None, **kwargs) -> str:
        if system and "importance scorer" in system:
            self.calls.append(prompt)
            return json.dumps({"importance": self.importance})
        if system and "contradiction detector" in system:
            self.calls.append(prompt)
            return json.dumps({"contradiction": self.contradiction})
        return super().__call__(prompt, system=system, **kwargs)


class _ReversingLLM(_FakeLLM):
    """Rerank replies that REVERSE Stage-1 order — a deterministic stage contrast."""

    def __call__(self, prompt: str, system: str | None = None, **kwargs) -> str:
        if "relevance reranker" in prompt:
            self.calls.append(prompt)
            ids = list(
                re.findall(r"^- ID: (.+)$", prompt, flags=re.MULTILINE)
            )
            ids.reverse()
            rankings = [
                {"id": eid, "score": max(0.0, round(0.9 - 0.05 * i, 3))}
                for i, eid in enumerate(ids)
            ]
            return json.dumps({"rankings": rankings})
        return super().__call__(prompt, system=system, **kwargs)


# ---------------------------------------------------------------------------
# Programmatic helpers (same flow the CLI runs, structured results out)
# ---------------------------------------------------------------------------


def _locomo(tmp_path):
    from src.eval.adapters import load_dataset

    return load_dataset("locomo", data_dir=str(tmp_path))


def _run_real_comparison(names, tmp_path, k=5, token_budget=512):
    """build_real_comparison_specs -> compare_strategies (the CLI's flow)."""
    from src.eval.comparison import compare_strategies
    from src.eval.run_benchmark import build_real_comparison_specs

    specs, answer_fn = build_real_comparison_specs(
        names, token_budget, data_dir=str(tmp_path), embedder=_HashEmbedder()
    )
    assert specs is not None, "all five strategies must build with injected deps"
    return compare_strategies(_locomo(tmp_path), specs, answer_fn, k=k)


def _run_real_ablation(configs, ablation, tmp_path, k=5, token_budget=512,
                       dataset=None):
    """run_ablation with build_real_ablation_spec; records built specs."""
    from src.eval.ablation import run_ablation
    from src.eval.run_benchmark import _build_answer_fn, build_real_ablation_spec

    recorded = {}

    def builder(cfg):
        spec = build_real_ablation_spec(
            cfg, token_budget=token_budget, embedder=_HashEmbedder()
        )
        assert spec is not None, f"config {cfg.label!r} must build"
        recorded[cfg.label] = spec
        return spec

    result = run_ablation(
        dataset if dataset is not None else _locomo(tmp_path),
        configs,
        builder,
        _build_answer_fn(_SignalLLM()),
        k=k,
        ablation=ablation,
    )
    return result, recorded


def _state(spec):
    """Diagnostic state of a built spec's memory_prepare (populated after a run)."""
    return spec.memory_prepare_fn.state


# ---------------------------------------------------------------------------
# 1-2. Real comparison on tiny fixtures (LoCoMo / LongMemEval), CLI level
# ---------------------------------------------------------------------------


def test_locomo_comparison_real_via_cli(capsys, tmp_path, monkeypatch):
    _write_locomo_fixture(tmp_path)
    _patch_llm(monkeypatch, _FakeLLM(answer="Chennai"))
    _patch_store_embedder(monkeypatch, _HashEmbedder())

    code = main(
        [
            "--dataset", "locomo",
            "--compare", "none,full_history,memoryos",
            "--data-dir", str(tmp_path),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Loaded locomo: 2 cases" in out
    assert "Comparison (locomo, k=5): 2 cases, 3 strategies" in out
    assert "Strategy" in out and "Recall@5" in out and "MRR" in out
    for name in ["none", "full_history", "memoryos"]:
        assert any(line.startswith(name) for line in out.splitlines()), name


def test_longmemeval_comparison_real_via_cli(capsys, tmp_path, monkeypatch):
    _write_longmemeval_fixture(tmp_path)
    _patch_llm(monkeypatch, _FakeLLM(answer="Whiskers"))
    _patch_store_embedder(monkeypatch, _HashEmbedder())

    code = main(
        [
            "--dataset", "longmemeval",
            "--compare", "none,full_history,memory_r1,hmem",
            "--data-dir", str(tmp_path),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Loaded longmemeval: 1 cases" in out
    assert "Comparison (longmemeval, k=5): 1 cases, 4 strategies" in out
    assert "Strategy" in out and "MRR" in out
    for name in ["none", "full_history", "memory_r1", "hmem"]:
        assert any(line.startswith(name) for line in out.splitlines()), name


# ---------------------------------------------------------------------------
# 3 + 7. All five strategies in one comparison; ComparisonResult structure
# ---------------------------------------------------------------------------


def test_all_five_strategies_one_real_comparison(tmp_path, monkeypatch):
    _write_locomo_fixture(tmp_path)
    _patch_llm(monkeypatch, _FakeLLM(answer="Chennai"))

    names = ["none", "full_history", "memoryos", "memory_r1", "hmem"]
    result = _run_real_comparison(names, tmp_path)

    assert result.names == names            # input order preserved
    assert result.num_cases == 2
    assert result.k == 5
    for name in ["none", "memoryos", "memory_r1", "hmem"]:
        rep = result.get(name).report
        assert rep.num_queries == 2
        # entry/pipeline strategies rank -> retrieval metrics are real numbers
        assert rep.mrr is not None
        assert rep.mean_recall_at_k is not None
        assert rep.mean_precision_at_k is not None
    # full_history formats the entire history (no ranking) -> metrics are None
    full = result.get("full_history").report
    assert full.num_queries == 2
    assert full.mrr is None
    assert full.mean_recall_at_k is None


def test_comparison_result_structure_and_table(tmp_path, monkeypatch):
    _write_locomo_fixture(tmp_path)
    _patch_llm(monkeypatch, _FakeLLM())

    from src.eval.comparison import format_comparison_table

    result = _run_real_comparison(["none", "full_history"], tmp_path, k=3)
    assert result.dataset_name == "locomo"
    assert result.k == 3                    # metric cutoff flows through
    assert result.num_cases == 2
    assert result.names == ["none", "full_history"]
    assert result.get("none").report.num_queries == 2
    with pytest.raises(KeyError):
        result.get("nope")

    table = format_comparison_table(result)
    lines = table.splitlines()
    assert lines[0].startswith("Strategy")
    assert "Recall@3" in lines[0] and "P@3" in lines[0]
    assert any(line.startswith("none") for line in lines)
    assert any(line.startswith("full_history") for line in lines)


# ---------------------------------------------------------------------------
# 4. Strategy order independence
# ---------------------------------------------------------------------------


def test_strategy_order_independence_real(tmp_path, monkeypatch):
    _write_locomo_fixture(tmp_path)
    _patch_llm(monkeypatch, _FakeLLM())

    names = ["none", "full_history", "memoryos", "memory_r1", "hmem"]
    forward = _run_real_comparison(names, tmp_path)
    backward = _run_real_comparison(list(reversed(names)), tmp_path)

    assert backward.names == list(reversed(names))
    for name in names:
        a = forward.get(name).report
        b = backward.get(name).report
        # identical aggregates regardless of strategy execution order
        assert a.mean_f1 == b.mean_f1
        assert a.mean_bleu1 == b.mean_bleu1
        assert a.mrr == b.mrr
        assert a.mean_recall_at_k == b.mean_recall_at_k
        assert a.mean_tokens == b.mean_tokens


# ---------------------------------------------------------------------------
# 5-6. Case isolation and meaningful retrieval IDs
# ---------------------------------------------------------------------------


def test_case_isolation_and_meaningful_ids_real(tmp_path, monkeypatch):
    _write_locomo_fixture(tmp_path)
    _patch_llm(monkeypatch, _FakeLLM())

    result = _run_real_comparison(["none", "memoryos"], tmp_path)
    for name in ["none", "memoryos"]:
        per = {r.case_id: r for r in result.get(name).report.per_case}
        c1, c2 = per["sample_1::qa0"], per["sample_2::qa0"]
        # Each case only ever sees its OWN dataset memory IDs (isolation).
        assert set(c1.retrieved_ids) <= {"D1:1", "D1:2"}
        assert set(c2.retrieved_ids) <= {"D2:1"}
        assert not any(i.startswith("D2") for i in c1.retrieved_ids)
        assert not any(i.startswith("D1") for i in c2.retrieved_ids)
        # IDs are meaningful: the relevant memory is actually retrieved.
        assert c1.retrieval_recall == pytest.approx(1.0)
        assert c2.retrieval_recall == pytest.approx(1.0)
        assert c1.relevant_ids == ["D1:1"]


def test_real_comparison_does_not_mutate_cases(tmp_path, monkeypatch):
    _write_locomo_fixture(tmp_path)
    _patch_llm(monkeypatch, _FakeLLM())

    import copy

    dataset = _locomo(tmp_path)
    before = copy.deepcopy(
        [case.metadata.get("turns") for case in dataset.cases]
    )
    _run_real_comparison(["none", "full_history", "memoryos"], tmp_path)
    after = [case.metadata.get("turns") for case in dataset.cases]
    assert after == before


# ---------------------------------------------------------------------------
# 8-10. Real ablations: stage / k / budget
# ---------------------------------------------------------------------------


def test_real_stage_ablation(tmp_path, monkeypatch):
    from src.eval.ablation import PassthroughReranker, run_ablation, stage_configs
    from src.eval.run_benchmark import _build_answer_fn, build_real_ablation_spec
    from src.retrieval.reranker import LLMReranker

    _write_locomo_fixture(tmp_path)
    configs = stage_configs()

    # One fake per config so rerank-call attribution is exact.
    fake_coarse = _ReversingLLM()
    _patch_llm(monkeypatch, fake_coarse)
    spec_coarse = build_real_ablation_spec(
        configs[0], token_budget=512, embedder=_HashEmbedder()
    )
    fake_two = _ReversingLLM()
    _patch_llm(monkeypatch, fake_two)
    spec_two = build_real_ablation_spec(
        configs[1], token_budget=512, embedder=_HashEmbedder()
    )
    assert spec_coarse is not None and spec_two is not None

    prebuilt = {configs[0].label: spec_coarse, configs[1].label: spec_two}
    result = run_ablation(
        _locomo(tmp_path),
        configs,
        lambda cfg: prebuilt[cfg.label],
        _build_answer_fn(fake_two),
        k=5,
        ablation="retrieval_stage",
    )

    # Existing abstractions: labels, metric cutoff fixed across configs.
    assert result.labels == ["stage=coarse_only", "stage=two_stage"]
    assert result.comparison.k == 5

    # coarse_only bypasses Stage-2 with the existing PassthroughReranker...
    assert isinstance(_state(spec_coarse)["ctx"].reranker, PassthroughReranker)
    assert isinstance(_state(spec_two)["ctx"].reranker, LLMReranker)
    # ...so it never issues a rerank call, while two_stage does.
    assert not any("relevance reranker" in p for p in fake_coarse.calls)
    assert any("relevance reranker" in p for p in fake_two.calls)

    # The reversal fake makes the two stages genuinely rank differently:
    # coarse keeps Stage-1 order (relevant first) -> MRR 1.0; two_stage
    # reverses it -> case1 RR 0.5, case2 RR 1.0 -> mean 0.75.
    m_coarse = result.get("stage=coarse_only").report.mrr
    m_two = result.get("stage=two_stage").report.mrr
    assert m_coarse == pytest.approx(1.0)
    assert m_two == pytest.approx(0.75)
    assert m_coarse != m_two


def test_real_k_ablation(tmp_path, monkeypatch):
    from src.eval.ablation import k_configs

    _write_locomo_fixture(tmp_path)
    fake = _FakeLLM()
    _patch_llm(monkeypatch, fake)

    configs = k_configs([1, 3])
    result, recorded = _run_real_ablation(configs, "k", tmp_path)

    assert result.labels == ["k=1", "k=3"]
    assert result.comparison.k == 5           # metric cutoff NOT varied
    # Stage-1 candidate count really varied per config.
    assert _state(recorded["k=1"])["ctx"].k == 1
    assert _state(recorded["k=3"])["ctx"].k == 3
    # Fewer Stage-1 candidates -> fewer IDs flow into metrics:
    # case1 has 2 memories (k=3 sees both, k=1 only the top one),
    # case2 has 1 memory (both configs retrieve it).
    counts1 = [len(r.retrieved_ids) for r in result.get("k=1").report.per_case]
    counts3 = [len(r.retrieved_ids) for r in result.get("k=3").report.per_case]
    assert counts1 == [1, 1]
    assert counts3 == [2, 1]
    p1 = result.get("k=1").report.mean_precision_at_k
    p3 = result.get("k=3").report.mean_precision_at_k
    r1 = result.get("k=1").report.mean_recall_at_k
    r3 = result.get("k=3").report.mean_recall_at_k
    assert p1 is not None and p3 is not None
    # precision@5's denominator is the number retrieved: k=3 pulls in the
    # extra non-relevant D1:2, so precision drops while recall holds.
    assert p1 > p3
    assert r1 <= r3


def test_real_budget_ablation(tmp_path, monkeypatch):
    from src.eval.ablation import budget_configs

    _write_locomo_fixture(tmp_path)
    _patch_llm(monkeypatch, _FakeLLM())

    configs = budget_configs([12, 512])
    result, _recorded = _run_real_ablation(configs, "token_budget", tmp_path)

    assert result.labels == ["token_budget=12", "token_budget=512"]
    assert result.comparison.k == 5           # metric cutoff NOT varied
    # The config's own budget drives the pipeline (params["token_budget"],
    # not the builder's default): at 12 tokens case1's second memory block
    # (~10 tokens + separator) no longer fits, at 512 it does.
    counts12 = [
        len(r.retrieved_ids)
        for r in result.get("token_budget=12").report.per_case
    ]
    counts512 = [
        len(r.retrieved_ids)
        for r in result.get("token_budget=512").report.per_case
    ]
    assert counts12 == [1, 1]
    assert counts512 == [2, 1]
    p12 = result.get("token_budget=12").report.mean_precision_at_k
    p512 = result.get("token_budget=512").report.mean_precision_at_k
    r12 = result.get("token_budget=12").report.mean_recall_at_k
    r512 = result.get("token_budget=512").report.mean_recall_at_k
    assert p12 is not None and p512 is not None
    # Tighter budget drops the extra non-relevant memory -> precision rises
    # (denominator = number retrieved); recall is unchanged on this fixture.
    assert p12 > p512
    assert r12 <= r512


# ---------------------------------------------------------------------------
# 11. Signal ablation actually changes controller behavior
# ---------------------------------------------------------------------------


def test_real_signal_importance_changes_controller(tmp_path, monkeypatch):
    from src.eval.signal_ablation import importance_configs

    _write_locomo_fixture(tmp_path)
    _patch_llm(monkeypatch, _SignalLLM(importance=0.5, contradiction=0.1))

    configs = importance_configs()
    result, recorded = _run_real_ablation(configs, "signal:importance", tmp_path)

    assert result.labels == [
        "importance_min=0.0", "importance_min=0.3", "importance_min=0.6"
    ]
    # Thresholds were applied: the strict gate stores nothing, the permissive
    # ones store the case's memory (last prepared case = sample_2, 1 turn).
    loose = _state(recorded["importance_min=0.0"])["store"].count()
    mid = _state(recorded["importance_min=0.3"])["store"].count()
    strict = _state(recorded["importance_min=0.6"])["store"].count()
    assert sum(loose.values()) >= 1
    assert sum(mid.values()) >= 1
    assert sum(strict.values()) == 0
    # And the difference shows up in the shared BenchmarkRunner metrics.
    assert result.get("importance_min=0.0").report.mrr == pytest.approx(1.0)
    assert result.get("importance_min=0.6").report.mrr == pytest.approx(0.0)


def test_real_signal_temporal_gates_ingestion(tmp_path, monkeypatch):
    from src.eval.dataset import BenchmarkCase
    from src.eval.signal_ablation import temporal_configs

    _patch_llm(monkeypatch, _SignalLLM())
    # Two turns with IDENTICAL timestamps -> temporal validity 0.5 once a
    # match exists, so temporal_min=0.6 must suppress the second memory.
    case = BenchmarkCase(
        query="What did the user do?",
        reference_answer="adopted a dog",
        relevant_ids=["T1", "T2"],
        metadata={
            "turns": [
                {"id": "T1", "text": "User moved to Lisbon", "speaker": "u",
                 "session": "s1", "timestamp": "2024-03-01T10:00:00"},
                {"id": "T2", "text": "User adopted a dog", "speaker": "u",
                 "session": "s1", "timestamp": "2024-03-01T10:00:00"},
            ]
        },
    )
    configs = temporal_configs()
    result, recorded = _run_real_ablation(
        configs, "signal:temporal", tmp_path, dataset=[case]
    )

    counts = {
        label: sum(_state(spec)["store"].count().values())
        for label, spec in recorded.items()
    }
    assert counts["temporal_min=0.0"] == 2
    assert counts["temporal_min=0.3"] == 2
    assert counts["temporal_min=0.6"] == 1   # second memory gated by time
    assert result.comparison.k == 5


def test_real_signal_contradiction_changes_decisions(tmp_path, monkeypatch):
    from src.eval.dataset import BenchmarkCase
    from src.eval.signal_ablation import contradiction_configs

    _patch_llm(monkeypatch, _SignalLLM(contradiction=0.55))
    case = BenchmarkCase(
        query="Where does the user live?",
        reference_answer="Lisbon",
        relevant_ids=["T1"],
        metadata={
            "turns": [
                {"id": "T1", "text": "User lives in Lisbon", "speaker": "u",
                 "session": "s1", "timestamp": "2024-03-01"},
                {"id": "T2", "text": "User relocated to Porto", "speaker": "u",
                 "session": "s1", "timestamp": "2024-03-02"},
            ]
        },
    )
    configs = contradiction_configs()
    _result, recorded = _run_real_ablation(
        configs, "signal:contradiction", tmp_path, dataset=[case]
    )

    counts = {
        label: sum(_state(spec)["store"].count().values())
        for label, spec in recorded.items()
    }
    # contradiction 0.55: UPDATE for loose (>=0.3) and default (>=0.5),
    # plain ADD for strict (<0.7) -> different store contents per config.
    assert counts["contradiction=loose"] == 1
    assert counts["contradiction=default"] == 1
    assert counts["contradiction=strict"] == 2


def test_real_signal_storage_thresholds_fire_under_pressure(tmp_path, monkeypatch):
    from src.eval.dataset import BenchmarkCase
    from src.eval.signal_ablation import storage_configs

    _patch_llm(monkeypatch, _SignalLLM(importance=0.5, contradiction=0.1))
    # 150 unique turns: STM (20) + MTM (100) capacity 120, so pressure
    # (STM+MTM)/120 crosses 0.7 / 0.9 / 0.95 at different points while
    # importance 0.5 < 0.6 makes low-pressure ADDs suppressible.
    n = 150
    turns = [
        {"id": f"S{i}", "text": f"Unique stored fact {i} with detail {i * 7}",
         "speaker": "u", "session": "s1",
         "timestamp": f"2024-04-01T00:{i % 60:02d}:{i // 60:02d}"}
        for i in range(n)
    ]
    case = BenchmarkCase(
        query="Which fact matters most?",
        reference_answer="fact 42",
        relevant_ids=["S42"],
        metadata={"turns": turns},
    )
    configs = storage_configs()
    _result, recorded = _run_real_ablation(
        configs, "signal:storage", tmp_path, dataset=[case]
    )

    counts = {
        label: sum(_state(spec)["store"].count().values())
        for label, spec in recorded.items()
    }
    c07 = counts["pressure_suppress=0.7"]
    c09 = counts["pressure_suppress=0.9"]
    c095 = counts["pressure_suppress=0.95"]
    # The threshold genuinely gates ingestion: earlier (lower) thresholds
    # suppress more of the 150 low-importance ADDs.
    assert c07 < c09 <= c095
    assert c07 < n                     # 0.7 fired mid-ingestion
    assert c095 == n                   # 0.95 never reached at n=150


def test_cli_real_signal_ablation(capsys, tmp_path, monkeypatch):
    _write_locomo_fixture(tmp_path)
    _patch_llm(monkeypatch, _SignalLLM())
    _patch_store_embedder(monkeypatch, _HashEmbedder())

    code = main(
        [
            "--dataset", "locomo",
            "--ablate", "signal:importance",
            "--data-dir", str(tmp_path),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Ablation signal:importance (locomo, metric_k=5)" in out
    assert "importance_min=0.6" in out
    assert "Strategy" in out and "MRR" in out


def test_synthetic_signal_ablation_is_functional():
    # Thresholds are applied through the real controller policy on the demo
    # memories (the historical record-but-ignore no-op is gone).
    from src.eval.run_benchmark import _SYNTHETIC_MEMORIES, _gate_synthetic_memories

    kept_off = _gate_synthetic_memories(_SYNTHETIC_MEMORIES, {"importance_min": 0.0})
    kept_strict = _gate_synthetic_memories(_SYNTHETIC_MEMORIES, {"importance_min": 0.6})
    assert len(kept_off) >= 1
    assert kept_strict == []               # importance 0.5 < 0.6 -> all NOOP

    kept_loose = _gate_synthetic_memories(
        _SYNTHETIC_MEMORIES,
        {"contradiction_update": 0.3, "contradiction_delete_existing": 0.6},
    )
    kept_strict_c = _gate_synthetic_memories(
        _SYNTHETIC_MEMORIES,
        {"contradiction_update": 0.7, "contradiction_delete_existing": 0.9},
    )
    assert len(kept_loose) == 1            # UPDATE folds the second memory
    assert len(kept_strict_c) == 2         # below update threshold -> ADD

    # Originals are never mutated (repeatability within one process).
    assert [m["id"] for m in _SYNTHETIC_MEMORIES] == ["mem_001", "mem_002"]
    assert _SYNTHETIC_MEMORIES[0]["text"] == "User prefers dark mode"
    assert _SYNTHETIC_MEMORIES[1]["text"] == "User lives in Berlin"


def test_strategy_spec_memory_prepare_validation(tmp_path, monkeypatch):
    from src.eval.comparison import StrategySpec, compare_strategies

    _write_locomo_fixture(tmp_path)
    _patch_llm(monkeypatch, _FakeLLM())

    # Non-callable memory_prepare_fn is rejected like the other callables.
    with pytest.raises(ValueError, match="memory_prepare_fn"):
        compare_strategies(
            _locomo(tmp_path),
            [StrategySpec("bad", lambda q: "", memory_prepare_fn="nope")],
            lambda q, c: "A",
            k=5,
        )
    # None is fine, and the appended field keeps positional construction
    # (name, retrieval_fn) working.
    spec = StrategySpec("ok", lambda q: "")
    assert spec.memory_prepare_fn is None
    result = compare_strategies(_locomo(tmp_path), [spec], lambda q, c: "A", k=5)
    assert result.names == ["ok"]


# ---------------------------------------------------------------------------
# 12 + 15. Dataset-unavailable errors and CLI exit codes
# ---------------------------------------------------------------------------


def test_dataset_unavailable_errors(capsys, tmp_path):
    # empty data dir: unavailable errors stay testable now that the repo
    # ships real datasets under data/
    assert main(["--dataset", "locomo", "--compare", "none,full_history",
                 "--data-dir", str(tmp_path)]) == 2
    assert "not available locally" in capsys.readouterr().err
    assert main(["--dataset", "longmemeval", "--ablate", "k:1,3",
                 "--data-dir", str(tmp_path)]) == 2
    assert "not available locally" in capsys.readouterr().err
    assert main(["--dataset", "locomo", "--ablate", "signal:importance",
                 "--data-dir", str(tmp_path)]) == 2
    assert "not available locally" in capsys.readouterr().err


def test_cli_exit_codes_real_modes(capsys, tmp_path, monkeypatch):
    _write_locomo_fixture(tmp_path)
    _patch_llm(monkeypatch, _FakeLLM())
    _patch_store_embedder(monkeypatch, _HashEmbedder())

    # mutual exclusion still wins (1) regardless of dataset readiness
    assert (
        main(
            ["--dataset", "locomo", "--compare", "none", "--ablate", "stage",
             "--data-dir", str(tmp_path)]
        )
        == 1
    )
    capsys.readouterr()

    # real modes run end-to-end on the fixture -> 0
    assert (
        main(["--dataset", "locomo", "--compare", "none",
              "--data-dir", str(tmp_path)])
        == 0
    )
    out = capsys.readouterr().out
    assert "Comparison (locomo, k=5)" in out and "MRR" in out

    assert (
        main(["--dataset", "locomo", "--ablate", "k:1,3",
              "--data-dir", str(tmp_path)])
        == 0
    )
    out = capsys.readouterr().out
    assert "Ablation k (locomo, metric_k=5)" in out

    # unavailable dataset -> 2 (empty data dir: hermetic even though the
    # repo's data/ folder now contains the real datasets)
    empty = tmp_path / "empty_data_dir"
    assert main(["--dataset", "locomo", "--compare", "none",
                 "--data-dir", str(empty)]) == 2
    assert "not available locally" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 13-14. Synthetic regression: comparison + all ablation families
# ---------------------------------------------------------------------------


def test_synthetic_comparison_regression(capsys):
    code = main(
        [
            "--dataset", "synthetic",
            "--compare", "none,full_history,memoryos,memory_r1,hmem",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Comparison (synthetic_demo, k=5): 2 cases, 5 strategies" in out
    for name in ["none", "full_history", "memoryos", "memory_r1", "hmem"]:
        assert any(line.startswith(name) for line in out.splitlines()), name


def test_synthetic_ablation_regression(capsys):
    assert main(["--dataset", "synthetic", "--ablate", "stage"]) == 0
    assert main(["--dataset", "synthetic", "--ablate", "k:1,3,5"]) == 0
    assert main(["--dataset", "synthetic", "--ablate", "budget:12,64,512"]) == 0
    assert main(["--dataset", "synthetic", "--ablate", "signal:importance"]) == 0
    out = capsys.readouterr().out
    assert "Ablation signal:importance (synthetic_demo, metric_k=5)" in out
    assert "importance_min=0.6" in out
    assert "Strategy" in out and "MRR" in out


# ---------------------------------------------------------------------------
# Export verification for a real comparison and a real ablation
# ---------------------------------------------------------------------------


def test_cli_real_comparison_exports(capsys, tmp_path, monkeypatch):
    _write_locomo_fixture(tmp_path)
    _patch_llm(monkeypatch, _FakeLLM())
    _patch_store_embedder(monkeypatch, _HashEmbedder())

    jpath = tmp_path / "cmp.json"
    cpath = tmp_path / "cmp.csv"
    code = main(
        [
            "--dataset", "locomo",
            "--compare", "none,full_history",
            "--data-dir", str(tmp_path),
            "--export-json", str(jpath),
            "--export-csv", str(cpath),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert f"Exported JSON: {jpath}" in out
    assert f"Exported CSV: {cpath}" in out

    data = json.loads(jpath.read_text(encoding="utf-8"))
    assert data["dataset_name"] == "locomo"
    assert data["k"] == 5
    assert data["num_cases"] == 2
    assert [r["name"] for r in data["results"]] == ["none", "full_history"]
    assert data["results"][0]["report"]["num_queries"] == 2

    csv_lines = cpath.read_text(encoding="utf-8").splitlines()
    assert csv_lines[0].startswith("name,kind,mean_f1")
    assert len(csv_lines) == 3               # header + 2 strategies
    assert csv_lines[1].startswith("none,")
    assert csv_lines[2].startswith("full_history,")


def test_cli_real_ablation_exports(capsys, tmp_path, monkeypatch):
    _write_locomo_fixture(tmp_path)
    _patch_llm(monkeypatch, _ReversingLLM())
    _patch_store_embedder(monkeypatch, _HashEmbedder())

    jpath = tmp_path / "abl.json"
    cpath = tmp_path / "abl.csv"
    code = main(
        [
            "--dataset", "locomo",
            "--ablate", "stage",
            "--data-dir", str(tmp_path),
            "--export-json", str(jpath),
            "--export-csv", str(cpath),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert f"Exported JSON: {jpath}" in out
    assert f"Exported CSV: {cpath}" in out

    data = json.loads(jpath.read_text(encoding="utf-8"))
    assert data["ablation"] == "retrieval_stage"
    assert [c["label"] for c in data["configs"]] == [
        "stage=coarse_only", "stage=two_stage"
    ]
    assert all(c["kind"] == "retrieval_stage" for c in data["configs"])
    assert len(data["comparison"]["results"]) == 2
    assert data["comparison"]["k"] == 5

    csv_lines = cpath.read_text(encoding="utf-8").splitlines()
    assert csv_lines[0].startswith("name,kind,mean_f1")
    assert csv_lines[1].startswith("stage=coarse_only,retrieval_stage,")
    assert csv_lines[2].startswith("stage=two_stage,retrieval_stage,")
