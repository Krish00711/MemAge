"""Tests for src.eval.comparison — multi-strategy orchestration.

Covers: comparing 2+ strategies, all five supported strategies, result
isolation, aggregate metrics, empty dataset/edge cases, CLI comparison mode,
and regression of the existing single-strategy CLI behavior.
No network, no store, no LLM, no serialization.
"""

from dataclasses import asdict, is_dataclass

import pytest

from src.eval.baselines import (
    FullHistoryBaseline,
    HMemBaseline,
    MemoryOSBaseline,
    MemoryR1Baseline,
)
from src.eval.benchmark import BenchmarkRunner, make_retrieval_fn
from src.eval.comparison import (
    ComparisonResult,
    StrategySpec,
    StrategyResult,
    compare_strategies,
    format_comparison_table,
)
from src.eval.dataset import BenchmarkCase, BenchmarkDataset
from src.eval.run_benchmark import (
    _synthetic_answer,
    build_synthetic_strategies,
    main,
)


def _synthetic_dataset():
    cases = [
        BenchmarkCase(
            query="What is the user's preference?",
            reference_answer="User prefers dark mode",
            relevant_ids=["mem_001"],
            case_id="syn_001",
        ),
        BenchmarkCase(
            query="Where does the user live?",
            reference_answer="User lives in Berlin",
            relevant_ids=["mem_002"],
            case_id="syn_002",
        ),
    ]
    return BenchmarkDataset(cases=cases, name="synthetic_demo")


# ---------------------------------------------------------------------------
# Comparing 2+ strategies on the same dataset
# ---------------------------------------------------------------------------


def test_compare_two_strategies_on_same_dataset():
    dataset = _synthetic_dataset()
    strategies = [
        StrategySpec("full_history", FullHistoryBaseline([])),
        StrategySpec("memoryos", MemoryOSBaseline([])),
    ]
    result = compare_strategies(dataset, strategies, _synthetic_answer, k=5)

    assert isinstance(result, ComparisonResult)
    assert result.dataset_name == "synthetic_demo"
    assert result.k == 5
    assert result.num_cases == 2
    assert result.names == ["full_history", "memoryos"]
    for r in result.results:
        assert isinstance(r, StrategyResult)
        assert r.report.num_queries == 2  # same dataset for every strategy
    assert result.get("full_history").name == "full_history"
    with pytest.raises(KeyError):
        result.get("nope")


# ---------------------------------------------------------------------------
# All currently supported strategies
# ---------------------------------------------------------------------------


def test_all_supported_strategies_in_one_comparison():
    names = ["none", "full_history", "memoryos", "memory_r1", "hmem"]
    specs = build_synthetic_strategies(names, token_budget=512, k=5)
    assert [s.name for s in specs] == names

    result = compare_strategies(_synthetic_dataset(), specs, _synthetic_answer, k=5)
    assert result.names == names
    assert len(result.results) == 5
    for r in result.results:
        assert r.report.num_queries == 2
        assert r.report.mean_f1 == pytest.approx(1.0)
        assert r.report.mean_bleu1 == pytest.approx(1.0)
        assert r.report.mean_tokens is not None and r.report.mean_tokens > 0
        assert r.report.mean_latency_ms is not None
        assert r.report.per_case and len(r.report.per_case) == 2


def test_retrieval_metrics_present_where_supported():
    specs = build_synthetic_strategies(
        ["none", "full_history", "memoryos", "memory_r1", "hmem"], k=5
    )
    result = compare_strategies(_synthetic_dataset(), specs, _synthetic_answer, k=5)

    for name in ["none", "memoryos", "memory_r1", "hmem"]:
        rep = result.get(name).report
        assert rep.mean_recall_at_k == pytest.approx(1.0), name
        assert rep.mean_precision_at_k is not None, name
        assert rep.mrr is not None, name
    # full_history does not rank -> retrieval metrics are None (as in single mode)
    rep_fh = result.get("full_history").report
    assert rep_fh.mean_recall_at_k is None
    assert rep_fh.mean_precision_at_k is None
    assert rep_fh.mrr is None


# ---------------------------------------------------------------------------
# Result isolation between strategies
# ---------------------------------------------------------------------------


def test_dataset_not_mutated_by_comparison():
    dataset = _synthetic_dataset()
    snapshot = [asdict(c) for c in dataset.cases]
    specs = build_synthetic_strategies(
        ["none", "full_history", "memoryos", "memory_r1", "hmem"], k=5
    )
    compare_strategies(dataset, specs, _synthetic_answer, k=5)
    assert [asdict(c) for c in dataset.cases] == snapshot


def test_each_strategy_matches_its_solo_run():
    dataset = _synthetic_dataset()
    specs = build_synthetic_strategies(
        ["none", "full_history", "memoryos", "memory_r1", "hmem"], k=5
    )
    result = compare_strategies(dataset, specs, _synthetic_answer, k=5)

    for spec in specs:
        solo = BenchmarkRunner(
            retrieval_fn=spec.retrieval_fn, answer_fn=_synthetic_answer, k=5
        ).run_dataset(dataset)
        combined = result.get(spec.name).report
        assert combined.mean_f1 == pytest.approx(solo.mean_f1), spec.name
        assert combined.mean_bleu1 == pytest.approx(solo.mean_bleu1), spec.name
        assert combined.mean_recall_at_k == solo.mean_recall_at_k, spec.name
        assert combined.mrr == solo.mrr, spec.name
        assert combined.mean_tokens == pytest.approx(solo.mean_tokens), spec.name


def test_strategy_order_does_not_change_results():
    dataset = _synthetic_dataset()
    names = ["none", "memoryos", "memory_r1", "hmem", "full_history"]
    forward = compare_strategies(
        dataset, build_synthetic_strategies(names, k=5), _synthetic_answer, k=5
    )
    backward = compare_strategies(
        dataset,
        build_synthetic_strategies(list(reversed(names)), k=5),
        _synthetic_answer,
        k=5,
    )
    for name in names:
        a = forward.get(name).report
        b = backward.get(name).report
        assert a.mean_f1 == pytest.approx(b.mean_f1), name
        assert a.mean_tokens == pytest.approx(b.mean_tokens), name
        assert a.mrr == b.mrr, name


def test_no_shared_strategy_instances_between_specs():
    specs = build_synthetic_strategies(
        ["full_history", "memoryos", "memory_r1", "hmem"], k=5
    )
    fns = [s.retrieval_fn for s in specs]
    assert all(callable(f) for f in fns)
    # every builder produced a distinct instance
    assert len({id(f) for f in fns}) == len(fns)


def test_shared_history_list_untouched_after_comparison():
    shared = [
        {"id": "m1", "text": "shared entry one", "timestamp": "t1"},
        {"id": "m2", "text": "shared entry two", "timestamp": "t2"},
    ]
    snapshot = [dict(e) for e in shared]
    strategies = [
        StrategySpec("memoryos", MemoryOSBaseline(shared, k=1)),
        StrategySpec("memory_r1", MemoryR1Baseline(shared, token_budget=512)),
        StrategySpec("hmem", HMemBaseline(shared, token_budget=512)),
    ]
    cases = [
        BenchmarkCase(
            query="shared entry one",
            reference_answer="shared entry one",
            relevant_ids=["m1"],
        )
    ]
    compare_strategies(cases, strategies, _synthetic_answer, k=5)
    assert shared == snapshot  # baselines never mutate the injected history


# ---------------------------------------------------------------------------
# Aggregate metrics + structured result
# ---------------------------------------------------------------------------


def test_aggregate_metrics_cover_all_supported_signals():
    specs = build_synthetic_strategies(["none", "hmem"], k=5)
    result = compare_strategies(_synthetic_dataset(), specs, _synthetic_answer, k=5)
    rep = result.get("none").report
    # exactly the metrics already supported by the project (no new ones)
    for attr in (
        "mean_f1",
        "mean_bleu1",
        "mean_recall_at_k",
        "mean_precision_at_k",
        "mrr",
        "mean_tokens",
        "mean_latency_ms",
        "num_queries",
    ):
        assert hasattr(rep, attr)
    assert rep.num_queries == 2
    # results are plain dataclasses of primitives -> asdict() works for the
    # upcoming serialization task (no serialization implemented here)
    d = asdict(result)
    assert d["dataset_name"] == "synthetic_demo"
    assert d["k"] == 5
    assert d["num_cases"] == 2
    assert [r["name"] for r in d["results"]] == ["none", "hmem"]
    assert is_dataclass(result.results[0].report)  # source object is a dataclass
    assert isinstance(d["results"][0]["report"], dict)  # asdict gives plain dicts


def test_format_comparison_table_output():
    specs = build_synthetic_strategies(
        ["none", "full_history", "memoryos", "memory_r1", "hmem"], k=5
    )
    result = compare_strategies(_synthetic_dataset(), specs, _synthetic_answer, k=5)
    table = format_comparison_table(result)
    lines = table.splitlines()
    assert lines[0].split() == ["Strategy", "F1", "BLEU-1", "Recall@5", "P@5", "MRR", "Tokens", "Latency(ms)"]
    for name in ["none", "full_history", "memoryos", "memory_r1", "hmem"]:
        assert any(line.startswith(name) for line in lines[1:])
    # full_history has no retrieval metrics -> dashes in those columns
    fh_line = [l for l in lines if l.startswith("full_history")][0]
    assert "  -  " in fh_line or fh_line.rstrip().endswith("-") or " -" in fh_line
    # deterministic: same input -> identical table
    assert format_comparison_table(result) == table


# ---------------------------------------------------------------------------
# Empty dataset / edge cases
# ---------------------------------------------------------------------------


def test_empty_dataset_yields_zeroed_reports():
    empty = BenchmarkDataset(cases=[], name="empty")
    specs = build_synthetic_strategies(["none", "full_history"], k=5)
    result = compare_strategies(empty, specs, _synthetic_answer, k=5)
    assert result.num_cases == 0
    assert len(result.results) == 2
    for r in result.results:
        assert r.report.num_queries == 0
        assert r.report.mean_f1 == 0.0
        assert r.report.mean_bleu1 == 0.0
        assert r.report.mean_tokens is None
        assert r.report.mean_recall_at_k is None
        assert r.report.mrr is None
    # table still renders
    assert "Strategy" in format_comparison_table(result)


def test_empty_strategies_rejected():
    with pytest.raises(ValueError):
        compare_strategies(_synthetic_dataset(), [], _synthetic_answer, k=5)


def test_invalid_arguments_rejected():
    specs = build_synthetic_strategies(["none"], k=5)
    with pytest.raises(ValueError):
        compare_strategies(_synthetic_dataset(), specs, _synthetic_answer, k=0)
    with pytest.raises(TypeError):
        compare_strategies(_synthetic_dataset(), specs, "not-callable", k=5)
    with pytest.raises(TypeError):
        compare_strategies(_synthetic_dataset(), "none", _synthetic_answer, k=5)
    # duplicate / bad names
    with pytest.raises(ValueError):
        compare_strategies(
            _synthetic_dataset(),
            [StrategySpec("x", lambda q: ""), StrategySpec("x", lambda q: "")],
            _synthetic_answer,
            k=5,
        )
    with pytest.raises(ValueError):
        compare_strategies(
            _synthetic_dataset(), [StrategySpec("", lambda q: "")], _synthetic_answer, k=5
        )
    with pytest.raises(ValueError):
        compare_strategies(
            _synthetic_dataset(), [StrategySpec("bad", "not-callable")], _synthetic_answer, k=5
        )
    with pytest.raises(ValueError):
        build_synthetic_strategies(["does_not_exist"])


def test_per_strategy_answer_fn_override():
    def always_one(query, context):
        return "User prefers dark mode"

    strategies = [
        StrategySpec("custom", FullHistoryBaseline([]), answer_fn=always_one),
        StrategySpec("default", FullHistoryBaseline([])),
    ]
    result = compare_strategies(
        _synthetic_dataset(), strategies, _synthetic_answer, k=5
    )
    # "custom" used the override: exact on case 1, partial overlap on case 2
    # ((1.0 + 0.25) / 2 = 0.625) -- proves this spec's answer_fn was applied
    assert result.get("custom").report.mean_f1 == pytest.approx(0.625)
    # "default" used the shared _synthetic_answer with EMPTY context -> both
    # cases answer "Unknown" -> F1 0.0 (i.e. the override did NOT leak)
    assert result.get("default").report.mean_f1 == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CLI comparison mode
# ---------------------------------------------------------------------------


def test_cli_compare_two_strategies(capsys):
    code = main(["--dataset", "synthetic", "--compare", "none,full_history"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Comparison (synthetic_demo, k=5): 2 cases, 2 strategies" in out
    assert "Strategy" in out and "Recall@5" in out and "MRR" in out
    assert "none" in out and "full_history" in out


def test_cli_compare_all_strategies(capsys):
    code = main(
        [
            "--dataset",
            "synthetic",
            "--compare",
            "none,full_history,memoryos,memory_r1,hmem",
            "--k",
            "3",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "5 strategies" in out
    assert "Recall@3" in out  # --k flows into the table header
    for name in ["none", "full_history", "memoryos", "memory_r1", "hmem"]:
        assert any(l.startswith(name) for l in out.splitlines()), name


def test_cli_compare_errors(capsys, tmp_path):
    # unknown name
    assert main(["--dataset", "synthetic", "--compare", "none,bogus"]) == 1
    assert "unknown baseline" in capsys.readouterr().err
    # empty list
    assert main(["--dataset", "synthetic", "--compare", ",, "]) == 1
    assert "at least one baseline" in capsys.readouterr().err
    # duplicates
    assert main(["--dataset", "synthetic", "--compare", "none,none"]) == 1
    assert "duplicate" in capsys.readouterr().err
    # non-synthetic dataset with no local files: same actionable exit 2 as
    # the single-baseline path (H-5 lifted the synthetic-only guard; real
    # datasets run when their files exist under --data-dir).  The empty
    # data dir keeps this hermetic now that the repo ships real datasets.
    assert main(["--dataset", "locomo", "--compare", "none",
                 "--data-dir", str(tmp_path)]) == 2
    assert "not available locally" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Regression: single-strategy CLI behavior unchanged
# ---------------------------------------------------------------------------


def test_cli_single_baseline_regression(capsys):
    # default single mode (no --compare): title unchanged, table absent
    assert main(["--dataset", "synthetic"]) == 0
    out0 = capsys.readouterr().out
    assert "Synthetic benchmark (synthetic_demo" in out0
    assert "Mean F1" in out0
    assert "Strategy" not in out0.splitlines()[0]

    # each single baseline still runs with its established title
    for baseline, title in [
        ("full_history", "Synthetic benchmark (synthetic_demo, baseline=full_history)"),
        ("memoryos", "Synthetic benchmark (synthetic_demo, baseline=memoryos)"),
        ("memory_r1", "Synthetic benchmark (synthetic_demo, baseline=memory_r1)"),
        ("hmem", "Synthetic benchmark (synthetic_demo, baseline=hmem)"),
    ]:
        assert main(["--dataset", "synthetic", "--baseline", baseline]) == 0
        out = capsys.readouterr().out
        assert title in out


def test_cli_single_baseline_ignores_compare_absent(capsys):
    # token-budget propagation still works in single mode
    assert main(["--dataset", "synthetic", "--token-budget", "12"]) == 0
    out = capsys.readouterr().out
    tokens = float([l for l in out.splitlines() if "Mean tokens" in l][0].split(":")[1])
    assert tokens <= 12
