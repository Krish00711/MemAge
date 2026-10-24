"""Tests for src.eval.ablation — ablation experiment infrastructure.

Covers: each ablation type (stage / k / budget), multiple K values, multiple
token budgets, coarse-only vs two-stage, deterministic config naming, result
isolation, structured results, CLI execution, invalid/empty configurations,
and regression of existing comparison + single-strategy CLI behavior.
No network, no store, no LLM, no serialization.
"""

from dataclasses import asdict, is_dataclass

import pytest

from src.eval.ablation import (
    STAGE_VALUES,
    AblationConfig,
    AblationResult,
    PassthroughReranker,
    budget_configs,
    k_configs,
    run_ablation,
    stage_configs,
)
from src.eval.benchmark import BenchmarkRunner, make_retrieval_fn
from src.eval.comparison import StrategySpec, compare_strategies, format_comparison_table
from src.eval.dataset import BenchmarkCase, BenchmarkDataset
from src.eval.run_benchmark import (
    _synthetic_answer,
    build_synthetic_ablation_spec,
    build_synthetic_strategies,
    main,
)


def _dataset():
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


def _run(configs, ablation, k=5, token_budget=512):
    return run_ablation(
        _dataset(),
        configs,
        lambda cfg: build_synthetic_ablation_spec(cfg, token_budget=token_budget),
        _synthetic_answer,
        k=k,
        ablation=ablation,
    )


# ---------------------------------------------------------------------------
# Config constructors / deterministic naming
# ---------------------------------------------------------------------------


def test_stage_configs_fixed_pair_and_labels():
    configs = stage_configs()
    assert len(configs) == 2
    assert [c.label for c in configs] == ["stage=coarse_only", "stage=two_stage"]
    assert [c.params["stage"] for c in configs] == list(STAGE_VALUES)
    assert all(c.kind == "retrieval_stage" for c in configs)
    # deterministic: repeated construction is identical
    assert stage_configs() == configs


def test_k_configs_multiple_values_and_labels():
    configs = k_configs([1, 3, 5])
    assert [c.label for c in configs] == ["k=1", "k=3", "k=5"]
    assert [c.params["k"] for c in configs] == [1, 3, 5]
    assert all(c.kind == "k" for c in configs)
    assert k_configs([1, 3, 5]) == configs  # deterministic naming


def test_budget_configs_multiple_values_and_labels():
    configs = budget_configs([5, 12, 512])
    assert [c.label for c in configs] == ["token_budget=5", "token_budget=12", "token_budget=512"]
    assert [c.params["token_budget"] for c in configs] == [5, 12, 512]
    assert all(c.kind == "token_budget" for c in configs)
    assert budget_configs([5, 12, 512]) == configs


def test_invalid_config_values_rejected():
    for bad in ([], [0], [-1], [2.5], [True], ["3"], "123"):
        with pytest.raises((ValueError, TypeError)):
            k_configs(bad)
        with pytest.raises((ValueError, TypeError)):
            budget_configs(bad)
    with pytest.raises(ValueError):  # duplicates -> duplicate labels
        k_configs([1, 1])
    with pytest.raises(ValueError):
        budget_configs([512, 512])


# ---------------------------------------------------------------------------
# Ablation type 1: retrieval stage (coarse-only vs two-stage)
# ---------------------------------------------------------------------------


def test_stage_ablation_coarse_only_vs_two_stage():
    result = _run(stage_configs(), ablation="retrieval_stage")
    assert result.ablation == "retrieval_stage"
    assert result.labels == ["stage=coarse_only", "stage=two_stage"]

    coarse = result.get("stage=coarse_only").report
    two = result.get("stage=two_stage").report
    # same dataset, same budget, same metric k -> same QA metrics here
    for rep in (coarse, two):
        assert rep.num_queries == 2
        assert rep.mean_f1 == pytest.approx(1.0)
        assert rep.mean_recall_at_k == pytest.approx(1.0)
    # the stages genuinely differ: reranking reorders (query 2 tie is broken
    # by the lexical reranker), so MRR differs from store-order coarse-only
    assert coarse.mrr == pytest.approx(0.75)
    assert two.mrr == pytest.approx(1.0)
    assert coarse.mrr != two.mrr


def test_stage_ablation_builder_wiring():
    coarse_spec = build_synthetic_ablation_spec(stage_configs()[0])
    two_spec = build_synthetic_ablation_spec(stage_configs()[1])
    assert isinstance(coarse_spec, StrategySpec)
    # both callable retrieval_fns with different behavior on the same query
    out_coarse = coarse_spec.retrieval_fn("Where does the user live?")
    out_two = two_spec.retrieval_fn("Where does the user live?")
    assert out_coarse.retrieved_ids != out_two.retrieved_ids  # order differs
    assert sorted(out_coarse.retrieved_ids) == sorted(out_two.retrieved_ids)


def test_passthrough_reranker_keeps_order_and_copies():
    r = PassthroughReranker()
    items = [{"id": "a"}, {"id": "b"}]
    out = r.rerank("q", items)
    assert [i["id"] for i in out] == ["a", "b"]
    assert out is not items  # returns a copy, never mutates in place
    with pytest.raises(ValueError):
        build_synthetic_ablation_spec(
            AblationConfig(kind="retrieval_stage", label="stage=weird",
                           params={"stage": "weird"})
        )


# ---------------------------------------------------------------------------
# Ablation type 2: K values
# ---------------------------------------------------------------------------


def test_k_ablation_multiple_values():
    result = _run(k_configs([1, 2]), ablation="k")
    assert result.ablation == "k"
    assert result.labels == ["k=1", "k=2"]

    k1 = result.get("k=1").report
    k2 = result.get("k=2").report
    # Stage-1 k=1 only ever returns mem_001 -> query 2 misses entirely
    assert k1.mean_recall_at_k == pytest.approx(0.5)
    assert k1.mean_f1 == pytest.approx(0.5)
    # Stage-1 k=2 returns both memories -> perfect on the synthetic set
    assert k2.mean_recall_at_k == pytest.approx(1.0)
    assert k2.mean_f1 == pytest.approx(1.0)
    # metric cutoff stays fixed (5) for both; only strategy k varies
    assert k1.num_queries == k2.num_queries == 2


def test_k_ablation_spec_uses_config_k():
    spec_one = build_synthetic_ablation_spec(k_configs([1])[0])
    spec_two = build_synthetic_ablation_spec(k_configs([2])[0])
    assert spec_one.retrieval_fn("Where does the user live?").retrieved_ids == ["mem_001"]
    # two-stage reranking ranks the lexically-better match (mem_002) first
    assert spec_two.retrieval_fn("Where does the user live?").retrieved_ids == [
        "mem_002",
        "mem_001",
    ]


# ---------------------------------------------------------------------------
# Ablation type 3: token budgets
# ---------------------------------------------------------------------------


def test_budget_ablation_multiple_values():
    result = _run(budget_configs([5, 12, 512]), ablation="token_budget")
    assert result.ablation == "token_budget"
    assert result.labels == ["token_budget=5", "token_budget=12", "token_budget=512"]

    b5 = result.get("token_budget=5").report
    b12 = result.get("token_budget=12").report
    b512 = result.get("token_budget=512").report
    # nothing fits in 5 tokens; one 11-token block fits in 12; both (22) in 512
    assert b5.mean_tokens == 0.0 and b5.mean_f1 == 0.0
    assert b12.mean_tokens == pytest.approx(11.0) and b12.mean_f1 == pytest.approx(1.0)
    assert b512.mean_tokens == pytest.approx(22.0) and b512.mean_f1 == pytest.approx(1.0)
    # budgets monotonically admit more tokens
    assert b5.mean_tokens < b12.mean_tokens < b512.mean_tokens


def test_budget_ablation_spec_uses_config_budget():
    tiny = build_synthetic_ablation_spec(budget_configs([5])[0])
    roomy = build_synthetic_ablation_spec(budget_configs([512])[0])
    assert tiny.retrieval_fn("preference").context == ""
    assert "[Memory 1]" in roomy.retrieval_fn("preference").context


# ---------------------------------------------------------------------------
# Deterministic naming + structured results
# ---------------------------------------------------------------------------


def test_run_ablation_labels_are_deterministic_and_unique():
    r1 = _run(k_configs([1, 2]), ablation="k")
    r2 = _run(k_configs([1, 2]), ablation="k")
    assert r1.labels == r2.labels == ["k=1", "k=2"]
    assert r1.comparison.names == r1.labels  # labels become strategy names
    assert r1.get("k=1").report.mean_f1 == pytest.approx(r2.get("k=1").report.mean_f1)


def test_structured_result_serialization_ready():
    result = _run(stage_configs(), ablation="retrieval_stage")
    assert isinstance(result, AblationResult)
    d = asdict(result)
    assert d["ablation"] == "retrieval_stage"
    assert [c["label"] for c in d["configs"]] == ["stage=coarse_only", "stage=two_stage"]
    assert d["configs"][0]["params"] == {"stage": "coarse_only"}  # primitives
    assert isinstance(d["configs"][0]["params"], dict)
    assert is_dataclass(result.comparison)  # source objects stay dataclasses
    assert d["comparison"]["dataset_name"] == "synthetic_demo"
    # covers every metric the project supports (no new ones added)
    rep = d["comparison"]["results"][0]["report"]
    for key in (
        "mean_f1", "mean_bleu1", "mean_recall_at_k", "mean_precision_at_k",
        "mrr", "mean_tokens", "mean_latency_ms", "num_queries",
    ):
        assert key in rep


def test_result_accessors():
    result = _run(budget_configs([12, 512]), ablation="token_budget")
    assert result.get("token_budget=12").name == "token_budget=12"
    with pytest.raises(KeyError):
        result.get("token_budget=999")


# ---------------------------------------------------------------------------
# Result isolation
# ---------------------------------------------------------------------------


def test_dataset_not_mutated_by_ablation():
    dataset = _dataset()
    snapshot = [asdict(c) for c in dataset.cases]
    run_ablation(
        dataset,
        k_configs([1, 2]),
        lambda cfg: build_synthetic_ablation_spec(cfg),
        _synthetic_answer,
        k=5,
        ablation="k",
    )
    assert [asdict(c) for c in dataset.cases] == snapshot


def test_each_config_matches_solo_runner_run():
    dataset = _dataset()
    configs = stage_configs() + k_configs([2]) + budget_configs([512])
    result = run_ablation(
        dataset,
        configs,
        lambda cfg: build_synthetic_ablation_spec(cfg),
        _synthetic_answer,
        k=5,
        ablation="mixed",
    )
    for cfg in configs:
        spec = build_synthetic_ablation_spec(cfg)
        solo = BenchmarkRunner(
            retrieval_fn=spec.retrieval_fn, answer_fn=_synthetic_answer, k=5
        ).run_dataset(dataset)
        combined = result.get(cfg.label).report
        assert combined.mean_f1 == pytest.approx(solo.mean_f1), cfg.label
        assert combined.mrr == solo.mrr, cfg.label
        assert combined.mean_tokens == pytest.approx(solo.mean_tokens), cfg.label


def test_config_order_does_not_change_results():
    dataset = _dataset()
    forward = run_ablation(
        dataset, k_configs([1, 2]),
        lambda cfg: build_synthetic_ablation_spec(cfg), _synthetic_answer,
        k=5, ablation="k",
    )
    backward = run_ablation(
        dataset, k_configs([2, 1]),
        lambda cfg: build_synthetic_ablation_spec(cfg), _synthetic_answer,
        k=5, ablation="k",
    )
    for label in ("k=1", "k=2"):
        a, b = forward.get(label).report, backward.get(label).report
        assert a.mean_f1 == pytest.approx(b.mean_f1)
        assert a.mean_recall_at_k == b.mean_recall_at_k


# ---------------------------------------------------------------------------
# Invalid / empty configurations
# ---------------------------------------------------------------------------


def test_run_ablation_invalid_arguments():
    builder = lambda cfg: build_synthetic_ablation_spec(cfg)
    with pytest.raises(ValueError):
        run_ablation(_dataset(), [], builder, _synthetic_answer, ablation="k")
    with pytest.raises(ValueError):
        run_ablation(_dataset(), k_configs([1]), builder, _synthetic_answer, ablation="")
    with pytest.raises(TypeError):
        run_ablation(_dataset(), k_configs([1]), "not-callable", _synthetic_answer,
                     ablation="k")
    with pytest.raises(TypeError):
        run_ablation(_dataset(), "k", builder, _synthetic_answer, ablation="k")
    with pytest.raises(TypeError):  # builder must return StrategySpec
        run_ablation(_dataset(), k_configs([1]), lambda cfg: "oops",
                     _synthetic_answer, ablation="k")
    with pytest.raises(TypeError):  # wrong config type
        run_ablation(_dataset(), ["k=1"], builder, _synthetic_answer, ablation="k")
    with pytest.raises(ValueError):  # duplicate labels
        run_ablation(
            _dataset(),
            [AblationConfig("k", "k=1", {"k": 1}), AblationConfig("k", "k=1", {"k": 2})],
            builder, _synthetic_answer, ablation="k",
        )
    with pytest.raises(ValueError):  # empty label
        run_ablation(_dataset(), [AblationConfig("k", "", {"k": 1})],
                     builder, _synthetic_answer, ablation="k")
    with pytest.raises(ValueError):  # bad metric k
        run_ablation(_dataset(), k_configs([1]), builder, _synthetic_answer,
                     k=0, ablation="k")


def test_empty_dataset_ablation():
    empty = BenchmarkDataset(cases=[], name="empty")
    result = run_ablation(
        empty, stage_configs(),
        lambda cfg: build_synthetic_ablation_spec(cfg),
        _synthetic_answer, k=5, ablation="retrieval_stage",
    )
    assert result.comparison.num_cases == 0
    for r in result.comparison.results:
        assert r.report.num_queries == 0
        assert r.report.mean_f1 == 0.0
        assert r.report.mrr is None
    assert "Strategy" in format_comparison_table(result.comparison)


# ---------------------------------------------------------------------------
# CLI execution
# ---------------------------------------------------------------------------


def test_cli_ablate_stage(capsys):
    code = main(["--dataset", "synthetic", "--ablate", "stage"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Ablation retrieval_stage (synthetic_demo, metric_k=5): 2 cases, 2 configurations" in out
    assert "stage=coarse_only" in out and "stage=two_stage" in out
    assert "Strategy" in out and "Recall@5" in out and "MRR" in out


def test_cli_ablate_multiple_k_values(capsys):
    code = main(["--dataset", "synthetic", "--ablate", "k:1,2", "--k", "5"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Ablation k (synthetic_demo, metric_k=5)" in out
    assert "3 configurations" in out or "2 configurations" in out
    assert any(l.startswith("k=1") for l in out.splitlines())
    assert any(l.startswith("k=2") for l in out.splitlines())


def test_cli_ablate_multiple_budgets(capsys):
    code = main(["--dataset", "synthetic", "--ablate", "budget:5,12,512"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Ablation token_budget" in out
    for label in ("token_budget=5", "token_budget=12", "token_budget=512"):
        assert any(l.startswith(label) for l in out.splitlines()), label
    # budget actually varied: token column shows 0.0 / 11.0 / 22.0
    tokens = [
        float(l.split()[-2]) for l in out.splitlines() if l.startswith("token_budget=")
    ]
    assert tokens == sorted(tokens) and len(tokens) == 3


def test_cli_ablate_invalid_configs(capsys):
    assert main(["--dataset", "synthetic", "--ablate", "bogus"]) == 1
    assert "unknown ablation kind" in capsys.readouterr().err
    assert main(["--dataset", "synthetic", "--ablate", "k"]) == 1
    assert "requires values" in capsys.readouterr().err
    assert main(["--dataset", "synthetic", "--ablate", "k:0"]) == 1
    assert "invalid value" in capsys.readouterr().err
    assert main(["--dataset", "synthetic", "--ablate", "k:abc"]) == 1
    assert main(["--dataset", "synthetic", "--ablate", "k:1,1"]) == 1
    assert "duplicate" in capsys.readouterr().err
    assert main(["--dataset", "synthetic", "--ablate", "budget:"]) == 1
    assert main(["--dataset", "synthetic", "--ablate", "stage:1,2"]) == 1
    assert "takes no values" in capsys.readouterr().err
    # mutually exclusive with --compare
    assert main(["--dataset", "synthetic", "--ablate", "stage", "--compare", "none"]) == 1
    assert "only one" in capsys.readouterr().err
    # non-synthetic dataset
    assert main(["--dataset", "locomo", "--ablate", "stage"]) == 2
    assert "requires --dataset synthetic" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Regression: existing comparison + single-strategy CLI
# ---------------------------------------------------------------------------


def test_regression_compare_cli_still_works(capsys):
    code = main(["--dataset", "synthetic", "--compare", "none,hmem"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Comparison (synthetic_demo, k=5): 2 cases, 2 strategies" in out
    assert "none" in out and "hmem" in out


def test_regression_single_strategy_cli_unchanged(capsys):
    assert main(["--dataset", "synthetic"]) == 0
    out0 = capsys.readouterr().out
    assert "Synthetic benchmark (synthetic_demo)" in out0
    assert "Mean F1" in out0
    assert "Ablation" not in out0

    assert main(["--dataset", "synthetic", "--baseline", "memoryos"]) == 0
    assert "Synthetic benchmark (synthetic_demo, baseline=memoryos)" in (
        capsys.readouterr().out
    )
    assert main(["--dataset", "synthetic", "--baseline", "full_history"]) == 0
    assert "baseline=full_history" in capsys.readouterr().out


def test_regression_comparison_module_untouched_by_ablation():
    # compare_strategies still behaves as before (used internally, unchanged)
    specs = build_synthetic_strategies(["none", "full_history"], k=5)
    result = compare_strategies(_dataset(), specs, _synthetic_answer, k=5)
    assert result.names == ["none", "full_history"]
    assert result.get("none").report.mean_f1 == pytest.approx(1.0)
