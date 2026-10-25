"""Tests for src.eval.baselines.HMemBaseline — hierarchical STM/MTM/LTM approximation.

Explicitly a project-level approximation (see the class docstring): these
tests pin down the behavior WE define (three static recency tiers,
tier-priority coarse-to-fine ordering, repository budget selection), not any
external system's behavior. No network, no store, no LLM.
"""

import pytest

from src.eval.baselines import (
    FullHistoryBaseline,
    HMemBaseline,
    MemoryOSBaseline,
    MemoryR1Baseline,
    hmem_retrieval_fn,
)
from src.eval.benchmark import BenchmarkRunner, RetrievalOutcome
from src.eval.dataset import BenchmarkCase
from src.eval.run_benchmark import main
from src.retrieval.token_budget import estimate_tokens


def _history():
    """Five distinct entries spanning all three tiers (n=5 -> LTM 1, MTM 2, STM 2)."""
    return [
        {"id": "m1", "text": "User drinks coffee every morning", "timestamp": "t1"},
        {"id": "m2", "text": "User adopted a cat named Miso", "timestamp": "t2"},
        {"id": "m3", "text": "User commutes by bicycle to work", "timestamp": "t3"},
        {"id": "m4", "text": "User plays the violin on weekends", "timestamp": "t4"},
        {"id": "m5", "text": "User plants tomatoes in the garden", "timestamp": "t5"},
    ]


# ---------------------------------------------------------------------------
# Tier assignment / hierarchy structure
# ---------------------------------------------------------------------------


def test_tier_counts_by_recency_boundaries():
    # n=5 -> boundaries 1 and 3: LTM=[m1], MTM=[m2,m3], STM=[m4,m5]
    b = HMemBaseline(_history())
    assert (b.num_ltm, b.num_mtm, b.num_stm) == (1, 2, 2)
    # small histories may leave a tier empty (documented behavior)
    b2 = HMemBaseline(_history()[:2])  # n=2 -> boundaries 0,1
    assert (b2.num_ltm, b2.num_mtm, b2.num_stm) == (0, 1, 1)
    b3 = HMemBaseline(_history()[:1])  # n=1 -> boundaries 0,0
    assert (b3.num_ltm, b3.num_mtm, b3.num_stm) == (0, 0, 1)
    assert b3.num_entries == 1


def test_invalid_token_budget_rejected():
    for bad in (0, -1, True, 2.5, None):
        with pytest.raises(ValueError):
            HMemBaseline(_history(), token_budget=bad)


# ---------------------------------------------------------------------------
# Selection / ranking behavior
# ---------------------------------------------------------------------------


def test_best_lexical_match_comes_first_regardless_of_tier():
    # m1 (LTM tier, oldest) uniquely matches "coffee" -> coarse step promotes LTM
    b = HMemBaseline(_history(), token_budget=512)
    outcome = b("coffee morning routine")
    assert outcome.retrieved_ids[0] == "m1"
    first_block = outcome.context.split("\n\n")[0]
    assert "drinks coffee" in first_block


def test_tier_priority_uses_best_entry_then_recency_breaks_ties():
    # identical-score history: every entry "user activity N" scores the same
    # for query "user activity" -> all tier maxima tie -> recency decides:
    # STM first, then MTM, then LTM
    history = [
        {"id": f"e{i}", "text": f"user activity {i}", "timestamp": f"t{i}"}
        for i in range(1, 6)
    ]
    b = HMemBaseline(history, token_budget=512)
    ranked = b.rank("user activity")
    ids = [b._entries[i].get("id") for i, _n, _s in ranked]
    assert ids == ["e5", "e4", "e3", "e2", "e1"]  # STM, MTM, LTM; recent first


def test_tier_priority_promotes_tier_with_best_match():
    # m1 (LTM tier, oldest) uniquely matches "coffee" -> coarse step promotes
    # the LTM tier above STM/MTM despite it being the oldest tier
    b = HMemBaseline(_history(), token_budget=512)
    ranked = b.rank("coffee morning routine")
    assert b._entries[ranked[0][0]]["id"] == "m1"


def test_within_tier_scores_descend():
    b = HMemBaseline(_history(), token_budget=512)
    ranked = b.rank("bicycle commute violin")
    scores = [s for _i, _n, s in ranked]
    # the global maximum is always first (its tier wins or ties the coarse step)
    assert ranked[0][2] == max(scores)
    assert ranked[0][2] > 0
    # hierarchical order is tier-first: within each tier block, scores descend
    # (n=5: STM = original indices {3,4}, MTM = {1,2}, LTM = {0})
    tier_of = {3: "stm", 4: "stm", 1: "mtm", 2: "mtm", 0: "ltm"}
    seen_blocks = []
    for tier, group in _group_by_tier(ranked, tier_of):
        if not seen_blocks or seen_blocks[-1] != tier:
            seen_blocks.append(tier)
            group_scores = [s for _i, _n, s in group]
            assert group_scores == sorted(group_scores, reverse=True)
    # each tier appears as one contiguous block (coarse-to-fine ordering)
    assert len(seen_blocks) == len(set(seen_blocks))


def _group_by_tier(ranked, tier_of):
    """Yield (tier_name, contiguous_group) in ranked order."""
    current_tier, current_group = None, []
    for item in ranked:
        t = tier_of[item[0]]
        if t != current_tier:
            if current_group:
                yield current_tier, current_group
            current_tier, current_group = t, []
        current_group.append(item)
    if current_group:
        yield current_tier, current_group


def test_ranking_is_deterministic():
    b = HMemBaseline(_history(), token_budget=512)
    a1 = b("user")
    a2 = b("user")
    assert [r for r in b.rank("user")] == [r for r in b.rank("user")]
    assert a1.retrieved_ids == a2.retrieved_ids
    assert a1.context == a2.context


def test_budget_is_enforced():
    # 5 blocks would far exceed 12 tokens; only the top-ranked block fits
    b = HMemBaseline(_history(), token_budget=12)
    outcome = b("coffee morning routine")
    assert outcome.context.count("[Memory ") == 1
    assert estimate_tokens(outcome.context) <= 12
    assert outcome.retrieved_ids == ["m1"]


def test_large_budget_admits_everything():
    b = HMemBaseline(_history(), token_budget=512)
    outcome = b("user")
    assert outcome.context.count("[Memory ") == 5
    assert estimate_tokens(outcome.context) <= 512


# ---------------------------------------------------------------------------
# Empty input / id conventions
# ---------------------------------------------------------------------------


def test_empty_history_yields_empty_context_and_no_ids():
    for empty in ([], None, ""):
        b = HMemBaseline(empty)
        outcome = b("anything")
        assert outcome.context == ""
        assert outcome.retrieved_ids == []
        assert outcome.tokens == 0
        assert b.num_entries == 0
    default = HMemBaseline()
    assert default.token_budget == HMemBaseline.DEFAULT_TOKEN_BUDGET
    assert default("").context == ""


def test_entries_without_ids_in_context_but_not_ids():
    history = [
        {"text": "plain entry about tea", "timestamp": "t1"},  # no id
        {"id": "with_id", "text": "entry about tea and biscuits", "timestamp": "t2"},
    ]
    outcome = HMemBaseline(history, token_budget=512)("tea")
    assert "plain entry about tea" in outcome.context
    assert outcome.retrieved_ids == ["with_id"]


# ---------------------------------------------------------------------------
# Benchmark integration
# ---------------------------------------------------------------------------


def test_benchmark_integration_reports_retrieval_metrics():
    cases = [
        BenchmarkCase(
            query="coffee morning routine",
            reference_answer="drinks coffee",
            relevant_ids=["m1"],
            case_id="c1",
        ),
        BenchmarkCase(
            query="bicycle commute to work",
            reference_answer="commutes by bicycle",
            relevant_ids=["m3"],
            case_id="c2",
        ),
    ]

    def answer_fn(query, context):
        if "coffee" in query:
            return "drinks coffee" if "coffee" in context else "Unknown"
        return "commutes by bicycle" if "bicycle" in context else "Unknown"

    runner = BenchmarkRunner(
        retrieval_fn=HMemBaseline(_history(), token_budget=512),
        answer_fn=answer_fn,
        k=5,
    )
    report = runner.run_dataset(cases)

    assert report.num_queries == 2
    assert report.mean_f1 == pytest.approx(1.0)
    assert report.mean_recall_at_k == pytest.approx(1.0)
    assert report.mrr == pytest.approx(1.0)
    for r in report.per_case:
        assert r.retrieved_ids is not None
        assert r.tokens is not None and r.tokens > 0


def test_benchmark_tight_budget_runs_gracefully():
    case = BenchmarkCase(
        query="coffee morning routine",
        reference_answer="drinks coffee",
        relevant_ids=["m1"],
        case_id="c1",
    )
    runner = BenchmarkRunner(
        retrieval_fn=HMemBaseline(_history(), token_budget=5),
        answer_fn=lambda q, c: "drinks coffee" if "coffee" in c else "Unknown",
    )
    result = runner.run_single(case)
    assert result.predicted_answer == "Unknown"  # nothing fits a 5-token budget
    assert result.tokens == 0
    assert result.retrieval_recall == 0.0


def test_hmem_retrieval_fn_factory():
    fn = hmem_retrieval_fn(_history(), token_budget=64)
    assert isinstance(fn, HMemBaseline)
    assert isinstance(fn("coffee"), RetrievalOutcome)


# ---------------------------------------------------------------------------
# CLI execution
# ---------------------------------------------------------------------------


def test_cli_hmem_synthetic(capsys):
    code = main(["--dataset", "synthetic", "--baseline", "hmem"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Synthetic benchmark (synthetic_demo, baseline=hmem)" in out
    assert "Mean F1" in out
    assert "Mean Recall" in out  # ids present -> retrieval metrics reported
    assert "MRR" in out


def test_cli_hmem_respects_token_budget(capsys):
    assert main(["--dataset", "synthetic", "--baseline", "hmem"]) == 0
    default_tokens = float(
        [l for l in capsys.readouterr().out.splitlines() if "Mean tokens" in l][0].split(":")[1]
    )
    assert (
        main(["--dataset", "synthetic", "--baseline", "hmem", "--token-budget", "12"]) == 0
    )
    tight_tokens = float(
        [l for l in capsys.readouterr().out.splitlines() if "Mean tokens" in l][0].split(":")[1]
    )
    assert tight_tokens <= 12
    assert tight_tokens < default_tokens


def test_cli_hmem_real_dataset_blocked_on_llm(tmp_path, capsys):
    import json

    fixture = [
        {
            "sample_id": "sample_1",
            "conversation": {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "hello"}],
                "session_1_date_time": "t",
            },
            "qa": [{"question": "Q?", "answer": "A", "evidence": ["D1:1"]}],
        }
    ]
    d = tmp_path / "locomo"
    d.mkdir(parents=True, exist_ok=True)
    (d / "locomo10.json").write_text(json.dumps(fixture), encoding="utf-8")

    code = main(["--dataset", "locomo", "--baseline", "hmem", "--data-dir", str(tmp_path)])
    captured = capsys.readouterr()
    assert code in (0, 2)
    assert "Loaded locomo: 1 cases" in captured.out


# ---------------------------------------------------------------------------
# Regression: every other baseline still behaves as before
# ---------------------------------------------------------------------------


def test_regression_all_existing_baselines(capsys):
    # none (two-stage pipeline): unqualified title, retrieval metrics, budget applies
    assert main(["--dataset", "synthetic", "--token-budget", "12"]) == 0
    out0 = capsys.readouterr().out
    assert "Synthetic benchmark (synthetic_demo" in out0
    assert "Mean Recall" in out0
    tokens0 = float([l for l in out0.splitlines() if "Mean tokens" in l][0].split(":")[1])
    assert tokens0 <= 12

    # full_history: title pattern, no retrieval metrics
    assert main(["--dataset", "synthetic", "--baseline", "full_history"]) == 0
    out1 = capsys.readouterr().out
    assert "baseline=full_history" in out1
    assert "Mean Recall" not in out1

    # memoryos: title pattern, retrieval metrics
    assert main(["--dataset", "synthetic", "--baseline", "memoryos"]) == 0
    out2 = capsys.readouterr().out
    assert "baseline=memoryos" in out2
    assert "Mean Recall" in out2

    # memory_r1: title pattern, retrieval metrics, budget applies
    assert main(["--dataset", "synthetic", "--baseline", "memory_r1"]) == 0
    out3 = capsys.readouterr().out
    assert "baseline=memory_r1" in out3
    assert "Mean Recall" in out3


def test_regression_baseline_classes_construct():
    h = [{"id": "x", "text": "some text about things", "timestamp": "t"}]
    assert FullHistoryBaseline(h).context.startswith("[Memory 1]")
    assert isinstance(MemoryOSBaseline(h), MemoryOSBaseline)
    assert isinstance(MemoryR1Baseline(h), MemoryR1Baseline)
    assert isinstance(HMemBaseline(h), HMemBaseline)
