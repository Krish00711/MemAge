"""Tests for src.eval.baselines.MemoryR1Baseline — budgeted ranked approximation.

Explicitly a project-level approximation (see the class docstring): these
tests pin down the behavior WE define (single-stage lexical ranking +
repository budget selection), not any external system's behavior.
No network, no store, no LLM.
"""

import pytest

from src.eval.baselines import (
    FullHistoryBaseline,
    MemoryOSBaseline,
    MemoryR1Baseline,
    memory_r1_retrieval_fn,
)
from src.eval.benchmark import BenchmarkRunner, RetrievalOutcome
from src.eval.dataset import BenchmarkCase
from src.eval.metrics import tokens_returned
from src.eval.run_benchmark import main
from src.retrieval.token_budget import estimate_tokens


def _history():
    """Distinct-overlap history: each entry matches a different query topic."""
    return [
        {"id": "m1", "text": "User drinks coffee every morning", "timestamp": "t1"},
        {"id": "m2", "text": "User adopted a cat named Miso", "timestamp": "t2"},
        {"id": "m3", "text": "User commutes by bicycle to work", "timestamp": "t3"},
    ]


# ---------------------------------------------------------------------------
# Selection / ranking
# ---------------------------------------------------------------------------


def test_ranking_puts_best_lexical_match_first():
    baseline = MemoryR1Baseline(_history(), token_budget=512)
    outcome = baseline("cat named Miso")
    assert outcome.retrieved_ids[0] == "m2"
    first_block = outcome.context.split("\n\n")[0]
    assert "cat named Miso" in first_block


def test_ranking_is_deterministic_and_recency_breaks_ties():
    history = [
        {"id": "old", "text": "same words here", "timestamp": "t1"},
        {"id": "new", "text": "same words here", "timestamp": "t2"},
    ]
    baseline = MemoryR1Baseline(history, token_budget=512)
    a = baseline("same words here")
    b = baseline("same words here")
    assert a.retrieved_ids == ["new", "old"]
    assert a.context == b.context


def test_rank_scores_descend():
    baseline = MemoryR1Baseline(_history(), token_budget=512)
    ranked = baseline.rank("bicycle commute to work")
    scores = [s for _i, _n, s in ranked]
    assert scores == sorted(scores, reverse=True)
    assert ranked[0][2] > 0  # best match has nonzero overlap


# ---------------------------------------------------------------------------
# Budget enforcement (the defining behavior of this baseline)
# ---------------------------------------------------------------------------


def test_budget_is_enforced_on_context():
    # two blocks ~22 tokens; budget 12 allows only the highest-ranked block
    baseline = MemoryR1Baseline(_history(), token_budget=12)
    outcome = baseline("coffee morning routine")
    assert outcome.context.count("[Memory ") == 1
    assert estimate_tokens(outcome.context) <= 12
    assert outcome.retrieved_ids == ["m1"]


def test_larger_budget_admits_more_entries():
    small = MemoryR1Baseline(_history(), token_budget=12)("coffee cat bicycle")
    large = MemoryR1Baseline(_history(), token_budget=512)("coffee cat bicycle")
    assert estimate_tokens(small.context) <= 12
    assert small.context.count("[Memory ") < large.context.count("[Memory ")
    assert large.context.count("[Memory ") == 3
    assert small.retrieved_ids == large.retrieved_ids[: len(small.retrieved_ids)]


def test_skip_too_large_keeps_lower_ranked(tmp_path=None):
    # a huge first-ranked entry does not block smaller later entries
    history = [
        {"id": "big", "text": " ".join(["word"] * 40)},
        {"id": "small", "text": "short relevant entry"},
    ]
    outcome = MemoryR1Baseline(history, token_budget=15)("word entry")
    # "big" outranks (shares "word") but cannot fit; "small" still selected
    assert outcome.retrieved_ids == ["small"]
    assert estimate_tokens(outcome.context) <= 15


def test_invalid_token_budget_rejected():
    for bad in (0, -1, True, 2.5, None):
        with pytest.raises(ValueError):
            MemoryR1Baseline(_history(), token_budget=bad)


def test_token_budget_property_and_default():
    assert MemoryR1Baseline(_history(), token_budget=77).token_budget == 77
    assert MemoryR1Baseline().token_budget == MemoryR1Baseline.DEFAULT_TOKEN_BUDGET
    assert MemoryR1Baseline(_history()).num_entries == 3


# ---------------------------------------------------------------------------
# Empty input / id conventions
# ---------------------------------------------------------------------------


def test_empty_history_yields_empty_context_and_no_ids():
    for empty in ([], None, ""):
        baseline = MemoryR1Baseline(empty)
        outcome = baseline("anything")
        assert outcome.context == ""
        assert outcome.retrieved_ids == []
        assert outcome.tokens == 0
    default = MemoryR1Baseline()
    assert default.num_entries == 0
    assert default("").context == ""


def test_entries_without_ids_in_context_but_not_ids():
    history = [
        {"text": "plain entry about tea", "timestamp": "t1"},  # no id
        {"id": "with_id", "text": "entry about tea and biscuits", "timestamp": "t2"},
    ]
    outcome = MemoryR1Baseline(history, token_budget=512)("tea")
    assert "plain entry about tea" in outcome.context
    assert outcome.retrieved_ids == ["with_id"]


# ---------------------------------------------------------------------------
# Benchmark integration
# ---------------------------------------------------------------------------


def test_benchmark_integration_reports_metrics():
    cases = [
        BenchmarkCase(
            query="What is the user's cat name?",
            reference_answer="Miso",
            relevant_ids=["m2"],
            case_id="c1",
        ),
        BenchmarkCase(
            query="bicycle commute to work",
            reference_answer="by bicycle",
            relevant_ids=["m3"],
            case_id="c2",
        ),
    ]

    def answer_fn(query, context):
        if "cat" in query:
            return "Miso" if "Miso" in context else "Unknown"
        return "by bicycle" if "bicycle" in context else "Unknown"

    runner = BenchmarkRunner(
        retrieval_fn=MemoryR1Baseline(_history(), token_budget=512),
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


def test_benchmark_tight_budget_still_runs_without_crashing():
    case = BenchmarkCase(
        query="What is the user's cat name?",
        reference_answer="Miso",
        relevant_ids=["m2"],
        case_id="c1",
    )
    runner = BenchmarkRunner(
        retrieval_fn=MemoryR1Baseline(_history(), token_budget=5),
        answer_fn=lambda q, c: "Miso" if "Miso" in c else "Unknown",
    )
    result = runner.run_single(case)
    # nothing fits a 5-token budget -> empty context handled gracefully
    assert result.predicted_answer == "Unknown"
    assert result.tokens == 0
    assert result.retrieval_recall == 0.0  # ids present but relevant missed


def test_memory_r1_retrieval_fn_factory():
    fn = memory_r1_retrieval_fn(_history(), token_budget=64)
    assert isinstance(fn, MemoryR1Baseline)
    assert isinstance(fn("coffee"), RetrievalOutcome)


# ---------------------------------------------------------------------------
# CLI execution
# ---------------------------------------------------------------------------


def test_cli_memory_r1_synthetic(capsys):
    code = main(["--dataset", "synthetic", "--baseline", "memory_r1"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Synthetic benchmark (synthetic_demo, baseline=memory_r1)" in out
    assert "Mean F1" in out
    assert "Mean Recall" in out
    assert "MRR" in out


def test_cli_memory_r1_respects_token_budget(capsys):
    # default budget: both synthetic memories fit
    assert main(["--dataset", "synthetic", "--baseline", "memory_r1"]) == 0
    default_tokens = float(
        [l for l in capsys.readouterr().out.splitlines() if "Mean tokens" in l][0].split(":")[1]
    )
    # tight budget: only one block fits
    assert (
        main(
            [
                "--dataset",
                "synthetic",
                "--baseline",
                "memory_r1",
                "--token-budget",
                "12",
            ]
        )
        == 0
    )
    tight_tokens = float(
        [l for l in capsys.readouterr().out.splitlines() if "Mean tokens" in l][0].split(":")[1]
    )
    assert tight_tokens <= 12
    assert tight_tokens < default_tokens


def test_cli_memory_r1_real_dataset_blocked_on_llm(tmp_path, capsys):
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

    code = main(
        ["--dataset", "locomo", "--baseline", "memory_r1", "--data-dir", str(tmp_path)]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "Loaded locomo: 1 cases" in captured.out
    assert "src/llm" in captured.err


def test_cli_other_baselines_unbroken(capsys):
    # regression: full_history, memoryos, none still behave as before
    assert main(["--dataset", "synthetic", "--baseline", "full_history"]) == 0
    out1 = capsys.readouterr().out
    assert "baseline=full_history" in out1
    assert "Mean Recall" not in out1

    assert main(["--dataset", "synthetic", "--baseline", "memoryos"]) == 0
    out2 = capsys.readouterr().out
    assert "baseline=memoryos" in out2
    assert "Mean Recall" in out2

    assert main(["--dataset", "synthetic", "--baseline", "none"]) == 0
    out3 = capsys.readouterr().out
    assert "Synthetic benchmark (synthetic_demo)" in out3
    assert "Mean Recall" in out3


def test_cli_hmem_runs_alongside_memory_r1(capsys):
    code = main(["--dataset", "synthetic", "--baseline", "hmem"])
    assert code == 0
    assert "baseline=hmem" in capsys.readouterr().out
