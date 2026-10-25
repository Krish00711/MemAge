"""Tests for src.eval.baselines.MemoryOSBaseline — tiered STM/LTM approximation.

Explicitly a project-level approximation (see the class docstring): these
tests pin down the behavior WE define (tiering by recency, lexical top-k
selection, deterministic ordering), not any external system's behavior.
No network, no store, no LLM.
"""

import pytest

from src.eval.baselines import (
    FullHistoryBaseline,
    MemoryOSBaseline,
    memoryos_retrieval_fn,
)
from src.eval.benchmark import BenchmarkRunner, RetrievalOutcome
from src.eval.dataset import BenchmarkCase, BenchmarkDataset
from src.eval.metrics import tokens_returned
from src.eval.run_benchmark import main


def _history():
    """Distinct-overlap history: each entry matches a different query topic."""
    return [
        {"id": "m1", "text": "User drinks coffee every morning", "timestamp": "t1", "source": "s1"},
        {"id": "m2", "text": "User adopted a cat named Miso", "timestamp": "t2", "source": "s2"},
        {"id": "m3", "text": "User commutes by bicycle to work", "timestamp": "t3", "source": "s3"},
    ]


# ---------------------------------------------------------------------------
# Context generation
# ---------------------------------------------------------------------------


def test_context_generation_uses_shared_block_format():
    baseline = MemoryOSBaseline(_history(), k=3)
    outcome = baseline("coffee habits")
    assert isinstance(outcome, RetrievalOutcome)
    assert outcome.context.startswith("[Memory 1]")
    assert "Content: " in outcome.context
    assert outcome.tokens == tokens_returned(outcome.context)


def test_selected_entries_appear_with_their_text():
    baseline = MemoryOSBaseline(_history(), k=2)
    outcome = baseline("coffee morning routine")
    # top-1 must be the coffee entry; k=2 keeps one more
    assert "drinks coffee" in outcome.context
    assert outcome.context.count("[Memory ") == 2


# ---------------------------------------------------------------------------
# Ordering / selection behavior
# ---------------------------------------------------------------------------


def test_selection_orders_by_lexical_overlap():
    baseline = MemoryOSBaseline(_history(), k=3)
    outcome = baseline("What is the user's cat name?")
    assert "cat named Miso" in outcome.context
    # best match is first block
    first_block = outcome.context.split("\n\n")[0]
    assert "cat named Miso" in first_block
    assert outcome.retrieved_ids[0] == "m2"


def test_k_limits_selection():
    baseline = MemoryOSBaseline(_history(), k=1)
    outcome = baseline("coffee")  # only m1 has "coffee"
    assert outcome.context.count("[Memory ") == 1
    assert outcome.retrieved_ids == ["m1"]

    baseline2 = MemoryOSBaseline(_history(), k=2)
    outcome2 = baseline2("coffee")
    assert outcome2.context.count("[Memory ") == 2


def test_recency_breaks_score_ties():
    # identical text -> identical scores; newer entry (higher index) first
    history = [
        {"id": "old", "text": "same words here", "timestamp": "t1"},
        {"id": "new", "text": "same words here", "timestamp": "t2"},
    ]
    outcome = MemoryOSBaseline(history, k=2)("same words here")
    assert outcome.retrieved_ids == ["new", "old"]


def test_selection_is_deterministic():
    baseline = MemoryOSBaseline(_history(), k=3)
    a = baseline("coffee cat bicycle")
    b = baseline("coffee cat bicycle")
    assert a.context == b.context
    assert a.retrieved_ids == b.retrieved_ids


def test_tier_counts_and_stm_is_most_recent():
    baseline = MemoryOSBaseline(_history(), k=2, stm_size=1)
    assert baseline.num_stm == 1
    assert baseline.num_ltm == 2
    assert baseline.k == 2
    assert baseline.stm_size == 1
    # all entries remain eligible for selection regardless of tier
    outcome = baseline("bicycle commute")
    assert outcome.retrieved_ids[0] == "m3"


def test_invalid_k_and_stm_size_rejected():
    for bad in (0, -1, True, 2.5):
        with pytest.raises(ValueError):
            MemoryOSBaseline(_history(), k=bad)
        with pytest.raises(ValueError):
            MemoryOSBaseline(_history(), stm_size=bad)


# ---------------------------------------------------------------------------
# Empty history / input handling
# ---------------------------------------------------------------------------


def test_empty_history_yields_empty_context_and_no_ids():
    for empty in ([], None, ""):
        baseline = MemoryOSBaseline(empty)
        outcome = baseline("anything")
        assert outcome.context == ""
        assert outcome.retrieved_ids == []
        assert outcome.tokens == 0
    default = MemoryOSBaseline()  # default args constructible (empty history)
    assert default.num_stm == 0
    assert default("").context == ""


def test_entries_without_ids_appear_in_context_but_not_ids():
    history = [
        {"text": "plain entry about tea", "timestamp": "t1"},  # no id
        {"id": "with_id", "text": "entry about tea and biscuits", "timestamp": "t2"},
    ]
    outcome = MemoryOSBaseline(history, k=2)("tea")
    assert "plain entry about tea" in outcome.context
    assert outcome.retrieved_ids == ["with_id"]


# ---------------------------------------------------------------------------
# Benchmark integration
# ---------------------------------------------------------------------------


def test_benchmark_integration_reports_retrieval_metrics():
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

    runner = BenchmarkRunner(retrieval_fn=MemoryOSBaseline(_history(), k=2), answer_fn=answer_fn, k=5)
    report = runner.run_dataset(cases)

    assert report.num_queries == 2
    assert report.mean_f1 == pytest.approx(1.0)
    # ranking present -> retrieval metrics computed (not None)
    assert report.mean_recall_at_k == pytest.approx(1.0)
    assert report.mrr == pytest.approx(1.0)
    for r in report.per_case:
        assert r.retrieved_ids is not None
        assert r.tokens is not None and r.tokens > 0


def test_memoryos_retrieval_fn_factory():
    fn = memoryos_retrieval_fn(_history(), k=1)
    assert isinstance(fn, MemoryOSBaseline)
    assert isinstance(fn("coffee"), RetrievalOutcome)


def test_token_budget_not_enforced_selection_is_k_based():
    # tiny token budget must NOT truncate the k-based selection
    baseline = MemoryOSBaseline(_history(), k=3)
    outcome = baseline("coffee cat bicycle")
    assert outcome.tokens > 5
    assert outcome.context.count("[Memory ") == 3


# ---------------------------------------------------------------------------
# CLI execution
# ---------------------------------------------------------------------------


def test_cli_memoryos_synthetic(capsys):
    code = main(["--dataset", "synthetic", "--baseline", "memoryos"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Synthetic benchmark (synthetic_demo, baseline=memoryos)" in out
    assert "Mean F1" in out
    # ranking present -> retrieval metrics ARE reported
    assert "Mean Recall" in out
    assert "MRR" in out


def test_cli_memoryos_k_controls_selection(capsys):
    code = main(["--dataset", "synthetic", "--baseline", "memoryos", "--k", "1"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Mean tokens" in out
    tokens = float([l for l in out.splitlines() if "Mean tokens" in l][0].split(":")[1])
    # k=1 -> exactly one memory block (~11 tokens per block)
    assert tokens < 20


def test_cli_memoryos_not_blocked_by_missing_data_files(capsys):
    # memoryos no longer in NOT_YET_IMPLEMENTED_BASELINES: with synthetic it runs;
    # with a real dataset it loads then reports the src/llm blocker
    code = main(["--dataset", "synthetic", "--baseline", "memoryos"])
    assert code == 0


def test_cli_memoryos_real_dataset_blocked_on_llm(tmp_path, capsys):
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
        ["--dataset", "locomo", "--baseline", "memoryos", "--data-dir", str(tmp_path)]
    )
    captured = capsys.readouterr()
    assert code in (0, 2)
    assert "Loaded locomo: 1 cases" in captured.out


def test_cli_full_history_and_none_still_work(capsys):
    # requirement: do not break the other baselines
    assert main(["--dataset", "synthetic", "--baseline", "full_history"]) == 0
    assert "baseline=full_history" in capsys.readouterr().out
    assert main(["--dataset", "synthetic", "--baseline", "none"]) == 0
    assert "Synthetic benchmark (synthetic_demo" in capsys.readouterr().out
