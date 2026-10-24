"""Tests for src.eval.baselines — Full-History baseline strategy.

Covers context generation, empty history, multiple turns, benchmark
execution, token accounting, token-budget (non-)application, and CLI wiring.
No network, no store, no LLM.
"""

import pytest

from src.eval.baselines import (
    FullHistoryBaseline,
    format_full_history,
    full_history_retrieval_fn,
    history_from_case,
)
from src.eval.benchmark import BenchmarkRunner, RetrievalOutcome
from src.eval.dataset import BenchmarkCase, BenchmarkDataset
from src.eval.metrics import tokens_returned


def _turns(n=3):
    return [
        {
            "text": f"turn number {i} content",
            "timestamp": f"2026-01-0{i}T00:00:00Z",
            "source": f"s{i}",
        }
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# Context generation
# ---------------------------------------------------------------------------


def test_full_history_context_contains_all_turns_in_order():
    baseline = FullHistoryBaseline(_turns(4))
    outcome = baseline("any query at all")

    assert isinstance(outcome, RetrievalOutcome)
    assert outcome.context == baseline.context
    for i in range(1, 5):
        assert f"turn number {i} content" in outcome.context
    # order preserved
    positions = [outcome.context.index(f"turn number {i} content") for i in range(1, 5)]
    assert positions == sorted(positions)
    # deterministic: same context for every query, repeated calls
    assert baseline("q1").context == baseline("q2").context
    assert baseline("q1").context == baseline("q1").context


def test_full_history_is_not_query_dependent_and_has_no_retrieved_ids():
    baseline = FullHistoryBaseline(_turns(2))
    outcome = baseline("a completely unrelated question")
    # no ranking implied by this baseline -> no retrieval metrics
    assert outcome.retrieved_ids is None
    assert outcome.tokens == tokens_returned(outcome.context)


def test_full_history_accepts_plain_string_and_mixed_entries():
    from_str = FullHistoryBaseline("raw history text")
    assert from_str.context == "raw history text"
    assert from_str("q").context == "raw history text"

    mixed = FullHistoryBaseline(
        ["plain string turn", {"content": "chat-style turn", "role": "user"}]
    )
    ctx = mixed("q").context
    assert "plain string turn" in ctx
    assert "chat-style turn" in ctx


def test_format_full_history_edge_cases():
    assert format_full_history(None) == ""
    assert format_full_history([]) == ""
    assert format_full_history("") == ""
    assert format_full_history("x") == "x"


# ---------------------------------------------------------------------------
# Empty history
# ---------------------------------------------------------------------------


def test_empty_history_yields_empty_context_and_tokens():
    for empty in ([], None, ""):
        baseline = FullHistoryBaseline(empty)
        outcome = baseline("q")
        assert outcome.context == ""
        assert outcome.tokens == 0
        assert outcome.retrieved_ids is None
    assert FullHistoryBaseline().context == ""  # default arg


def test_benchmark_runs_with_empty_history():
    cases = [
        BenchmarkCase(query="q1", reference_answer="hello world", relevant_ids=["m"], case_id="c1"),
        BenchmarkCase(query="q2", reference_answer="other", case_id="c2"),
    ]
    runner = BenchmarkRunner(
        retrieval_fn=FullHistoryBaseline([]),
        answer_fn=lambda q, ctx: "hello world" if not ctx else "unexpected",
    )
    report = runner.run_dataset(cases)
    assert report.num_queries == 2
    assert report.mean_f1 == pytest.approx(0.5)  # case1 exact, case2 miss
    assert report.mean_tokens == pytest.approx(0.0)
    # no retrieved ids -> retrieval metrics stay None (not zero, not crash)
    assert report.mean_recall_at_k is None
    assert report.mrr is None


# ---------------------------------------------------------------------------
# Benchmark execution with the baseline
# ---------------------------------------------------------------------------


def test_benchmark_execution_using_full_history_baseline():
    history = _turns(3)
    cases = [
        BenchmarkCase(
            query="What happened in turn 2?",
            reference_answer="turn number 2 content",
            relevant_ids=["m2"],
            case_id="c1",
        ),
        BenchmarkCase(
            query="Summarize everything",
            reference_answer="turn number 1 content turn number 2 content turn number 3 content",
            case_id="c2",
        ),
    ]

    def answer_fn(query, context):
        # naive upper-bound answerer: echoes everything the baseline provided
        return " ".join(
            line.split("Content: ", 1)[1]
            for line in context.splitlines()
            if line.startswith("Content: ")
        )

    runner = BenchmarkRunner(retrieval_fn=FullHistoryBaseline(history), answer_fn=answer_fn, k=5)
    report = runner.run_dataset(cases)

    assert report.num_queries == 2
    # every case saw the same full-history context
    for result in report.per_case:
        assert result.retrieved_ids is None
        assert result.retrieval_recall is None
        assert result.tokens == tokens_returned(FullHistoryBaseline(history).context)
    assert report.mean_tokens == tokens_returned(FullHistoryBaseline(history).context)
    assert report.mean_f1 == pytest.approx(
        sum(r.f1 for r in report.per_case) / 2
    )
    # case2 should be an exact match under the naive answerer
    assert report.per_case[1].f1 == pytest.approx(1.0)


def test_full_history_retrieval_fn_factory():
    fn = full_history_retrieval_fn(_turns(1))
    assert isinstance(fn, FullHistoryBaseline)
    assert isinstance(fn("q"), RetrievalOutcome)


def test_history_from_case_variants():
    # LoCoMo-style metadata (adapter shape)
    locomo_case = BenchmarkCase(
        query="q",
        reference_answer="a",
        metadata={"turns": [{"id": "D1:1", "text": "hello there", "speaker": "A", "session": "session_1", "timestamp": "t1"}]},
    )
    ctx = history_from_case(locomo_case)
    assert "hello there" in ctx

    # LongMemEval-style metadata (sessions zipped with dates)
    lme_case = BenchmarkCase(
        query="q",
        reference_answer="a",
        metadata={
            "sessions": [
                {
                    "session_id": "s1",
                    "date": "2023/05/01",
                    "turns": [
                        {"role": "user", "content": "I moved to Paris"},
                        {"role": "assistant", "content": "Nice!", "has_answer": True},
                    ],
                }
            ]
        },
    )
    ctx2 = history_from_case(lme_case)
    assert "I moved to Paris" in ctx2 and "Nice!" in ctx2

    # no history metadata -> empty string (valid empty history)
    assert history_from_case(BenchmarkCase(query="q", reference_answer="a")) == ""
    assert history_from_case(BenchmarkCase(query="q", reference_answer="a", metadata={})) == ""


def test_full_history_baseline_over_case_metadata_end_to_end():
    case = BenchmarkCase(
        query="q",
        reference_answer="hello there",
        metadata={"turns": [{"id": "D1:1", "text": "hello there", "speaker": "A"}]},
    )
    runner = BenchmarkRunner(
        retrieval_fn=FullHistoryBaseline(history_from_case(case)),
        answer_fn=lambda q, c: "hello there" if "hello there" in c else "",
    )
    result = runner.run_single(case)
    assert result.f1 == pytest.approx(1.0)
    assert result.tokens == tokens_returned(history_from_case(case))


# ---------------------------------------------------------------------------
# Token usage / token-budget behavior
# ---------------------------------------------------------------------------


def test_token_usage_reported_for_full_history():
    baseline = FullHistoryBaseline(_turns(3))
    assert baseline.tokens == tokens_returned(baseline.context)
    assert baseline.tokens > 0


def test_token_budget_is_not_applicable_to_full_history():
    # The baseline contract has no budget: tiny budgets must NOT truncate.
    history = _turns(3)
    baseline = FullHistoryBaseline(history)
    outcome = baseline("q")
    full_tokens = tokens_returned(outcome.context)

    # even though "budget=5" would exclude everything in the retrieval
    # pipeline, the full-history context is returned whole
    assert full_tokens > 5
    assert outcome.context == baseline.context
    assert outcome.context.count("[Memory ") == len(history)
