"""Tests for src.eval.benchmark — end-to-end orchestration."""

import pytest

from src.eval.benchmark import BenchmarkRunner, RetrievalOutcome
from src.eval.dataset import BenchmarkCase, BenchmarkDataset
from src.eval.metrics import tokens_returned


# 3. BenchmarkCase
def test_benchmark_case():
    case = BenchmarkCase(
        query="q1",
        reference_answer="answer",
        relevant_ids=["m1"],
        case_id="c1",
        metadata={"k": 5},
        conversation_id="sess_1",
    )
    assert case.query == "q1"
    assert case.reference_answer == "answer"
    assert case.relevant_ids == ["m1"]
    assert case.case_id == "c1"
    assert case.conversation_id == "sess_1"
    # to_retrieval_case
    rc = case.to_retrieval_case()
    assert rc.query == "q1"
    assert rc.relevant_ids == ["m1"]


# 4. synthetic benchmark dataset
def test_synthetic_benchmark_dataset():
    ds = BenchmarkDataset.from_synthetic(
        [
            {"query": "q1", "reference_answer": "a1", "relevant_ids": ["m1"], "case_id": "id1"},
            {"query": "q2", "reference_answer": "a2", "relevant_ids": ["m2"], "metadata": {"x": 1}},
        ]
    )
    assert len(ds) == 2
    assert ds[0].query == "q1"
    assert ds[0].case_id == "id1"
    assert ds[1].reference_answer == "a2"
    assert not ds.is_empty()
    empty = BenchmarkDataset(cases=[])
    assert empty.is_empty()


# 5. single benchmark case execution
def test_single_benchmark_case_execution():
    case = BenchmarkCase(query="What is preference?", reference_answer="dark mode", relevant_ids=["m1"], case_id="c1")

    def retrieval_fn(q):
        return RetrievalOutcome(context="dark mode context", retrieved_ids=["m1"], tokens=5)

    def answer_fn(q, ctx):
        return "dark mode"

    runner = BenchmarkRunner(retrieval_fn=retrieval_fn, answer_fn=answer_fn, k=5)
    result = runner.run_single(case)
    assert result.case_id == "c1"
    assert result.predicted_answer == "dark mode"
    assert result.reference_answer == "dark mode"
    assert result.f1 == pytest.approx(1.0)
    assert result.bleu1 == pytest.approx(1.0)
    assert result.latency_ms >= 0
    assert result.tokens is not None


# 6. multiple-case aggregation
def test_multiple_case_aggregation():
    cases = [
        BenchmarkCase(query="q1", reference_answer="hello world", relevant_ids=["m1"], case_id="c1"),
        BenchmarkCase(query="q2", reference_answer="foo bar", relevant_ids=["m2"], case_id="c2"),
    ]

    def retrieval_fn(q):
        return "context"

    def answer_fn(q, ctx):
        # perfect for q1, zero for q2
        if "q1" in q:
            return "hello world"
        return "wrong answer"

    runner = BenchmarkRunner(retrieval_fn=retrieval_fn, answer_fn=answer_fn)
    report = runner.run_dataset(cases)
    assert report.num_queries == 2
    assert report.mean_f1 == pytest.approx(0.5)
    # mean_bleu1 also 0.5 + 0 => 0.5
    assert report.mean_latency_ms >= 0
    assert len(report.per_case) == 2

    # Empty dataset
    empty_report = runner.run_dataset([])
    assert empty_report.num_queries == 0
    assert empty_report.mean_f1 == pytest.approx(0.0)


# 7. latency recording
def test_latency_recording():
    case = BenchmarkCase(query="q", reference_answer="ref", case_id="c")

    def retrieval_fn(q):
        return "ctx"

    def answer_fn(q, ctx):
        return "ref"

    runner = BenchmarkRunner(retrieval_fn=retrieval_fn, answer_fn=answer_fn)
    result = runner.run_single(case)
    assert result.latency_ms >= 0
    assert result.latency_ms < 5000  # generous bound, not exact

    report = runner.run_dataset([case, case])
    assert report.mean_latency_ms >= 0


# 8. retrieval metric integration when retrieval metadata exists
def test_retrieval_metric_integration_exists():
    case = BenchmarkCase(query="q", reference_answer="ref", relevant_ids=["a", "b"], case_id="c")

    def retrieval_fn(q):
        return RetrievalOutcome(context="ctx", retrieved_ids=["a", "x", "b"])

    runner = BenchmarkRunner(retrieval_fn=retrieval_fn, answer_fn=lambda q, ctx: "ref", k=3)
    result = runner.run_single(case)
    # retrieved ["a","x","b"], relevant ["a","b"], k=3 => recall 1.0, prec 2/3, rr 1.0
    assert result.retrieval_recall == pytest.approx(1.0)
    assert result.retrieval_precision == pytest.approx(2 / 3)
    assert result.retrieval_rr == pytest.approx(1.0)
    assert result.retrieved_ids == ["a", "x", "b"]

    # Aggregation includes retrieval means
    report = runner.run_dataset([case])
    assert report.mean_recall_at_k == pytest.approx(1.0)
    assert report.mrr == pytest.approx(1.0)


# 9. benchmark operation when retrieval metadata does NOT exist
def test_no_retrieval_metadata():
    case = BenchmarkCase(query="q", reference_answer="ref", relevant_ids=["a"], case_id="c")

    # retrieval returns only string context, no ids
    def retrieval_fn(q):
        return "just context"

    runner = BenchmarkRunner(retrieval_fn=retrieval_fn, answer_fn=lambda q, ctx: "ref")
    result = runner.run_single(case)
    assert result.retrieval_recall is None
    assert result.retrieval_precision is None
    assert result.retrieval_rr is None
    # Tokens still computed from context
    assert result.tokens == tokens_returned("just context")

    # Also when retrieval_fn is None
    runner2 = BenchmarkRunner(retrieval_fn=None, answer_fn=lambda q, ctx: "ref")
    result2 = runner2.run_single(case)
    assert result2.retrieval_recall is None
    assert result2.predicted_answer == "ref"

    # Aggregation with mixed availability — mean over available only
    report = runner.run_dataset([case])
    assert report.mean_recall_at_k is None
    assert report.mrr is None


# 10. full-history strategy interface
def test_full_history_strategy_interface():
    # Memory-based strategy
    mem_case = BenchmarkCase(query="q", reference_answer="ref", relevant_ids=["m1"], case_id="c")
    full_case = BenchmarkCase(query="q", reference_answer="ref", relevant_ids=["m1"], case_id="c")

    def memory_retrieval(q):
        return RetrievalOutcome(context="memory context", retrieved_ids=["m1"])

    def full_history_retrieval(q):
        return RetrievalOutcome(context="FULL HISTORY TEXT WITH ALL DATA", retrieved_ids=None)

    def memory_answer(q, ctx):
        return "answer via memory"

    def full_answer(q, ctx):
        return "answer via full history"

    mem_runner = BenchmarkRunner(retrieval_fn=memory_retrieval, answer_fn=memory_answer)
    full_runner = BenchmarkRunner(retrieval_fn=full_history_retrieval, answer_fn=full_answer)

    mem_result = mem_runner.run_single(mem_case)
    full_result = full_runner.run_single(full_case)

    assert mem_result.predicted_answer == "answer via memory"
    assert full_result.predicted_answer == "answer via full history"
    # Full history returns no retrieval metrics (as no retrieved_ids), memory does
    assert mem_result.retrieval_recall is not None
    assert full_result.retrieval_recall is None
    # Both share same BenchmarkCase objects (strategy-agnostic)
    assert mem_result.query == full_result.query == "q"


def test_token_integration_consistency():
    # Ensure tokens uses same estimator as retrieval
    case = BenchmarkCase(query="q", reference_answer="ref", case_id="c")
    ctx_text = "hello world this is test"

    def retrieval_fn(q):
        return ctx_text

    runner = BenchmarkRunner(retrieval_fn=retrieval_fn, answer_fn=lambda q, ctx: "ref")
    result = runner.run_single(case)
    assert result.tokens == tokens_returned(ctx_text)
