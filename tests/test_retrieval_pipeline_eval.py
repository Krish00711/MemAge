"""End-to-end: real two-stage retrieval pipeline → BenchmarkRunner → metrics.

Verifies that evaluation metrics (Recall@K, Precision@K, MRR, tokens) work
against the *real* retrieval output (context + ordered selected IDs), not
just hand-written fakes. Uses only in-memory fakes — no network, no ChromaDB.
"""

import json
import re

from src.eval.benchmark import BenchmarkRunner, RetrievalOutcome, make_retrieval_fn
from src.eval.dataset import BenchmarkDataset
from src.eval.evaluator import RetrievalEvaluator
from src.eval.metrics import precision_at_k, recall_at_k, reciprocal_rank, tokens_returned
from src.retrieval.context import RetrievalContext
from src.retrieval.reranker import LLMReranker
from src.retrieval.retriever import CoarseRetriever
from src.retrieval.token_budget import estimate_tokens

TOKEN_BUDGET = 45


class FakeStore:
    def __init__(self, items):
        self._items = items

    def query(self, text: str, k: int, tier: str):
        return self._items[:k]


def _items():
    return [
        {"id": "mem_001", "text": "User prefers dark mode.", "timestamp": "2026-01-15T10:00:00Z", "source": "s1", "embedding": None},
        {"id": "mem_002", "text": "User lives in Berlin.", "timestamp": "2026-01-16T12:30:00Z", "source": "s2", "embedding": None},
        {"id": "mem_003", "text": "User has a cat named Momo.", "timestamp": "2026-01-17T09:00:00Z", "source": "s3", "embedding": None},
    ]


def build_pipeline() -> RetrievalContext:
    def fake_llm(prompt, system=None, **kwargs):
        # Deterministic Stage-2 ordering: mem_002 > mem_001, mem_003 via fallback
        return json.dumps({"rankings": [{"id": "mem_002", "score": 0.9}, {"id": "mem_001", "score": 0.8}]})

    return RetrievalContext(
        CoarseRetriever(FakeStore(_items())),
        LLMReranker(fake_llm),
        k=3,
        tier="all",
    )


def _dataset() -> BenchmarkDataset:
    return BenchmarkDataset.from_synthetic(
        [
            {
                "query": "Where does the user live?",
                "reference_answer": "User lives in Berlin",
                "relevant_ids": ["mem_002"],
                "case_id": "c1",
            },
            {
                "query": "What is the user's preference?",
                "reference_answer": "User prefers dark mode",
                "relevant_ids": ["mem_001"],
                "case_id": "c2",
            },
            {
                "query": "Tell me about the user's cat",
                "reference_answer": "The user has a cat named Momo",
                "relevant_ids": ["mem_003"],
                "case_id": "c3",
            },
        ],
        name="pipeline_eval",
    )


def _run_benchmark():
    ctx = build_pipeline()
    runner = BenchmarkRunner(
        retrieval_fn=make_retrieval_fn(ctx, token_budget=TOKEN_BUDGET),
        answer_fn=lambda q, c: "ok" if c else "",
        k=5,
    )
    return ctx, runner.run_dataset(_dataset())


def test_benchmark_report_has_retrieval_metrics_from_real_pipeline():
    _ctx, report = _run_benchmark()
    assert report.num_queries == 3
    # IDs flowed through -> retrieval aggregates are real numbers, not None
    assert report.mean_recall_at_k is not None
    assert report.mean_precision_at_k is not None
    assert report.mrr is not None
    assert 0.0 <= report.mean_recall_at_k <= 1.0
    assert 0.0 <= report.mean_precision_at_k <= 1.0
    assert 0.0 <= report.mrr <= 1.0
    # Stage-2 puts mem_002 first: every relevant memory is inside top-5
    assert report.mean_recall_at_k == 1.0
    for r in report.per_case:
        assert r.retrieved_ids  # real IDs on every case
        assert r.retrieval_recall is not None
        assert r.retrieval_precision is not None
        assert r.retrieval_rr is not None


def test_evaluator_metrics_match_direct_metric_functions_on_real_output():
    _ctx, report = _run_benchmark()
    retrieved = [r.retrieved_ids for r in report.per_case]
    relevant = [r.relevant_ids for r in report.per_case]

    evaluator = RetrievalEvaluator(k=5)
    agg = evaluator.evaluate_dataset(retrieved, relevant, k=5)

    # Aggregator agrees with the benchmark's own aggregates
    assert agg.num_queries == 3
    assert agg.mean_recall_at_k == report.mean_recall_at_k
    assert agg.mrr == report.mrr
    # Per-query evaluator agrees with direct metric calls
    for r in report.per_case:
        qm = evaluator.evaluate_query(r.retrieved_ids, r.relevant_ids, k=5)
        assert qm.recall_at_k == recall_at_k(r.retrieved_ids, r.relevant_ids, 5)
        assert qm.precision_at_k == precision_at_k(r.retrieved_ids, r.relevant_ids, 5)
        assert qm.reciprocal_rank == reciprocal_rank(r.retrieved_ids, r.relevant_ids)


def test_real_pipeline_output_respects_token_budget_and_token_metric():
    ctx = build_pipeline()
    retrieval_fn = make_retrieval_fn(ctx, token_budget=TOKEN_BUDGET)

    for case in _dataset():
        outcome = retrieval_fn(case.query)
        assert isinstance(outcome, RetrievalOutcome)
        # Budget invariant, measured with the shared estimator
        assert estimate_tokens(outcome.context) <= TOKEN_BUDGET
        # Reported tokens match the evaluator's token metric
        assert outcome.tokens == tokens_returned(outcome.context)
        # Selected IDs are 1:1 in order with the formatted memory blocks
        blocks = re.findall(r"\[Memory (\d+)\]", outcome.context)
        assert blocks == [str(i) for i in range(1, len(outcome.retrieved_ids) + 1)]


def test_ids_shrink_when_token_budget_shrinks():
    ctx = build_pipeline()
    wide = make_retrieval_fn(ctx, token_budget=1000)("Where does the user live?")
    tight = make_retrieval_fn(ctx, token_budget=12)("Where does the user live?")

    assert len(tight.retrieved_ids) < len(wide.retrieved_ids)
    assert tight.retrieved_ids == ["mem_002"]  # only the top-ranked memory fits
    assert estimate_tokens(tight.context) <= 12
