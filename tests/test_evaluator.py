"""Tests for src.eval.evaluator — RetrievalEvaluator."""

import json

import pytest

from src.eval.dataset import EvalDataset, RetrievalEvalCase
from src.eval.evaluator import RetrievalEvaluator
from src.eval.metrics import tokens_returned
from src.retrieval.formatter import format_context
from src.retrieval.token_budget import estimate_tokens


# 20. complete synthetic evaluation case produces expected metrics
def test_complete_synthetic_case():
    evaluator = RetrievalEvaluator(k=3)
    retrieved = ["a", "b", "c"]
    relevant = ["a", "b"]
    result = evaluator.evaluate_query(retrieved, relevant, k=2, latency_ms=12.5, formatted_context="hello world")
    # recall@2: 1 of 2? Actually retrieved top-2 is ["a","b"] both relevant => recall 1.0? Wait relevant 2, top-2 contains a and b? retrieved[:2]=["a","b"] both in relevant => recall 2/2=1.0? No with retrieved ["a","b","c"], top2 ["a","b"] hits 1? relevant ["a","b"] -> both hits => 1.0. Let's use known case:
    # Use retrieved ["a","x","y"], relevant ["a","b"], k=3 => recall 0.5, prec 1/3
    result2 = evaluator.evaluate_query(["a", "x", "y"], ["a", "b"], k=3)
    assert result2.recall_at_k == pytest.approx(0.5)
    assert result2.precision_at_k == pytest.approx(1 / 3)
    assert result2.reciprocal_rank == pytest.approx(1.0)
    # With formatted context tokens
    assert result2.tokens is None
    result3 = evaluator.evaluate_query(["a", "b"], ["a", "b"], k=2, formatted_context="one two three")
    assert result3.tokens == 3


# 21. Stage-1 and Stage-2 results can be evaluated independently
def test_stage1_vs_stage2_independent():
    evaluator = RetrievalEvaluator(k=5)
    relevant = ["gold1", "gold2", "gold3"]
    # Stage-1 (coarse) retrieved order poorly ranked
    stage1 = ["x", "y", "gold1", "gold2", "z"]
    # Stage-2 reranked puts gold at top
    stage2 = ["gold1", "gold2", "gold3", "x", "y"]
    m1 = evaluator.evaluate_query(stage1, relevant, k=3)
    m2 = evaluator.evaluate_query(stage2, relevant, k=3)
    # Stage-2 should have better recall/precision/MRR in this synthetic example
    assert m2.recall_at_k >= m1.recall_at_k
    assert m2.precision_at_k >= m1.precision_at_k
    assert m2.reciprocal_rank >= m1.reciprocal_rank
    # But evaluator doesn't assume stage2 always better; we verify it computes neutrally
    # Reverse case where stage2 worse
    stage1_better = ["gold3", "gold1", "gold2"]
    stage2_worse = ["x", "y", "z"]
    mb = evaluator.evaluate_query(stage1_better, relevant, k=3)
    mw = evaluator.evaluate_query(stage2_worse, relevant, k=3)
    assert mb.recall_at_k > mw.recall_at_k


# 22. same ground truth is used for both
def test_same_ground_truth_both():
    evaluator = RetrievalEvaluator(k=2)
    relevant = ["a", "b"]
    stage1 = ["a", "x"]
    stage2 = ["a", "b"]
    r1 = evaluator.evaluate_query(stage1, relevant, k=2)
    r2 = evaluator.evaluate_query(stage2, relevant, k=2)
    # Both evaluated against same relevant set
    assert r1.recall_at_k == pytest.approx(0.5)  # 1 of 2
    assert r2.recall_at_k == pytest.approx(1.0)  # 2 of 2
    assert r1.precision_at_k == pytest.approx(0.5)
    assert r2.precision_at_k == pytest.approx(1.0)


def test_evaluator_dataset_aggregation():
    evaluator = RetrievalEvaluator(k=2)
    retrieved_lists = [["a", "b"], ["x", "y"]]
    relevant_lists = [["a", "b"], ["a", "b"]]
    agg = evaluator.evaluate_dataset(retrieved_lists, relevant_lists, k=2)
    # mean recall: (1.0 + 0.0)/2 =0.5
    assert agg.mean_recall_at_k == pytest.approx(0.5)
    assert agg.mean_precision_at_k == pytest.approx(0.5)
    assert agg.mrr == pytest.approx(0.5)
    assert agg.num_queries == 2

    # Empty dataset
    agg2 = evaluator.evaluate_dataset([], [])
    assert agg2.mean_recall_at_k == pytest.approx(0.0)
    assert agg2.mrr == pytest.approx(0.0)
    assert agg2.num_queries == 0

    # With latencies and tokens
    contexts = ["hello world", "a b c d"]
    latencies = [10.0, 20.0]
    agg3 = evaluator.evaluate_dataset(
        [["a"], ["a"]], [["a"], ["a"]], k=1, latencies_ms=latencies, formatted_contexts=contexts
    )
    assert agg3.mean_latency_ms == pytest.approx(15.0)
    assert agg3.mean_tokens == pytest.approx((2 + 4) / 2)


def test_evaluator_tokens_via_estimator():
    evaluator = RetrievalEvaluator(k=5)
    ctx = format_context([{"id": "m1", "text": "hello world test", "timestamp": "2026-01-01T00:00:00Z", "source": "s1", "embedding": None}])
    expected = estimate_tokens(ctx)
    qm = evaluator.evaluate_query(["m1"], ["m1"], k=1, formatted_context=ctx)
    assert qm.tokens == expected
    assert qm.tokens == tokens_returned(ctx)


def test_evaluator_dict_api():
    evaluator = RetrievalEvaluator(k=2)
    d = evaluator.evaluate(["a", "b"], ["a"], k=2)
    assert "recall_at_k" in d
    assert "precision_at_k" in d
    assert "reciprocal_rank" in d
    assert "latency_ms" in d
    assert "tokens" in d
