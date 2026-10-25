"""Tests for src.eval.metrics — Recall@K, Precision@K, MRR, latency, tokens."""

import time

import pytest

from src.eval.metrics import (
    Timer,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    timed_block,
    tokens_returned,
)
from src.retrieval.token_budget import estimate_tokens


# 1. perfect recall
def test_recall_perfect():
    assert recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == pytest.approx(1.0)
    assert recall_at_k(["a"], ["a"], k=1) == pytest.approx(1.0)


# 2. partial recall
def test_recall_partial():
    # 1 of 2 relevant retrieved in top-3
    assert recall_at_k(["a", "x", "y"], ["a", "b"], k=3) == pytest.approx(0.5)
    assert recall_at_k(["a", "b", "c", "d"], ["a", "b", "c"], k=2) == pytest.approx(2 / 3)


# 3. zero recall
def test_recall_zero():
    assert recall_at_k(["x", "y"], ["a", "b"], k=2) == pytest.approx(0.0)
    assert recall_at_k([], ["a"], k=5) == pytest.approx(0.0)


# 4. K smaller than result list
def test_recall_k_smaller_than_results():
    # 3 relevant total, but only top-1 considered contains 1 relevant
    assert recall_at_k(["a", "b", "c", "d"], ["a", "b", "c"], k=1) == pytest.approx(1 / 3)
    # top-2 contains a,b
    assert recall_at_k(["a", "b", "c"], ["a", "b", "c"], k=2) == pytest.approx(2 / 3)


# 5. K larger than result list
def test_recall_k_larger_than_results():
    # only 1 retrieved, but 2 relevant; even with K=10 only 1 hit
    assert recall_at_k(["a"], ["a", "b"], k=10) == pytest.approx(0.5)
    assert recall_at_k(["a", "b"], ["a", "b", "c"], k=10) == pytest.approx(2 / 3)


# 6. no relevant ground truth
def test_recall_no_relevant():
    # Convention: 0.0 when no relevant (avoids div-by-zero)
    assert recall_at_k(["a", "b"], [], k=2) == pytest.approx(0.0)
    assert recall_at_k([], [], k=5) == pytest.approx(0.0)


# 7. perfect precision
def test_precision_perfect():
    assert precision_at_k(["a", "b"], ["a", "b"], k=2) == pytest.approx(1.0)
    assert precision_at_k(["a"], ["a", "b"], k=1) == pytest.approx(1.0)


# 8. partial precision
def test_precision_partial():
    # 1 relevant of 2 retrieved
    assert precision_at_k(["a", "x"], ["a", "b"], k=2) == pytest.approx(0.5)
    assert precision_at_k(["a", "x", "y"], ["a"], k=3) == pytest.approx(1 / 3)


# 9. zero precision
def test_precision_zero():
    assert precision_at_k(["x", "y"], ["a", "b"], k=2) == pytest.approx(0.0)
    assert precision_at_k([], ["a"], k=2) == pytest.approx(0.0)


# 10. fewer than K results
def test_precision_fewer_than_k():
    # K=5 but only 2 retrieved, 1 relevant => prec = 1/2 (denominator = retrieved count)
    assert precision_at_k(["a", "x"], ["a"], k=5) == pytest.approx(0.5)
    assert precision_at_k(["a"], ["a"], k=5) == pytest.approx(1.0)
    assert precision_at_k([], ["a"], k=5) == pytest.approx(0.0)


# 11. first result relevant
def test_mrr_first_relevant():
    assert reciprocal_rank(["a", "b", "c"], ["a"]) == pytest.approx(1.0)


# 12. relevant at rank 2
def test_mrr_rank_2():
    assert reciprocal_rank(["x", "a", "b"], ["a"]) == pytest.approx(0.5)


# 13. relevant later
def test_mrr_later():
    assert reciprocal_rank(["x", "y", "z", "a"], ["a"]) == pytest.approx(0.25)
    assert reciprocal_rank(["x", "y", "a"], ["a", "b"]) == pytest.approx(1 / 3)


# 14. no relevant result
def test_mrr_no_relevant():
    assert reciprocal_rank(["x", "y"], ["a"]) == pytest.approx(0.0)
    assert reciprocal_rank([], ["a"]) == pytest.approx(0.0)
    assert reciprocal_rank(["a"], []) == pytest.approx(0.0)


# 15. multiple evaluation cases (MRR)
def test_mrr_multiple_cases():
    retrieved = [["a", "b"], ["x", "a"], ["x", "y"]]
    relevant = [["a"], ["a"], ["a"]]
    # RR: 1, 0.5, 0 => MRR 0.5
    assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(0.5)
    # perfect
    assert mean_reciprocal_rank([["a"], ["a"]], [["a"], ["a"]]) == pytest.approx(1.0)
    # empty
    assert mean_reciprocal_rank([], []) == pytest.approx(0.0)


# 16. duplicate retrieved IDs handled consistently
def test_duplicate_handling():
    # duplicate "a" should not inflate recall/precision
    # retrieved ["a","a","b"], relevant ["a","b"] => unique top-3 is ["a","b"] => recall 1.0, precision 1.0
    assert recall_at_k(["a", "a", "b"], ["a", "b"], k=3) == pytest.approx(1.0)
    assert precision_at_k(["a", "a", "b"], ["a", "b"], k=3) == pytest.approx(1.0)
    # duplicate irrelevant
    assert recall_at_k(["a", "a", "a"], ["a", "b"], k=3) == pytest.approx(0.5)
    # RR with duplicate first relevant still rank 1
    assert reciprocal_rank(["a", "a", "b"], ["a"]) == pytest.approx(1.0)
    # duplicate irrelevant before relevant: ["x","x","a"] dedup rank is 2 (x,a) => RR 0.5
    assert reciprocal_rank(["x", "x", "a"], ["a"]) == pytest.approx(0.5)


# 17. metric uses existing retrieval token estimator
def test_tokens_uses_existing_estimator():
    text = "hello world  this is a test"
    # tokens_returned should delegate to same estimator
    assert tokens_returned(text) == estimate_tokens(text)
    assert tokens_returned("") == estimate_tokens("")
    assert tokens_returned("a b c") == 3
    # Verify formatted context token count consistent
    from src.retrieval.formatter import format_context

    ctx = format_context([{"id": "m1", "text": "hello world", "timestamp": "2026-01-01T00:00:00Z", "source": "s1", "embedding": None}])
    assert tokens_returned(ctx) == estimate_tokens(ctx)


# 18. timing utility returns non-negative milliseconds
def test_latency_non_negative():
    t = Timer()
    t.start()
    time.sleep(0.01)
    ms = t.stop()
    assert ms >= 0
    assert t.elapsed_ms >= 0
    # with context manager
    with Timer() as t2:
        time.sleep(0.005)
    assert t2.elapsed_ms >= 0
    with timed_block() as tb:
        time.sleep(0.005)
    assert tb.elapsed_ms >= 0


# 19. do not assert exact timing (just non-negative and plausible)
def test_latency_no_exact_assert():
    with timed_block() as t:
        # short busy work
        sum(range(10000))
    # Should be non-negative and not flaky; upper bound generous
    assert t.elapsed_ms is not None
    assert 0 <= t.elapsed_ms < 5000  # generous upper bound
