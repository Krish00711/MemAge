"""Tests for src.eval.dataset — synthetic dataset and EvaluationCase."""

import pytest

from src.eval.dataset import EvalDataset, EvaluationCase, RetrievalEvalCase


# 23. synthetic dataset can be represented and iterated
def test_synthetic_dataset_representation():
    cases = [
        RetrievalEvalCase(query="q1", relevant_ids=["a", "b"], case_id="c1"),
        RetrievalEvalCase(query="q2", relevant_ids=["c"], metadata={"source": "synthetic"}),
    ]
    ds = EvalDataset(cases=cases, name="test")
    assert len(ds) == 2
    assert ds.name == "test"
    # Iteration preserves order
    queries = [c.query for c in ds]
    assert queries == ["q1", "q2"]
    assert ds[0].relevant_ids == ["a", "b"]
    assert ds[1].metadata["source"] == "synthetic"
    # Optional fields default
    assert ds[0].candidates is None
    assert ds[1].case_id is None
    # from_synthetic helper
    ds2 = EvalDataset.from_synthetic(
        [
            {"query": "hello", "relevant_ids": ["m1"], "case_id": "id1"},
            {"query": "world", "relevant_ids": ["m2", "m3"], "metadata": {"k": 5}},
        ]
    )
    assert len(ds2) == 2
    assert ds2[0].query == "hello"
    assert ds2[0].case_id == "id1"
    assert ds2[1].relevant_ids == ["m2", "m3"]


# 24. empty dataset is handled
def test_empty_dataset():
    ds = EvalDataset(cases=[])
    assert len(ds) == 0
    assert ds.is_empty()
    assert list(ds) == []
    # from_synthetic empty
    ds2 = EvalDataset.from_synthetic([])
    assert len(ds2) == 0
    assert ds2.is_empty()
    # Add case
    ds.add_case(RetrievalEvalCase(query="q", relevant_ids=["a"]))
    assert len(ds) == 1
    assert not ds.is_empty()


def test_evaluation_case_fields():
    case = RetrievalEvalCase(
        query="what is user's preference?",
        relevant_ids=["mem_001"],
        candidates=[{"id": "mem_001", "text": "dark mode", "timestamp": "2026-01-01T00:00:00Z", "source": "s1", "embedding": None}],
        metadata={"latency_ms": 12.3},
        case_id="case_001",
    )
    assert case.query == "what is user's preference?"
    assert case.relevant_ids == ["mem_001"]
    assert len(case.candidates) == 1
    assert case.metadata["latency_ms"] == 12.3
    assert case.case_id == "case_001"
    # Alias works
    assert EvaluationCase is RetrievalEvalCase
