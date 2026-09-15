"""src.eval — retrieval evaluation framework for MemAge."""

from src.eval.dataset import EvalDataset, EvaluationCase, RetrievalEvalCase
from src.eval.evaluator import AggregatedMetrics, QueryMetrics, RetrievalEvaluator
from src.eval.metrics import (
    Timer,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    timed_block,
    tokens_returned,
)

__all__ = [
    "RetrievalEvalCase",
    "EvaluationCase",
    "EvalDataset",
    "RetrievalEvaluator",
    "QueryMetrics",
    "AggregatedMetrics",
    "recall_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "mean_reciprocal_rank",
    "Timer",
    "timed_block",
    "tokens_returned",
]
