"""src.eval — retrieval evaluation + benchmark framework for MemAge."""

from src.eval.adapters import (
    DatasetFileNotFoundError,
    DatasetFormatError,
    load_dataset,
    load_locomo,
    load_longmemeval,
    resolve_dataset_file,
    to_eval_dataset,
)
from src.eval.benchmark import (
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkRunner,
    RetrievalOutcome,
    make_retrieval_fn,
)
from src.eval.dataset import (
    BenchmarkCase,
    BenchmarkDataset,
    EvalDataset,
    EvaluationCase,
    RetrievalEvalCase,
)
from src.eval.evaluator import AggregatedMetrics, QueryMetrics, RetrievalEvaluator
from src.eval.metrics import (
    Timer,
    bleu1_score,
    f1_score,
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
    "BenchmarkCase",
    "BenchmarkDataset",
    "RetrievalEvaluator",
    "QueryMetrics",
    "AggregatedMetrics",
    "BenchmarkRunner",
    "BenchmarkReport",
    "BenchmarkResult",
    "RetrievalOutcome",
    "make_retrieval_fn",
    "load_locomo",
    "load_longmemeval",
    "load_dataset",
    "resolve_dataset_file",
    "to_eval_dataset",
    "DatasetFormatError",
    "DatasetFileNotFoundError",
    "recall_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "mean_reciprocal_rank",
    "f1_score",
    "bleu1_score",
    "Timer",
    "timed_block",
    "tokens_returned",
]
