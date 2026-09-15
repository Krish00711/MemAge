"""RetrievalEvaluator for MemAge.

Computes Recall@K, Precision@K, MRR, latency, and tokens returned
without coupling to ChromaDB, LLM providers, or network.
Execution (retrieval) and metric calculation are kept separate;
the evaluator consumes already-retrieved ID lists and ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.eval.metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    tokens_returned,
)


@dataclass
class QueryMetrics:
    """Metrics for a single query."""

    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    latency_ms: Optional[float] = None
    tokens: Optional[int] = None


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across a dataset."""

    mean_recall_at_k: float
    mean_precision_at_k: float
    mrr: float
    mean_latency_ms: Optional[float] = None
    mean_tokens: Optional[float] = None
    num_queries: int = 0


class RetrievalEvaluator:
    """Evaluator for retrieval quality.

    Independent of retrieval implementation; evaluates given retrieved ID
    lists against ground-truth relevant IDs using the same metrics.
    This allows side-by-side Stage-1 vs Stage-1+Stage-2 comparison.

    Args:
        k: Default cutoff for Recall@K / Precision@K if not supplied per call.
    """

    def __init__(self, k: int = 5) -> None:
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError(f"k must be positive int, got {k}")
        self.k = k

    # ------------------------------------------------------------------
    # Single-query evaluation
    # ------------------------------------------------------------------

    def evaluate_query(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: Optional[int] = None,
        latency_ms: Optional[float] = None,
        formatted_context: Optional[str] = None,
    ) -> QueryMetrics:
        """Evaluate a single query.

        Args:
            retrieved_ids: Ordered retrieved IDs (rank most relevant first).
            relevant_ids: Ground-truth relevant IDs.
            k: Cutoff (defaults to evaluator's k).
            latency_ms: Optional retrieval latency in milliseconds.
            formatted_context: Optional final formatted context string;
                tokens will be estimated via the shared token estimator.

        Returns:
            QueryMetrics with recall, precision, RR, and optional latency/tokens.
        """
        kk = k if k is not None else self.k
        rec = recall_at_k(retrieved_ids, relevant_ids, kk)
        prec = precision_at_k(retrieved_ids, relevant_ids, kk)
        rr = reciprocal_rank(retrieved_ids, relevant_ids)
        tokens = tokens_returned(formatted_context) if formatted_context is not None else None
        return QueryMetrics(
            recall_at_k=rec,
            precision_at_k=prec,
            reciprocal_rank=rr,
            latency_ms=latency_ms,
            tokens=tokens,
        )

    # ------------------------------------------------------------------
    # Dataset / multi-query evaluation
    # ------------------------------------------------------------------

    def evaluate_dataset(
        self,
        retrieved_lists: List[List[str]],
        relevant_lists: List[List[str]],
        k: Optional[int] = None,
        latencies_ms: Optional[List[Optional[float]]] = None,
        formatted_contexts: Optional[List[Optional[str]]] = None,
    ) -> AggregatedMetrics:
        """Evaluate multiple queries and aggregate.

        Args:
            retrieved_lists: One retrieved list per query.
            relevant_lists: One relevant list per query (same order).
            k: Cutoff (defaults to evaluator's k).
            latencies_ms: Optional per-query latencies.
            formatted_contexts: Optional per-query formatted contexts for token metric.

        Returns:
            AggregatedMetrics (mean recall, mean precision, MRR, mean latency/tokens).

        Raises:
            ValueError: If retrieved/relevant length mismatch.
        """
        if len(retrieved_lists) != len(relevant_lists):
            raise ValueError(
                f"retrieved_lists and relevant_lists length mismatch: {len(retrieved_lists)} vs {len(relevant_lists)}"
            )
        n = len(retrieved_lists)
        if n == 0:
            return AggregatedMetrics(
                mean_recall_at_k=0.0,
                mean_precision_at_k=0.0,
                mrr=0.0,
                mean_latency_ms=None,
                mean_tokens=None,
                num_queries=0,
            )
        kk = k if k is not None else self.k
        per_query = [
            self.evaluate_query(ret, rel, k=kk)
            for ret, rel in zip(retrieved_lists, relevant_lists)
        ]
        mean_rec = sum(q.recall_at_k for q in per_query) / n
        mean_prec = sum(q.precision_at_k for q in per_query) / n
        mrr = mean_reciprocal_rank(retrieved_lists, relevant_lists)

        # Optional latencies
        mean_lat = None
        if latencies_ms is not None:
            vals = [v for v in latencies_ms if v is not None]
            mean_lat = sum(vals) / len(vals) if vals else None

        # Optional tokens
        mean_tokens = None
        if formatted_contexts is not None:
            token_vals = [
                tokens_returned(ctx) if ctx is not None else 0
                for ctx in formatted_contexts
            ]
            mean_tokens = sum(token_vals) / len(token_vals) if token_vals else None

        return AggregatedMetrics(
            mean_recall_at_k=mean_rec,
            mean_precision_at_k=mean_prec,
            mrr=mrr,
            mean_latency_ms=mean_lat,
            mean_tokens=mean_tokens,
            num_queries=n,
        )

    # Convenience: evaluate with explicit dict return
    def evaluate(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: Optional[int] = None,
        latency_ms: Optional[float] = None,
        formatted_context: Optional[str] = None,
    ) -> Dict:
        """Evaluate single query and return plain dict (JSON-friendly)."""
        qm = self.evaluate_query(retrieved_ids, relevant_ids, k, latency_ms, formatted_context)
        return {
            "recall_at_k": qm.recall_at_k,
            "precision_at_k": qm.precision_at_k,
            "reciprocal_rank": qm.reciprocal_rank,
            "latency_ms": qm.latency_ms,
            "tokens": qm.tokens,
        }
