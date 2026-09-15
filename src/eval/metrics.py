"""Retrieval-quality metrics for MemAge evaluation.

Implements Recall@K, Precision@K, reciprocal rank / MRR, latency timing,
and token-count (reusing deterministic estimator from retrieval).

Conventions (explicit):
  Recall@K = |relevant ∩ retrieved[:K]| / |relevant|
    - If no relevant items, returns 0.0 (undefined recall; safe default).
    - Duplicate retrieved IDs are deduplicated; each relevant counted once.
    - If K > len(retrieved), considers all retrieved.

  Precision@K = |relevant ∩ retrieved[:K]| / |retrieved[:K] considered|
    - Denominator = min(K, len(unique retrieved[:K])) ; if no retrieved, 0.0.
    - Duplicate IDs deduplicated consistently with recall.

  MRR: mean of reciprocal_rank across queries.
    - reciprocal_rank = 1 / rank_of_first_relevant (1-indexed), 0 if none.

  Latency: monotonic clock, milliseconds (time.monotonic).

  Tokens: whitespace estimate via src.retrieval.token_budget.estimate_tokens
    - Reuses same estimator as retrieval budget for consistency.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterable, List


# Reuse existing deterministic estimator for consistency
try:
    from src.retrieval.token_budget import estimate_tokens as _estimate_tokens
except Exception:  # fallback if retrieval not importable in isolated tests
    def _estimate_tokens(text: str) -> int:
        if not text or not text.strip():
            return 0
        return len(text.strip().split())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deduplicate_preserve_order(ids: List[str]) -> List[str]:
    """Deduplicate while preserving first-occurrence order."""
    seen: set[str] = set()
    out: List[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _top_k_unique(retrieved_ids: List[str], k: int) -> List[str]:
    """Return top-K unique IDs preserving order."""
    if k <= 0:
        return []
    # Deduplicate first, then slice
    # Note: deduplicate then slice vs slice then dedup — both are defensible.
    # We choose dedup then slice to maximize unique coverage within K,
    # but also handle duplicates inside top-K by deduping the slice.
    # Implementation: slice first, then dedup to avoid counting duplicates
    # beyond K as filling K slots. This matches "consider retrieved[:K] items".
    sliced = retrieved_ids[:k]
    return _deduplicate_preserve_order(sliced)


# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------


def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """Compute Recall@K.

    Recall@K = #relevant retrieved in top-K / #relevant total.

    Args:
        retrieved_ids: Ordered list of retrieved memory IDs (most relevant first).
        relevant_ids: Ground-truth relevant IDs for the query.
        k: Cutoff rank; top-K retrieved considered.

    Returns:
        Recall in [0, 1]. Returns 0.0 if no relevant items (safe handling).
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if not relevant_ids:
        # No relevant ground truth: recall is 0.0 by convention (undefined).
        # Documented explicitly; avoids division by zero.
        return 0.0
    if not retrieved_ids:
        return 0.0
    top_k = _top_k_unique(retrieved_ids, k)
    relevant_set = set(relevant_ids)
    retrieved_set = set(top_k)
    hits = len(relevant_set & retrieved_set)
    return hits / len(relevant_set)


# ---------------------------------------------------------------------------
# Precision@K
# ---------------------------------------------------------------------------


def precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """Compute Precision@K.

    Precision@K = #relevant in top-K / #considered in top-K.

    Denominator is min(K, number of unique retrieved considered);
    if no retrieved considered, returns 0.0.

    Args:
        retrieved_ids: Ordered retrieved IDs.
        relevant_ids: Ground-truth relevant IDs.
        k: Cutoff rank.

    Returns:
        Precision in [0, 1]. Returns 0.0 if no items retrieved.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if not retrieved_ids:
        return 0.0
    top_k = _top_k_unique(retrieved_ids, k)
    if not top_k:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = len(set(top_k) & relevant_set)
    denominator = len(top_k)  # unique count considered in top-K
    # Alternative convention denominator = min(k, len(retrieved)) would count
    # duplicates; we use unique count consistently with recall.
    return hits / denominator if denominator > 0 else 0.0


# ---------------------------------------------------------------------------
# Reciprocal Rank / MRR
# ---------------------------------------------------------------------------


def reciprocal_rank(
    retrieved_ids: List[str],
    relevant_ids: List[str],
) -> float:
    """Compute reciprocal rank for a single query.

    RR = 1 / rank_of_first_relevant (1-indexed), 0 if none.

    Args:
        retrieved_ids: Ordered retrieved IDs.
        relevant_ids: Ground-truth relevant IDs.

    Returns:
        Reciprocal rank in [0, 1].
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    # Consider deduplicated order for rank? We consider original order but
    # skip duplicate IDs (first occurrence determines rank).
    seen: set[str] = set()
    rank = 0
    for rid in retrieved_ids:
        if rid in seen:
            continue
        seen.add(rid)
        rank += 1
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    retrieved_lists: List[List[str]],
    relevant_lists: List[List[str]],
) -> float:
    """Compute MRR across multiple queries.

    MRR = mean of reciprocal_rank for each query.

    Args:
        retrieved_lists: List of retrieved ID lists, one per query.
        relevant_lists: List of relevant ID lists, one per query.

    Returns:
        Mean reciprocal rank in [0,1]. Returns 0.0 if no queries.

    Raises:
        ValueError: If lists have mismatched lengths.
    """
    if len(retrieved_lists) != len(relevant_lists):
        raise ValueError(
            f"retrieved_lists and relevant_lists must have same length, got {len(retrieved_lists)} vs {len(relevant_lists)}"
        )
    if not retrieved_lists:
        return 0.0
    scores = [
        reciprocal_rank(ret, rel)
        for ret, rel in zip(retrieved_lists, relevant_lists)
    ]
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


class Timer:
    """Monotonic timer reporting milliseconds.

    Usage:
        t = Timer()
        t.start()
        # ... retrieval ...
        ms = t.stop()  # or t.elapsed_ms

    Also supports context-manager: `with Timer() as t: ...`
    """

    def __init__(self) -> None:
        self._start: float | None = None
        self._elapsed_ms: float | None = None

    def start(self) -> "Timer":
        self._start = time.monotonic()
        self._elapsed_ms = None
        return self

    def stop(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer not started")
        end = time.monotonic()
        self._elapsed_ms = (end - self._start) * 1000.0
        return self._elapsed_ms

    @property
    def elapsed_ms(self) -> float | None:
        return self._elapsed_ms

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            self.stop()
        except Exception:
            pass


@contextmanager
def timed_block():
    """Context manager yielding Timer, using monotonic clock.

    Example:
        with timed_block() as t:
            do_retrieval()
        print(t.elapsed_ms)
    """
    t = Timer().start()
    try:
        yield t
    finally:
        try:
            if t.elapsed_ms is None:
                t.stop()
        except Exception:
            pass


def measure_latency_ms(start_monotonic: float, end_monotonic: float) -> float:
    """Convert monotonic start/end to milliseconds."""
    return (end_monotonic - start_monotonic) * 1000.0


# ---------------------------------------------------------------------------
# Tokens returned
# ---------------------------------------------------------------------------


def tokens_returned(formatted_context: str) -> int:
    """Estimate tokens in final formatted retrieval context.

    Reuses ``src.retrieval.token_budget.estimate_tokens`` for consistency
    with budget constraint. Deterministic whitespace-based approximation.

    Args:
        formatted_context: Final context string from formatter.

    Returns:
        Approximate token count (int).
    """
    return _estimate_tokens(formatted_context)


def tokens_for_formatted_context(formatted_context: str) -> int:
    """Alias for tokens_returned."""
    return tokens_returned(formatted_context)
