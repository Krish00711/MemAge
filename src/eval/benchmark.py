"""End-to-end benchmark orchestration for MemAge.

Orchestrates:

  BenchmarkCase
    → memory preparation (injected, optional)
    → retrieval (injected retrieval_fn → context + optional retrieved_ids)
    → answer generation (injected answer_fn)
    → QA evaluation (F1/BLEU-1) + optional retrieval metrics

Uses injected callables/protocols — no direct store/controller/LLM access.
Supports both memory-based system and full-history upper bound via different
injected retrieval_fn/answer_fn (strategy interface).

Reuses Phase-5 metrics: Recall@K, Precision@K, MRR, tokens, Timer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from src.eval.dataset import BenchmarkCase, BenchmarkDataset
from src.eval.metrics import (
    Timer,
    bleu1_score,
    f1_score,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    tokens_returned,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RetrievalOutcome:
    """Result of a retrieval step.

    Attributes:
        context: Formatted retrieval context string (may be empty).
        retrieved_ids: Optional ordered retrieved memory IDs for metric
            integration. None if unavailable.
        tokens: Optional token count (computed from context if not supplied).
    """

    context: str
    retrieved_ids: Optional[List[str]] = None
    tokens: Optional[int] = None


@dataclass
class BenchmarkResult:
    """Per-query benchmark result."""

    case_id: Optional[str]
    query: str
    predicted_answer: str
    reference_answer: str
    f1: float
    bleu1: float
    latency_ms: float
    tokens: Optional[int] = None
    # Optional retrieval integration
    retrieval_recall: Optional[float] = None
    retrieval_precision: Optional[float] = None
    retrieval_rr: Optional[float] = None
    retrieved_ids: Optional[List[str]] = None
    relevant_ids: Optional[List[str]] = None


@dataclass
class BenchmarkReport:
    """Aggregate benchmark report across a dataset."""

    num_queries: int
    mean_f1: float
    mean_bleu1: float
    mean_latency_ms: float
    mean_tokens: Optional[float] = None
    mean_recall_at_k: Optional[float] = None
    mean_precision_at_k: Optional[float] = None
    mrr: Optional[float] = None
    per_case: List[BenchmarkResult] = field(default_factory=list)


# Type aliases for injected callables
# retrieval_fn: (query: str) -> str | RetrievalOutcome | dict
RetrievalFn = Callable[[str], Union[str, RetrievalOutcome, Dict[str, Any]]]
# answer_fn: (query: str, context: str) -> str
AnswerFn = Callable[[str, str], str]
# memory_prepare_fn: (case: BenchmarkCase) -> None
MemoryPrepareFn = Callable[[BenchmarkCase], Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_retrieval_output(raw: Union[str, RetrievalOutcome, Dict[str, Any]]) -> RetrievalOutcome:
    """Normalize retrieval_fn return to RetrievalOutcome.

    Supports:
      - str → context only
      - RetrievalOutcome → as-is
      - dict with keys 'context'/'retrieved_ids'/'tokens'
    """
    if isinstance(raw, RetrievalOutcome):
        # Ensure tokens computed if not supplied
        if raw.tokens is None:
            raw.tokens = tokens_returned(raw.context)
        return raw
    if isinstance(raw, str):
        return RetrievalOutcome(context=raw, tokens=tokens_returned(raw))
    if isinstance(raw, dict):
        ctx = raw.get("context", raw.get("formatted_context", ""))
        rids = raw.get("retrieved_ids", raw.get("ids"))
        toks = raw.get("tokens")
        if toks is None and isinstance(ctx, str):
            toks = tokens_returned(ctx)
        return RetrievalOutcome(context=ctx, retrieved_ids=rids, tokens=toks)
    # Fallback: stringify
    ctx = str(raw)
    return RetrievalOutcome(context=ctx, tokens=tokens_returned(ctx))


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Orchestrates end-to-end benchmark for a strategy.

    The runner is strategy-agnostic: inject different ``retrieval_fn`` and
    ``answer_fn`` to represent memory-based system vs full-history upper bound.

    Args:
        retrieval_fn: Callable ``(query) -> context`` or ``RetrievalOutcome``.
            If None, answer is generated with empty context (or caller provides
            context via answer_fn). For full-history, inject a function that
            returns full-history context.
        answer_fn: Callable ``(query, context) -> predicted_answer``. Required.
        memory_prepare_fn: Optional callable ``(BenchmarkCase) -> None`` for
            per-case store preparation (no-op if None). Keeps benchmark valid
            when memory construction is unavailable.
        k: Cutoff for retrieval metrics when retrieved_ids are available.

    Example:
        memory_runner = BenchmarkRunner(
            retrieval_fn=lambda q: retrieval_context.retrieve_context(q, 500),
            answer_fn=lambda q, ctx: fake_llm(f"Q:{q}\\nContext:{ctx}"),
        )
        full_history_runner = BenchmarkRunner(
            retrieval_fn=lambda q: full_history_text,
            answer_fn=lambda q, ctx: fake_llm(f"Q:{q}\\nHistory:{ctx}"),
        )
        report = memory_runner.run_dataset(benchmark_cases)
    """

    def __init__(
        self,
        retrieval_fn: Optional[RetrievalFn] = None,
        answer_fn: Optional[AnswerFn] = None,
        memory_prepare_fn: Optional[MemoryPrepareFn] = None,
        k: int = 5,
    ) -> None:
        if answer_fn is not None and not callable(answer_fn):
            raise TypeError("answer_fn must be callable")
        if retrieval_fn is not None and not callable(retrieval_fn):
            raise TypeError("retrieval_fn must be callable")
        if memory_prepare_fn is not None and not callable(memory_prepare_fn):
            raise TypeError("memory_prepare_fn must be callable")
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError(f"k must be positive int, got {k}")
        self.retrieval_fn = retrieval_fn
        self.answer_fn = answer_fn
        self.memory_prepare_fn = memory_prepare_fn
        self.k = k

    def run_single(self, case: BenchmarkCase) -> BenchmarkResult:
        """Run benchmark for a single case.

        Steps:
          1. memory preparation (optional)
          2. retrieval (timed as part of end-to-end latency)
          3. answer generation
          4. QA + optional retrieval metrics

        End-to-end latency is measured from start of retrieval to answer
        completion using monotonic clock.

        Args:
            case: BenchmarkCase with query and reference.

        Returns:
            BenchmarkResult with prediction, metrics, latency, tokens.
        """
        # Memory preparation (optional, not timed as part of latency? Include as setup)
        if self.memory_prepare_fn is not None:
            try:
                self.memory_prepare_fn(case)
            except Exception:
                # Preparation failure should not crash benchmark; continue
                pass

        timer = Timer().start()
        # Retrieval
        retrieval_out: RetrievalOutcome
        if self.retrieval_fn is not None:
            try:
                raw = self.retrieval_fn(case.query)
                retrieval_out = _normalize_retrieval_output(raw)
            except Exception:
                # Retrieval failure → empty context
                retrieval_out = RetrievalOutcome(context="", retrieved_ids=None, tokens=0)
        else:
            retrieval_out = RetrievalOutcome(context="", retrieved_ids=None, tokens=0)

        context = retrieval_out.context
        retrieved_ids = retrieval_out.retrieved_ids

        # Answer generation
        predicted = ""
        if self.answer_fn is not None:
            try:
                predicted = self.answer_fn(case.query, context)
                if predicted is None:
                    predicted = ""
                elif not isinstance(predicted, str):
                    predicted = str(predicted)
            except Exception:
                predicted = ""
        # Stop timer after answer generation (end-to-end)
        latency_ms = timer.stop()

        # QA metrics
        f1 = f1_score(predicted, case.reference_answer)
        bleu1 = bleu1_score(predicted, case.reference_answer)

        # Tokens (from retrieval outcome or computed)
        tokens = retrieval_out.tokens
        if tokens is None:
            tokens = tokens_returned(context)

        # Retrieval metric integration (only if retrieved_ids available and relevant_ids exist)
        ret_recall: Optional[float] = None
        ret_prec: Optional[float] = None
        ret_rr: Optional[float] = None
        if retrieved_ids is not None and case.relevant_ids is not None:
            try:
                ret_recall = recall_at_k(retrieved_ids, case.relevant_ids, self.k)
                ret_prec = precision_at_k(retrieved_ids, case.relevant_ids, self.k)
                ret_rr = reciprocal_rank(retrieved_ids, case.relevant_ids)
            except Exception:
                ret_recall = ret_prec = ret_rr = None

        return BenchmarkResult(
            case_id=case.case_id,
            query=case.query,
            predicted_answer=predicted,
            reference_answer=case.reference_answer,
            f1=f1,
            bleu1=bleu1,
            latency_ms=latency_ms,
            tokens=tokens,
            retrieval_recall=ret_recall,
            retrieval_precision=ret_prec,
            retrieval_rr=ret_rr,
            retrieved_ids=retrieved_ids,
            relevant_ids=list(case.relevant_ids) if case.relevant_ids else [],
        )

    def run_dataset(
        self, dataset: Union[BenchmarkDataset, List[BenchmarkCase]]
    ) -> BenchmarkReport:
        """Run benchmark across a dataset.

        Args:
            dataset: BenchmarkDataset or list of BenchmarkCase.

        Returns:
            BenchmarkReport with per-case results and aggregated means.
        """
        cases: List[BenchmarkCase]
        if isinstance(dataset, BenchmarkDataset):
            cases = list(dataset.cases)
        elif isinstance(dataset, list):
            cases = dataset
        else:
            # Generic iterable
            cases = list(dataset)  # type: ignore[arg-type]

        per_case: List[BenchmarkResult] = []
        for case in cases:
            per_case.append(self.run_single(case))

        n = len(per_case)
        if n == 0:
            return BenchmarkReport(
                num_queries=0,
                mean_f1=0.0,
                mean_bleu1=0.0,
                mean_latency_ms=0.0,
                mean_tokens=None,
                mean_recall_at_k=None,
                mean_precision_at_k=None,
                mrr=None,
                per_case=[],
            )
        mean_f1 = sum(r.f1 for r in per_case) / n
        mean_bleu1 = sum(r.bleu1 for r in per_case) / n
        mean_lat = sum(r.latency_ms for r in per_case) / n
        # Mean tokens (average over available)
        token_vals = [r.tokens for r in per_case if r.tokens is not None]
        mean_tokens = sum(token_vals) / len(token_vals) if token_vals else None
        # Retrieval aggregates (only where available)
        recall_vals = [r.retrieval_recall for r in per_case if r.retrieval_recall is not None]
        prec_vals = [r.retrieval_precision for r in per_case if r.retrieval_precision is not None]
        rr_vals = [r.retrieval_rr for r in per_case if r.retrieval_rr is not None]
        mean_rec = sum(recall_vals) / len(recall_vals) if recall_vals else None
        mean_prec = sum(prec_vals) / len(prec_vals) if prec_vals else None
        mrr = sum(rr_vals) / len(rr_vals) if rr_vals else None

        return BenchmarkReport(
            num_queries=n,
            mean_f1=mean_f1,
            mean_bleu1=mean_bleu1,
            mean_latency_ms=mean_lat,
            mean_tokens=mean_tokens,
            mean_recall_at_k=mean_rec,
            mean_precision_at_k=mean_prec,
            mrr=mrr,
            per_case=per_case,
        )
