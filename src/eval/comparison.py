"""Baseline comparison orchestration for MemAge.

Runs multiple strategies (``retrieval_fn`` callables) against the SAME
``BenchmarkDataset`` and aggregates one :class:`StrategyResult` per strategy
into a single :class:`ComparisonResult`.

Design:
    * Orchestration lives here, not in the CLI: ``src/eval/run_benchmark.py``
      only wires strategy construction and prints.
    * Reuses the existing ``BenchmarkRunner`` per strategy — no metric is
      reimplemented; every aggregate is exactly what ``BenchmarkReport``
      already computes (F1, BLEU-1, Recall@K, Precision@K, MRR, mean tokens,
      mean latency). Retrieval metrics appear as ``None`` for strategies that
      do not rank (e.g. ``full_history``), same as single-baseline runs.
    * Isolation: each strategy gets its own ``BenchmarkRunner`` and a fresh
      copy of the case list; cases are treated as read-only (the runner never
      mutates them), and strategies share nothing mutable. Strategy order
      does not affect any strategy's numbers.
    * Serialization-ready (but NOT serialized here): all result fields are
      plain dataclasses of primitives, so ``dataclasses.asdict(result)`` is
      enough for the upcoming JSON/CSV task. No file I/O happens in this
      module.

Example:
    result = compare_strategies(
        dataset,
        strategies=[StrategySpec("none", none_fn), StrategySpec("hmem", hmem_fn)],
        answer_fn=answer_fn,
        k=5,
    )
    print(format_comparison_table(result))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from src.eval.benchmark import BenchmarkReport, BenchmarkRunner
from src.eval.dataset import BenchmarkCase, BenchmarkDataset

# retrieval_fn contract: (query) -> str | RetrievalOutcome | dict | (context, ids)
RetrievalFn = Callable[[str], Any]
# answer_fn contract: (query, context) -> predicted_answer
AnswerFn = Callable[[str, str], str]


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategySpec:
    """One enabled strategy for a comparison.

    Attributes:
        name: Unique label used in results (e.g. ``"none"``, ``"memoryos"``).
        retrieval_fn: Callable ``(query) ->`` any value the existing
            ``BenchmarkRunner._normalize_retrieval_output`` accepts
            (``RetrievalOutcome``, ``str``, ``dict``, or ``(context, ids)``).
        answer_fn: Optional per-strategy answer function; falls back to the
            comparison-level ``answer_fn`` when ``None`` (keeps comparisons
            fair by default: every strategy answers with the same function).
    """

    name: str
    retrieval_fn: RetrievalFn
    answer_fn: Optional[AnswerFn] = None


@dataclass
class StrategyResult:
    """Aggregate result of one strategy on the shared dataset.

    Attributes:
        name: The ``StrategySpec.name`` this result belongs to.
        report: The unmodified ``BenchmarkReport`` produced by
            ``BenchmarkRunner.run_dataset`` (mean_f1, mean_bleu1,
            mean_latency_ms, mean_tokens, mean_recall_at_k,
            mean_precision_at_k, mrr, num_queries, per_case).
    """

    name: str
    report: BenchmarkReport


@dataclass
class ComparisonResult:
    """Single comparison outcome across strategies.

    Attributes:
        dataset_name: Name of the dataset all strategies ran on.
        k: Metric cutoff used for every strategy's ``BenchmarkRunner``.
        num_cases: Number of dataset cases run per strategy.
        results: One :class:`StrategyResult` per strategy, input order.
    """

    dataset_name: str
    k: int
    num_cases: int
    results: List[StrategyResult] = field(default_factory=list)

    @property
    def names(self) -> List[str]:
        """Strategy names in input order."""
        return [r.name for r in self.results]

    def get(self, name: str) -> StrategyResult:
        """Return the result for *name*.

        Raises:
            KeyError: If no strategy with that name ran.
        """
        for r in self.results:
            if r.name == name:
                return r
        raise KeyError(f"no strategy named {name!r} in comparison (ran: {self.names})")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def compare_strategies(
    dataset: Union[BenchmarkDataset, Sequence[BenchmarkCase]],
    strategies: Sequence[StrategySpec],
    answer_fn: AnswerFn,
    k: int = 5,
) -> ComparisonResult:
    """Run every strategy on the same cases and collect aggregate results.

    Each strategy runs in its own :class:`BenchmarkRunner` over a fresh copy
    of the case list, so strategies cannot affect each other (no shared
    runner state, no list aliasing, cases treated as read-only).

    Args:
        dataset: ``BenchmarkDataset`` or sequence of ``BenchmarkCase``.
        strategies: Enabled strategies (>= 1), names must be non-empty and
            unique (results are keyed by name).
        answer_fn: Shared ``(query, context) -> answer`` used by any strategy
            that does not supply its own ``answer_fn``.
        k: Metric cutoff passed to every ``BenchmarkRunner``.

    Returns:
        ``ComparisonResult`` with one ``StrategyResult`` per strategy, in
        input order.

    Raises:
        TypeError: If *strategies* is not a sequence of ``StrategySpec`` or
            *answer_fn* is not callable.
        ValueError: If *strategies* is empty, a name is empty/duplicated, a
            ``retrieval_fn`` is not callable, or *k* is not a positive int.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    if not callable(answer_fn):
        raise TypeError("answer_fn must be callable")
    if isinstance(strategies, (str, bytes)) or not isinstance(strategies, (list, tuple)):
        raise TypeError("strategies must be a sequence of StrategySpec")
    if not strategies:
        raise ValueError("strategies must contain at least one StrategySpec")

    seen: set = set()
    for spec in strategies:
        if not isinstance(spec, StrategySpec):
            raise TypeError(f"strategies entries must be StrategySpec, got {type(spec)!r}")
        if not isinstance(spec.name, str) or not spec.name:
            raise ValueError(f"strategy name must be a non-empty string, got {spec.name!r}")
        if spec.name in seen:
            raise ValueError(f"duplicate strategy name {spec.name!r}")
        seen.add(spec.name)
        if not callable(spec.retrieval_fn):
            raise ValueError(f"strategy {spec.name!r} retrieval_fn must be callable")
        if spec.answer_fn is not None and not callable(spec.answer_fn):
            raise ValueError(f"strategy {spec.name!r} answer_fn must be callable or None")

    if isinstance(dataset, BenchmarkDataset):
        cases: List[BenchmarkCase] = list(dataset.cases)
        dataset_name = dataset.name
    else:
        cases = list(dataset)
        dataset_name = "custom"

    results: List[StrategyResult] = []
    for spec in strategies:
        runner = BenchmarkRunner(
            retrieval_fn=spec.retrieval_fn,
            answer_fn=spec.answer_fn if spec.answer_fn is not None else answer_fn,
            k=k,
        )
        # Fresh list per strategy: strategies can never share/alias the case
        # sequence they iterate (isolation guarantee).
        report = runner.run_dataset(list(cases))
        results.append(StrategyResult(name=spec.name, report=report))

    return ComparisonResult(
        dataset_name=dataset_name,
        k=k,
        num_cases=len(cases),
        results=results,
    )


# ---------------------------------------------------------------------------
# Presentation (no serialization - that is a later task)
# ---------------------------------------------------------------------------


def _fmt(value: Optional[float], spec: str) -> str:
    """Format an optional aggregate; ``None`` renders as ``-``."""
    if value is None:
        return "-"
    return format(value, spec)


def format_comparison_table(result: ComparisonResult) -> str:
    """Render *result* as a deterministic aligned text table.

    Columns are the metrics the project already supports: F1, BLEU-1,
    Recall@K, Precision@K, MRR, mean tokens, mean latency (ms). Strategies
    without retrieval metrics (e.g. ``full_history``) show ``-``.

    Returns:
        Multi-line string (no trailing newline); empty-strategy results are
        impossible (``compare_strategies`` requires >= 1 strategy).
    """
    header = [
        "Strategy",
        "F1",
        "BLEU-1",
        f"Recall@{result.k}",
        f"P@{result.k}",
        "MRR",
        "Tokens",
        "Latency(ms)",
    ]
    rows: List[List[str]] = []
    for r in result.results:
        rep = r.report
        rows.append(
            [
                r.name,
                _fmt(rep.mean_f1, ".3f"),
                _fmt(rep.mean_bleu1, ".3f"),
                _fmt(rep.mean_recall_at_k, ".3f"),
                _fmt(rep.mean_precision_at_k, ".3f"),
                _fmt(rep.mrr, ".3f"),
                _fmt(rep.mean_tokens, ".1f"),
                _fmt(rep.mean_latency_ms, ".2f"),
            ]
        )

    widths = [
        max(len(header[i]), max((len(row[i]) for row in rows), default=0))
        for i in range(len(header))
    ]

    def line(cells: List[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            if i == 0:
                parts.append(cell.ljust(widths[i]))
            else:
                parts.append(cell.rjust(widths[i]))
        return "  ".join(parts).rstrip()

    lines = [line(header)]
    lines.extend(line(row) for row in rows)
    return "\n".join(lines)
