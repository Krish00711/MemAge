"""Ablation experiment infrastructure for MemAge.

Runs one *ablation family* — a set of labeled configurations of a single
strategy — against the same dataset and aggregates everything into one
structured result.

Three required families (constructors below):

1. **Retrieval-stage** (:func:`stage_configs`): coarse/vector retrieval only
   (Stage-2 bypassed via :class:`PassthroughReranker`) vs. the full
   two-stage pipeline with reranking.
2. **K** (:func:`k_configs`): the retrieval strategy's Stage-1 candidate
   count ``k`` swept over caller-supplied values (never hard-coded here).
3. **Token-budget** (:func:`budget_configs`): the strategy's token budget
   swept over caller-supplied values.

Design:
    * Reuses the comparison layer wholesale: every ablation is a
      :func:`src.eval.comparison.compare_strategies` call with one
      ``StrategySpec`` per configuration (label = spec name). That gives us
      the existing ``BenchmarkRunner`` per config, the existing metrics
      (no metric is added or reimplemented), per-config isolation (fresh
      case list per run, cases read-only), and ``format_comparison_table``.
    * Deterministic, unique labels: ``stage=coarse_only``, ``k=3``,
      ``token_budget=64`` — derived only from config params, validated for
      uniqueness before any run.
    * Structured for the future JSON/CSV task: ``AblationResult`` nests
      plain dataclasses of primitives (``AblationConfig.params`` is a dict
      of primitives), so ``dataclasses.asdict(result)`` is serialization-
      ready. No export happens in this module.
    * Programmatic-first: call :func:`run_ablation` with any
      ``strategy_builder``; the CLI is a thin wrapper. No dependency on
      ``src/store``, ``src/controller``, ``src/extraction``, or ``src/llm``.

Example:
    configs = k_configs([1, 3, 5])
    result = run_ablation(dataset, configs, my_builder, answer_fn, k=5, ablation="k")
    print(format_comparison_table(result.comparison))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Union

from src.eval.benchmark import BenchmarkReport
from src.eval.comparison import (
    ComparisonResult,
    StrategyResult,
    StrategySpec,
    compare_strategies,
)
from src.eval.dataset import BenchmarkCase, BenchmarkDataset

# Builder contract: one strategy spec for a given ablation configuration.
StrategyBuilder = Callable[["AblationConfig"], StrategySpec]

# The two required retrieval-stage values (fixed pair by requirement).
STAGE_VALUES = ("coarse_only", "two_stage")


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AblationConfig:
    """One ablation configuration with a deterministic unique label.

    Attributes:
        kind: Ablation family (``"retrieval_stage"``, ``"k"``,
            ``"token_budget"``, or a custom family name).
        label: Deterministic unique display name, e.g. ``"k=3"``. Used as
            the ``StrategySpec.name`` so results are keyed by it.
        params: Kind-specific parameters as primitives (serialization-ready
            for the upcoming JSON/CSV task), e.g. ``{"k": 3}``.
    """

    kind: str
    label: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AblationResult:
    """Result of one ablation family: configs + the shared-dataset comparison.

    Attributes:
        ablation: Family name supplied to :func:`run_ablation`.
        configs: The validated configurations, input order.
        comparison: The reused ``ComparisonResult`` — one
            ``StrategyResult`` per config, in input order, each holding the
            existing ``BenchmarkReport`` metrics.
    """

    ablation: str
    configs: List[AblationConfig]
    comparison: ComparisonResult

    @property
    def labels(self) -> List[str]:
        """Configuration labels in input order (= strategy names)."""
        return [c.label for c in self.configs]

    def get(self, label: str) -> StrategyResult:
        """Return the ``StrategyResult`` for *label* (KeyError if absent)."""
        return self.comparison.get(label)


# ---------------------------------------------------------------------------
# Stage-2 bypass (retrieval-stage ablation)
# ---------------------------------------------------------------------------


class PassthroughReranker:
    """Stage-2 bypass: returns candidates unchanged, in Stage-1 order.

    Used by the coarse/vector-only ablation variant so the pipeline runs
    ``coarse retrieval -> (no reranking) -> budget -> format``. Satisfies
    ``RetrievalContext``'s ``rerank(query, candidates)`` contract; does not
    score, reorder, or mutate candidates (a fresh list is returned).
    """

    def rerank(self, query: str, candidates: List) -> List:
        """Return *candidates* as-is (copied list, original order)."""
        return list(candidates)


# ---------------------------------------------------------------------------
# Configuration constructors (deterministic, validated)
# ---------------------------------------------------------------------------


def _validate_sweep_values(values: Sequence[int], name: str) -> List[int]:
    """Validate a sweep list: non-empty, positive ints (no bools), unique."""
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a sequence of positive integers")
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    out: List[int] = []
    for v in values:
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            raise ValueError(f"{name} values must be positive integers, got {v!r}")
        out.append(v)
    if len(set(out)) != len(out):
        raise ValueError(f"{name} contains duplicate values: {out}")
    return out


def stage_configs() -> List[AblationConfig]:
    """The two required retrieval-stage configurations (fixed pair).

    Returns:
        ``[coarse_only, two_stage]`` — labels ``stage=coarse_only`` and
        ``stage=two_stage``; params ``{"stage": ...}``.
    """
    return [
        AblationConfig(kind="retrieval_stage", label="stage=coarse_only",
                       params={"stage": "coarse_only"}),
        AblationConfig(kind="retrieval_stage", label="stage=two_stage",
                       params={"stage": "two_stage"}),
    ]


def k_configs(ks: Sequence[int]) -> List[AblationConfig]:
    """K ablation over caller-supplied Stage-1 candidate counts.

    Args:
        ks: Positive, unique integers (e.g. ``[1, 3, 5]``). Nothing is
            hard-coded — the caller/CLI supplies the set.

    Returns:
        Configs labeled ``k=<v>`` with params ``{"k": v}``.

    Raises:
        TypeError/ValueError: Invalid or empty/duplicate *ks*.
    """
    values = _validate_sweep_values(ks, "ks")
    return [
        AblationConfig(kind="k", label=f"k={v}", params={"k": v}) for v in values
    ]


def budget_configs(budgets: Sequence[int]) -> List[AblationConfig]:
    """Token-budget ablation over caller-supplied budgets.

    Args:
        budgets: Positive, unique integers (e.g. ``[12, 512]``).

    Returns:
        Configs labeled ``token_budget=<v>`` with params
        ``{"token_budget": v}``.

    Raises:
        TypeError/ValueError: Invalid or empty/duplicate *budgets*.
    """
    values = _validate_sweep_values(budgets, "budgets")
    return [
        AblationConfig(kind="token_budget", label=f"token_budget={v}",
                       params={"token_budget": v})
        for v in values
    ]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_ablation(
    dataset: Union[BenchmarkDataset, Sequence[BenchmarkCase]],
    configs: Sequence[AblationConfig],
    strategy_builder: StrategyBuilder,
    answer_fn: Callable[[str, str], str],
    k: int = 5,
    ablation: str = "",
) -> AblationResult:
    """Run one ablation family: one strategy per config on the same cases.

    Delegates the actual runs to :func:`compare_strategies`, so every
    configuration gets its own ``BenchmarkRunner`` over a fresh copy of the
    case list (results isolated; datasets never mutated), and all metrics
    come from the existing ``BenchmarkReport`` — nothing is recomputed
    here. Each config's ``label`` becomes the strategy name (overriding any
    name on the builder's spec), guaranteeing deterministic unique keys.

    Args:
        dataset: ``BenchmarkDataset`` or sequence of ``BenchmarkCase``.
        configs: Validated configurations (>= 1; labels must be unique —
            enforced by the underlying comparison as well).
        strategy_builder: ``(AblationConfig) -> StrategySpec`` producing the
            strategy for one config (its spec name is overridden by the
            config label; its ``retrieval_fn``/``answer_fn``/
            ``memory_prepare_fn`` are kept — the latter lets real-dataset
            configs prepare per-case state, e.g. an isolated store).
        answer_fn: Shared ``(query, context) -> answer`` for configs whose
            spec does not carry its own.
        k: Metric cutoff for every runner (kept FIXED across configs so the
            ablation varies exactly one factor).
        ablation: Family name recorded on the result (e.g. ``"k"``).

    Returns:
        ``AblationResult`` (serialization-ready nested dataclasses).

    Raises:
        TypeError: Non-callable builder, or builder not returning a
            ``StrategySpec``; bad *configs*/*answer_fn* types.
        ValueError: Empty *configs*/*ablation*, empty/duplicate labels, or
            invalid *k* (surfaced from the comparison layer too).
    """
    if not isinstance(ablation, str) or not ablation:
        raise ValueError(f"ablation must be a non-empty string, got {ablation!r}")
    if not callable(strategy_builder):
        raise TypeError("strategy_builder must be callable")
    if isinstance(configs, (str, bytes)) or not isinstance(configs, (list, tuple)):
        raise TypeError("configs must be a sequence of AblationConfig")
    if not configs:
        raise ValueError("configs must contain at least one AblationConfig")

    specs: List[StrategySpec] = []
    seen_labels: set = set()
    for cfg in configs:
        if not isinstance(cfg, AblationConfig):
            raise TypeError(f"configs entries must be AblationConfig, got {type(cfg)!r}")
        if not isinstance(cfg.label, str) or not cfg.label:
            raise ValueError(f"config label must be a non-empty string, got {cfg.label!r}")
        if cfg.label in seen_labels:
            raise ValueError(f"duplicate config label {cfg.label!r}")
        seen_labels.add(cfg.label)

        spec = strategy_builder(cfg)
        if not isinstance(spec, StrategySpec):
            raise TypeError(
                f"strategy_builder must return StrategySpec for {cfg.label!r}, "
                f"got {type(spec)!r}"
            )
        # Label wins: deterministic, unique keying for results/serialization.
        specs.append(
            StrategySpec(
                name=cfg.label,
                retrieval_fn=spec.retrieval_fn,
                answer_fn=spec.answer_fn,
                memory_prepare_fn=spec.memory_prepare_fn,
            )
        )

    comparison = compare_strategies(dataset, specs, answer_fn, k=k)
    return AblationResult(
        ablation=ablation, configs=list(configs), comparison=comparison
    )
