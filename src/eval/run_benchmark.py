"""CLI entry point for MemAge benchmarking.

Usage:
    python -m src.eval.run_benchmark --dataset synthetic --token-budget 512
    python -m src.eval.run_benchmark --dataset locomo --baseline none --data-dir data
    python -m src.eval.run_benchmark --dataset longmemeval --baseline none

Dataset resolution: ``<data-dir>/<dataset>/*.json`` (default ``data-dir`` is
the repository ``data/`` folder — see ``data/README.md`` for the expected
file names). No network, no downloads, no ChromaDB access.

Baselines: ``none`` (our system), ``full_history`` (upper bound),
``memoryos`` (tiered approximation), ``memory_r1`` (budgeted approximation),
and ``hmem`` (hierarchical approximation — see src/eval/baselines.py) all run
on synthetic data today. All checklist baselines are now implemented.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

from src.eval.adapters import (
    DatasetFileNotFoundError,
    DatasetFormatError,
    load_dataset,
    resolve_dataset_file,
)
from src.eval.ablation import (
    AblationConfig,
    AblationResult,
    PassthroughReranker,
    budget_configs,
    k_configs,
    run_ablation,
    stage_configs,
)
from src.eval.baselines import (
    FullHistoryBaseline,
    HMemBaseline,
    MemoryOSBaseline,
    MemoryR1Baseline,
)
from src.eval.benchmark import BenchmarkRunner, make_retrieval_fn
from src.eval.comparison import StrategySpec, compare_strategies, format_comparison_table
from src.eval.serialization import write_csv, write_json
from src.eval.dataset import BenchmarkDataset
from src.retrieval.context import RetrievalContext
from src.retrieval.retriever import CoarseRetriever
from src.retrieval.reranker import RankedCandidate


SUPPORTED_DATASETS = {
    "synthetic": "Synthetic in-memory cases (always available, for testing)",
    "locomo": "LoCoMo dataset (requires data/locomo/locomo10.json)",
    "longmemeval": "LongMemEval dataset (requires data/longmemeval/*.json)",
}

SUPPORTED_BASELINES = {
    "none": "Our System — two-stage retrieval + QA (memory-based)",
    "full_history": "Full-history upper bound (entire history as context, no retrieval)",
    "memoryos": "MemoryOS-style tiered baseline (project-level approximation, not a reproduction)",
    "memory_r1": "Memory-R1-style budgeted baseline (project-level approximation, not a reproduction)",
    "hmem": "H-MEM-style hierarchical baseline (project-level approximation, not a reproduction)",
}

# Baselines that are not yet implemented but will be
NOT_YET_IMPLEMENTED_BASELINES: set = set()

# Default token budget forwarded to the retrieval pipeline (used when the
# --token-budget flag is omitted).
DEFAULT_TOKEN_BUDGET = 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _repo_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "data"


def _resolve_data_dir(data_dir: Optional[str]) -> Path:
    return Path(data_dir) if data_dir else _repo_data_dir()


def _dataset_available(dataset: str, data_dir: Optional[str] = None) -> bool:
    """Check whether *dataset* can be loaded from *data_dir* (no network)."""
    if dataset == "synthetic":
        return True
    try:
        resolve_dataset_file(dataset, data_dir)
    except (DatasetFileNotFoundError, ValueError):
        return False
    return True


def _get_unavailable_message(
    dataset: str, baseline: str, data_dir: Optional[str] = None
) -> Optional[str]:
    """Return error message if dataset/baseline unavailable, else None."""
    if dataset not in SUPPORTED_DATASETS:
        return f"Unknown dataset '{dataset}'. Supported: {sorted(SUPPORTED_DATASETS)}"
    if baseline not in SUPPORTED_BASELINES:
        return f"Unknown baseline '{baseline}'. Supported: {sorted(SUPPORTED_BASELINES)}"
    if baseline in NOT_YET_IMPLEMENTED_BASELINES:
        return (
            f"Baseline '{baseline}' is not yet implemented. "
            f"It will be added in Phase 7 (baselines phase). "
            f"Available now: 'none' (our system), 'full_history' (upper bound, injectable)."
        )
    if not _dataset_available(dataset, data_dir):
        expected = _resolve_data_dir(data_dir) / dataset
        return (
            f"Dataset '{dataset}' is not available locally. "
            f"Expected dataset files under '{expected}/' "
            f"(see data/README.md for download instructions; e.g. "
            f"data/locomo/locomo10.json, data/longmemeval/longmemeval_s.json). "
            f"Dataset adapters are implemented; files must be downloaded "
            f"manually - no download is performed automatically."
        )
    return None


def _positive_int(value: str) -> int:
    """argparse type: positive integer (errors exit with code 2)."""
    try:
        iv = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}")
    if iv <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {iv}")
    return iv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="src.eval.run_benchmark",
        description="MemAge benchmark runner (datasets: synthetic, locomo, longmemeval).",
        epilog="Real datasets load from <data-dir>/<dataset>/*.json; no download.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=sorted(SUPPORTED_DATASETS.keys()),
        help="Dataset to benchmark (default: synthetic).",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="none",
        choices=sorted(SUPPORTED_BASELINES.keys()),
        help="Strategy/baseline to evaluate (default: none).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Cutoff K for retrieval metrics (default: 5).",
    )
    parser.add_argument(
        "--token-budget",
        type=_positive_int,
        default=DEFAULT_TOKEN_BUDGET,
        help=(
            "Token budget passed through to the retrieval pipeline "
            f"(default: {DEFAULT_TOKEN_BUDGET})."
        ),
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help=(
            "Comma-separated baselines to compare on the same dataset, e.g. "
            "'none,full_history,memoryos,memory_r1,hmem' (synthetic only). "
            "When omitted, the single --baseline behavior is unchanged."
        ),
    )
    parser.add_argument(
        "--ablate",
        type=str,
        default=None,
        help=(
            "Run one ablation family on the synthetic dataset: 'stage' "
            "(coarse-only vs two-stage), 'k:<v,v,...>' (Stage-1 candidate "
            "counts), or 'budget:<v,v,...>' (token budgets), e.g. "
            "'--ablate k:1,3,5'. Mutually exclusive with --compare."
        ),
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Write the --compare/--ablate result as JSON to PATH "
            "(ignored when the export flags are omitted; requires "
            "--compare or --ablate)."
        ),
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Write the --compare/--ablate result as CSV to PATH (one row "
            "per strategy/configuration; requires --compare or --ablate)."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Directory containing <dataset>/ subfolders with dataset files "
            "(default: the repository data/ folder)."
        ),
    )
    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------


def _print_report(report, k: int, title: str) -> None:
    print(f"{title}: {report.num_queries} queries")
    print(f"  Mean F1: {report.mean_f1:.3f}")
    print(f"  Mean BLEU-1: {report.mean_bleu1:.3f}")
    print(f"  Mean latency: {report.mean_latency_ms:.2f} ms")
    if report.mean_tokens is not None:
        print(f"  Mean tokens: {report.mean_tokens:.1f}")
    if report.mean_recall_at_k is not None:
        print(f"  Mean Recall@{k}: {report.mean_recall_at_k:.3f}")
    if report.mean_precision_at_k is not None:
        print(f"  Mean Precision@{k}: {report.mean_precision_at_k:.3f}")
    if report.mrr is not None:
        print(f"  MRR: {report.mrr:.3f}")


# ---------------------------------------------------------------------------
# Synthetic benchmark (in-memory store + deterministic reranker, no network)
# ---------------------------------------------------------------------------

_SYNTHETIC_MEMORIES = [
    {
        "id": "mem_001",
        "text": "User prefers dark mode",
        "timestamp": "2026-01-01T00:00:00Z",
        "source": "s1",
        "embedding": None,
    },
    {
        "id": "mem_002",
        "text": "User lives in Berlin",
        "timestamp": "2026-01-01T00:00:00Z",
        "source": "s2",
        "embedding": None,
    },
]

_SYNTHETIC_CASES = [
    {
        "query": "What is the user's preference?",
        "reference_answer": "User prefers dark mode",
        "relevant_ids": ["mem_001"],
        "case_id": "syn_001",
    },
    {
        "query": "Where does the user live?",
        "reference_answer": "User lives in Berlin",
        "relevant_ids": ["mem_002"],
        "case_id": "syn_002",
    },
]


class _StaticStore:
    """In-memory store for the synthetic demo (``query`` contract only)."""

    def __init__(self, items):
        self._items = items

    def query(self, text: str, k: int, tier: str):
        return self._items[:k]


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _stem(token: str) -> str:
    """Naive singular stem (``prefers`` → ``prefer``); deterministic."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(text: str) -> set:
    return {_stem(t) for t in _TOKEN_RE.findall((text or "").lower())}


class _LexicalReranker:
    """Deterministic Stage-2 reranker for the synthetic demo (no LLM).

    Scores candidates by stemmed Jaccard overlap with the query; ties keep
    the Stage-1 order (stable sort).
    """

    def rerank(self, query: str, candidates):
        query_tokens = _tokens(query)
        ranked: List[RankedCandidate] = []
        for candidate in candidates:
            cand_tokens = _tokens(candidate.get("text", ""))
            if query_tokens and cand_tokens:
                overlap = len(query_tokens & cand_tokens)
                union = len(query_tokens | cand_tokens)
                score = overlap / union if union else 0.0
            else:
                score = 0.0
            ranked.append(RankedCandidate(candidate=candidate, score=score))
        return sorted(ranked, key=lambda r: -r.score)


def _synthetic_pipeline() -> RetrievalContext:
    """Real two-stage pipeline over the in-memory synthetic memories."""
    return RetrievalContext(
        CoarseRetriever(_StaticStore(_SYNTHETIC_MEMORIES)),
        _LexicalReranker(),
        k=len(_SYNTHETIC_MEMORIES),
        tier="all",
    )


def _synthetic_answer(query: str, context: str) -> str:
    """Deterministic shared answer function for the synthetic demo."""
    if "preference" in query:
        return "User prefers dark mode" if "dark mode" in context else "Unknown"
    if "where" in query.lower() or "live" in query:
        return "User lives in Berlin" if "Berlin" in context else "Unknown"
    return "Unknown"


def run_synthetic_demo(
    k: int = 5, token_budget: int = DEFAULT_TOKEN_BUDGET, baseline: str = "none"
) -> int:
    """Run the synthetic benchmark (no network, no store, no API keys).

    Baselines:
        * ``none`` - the real two-stage pipeline (``RetrievalContext``) over
          an in-memory store; ``token_budget`` is passed through to it via
          ``make_retrieval_fn``.
        * ``full_history`` - the entire history as context, no retrieval
          (``FullHistoryBaseline``); ``token_budget`` does not apply.
        * ``memoryos`` - tiered STM/LTM approximation (``MemoryOSBaseline``);
          selects top-``k`` entries by lexical overlap; ``token_budget`` does
          not apply.
        * ``memory_r1`` - budgeted ranked approximation (``MemoryR1Baseline``);
          ranks the whole history lexically, then enforces ``token_budget``
          via the repository's greedy budget selector.
        * ``hmem`` - hierarchical approximation (``HMemBaseline``); orders
          candidates tier-first (STM/MTM/LTM coarse-to-fine), then enforces
          ``token_budget`` via the same greedy budget selector.

    Returns exit code 0.
    """
    dataset = BenchmarkDataset.from_synthetic(_SYNTHETIC_CASES, name="synthetic_demo")

    if baseline == "full_history":
        retrieval_fn = FullHistoryBaseline(_SYNTHETIC_MEMORIES)
        title = f"Synthetic benchmark ({dataset.name}, baseline={baseline})"
    elif baseline == "memoryos":
        # ``--k`` doubles as the baseline's selection size (metric cutoff
        # stays k as well, matching the BenchmarkRunner contract).
        retrieval_fn = MemoryOSBaseline(_SYNTHETIC_MEMORIES, k=k)
        title = f"Synthetic benchmark ({dataset.name}, baseline={baseline})"
    elif baseline == "memory_r1":
        retrieval_fn = MemoryR1Baseline(
            _SYNTHETIC_MEMORIES, token_budget=token_budget
        )
        title = f"Synthetic benchmark ({dataset.name}, baseline={baseline})"
    elif baseline == "hmem":
        retrieval_fn = HMemBaseline(_SYNTHETIC_MEMORIES, token_budget=token_budget)
        title = f"Synthetic benchmark ({dataset.name}, baseline={baseline})"
    else:
        retrieval_fn = make_retrieval_fn(_synthetic_pipeline(), token_budget)
        title = f"Synthetic benchmark ({dataset.name})"

    runner = BenchmarkRunner(
        retrieval_fn=retrieval_fn, answer_fn=_synthetic_answer, k=k
    )
    report = runner.run_dataset(dataset)
    _print_report(report, k, title)
    return 0


# ---------------------------------------------------------------------------
# Comparison mode (orchestration lives in src/eval/comparison.py)
# ---------------------------------------------------------------------------


def _export_result(result, export_json: Optional[str], export_csv: Optional[str]) -> Optional[int]:
    """Write the requested exports for *result* after a run.

    Returns ``None`` on success (or when no export was requested), or an
    exit code (2) if writing failed — the error is printed to stderr.
    """
    if not export_json and not export_csv:
        return None
    try:
        if export_json:
            write_json(result, export_json)
            print(f"Exported JSON: {export_json}")
        if export_csv:
            write_csv(result, export_csv)
            print(f"Exported CSV: {export_csv}")
    except (OSError, ValueError, TypeError) as exc:
        print(f"Error: could not export result: {exc}", file=sys.stderr)
        return 2
    return None


def build_synthetic_strategies(
    names: List[str], token_budget: int = DEFAULT_TOKEN_BUDGET, k: int = 5
) -> List[StrategySpec]:
    """Build one isolated ``StrategySpec`` per baseline name for synthetic data.

    Each name gets its OWN strategy instance (no sharing between specs).
    ``none`` is the real two-stage pipeline via ``make_retrieval_fn`` with
    the CLI ``token_budget``; the four baselines reuse ``_SYNTHETIC_MEMORIES``.

    Raises:
        ValueError: If any *name* is not a supported baseline.
    """
    unknown = [n for n in names if n not in SUPPORTED_BASELINES]
    if unknown:
        raise ValueError(
            f"unknown baseline(s): {unknown}. Supported: {sorted(SUPPORTED_BASELINES)}"
        )
    builders = {
        "none": lambda: make_retrieval_fn(_synthetic_pipeline(), token_budget),
        "full_history": lambda: FullHistoryBaseline(_SYNTHETIC_MEMORIES),
        "memoryos": lambda: MemoryOSBaseline(_SYNTHETIC_MEMORIES, k=k),
        "memory_r1": lambda: MemoryR1Baseline(
            _SYNTHETIC_MEMORIES, token_budget=token_budget
        ),
        "hmem": lambda: HMemBaseline(_SYNTHETIC_MEMORIES, token_budget=token_budget),
    }
    return [StrategySpec(name=n, retrieval_fn=builders[n]()) for n in names]


def run_synthetic_comparison(
    names: List[str],
    k: int = 5,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    export_json: Optional[str] = None,
    export_csv: Optional[str] = None,
) -> int:
    """Run a comparison of *names* on the synthetic dataset and print a table.

    Optional *export_json*/*export_csv* paths serialize the
    ``ComparisonResult`` (see ``src/eval/serialization.py``).

    Returns exit code 0 (2 on validation/export failure, before or after the
    run respectively).
    """
    try:
        specs = build_synthetic_strategies(names, token_budget=token_budget, k=k)
        result = compare_strategies(
            BenchmarkDataset.from_synthetic(_SYNTHETIC_CASES, name="synthetic_demo"),
            specs,
            _synthetic_answer,
            k=k,
        )
    except (ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Comparison ({result.dataset_name}, k={result.k}): "
        f"{result.num_cases} cases, {len(result.results)} strategies"
    )
    print(format_comparison_table(result))
    export_code = _export_result(result, export_json, export_csv)
    if export_code is not None:
        return export_code
    return 0


# ---------------------------------------------------------------------------
# Ablation mode (infrastructure in src/eval/ablation.py)
# ---------------------------------------------------------------------------


def build_synthetic_ablation_spec(
    cfg: AblationConfig, token_budget: int = DEFAULT_TOKEN_BUDGET
) -> StrategySpec:
    """Map one ablation config onto a synthetic two-stage strategy spec.

    Config params honored (defaults = the current synthetic baseline setup):
        * ``stage``: ``"two_stage"`` (default) uses the demo lexical reranker;
          ``"coarse_only"`` bypasses Stage-2 via ``PassthroughReranker``.
        * ``k``: Stage-1 candidate count (default: all synthetic memories).
        * ``token_budget``: budget forwarded to ``make_retrieval_fn``
          (default: the CLI/config budget).

    Raises:
        ValueError: Unknown ``stage`` value.
    """
    params = cfg.params
    stage = params.get("stage", "two_stage")
    if stage not in ("coarse_only", "two_stage"):
        raise ValueError(
            f"unknown stage {stage!r} (expected coarse_only | two_stage)"
        )
    pipe_k = params.get("k", len(_SYNTHETIC_MEMORIES))
    budget = params.get("token_budget", token_budget)
    reranker = _LexicalReranker() if stage == "two_stage" else PassthroughReranker()
    ctx = RetrievalContext(
        CoarseRetriever(_StaticStore(_SYNTHETIC_MEMORIES)),
        reranker,
        k=pipe_k,
        tier="all",
    )
    return StrategySpec(name=cfg.label, retrieval_fn=make_retrieval_fn(ctx, budget))


def run_synthetic_ablation(
    configs: List[AblationConfig],
    ablation: str,
    k: int = 5,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    export_json: Optional[str] = None,
    export_csv: Optional[str] = None,
) -> int:
    """Run one ablation family on the synthetic dataset and print a table.

    Optional *export_json*/*export_csv* paths serialize the
    ``AblationResult`` (see ``src/eval/serialization.py``).

    Returns exit code 0 (2 on validation/export failure, before or after the
    run respectively).
    """
    try:
        result = run_ablation(
            BenchmarkDataset.from_synthetic(_SYNTHETIC_CASES, name="synthetic_demo"),
            configs,
            lambda cfg: build_synthetic_ablation_spec(cfg, token_budget=token_budget),
            _synthetic_answer,
            k=k,
            ablation=ablation,
        )
    except (ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    comp = result.comparison
    print(
        f"Ablation {result.ablation} ({comp.dataset_name}, metric_k={comp.k}): "
        f"{comp.num_cases} cases, {len(result.configs)} configurations"
    )
    print(format_comparison_table(comp))
    export_code = _export_result(result, export_json, export_csv)
    if export_code is not None:
        return export_code
    return 0


# ---------------------------------------------------------------------------
# Real datasets (LoCoMo / LongMemEval)
# ---------------------------------------------------------------------------


def _build_strategy(baseline: str, token_budget: int, data_dir: Optional[str]):
    """Build ``(retrieval_fn, answer_fn)`` for a real dataset — or ``None``.

    Once ``src/store`` (tiered memory store) and ``src/llm`` (``call_llm``)
    are available, this constructs the real strategy:

        ctx = RetrievalContext(CoarseRetriever(store), LLMReranker(call_llm), ...)
        retrieval_fn = make_retrieval_fn(ctx, token_budget)   # budget passes here
        answer_fn = lambda q, context: call_llm(...)

    For ``full_history``, ``memoryos``, ``memory_r1``, and ``hmem`` the
    retrieval side needs no store — the context can be built per case from
    adapter metadata via the baselines in ``src/eval/baselines.py`` — but
    answer generation still requires ``src/llm``.

    Neither module exists yet (owned by teammates), so no strategy can be
    constructed without inventing interfaces. The dataset loading path above
    is fully exercised regardless.
    """
    return None


def run_dataset_benchmark(
    dataset: str,
    baseline: str,
    k: int,
    token_budget: int,
    data_dir: Optional[str] = None,
) -> int:
    """Load a real dataset, report it, and benchmark if a strategy exists.

    Returns:
        0 on success, 1 on dataset format errors, 2 when the strategy
        (retrieval store / LLM) is not yet available.
    """
    try:
        loaded = load_dataset(dataset, data_dir=data_dir)
    except DatasetFileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except DatasetFormatError as exc:
        print(f"Error: dataset '{dataset}' could not be parsed: {exc}", file=sys.stderr)
        return 1

    path = resolve_dataset_file(dataset, data_dir)
    print(f"Loaded {dataset}: {len(loaded)} cases from {path}")

    strategy = _build_strategy(baseline, token_budget, data_dir)
    if strategy is None:
        if baseline == "full_history":
            reason = (
                "the full-history context can be built from dataset metadata "
                "(src/eval/baselines.py) but answer generation still requires "
                "src/llm (call_llm), which is not implemented yet"
            )
        elif baseline == "memoryos":
            reason = (
                "the tiered-memory context can be built from dataset metadata "
                "(src/eval/baselines.py) but answer generation still requires "
                "src/llm (call_llm), which is not implemented yet"
            )
        elif baseline == "memory_r1":
            reason = (
                "the budgeted-ranked context can be built from dataset metadata "
                "(src/eval/baselines.py) but answer generation still requires "
                "src/llm (call_llm), which is not implemented yet"
            )
        elif baseline == "hmem":
            reason = (
                "the hierarchical context can be built from dataset metadata "
                "(src/eval/baselines.py) but answer generation still requires "
                "src/llm (call_llm), which is not implemented yet"
            )
        else:
            reason = (
                "building the retrieval/answer strategy requires src/store "
                "(tiered memory store) and src/llm (call_llm), which are not "
                "implemented yet"
            )
        print(
            f"Error: cannot execute baseline '{baseline}' yet - {reason}. "
            f"Dataset parsing succeeded ({len(loaded)} cases, "
            f"token_budget={token_budget}).",
            file=sys.stderr,
        )
        return 2

    retrieval_fn, answer_fn = strategy
    runner = BenchmarkRunner(retrieval_fn=retrieval_fn, answer_fn=answer_fn, k=k)
    report = runner.run_dataset(loaded)
    _print_report(report, k, f"Benchmark ({dataset}, baseline={baseline})")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """CLI main entry point.

    Returns:
        Exit code (0 success, 1 format/usage error, 2 unavailable).
    """
    args = parse_args(argv)

    # --compare and --ablate are mutually exclusive modes.
    if args.compare is not None and args.ablate is not None:
        print("Error: use only one of --compare / --ablate", file=sys.stderr)
        return 1

    # Export flags apply only to the structured-result modes.
    if (args.export_json or args.export_csv) and (
        args.compare is None and args.ablate is None
    ):
        print(
            "Error: --export-json/--export-csv require --compare or --ablate",
            file=sys.stderr,
        )
        return 1

    # Comparison mode (orthogonal to --baseline; validated here, runs on
    # synthetic only until real-dataset strategies exist).
    if args.compare is not None:
        names = [n.strip() for n in args.compare.split(",") if n.strip()]
        unknown = [n for n in names if n not in SUPPORTED_BASELINES]
        if not names:
            print(
                "Error: --compare requires at least one baseline name, e.g. "
                "--compare none,full_history",
                file=sys.stderr,
            )
            return 1
        if unknown:
            print(
                f"Error: unknown baseline(s) in --compare: {unknown}. "
                f"Supported: {sorted(SUPPORTED_BASELINES)}",
                file=sys.stderr,
            )
            return 1
        if len(set(names)) != len(names):
            print("Error: duplicate baseline names in --compare", file=sys.stderr)
            return 1
        if args.dataset != "synthetic":
            print(
                "Error: --compare currently requires --dataset synthetic "
                "(strategy construction for real datasets awaits src/llm).",
                file=sys.stderr,
            )
            return 2
        return run_synthetic_comparison(
            names,
            k=args.k,
            token_budget=args.token_budget,
            export_json=args.export_json,
            export_csv=args.export_csv,
        )

    # Ablation mode (also synthetic-only; mutually exclusive with --compare).
    if args.ablate is not None:
        kind, _, raw = args.ablate.partition(":")
        kind, raw = kind.strip(), raw.strip()
        if kind == "stage":
            if raw:
                print(
                    "Error: --ablate stage takes no values "
                    "(stage=coarse_only,stage=two_stage are fixed)",
                    file=sys.stderr,
                )
                return 1
            configs, ablation = stage_configs(), "retrieval_stage"
        elif kind in ("k", "budget"):
            if not raw:
                print(
                    f"Error: --ablate {kind} requires values, e.g. "
                    f"--ablate {kind}:1,3,5",
                    file=sys.stderr,
                )
                return 1
            values: List[int] = []
            for part in raw.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    values.append(_positive_int(part))
                except argparse.ArgumentTypeError as exc:
                    print(f"Error: invalid value {part!r}: {exc}", file=sys.stderr)
                    return 1
            if not values:
                print(f"Error: --ablate {kind} requires values", file=sys.stderr)
                return 1
            try:
                configs = k_configs(values) if kind == "k" else budget_configs(values)
            except (TypeError, ValueError) as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
            ablation = "k" if kind == "k" else "token_budget"
        else:
            print(
                f"Error: unknown ablation kind {kind!r}. "
                f"Supported: stage, k, budget",
                file=sys.stderr,
            )
            return 1
        if args.dataset != "synthetic":
            print(
                "Error: --ablate currently requires --dataset synthetic "
                "(strategy construction for real datasets awaits src/llm).",
                file=sys.stderr,
            )
            return 2
        return run_synthetic_ablation(
            configs,
            ablation=ablation,
            k=args.k,
            token_budget=args.token_budget,
            export_json=args.export_json,
            export_csv=args.export_csv,
        )

    msg = _get_unavailable_message(args.dataset, args.baseline, args.data_dir)
    if msg is not None:
        print(f"Error: {msg}", file=sys.stderr)
        # Distinguish not-implemented vs missing data via exit code 2
        if "not yet implemented" in msg or "not available locally" in msg:
            return 2
        return 1

    if args.dataset == "synthetic":
        return run_synthetic_demo(
            k=args.k, token_budget=args.token_budget, baseline=args.baseline
        )

    return run_dataset_benchmark(
        dataset=args.dataset,
        baseline=args.baseline,
        k=args.k,
        token_budget=args.token_budget,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
