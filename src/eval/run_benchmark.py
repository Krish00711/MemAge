"""CLI entry point for MemAge benchmarking.

Usage:
    python -m src.eval.run_benchmark --dataset synthetic --token-budget 512
    python -m src.eval.run_benchmark --dataset locomo --baseline none --data-dir data
    python -m src.eval.run_benchmark --dataset longmemeval --baseline none

Dataset resolution: ``<data-dir>/<dataset>/*.json`` (default ``data-dir`` is
the repository ``data/`` folder — see ``data/README.md`` for the expected
file names). No network, no downloads, no ChromaDB access.

Baselines MemoryOS / Memory-R1 / H-MEM are Phase 7 and not implemented here.
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
from src.eval.benchmark import BenchmarkRunner, make_retrieval_fn
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
    "full_history": "Full-history upper bound (injectable strategy, no store)",
    # Future baselines — not yet implemented
    "memoryos": "MemoryOS baseline (Phase 7)",
    "memory_r1": "Memory-R1 baseline (Phase 7)",
    "hmem": "H-MEM baseline (Phase 7)",
}

# Baselines that are not yet implemented but will be
NOT_YET_IMPLEMENTED_BASELINES = {"memoryos", "memory_r1", "hmem"}

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


def run_synthetic_demo(
    k: int = 5, token_budget: int = DEFAULT_TOKEN_BUDGET
) -> int:
    """Run the synthetic benchmark (no network, no store, no API keys).

    The retrieval step is the real pipeline (``RetrievalContext``) over an
    in-memory store; ``token_budget`` is passed through to it via
    ``make_retrieval_fn``. Returns exit code 0.
    """
    dataset = BenchmarkDataset.from_synthetic(_SYNTHETIC_CASES, name="synthetic_demo")

    retrieval_fn = make_retrieval_fn(_synthetic_pipeline(), token_budget)

    def fake_answer(query: str, context: str) -> str:
        if "preference" in query:
            return "User prefers dark mode" if "dark mode" in context else "Unknown"
        if "where" in query.lower() or "live" in query:
            return "User lives in Berlin" if "Berlin" in context else "Unknown"
        return "Unknown"

    runner = BenchmarkRunner(retrieval_fn=retrieval_fn, answer_fn=fake_answer, k=k)
    report = runner.run_dataset(dataset)
    _print_report(report, k, f"Synthetic benchmark ({dataset.name})")
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
        print(
            f"Error: cannot execute baseline '{baseline}' yet - building the "
            "retrieval/answer strategy requires src/store (tiered memory "
            "store) and src/llm (call_llm), which are not implemented yet. "
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

    msg = _get_unavailable_message(args.dataset, args.baseline, args.data_dir)
    if msg is not None:
        print(f"Error: {msg}", file=sys.stderr)
        # Distinguish not-implemented vs missing data via exit code 2
        if "not yet implemented" in msg or "not available locally" in msg:
            return 2
        return 1

    if args.dataset == "synthetic":
        return run_synthetic_demo(k=args.k, token_budget=args.token_budget)

    return run_dataset_benchmark(
        dataset=args.dataset,
        baseline=args.baseline,
        k=args.k,
        token_budget=args.token_budget,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
