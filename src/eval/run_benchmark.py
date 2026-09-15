"""Thin CLI entry point for MemAge benchmarking.

Intended eventual usage:
    python -m src.eval.run_benchmark --dataset locomo --baseline none
    python -m src.eval.run_benchmark --dataset longmemeval --baseline none
    python -m src.eval.run_benchmark --dataset synthetic --baseline none

Real dataset adapters and baseline implementations are not yet available;
this CLI provides argument parsing and clean error messages when a requested
real adapter is unavailable, while allowing synthetic in-memory benchmarks
for testing.

No network, no downloads, no ChromaDB access.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional


SUPPORTED_DATASETS = {
    "synthetic": "Synthetic in-memory cases (always available, for testing)",
    "locomo": "LoCoMo dataset (requires data/locomo and adapter — not yet available)",
    "longmemeval": "LongMemEval dataset (requires data/longmemeval and adapter — not yet available)",
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


def _dataset_available(dataset: str) -> bool:
    """Check if dataset is locally available.

    - synthetic: always True
    - locomo/longmemeval: check for data/<name> directory with files
    """
    if dataset == "synthetic":
        return True
    # Check for real dataset files (no download)
    base = Path(__file__).parent.parent.parent / "data" / dataset
    if not base.exists():
        return False
    # Consider available only if non-empty (has actual files beyond README)
    files = list(base.rglob("*"))
    # Filter to actual data files (ignore README)
    data_files = [f for f in files if f.is_file() and f.name.lower() != "readme.md"]
    return len(data_files) > 0


def _get_unavailable_message(dataset: str, baseline: str) -> Optional[str]:
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
    if not _dataset_available(dataset):
        return (
            f"Dataset '{dataset}' is not available locally. "
            f"Expected data at 'data/{dataset}/' but not found or empty. "
            f"Synthetic benchmark is available via '--dataset synthetic' for testing. "
            f"Real LoCoMo/LongMemEval adapters will be added when dataset files and "
            f"specifications are present. No download is performed automatically."
        )
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="src.eval.run_benchmark",
        description="MemAge benchmark runner (Phase 6 — thin CLI boundary).",
        epilog="Phase 6: synthetic available; real datasets/baselines plug in later.",
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
        type=int,
        default=None,
        help="Optional token budget hint (not yet used by synthetic runner).",
    )
    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)


def run_synthetic_demo(k: int = 5) -> int:
    """Run a tiny synthetic benchmark demo (no network, no store).

    Uses BenchmarkRunner with injected fake retrieval/answer functions.
    Returns exit code 0.
    """
    from src.eval.benchmark import BenchmarkRunner
    from src.eval.dataset import BenchmarkDataset

    # Synthetic cases
    dataset = BenchmarkDataset.from_synthetic(
        [
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
        ],
        name="synthetic_demo",
    )

    # Fake retrieval: returns context with gold memory text
    def fake_retrieval(query: str):
        from src.eval.benchmark import RetrievalOutcome

        if "preference" in query:
            return RetrievalOutcome(
                context="[Memory 1]\nTimestamp: 2026-01-01T00:00:00Z\nSource: s1\nContent: User prefers dark mode",
                retrieved_ids=["mem_001", "mem_002"],
            )
        else:
            return RetrievalOutcome(
                context="[Memory 1]\nTimestamp: 2026-01-01T00:00:00Z\nSource: s2\nContent: User lives in Berlin",
                retrieved_ids=["mem_002", "mem_001"],
            )

    def fake_answer(query: str, context: str) -> str:
        # Simple echo: if context contains answer, return reference-like
        if "dark mode" in context:
            return "User prefers dark mode"
        if "Berlin" in context:
            return "User lives in Berlin"
        return "Unknown"

    runner = BenchmarkRunner(retrieval_fn=fake_retrieval, answer_fn=fake_answer, k=k)
    report = runner.run_dataset(dataset)
    print(f"Synthetic benchmark ({dataset.name}): {report.num_queries} queries")
    print(f"  Mean F1: {report.mean_f1:.3f}")
    print(f"  Mean BLEU-1: {report.mean_bleu1:.3f}")
    print(f"  Mean latency: {report.mean_latency_ms:.2f} ms")
    if report.mean_tokens is not None:
        print(f"  Mean tokens: {report.mean_tokens:.1f}")
    if report.mean_recall_at_k is not None:
        print(f"  Mean Recall@{k}: {report.mean_recall_at_k:.3f}")
        print(f"  Mean Precision@{k}: {report.mean_precision_at_k:.3f}")
        print(f"  MRR: {report.mrr:.3f}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI main entry point.

    Returns:
        Exit code (0 success, 1/2 for unavailable adapter).
    """
    args = parse_args(argv)

    msg = _get_unavailable_message(args.dataset, args.baseline)
    if msg is not None:
        print(f"Error: {msg}", file=sys.stderr)
        # Distinguish not-implemented vs missing data via exit code 2 for missing data
        if "not yet implemented" in msg or "not available locally" in msg:
            return 2
        return 1

    # At this point dataset+baseline are considered available
    if args.dataset == "synthetic":
        return run_synthetic_demo(k=args.k)

    # Real dataset path (would be implemented in Phase 7 with actual adapters)
    # For Phase 6, this branch is unreachable because _dataset_available would be False
    # for locomo/longmemeval without files; kept for future.
    print(f"Running benchmark: dataset={args.dataset}, baseline={args.baseline}, k={args.k}")
    print("Note: Real dataset adapters not yet implemented in Phase 6.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
