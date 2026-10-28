"""CLI entry point for MemAge benchmarking.

Usage:
    python -m src.eval.run_benchmark --dataset synthetic --token-budget 512
    python -m src.eval.run_benchmark --dataset locomo --baseline none --data-dir data
    python -m src.eval.run_benchmark --dataset locomo --compare none,full_history,memoryos,memory_r1,hmem --data-dir data
    python -m src.eval.run_benchmark --dataset longmemeval --ablate stage --data-dir data

Dataset resolution: ``<data-dir>/<dataset>/*.json`` (default ``data-dir`` is
the repository ``data/`` folder — see ``data/README.md`` for the expected
file names). No network, no downloads, no ChromaDB access.

Baselines: ``none`` (our system), ``full_history`` (upper bound),
``memoryos`` (tiered approximation), ``memory_r1`` (budgeted approximation),
and ``hmem`` (hierarchical approximation — see src/eval/baselines.py).  All
five run on the synthetic demo; on real datasets (LoCoMo / LongMemEval)
``_build_strategy`` builds every one of them — ``none`` via a per-case
TieredMemoryStore and the real two-stage retrieval pipeline.

``--compare`` and ``--ablate`` run on BOTH the synthetic demo and real
datasets: they reuse the same ``StrategySpec``/``compare_strategies``/
``run_ablation``/``BenchmarkRunner`` abstractions, feeding them real-dataset
strategies built by the same H-4 builders (per-case isolated stores,
dataset memory IDs preserved).
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
from src.eval.signal_ablation import signal_configs, SIGNAL_NAMES
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

# Default Stage-1 shortlist size for the real-dataset ``none`` pipeline:
# the coarse retriever scores every memory of the case, but only the top
# min(item count, DEFAULT_STAGE1_K) candidates are handed to the LLM
# reranker.  Unbounded shortlists (one per item) produced rerank prompts
# that exceeded provider request/TPM limits on the real datasets (Groq
# 413) and silently degraded Stage-2 to the lexical fallback; 30 keeps the
# bounded prompt (see reranker text caps) small enough that the model's
# full ranking response also fits the provider's per-minute token budget.
DEFAULT_STAGE1_K = 30


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
            "'none,full_history,memoryos,memory_r1,hmem' (works on synthetic "
            "and on real datasets; real datasets need --data-dir files). "
            "When omitted, the single --baseline behavior is unchanged."
        ),
    )
    parser.add_argument(
        "--ablate",
        type=str,
        default=None,
        help=(
            "Run one ablation family on the selected dataset: 'stage' "
            "(coarse-only vs two-stage), 'k:<v,v,...>' (Stage-1 candidate "
            "counts), 'budget:<v,v,...>' (token budgets), or "
            "'signal:<name>' (controller thresholds; choices: importance, "
            "temporal, contradiction, storage), e.g. '--ablate k:1,3,5'. "
            "Mutually exclusive with --compare."
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
    parser.add_argument(
        "--sample",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Limit evaluation to the first N cases (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Random seed for reproducibility (recorded in metadata; no shuffling).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Write per-example predictions (predictions.json) and aggregate "
            "results (results.json, results.csv) to this directory. "
            "Directory is created if it does not exist."
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
    k: int = 5,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    baseline: str = "none",
    sample: Optional[int] = None,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> int:
    """Run the synthetic benchmark (no network, no store, no API keys).

    Returns exit code 0.
    """
    dataset = BenchmarkDataset.from_synthetic(_SYNTHETIC_CASES, name="synthetic_demo")
    if sample is not None:
        dataset = BenchmarkDataset(cases=list(dataset.cases)[:sample], name=dataset.name)

    if baseline == "full_history":
        retrieval_fn = FullHistoryBaseline(_SYNTHETIC_MEMORIES)
    elif baseline == "memoryos":
        retrieval_fn = MemoryOSBaseline(_SYNTHETIC_MEMORIES, k=k)
    elif baseline == "memory_r1":
        retrieval_fn = MemoryR1Baseline(_SYNTHETIC_MEMORIES, token_budget=token_budget)
    elif baseline == "hmem":
        retrieval_fn = HMemBaseline(_SYNTHETIC_MEMORIES, token_budget=token_budget)
    else:
        retrieval_fn = make_retrieval_fn(_synthetic_pipeline(), token_budget)

    runner = BenchmarkRunner(retrieval_fn=retrieval_fn, answer_fn=_synthetic_answer, k=k)
    report = runner.run_dataset(dataset)
    title = f"Synthetic benchmark ({dataset.name}, baseline={baseline})"
    _print_report(report, k, title)
    _save_run(report, output_dir=output_dir, dataset="synthetic", baseline=baseline,
              token_budget=token_budget, k=k, sample=sample, seed=seed)
    return 0


# ---------------------------------------------------------------------------
# Comparison mode (orchestration lives in src/eval/comparison.py)
# ---------------------------------------------------------------------------


def _save_run(
    report,
    output_dir: Optional[str],
    dataset: str,
    baseline: str,
    token_budget: int,
    k: int,
    sample: Optional[int] = None,
    seed: Optional[int] = None,
) -> None:
    """Write per-example predictions and aggregate results to *output_dir*."""
    if output_dir is None:
        return
    import json as _json
    import time as _time
    from dataclasses import asdict
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = {
        "dataset": dataset,
        "baseline": baseline,
        "token_budget": token_budget,
        "k": k,
        "sample": sample,
        "seed": seed,
        "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "llm_provider": __import__("os").environ.get("LLM_PROVIDER", ""),
    }

    # Per-example predictions
    predictions = []
    for r in report.per_case:
        predictions.append({
            "case_id": r.case_id,
            "query": r.query,
            "predicted_answer": r.predicted_answer,
            "reference_answer": r.reference_answer,
            "f1": r.f1,
            "bleu1": r.bleu1,
            "latency_ms": r.latency_ms,
            "tokens": r.tokens,
            "retrieval_recall": r.retrieval_recall,
            "retrieval_precision": r.retrieval_precision,
            "retrieval_rr": r.retrieval_rr,
            "retrieved_ids": r.retrieved_ids,
            "relevant_ids": r.relevant_ids,
        })
    pred_path = out / "predictions.json"
    pred_path.write_text(
        _json.dumps({"meta": meta, "predictions": predictions}, indent=2),
        encoding="utf-8",
    )

    # Aggregate results
    agg = {
        "meta": meta,
        "results": {
            "num_queries": report.num_queries,
            "mean_f1": report.mean_f1,
            "mean_bleu1": report.mean_bleu1,
            "mean_latency_ms": report.mean_latency_ms,
            "mean_tokens": report.mean_tokens,
            "mean_recall_at_k": report.mean_recall_at_k,
            "mean_precision_at_k": report.mean_precision_at_k,
            "mrr": report.mrr,
        },
    }
    results_path = out / "results.json"
    results_path.write_text(_json.dumps(agg, indent=2), encoding="utf-8")
    print(f"Saved predictions -> {pred_path}")
    print(f"Saved results     -> {results_path}")


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


def _deterministic_signal_llm(prompt: str, system: str | None = None, **kwargs) -> str:
    """No-network scoring stand-in for the synthetic demo's controller.

    Returns JSON the controller's LLM scorers parse
    (``src/controller/scoring.py``):

    * importance ``0.5`` — mid band: kept by the permissive configs
      (``0.0``/``0.3``) and suppressed by the strict ``0.6`` config.
    * contradiction ``0.55`` — between the loose/default update thresholds
      (``0.3``/``0.5``) and the strict ``0.7`` one, so UPDATE vs ADD can
      actually diverge.

    Only the controller scoring path ever sees this callable: the synthetic
    demo answers via ``_synthetic_answer`` and reranks via
    ``_LexicalReranker`` (both offline by design).
    """
    if system and "contradiction detector" in system:
        return '{"contradiction": 0.55}'
    return '{"importance": 0.5}'


def _gate_synthetic_memories(memories, thresholds) -> List[dict]:
    """Apply the REAL controller decision policy to the demo memories.

    Runs ``src.controller.controller.decide`` (real scoring signals + real
    joint policy with *thresholds*) per memory and applies the resulting
    ADD/UPDATE/DELETE/NOOP decision to a plain list — the synthetic demo
    has no embedding store (no network / no dependencies allowed), so the
    demo list stands in for the store.  Thresholds therefore genuinely
    change which demo memories survive ingestion (the historical no-op
    behavior — thresholds recorded but never applied — is gone).

    Never mutates the input items (copies are stored), so repeated runs in
    one process see the original ``_SYNTHETIC_MEMORIES``.
    """
    from src.controller.controller import decide as controller_decide

    kept: List[dict] = []
    for item in memories:
        existing = kept[:5]  # stand-in for store.query(text, k=5, "all")
        decision = controller_decide(
            item,
            existing,
            llm=_deterministic_signal_llm,
            thresholds=thresholds,
            counts={"STM": len(kept), "MTM": 0, "LTM": 0},
        )
        action = decision.get("action")
        target = decision.get("target_id")
        if action == "ADD":
            kept.append(dict(item))
        elif action == "UPDATE" and target:
            for slot in kept:
                if slot.get("id") == target:
                    slot["text"] = item.get("text", "")
                    break
        elif action == "DELETE" and target:
            kept = [s for s in kept if s.get("id") != target]
            kept.append(dict(item))
        # NOOP: the policy decided not to store this memory.
    return kept


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
        * ``thresholds`` (signal families): controller-gate the demo
          memories through the REAL ``decide`` policy before the pipeline
          sees them (see :func:`_gate_synthetic_memories`) — this is what
          makes ``--ablate signal:...`` functional instead of a no-op.

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
    thresholds = params.get("thresholds")
    memories = _SYNTHETIC_MEMORIES
    if thresholds is not None:
        memories = _gate_synthetic_memories(_SYNTHETIC_MEMORIES, thresholds)
    reranker = _LexicalReranker() if stage == "two_stage" else PassthroughReranker()
    ctx = RetrievalContext(
        CoarseRetriever(_StaticStore(memories)),
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


def _embedder_available() -> bool:
    """True when the store's default embedder chain can resolve.

    Mirrors the fallback order in ``TieredMemoryStore.__init__``
    (src/store/memory_store.py): ``sentence-transformers`` first, then
    ``fastembed`` via ``FastEmbedAdapter``.  An injected embedder bypasses
    this check (the caller only invokes this when no embedder was given).
    """
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import fastembed  # noqa: F401
        return True
    except ImportError:
        return False


def _store_items_from_entries(entries) -> List[dict]:
    """Turn case history entries into store-ready ``CandidateMemoryItem`` dicts.

    One item per *dataset memory id* — the unit ``case.relevant_ids`` refers
    to, so retrieved ids stay Recall@K / P@K / MRR compatible:

    * LoCoMo: ``dia_id`` per turn (unique) -> one item per turn.
    * LongMemEval: ``session_id``; all turns of one session are joined into
      a single item (the store's primary key would otherwise collide, and
      the relevance unit *is* the session).

    Text/timestamp/source use the same precedence as the baselines'
    ``_normalize_entry`` (``text`` else ``content``; ``source`` else ``id``)
    so ``none`` sees the same memory content as the other strategies for the
    same history.  Original entry dicts are never mutated; entries without an
    id or without any usable text are skipped.

    Returns ``[]`` for empty/unusable input.
    """
    items: List[dict] = []
    index: dict = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        eid = str(entry.get("id") or "").strip()
        if not eid:
            continue
        text = entry.get("text")
        if text is None or text == "":
            text = entry.get("content", "")
        text = str(text).strip()
        timestamp = str(entry.get("timestamp") or entry.get("date") or "")
        source = str(entry.get("source") or eid)
        pos = index.get(eid)
        if pos is None:
            index[eid] = len(items)
            items.append({
                "id": eid,
                "text": text,
                "timestamp": timestamp,
                "source": source,
                "embedding": None,  # filled in by the store
            })
        elif text:
            prev = items[pos]
            prev["text"] = (
                f"{prev['text']}\n{text}" if prev["text"] else text
            )
    return [item for item in items if item["text"]]


def _build_none_strategy(token_budget: int, call_llm, embedder, answer_fn,
                         stage: str = "two_stage",
                         pipe_k: Optional[int] = None,
                         thresholds: Optional[dict] = None):
    """Build the real-data ``none`` pipeline — our system.

    Per-case execution flow (every case gets an ISOLATED memory state):

        case -> memory_entries_from_case -> _store_items_from_entries
            -> fresh TieredMemoryStore (dataset memory ids preserved)
            -> RetrievalContext(CoarseRetriever -> reranker
               -> token-budget selection -> format_context)
            -> make_retrieval_fn(ctx, token_budget)

    Semantics of ``none``: **no selective-memory controller filtering** —
    every memory of the case is added to the store as-is (no extraction LLM,
    no controller ADD/UPDATE/DELETE decisions), and answering is plain
    two-stage retrieval + ``call_llm`` (provider-agnostic, never hard-codes
    a provider).  Stage-1 coarse-scores every memory of the case and
    shortlists the top ``min(item count, DEFAULT_STAGE1_K)`` for the LLM
    reranker (an unbounded shortlist overflows provider prompt limits on
    the real datasets).

    This builder doubles as the real-dataset **ablation** strategy builder
    (same components, one factor varied per config):

        stage      ``"two_stage"`` (default) uses ``LLMReranker(call_llm)``;
                   ``"coarse_only"`` bypasses Stage-2 with the existing
                   ``PassthroughReranker`` (candidates pass through in
                   Stage-1 order).
        pipe_k     Stage-1 candidate count for the retrieval context
                   (``None`` = ``min(item count, DEFAULT_STAGE1_K)``).  The
                   metric cutoff ``k`` used by ``BenchmarkRunner`` is NOT
                   touched here — it is fixed by the caller for fair
                   ablations.
        thresholds When not ``None`` (**signal ablation scope**), each case
                   memory is ingested through the real selective-memory
                   controller — ``src.controller.controller.process(candidate,
                   store, llm=call_llm, thresholds=...)`` — instead of a
                   blind ``store.add``.  Thresholds therefore genuinely
                   change what is stored (ADD/UPDATE/DELETE/NOOP) and, in
                   turn, retrieval/QA metrics.  When ``None`` (the ``none``
                   baseline and stage/k/budget ablations) the controller is
                   NOT invoked at all — the H-4 baseline semantics are
                   unchanged.

    Isolation: the state dict is replaced wholesale by each
    ``memory_prepare``; with chromadb installed the store persists into a
    unique temp directory per case (the previous case's directory is
    removed), and without chromadb the store already uses per-instance
    in-memory collections.

    Returns ``(retrieval_fn, answer_fn, memory_prepare)`` — both callables
    carry a diagnostic ``.state`` dict (``store`` / ``ctx`` /
    ``retrieval_fn`` / ``persist_dir`` / ``thresholds`` / ``error``) — or
    ``None`` when the default embedder chain is unavailable and none was
    injected (reported on stderr rather than silently degrading).
    """
    import shutil
    import tempfile

    from src.eval.baselines import memory_entries_from_case
    from src.eval.benchmark import RetrievalOutcome, make_retrieval_fn
    from src.retrieval.reranker import LLMReranker

    if stage not in ("two_stage", "coarse_only"):
        raise ValueError(
            f"unknown stage {stage!r} (expected coarse_only | two_stage)"
        )
    if thresholds is not None and not isinstance(thresholds, dict):
        raise ValueError(f"thresholds must be a dict or None, got {type(thresholds)!r}")

    controller_process = None
    if thresholds is not None:
        # Signal ablation only: real controller ingestion (scoring + policy).
        from src.controller.controller import process as controller_process

    if embedder is None and not _embedder_available():
        print(
            "Error: baseline 'none' needs a memory-store embedder but neither "
            "sentence-transformers nor fastembed is installed (see "
            "src/store/memory_store.py), and none was injected.",
            file=sys.stderr,
        )
        return None

    # With chromadb the store persists to disk (unique dir per case keeps
    # cases disjoint); without it TieredMemoryStore uses its own per-instance
    # in-memory collections and persist_dir is irrelevant.
    try:
        import chromadb  # noqa: F401
        chroma_available = True
    except ImportError:
        chroma_available = False

    # One shared state, replaced wholesale by every memory_prepare call, so
    # memories can never leak from one benchmark case into the next.
    state: dict = {
        "retrieval_fn": None,
        "store": None,
        "ctx": None,
        "persist_dir": None,
        "thresholds": thresholds,
        "error": None,
    }

    def memory_prepare(case: BenchmarkCase) -> None:
        """Give *case* its own store: fresh TieredMemoryStore + pipeline."""
        from src.store.memory_store import TieredMemoryStore

        prev_dir = state.get("persist_dir")
        # Clear first: a failed prepare must never serve the previous case.
        state.update(retrieval_fn=None, store=None, ctx=None, error=None)
        if prev_dir:
            shutil.rmtree(prev_dir, ignore_errors=True)
            state["persist_dir"] = None
        try:
            items = _store_items_from_entries(memory_entries_from_case(case))
            kwargs = {"embedder": embedder}
            if chroma_available:
                persist_dir = tempfile.mkdtemp(prefix="memage_none_")
                state["persist_dir"] = persist_dir
                kwargs["persist_dir"] = persist_dir
            store = TieredMemoryStore(**kwargs)
            if controller_process is None:
                # none baseline / stage+k+budget ablations: store as-is.
                for item in items:
                    store.add(item)  # store fills the embedding, rebalances tiers
            else:
                # Signal ablation: controller decides per memory; the applied
                # thresholds decide what is stored (genuinely functional).
                for item in items:
                    controller_process(
                        item, store, llm=call_llm, thresholds=thresholds
                    )
            reranker = (
                LLMReranker(call_llm) if stage == "two_stage"
                else PassthroughReranker()
            )
            ctx = RetrievalContext(
                retriever=CoarseRetriever(store),
                reranker=reranker,
                k=(
                    pipe_k
                    if pipe_k is not None
                    else min(max(len(items), 1), DEFAULT_STAGE1_K)
                ),
                tier="all",
            )
        except Exception as exc:
            state["error"] = str(exc)
            print(
                f"Error: could not prepare memories for case "
                f"{getattr(case, 'case_id', '?')}: {exc}",
                file=sys.stderr,
            )
            return
        state["store"] = store
        state["ctx"] = ctx
        state["retrieval_fn"] = make_retrieval_fn(ctx, token_budget)

    def retrieval_fn(query: str) -> RetrievalOutcome:
        fn = state.get("retrieval_fn")
        if fn is None:
            return RetrievalOutcome(context="", retrieved_ids=None, tokens=0)
        return fn(query)

    retrieval_fn.state = state
    memory_prepare.state = state
    return retrieval_fn, answer_fn, memory_prepare


# Provider request-size guard for answer prompts (same class of fix as
# ``_PROMPT_TEXT_MAX_*`` in src/retrieval/reranker.py): Groq rejects any
# request that would push the rolling tokens-per-minute window past its
# limit (observed 413 "Limit 8000" = sibling calls in the same minute +
# this request), and ``full_history`` contexts on the real datasets are
# far larger (LoCoMo ~15k, LongMemEval ~84k whitespace tokens).  Caps are
# sized from an observed 3.3 chars/token for the real LoCoMo history:
# 16000 chars ≈ 4800 tokens + system/question ≈ 4900, leaving ~3000 TPM
# window headroom for sibling strategy calls (rerank/answer) in the same
# minute.  Budgeted strategies (budget <= ~4096 whitespace tokens) stay
# under the char cap in practice, so their behavior is unchanged.
_ANSWER_CONTEXT_MAX_WORDS = 3000
_ANSWER_CONTEXT_MAX_CHARS = 16000


def _truncate_answer_context(context: str) -> str:
    """Head-truncate *context* to the answer prompt size caps (chronological
    start kept, matching ``_truncate_for_prompt`` in the reranker)."""
    if not context:
        return context
    words = context.split()
    if len(words) > _ANSWER_CONTEXT_MAX_WORDS:
        context = " ".join(words[:_ANSWER_CONTEXT_MAX_WORDS])
    if len(context) > _ANSWER_CONTEXT_MAX_CHARS:
        context = context[:_ANSWER_CONTEXT_MAX_CHARS]
    return context


def _build_answer_fn(call_llm):
    """Shared real-dataset answer function for every strategy/config.

    One ``(query, context) -> answer`` for the whole comparison/ablation so
    all strategies answer identically (fairness: only retrieval differs).
    Provider-agnostic — uses the injected/callable ``call_llm`` only.
    Contexts above the provider request-size caps are head-truncated
    (``_truncate_answer_context``); budgeted strategies never reach the cap.
    """
    _ANSWER_SYSTEM = (
        "You are a helpful assistant with access to conversation history. "
        "Answer concisely using only the provided context. "
        "If the context does not contain the answer, say you don't know."
    )

    def _answer_fn(query: str, context: str) -> str:
        context = _truncate_answer_context(context)
        if context.strip():
            prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        else:
            prompt = f"Question: {query}\nAnswer (no relevant context available):"
        try:
            return call_llm(prompt, system=_ANSWER_SYSTEM)
        except Exception as exc:
            return f"[LLM error: {exc}]"

    return _answer_fn


def _build_strategy(baseline: str, token_budget: int, data_dir: Optional[str],
                    embedder=None):
    """Build ``(retrieval_fn, answer_fn, memory_prepare)`` for a real dataset.

    ``none`` (our system) uses the REAL MemAge components per case: an
    isolated ``TieredMemoryStore`` holding the case's own memory entries
    (dataset ids preserved), searched by the actual two-stage pipeline
    (``CoarseRetriever`` -> ``LLMReranker`` -> token budget ->
    ``format_context``) and answered via ``call_llm`` — see
    :func:`_build_none_strategy`.  The ``full_history`` baseline uses
    ``history_from_case`` (formatted context string); the entry-based
    baselines (memoryos / memory_r1 / hmem) use
    ``memory_entries_from_case`` (individual entries with preserved dataset
    IDs).  All strategies answer with ``call_llm`` (provider-agnostic).

    Args:
        token_budget: Token budget for retrieval (forwarded to the pipeline
            / entry baselines).
        data_dir: Unused here (dataset files are loaded by the caller);
            kept for signature compatibility.
        embedder: Optional store embedder for ``none`` (tests inject a
            deterministic one).  ``None`` -> the store's default chain
            (sentence-transformers -> fastembed).

    Returns ``(retrieval_fn, answer_fn, memory_prepare)`` or ``None`` if a
    runtime dependency is unavailable (``src.llm`` import; or, for ``none``,
    a default store embedder when *embedder* is ``None``).
    """
    # All strategies answer via call_llm; `none` also reranks with it.
    try:
        from src.llm.client import call_llm
    except ImportError:
        return None

    from src.eval.baselines import (
        history_from_case,
        memory_entries_from_case,
        full_history_retrieval_fn,
        memoryos_retrieval_fn,
        memory_r1_retrieval_fn,
        hmem_retrieval_fn,
    )
    from src.eval.dataset import BenchmarkCase
    from src.eval.benchmark import RetrievalOutcome

    _answer_fn = _build_answer_fn(call_llm)

    # Our system baseline: real per-case store + real two-stage pipeline.
    if baseline == "none":
        return _build_none_strategy(token_budget, call_llm, embedder, _answer_fn)

    # Per-case retrieval: build baseline from this case's history each time.
    def _make_retrieval_fn(case: BenchmarkCase, bsl: str, budget: int):
        if bsl == "full_history":
            # Full-history baseline formats the whole history itself.
            return full_history_retrieval_fn(history_from_case(case))
        # Entry-based baselines (memoryos / memory_r1 / hmem) expect a list
        # of individual memory entries, not one preformatted history string:
        # a single giant string would collapse into one id-less "memory",
        # losing dataset turn IDs and making retrieval metrics degenerate.
        entries = memory_entries_from_case(case)
        if bsl == "memoryos":
            return memoryos_retrieval_fn(entries)
        if bsl == "memory_r1":
            return memory_r1_retrieval_fn(entries, token_budget=budget)
        if bsl == "hmem":
            return hmem_retrieval_fn(entries, token_budget=budget)
        raise ValueError(f"Unknown baseline: {bsl}")

    # Closure captures the per-case retrieval function.
    # BenchmarkRunner calls retrieval_fn(query), so we need to stash the
    # current case. We inject via memory_prepare_fn which sets _current_case.
    state: dict = {"retrieval_fn": None}

    def memory_prepare(case: BenchmarkCase) -> None:
        state["retrieval_fn"] = _make_retrieval_fn(case, baseline, token_budget)

    def retrieval_fn(query: str) -> RetrievalOutcome:
        fn = state.get("retrieval_fn")
        if fn is None:
            return RetrievalOutcome(context="", retrieved_ids=None, tokens=0)
        return fn(query)

    return retrieval_fn, _answer_fn, memory_prepare


def _load_real_dataset(
    dataset: str, data_dir: Optional[str], sample: Optional[int]
):
    """Load and sample-slice a real dataset, printing the standard banner.

    Returns ``(loaded, total, exit_code)``.  On failure ``loaded`` is
    ``None`` and *exit_code* is 2 (dataset files unavailable) or 1 (parse
    error); the actionable error has already been printed to stderr.
    """
    try:
        loaded = load_dataset(dataset, data_dir=data_dir)
    except DatasetFileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return None, 0, 2
    except DatasetFormatError as exc:
        print(f"Error: dataset '{dataset}' could not be parsed: {exc}", file=sys.stderr)
        return None, 0, 1

    path = resolve_dataset_file(dataset, data_dir)
    total = len(loaded)
    if sample is not None:
        from src.eval.dataset import BenchmarkDataset as _BD
        loaded = _BD(cases=list(loaded.cases)[:sample], name=loaded.name)
    print(f"Loaded {dataset}: {total} cases from {path}"
          + (f" (using first {len(loaded)})" if sample else ""))
    return loaded, total, 0


_DEP_ERROR_HINT = (
    "a runtime dependency is unavailable (src/llm client; for the 'none' "
    "pipeline also a src/store embedder: sentence-transformers or fastembed)"
)


def run_dataset_benchmark(
    dataset: str,
    baseline: str,
    k: int,
    token_budget: int,
    data_dir: Optional[str] = None,
    sample: Optional[int] = None,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> int:
    """Load a real dataset, report it, and benchmark if a strategy exists."""
    loaded, total, code = _load_real_dataset(dataset, data_dir, sample)
    if loaded is None:
        return code

    strategy = _build_strategy(baseline, token_budget, data_dir)
    if strategy is None:
        print(
            f"Error: cannot execute baseline '{baseline}' on real data - "
            f"{_DEP_ERROR_HINT}. "
            f"Dataset parsing succeeded ({total} cases).",
            file=sys.stderr,
        )
        return 2

    retrieval_fn, answer_fn, memory_prepare_fn = strategy
    runner = BenchmarkRunner(
        retrieval_fn=retrieval_fn,
        answer_fn=answer_fn,
        memory_prepare_fn=memory_prepare_fn,
        k=k,
    )
    report = runner.run_dataset(loaded)
    _print_report(report, k, f"Benchmark ({dataset}, baseline={baseline})")
    _save_run(report, output_dir=output_dir, dataset=dataset, baseline=baseline,
              token_budget=token_budget, k=k, sample=sample, seed=seed)
    return 0


# ---------------------------------------------------------------------------
# Real-dataset comparison & ablation (same abstractions as the synthetic path)
# ---------------------------------------------------------------------------


def build_real_comparison_specs(
    names: List[str],
    token_budget: int,
    data_dir: Optional[str] = None,
    embedder=None,
):
    """Build one isolated ``StrategySpec`` per baseline for a real dataset.

    Reuses :func:`_build_strategy` (the H-4 builders) once per name — no
    strategy shares state with another.  Each spec carries its own
    ``memory_prepare_fn``, so every case of every strategy gets a fresh
    preparation (case isolation).  All specs leave ``answer_fn`` unset so
    the comparison uses ONE shared answer function (fairness: only
    retrieval differs between strategies).

    Returns ``(specs, answer_fn)`` or ``(None, None)`` when a runtime
    dependency is missing (an actionable message is printed by the caller).
    """
    try:
        from src.llm.client import call_llm
    except ImportError:
        return None, None

    answer_fn = _build_answer_fn(call_llm)
    specs: List[StrategySpec] = []
    for name in names:
        strategy = _build_strategy(name, token_budget, data_dir, embedder=embedder)
        if strategy is None:
            return None, None
        retrieval_fn, _strategy_answer_fn, memory_prepare_fn = strategy
        specs.append(
            StrategySpec(
                name=name,
                retrieval_fn=retrieval_fn,
                answer_fn=None,          # shared answer_fn keeps the run fair
                memory_prepare_fn=memory_prepare_fn,
            )
        )
    return specs, answer_fn


def run_dataset_comparison(
    dataset: str,
    names: List[str],
    k: int,
    token_budget: int,
    data_dir: Optional[str] = None,
    sample: Optional[int] = None,
    export_json: Optional[str] = None,
    export_csv: Optional[str] = None,
    embedder=None,
) -> int:
    """Run ``--compare`` on a real dataset (LoCoMo / LongMemEval).

    Flow: load via the existing adapters → build one isolated strategy per
    name (reusing the H-4 builders; ``none`` included verbatim) →
    ``compare_strategies`` (existing ``BenchmarkRunner`` + metrics per
    strategy) → deterministic ``format_comparison_table`` → optional
    JSON/CSV export.

    Returns exit code 0 (2 dataset/dependency/validation, 1 parse error).
    """
    loaded, total, code = _load_real_dataset(dataset, data_dir, sample)
    if loaded is None:
        return code

    specs, answer_fn = build_real_comparison_specs(
        names, token_budget, data_dir=data_dir, embedder=embedder
    )
    if specs is None:
        print(
            f"Error: cannot execute --compare ({','.join(names)}) on real "
            f"data - {_DEP_ERROR_HINT}. "
            f"Dataset parsing succeeded ({total} cases).",
            file=sys.stderr,
        )
        return 2

    try:
        result = compare_strategies(loaded, specs, answer_fn, k=k)
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


def build_real_ablation_spec(
    cfg: AblationConfig,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    data_dir: Optional[str] = None,
    embedder=None,
) -> Optional[StrategySpec]:
    """Map one ablation config onto the real two-stage pipeline strategy.

    All four real ablation families reuse :func:`_build_none_strategy`
    (the very builder behind the ``none`` baseline) with exactly one factor
    varied per config — no parallel strategy stack:

        * ``stage`` (params ``stage``): ``"coarse_only"`` swaps Stage-2 for
          the existing ``PassthroughReranker``; ``"two_stage"`` keeps the
          real ``LLMReranker``.
        * ``k`` (params ``k``): Stage-1 candidate count (``pipe_k``).  The
          metric cutoff ``k`` lives on ``BenchmarkRunner`` and is untouched.
        * ``token_budget`` (params ``token_budget``): budget forwarded to
          ``make_retrieval_fn`` per config.
        * signal families (params ``thresholds``): controller-gated
          ingestion (see ``_build_none_strategy``) — the selective-memory
          controller decides what is stored; retrieval/answering are the
          same pipeline as ``none``.

    Returns ``None`` when a runtime dependency is unavailable (the caller
    reports it).  Spec ``name`` is the config label (``run_ablation``
    re-applies it anyway); ``answer_fn`` stays unset so all configs share
    the one comparison-level answer function.
    """
    params = cfg.params
    stage = params.get("stage", "two_stage")
    pipe_k = params.get("k")
    thresholds = params.get("thresholds")
    budget = params.get("token_budget", token_budget)

    try:
        from src.llm.client import call_llm
    except ImportError:
        return None

    built = _build_none_strategy(
        budget,
        call_llm,
        embedder,
        _build_answer_fn(call_llm),
        stage=stage,
        pipe_k=pipe_k,
        thresholds=thresholds,
    )
    if built is None:
        return None
    retrieval_fn, _answer_fn, memory_prepare_fn = built
    return StrategySpec(
        name=cfg.label,
        retrieval_fn=retrieval_fn,
        answer_fn=None,
        memory_prepare_fn=memory_prepare_fn,
    )


def run_dataset_ablation(
    dataset: str,
    configs: List[AblationConfig],
    ablation: str,
    k: int,
    token_budget: int,
    data_dir: Optional[str] = None,
    sample: Optional[int] = None,
    export_json: Optional[str] = None,
    export_csv: Optional[str] = None,
    embedder=None,
) -> int:
    """Run ``--ablate`` (stage / k / budget / signal) on a real dataset.

    Flow: load via the existing adapters → build one isolated strategy per
    config (all configs share the same builder, one factor differs) →
    ``run_ablation`` (delegates to ``compare_strategies`` → existing
    ``BenchmarkRunner`` + metrics, metric cutoff ``k`` fixed across
    configs) → deterministic table → optional JSON/CSV export.

    Returns exit code 0 (2 dataset/dependency/validation, 1 parse error).
    """
    loaded, total, code = _load_real_dataset(dataset, data_dir, sample)
    if loaded is None:
        return code

    try:
        from src.llm.client import call_llm
    except ImportError:
        call_llm = None
    if call_llm is None:
        print(
            f"Error: cannot execute --ablate {ablation} on real "
            f"data - {_DEP_ERROR_HINT}. "
            f"Dataset parsing succeeded ({total} cases).",
            file=sys.stderr,
        )
        return 2
    answer_fn = _build_answer_fn(call_llm)

    # Probe the first config for an actionable dependency error BEFORE the
    # run (e.g. no store embedder for the none pipeline).
    probe = build_real_ablation_spec(
        configs[0], token_budget=token_budget, data_dir=data_dir, embedder=embedder
    )
    if probe is None:
        print(
            f"Error: cannot execute --ablate {ablation} on real data - "
            f"{_DEP_ERROR_HINT}. "
            f"Dataset parsing succeeded ({total} cases).",
            file=sys.stderr,
        )
        return 2

    def builder(cfg: AblationConfig) -> StrategySpec:
        spec = build_real_ablation_spec(
            cfg, token_budget=token_budget, data_dir=data_dir, embedder=embedder
        )
        if spec is None:  # cannot happen: probe succeeded with same deps
            raise ValueError(f"could not build strategy for config {cfg.label!r}")
        return spec

    try:
        result = run_ablation(
            loaded, configs, builder, answer_fn, k=k, ablation=ablation
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

    # Comparison mode (orthogonal to --baseline; validated here, then run on
    # the selected dataset — synthetic demo or a real dataset's cases).
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
        if args.dataset == "synthetic":
            return run_synthetic_comparison(
                names,
                k=args.k,
                token_budget=args.token_budget,
                export_json=args.export_json,
                export_csv=args.export_csv,
            )
        # Real dataset: same name validation as the single-baseline path.
        msg = _get_unavailable_message(args.dataset, "none", args.data_dir)
        if msg is not None:
            print(f"Error: {msg}", file=sys.stderr)
            if "not yet implemented" in msg or "not available locally" in msg:
                return 2
            return 1
        return run_dataset_comparison(
            dataset=args.dataset,
            names=names,
            k=args.k,
            token_budget=args.token_budget,
            data_dir=args.data_dir,
            sample=args.sample,
            export_json=args.export_json,
            export_csv=args.export_csv,
        )

    # Ablation mode (mutually exclusive with --compare; runs on the selected
    # dataset — synthetic demo or a real dataset's cases).
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
        elif kind == "signal":
            if not raw:
                print(
                    f"Error: --ablate signal requires a signal name, e.g. "
                    f"--ablate signal:importance  (choices: {sorted(SIGNAL_NAMES)})",
                    file=sys.stderr,
                )
                return 1
            try:
                configs = signal_configs(raw)
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
            ablation = f"signal:{raw}"
        else:
            print(
                f"Error: unknown ablation kind {kind!r}. "
                f"Supported: stage, k, budget, signal",
                file=sys.stderr,
            )
            return 1
        if args.dataset == "synthetic":
            return run_synthetic_ablation(
                configs,
                ablation=ablation,
                k=args.k,
                token_budget=args.token_budget,
                export_json=args.export_json,
                export_csv=args.export_csv,
            )
        # Real dataset: same availability contract as the single-baseline path.
        msg = _get_unavailable_message(args.dataset, "none", args.data_dir)
        if msg is not None:
            print(f"Error: {msg}", file=sys.stderr)
            if "not yet implemented" in msg or "not available locally" in msg:
                return 2
            return 1
        return run_dataset_ablation(
            dataset=args.dataset,
            configs=configs,
            ablation=ablation,
            k=args.k,
            token_budget=args.token_budget,
            data_dir=args.data_dir,
            sample=args.sample,
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
            k=args.k,
            token_budget=args.token_budget,
            baseline=args.baseline,
            sample=args.sample,
            seed=args.seed,
            output_dir=args.output_dir,
        )

    return run_dataset_benchmark(
        dataset=args.dataset,
        baseline=args.baseline,
        k=args.k,
        token_budget=args.token_budget,
        data_dir=args.data_dir,
        sample=args.sample,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
