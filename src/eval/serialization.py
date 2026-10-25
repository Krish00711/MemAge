"""Deterministic JSON/CSV serialization for MemAge benchmark results.

Exports the existing result dataclasses — ``BenchmarkReport``,
``ComparisonResult`` (from ``src/eval/comparison.py``), and ``AblationResult``
(from ``src/eval/ablation.py``) — without recomputing anything: JSON is
``dataclasses.asdict()`` + ``json.dumps``; CSV is one row per
strategy/configuration using the aggregate fields already on
``BenchmarkReport``.

Contract:
    * **JSON** preserves the full structure: result-level metadata
      (``dataset_name``/``k``/``num_cases``/``ablation``/``configs``) plus
      each strategy's complete report including ``per_case``. ``None``
      aggregates serialize as JSON ``null`` — never as fake zeros.
    * **CSV** is a flat table, one row per strategy/configuration, columns::

        name,kind,mean_f1,mean_bleu1,mean_recall_at_k,mean_precision_at_k,
        mrr,mean_tokens,mean_latency_ms,num_queries

      ``name`` is the strategy name (comparison) or configuration label
      (ablation); ``kind`` is the ``AblationConfig.kind`` (empty for
      comparison/bare-report rows). ``None`` aggregates render as empty
      cells — never ``0.0``.
    * **Deterministic**: field order follows dataclass definitions, CSV uses
      ``\\n`` line terminators and no quoting unless needed; serializing the
      same values always yields byte-identical output.
    * No new metrics, no metric recalculation, no I/O beyond writing the
      file the caller asked for (parent directories are NOT created).

Programmatic API::

    to_json(result, indent=2) -> str
    write_json(result, path) -> Path
    to_csv(result) -> str
    write_csv(result, path) -> Path
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Union

from src.eval.ablation import AblationResult
from src.eval.benchmark import BenchmarkReport
from src.eval.comparison import ComparisonResult

# The three supported result types.
SerializableResult = Union[BenchmarkReport, ComparisonResult, AblationResult]

# Fixed CSV column order (identifiers first, then the aggregates the
# benchmark already produces; num_queries last for context).
CSV_HEADER: List[str] = [
    "name",
    "kind",
    "mean_f1",
    "mean_bleu1",
    "mean_recall_at_k",
    "mean_precision_at_k",
    "mrr",
    "mean_tokens",
    "mean_latency_ms",
    "num_queries",
]


# ---------------------------------------------------------------------------
# Validation / row helpers
# ---------------------------------------------------------------------------


def _as_dict(result: SerializableResult) -> dict:
    """Validate *result* and convert it with ``dataclasses.asdict()``."""
    if not isinstance(result, (BenchmarkReport, ComparisonResult, AblationResult)):
        raise TypeError(
            "result must be BenchmarkReport, ComparisonResult, or "
            f"AblationResult, got {type(result)!r}"
        )
    return asdict(result)


def _cell(value: Any) -> Any:
    """CSV cell for an aggregate: ``None`` becomes an EMPTY cell (not 0.0)."""
    return "" if value is None else value


def _metric_cells(report: BenchmarkReport) -> List[Any]:
    """Aggregate metric cells for *report*, in ``CSV_HEADER`` order."""
    return [
        _cell(report.mean_f1),
        _cell(report.mean_bleu1),
        _cell(report.mean_recall_at_k),
        _cell(report.mean_precision_at_k),
        _cell(report.mrr),
        _cell(report.mean_tokens),
        _cell(report.mean_latency_ms),
        _cell(report.num_queries),
    ]


def _csv_rows(result: SerializableResult) -> List[List[Any]]:
    """Build CSV rows (header excluded) for *result*."""
    if isinstance(result, BenchmarkReport):
        return [["", "", *_metric_cells(result)]]
    if isinstance(result, ComparisonResult):
        return [[r.name, "", *_metric_cells(r.report)] for r in result.results]
    # AblationResult: one row per config, aligned 1:1 with comparison results
    rows: List[List[Any]] = []
    results = result.comparison.results
    if len(results) != len(result.configs):
        raise ValueError(
            "AblationResult configs/results length mismatch: "
            f"{len(result.configs)} vs {len(results)}"
        )
    for cfg, strategy_result in zip(result.configs, results):
        rows.append([cfg.label, cfg.kind, *_metric_cells(strategy_result.report)])
    return rows


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def to_json(result: SerializableResult, indent: int = 2) -> str:
    """Serialize *result* to a deterministic JSON string.

    Args:
        result: ``BenchmarkReport``, ``ComparisonResult``, or ``AblationResult``.
        indent: ``json.dumps`` indent (default 2); ``None`` for compact.

    Returns:
        JSON text (dataclass field order preserved; ``None`` -> ``null``).

    Raises:
        TypeError: Unsupported *result* type.
    """
    return json.dumps(_as_dict(result), indent=indent, ensure_ascii=True)


def write_json(result: SerializableResult, path: Union[str, Path], indent: int = 2) -> Path:
    """Write :func:`to_json` output to *path* (UTF-8). Parent dirs are not created.

    Returns:
        The resolved ``Path`` written.

    Raises:
        TypeError: Unsupported *result* type.
        OSError: The location is not writable (e.g. missing parent, a
            directory, permission denied).
        ValueError: The path is otherwise invalid (e.g. embedded NUL).
    """
    out = Path(path)
    out.write_text(to_json(result, indent=indent), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def to_csv(result: SerializableResult) -> str:
    """Serialize *result* to a deterministic CSV string.

    One row per strategy (comparison) or configuration (ablation); a bare
    ``BenchmarkReport`` yields one row with empty identifiers. ``None``
    metrics become empty cells.

    Returns:
        CSV text with ``\\n`` line terminators and a header row.

    Raises:
        TypeError: Unsupported *result* type.
        ValueError: Ablation configs/results are misaligned (should be
            impossible for results produced by ``run_ablation``).
    """
    _as_dict(result)  # validate type early (also guarantees dataclass-ness)
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(CSV_HEADER)
    writer.writerows(_csv_rows(result))
    return buf.getvalue()


def write_csv(result: SerializableResult, path: Union[str, Path]) -> Path:
    """Write :func:`to_csv` output to *path* (UTF-8). Parent dirs are not created.

    Returns:
        The resolved ``Path`` written.

    Raises:
        TypeError: Unsupported *result* type.
        OSError: The location is not writable.
        ValueError: Invalid path (or misaligned ablation results).
    """
    out = Path(path)
    out.write_text(to_csv(result), encoding="utf-8")
    return out
