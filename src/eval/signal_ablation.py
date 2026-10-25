"""Controller signal ablation configs for MemAge evaluation.

Provides four additional AblationConfig families corresponding to the four
controller scoring signals documented in docs/interfaces.md:

    importance          — vary importance_min threshold
    temporal_validity   — enable/disable temporal gating
    contradiction       — vary contradiction_update / contradiction_delete thresholds
    storage_pressure    — vary pressure_suppress threshold

Each family produces a list[AblationConfig] suitable for run_ablation() via
build_synthetic_ablation_spec() in run_benchmark.py.

Usage (CLI):
    python -m src.eval.run_benchmark --ablate signal:importance
    python -m src.eval.run_benchmark --ablate signal:temporal
    python -m src.eval.run_benchmark --ablate signal:contradiction
    python -m src.eval.run_benchmark --ablate signal:storage

These can also be consumed programmatically:

    from src.eval.signal_ablation import signal_configs
    configs = signal_configs("importance")
    # -> list[AblationConfig] with labels like "importance_min=0.1"
"""
from __future__ import annotations

from typing import List

from src.eval.ablation import AblationConfig
from src.controller.policy import DEFAULT_THRESHOLDS


# Canonical signal names exposed via --ablate signal:<name>
SIGNAL_NAMES = ("importance", "temporal", "contradiction", "storage")


def _cfg(kind: str, label: str, **params) -> AblationConfig:
    return AblationConfig(kind=kind, label=label, params=dict(params))


def importance_configs() -> List[AblationConfig]:
    """Vary the importance_min threshold: 0.0 (off), 0.3 (default), 0.6 (high)."""
    default = DEFAULT_THRESHOLDS["importance_min"]
    return [
        _cfg("importance_min", "importance_min=0.0",
             thresholds={"importance_min": 0.0}),
        _cfg("importance_min", f"importance_min={default}",
             thresholds={"importance_min": default}),
        _cfg("importance_min", "importance_min=0.6",
             thresholds={"importance_min": 0.6}),
    ]


def temporal_configs() -> List[AblationConfig]:
    """Enable/disable temporal gating: off (default=0.0), mid (0.3), strict (0.6)."""
    return [
        _cfg("temporal_min", "temporal_min=0.0",
             thresholds={"temporal_min": 0.0}),
        _cfg("temporal_min", "temporal_min=0.3",
             thresholds={"temporal_min": 0.3}),
        _cfg("temporal_min", "temporal_min=0.6",
             thresholds={"temporal_min": 0.6}),
    ]


def contradiction_configs() -> List[AblationConfig]:
    """Vary contradiction thresholds: loose, default, strict."""
    return [
        _cfg("contradiction", "contradiction=loose",
             thresholds={"contradiction_update": 0.3,
                         "contradiction_delete_existing": 0.6}),
        _cfg("contradiction", "contradiction=default",
             thresholds={
                 "contradiction_update": DEFAULT_THRESHOLDS["contradiction_update"],
                 "contradiction_delete_existing": DEFAULT_THRESHOLDS["contradiction_delete_existing"],
             }),
        _cfg("contradiction", "contradiction=strict",
             thresholds={"contradiction_update": 0.7,
                         "contradiction_delete_existing": 0.9}),
    ]


def storage_configs() -> List[AblationConfig]:
    """Vary pressure_suppress threshold: lenient (0.95), default (0.9), aggressive (0.7)."""
    return [
        _cfg("pressure_suppress", "pressure_suppress=0.7",
             thresholds={"pressure_suppress": 0.7}),
        _cfg("pressure_suppress", f"pressure_suppress=0.9",
             thresholds={"pressure_suppress": DEFAULT_THRESHOLDS["pressure_suppress"]}),
        _cfg("pressure_suppress", "pressure_suppress=0.95",
             thresholds={"pressure_suppress": 0.95}),
    ]


_FAMILIES = {
    "importance": importance_configs,
    "temporal": temporal_configs,
    "contradiction": contradiction_configs,
    "storage": storage_configs,
}


def signal_configs(signal: str) -> List[AblationConfig]:
    """Return the ablation config list for a named signal.

    Args:
        signal: one of "importance", "temporal", "contradiction", "storage".

    Raises:
        ValueError: unknown signal name.
    """
    if signal not in _FAMILIES:
        raise ValueError(
            f"Unknown signal {signal!r}. Supported: {sorted(_FAMILIES)}"
        )
    return _FAMILIES[signal]()
