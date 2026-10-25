"""Public API for the Selective Memory Controller.

Exposes:
    decide(candidate, existing_matches, llm=None, ...) -> Decision
    process(candidate, store, llm=None, ...)            -> Decision

Owner: Krish. Interface contract: docs/interfaces.md
"""
from __future__ import annotations

from typing import Any, Callable

from src.controller.scoring import (
    score_importance,
    score_temporal_validity,
    score_contradiction,
    score_storage_pressure,
)
from src.controller.policy import decide as _policy_decide, DEFAULT_THRESHOLDS

# Default store capacities — must match TieredMemoryStore defaults so the
# pressure signal is calibrated correctly when no store is available.
_DEFAULT_STM_CAP = 20
_DEFAULT_MTM_CAP = 100


def decide(
    candidate: dict,
    existing_matches: list[dict],
    llm: Callable | None = None,
    thresholds: dict[str, float] | None = None,
    counts: dict[str, int] | None = None,
    stm_capacity: int = _DEFAULT_STM_CAP,
    mtm_capacity: int = _DEFAULT_MTM_CAP,
) -> dict[str, Any]:
    """Score a candidate and return a Decision dict (see docs/interfaces.md).

    Parameters
    ----------
    candidate:
        A CandidateMemoryItem plain dict.
    existing_matches:
        Items from ``store.query(candidate["text"], k, "all")``.
        Pass ``[]`` when no similar memories exist.
    llm:
        Injectable LLM callable ``(prompt, system=None, **kwargs) -> str``.
        Defaults to ``src.llm.client.call_llm`` when None.
    thresholds:
        Override any subset of DEFAULT_THRESHOLDS.
    counts:
        Pre-fetched ``store.count()`` dict.  Computed from ``existing_matches``
        tier distribution when not supplied (rough approximation only).
    stm_capacity / mtm_capacity:
        Must match the store's configured capacities for accurate pressure.

    Returns
    -------
    Decision = {
        "action": "ADD" | "UPDATE" | "DELETE" | "NOOP",
        "target_id": str | None,
        "scores": {"importance": float, "temporal_validity": float,
                   "contradiction": float, "storage_pressure": float},
    }
    """
    if llm is None:
        from src.llm.client import call_llm as llm  # type: ignore[assignment]

    # 1. Compute storage-pressure from counts or a best-effort estimate
    if counts is None:
        counts = _estimate_counts(existing_matches)

    # 2. Run all four signals
    importance = score_importance(candidate, llm)
    temporal = score_temporal_validity(candidate, existing_matches)
    contradiction = score_contradiction(candidate, existing_matches, llm)
    pressure = score_storage_pressure(counts, stm_capacity, mtm_capacity)

    scores = {
        "importance": importance,
        "temporal_validity": temporal,
        "contradiction": contradiction,
        "storage_pressure": pressure,
    }

    # 3. Apply joint policy
    return _policy_decide(candidate, existing_matches, scores, thresholds)


def process(
    candidate: dict,
    store,  # TieredMemoryStore — not imported to avoid chromadb at module load
    llm: Callable | None = None,
    thresholds: dict[str, float] | None = None,
    k: int = 5,
) -> dict[str, Any]:
    """Full pipeline: query store → score → decide → apply decision.

    This is the high-level function that Krish wires into the ingestion loop.
    It does NOT import ChromaDB directly; `store` is injected by the caller.

    Parameters
    ----------
    candidate:
        CandidateMemoryItem dict from ``extract_candidates()``.
    store:
        A ``TieredMemoryStore`` instance (or any object with the same interface).
    llm:
        Injectable LLM callable.  Defaults to ``src.llm.client.call_llm``.
    thresholds:
        Threshold overrides forwarded to ``decide()``.
    k:
        Number of nearest neighbours to retrieve from the store.

    Returns
    -------
    The Decision dict, after the action has been applied to the store.
    """
    existing = store.query(candidate["text"], k=k, tier="all")
    counts = store.count()

    decision = decide(
        candidate,
        existing,
        llm=llm,
        thresholds=thresholds,
        counts=counts,
        stm_capacity=getattr(store, "stm_capacity", _DEFAULT_STM_CAP),
        mtm_capacity=getattr(store, "mtm_capacity", _DEFAULT_MTM_CAP),
    )

    action = decision["action"]
    target_id = decision["target_id"]

    if action == "ADD":
        store.add(candidate)
    elif action == "UPDATE" and target_id:
        store.update(target_id, candidate["text"],
                     timestamp=candidate.get("timestamp"))
    elif action == "DELETE":
        if target_id:
            # Verify the target still exists before deleting it.
            # store.get() is available on TieredMemoryStore; fall back to
            # checking store.delete()'s return value if get() is absent.
            target_exists = True
            if hasattr(store, "get"):
                target_exists = store.get(target_id) is not None
            if target_exists:
                deleted = store.delete(target_id)
                if not deleted:
                    target_exists = False
            # Candidate is always preserved regardless of whether the delete
            # succeeded — the spec forbids silently discarding it.
            store.add(candidate)
        else:
            # target_id is None (best match had no "id" field):
            # cannot safely delete anything, so just ADD the candidate.
            store.add(candidate)
    # NOOP: do nothing

    return decision


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _estimate_counts(existing_matches: list[dict]) -> dict[str, int]:
    """Rough tier count from query results when store.count() is not available."""
    counts: dict[str, int] = {"STM": 0, "MTM": 0, "LTM": 0}
    for m in existing_matches:
        tier = m.get("tier", "STM")
        if tier in counts:
            counts[tier] += 1
    return counts
