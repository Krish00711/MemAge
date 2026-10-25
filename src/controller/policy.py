"""Joint decision policy for the Selective Memory Controller.

Combines the four scoring signals into a single ADD / UPDATE / DELETE / NOOP
decision, following the documented thresholds.

Thresholds are keyword-configurable so callers can tune them without subclassing.

Owner: Krish. Interface contract: docs/interfaces.md
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Default thresholds (initial values from the project handoff)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: dict[str, float] = {
    # Minimum importance to consider storing at all
    "importance_min": 0.3,
    # Contradiction level above which we prefer UPDATE over ADD
    "contradiction_update": 0.5,
    # Contradiction level above which the *existing* item should be deleted
    # and the candidate kept (strong direct conflict — the old fact is stale)
    "contradiction_delete_existing": 0.8,
    # Storage pressure above which we suppress low-importance ADDs
    "pressure_suppress": 0.9,
    # Temporal-validity minimum — below this, skip an otherwise-valid candidate
    "temporal_min": 0.0,   # off by default; set >0 to prune stale candidates
}


def decide(
    candidate: dict,
    existing_matches: list[dict],
    scores: dict[str, float],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Apply the joint decision policy.

    Parameters
    ----------
    candidate:
        A CandidateMemoryItem dict (plain dict, never mutated here).
    existing_matches:
        Items returned by ``store.query(candidate["text"], k, "all")``.
    scores:
        Dict with keys ``importance``, ``temporal_validity``, ``contradiction``,
        ``storage_pressure`` — all floats in [0, 1].
    thresholds:
        Override any subset of DEFAULT_THRESHOLDS.

    Returns
    -------
    A Decision dict as documented in docs/interfaces.md::

        {
            "action": "ADD" | "UPDATE" | "DELETE" | "NOOP",
            "target_id": str | None,
            "scores": { <the four scores> },
        }

    DELETE semantics
    ----------------
    DELETE always identifies an *existing* target_id (the stale/contradicted
    memory that should be evicted).  The incoming candidate itself is ADD-worthy
    after the old item is removed; the store layer is responsible for calling
    ``store.delete(target_id)`` then ``store.add(candidate)``.
    The controller never silently discards the incoming candidate.
    """
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    importance = scores["importance"]
    temporal = scores["temporal_validity"]
    contradiction = scores["contradiction"]
    storage_pressure = scores.get("storage_pressure", 0.0)

    # -----------------------------------------------------------------------
    # Rule 1 — Temporal validity gate
    # -----------------------------------------------------------------------
    if temporal < thr["temporal_min"]:
        return _decision("NOOP", None, scores)

    # -----------------------------------------------------------------------
    # Rule 2 — Not important enough to store
    # -----------------------------------------------------------------------
    if importance < thr["importance_min"]:
        return _decision("NOOP", None, scores)

    # -----------------------------------------------------------------------
    # Rule 3 — No existing matches → clean ADD
    # -----------------------------------------------------------------------
    if not existing_matches:
        # Still suppress if memory is bursting and this is low-priority
        if storage_pressure >= thr["pressure_suppress"] and importance < 0.6:
            return _decision("NOOP", None, scores)
        return _decision("ADD", None, scores)

    # -----------------------------------------------------------------------
    # Pick the best existing match (first in list; caller sorts by similarity)
    # -----------------------------------------------------------------------
    best_match = existing_matches[0]
    best_id = best_match.get("id")

    # -----------------------------------------------------------------------
    # Rule 4 — Strong contradiction: the existing fact is wrong/stale → DELETE it
    #          and implicitly ADD the candidate (caller handles the two-step).
    # -----------------------------------------------------------------------
    if contradiction >= thr["contradiction_delete_existing"]:
        return _decision("DELETE", best_id, scores)

    # -----------------------------------------------------------------------
    # Rule 5 — Moderate contradiction: UPDATE the existing item with new info
    # -----------------------------------------------------------------------
    if contradiction >= thr["contradiction_update"]:
        return _decision("UPDATE", best_id, scores)

    # -----------------------------------------------------------------------
    # Rule 6 — Candidate is important, no real conflict → ADD
    #          but suppress if storage is nearly full and importance is low
    # -----------------------------------------------------------------------
    if storage_pressure >= thr["pressure_suppress"] and importance < 0.6:
        return _decision("NOOP", None, scores)

    return _decision("ADD", None, scores)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _decision(action: str, target_id: str | None, scores: dict) -> dict:
    return {
        "action": action,
        "target_id": target_id,
        "scores": {
            "importance": scores.get("importance", 0.0),
            "temporal_validity": scores.get("temporal_validity", 1.0),
            "contradiction": scores.get("contradiction", 0.0),
            "storage_pressure": scores.get("storage_pressure", 0.0),
        },
    }
