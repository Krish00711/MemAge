"""End-to-end MemAge ingestion pipeline.

Public entry point:

    run_turn(turn, metadata, store, llm=None, k=5, thresholds=None)
        -> TurnResult

Wires together extraction -> controller (score + decide + apply) for every
candidate fact extracted from a single conversation turn.

All dependencies (store, llm) are injected so unit tests need neither
ChromaDB nor a real API key.

Owner: Krish.  Interface contract: docs/interfaces.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CandidateResult:
    """Outcome for a single candidate memory item."""
    candidate: dict       # plain CandidateMemoryItem dict (never mutated here)
    decision: dict        # Decision dict from controller
    ok: bool              # True iff the store operation completed without error
    error: str | None = None  # exception message when ok=False


@dataclass
class TurnResult:
    """Aggregated result for one conversation turn."""
    turn: str
    candidates: list[CandidateResult] = field(default_factory=list)

    @property
    def added(self) -> list[str]:
        return [r.candidate["id"] for r in self.candidates
                if r.decision["action"] == "ADD" and r.ok]

    @property
    def updated(self) -> list[str]:
        return [r.candidate["id"] for r in self.candidates
                if r.decision["action"] == "UPDATE" and r.ok]

    @property
    def deleted(self) -> list[str]:
        """IDs of the *existing* memories that were evicted (target_id on DELETE)."""
        return [r.decision["target_id"] for r in self.candidates
                if r.decision["action"] == "DELETE" and r.ok
                and r.decision["target_id"] is not None]

    @property
    def noops(self) -> list[str]:
        return [r.candidate["id"] for r in self.candidates
                if r.decision["action"] == "NOOP"]

    @property
    def errors(self) -> list[str]:
        return [r.error for r in self.candidates if not r.ok and r.error]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_turn(
    turn: str,
    metadata: dict | None = None,
    store: Any = None,
    llm: Callable | None = None,
    k: int = 5,
    thresholds: dict[str, float] | None = None,
) -> TurnResult:
    """Process one conversation turn end-to-end.

    Extracts candidate memories, scores each one, decides an action and
    applies it to the store.

    Parameters
    ----------
    turn:
        Raw text of the dialogue turn.
    metadata:
        Optional dict with keys ``speaker``, ``timestamp`` (ISO 8601),
        ``source`` — forwarded to ``extract_candidates()``.
    store:
        Object satisfying the TieredMemoryStore interface:
        ``query``, ``count``, ``add``, ``update``, ``delete``, ``get``.
        Must be provided; raises ``ValueError`` if None.
    llm:
        Injectable callable ``(prompt, system=None, **kwargs) -> str``.
        Defaults to ``src.llm.client.call_llm`` when None.
    k:
        Nearest-neighbour matches to retrieve per candidate.
    thresholds:
        Optional controller threshold overrides.

    Returns
    -------
    TurnResult
        Structured result with each candidate's decision and store outcome.
        Never raises; errors are captured in ``CandidateResult.error``.

    Notes
    -----
    Real-store integration has not been verified in this environment
    (chromadb/sentence-transformers not installed in the test venv).
    """
    if store is None:
        raise ValueError("store must be provided.")

    result = TurnResult(turn=turn)

    # ------------------------------------------------------------------
    # Step 1 — Extract candidate memory items
    # ------------------------------------------------------------------
    from src.extraction.extractor import extract_candidates

    try:
        candidates = extract_candidates(turn, metadata=metadata, llm=llm)
    except Exception as exc:
        # Surface extraction failure as a sentinel NOOP entry so the caller
        # always gets a well-formed TurnResult.
        sentinel = {
            "id": "__extraction_error__",
            "text": turn[:120],
            "timestamp": (metadata or {}).get("timestamp", ""),
            "source": (metadata or {}).get("source", "unknown"),
            "embedding": None,
        }
        noop_decision = {
            "action": "NOOP",
            "target_id": None,
            "scores": {
                "importance": 0.0, "temporal_validity": 0.0,
                "contradiction": 0.0, "storage_pressure": 0.0,
            },
        }
        result.candidates.append(
            CandidateResult(
                candidate=sentinel,
                decision=noop_decision,
                ok=False,
                error=f"extraction failed: {exc}",
            )
        )
        return result

    # Empty turn or small talk — return with no candidates.
    if not candidates:
        return result

    # ------------------------------------------------------------------
    # Step 2 — Run controller.process() for each candidate
    # ------------------------------------------------------------------
    from src.controller.controller import process as _ctrl_process

    for candidate in candidates:
        try:
            decision = _ctrl_process(
                candidate,
                store,
                llm=llm,
                thresholds=thresholds,
                k=k,
            )
            result.candidates.append(
                CandidateResult(candidate=candidate, decision=decision, ok=True)
            )
        except Exception as exc:
            # Store or scoring failure: preserve candidate, report error.
            # Use a NOOP decision so the result is always schema-complete.
            fallback_decision = {
                "action": "NOOP",
                "target_id": None,
                "scores": {
                    "importance": 0.0, "temporal_validity": 0.0,
                    "contradiction": 0.0, "storage_pressure": 0.0,
                },
            }
            result.candidates.append(
                CandidateResult(
                    candidate=candidate,
                    decision=fallback_decision,
                    ok=False,
                    error=str(exc),
                )
            )

    return result
