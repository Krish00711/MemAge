"""Scoring signals for the Selective Memory Controller.

Each function returns a float in [0, 1].
All signals accept an injectable `llm` callable so tests need no real API key.

Owner: Krish. Interface contract: docs/interfaces.md
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(v: float) -> float:
    """Clamp a value to [0, 1], defaulting to 0.5 on NaN/inf."""
    if not math.isfinite(v):
        return 0.5
    return max(0.0, min(1.0, float(v)))


def _parse_score(raw: str, key: str) -> float | None:
    """Pull a float from a JSON object returned by the LLM.

    Returns None if the reply cannot be parsed — callers must handle fallback.
    """
    try:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            return None
        data = json.loads(raw[start:end + 1])
        val = data.get(key)
        if val is None:
            return None
        return _clamp(float(val))
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Signal 1 — Importance
# ---------------------------------------------------------------------------

_IMPORTANCE_SYSTEM = (
    "You are a memory importance scorer. "
    "Reply with ONLY a JSON object, nothing else."
)

_IMPORTANCE_PROMPT = """\
Rate the importance of the following fact for long-term memory on a scale from 0.0 to 1.0.
Important = personal detail, concrete plan, preference, key event, relationship, or decision.
Unimportant = small talk, filler, trivial observation.

Fact: {text}

Reply with exactly: {{"importance": <float between 0 and 1>}}
"""


def score_importance(candidate: dict, llm, fallback: float = 0.5) -> float:
    """Ask the LLM how important this fact is. Falls back to `fallback` on error."""
    prompt = _IMPORTANCE_PROMPT.format(text=candidate.get("text", ""))
    try:
        raw = llm(prompt, system=_IMPORTANCE_SYSTEM)
    except Exception:
        return fallback
    result = _parse_score(raw, "importance")
    return result if result is not None else fallback


# ---------------------------------------------------------------------------
# Signal 2 — Temporal Validity
# ---------------------------------------------------------------------------

def score_temporal_validity(candidate: dict, existing_matches: list[dict],
                             fallback: float = 1.0) -> float:
    """How time-valid is the candidate compared to the best matching existing memory?

    - No matches → full validity (1.0).
    - Candidate newer than closest match → 1.0.
    - Same timestamp → 0.5.
    - Candidate older → 0.0.
    - Unparseable timestamps → fallback.
    """
    if not existing_matches:
        return 1.0

    def _parse_ts(ts: str | None) -> datetime | None:
        if not ts:
            return None
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S.%f"):
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            return None

    cand_ts = _parse_ts(candidate.get("timestamp"))
    if cand_ts is None:
        return fallback

    def _naive(dt: datetime) -> datetime:
        if dt.tzinfo is not None:
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt

    best_ts: datetime | None = None
    for m in existing_matches:
        mt = _parse_ts(m.get("timestamp"))
        if mt is None:
            continue
        mt_naive = _naive(mt)
        if best_ts is None or mt_naive > best_ts:
            best_ts = mt_naive

    if best_ts is None:
        return fallback

    cn = _naive(cand_ts)
    if cn > best_ts:
        return 1.0
    if cn == best_ts:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Signal 3 — Contradiction
# ---------------------------------------------------------------------------

_CONTRADICTION_SYSTEM = (
    "You are a contradiction detector. "
    "Reply with ONLY a JSON object, nothing else."
)

_CONTRADICTION_PROMPT = """\
Does the new fact contradict any of the existing facts listed below?
A contradiction is a direct factual conflict (e.g. different location, opposite statement).

New fact: {new}

Existing facts:
{existing}

Reply with exactly: {{"contradiction": <float between 0 and 1>}}
where 1.0 = strong contradiction and 0.0 = no contradiction.
"""


def score_contradiction(candidate: dict, existing_matches: list[dict],
                        llm, fallback: float = 0.0) -> float:
    """LLM-based contradiction detector. Returns 0.0 when there are no matches."""
    if not existing_matches:
        return 0.0
    existing_text = "\n".join(
        f"- {m.get('text', '')}" for m in existing_matches
    )
    prompt = _CONTRADICTION_PROMPT.format(
        new=candidate.get("text", ""),
        existing=existing_text,
    )
    try:
        raw = llm(prompt, system=_CONTRADICTION_SYSTEM)
    except Exception:
        return fallback
    result = _parse_score(raw, "contradiction")
    return result if result is not None else fallback


# ---------------------------------------------------------------------------
# Signal 4 — Storage Pressure
# ---------------------------------------------------------------------------

def score_storage_pressure(counts: dict[str, int],
                            stm_capacity: int = 20,
                            mtm_capacity: int = 100) -> float:
    """Fraction of bounded capacity currently used.

    counts = {"STM": n, "MTM": n, "LTM": n}  (from TieredMemoryStore.count())
    LTM is unbounded — only STM + MTM contribute.
    Returns a value in [0, 1].
    """
    total_cap = stm_capacity + mtm_capacity
    if total_cap <= 0:
        return 0.0
    used = counts.get("STM", 0) + counts.get("MTM", 0)
    return _clamp(used / total_cap)
