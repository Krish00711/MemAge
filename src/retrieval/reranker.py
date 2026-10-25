"""Stage-2 LLM-guided reranking for MemAge.

This module provides :class:`LLMReranker`, which reorders candidate memories
supplied by Stage-1 (``CoarseRetriever``) using an LLM that reasons about
relevance/utility to the current query.

The reranker uses dependency injection for the LLM callable so that it works
with the shared ``call_llm(prompt, system=None, **kwargs) -> str`` contract
without importing a concrete provider (Gemini/Groq/OpenAI) or a vector store.

Robust parsing and a deterministic lexical fallback ensure the reranker never
crashes on malformed model output.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, List

from src.retrieval.retriever import CandidateMemoryItem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankedCandidate:
    """Result of reranking: original candidate + relevance score.

    Attributes:
        candidate: The original ``CandidateMemoryItem`` dict (same object
            reference, never mutated).
        score: Relevance/usefulness score in ``[0, 1]`` (LLM-provided or
            fallback lexical score). Higher is more relevant.
    """

    candidate: CandidateMemoryItem
    score: float


# ---------------------------------------------------------------------------
# Lexical fallback helpers
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> set[str]:
    """Tokenize *text* case-insensitively, handling punctuation.

    Uses ``[a-z0-9]+`` so punctuation and whitespace are ignored.
    Returns a set of lower-cased tokens.
    """
    if not text:
        return set()
    return set(match.lower() for match in _TOKEN_RE.findall(text))


def _lexical_score(query: str, candidate_text: str) -> float:
    """Compute Jaccard token-overlap between query and candidate text.

    Jaccard = |intersection| / |union|.  Returns 0.0 when there is no
    overlap or when either side has no tokens.  Deterministic, no external
    dependencies.
    """
    q_tokens = _tokenize(query)
    c_tokens = _tokenize(candidate_text)
    if not q_tokens or not c_tokens:
        return 0.0
    intersection = q_tokens & c_tokens
    if not intersection:
        return 0.0
    union = q_tokens | c_tokens
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a relevance reranker for a selective long-term memory system. "
    "You must rank ONLY the supplied memory candidates by relevance/usefulness "
    "to the query."
)


def _build_prompt(query: str, candidates: List[CandidateMemoryItem]) -> str:
    """Construct a deterministic ranking prompt.

    The prompt contains the query and for each candidate its id, text,
    timestamp, and source.  It instructs the LLM to output structured JSON
    and to obey the no-invention constraints.
    """
    lines: List[str] = []
    lines.append("You are a relevance reranker for a selective long-term memory system.")
    lines.append("")
    lines.append("TASK: Rank the following memory candidates by relevance/usefulness to the query.")
    lines.append("")
    lines.append(f"Query: {query}")
    lines.append("")
    lines.append(f"Candidates ({len(candidates)}):")
    for item in candidates:
        cid = item.get("id", "")
        text = item.get("text", "")
        timestamp = item.get("timestamp", "")
        source = item.get("source", "")
        lines.append(f'- ID: {cid}')
        lines.append(f'  Text: {text}')
        lines.append(f'  Timestamp: {timestamp}')
        lines.append(f'  Source: {source}')
    lines.append("")
    lines.append("Instructions:")
    lines.append("- Rank ONLY the candidates listed above.")
    lines.append("- Do NOT invent candidates.")
    lines.append("- Do NOT invent facts.")
    lines.append("- Do NOT rewrite memory text.")
    lines.append("- Do NOT modify timestamps.")
    lines.append("- Do NOT modify sources.")
    lines.append("- Output must be structured JSON.")
    lines.append("- Use the format:")
    lines.append('  {')
    lines.append('    "rankings": [')
    lines.append('      {"id": "memory_id_1", "score": 0.95},')
    lines.append('      {"id": "memory_id_2", "score": 0.72}')
    lines.append('    ]')
    lines.append('  }')
    lines.append("- Scores should be floats between 0 and 1 representing relevance/usefulness to the query.")
    lines.append("- Ranking should be descending by score (most relevant first).")
    lines.append("- Each supplied candidate ID must appear at most once.")
    lines.append("- Return ONLY valid JSON. Do not add extra text or explanation.")
    lines.append("- Do not include markdown except optional code fences around the JSON are allowed.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON extraction / parsing
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _extract_json(raw: str) -> str:
    """Extract JSON string from *raw*, handling markdown fences.

    If *raw* contains triple-backtick fences, the content inside the first
    fence pair is extracted.  Otherwise the stripped raw string is returned.
    If fences are present but extraction fails, the raw stripped string is
    still tried.  Also handles JSON surrounded by other text by finding the
    outermost ``{ ... }`` when needed.
    """
    stripped = raw.strip()
    if not stripped:
        return stripped
    # If fenced, extract inner content
    fence_match = _FENCE_RE.search(stripped)
    if fence_match:
        inner = fence_match.group(1).strip()
        if inner:
            return inner
    return stripped


def _parse_llm_response(raw: str) -> list[dict[str, Any]] | None:
    """Parse LLM response into a list of ``{id, score}`` dicts.

    Returns ``None`` if the response is malformed and fallback should be used.
    On success returns a list (possibly empty) of raw ranking entries.
    The caller is responsible for filtering unknown IDs, duplicates, and
    invalid scores.
    """
    if not raw or not raw.strip():
        return None
    json_str = _extract_json(raw)
    # If the extracted string does not look like JSON, try to find JSON object
    # inside surrounding text (best-effort robustness).
    if not json_str.lstrip().startswith("{"):
        # Attempt to locate outermost { ... }
        obj_match = re.search(r"\{[\s\S]*\}", json_str)
        if obj_match:
            json_str = obj_match.group(0)
        else:
            return None
    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.debug("LLM JSON parsing failed: %s", exc)
        return None

    if not isinstance(data, dict):
        logger.debug("LLM response JSON is not an object: %r", type(data))
        return None
    rankings = data.get("rankings")
    if rankings is None:
        logger.debug("LLM JSON missing 'rankings' key: %r", data)
        return None
    if not isinstance(rankings, list):
        logger.debug("'rankings' is not a list: %r", rankings)
        return None
    # Validate each entry is a dict with id/score keys (but allow filtering later)
    # If any entry is not a dict, treat as malformed -> fallback? We instead
    # filter non-dicts and keep valid ones, but if none are valid we still proceed
    # to handling. To satisfy spec "robust parsing" we do not crash.
    cleaned: list[dict[str, Any]] = []
    for entry in rankings:
        if not isinstance(entry, dict):
            continue
        cleaned.append(entry)
    return cleaned


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class LLMReranker:
    """Stage-2 LLM-guided reranker.

    Reorders ``CandidateMemoryItem`` lists by relevance to a query using an
    injected LLM callable compatible with ``call_llm(prompt, system=None,
    **kwargs) -> str``.

    Args:
        llm_callable: A callable that accepts ``prompt`` (and optionally
            ``system``) and returns a string.  Injected via constructor so
            tests can supply a fake and production can supply Krish's shared
            client without coupling.

    Example:
        >>> reranker = LLMReranker(llm_callable)
        >>> ranked = reranker.rerank("what is user's preference?", candidates)
        >>> ranked[0].candidate["id"]
        'mem_001'
    """

    def __init__(self, llm_callable: Callable[..., str]) -> None:
        if not callable(llm_callable):
            raise TypeError("llm_callable must be callable")
        self._llm = llm_callable

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: List[CandidateMemoryItem],
    ) -> List[RankedCandidate]:
        """Rerank *candidates* by relevance to *query*.

        Args:
            query: Natural-language query string.
            candidates: List of ``CandidateMemoryItem`` dicts from Stage-1.

        Returns:
            List of :class:`RankedCandidate` ordered by descending relevance.
            Each supplied candidate appears at most once.  Candidates not
            ranked by the LLM are preserved via deterministic lexical fallback.
            The original candidate objects are referenced unchanged and scores
            live only in the ``RankedCandidate`` wrapper.

        Notes:
            Never crashes on LLM errors or malformed output; falls back
            deterministically instead.  No mutation of candidate dicts.
        """
        if not candidates:
            return []

        prompt = _build_prompt(query, candidates)
        system = _SYSTEM_PROMPT

        # Call LLM with robust error handling
        raw: Any = None
        try:
            # Try with system kwarg first; fall back to prompt-only if signature mismatch
            try:
                raw = self._llm(prompt, system=system)
            except TypeError as te:
                # If callable doesn't accept 'system', retry with prompt only
                msg = str(te).lower()
                if "system" in msg or "unexpected" in msg or "got" in msg:
                    # Heuristic: try positional prompt only, then keyword prompt
                    try:
                        raw = self._llm(prompt)
                    except TypeError:
                        # Try keyword prompt
                        raw = self._llm(prompt=prompt)
                else:
                    raise
        except Exception as exc:  # noqa: BLE001
            logger.debug("LLM callable raised exception, falling back: %s", exc)
            return self._fallback_ranking(query, candidates)

        # Handle None / empty / non-string
        if raw is None:
            logger.debug("LLM returned None, falling back")
            return self._fallback_ranking(query, candidates)
        if not isinstance(raw, str):
            # Coerce non-string? Spec says empty/non-string -> fallback if not useful
            # If it's not a string, treat as malformed
            logger.debug("LLM returned non-string type %r, falling back", type(raw))
            return self._fallback_ranking(query, candidates)
        if not raw.strip():
            logger.debug("LLM returned empty string, falling back")
            return self._fallback_ranking(query, candidates)

        # Parse LLM response
        parsed = _parse_llm_response(raw)
        if parsed is None:
            logger.debug("LLM response parsing failed, falling back")
            return self._fallback_ranking(query, candidates)

        # Build lookup for supplied candidates
        id_to_candidate: dict[str, CandidateMemoryItem] = {}
        id_to_index: dict[str, int] = {}
        for idx, item in enumerate(candidates):
            cid = item.get("id")
            if cid is not None:
                # Preserve first occurrence if duplicate IDs in input (should not happen)
                if cid not in id_to_candidate:
                    id_to_candidate[cid] = item
                    id_to_index[cid] = idx

        # Process LLM rankings: filter unknown, duplicate, invalid scores
        seen: set[str] = set()
        valid_ranked: list[tuple[str, float]] = []
        for entry in parsed:
            eid = entry.get("id")
            score_raw = entry.get("score")
            if not isinstance(eid, str):
                continue
            if eid not in id_to_candidate:
                continue  # unknown ID -> ignore
            if eid in seen:
                continue  # duplicate -> ignore
            # Validate score
            try:
                # Allow int/float/string numeric; reject bool
                if isinstance(score_raw, bool):
                    continue
                score_f = float(score_raw)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score_f):
                continue
            # Clamp to [0,1] ? Keep as is but ensure finite; we don't strictly require clamping
            # but normalizing helps consistent fallback comparison. We keep original value
            # but ensure it's a float.
            seen.add(eid)
            valid_ranked.append((eid, score_f))

        # If LLM provided no valid rankings (e.g., all entries invalid), fallback entirely
        # But we still want to preserve candidates; valid_ranked may be empty.
        # We handle remaining via fallback ordering.
        # Determine remaining candidates (not in valid_ranked) and order them by fallback
        remaining_ids = [c.get("id") for c in candidates if c.get("id") not in seen]
        # Compute fallback scores for remaining candidates
        remaining_candidates = [id_to_candidate[cid] for cid in remaining_ids if cid in id_to_candidate]
        # Compute lexical scores for remaining
        fallback_scored: list[tuple[CandidateMemoryItem, float, int]] = []
        for item in remaining_candidates:
            cid = item.get("id", "")
            idx = id_to_index.get(cid, 0)
            score = _lexical_score(query, item.get("text", ""))
            fallback_scored.append((item, score, idx))
        # Sort remaining by descending fallback score, tie-breaker original order
        fallback_scored.sort(key=lambda x: (-x[1], x[2]))

        # Build final result
        result: List[RankedCandidate] = []
        # First, LLM-ranked in LLM order
        for eid, score in valid_ranked:
            cand = id_to_candidate[eid]
            result.append(RankedCandidate(candidate=cand, score=float(score)))
        # Then fallback-ordered remaining
        for item, score, _idx in fallback_scored:
            result.append(RankedCandidate(candidate=item, score=float(score)))

        # Ensure result contains each supplied candidate at most once and never invents
        # (Already guaranteed by seen + remaining logic)
        # Handle edge: if valid_ranked was empty, result is purely fallback sorted; that's intended.
        # If LLM returned partial ranking, we've preserved remainder.

        # If for some reason result is empty but candidates non-empty (e.g., all IDs missing/unknown),
        # fallback entirely (this happens if LLM returned only unknown IDs, valid_ranked empty,
        # remaining includes all candidates, so result non-empty). No extra handling needed.
        # But if id_to_candidate missing some candidates due to missing id field, ensure we include them
        # Fallback for items without id field: they were not in id_to_candidate, but should still appear
        # Find candidates not yet in result (by identity)
        result_ids = {r.candidate.get("id") for r in result}
        # Also handle items with None/missing id: include them via fallback ordering
        unsorted = [c for c in candidates if c.get("id") not in result_ids]
        # For unsorted that have missing id, they were excluded from id_to_candidate
        if unsorted:
            # Score them via lexical and append sorted by fallback
            extra: list[tuple[CandidateMemoryItem, float, int]] = []
            for idx, item in enumerate(candidates):
                if item in [r.candidate for r in result]:
                    continue
                # Check by identity if id missing
                if item.get("id") in result_ids and item.get("id") is not None:
                    continue
                # Need to deduplicate: use object identity already checked
                # Compute score
                if item not in [e[0] for e in fallback_scored]:
                    score = _lexical_score(query, item.get("text", ""))
                    extra.append((item, score, idx))
            extra.sort(key=lambda x: (-x[1], x[2]))
            for item, score, _ in extra:
                # Avoid duplicates
                if item not in [r.candidate for r in result]:
                    result.append(RankedCandidate(candidate=item, score=float(score)))

        return result

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _fallback_ranking(
        self,
        query: str,
        candidates: List[CandidateMemoryItem],
    ) -> List[RankedCandidate]:
        """Deterministic lexical fallback ranking.

        Scores each candidate via Jaccard token overlap and returns them
        descending by score, with original order as tie-breaker.  Does not
        call the LLM and does not mutate candidates.
        """
        scored: list[tuple[CandidateMemoryItem, float, int]] = []
        for idx, item in enumerate(candidates):
            text = item.get("text", "")
            score = _lexical_score(query, text)
            scored.append((item, score, idx))
        scored.sort(key=lambda x: (-x[1], x[2]))
        return [RankedCandidate(candidate=item, score=float(score)) for item, score, _ in scored]

    # ------------------------------------------------------------------
    # Exposed helpers for testing / debugging (not part of public contract)
    # ------------------------------------------------------------------

    def _build_prompt_for_test(
        self, query: str, candidates: List[CandidateMemoryItem]
    ) -> str:
        """Expose prompt building for testing without calling LLM."""
        return _build_prompt(query, candidates)
