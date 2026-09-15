"""Deterministic formatting for MemAge retrieval context.

Produces the prompt-ready string that is inserted into the final agent
prompt.  Each memory includes timestamp, source, and text exactly as stored;
no hallucination, rewriting, or score leakage.
"""

from __future__ import annotations

from typing import List


def _extract_candidate(item) -> dict:
    """Extract underlying ``CandidateMemoryItem`` dict from wrapper or dict.

    Supports both ``RankedCandidate`` (has ``.candidate``) and plain dict.
    Does not mutate the input.
    """
    # Check for RankedCandidate-like object (has 'candidate' attribute)
    if hasattr(item, "candidate") and isinstance(getattr(item, "candidate"), dict):
        # RankedCandidate case
        return getattr(item, "candidate")  # type: ignore[return-value]
    if isinstance(item, dict):
        # Support both CandidateMemoryItem and {"candidate": ..., "score": ...} dict style
        # But prefer direct dict
        return item
    # Fallback: try dict-like access
    return item  # type: ignore[return-value]


def format_memory(item, index: int) -> str:
    """Format a single memory into a stable block.

    Args:
        item: ``CandidateMemoryItem`` dict or ``RankedCandidate`` wrapper.
        index: 1-based memory number used in header ``[Memory {index}]``.

    Returns:
        Formatted block string for this memory. Always includes
        Timestamp, Source, Content labels even if fields are empty.
    """
    candidate = _extract_candidate(item)
    # Preserve exact values; default to "" if missing
    text = candidate.get("text", "") if isinstance(candidate, dict) else ""
    timestamp = candidate.get("timestamp", "") if isinstance(candidate, dict) else ""
    source = candidate.get("source", "") if isinstance(candidate, dict) else ""
    # Ensure we preserve exactly as stored (including empty strings)
    # Convert None to "" for formatting safety
    if text is None:
        text = ""
    if timestamp is None:
        timestamp = ""
    if source is None:
        source = ""
    lines = [
        f"[Memory {index}]",
        f"Timestamp: {timestamp}",
        f"Source: {source}",
        f"Content: {text}",
    ]
    return "\n".join(lines)


def format_context(items: List) -> str:
    """Format a list of selected memories into final context string.

    Args:
        items: List of ``CandidateMemoryItem`` dicts or ``RankedCandidate``
            wrappers, already ordered by relevance / budget selection.

    Returns:
        Deterministic context string. Memories are joined by ``\\n\\n``.
        Empty input returns ``""`` (stable empty-context representation).
        Ordering is preserved, text/timestamp/source are not rewritten,
        and no reranking scores are included.
    """
    if not items:
        return ""
    blocks: List[str] = []
    for idx, item in enumerate(items, start=1):
        blocks.append(format_memory(item, idx))
    return "\n\n".join(blocks)
