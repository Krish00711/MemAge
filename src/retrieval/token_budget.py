"""Token-budget selection for MemAge retrieval.

Provides deterministic, whitespace-based token estimation and a greedy
skip-if-too-large selector that preserves ranking order while maximizing
useful context under a fixed budget.

The estimator is intentionally an approximation (whitespace split) and is
documented as such; the invariant is that ``estimate_tokens(format_context(selected))
<= token_budget`` using the same estimator throughout this module.
"""

from __future__ import annotations

from typing import List


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Deterministic whitespace-based token approximation.

    This is NOT an exact model tokenizer count. It counts whitespace-separated
    tokens via ``str.split()`` which is deterministic and requires no external
    dependency.

    Args:
        text: Input string to estimate.

    Returns:
        Approximate token count (number of whitespace-separated tokens).
        Empty or whitespace-only strings return 0.
    """
    if not text or not text.strip():
        return 0
    return len(text.strip().split())


def _extract_candidate(item) -> dict:
    """Extract underlying CandidateMemoryItem dict without mutation."""
    if hasattr(item, "candidate") and isinstance(getattr(item, "candidate"), dict):
        return getattr(item, "candidate")  # type: ignore[return-value]
    return item  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_by_token_budget(
    ranked: List,
    token_budget: int,
) -> List:
    """Select ranked memories that fit within ``token_budget``.

    Greedy skip-if-too-large behavior:

    * Iterate ranked candidates in supplied order (highest rank first).
    * For each candidate, check whether adding its formatted block to the
      *currently selected* context would keep the total within budget
      (including formatting overhead).
    * If it fits, select it; if not, skip it and continue to lower-ranked
      candidates.
    * Result preserves ranking order (no reordering).

    This allows lower-ranked memories to be selected when a higher-ranked
    memory is too large, e.g. ``A (too large), B (fits), C (fits) -> B, C``.

    Args:
        ranked: List of ``RankedCandidate`` or ``CandidateMemoryItem`` dicts,
            ordered by descending relevance (as produced by ``LLMReranker``).
        token_budget: Maximum allowed estimated tokens for the final formatted
            context. Must be a positive integer.

    Returns:
        Selected sub-list, preserving ranking order, with
        ``estimate_tokens(format_context(selected)) <= token_budget``.
        Empty list if no candidate fits or input is empty.

    Raises:
        ValueError: If ``token_budget`` is not a positive integer.
    """
    # Validate budget
    if not isinstance(token_budget, int) or isinstance(token_budget, bool):
        raise ValueError(f"token_budget must be a positive integer, got {token_budget!r}")
    if token_budget <= 0:
        raise ValueError(f"token_budget must be > 0, got {token_budget}")

    if not ranked:
        return []

    # Local import to avoid circular dependency at module load time
    from src.retrieval.formatter import format_context

    selected: List = []

    for item in ranked:
        # Tentatively add item and check total token count
        tentative = selected + [item]
        formatted = format_context(tentative)
        tokens = estimate_tokens(formatted)
        if tokens <= token_budget:
            selected.append(item)
        else:
            # Skip this candidate, continue checking lower-ranked ones
            continue

    # Invariant: selected is in original ranking order because we iterated
    # in order and appended only when fitting, never reordering.
    return selected


# Alias for convenience (both names refer to same implementation)
select = select_by_token_budget
