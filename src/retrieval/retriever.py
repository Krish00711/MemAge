"""Stage-1 coarse retrieval for MemAge.

This module provides :class:`CoarseRetriever`, a thin abstraction over
the ``store.query(text, k, tier)`` contract defined in ``docs/interfaces.md``.

It intentionally performs **no** reranking, LLM calls, or ChromaDB access —
it forwards the query to the injected store and returns
``CandidateMemoryItem`` objects unchanged.
"""

from __future__ import annotations

from typing import Any, Protocol, TypedDict


class CandidateMemoryItem(TypedDict, total=False):
    """Memory item as defined in docs/interfaces.md."""

    id: str
    text: str
    timestamp: str
    source: str
    embedding: list[float] | None  # type: ignore[valid-type]


class StoreProtocol(Protocol):
    """Minimal protocol for the tiered memory store.

    Any store-like object that implements this method is compatible
    with :class:`CoarseRetriever` via dependency injection.
    """

    def query(self, text: str, k: int, tier: str) -> list[CandidateMemoryItem]:  # type: ignore[valid-type]
        ...


_ALLOWED_TIERS: frozenset[str] = frozenset({"STM", "MTM", "LTM", "all"})


class CoarseRetriever:
    """Stage-1 semantic/vector candidate generator.

    Wraps the ``store.query(...)`` contract with validation and a clean
    retrieval interface.  No reranking, text mutation, or LLM calls are
    performed.

    Args:
        store: An object providing ``query(text, k, tier) ->
            list[CandidateMemoryItem]``.  Injected via constructor so that
            tests can supply a fake and production can supply the real
            tiered store without importing ChromaDB here.
    """

    def __init__(self, store: Any) -> None:
        if not hasattr(store, "query") or not callable(getattr(store, "query")):
            raise TypeError("store must provide a callable query(text, k, tier) method")
        self._store = store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int,
        tier: str = "all",
    ) -> list[CandidateMemoryItem]:
        """Retrieve *k* candidate memories for *query* from *tier*.

        Args:
            query: Natural-language query string.
            k: Number of candidates to request. Must be a positive integer.
            tier: One of ``"STM"``, ``"MTM"``, ``"LTM"``, ``"all"``.

        Returns:
            List of ``CandidateMemoryItem`` dicts returned by the store,
            **unchanged** and in the same order.

        Raises:
            ValueError: If *tier* is not allowed or *k* is not a positive int.
        """
        self._validate_k(k)
        self._validate_tier(tier)
        # Forward exactly as requested — no modification or reranking.
        return self._store.query(text=query, k=k, tier=tier)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_tier(tier: str) -> None:
        """Validate tier argument."""
        if tier not in _ALLOWED_TIERS:
            raise ValueError(
                f"Invalid tier '{tier}'. Allowed values are: {sorted(_ALLOWED_TIERS)}"
            )

    @staticmethod
    def _validate_k(k: int) -> None:
        """Validate k argument."""
        if not isinstance(k, int) or isinstance(k, bool):
            raise ValueError(f"k must be a positive integer, got {k!r}")
        if k <= 0:
            raise ValueError(f"k must be a positive integer (> 0), got {k}")
