"""Final retrieval composition for MemAge.

Orchestrates the two-stage pipeline:

    query
      -> CoarseRetriever.retrieve(...)
      -> LLMReranker.rerank(...)
      -> token-budget selection
      -> formatter
      -> final context string

Uses dependency injection; no global singletons, no concrete LLM or vector-store imports.
"""

from __future__ import annotations

from typing import Any, Callable, List


class RetrievalContext:
    """Compose Stage-1, Stage-2, budget and formatting into final context.

    Args:
        retriever: Injected ``CoarseRetriever`` instance (or any object with
            ``retrieve(query, k, tier)``).
        reranker: Injected ``LLMReranker`` instance (or any object with
            ``rerank(query, candidates)``).
        k: Number of candidates to request from Stage-1 (default 10).
        tier: Tier passed to ``retriever.retrieve`` (default ``"all"``).
        selector: Callable ``(ranked, token_budget) -> selected``. Defaults to
            ``src.retrieval.token_budget.select_by_token_budget``.
        formatter_fn: Callable ``(selected) -> str``. Defaults to
            ``src.retrieval.formatter.format_context``.

    Example:
        >>> ctx = RetrievalContext(retriever, reranker, k=10, tier="all")
        >>> prompt_context = ctx.retrieve_context("what is user's preference?", token_budget=500)
    """

    def __init__(
        self,
        retriever: Any,
        reranker: Any,
        k: int = 10,
        tier: str = "all",
        selector: Callable[[List, int], List] | None = None,
        formatter_fn: Callable[[List], str] | None = None,
    ) -> None:
        if not hasattr(retriever, "retrieve") or not callable(getattr(retriever, "retrieve")):
            raise TypeError("retriever must provide retrieve(query, k, tier) method")
        if not hasattr(reranker, "rerank") or not callable(getattr(reranker, "rerank")):
            raise TypeError("reranker must provide rerank(query, candidates) method")
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        if tier not in {"STM", "MTM", "LTM", "all"}:
            raise ValueError(f"Invalid tier '{tier}'. Allowed: STM, MTM, LTM, all")

        self.retriever = retriever
        self.reranker = reranker
        self.k = k
        self.tier = tier

        if selector is None:
            from src.retrieval.token_budget import select_by_token_budget

            self.selector = select_by_token_budget
        else:
            if not callable(selector):
                raise TypeError("selector must be callable")
            self.selector = selector

        if formatter_fn is None:
            from src.retrieval.formatter import format_context

            self.formatter_fn = format_context
        else:
            if not callable(formatter_fn):
                raise TypeError("formatter_fn must be callable")
            self.formatter_fn = formatter_fn

    def retrieve_context(self, query: str, token_budget: int) -> str:
        """Produce final formatted context for *query* within *token_budget*.

        Args:
            query: Natural-language query string.
            token_budget: Maximum estimated tokens for the final context.
                Must be a positive integer (validated by selector).

        Returns:
            Formatted context string ready to drop into an LLM prompt.
            Empty string if no memories were retrieved or none fit the budget.

        Raises:
            ValueError: If ``token_budget`` is not a positive integer.
        """
        if not isinstance(token_budget, int) or isinstance(token_budget, bool):
            raise ValueError(f"token_budget must be a positive integer, got {token_budget!r}")
        if token_budget <= 0:
            raise ValueError(f"token_budget must be > 0, got {token_budget}")

        # Stage-1: coarse retrieval
        candidates = self.retriever.retrieve(query, k=self.k, tier=self.tier)

        # Stage-2: LLM-guided reranking
        ranked = self.reranker.rerank(query, candidates)

        # Budget selection (selector handles its own token_budget validation,
        # but we validated above as well for immediate error)
        selected = self.selector(ranked, token_budget)

        # Formatting
        context = self.formatter_fn(selected)

        return context

    # Make instance callable as alternative API: instance(query, token_budget)
    def __call__(self, query: str, token_budget: int) -> str:
        return self.retrieve_context(query, token_budget)
