"""MemAge conversational memory agent.

Public entry point:

    agent = MemAgeAgent(store, llm=None, token_budget=500, k=10)
    response = agent.chat(message, session_id="default")  -> AgentResponse

Wires together the full pipeline:
    user message
      -> RetrievalContext (CoarseRetriever + LLMReranker + budget + formatter)
      -> answer generation via LLM + retrieved context
      -> run_turn() to ingest the new turn into memory

All heavy dependencies (store, LLM) are injected.  Unit tests need no API
keys and no ChromaDB.

Owner: Krish.  Interface contract: docs/interfaces.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from src.pipeline import TurnResult


# ---------------------------------------------------------------------------
# Answer-generation prompt
# ---------------------------------------------------------------------------

_ANSWER_SYSTEM = (
    "You are a helpful assistant with access to a long-term memory store. "
    "Answer the user's question using ONLY the information in the supplied "
    "memories. If the memories do not contain the answer, say so clearly — "
    "do NOT invent facts."
)

_ANSWER_PROMPT_WITH_CONTEXT = """\
Relevant memories:
{context}

User: {message}
Answer:"""

_ANSWER_PROMPT_NO_CONTEXT = """\
No relevant memories were found for this question.

User: {message}
Answer (be honest that you don't have stored information about this):"""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    """Full result of one agent.chat() call."""
    message: str                # original user message
    session_id: str
    answer: str                 # generated answer (or error message)
    context: str                # formatted memory context used (may be "")
    retrieved_ids: list[str]    # IDs of memories that were in the context
    turn_result: TurnResult | None  # ingestion result (None on ingestion error)
    ok: bool                    # False if answer generation itself raised
    error: str | None = None    # exception message when ok=False


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MemAgeAgent:
    """Conversational memory agent.

    Parameters
    ----------
    store:
        Any object satisfying the TieredMemoryStore interface:
        ``query``, ``count``, ``add``, ``update``, ``delete``, ``get``.
    llm:
        Injectable LLM callable ``(prompt, system=None, **kwargs) -> str``.
        Defaults to ``src.llm.client.call_llm`` when None.
    token_budget:
        Maximum estimated tokens for the retrieved context injected into the
        answer prompt.  Forwarded to RetrievalContext.
    k:
        Number of Stage-1 candidates to retrieve.
    retrieval_k:
        Number of candidates passed to CoarseRetriever (defaults to ``k``).
    thresholds:
        Controller threshold overrides forwarded to run_turn().
    pipeline_k:
        Number of nearest-neighbour matches used during ingestion (default 5).

    Example
    -------
    >>> agent = MemAgeAgent(store, llm=fake_llm)
    >>> resp = agent.chat("What city do I live in?", session_id="s1")
    >>> print(resp.answer)
    """

    def __init__(
        self,
        store: Any,
        llm: Callable | None = None,
        token_budget: int = 500,
        k: int = 10,
        thresholds: dict[str, float] | None = None,
        pipeline_k: int = 5,
    ) -> None:
        if store is None:
            raise ValueError("store must be provided.")
        if not isinstance(token_budget, int) or token_budget <= 0:
            raise ValueError("token_budget must be a positive integer.")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")

        self._store = store
        self._llm = llm  # None → resolved lazily via src.llm.client.call_llm
        self._token_budget = token_budget
        self._k = k
        self._thresholds = thresholds
        self._pipeline_k = pipeline_k

        # Build the retrieval pipeline using existing modules — no new logic.
        from src.retrieval.retriever import CoarseRetriever
        from src.retrieval.reranker import LLMReranker
        from src.retrieval.context import RetrievalContext

        self._retriever = CoarseRetriever(store)
        self._reranker = LLMReranker(self._effective_llm)
        self._retrieval_ctx = RetrievalContext(
            retriever=self._retriever,
            reranker=self._reranker,
            k=k,
            tier="all",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, message: str, session_id: str = "default") -> AgentResponse:
        """Process one user message end-to-end.

        Steps:
        1. Retrieve relevant memories via the two-stage retrieval pipeline.
        2. Generate an answer grounded in those memories (or admit absence).
        3. Ingest the new turn through run_turn() to update memory.

        Parameters
        ----------
        message:
            The user's natural-language message.
        session_id:
            Logical session identifier used as the ``source`` metadata field
            when ingesting the new turn.

        Returns
        -------
        AgentResponse
            Always returns a well-formed response — LLM errors are captured
            in ``ok=False`` / ``error`` rather than raised.
        """
        # ------------------------------------------------------------------
        # Step 1 — Retrieve relevant memories
        # ------------------------------------------------------------------
        context = ""
        retrieved_ids: list[str] = []
        try:
            context, retrieved_ids = self._retrieval_ctx.retrieve_ranked(
                message, self._token_budget
            )
        except Exception:
            # Retrieval failure is non-fatal — proceed with empty context.
            context = ""
            retrieved_ids = []

        # ------------------------------------------------------------------
        # Step 2 — Generate answer
        # ------------------------------------------------------------------
        answer = ""
        answer_ok = True
        answer_error: str | None = None
        try:
            answer = self._generate_answer(message, context)
        except Exception as exc:
            answer_ok = False
            answer_error = str(exc)
            answer = (
                "I encountered an error generating a response. "
                f"Details: {exc}"
            )

        # ------------------------------------------------------------------
        # Step 3 — Ingest the new turn into memory
        # ------------------------------------------------------------------
        from src.pipeline import run_turn

        turn_result: TurnResult | None = None
        try:
            from datetime import datetime, timezone
            ts = datetime.now(timezone.utc).isoformat()
            turn_result = run_turn(
                turn=message,
                metadata={"source": session_id, "timestamp": ts},
                store=self._store,
                llm=self._llm,
                k=self._pipeline_k,
                thresholds=self._thresholds,
            )
        except Exception:
            # Ingestion failure must not prevent returning the answer.
            pass

        return AgentResponse(
            message=message,
            session_id=session_id,
            answer=answer,
            context=context,
            retrieved_ids=retrieved_ids,
            turn_result=turn_result,
            ok=answer_ok,
            error=answer_error,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _effective_llm(self) -> Callable:
        """Return the injectable LLM or fall back to the real client."""
        if self._llm is not None:
            return self._llm
        from src.llm.client import call_llm
        return call_llm

    def _generate_answer(self, message: str, context: str) -> str:
        """Call the LLM to answer *message* given *context*.

        Uses the no-context prompt when context is empty, so the model is
        explicitly told it has no stored information rather than hallucinating.
        """
        llm = self._effective_llm
        if context.strip():
            prompt = _ANSWER_PROMPT_WITH_CONTEXT.format(
                context=context, message=message
            )
        else:
            prompt = _ANSWER_PROMPT_NO_CONTEXT.format(message=message)
        return llm(prompt, system=_ANSWER_SYSTEM)
