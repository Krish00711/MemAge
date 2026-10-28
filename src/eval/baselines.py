"""Baselines for MemAge evaluation: full-history upper bound, MemoryOS-style tiered baseline, Memory-R1-style budgeted baseline, H-MEM-style hierarchical baseline.

All four plug into the existing ``BenchmarkRunner`` as a ``retrieval_fn``:

    runner = BenchmarkRunner(
        retrieval_fn=FullHistoryBaseline(history),    # or MemoryOSBaseline /
        answer_fn=answer_fn,                          # MemoryR1Baseline /
    )                                                 # HMemBaseline

Shared design:
    * Dependency-injected (history supplied by the caller), deterministic,
      no store / no LLM / no network.
    * History may be a plain string or a list of turn/item dicts. Dicts may
      use either schema seen in this project:
        - memory-style: ``text`` / ``timestamp`` / ``source`` (or ``id``)
        - chat-style:   ``content`` / ``role`` (LongMemEval turns)
      Entries are normalized and formatted with the existing deterministic
      ``src.retrieval.formatter.format_context`` (same block layout as the
      retrieval pipeline), so answer functions and token accounting behave
      identically across baselines.
    * QA metrics (F1 / BLEU-1), latency and token usage are measured by the
      framework exactly as for any other strategy.
    * Selection semantics differ per baseline: ``full_history`` selects
      nothing (no budget), ``memoryos`` selects top-``k`` ignoring the budget,
      ``memory_r1`` enforces the injected ``token_budget`` via the
      repository's budget selector, and ``hmem`` orders candidates
      hierarchically (tier-first coarse-to-fine) before the same budget
      selector runs.

FullHistoryBaseline (:class:`FullHistoryBaseline`):
    Returns the ENTIRE history for every query - no retrieval, no ranking.
    Returns ``retrieved_ids=None`` so retrieval metrics are ``None``.

MemoryOS-style tiered baseline (:class:`MemoryOSBaseline`):
    See :class:`MemoryOSBaseline` for the explicit approximation scope.

Memory-R1-style budgeted baseline (:class:`MemoryR1Baseline`):
    See :class:`MemoryR1Baseline` for the explicit approximation scope.

H-MEM-style hierarchical baseline (:class:`HMemBaseline`):
    See :class:`HMemBaseline` for the explicit approximation scope.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

from src.eval.benchmark import RetrievalOutcome
from src.eval.dataset import BenchmarkCase
from src.eval.metrics import tokens_returned
from src.retrieval.formatter import format_context
from src.retrieval.reranker import _lexical_score
from src.retrieval.token_budget import select_by_token_budget


def _normalize_entry(entry) -> dict:
    """Map one history entry (str or dict, either schema) to format keys."""
    if isinstance(entry, str):
        return {"text": entry, "timestamp": "", "source": ""}
    if isinstance(entry, dict):
        text = entry.get("text")
        if text is None or text == "":
            text = entry.get("content", "")
        source = (
            entry.get("source")
            or entry.get("id")
            or entry.get("speaker")
            or entry.get("role")
            or entry.get("session_id")
            or ""
        )
        timestamp = entry.get("timestamp") or entry.get("date") or ""
        return {
            "text": str(text),
            "timestamp": str(timestamp),
            "source": str(source),
        }
    return {"text": str(entry), "timestamp": "", "source": ""}


def format_full_history(history: Union[str, List, None]) -> str:
    """Format the whole history deterministically into a context string.

    Args:
        history: ``None``, a plain string (used verbatim), or a list of
            turn/item dicts or strings (normalized then formatted with the
            shared retrieval formatter).

    Returns:
        Context string (``""`` for ``None`` / empty list / empty string).
        Never raises on empty input.
    """
    if history is None:
        return ""
    if isinstance(history, str):
        return history
    if isinstance(history, (list, tuple)):
        if not history:
            return ""
        return format_context([_normalize_entry(h) for h in history])
    return str(history)


def memory_entries_from_case(case: BenchmarkCase) -> List[dict]:
    """Extract a case's history as individual memory-entry dicts.

    Unlike :func:`history_from_case` (which formats the whole history into
    one context string), this returns one dict per conversation turn so the
    entry-based baselines (``memoryos`` / ``memory_r1`` / ``hmem``) can
    rank, budget, and id-annotate entries individually.

    The original dataset ids are preserved on every entry:

    * LoCoMo: each ``metadata["turns"]`` record keeps its ``dia_id`` as
      ``id`` (e.g. ``"D1:1"``) - the same unit referenced by the case's
      ``relevant_ids`` (``qa[].evidence``).
    * LongMemEval: each flattened turn gets ``id = session_id`` - the same
      unit referenced by the case's ``relevant_ids``
      (``answer_session_ids``). Session date is attached as ``timestamp``
      and ``source`` becomes ``"<session_id>:<role>"``.

    Entries are raw (never pre-formatted with ``[Memory N]`` markers);
    formatting happens exactly once, later, via the shared
    ``src.retrieval.formatter.format_context``.

    Returns ``[]`` when the case carries no history metadata.
    """
    metadata = getattr(case, "metadata", None) or {}

    turns = metadata.get("turns")
    if isinstance(turns, (list, tuple)) and turns:
        return list(turns)

    sessions = metadata.get("sessions")
    if isinstance(sessions, (list, tuple)) and sessions:
        entries: List[dict] = []
        for session in sessions:
            if not isinstance(session, dict):
                continue
            session_date = session.get("date") or ""
            session_id = session.get("session_id") or ""
            raw_turns = session.get("turns")
            if not isinstance(raw_turns, (list, tuple)):
                continue
            for turn in raw_turns:
                if not isinstance(turn, dict):
                    continue
                entry = dict(turn)
                entry.setdefault("timestamp", session_date or "")
                if session_id:
                    # Relevance unit for LongMemEval is the session id.
                    entry.setdefault("id", session_id)
                if "role" in entry and "source" not in entry:
                    entry["source"] = (
                        f"{session_id}:{entry['role']}" if session_id else entry["role"]
                    )
                entries.append(entry)
        return entries

    return []


def history_from_case(case: BenchmarkCase) -> str:
    """Extract a case's full history from its adapter-provided metadata.

    Supports the metadata shapes produced by ``src/eval/adapters.py``:

    * LoCoMo: ``metadata["turns"]`` (conversation turns, in order)
    * LongMemEval: ``metadata["sessions"]`` (zipped sessions; session date
      is attached to each turn as a timestamp fallback)

    Returns ``""`` when the case carries no history (e.g. synthetic cases
    built without metadata), which is a valid empty history.
    """
    return format_full_history(memory_entries_from_case(case))


class FullHistoryBaseline:
    """``retrieval_fn`` that returns the entire history as context.

    Callable with the ``retrieval_fn`` contract: ``(query) -> RetrievalOutcome``.
    The context is identical for every query (no query-dependent behavior).

    Args:
        history: ``str`` or list of turn/item dicts (see
            :func:`format_full_history`). ``None``/empty means empty history.

    Example:
        >>> baseline = FullHistoryBaseline([{"text": "hello", "source": "s1"}])
        >>> outcome = baseline("any query")
        >>> outcome.retrieved_ids is None
        True
    """

    def __init__(self, history: Union[str, List, None] = None) -> None:
        self._context = format_full_history(history)

    @property
    def context(self) -> str:
        """The full (formatted) history used as context."""
        return self._context

    @property
    def tokens(self) -> int:
        """Estimated token count of the full-history context."""
        return tokens_returned(self._context)

    def __call__(self, query: str) -> RetrievalOutcome:
        """Return the full history for *query* (ignored - same context always)."""
        return RetrievalOutcome(
            context=self._context,
            retrieved_ids=None,
            tokens=tokens_returned(self._context),
        )


def full_history_retrieval_fn(history: Union[str, List, None]) -> FullHistoryBaseline:
    """Convenience factory: build the ``retrieval_fn`` for a case's history."""
    return FullHistoryBaseline(history)


# ---------------------------------------------------------------------------
# MemoryOS-style tiered baseline (project-level approximation)
# ---------------------------------------------------------------------------


def _entry_id(entry) -> Optional[str]:
    """Return the entry's id if it is a non-empty string, else ``None``."""
    if isinstance(entry, dict):
        eid = entry.get("id")
        if isinstance(eid, str) and eid:
            return eid
    return None


def _split_tiers(
    entries: List, stm_size: int
) -> Tuple[List[Tuple[int, dict]], List[Tuple[int, dict]]]:
    """Split *entries* into (LTM, STM) as (original_index, normalized) pairs.

    Tiering is purely positional/recency based: the last ``stm_size``
    entries (most recent) form the short-term tier; everything before that
    forms the long-term tier. This is the approximation - see
    :class:`MemoryOSBaseline`.
    """
    stm = [
        (idx, _normalize_entry(entry))
        for idx, entry in list(enumerate(entries))[-stm_size:]
    ]
    ltm = [
        (idx, _normalize_entry(entry))
        for idx, entry in list(enumerate(entries))[:-stm_size]
    ]
    return ltm, stm


class MemoryOSBaseline:
    """MemoryOS-style tiered-memory baseline — a PROJECT-LEVEL APPROXIMATION.

    **This is NOT a faithful reproduction of any published MemoryOS system.**
    This repository documents no MemoryOS internals (only the name appears in
    ``README.md`` as a baseline to compare against), so no paper behavior is
    reproduced and no research claims are made. What is implemented is the
    simplest defensible tiered-memory strategy using only concepts already
    present in this repository:

    * **Tiering** - ``docs/interfaces.md`` defines the store query contract
      with tiers ``"STM" | "MTM" | "LTM"``. Here history is split by recency
      into two tiers: STM = most recent ``stm_size`` entries, LTM = the rest.
      There is no eviction, no promotion/consolidation, no summarization,
      no MTM tier, and no online updates - the tier assignment is static and
      computed once from the injected history.
    * **Query-time selection** - every entry (both tiers) is scored with the
      retrieval pipeline's existing deterministic lexical scorer
      (:func:`src.retrieval.reranker._lexical_score`, Jaccard token overlap);
      the top-``k`` entries are selected, ordered by score (desc), then
      recency (newer first), then original position (stable).
    * **Formatting** - selected entries use the shared
      ``src.retrieval.formatter.format_context`` (identical ``[Memory N]``
      blocks as the real pipeline).
    * **Retrieval metrics** - ``retrieved_ids`` contains the ``id`` of each
      selected entry that has one (entries without a string id appear in the
      context but are omitted from the id list, per ``docs/interfaces.md``),
      so Recall@K / Precision@K / MRR are computed by the framework.

    Args:
        history: ``str`` or list of turn/item dicts (see
            :func:`format_full_history`). ``None``/empty means empty history.
        k: Maximum number of entries selected per query (approximation
            constant, chosen here - not taken from any external system).
        stm_size: Number of most-recent entries in the short-term tier.

    Example:
        >>> baseline = MemoryOSBaseline(
        ...     [{"id": "m1", "text": "likes tea"}, {"id": "m2", "text": "owns a bike"}],
        ...     k=1,
        ... )
        >>> outcome = baseline("tea preference")
        >>> outcome.retrieved_ids  # deterministic top-1 by lexical overlap
        ['m1']
    """

    def __init__(
        self,
        history: Union[str, List, None] = None,
        k: int = 5,
        stm_size: int = 5,
    ) -> None:
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        if not isinstance(stm_size, int) or isinstance(stm_size, bool) or stm_size <= 0:
            raise ValueError(f"stm_size must be a positive integer, got {stm_size!r}")
        self._k = k
        self._stm_size = stm_size

        if isinstance(history, str):
            entries: List = [history] if history else []
        elif history is None:
            entries = []
        elif isinstance(history, (list, tuple)):
            entries = list(history)
        else:
            entries = [history]

        self._entries = entries
        self._ltm, self._stm = _split_tiers(entries, stm_size)
        # Full-history context (same formatting as FullHistoryBaseline)
        self._full_context = format_full_history(entries)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def k(self) -> int:
        """Maximum number of entries selected per query."""
        return self._k

    @property
    def stm_size(self) -> int:
        """Configured short-term-tier size."""
        return self._stm_size

    @property
    def num_ltm(self) -> int:
        """Number of entries in the long-term tier."""
        return len(self._ltm)

    @property
    def num_stm(self) -> int:
        """Number of entries in the short-term tier."""
        return len(self._stm)

    @property
    def full_context(self) -> str:
        """The whole history, formatted (not what the baseline returns)."""
        return self._full_context

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self, query: str) -> List[Tuple[int, dict, float]]:
        """Select top-``k`` entries for *query*.

        Returns:
            List of ``(original_index, normalized_entry, score)`` tuples in
            final context order: score desc, then recency (newer/original
            index higher first), then original index asc. Deterministic;
            entries are never mutated.
        """
        scored: List[Tuple[int, dict, float]] = []
        for idx, entry in enumerate(self._entries):
            norm = _normalize_entry(entry)
            score = _lexical_score(query, norm["text"])
            scored.append((idx, norm, score))
        scored.sort(key=lambda t: (-t[2], -t[0]))
        return scored[: self._k]

    def __call__(self, query: str) -> RetrievalOutcome:
        """Return tier-selected context + ids for *query*."""
        selected = self.select(query)
        context = format_context([norm for _idx, norm, _s in selected])
        ids: List[str] = []
        for idx, _norm, _s in selected:
            eid = _entry_id(self._entries[idx])
            if eid is not None:
                ids.append(eid)
        return RetrievalOutcome(
            context=context,
            retrieved_ids=ids,
            tokens=tokens_returned(context),
        )


def memoryos_retrieval_fn(
    history: Union[str, List, None], k: int = 5, stm_size: int = 5
) -> MemoryOSBaseline:
    """Convenience factory: build the tiered ``retrieval_fn`` for a history."""
    return MemoryOSBaseline(history, k=k, stm_size=stm_size)


# ---------------------------------------------------------------------------
# Memory-R1-style budgeted ranked baseline (project-level approximation)
# ---------------------------------------------------------------------------


class MemoryR1Baseline:
    """Memory-R1-style budgeted ranked baseline — a PROJECT-LEVEL APPROXIMATION.

    **This is NOT a faithful reproduction of any published Memory-R1 system.**
    This repository documents no Memory-R1 internals (only the name appears in
    ``README.md`` as a checklist baseline), so no paper behavior is reproduced
    and no research claims are made — in particular nothing about training,
    rewards, or agent loops is implemented here. What is implemented is the
    simplest defensible budgeted-retrieval strategy using only utilities that
    already exist in this repository:

    * **Ranking** - every history entry is scored single-stage with the
      retrieval pipeline's existing deterministic lexical scorer
      (:func:`src.retrieval.reranker._lexical_score`, Jaccard token overlap);
      ordered by score (desc), then recency (newer first), then original
      position (stable). No store, no stage-1 coarse retrieval, no LLM
      reranker.
    * **Selection** - the ranked list is passed through the repository's
      existing greedy budget selector
      (:func:`src.retrieval.token_budget.select_by_token_budget`) with the
      injected ``token_budget``: highest-ranked entries are kept while
      ``estimate_tokens(format_context(selected)) <= token_budget``.
      This is the defining difference from the sibling baselines:
      ``full_history`` selects nothing, ``memoryos`` selects top-``k``
      ignoring the budget, and this baseline enforces the budget.
    * **Formatting** - shared ``src.retrieval.formatter.format_context``
      (identical ``[Memory N]`` blocks as the real pipeline).
    * **Retrieval metrics** - ``retrieved_ids`` contains the ``id`` of each
      selected entry that has one (entries without a string id appear in the
      context but are omitted from the id list, per ``docs/interfaces.md``),
      so Recall@K / Precision@K / MRR are computed by the framework.

    Args:
        history: ``str`` or list of turn/item dicts (see
            :func:`format_full_history`). ``None``/empty means empty history.
        token_budget: Positive int budget enforced by the selector
            (approximation constant supplied by the caller — the CLI forwards
            ``--token-budget``; default 512 matches ``DEFAULT_TOKEN_BUDGET``).

    Example:
        >>> baseline = MemoryR1Baseline(
        ...     [{"id": "m1", "text": "likes tea"}, {"id": "m2", "text": "owns a bike"}],
        ...     token_budget=12,
        ... )
        >>> outcome = baseline("tea preference")
        >>> outcome.retrieved_ids  # best lexical match fits the budget
        ['m1']
    """

    DEFAULT_TOKEN_BUDGET = 512

    def __init__(
        self,
        history: Union[str, List, None] = None,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> None:
        if (
            not isinstance(token_budget, int)
            or isinstance(token_budget, bool)
            or token_budget <= 0
        ):
            raise ValueError(f"token_budget must be a positive integer, got {token_budget!r}")
        self._token_budget = token_budget

        if isinstance(history, str):
            entries: List = [history] if history else []
        elif history is None:
            entries = []
        elif isinstance(history, (list, tuple)):
            entries = list(history)
        else:
            entries = [history]
        self._entries = entries

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def token_budget(self) -> int:
        """Budget enforced by the selection stage."""
        return self._token_budget

    @property
    def num_entries(self) -> int:
        """Number of history entries available for ranking."""
        return len(self._entries)

    # ------------------------------------------------------------------
    # Ranking + budgeted selection
    # ------------------------------------------------------------------

    def rank(self, query: str) -> List[Tuple[int, dict, float]]:
        """Rank ALL entries for *query* (score desc, recency desc, index).

        Deterministic; entries are never mutated.
        """
        scored: List[Tuple[int, dict, float]] = []
        for idx, entry in enumerate(self._entries):
            norm = _normalize_entry(entry)
            score = _lexical_score(query, norm["text"])
            scored.append((idx, norm, score))
        scored.sort(key=lambda t: (-t[2], -t[0]))
        return scored

    def __call__(self, query: str) -> RetrievalOutcome:
        """Rank the history, keep what fits the budget, return context + ids."""
        ranked = self.rank(query)
        selected = select_by_token_budget(
            [norm for _idx, norm, _s in ranked], self._token_budget
        )
        selected_set = {id(norm) for norm in selected}
        ids: List[str] = []
        for idx, norm, _s in ranked:
            if id(norm) not in selected_set:
                continue
            eid = _entry_id(self._entries[idx])
            if eid is not None:
                ids.append(eid)
        context = format_context(selected)
        return RetrievalOutcome(
            context=context,
            retrieved_ids=ids,
            tokens=tokens_returned(context),
        )


def memory_r1_retrieval_fn(
    history: Union[str, List, None], token_budget: int = MemoryR1Baseline.DEFAULT_TOKEN_BUDGET
) -> MemoryR1Baseline:
    """Convenience factory: build the budgeted ``retrieval_fn`` for a history."""
    return MemoryR1Baseline(history, token_budget=token_budget)


# ---------------------------------------------------------------------------
# H-MEM-style hierarchical baseline (project-level approximation)
# ---------------------------------------------------------------------------


class HMemBaseline:
    """H-MEM-style hierarchical baseline — a PROJECT-LEVEL APPROXIMATION.

    **This is NOT a faithful reproduction of any published H-MEM system.**
    This repository documents no H-MEM internals (only the name appears in
    ``README.md`` as a checklist baseline), so no paper behavior is reproduced
    and no research claims are made. What is implemented is the simplest
    defensible *hierarchical* retrieval strategy using only concepts and
    utilities already present in this repository:

    * **Hierarchy** - ``docs/interfaces.md`` defines the store query contract
      with the tiers ``"STM" | "MTM" | "LTM"``. History is split by recency
      into exactly these three contiguous tiers (oldest = LTM, middle = MTM,
      newest = STM; boundary indices ``n // 3`` and ``2 * n // 3``, so small
      histories may leave a tier empty). The tier assignment is static,
      computed once from the injected history - no eviction, promotion,
      summarization, or online updates.
    * **Coarse step (tier priority)** - per query, each tier's representative
      score is the maximum lexical score of its entries (scorer:
      :func:`src.retrieval.reranker._lexical_score`); tiers are ordered by
      that score (desc), ties broken by recency (STM before MTM before LTM).
    * **Fine step (within tiers)** - inside each tier, entries are ordered by
      score (desc) then recency (newer first). Concatenating tiers in coarse
      order yields the final candidate order.
    * **Budgeted selection** - the ordered candidates are passed through the
      repository's greedy selector
      (:func:`src.retrieval.token_budget.select_by_token_budget`), so the
      invariant ``estimate_tokens(format_context(selected)) <= token_budget``
      holds exactly as for ``memory_r1``.
    * **Formatting / ids** - shared ``format_context``; ``retrieved_ids``
      contains the ``id`` of each selected entry that has one (id-less
      entries appear in context but are omitted, per ``docs/interfaces.md``),
      so Recall@K / Precision@K / MRR are computed by the framework.

    Contrast with the sibling baselines: ``full_history`` selects nothing,
    ``memoryos`` applies flat top-``k`` over two static tiers, ``memory_r1``
    ranks globally in one stage, and this baseline inserts a tier-priority
    (coarse-to-fine) ordering step before budget selection.

    Args:
        history: ``str`` or list of turn/item dicts (see
            :func:`format_full_history`). ``None``/empty means empty history.
        token_budget: Positive int budget enforced by the selector
            (approximation constant supplied by the caller - the CLI forwards
            ``--token-budget``; default 512 matches ``DEFAULT_TOKEN_BUDGET``).

    Example:
        >>> baseline = HMemBaseline(
        ...     [{"id": "m1", "text": "likes tea"}, {"id": "m2", "text": "owns a bike"}],
        ...     token_budget=512,
        ... )
        >>> outcome = baseline("tea preference")
        >>> isinstance(outcome.retrieved_ids, list)
        True
    """

    DEFAULT_TOKEN_BUDGET = 512

    def __init__(
        self,
        history: Union[str, List, None] = None,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> None:
        if (
            not isinstance(token_budget, int)
            or isinstance(token_budget, bool)
            or token_budget <= 0
        ):
            raise ValueError(f"token_budget must be a positive integer, got {token_budget!r}")
        self._token_budget = token_budget

        if isinstance(history, str):
            entries: List = [history] if history else []
        elif history is None:
            entries = []
        elif isinstance(history, (list, tuple)):
            entries = list(history)
        else:
            entries = [history]
        self._entries = entries

        # Static recency tiers: LTM = oldest, MTM = middle, STM = newest.
        # Boundaries n // 3 and 2 * n // 3 (empty tiers possible for small n).
        n = len(entries)
        b1, b2 = n // 3, (2 * n) // 3
        self._ltm = [(i, entries[i]) for i in range(0, b1)]
        self._mtm = [(i, entries[i]) for i in range(b1, b2)]
        self._stm = [(i, entries[i]) for i in range(b2, n)]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def token_budget(self) -> int:
        """Budget enforced by the selection stage."""
        return self._token_budget

    @property
    def num_entries(self) -> int:
        """Number of history entries available for ranking."""
        return len(self._entries)

    @property
    def num_ltm(self) -> int:
        """Number of entries in the long-term tier."""
        return len(self._ltm)

    @property
    def num_mtm(self) -> int:
        """Number of entries in the mid-term tier."""
        return len(self._mtm)

    @property
    def num_stm(self) -> int:
        """Number of entries in the short-term tier."""
        return len(self._stm)

    # ------------------------------------------------------------------
    # Hierarchical coarse-to-fine ranking
    # ------------------------------------------------------------------

    def rank(self, query: str) -> List[Tuple[int, dict, float]]:
        """Hierarchically rank ALL entries for *query*.

        Coarse: tier priority = max entry score per tier (desc), ties go to
        the more recent tier (STM > MTM > LTM). Fine: within a tier, score
        desc then recency desc. Deterministic; entries are never mutated.

        Returns:
            List of ``(original_index, normalized_entry, score)`` tuples in
            final candidate order (before budget selection).
        """
        tiers = [
            # (recency_rank, tier_members)  — recency_rank 0 = most recent
            (0, self._stm),
            (1, self._mtm),
            (2, self._ltm),
        ]
        tier_scores: List[Tuple[float, int, List[Tuple[int, dict, float]]]] = []
        for recency_rank, members in tiers:
            scored: List[Tuple[int, dict, float]] = []
            for idx, entry in members:
                norm = _normalize_entry(entry)
                scored.append((idx, norm, _lexical_score(query, norm["text"])))
            best = max((s for _i, _n, s in scored), default=0.0)
            # sort key for tiers: -best score, then recency_rank (recent first)
            tier_scores.append((-best, recency_rank, scored))

        tier_scores.sort(key=lambda t: (t[0], t[1]))

        ordered: List[Tuple[int, dict, float]] = []
        for _neg_best, _rr, scored in tier_scores:
            scored.sort(key=lambda t: (-t[2], -t[0]))
            ordered.extend(scored)
        return ordered

    def __call__(self, query: str) -> RetrievalOutcome:
        """Hierarchically rank, keep what fits the budget, return outcome."""
        ranked = self.rank(query)
        selected = select_by_token_budget(
            [norm for _idx, norm, _s in ranked], self._token_budget
        )
        selected_set = {id(norm) for norm in selected}
        ids: List[str] = []
        for idx, norm, _s in ranked:
            if id(norm) not in selected_set:
                continue
            eid = _entry_id(self._entries[idx])
            if eid is not None:
                ids.append(eid)
        context = format_context(selected)
        return RetrievalOutcome(
            context=context,
            retrieved_ids=ids,
            tokens=tokens_returned(context),
        )


def hmem_retrieval_fn(
    history: Union[str, List, None], token_budget: int = HMemBaseline.DEFAULT_TOKEN_BUDGET
) -> HMemBaseline:
    """Convenience factory: build the hierarchical ``retrieval_fn`` for a history."""
    return HMemBaseline(history, token_budget=token_budget)
