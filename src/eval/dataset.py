"""Dataset abstraction for retrieval evaluation.

Provides a minimal, dataset-agnostic evaluation case and dataset container
that supports synthetic unit tests now and LoCoMo/LongMemEval adapters later.

Do NOT download datasets automatically; no network required.
Future adapters must supply query and relevant_ids at minimum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass
class RetrievalEvalCase:
    """One retrieval evaluation case.

    Attributes:
        query: Natural-language query for retrieval.
        relevant_ids: Ground-truth IDs of memories that should be retrieved.
            These are the "useful" memories for the query; empty list means
            no relevant memories (handled safely by metrics).
        candidates: Optional list of candidate memories considered for this
            query (e.g., all store contents). Not required for metric
            computation but useful for debugging/adapters.
        metadata: Optional free-form metadata (e.g., source dataset,
            session, latency_ms, token_budget, etc.). Not interpreted by
            metrics but preserved for evaluation reporting.
        case_id: Optional unique identifier for the case (e.g., "locomo_001").

    Notes:
        Not coupled to ChromaDB or any specific dataset. Synthetic tests can
        create cases directly; future LoCoMo/LongMemEval adapters should
        map their schema to this dataclass (query + relevant_ids required).
    """

    query: str
    relevant_ids: List[str]
    candidates: Optional[List[Dict]] = None
    metadata: Dict = field(default_factory=dict)
    case_id: Optional[str] = None


# Alias for shorter import name; both refer to same dataclass
EvaluationCase = RetrievalEvalCase


class EvalDataset:
    """Container for retrieval evaluation cases.

    Minimal abstraction over a list of cases. Supports iteration,
    indexing, len(), and dataset-agnostic adapters.

    Future adapters (LoCoMo, LongMemEval) should convert their native
    examples into ``RetrievalEvalCase`` objects and return an ``EvalDataset``.

    Example:
        dataset = EvalDataset(cases=[...])
        for case in dataset:
            retrieved = retriever.retrieve(case.query, k=10)
            metrics = evaluator.evaluate(...)
    """

    def __init__(self, cases: Optional[List[RetrievalEvalCase]] = None, name: str = "synthetic") -> None:
        self.cases: List[RetrievalEvalCase] = list(cases) if cases is not None else []
        self.name: str = name

    def add_case(self, case: RetrievalEvalCase) -> None:
        """Add a case to the dataset."""
        self.cases.append(case)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterable[RetrievalEvalCase]:
        return iter(self.cases)

    def __getitem__(self, idx: int) -> RetrievalEvalCase:
        return self.cases[idx]

    def is_empty(self) -> bool:
        return len(self.cases) == 0

    @classmethod
    def from_synthetic(
        cls, examples: List[Dict], name: str = "synthetic"
    ) -> "EvalDataset":
        """Create dataset from list of dicts with keys query/relevant_ids.

        Args:
            examples: List of dicts each with at least ``query`` and
                ``relevant_ids``; optional ``case_id``, ``candidates``, ``metadata``.
            name: Dataset name label.

        Returns:
            EvalDataset instance.
        """
        cases: List[RetrievalEvalCase] = []
        for ex in examples:
            cases.append(
                RetrievalEvalCase(
                    query=ex["query"],
                    relevant_ids=list(ex.get("relevant_ids", [])),
                    candidates=ex.get("candidates"),
                    metadata=dict(ex.get("metadata", {})),
                    case_id=ex.get("case_id"),
                )
            )
        return cls(cases=cases, name=name)
