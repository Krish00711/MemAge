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


# ---------------------------------------------------------------------------
# Benchmark abstraction (end-to-end QA)
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkCase:
    """One end-to-end benchmark case (QA + optional retrieval ground truth).

    Attributes:
        query: Question / query string.
        reference_answer: Expected / ground-truth answer string.
        relevant_ids: Optional ground-truth relevant memory IDs (for retrieval
            metric integration). Empty list if not provided.
        case_id: Optional unique identifier (e.g., "locomo_42").
        metadata: Optional free-form metadata (conversation_id, session,
            dataset name, etc.).
        conversation_id: Optional conversation/session identifier (alternative
            to storing inside metadata).

    Notes:
        Designed to be dataset-agnostic. Synthetic tests can instantiate
        directly; future LoCoMo/LongMemEval adapters convert native records
        to this dataclass. The adapter boundary is: loader must produce
        ``query`` and ``reference_answer`` at minimum; ``relevant_ids`` and
        identifiers are optional.
    """

    query: str
    reference_answer: str
    relevant_ids: List[str] = field(default_factory=list)
    case_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    conversation_id: Optional[str] = None

    def to_retrieval_case(self) -> RetrievalEvalCase:
        """Convert to retrieval-only case (drops reference answer)."""
        return RetrievalEvalCase(
            query=self.query,
            relevant_ids=list(self.relevant_ids),
            candidates=None,
            metadata=dict(self.metadata),
            case_id=self.case_id,
        )


class BenchmarkDataset:
    """Container for benchmark cases (QA-level).

    Analogous to ``EvalDataset`` but stores ``BenchmarkCase`` objects.
    Provides ``len``, iteration, indexing, and synthetic helper.

    Future loaders (LoCoMo, LongMemEval) should return this type.

    Example:
        ds = BenchmarkDataset(cases=[BenchmarkCase(query="...", reference_answer="...")])
        for case in ds:
            result = benchmark.run_case(case)
    """

    def __init__(self, cases: Optional[List[BenchmarkCase]] = None, name: str = "synthetic") -> None:
        self.cases: List[BenchmarkCase] = list(cases) if cases is not None else []
        self.name: str = name

    def add_case(self, case: BenchmarkCase) -> None:
        self.cases.append(case)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterable[BenchmarkCase]:
        return iter(self.cases)

    def __getitem__(self, idx: int) -> BenchmarkCase:
        return self.cases[idx]

    def is_empty(self) -> bool:
        return len(self.cases) == 0

    @classmethod
    def from_synthetic(
        cls, examples: List[Dict], name: str = "synthetic"
    ) -> "BenchmarkDataset":
        """Create benchmark dataset from list of dicts.

        Each dict requires ``query`` and ``reference_answer``; optional
        ``relevant_ids``, ``case_id``, ``conversation_id``, ``metadata``.

        Args:
            examples: List of example dicts.
            name: Dataset name label.

        Returns:
            BenchmarkDataset instance.
        """
        cases: List[BenchmarkCase] = []
        for ex in examples:
            cases.append(
                BenchmarkCase(
                    query=ex["query"],
                    reference_answer=ex.get("reference_answer", ex.get("reference", "")),
                    relevant_ids=list(ex.get("relevant_ids", [])),
                    case_id=ex.get("case_id"),
                    metadata=dict(ex.get("metadata", {})),
                    conversation_id=ex.get("conversation_id"),
                )
            )
        return cls(cases=cases, name=name)
