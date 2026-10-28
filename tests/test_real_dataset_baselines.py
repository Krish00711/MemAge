"""H-3 regression tests: real-dataset baselines receive per-entry history.

The bug: ``_build_strategy`` used to pass ``history_from_case(case)`` - a
formatted full-history STRING - into the entry-based baselines (memoryos /
memory_r1 / hmem).  A string collapses into ONE id-less entry, so:

* the whole history became a single "memory" item;
* ``retrieved_ids`` was empty (no entry carried a dataset ID);
* Recall@K / P@K / MRR were structurally zero;
* the context double-formatted: ``Content: [Memory 1] ...``.

The fix: ``memory_entries_from_case`` returns individual entries with the
original dataset IDs (LoCoMo ``dia_id``; LongMemEval ``session_id``), and
``_make_retrieval_fn`` feeds those lists to the entry-based baselines.
"""

import json

import pytest

from src.eval.baselines import (
    FullHistoryBaseline,
    HMemBaseline,
    MemoryOSBaseline,
    MemoryR1Baseline,
    format_full_history,
    history_from_case,
    hmem_retrieval_fn,
    memory_entries_from_case,
    memory_r1_retrieval_fn,
    memoryos_retrieval_fn,
)
from src.eval.benchmark import BenchmarkRunner, RetrievalOutcome
from src.eval.dataset import BenchmarkCase


# ---------------------------------------------------------------------------
# Fixtures (smallest possible; no real dataset files needed)
# ---------------------------------------------------------------------------


def _locomo_case() -> BenchmarkCase:
    """One LoCoMo-style case: D1:1 relevant, D1:2 distractor."""
    return BenchmarkCase(
        query="Where does Maya live?",
        reference_answer="Chennai",
        relevant_ids=["D1:1"],
        case_id="sample_1::qa0",
        metadata={
            "dataset": "locomo",
            "turns": [
                {
                    "id": "D1:1",
                    "text": "Maya lives in Chennai",
                    "speaker": "Maya",
                    "session": "session_1",
                    "timestamp": "2024-01-01",
                },
                {
                    "id": "D1:2",
                    "text": "Bob likes painting",
                    "speaker": "Bob",
                    "session": "session_1",
                    "timestamp": "2024-01-01",
                },
            ],
        },
    )


def _longmemeval_case() -> BenchmarkCase:
    """One LongMemEval-style case: session s2 relevant, s1 distractor."""
    return BenchmarkCase(
        query="What is the name of the cat?",
        reference_answer="Whiskers",
        relevant_ids=["s2"],
        case_id="q1",
        metadata={
            "dataset": "longmemeval",
            "sessions": [
                {
                    "session_id": "s1",
                    "date": "2024/05/01",
                    "turns": [
                        {"role": "user", "content": "I moved to Paris"},
                        {"role": "assistant", "content": "Nice!"},
                    ],
                },
                {
                    "session_id": "s2",
                    "date": "2024/05/02",
                    "turns": [
                        {"role": "user", "content": "I adopted a cat named Whiskers"},
                    ],
                },
            ],
        },
    )


# ---------------------------------------------------------------------------
# Per-entry construction + ID preservation
# ---------------------------------------------------------------------------


def test_locomo_entries_are_per_turn_with_ids():
    entries = memory_entries_from_case(_locomo_case())
    assert isinstance(entries, list)
    assert len(entries) == 2
    assert [e["id"] for e in entries] == ["D1:1", "D1:2"]
    assert entries[0]["text"] == "Maya lives in Chennai"
    assert entries[1]["text"] == "Bob likes painting"


def test_longmemeval_entries_are_per_turn_with_session_ids():
    entries = memory_entries_from_case(_longmemeval_case())
    assert len(entries) == 3
    # Every entry carries the session id - the dataset's relevance unit.
    assert [e["id"] for e in entries] == ["s1", "s1", "s2"]
    # Session date attached as timestamp fallback; source is session:role.
    assert entries[0]["timestamp"] == "2024/05/01"
    assert entries[0]["source"] == "s1:user"
    assert entries[2]["source"] == "s2:user"
    assert entries[2]["content"] == "I adopted a cat named Whiskers"


def test_entries_are_raw_never_preformatted():
    # The old failure mode: entries were already "[Memory N]" strings that the
    # baselines wrapped AGAIN. Raw entries must contain no formatting markers.
    for case in (_locomo_case(), _longmemeval_case()):
        for entry in memory_entries_from_case(case):
            blob = json.dumps(entry)
            assert "[Memory" not in blob


def test_entries_empty_when_no_history_metadata():
    assert memory_entries_from_case(BenchmarkCase(query="q", reference_answer="a")) == []
    assert (
        memory_entries_from_case(BenchmarkCase(query="q", reference_answer="a", metadata={}))
        == []
    )


def test_history_from_case_still_returns_formatted_string():
    # full_history path unchanged: a formatted context string with [Memory N].
    ctx = history_from_case(_locomo_case())
    assert "Maya lives in Chennai" in ctx
    assert "Bob likes painting" in ctx
    assert ctx.count("[Memory ") == 2
    # history_from_case == format_full_history(entries): one formatter, once.
    assert ctx == format_full_history(memory_entries_from_case(_locomo_case()))


# ---------------------------------------------------------------------------
# Baselines receive entries: shape, IDs, no double formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_fn",
    [
        lambda entries: memoryos_retrieval_fn(entries, k=5),
        lambda entries: memory_r1_retrieval_fn(entries, token_budget=512),
        lambda entries: hmem_retrieval_fn(entries, token_budget=512),
    ],
    ids=["memoryos", "memory_r1", "hmem"],
)
def test_locomo_baselines_retrieve_individual_entry(make_fn):
    case = _locomo_case()
    baseline = make_fn(memory_entries_from_case(case))
    outcome = baseline(case.query)

    assert isinstance(outcome, RetrievalOutcome)
    # The relevant individual turn's ID is retrievable...
    assert outcome.retrieved_ids is not None
    assert "D1:1" in outcome.retrieved_ids
    # ...and it is the TOP hit for this query.
    assert outcome.retrieved_ids[0] == "D1:1"
    # Raw text present in the formatted context.
    assert "Maya lives in Chennai" in outcome.context


@pytest.mark.parametrize(
    "make_fn",
    [
        lambda entries: memoryos_retrieval_fn(entries, k=5),
        lambda entries: memory_r1_retrieval_fn(entries, token_budget=512),
        lambda entries: hmem_retrieval_fn(entries, token_budget=512),
    ],
    ids=["memoryos", "memory_r1", "hmem"],
)
def test_no_double_formatting(make_fn):
    case = _locomo_case()
    outcome = make_fn(memory_entries_from_case(case))(case.query)
    # Old signature: the giant preformatted string became one entry whose
    # Content line re-emitted "[Memory 1]".
    assert "Content: [Memory" not in outcome.context
    # Exactly one [Memory block per selected entry, numbered from 1.
    n_blocks = outcome.context.count("[Memory ")
    assert n_blocks == len(outcome.retrieved_ids)
    assert outcome.context.count("[Memory 1]") == 1


def test_longmemeval_baselines_retrieve_relevant_session():
    case = _longmemeval_case()
    baseline = memoryos_retrieval_fn(memory_entries_from_case(case), k=5)
    outcome = baseline(case.query)
    assert outcome.retrieved_ids is not None
    assert outcome.retrieved_ids[0] == "s2"
    assert "Whiskers" in outcome.context


# ---------------------------------------------------------------------------
# Retrieval metrics become meaningful (non-degenerate)
# ---------------------------------------------------------------------------


def test_retrieval_metrics_non_degenerate_through_runner():
    from src.eval.metrics import precision_at_k, recall_at_k, reciprocal_rank

    case = _locomo_case()
    entries = memory_entries_from_case(case)
    # Top-1 selection: only D1:1 survives -> metrics are exact.
    baseline = memoryos_retrieval_fn(entries, k=1)
    outcome = baseline(case.query)
    assert outcome.retrieved_ids == ["D1:1"]

    assert recall_at_k(outcome.retrieved_ids, case.relevant_ids, k=5) == 1.0
    assert precision_at_k(outcome.retrieved_ids, case.relevant_ids, k=5) == 1.0
    assert reciprocal_rank(outcome.retrieved_ids, case.relevant_ids) == 1.0

    # Same shape through the actual BenchmarkRunner.
    runner = BenchmarkRunner(
        retrieval_fn=baseline,
        answer_fn=lambda q, c: "Chennai" if "Chennai" in c else "",
        k=5,
    )
    result = runner.run_single(case)
    assert result.retrieved_ids == ["D1:1"]
    assert result.retrieval_recall == 1.0
    assert result.retrieval_precision == 1.0
    assert result.retrieval_rr == 1.0


def test_longmemeval_metrics_non_degenerate():
    from src.eval.metrics import recall_at_k

    case = _longmemeval_case()
    outcome = memoryos_retrieval_fn(memory_entries_from_case(case), k=5)(case.query)
    assert recall_at_k(outcome.retrieved_ids, case.relevant_ids, k=5) == 1.0


# ---------------------------------------------------------------------------
# _build_strategy end-to-end on a tiny local LoCoMo fixture (no LLM calls)
# ---------------------------------------------------------------------------


def _write_locomo_fixture(tmp_path, n_turns=2):
    import json as _json

    fixture = [
        {
            "sample_id": "sample_1",
            "conversation": {
                "speaker_a": "Maya",
                "speaker_b": "Bob",
                "session_1_date_time": "2024-01-01",
                "session_1": [
                    {"speaker": "Maya", "dia_id": "D1:1", "text": "Maya lives in Chennai"},
                    {"speaker": "Bob", "dia_id": "D1:2", "text": "Bob likes painting"},
                ][:n_turns],
            },
            "qa": [
                {
                    "question": "Where does Maya live?",
                    "answer": "Chennai",
                    "evidence": ["D1:1"],
                    "category": "adversarial",
                }
            ],
        }
    ]
    d = tmp_path / "locomo"
    d.mkdir(parents=True, exist_ok=True)
    (d / "locomo10.json").write_text(_json.dumps(fixture), encoding="utf-8")


@pytest.mark.parametrize("bsl", ["memoryos", "memory_r1", "hmem"])
def test_build_strategy_entry_baselines_end_to_end(tmp_path, bsl):
    """Full real-dataset path: fixture -> adapter -> _build_strategy -> ids."""
    from src.eval.adapters import load_dataset
    from src.eval.run_benchmark import _build_strategy

    _write_locomo_fixture(tmp_path)
    loaded = load_dataset("locomo", data_dir=str(tmp_path))
    case = loaded.cases[0]

    strategy = _build_strategy(bsl, 512, str(tmp_path))
    assert strategy is not None
    retrieval_fn, answer_fn, memory_prepare = strategy

    memory_prepare(case)
    outcome = retrieval_fn(case.query)

    assert outcome.retrieved_ids is not None
    assert outcome.retrieved_ids[0] == "D1:1"
    assert "Maya lives in Chennai" in outcome.context
    assert "Content: [Memory" not in outcome.context
    # ID loss (the old failure mode) is gone.
    assert outcome.retrieved_ids != []


def test_build_strategy_full_history_unchanged(tmp_path):
    from src.eval.adapters import load_dataset
    from src.eval.run_benchmark import _build_strategy

    _write_locomo_fixture(tmp_path)
    loaded = load_dataset("locomo", data_dir=str(tmp_path))
    case = loaded.cases[0]

    retrieval_fn, answer_fn, memory_prepare = _build_strategy("full_history", 512, str(tmp_path))
    memory_prepare(case)
    outcome = retrieval_fn(case.query)

    # Full-history semantics preserved: no retrieval ids, whole context.
    assert outcome.retrieved_ids is None
    assert "Maya lives in Chennai" in outcome.context
    assert "Bob likes painting" in outcome.context


def test_build_strategy_none_is_wired(tmp_path):
    from src.eval.run_benchmark import _build_strategy

    # H-4: the real `none` strategy now constructs.  Building only wires
    # callables (no store/LLM/model access yet), so no embedder injection is
    # needed here; construction fails only when no embedder chain exists.
    strategy = _build_strategy("none", 512, str(tmp_path))
    assert strategy is not None
    retrieval_fn, answer_fn, memory_prepare = strategy
    assert callable(retrieval_fn)
    assert callable(answer_fn)
    assert callable(memory_prepare)


# ---------------------------------------------------------------------------
# Class-level sanity: baselines still accept list-of-entries as before
# ---------------------------------------------------------------------------


def test_baseline_classes_accept_entries_lists_unchanged():
    entries = [
        {"id": "m1", "text": "likes tea", "timestamp": "t1", "source": "s1"},
        {"id": "m2", "text": "owns a bike", "timestamp": "t2", "source": "s2"},
    ]
    assert MemoryOSBaseline(entries, k=1)("tea preference").retrieved_ids == ["m1"]
    assert MemoryR1Baseline(entries, token_budget=64)("tea preference").retrieved_ids[0] == "m1"
    assert HMemBaseline(entries, token_budget=64)("tea preference").retrieved_ids[0] == "m1"
    assert FullHistoryBaseline(entries)("any").retrieved_ids is None


# ---------------------------------------------------------------------------
# Synthetic behavior remains unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bsl", ["none", "full_history", "memoryos", "memory_r1", "hmem"])
def test_synthetic_baselines_still_run(bsl, capsys):
    from src.eval.run_benchmark import main

    code = main(["--dataset", "synthetic", "--baseline", bsl])
    out = capsys.readouterr().out
    assert code == 0
    assert "Mean F1" in out
