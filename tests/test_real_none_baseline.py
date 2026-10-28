"""H-4 tests: the real ``none`` baseline (our system) on dataset fixtures.

Execution flow under test (the real MemAge wiring, no parallel fake stack):

    case -> memory preparation (isolated TieredMemoryStore per case)
         -> two-stage retrieval (CoarseRetriever -> LLMReranker -> budget -> format)
         -> answer generation via ``call_llm``
         -> BenchmarkRunner metrics (F1 / BLEU-1 / Recall@K / P@K / MRR / tokens)

Everything is deterministic and offline: a hash-based bag-of-words embedder
and a recording fake LLM are injected (the store and reranker themselves are
the real implementations).  Tiny local fixtures - no dataset files required.
"""

from __future__ import annotations

import copy
import json
import math
import re
import sys
import zlib

import pytest


# ---------------------------------------------------------------------------
# Deterministic injectables (offline stand-ins for embedder / LLM only)
# ---------------------------------------------------------------------------


class _HashEmbedder:
    """Deterministic bag-of-words embedder (crc32 token hashing, L2-normalized).

    Texts sharing tokens land closer together, so stage-1 ordering for the
    tiny fixtures is fixed across runs - no model download, no network.
    """

    DIM = 256

    def encode(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for tok in re.findall(r"[a-z0-9]+", (text or "").lower()):
            vec[zlib.crc32(tok.encode("utf-8")) % self.DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm:
            vec = [v / norm for v in vec]
        return vec


class _FakeLLM:
    """Recording ``call_llm(prompt, system=None) -> str`` stand-in.

    * Stage-2 rerank prompts (built by ``LLMReranker``) get a valid JSON
      ``rankings`` reply echoing the candidate IDs from the prompt, in order -
      exercising the real reranker parse/rank path.
    * Everything else (answer prompts) gets the configured canned answer.
    """

    def __init__(self, answer: str = "OK") -> None:
        self.answer = answer
        self.calls: list[str] = []

    def __call__(self, prompt: str, system: str | None = None, **kwargs) -> str:
        self.calls.append(prompt)
        if "relevance reranker" in prompt:
            ids = re.findall(r"^- ID: (.+)$", prompt, flags=re.MULTILINE)
            rankings = [
                {"id": eid, "score": max(0.0, round(0.9 - 0.05 * i, 3))}
                for i, eid in enumerate(ids)
            ]
            return json.dumps({"rankings": rankings})
        return self.answer

    @property
    def rerank_prompts(self) -> list[str]:
        return [p for p in self.calls if "relevance reranker" in p]

    @property
    def answer_prompts(self) -> list[str]:
        return [p for p in self.calls if "Question:" in p and "Answer:" in p]


def _patch_llm(monkeypatch, fake: _FakeLLM) -> None:
    """Route ``src.llm.client.call_llm`` (and everything importing it) to *fake*."""
    monkeypatch.setattr("src.llm.client.call_llm", fake)


def _patch_store_embedder(monkeypatch, embedder) -> None:
    """Make every ``TieredMemoryStore`` built after this line use *embedder*.

    Wraps (does not replace) the real class - the store under test is always
    a genuine ``TieredMemoryStore``; only its embedder is injected.  Used by
    CLI-level tests, which cannot pass ``embedder=`` through ``main()``.
    """
    import src.store.memory_store as ms

    original = ms.TieredMemoryStore

    def factory(*args, **kwargs):
        # memory_prepare passes embedder=None explicitly on real runs; treat
        # "missing or None" as "use the injected one".
        if kwargs.get("embedder") is None:
            kwargs["embedder"] = embedder
        return original(*args, **kwargs)

    monkeypatch.setattr(ms, "TieredMemoryStore", factory)


# ---------------------------------------------------------------------------
# Tiny dataset fixtures (schema per src/eval/adapters.py)
# ---------------------------------------------------------------------------


def _write_locomo_fixture(tmp_path) -> None:
    """Two samples -> two isolated cases with distinct memory IDs."""
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
                ],
            },
            "qa": [
                {
                    "question": "Where does Maya live?",
                    "answer": "Chennai",
                    "evidence": ["D1:1"],
                    "category": "multi-hop",
                }
            ],
        },
        {
            "sample_id": "sample_2",
            "conversation": {
                "speaker_a": "Zoe",
                "speaker_b": "Ben",
                "session_1_date_time": "2024-02-01",
                "session_1": [
                    {"speaker": "Zoe", "dia_id": "D2:1", "text": "Zoe plays violin"},
                ],
            },
            "qa": [
                {
                    "question": "What does Zoe play?",
                    "answer": "violin",
                    "evidence": ["D2:1"],
                    "category": "temporal",
                }
            ],
        },
    ]
    d = tmp_path / "locomo"
    d.mkdir(parents=True, exist_ok=True)
    (d / "locomo10.json").write_text(json.dumps(fixture), encoding="utf-8")


def _write_longmemeval_fixture(tmp_path) -> None:
    """One question, two haystack sessions -> relevance unit = session id."""
    fixture = [
        {
            "question_id": "q1",
            "question_type": "single-session-user",
            "question": "What is the name of the cat?",
            "answer": "Whiskers",
            "question_date": "2024/05/03",
            "haystack_session_ids": ["s1", "s2"],
            "haystack_dates": ["2024/05/01", "2024/05/02"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I moved to Paris"},
                    {"role": "assistant", "content": "Nice!"},
                ],
                [
                    {"role": "user", "content": "I adopted a cat named Whiskers"},
                ],
            ],
            "answer_session_ids": ["s2"],
        }
    ]
    d = tmp_path / "longmemeval"
    d.mkdir(parents=True, exist_ok=True)
    (d / "longmemeval_s.json").write_text(json.dumps(fixture), encoding="utf-8")


def _build_none(tmp_path, fake: _FakeLLM | None = None, monkeypatch=None):
    """Build the real ``none`` strategy for the fixtures in *tmp_path*.

    Returns ``(strategy_tuple, dataset)``.  Patches ``call_llm`` when a
    *monkeypatch* is supplied so no API key / network is ever touched.
    """
    from src.eval.adapters import load_dataset
    from src.eval.run_benchmark import _build_strategy

    if monkeypatch is not None:
        _patch_llm(monkeypatch, fake or _FakeLLM())
    strategy = _build_strategy("none", 512, str(tmp_path), embedder=_HashEmbedder())
    assert strategy is not None, "none strategy must be constructible"
    return strategy, load_dataset("locomo", data_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Store-item construction: IDs, grouping, no mutation
# ---------------------------------------------------------------------------


def test_store_items_locomo_preserve_dataset_ids():
    from src.eval.baselines import memory_entries_from_case
    from src.eval.dataset import BenchmarkCase
    from src.eval.run_benchmark import _store_items_from_entries

    case = BenchmarkCase(
        query="Where does Maya live?",
        reference_answer="Chennai",
        relevant_ids=["D1:1"],
        metadata={
            "turns": [
                {"id": "D1:1", "text": "Maya lives in Chennai",
                 "speaker": "Maya", "session": "session_1", "timestamp": "2024-01-01"},
                {"id": "D1:2", "text": "Bob likes painting",
                 "speaker": "Bob", "session": "session_1", "timestamp": "2024-01-01"},
            ]
        },
    )
    items = _store_items_from_entries(memory_entries_from_case(case))
    assert [i["id"] for i in items] == ["D1:1", "D1:2"]
    assert items[0]["text"] == "Maya lives in Chennai"
    assert items[0]["timestamp"] == "2024-01-01"
    assert items[0]["source"] == "D1:1"  # id fallback, _normalize_entry parity


def test_store_items_longmemeval_group_turns_under_session_id(tmp_path):
    from src.eval.adapters import load_dataset
    from src.eval.baselines import memory_entries_from_case
    from src.eval.run_benchmark import _store_items_from_entries

    _write_longmemeval_fixture(tmp_path)
    case = load_dataset("longmemeval", data_dir=str(tmp_path)).cases[0]
    items = _store_items_from_entries(memory_entries_from_case(case))
    # Relevance unit is the session: one store item per session id, turns
    # joined - the store's primary key would otherwise collide.
    assert [i["id"] for i in items] == ["s1", "s2"]
    assert "I moved to Paris" in items[0]["text"]
    assert "Nice!" in items[0]["text"]
    assert items[1]["text"] == "I adopted a cat named Whiskers"
    assert items[1]["timestamp"] == "2024/05/02"
    assert items[1]["source"] == "s2:user"  # first entry's source kept


def test_store_items_never_mutate_entries_and_skip_unusable():
    from src.eval.baselines import memory_entries_from_case
    from src.eval.dataset import BenchmarkCase
    from src.eval.run_benchmark import _store_items_from_entries

    case = BenchmarkCase(
        query="q",
        reference_answer="a",
        metadata={
            "turns": [
                {"id": "D1:1", "text": "keep me", "timestamp": "", "speaker": "A"},
                {"id": "", "text": "no id - skipped"},
                {"id": "D1:3", "text": "   "},       # blank text - skipped
                "not-a-dict",                        # malformed - skipped
            ]
        },
    )
    entries = memory_entries_from_case(case)
    snapshot = copy.deepcopy(entries)
    items = _store_items_from_entries(entries)
    assert [i["id"] for i in items] == ["D1:1"]
    assert entries == snapshot  # original entries untouched
    # and a case's adapter metadata stays pristine through a full prepare
    meta_snapshot = copy.deepcopy(case.metadata)
    _store_items_from_entries(memory_entries_from_case(case))
    assert case.metadata == meta_snapshot


# ---------------------------------------------------------------------------
# Strategy construction: real components, never the synthetic demo stack
# ---------------------------------------------------------------------------


def test_none_strategy_uses_real_components_not_synthetic_ones(tmp_path, monkeypatch):
    _write_locomo_fixture(tmp_path)
    strategy, dataset = _build_none(
        tmp_path, _FakeLLM(answer="Chennai"), monkeypatch=monkeypatch
    )
    retrieval_fn, answer_fn, memory_prepare = strategy
    memory_prepare(dataset.cases[0])
    state = retrieval_fn.state

    from src.retrieval.reranker import LLMReranker
    from src.retrieval.retriever import CoarseRetriever
    from src.store.memory_store import TieredMemoryStore
    from src.eval.run_benchmark import _LexicalReranker, _StaticStore

    # Real store + real two-stage pipeline...
    assert isinstance(state["store"], TieredMemoryStore)
    assert isinstance(state["ctx"].retriever, CoarseRetriever)
    assert isinstance(state["ctx"].reranker, LLMReranker)
    # ...never the synthetic demo's static store / lexical reranker.
    assert not isinstance(state["store"], _StaticStore)
    assert not isinstance(state["ctx"].reranker, _LexicalReranker)
    assert state["ctx"].retriever._store is state["store"]
    # Stage-1 considers every memory of the case (2 turns in this fixture).
    assert state["ctx"].k == 2
    assert state["ctx"].tier == "all"
    assert state["retrieval_fn"] is not None
    assert state["error"] is None


# ---------------------------------------------------------------------------
# End-to-end LoCoMo: store -> retrieval -> call_llm -> metrics
# ---------------------------------------------------------------------------


def test_none_locomo_end_to_end(tmp_path, monkeypatch):
    from src.eval.benchmark import BenchmarkRunner, RetrievalOutcome
    from src.eval.metrics import precision_at_k, recall_at_k, reciprocal_rank

    _write_locomo_fixture(tmp_path)
    fake = _FakeLLM(answer="Chennai")
    strategy, dataset = _build_none(tmp_path, fake, monkeypatch=monkeypatch)
    retrieval_fn, answer_fn, memory_prepare = strategy
    case = dataset.cases[0]

    memory_prepare(case)
    state = retrieval_fn.state

    # (1) case memories enter the store, dataset ids intact
    assert state["store"].count() == {"STM": 2, "MTM": 0, "LTM": 0}
    stored = state["store"].get("D1:1")
    assert stored is not None
    assert stored["text"] == "Maya lives in Chennai"

    # (2) retrieval returns the expected memory ids (D1:1 ranked first)
    outcome = retrieval_fn(case.query)
    assert isinstance(outcome, RetrievalOutcome)
    assert outcome.retrieved_ids is not None
    assert outcome.retrieved_ids[0] == "D1:1"
    assert set(outcome.retrieved_ids) == {"D1:1", "D1:2"}
    assert "Maya lives in Chennai" in outcome.context
    assert "Content: [Memory" not in outcome.context  # single formatting pass
    assert recall_at_k(outcome.retrieved_ids, case.relevant_ids, 5) == 1.0
    assert precision_at_k(outcome.retrieved_ids, case.relevant_ids, 5) == 0.5
    assert reciprocal_rank(outcome.retrieved_ids, case.relevant_ids) == 1.0

    # (3) full runner: answer via call_llm, metrics produced
    runner = BenchmarkRunner(
        retrieval_fn=retrieval_fn,
        answer_fn=answer_fn,
        memory_prepare_fn=memory_prepare,
        k=5,
    )
    result = runner.run_single(case)
    assert result.predicted_answer == "Chennai"
    assert result.f1 == 1.0
    assert result.bleu1 > 0.0
    assert result.tokens and result.tokens > 0
    assert result.latency_ms > 0
    assert result.retrieval_recall == 1.0
    assert result.retrieval_precision == 0.5
    assert result.retrieval_rr == 1.0

    # (4) both LLM touches went through the (patched) call_llm abstraction
    assert fake.rerank_prompts, "Stage-2 reranker must use call_llm"
    assert fake.answer_prompts, "answer generation must use call_llm"
    assert "D1:1" in fake.rerank_prompts[0]  # candidates carry dataset ids


# ---------------------------------------------------------------------------
# End-to-end LongMemEval fixture (adapter-supported schema)
# ---------------------------------------------------------------------------


def test_none_longmemeval_end_to_end(tmp_path, monkeypatch):
    from src.eval.adapters import load_dataset
    from src.eval.benchmark import BenchmarkRunner
    from src.eval.run_benchmark import _build_strategy

    _write_longmemeval_fixture(tmp_path)
    fake = _FakeLLM(answer="Whiskers")
    monkeypatch.setattr("src.llm.client.call_llm", fake)
    strategy = _build_strategy("none", 512, str(tmp_path), embedder=_HashEmbedder())
    assert strategy is not None
    retrieval_fn, answer_fn, memory_prepare = strategy

    dataset = load_dataset("longmemeval", data_dir=str(tmp_path))
    case = dataset.cases[0]
    memory_prepare(case)
    state = retrieval_fn.state

    # One store item per session id (relevance unit), 3 turns grouped into 2.
    assert state["store"].count() == {"STM": 2, "MTM": 0, "LTM": 0}
    assert state["store"].get("s1") is not None
    assert state["store"].get("s2") is not None
    assert "Whiskers" in state["store"].get("s2")["text"]

    outcome = retrieval_fn(case.query)
    assert outcome.retrieved_ids is not None
    assert outcome.retrieved_ids[0] == "s2"
    assert "Whiskers" in outcome.context

    runner = BenchmarkRunner(
        retrieval_fn=retrieval_fn,
        answer_fn=answer_fn,
        memory_prepare_fn=memory_prepare,
        k=5,
    )
    result = runner.run_single(case)
    assert result.predicted_answer == "Whiskers"
    assert result.retrieval_recall == 1.0
    assert result.retrieval_rr == 1.0


# ---------------------------------------------------------------------------
# Case isolation: no store memory leaks between benchmark cases
# ---------------------------------------------------------------------------


def test_none_cases_are_fully_isolated(tmp_path, monkeypatch):
    _write_locomo_fixture(tmp_path)
    strategy, dataset = _build_none(
        tmp_path, _FakeLLM(answer="x"), monkeypatch=monkeypatch
    )
    retrieval_fn, answer_fn, memory_prepare = strategy
    case_a, case_b = dataset.cases[0], dataset.cases[1]

    meta_a_before = copy.deepcopy(case_a.metadata)
    meta_b_before = copy.deepcopy(case_b.metadata)

    # Prepare case A: A's memories are present.
    memory_prepare(case_a)
    store_a = retrieval_fn.state["store"]
    persist_a = retrieval_fn.state["persist_dir"]
    assert store_a.get("D1:1") is not None
    assert store_a.get("D2:1") is None

    # Prepare case B: a NEW store holds only B's memories.
    memory_prepare(case_b)
    store_b = retrieval_fn.state["store"]
    persist_b = retrieval_fn.state["persist_dir"]
    assert store_b is not store_a
    assert store_b.get("D2:1") is not None
    assert store_b.get("D1:1") is None  # A did not leak into B

    # Distinct persistence roots when chromadb is installed (without it the
    # store falls back to per-instance in-memory collections, no dir at all).
    if persist_a is not None or persist_b is not None:
        assert persist_a != persist_b

    # Retrieval after preparing B never surfaces A's ids.
    outcome = retrieval_fn("Where does Maya live?")
    assert "D1:1" not in (outcome.retrieved_ids or [])

    # And back: preparing A again drops B's memories.
    memory_prepare(case_a)
    assert retrieval_fn.state["store"] is not store_b
    assert retrieval_fn.state["store"].get("D1:1") is not None
    assert retrieval_fn.state["store"].get("D2:1") is None
    outcome = retrieval_fn("What does Zoe play?")
    assert "D2:1" not in (outcome.retrieved_ids or [])

    # Dataset cases were never mutated by preparation.
    assert case_a.metadata == meta_a_before
    assert case_b.metadata == meta_b_before


def test_none_prepare_failure_resets_state_and_reports(tmp_path, monkeypatch, capsys):
    """A failing prepare must never serve the previous case's store."""
    _write_locomo_fixture(tmp_path)
    strategy, dataset = _build_none(
        tmp_path, _FakeLLM(answer="x"), monkeypatch=monkeypatch
    )
    retrieval_fn, answer_fn, memory_prepare = strategy

    memory_prepare(dataset.cases[0])
    assert retrieval_fn.state["store"] is not None

    # Break the store's embedder: prepare fails, state is cleared, error shown.
    class _Boom:
        def encode(self, text):
            raise RuntimeError("embedder exploded")

    monkeypatch.setattr(
        "src.store.memory_store.TieredMemoryStore",
        _failing_store_factory(_Boom()),
    )
    memory_prepare(dataset.cases[1])
    assert retrieval_fn.state["store"] is None
    assert retrieval_fn.state["retrieval_fn"] is None
    assert retrieval_fn.state["error"] is not None
    assert "could not prepare memories" in capsys.readouterr().err

    # Retrieval before/after failure degrades to an empty outcome (no leak).
    outcome = retrieval_fn(dataset.cases[0].query)
    assert outcome.context == ""
    assert outcome.retrieved_ids is None


def _failing_store_factory(bad_embedder):
    """Factory returning a real TieredMemoryStore with an always-raising embedder."""
    import src.store.memory_store as ms

    original = ms.TieredMemoryStore

    def factory(*args, **kwargs):
        kwargs["embedder"] = bad_embedder
        return original(*args, **kwargs)

    return factory


# ---------------------------------------------------------------------------
# Unavailable dependencies are reported, not silently degraded
# ---------------------------------------------------------------------------


def test_none_requires_an_embedder_and_says_so(tmp_path, monkeypatch, capsys):
    from src.eval.run_benchmark import _build_strategy

    # Simulate an environment with neither sentence-transformers nor fastembed.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setitem(sys.modules, "fastembed", None)

    assert _build_strategy("none", 512, str(tmp_path)) is None
    err = capsys.readouterr().err
    assert "embedder" in err

    # An injected embedder bypasses the default-chain check entirely.
    strategy = _build_strategy(
        "none", 512, str(tmp_path), embedder=_HashEmbedder()
    )
    assert strategy is not None


# ---------------------------------------------------------------------------
# CLI-level: real dataset + real none strategy exits 0 with full metrics
# ---------------------------------------------------------------------------


def test_main_locomo_none_runs_end_to_end(tmp_path, capsys, monkeypatch):
    from src.eval.run_benchmark import main

    _write_locomo_fixture(tmp_path)
    fake = _FakeLLM(answer="Chennai")
    _patch_llm(monkeypatch, fake)
    _patch_store_embedder(monkeypatch, _HashEmbedder())

    code = main(
        ["--dataset", "locomo", "--baseline", "none", "--data-dir", str(tmp_path)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Loaded locomo: 2 cases" in out
    assert "baseline=none" in out
    assert "Mean F1" in out
    assert "Mean BLEU-1" in out
    assert "Mean Recall@5: 1.000" in out  # non-degenerate retrieval metrics
    assert "Mean Precision@5: 0.750" in out  # (1/2 + 1/1) / 2
    assert "MRR: 1.000" in out
    assert fake.rerank_prompts and fake.answer_prompts


def test_main_longmemeval_none_runs_end_to_end(tmp_path, capsys, monkeypatch):
    from src.eval.run_benchmark import main

    _write_longmemeval_fixture(tmp_path)
    _patch_llm(monkeypatch, _FakeLLM(answer="Whiskers"))
    _patch_store_embedder(monkeypatch, _HashEmbedder())

    code = main(
        ["--dataset", "longmemeval", "--baseline", "none", "--data-dir", str(tmp_path)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Loaded longmemeval: 1 cases" in out
    assert "MRR: 1.000" in out


# ---------------------------------------------------------------------------
# Synthetic behavior remains untouched
# ---------------------------------------------------------------------------


def test_synthetic_none_still_uses_demo_pipeline(capsys):
    from src.eval.run_benchmark import main

    code = main(["--dataset", "synthetic", "--baseline", "none"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Synthetic benchmark (synthetic_demo, baseline=none)" in out
    assert "Mean F1" in out
