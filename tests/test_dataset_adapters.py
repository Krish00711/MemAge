"""Tests for src.eval.adapters — LoCoMo / LongMemEval loaders.

Fixtures follow the official schemas documented in the adapters module
docstring (snap-research/locomo, xiaowu0162/LongMemEval). No network, no
real dataset files required.
"""

import json

import pytest

from src.eval.adapters import (
    DatasetFileNotFoundError,
    DatasetFormatError,
    load_dataset,
    load_locomo,
    load_longmemeval,
    resolve_dataset_file,
    to_eval_dataset,
)
from src.eval.dataset import RetrievalEvalCase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def locomo_fixture():
    return [
        {
            "sample_id": "sample_1",
            "conversation": {
                # session_2 intentionally listed before session_1 to prove
                # numeric session ordering in the adapter.
                "speaker_a": "Caroline",
                "speaker_b": "Melanie",
                "session_2_date_time": "9:00 am on 12 May, 2023",
                "session_2": [
                    {"speaker": "Caroline", "dia_id": "D2:1", "text": "I adopted a cat named Momo."},
                    # image-only turn: text absent, blip_caption used
                    {"speaker": "Melanie", "dia_id": "D2:2", "img_url": ["u"], "blip_caption": "a photo of a cat"},
                    # neither text nor caption -> skipped
                    {"speaker": "Melanie", "dia_id": "D2:3"},
                ],
                "session_1_date_time": "10:30 am on 7 May, 2023",
                "session_1": [
                    {"speaker": "Caroline", "dia_id": "D1:1", "text": "Hey Melanie! How's it going?"},
                    {"speaker": "Melanie", "dia_id": "D1:2", "text": "Going great!"},
                ],
            },
            "qa": [
                {"question": "What is the name of Caroline's cat?", "answer": "Momo", "evidence": ["D2:1"], "category": 3},
                {"question": "How is Melanie doing?", "answer": "Going great", "evidence": ["D1:2"], "category": 1},
            ],
        },
        {
            "sample_id": "sample_2",
            "conversation": {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "I live in Berlin"}],
                "session_1_date_time": "morning",
            },
            "qa": [
                # adversarial-style: no evidence, adversarial_answer instead
                {"question": "Where does A live?", "adversarial_answer": "Somewhere in Europe", "category": 4},
                # malformed entries -> skipped
                {"answer": "orphan answer"},
                {"question": "no answer here"},
                {"question": 123, "answer": "x"},
                "not-an-object",
                {"question": "Bad evidence", "answer": "y", "evidence": "D1:1"},
            ],
        },
    ]


def longmemeval_fixture():
    return [
        {
            "question_id": "q1",
            "question_type": "single-session-user",
            "question": "What did I cook last Friday?",
            "answer": "Pasta",
            "question_date": "2023/05/20 (Sat) 14:00",
            "haystack_session_ids": ["s2", "s1"],
            "haystack_dates": ["2023/05/12", "2023/05/05"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I made pancakes"},
                    {"role": "assistant", "content": "Nice!"},
                ],
                [
                    {"role": "user", "content": "I cooked pasta"},
                    {"role": "assistant", "content": "Yum", "has_answer": True},
                    {"role": "assistant", "content": None},  # malformed turn -> skipped
                ],
            ],
            "answer_session_ids": ["s1"],
        },
        {
            # abstention question: empty evidence list is valid
            "question_id": "q2_abs",
            "question_type": "abstention",
            "question": "What is my dog's name?",
            "answer": "I don't know",
            "question_date": "2023/06/01",
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2023/01/01"],
            "haystack_sessions": [[{"role": "user", "content": "hello"}]],
            "answer_session_ids": [],
        },
        # malformed records -> skipped
        {"question_id": "bad1", "answer": "x", "haystack_sessions": [], "answer_session_ids": []},
        {"question_id": "bad2", "question": "q", "haystack_sessions": [], "answer_session_ids": []},
        {"question_id": "bad3", "question": "q", "answer": "a", "answer_session_ids": ["s1"]},
        {"question_id": "bad4", "question": "q", "answer": "a", "haystack_sessions": [[]]},
    ]


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# LoCoMo
# ---------------------------------------------------------------------------


def test_locomo_loads_valid_cases(tmp_path):
    path = write_json(tmp_path / "locomo" / "locomo10.json", locomo_fixture())
    ds = load_locomo(path)

    assert len(ds) == 3  # 2 valid in sample_1 + 1 valid in sample_2
    assert ds.name == "locomo"

    first = ds[0]
    assert first.query == "What is the name of Caroline's cat?"
    assert first.reference_answer == "Momo"
    assert first.relevant_ids == ["D2:1"]
    assert first.case_id == "sample_1::qa0"
    assert first.conversation_id == "sample_1"
    assert first.metadata["dataset"] == "locomo"
    assert first.metadata["category"] == 3
    assert first.metadata["adversarial"] is False

    # conversation turns: session_1 (numeric order) before session_2,
    # image turn kept via blip_caption, empty turn dropped
    turns = first.metadata["turns"]
    assert [t["id"] for t in turns] == ["D1:1", "D1:2", "D2:1", "D2:2"]
    assert turns[0]["session"] == "session_1"
    assert turns[0]["timestamp"] == "10:30 am on 7 May, 2023"
    assert turns[2]["session"] == "session_2"
    assert turns[3]["text"] == "a photo of a cat"

    # adversarial-style entry: no evidence -> empty relevant_ids, fallback answer
    adversarial = ds[2]
    assert adversarial.case_id == "sample_2::qa0"
    assert adversarial.relevant_ids == []
    assert adversarial.reference_answer == "Somewhere in Europe"
    assert adversarial.metadata["adversarial"] is True


def test_locomo_skips_malformed_qa_entries(tmp_path):
    path = write_json(tmp_path / "locomo10.json", locomo_fixture())
    ds = load_locomo(path)
    # only 3 valid of 8 total qa entries
    assert len(ds) == 3
    queries = [c.query for c in ds]
    assert "no answer here" not in queries
    assert all(isinstance(c, type(ds[0])) for c in ds)


def test_locomo_root_must_be_array(tmp_path):
    path = write_json(tmp_path / "locomo10.json", {"not": "a list"})
    with pytest.raises(DatasetFormatError, match="JSON array"):
        load_locomo(path)


def test_locomo_invalid_json(tmp_path):
    p = tmp_path / "locomo10.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(DatasetFormatError, match="parse"):
        load_locomo(p)


def test_locomo_missing_file(tmp_path):
    with pytest.raises(DatasetFileNotFoundError):
        load_locomo(tmp_path / "does_not_exist.json")


def test_locomo_malformed_sample_container(tmp_path):
    bad_conv = [{"sample_id": "s", "conversation": "oops", "qa": []}]
    path = write_json(tmp_path / "locomo10.json", bad_conv)
    with pytest.raises(DatasetFormatError, match="conversation"):
        load_locomo(path)

    bad_qa = [{"sample_id": "s", "conversation": {"session_1": []}, "qa": "oops"}]
    path = write_json(tmp_path / "locomo10.json", bad_qa)
    with pytest.raises(DatasetFormatError, match="'qa'"):
        load_locomo(path)


def test_locomo_non_object_sample(tmp_path):
    path = write_json(tmp_path / "locomo10.json", ["just-a-string"])
    with pytest.raises(DatasetFormatError, match="JSON object"):
        load_locomo(path)


def test_locomo_zero_valid_cases_raises(tmp_path):
    empty_qa = [{"sample_id": "s", "conversation": {"session_1": []}, "qa": [{"answer": "x"}]}]
    path = write_json(tmp_path / "locomo10.json", empty_qa)
    with pytest.raises(DatasetFormatError, match="No valid"):
        load_locomo(path)


# ---------------------------------------------------------------------------
# LongMemEval
# ---------------------------------------------------------------------------


def test_longmemeval_loads_valid_cases(tmp_path):
    path = write_json(tmp_path / "longmemeval_s.json", longmemeval_fixture())
    ds = load_longmemeval(path)

    assert len(ds) == 2  # 2 valid of 6 records
    assert ds.name == "longmemeval"

    first = ds[0]
    assert first.query == "What did I cook last Friday?"
    assert first.reference_answer == "Pasta"
    assert first.relevant_ids == ["s1"]
    assert first.case_id == "q1"
    assert first.conversation_id is None
    assert first.metadata["dataset"] == "longmemeval"
    assert first.metadata["question_type"] == "single-session-user"
    assert first.metadata["question_date"] == "2023/05/20 (Sat) 14:00"

    # parallel arrays zipped by index: haystack order preserved (s2 first)
    sessions = first.metadata["sessions"]
    assert [s["session_id"] for s in sessions] == ["s2", "s1"]
    assert sessions[0]["date"] == "2023/05/12"
    assert sessions[1]["date"] == "2023/05/05"
    assert len(sessions[0]["turns"]) == 2
    assert len(sessions[1]["turns"]) == 2  # malformed turn dropped
    assert sessions[1]["turns"][1]["has_answer"] is True

    # abstention: empty answer_session_ids kept as-is
    abstention = ds[1]
    assert abstention.case_id == "q2_abs"
    assert abstention.relevant_ids == []
    assert abstention.reference_answer == "I don't know"


def test_longmemeval_skips_malformed_records(tmp_path):
    path = write_json(tmp_path / "longmemeval_s.json", longmemeval_fixture())
    ds = load_longmemeval(path)
    assert [c.case_id for c in ds] == ["q1", "q2_abs"]


def test_longmemeval_root_must_be_array(tmp_path):
    path = write_json(tmp_path / "l.json", {"question_id": "x"})
    with pytest.raises(DatasetFormatError, match="JSON array"):
        load_longmemeval(path)


def test_longmemeval_non_object_record(tmp_path):
    path = write_json(tmp_path / "l.json", ["nope"])
    with pytest.raises(DatasetFormatError, match="JSON object"):
        load_longmemeval(path)


def test_longmemeval_missing_file(tmp_path):
    with pytest.raises(DatasetFileNotFoundError):
        load_longmemeval(tmp_path / "none.json")


def test_longmemeval_zero_valid_cases_raises(tmp_path):
    path = write_json(tmp_path / "l.json", [{"question": "", "answer": ""}])
    with pytest.raises(DatasetFormatError, match="No valid"):
        load_longmemeval(path)


# ---------------------------------------------------------------------------
# Path resolution / dispatch
# ---------------------------------------------------------------------------


def test_resolve_prefers_documented_filenames(tmp_path):
    write_json(tmp_path / "locomo" / "locomo10.json", locomo_fixture())
    write_json(tmp_path / "locomo" / "other.json", [])
    assert resolve_dataset_file("locomo", tmp_path).name == "locomo10.json"

    write_json(tmp_path / "longmemeval" / "longmemeval_s.json", [])
    write_json(tmp_path / "longmemeval" / "zzz_custom.json", [])
    assert resolve_dataset_file("longmemeval", tmp_path).name == "longmemeval_s.json"


def test_resolve_falls_back_to_any_json(tmp_path):
    write_json(tmp_path / "locomo" / "custom_export.json", locomo_fixture())
    assert resolve_dataset_file("locomo", tmp_path).name == "custom_export.json"


def test_resolve_missing_directory_and_files(tmp_path):
    with pytest.raises(DatasetFileNotFoundError, match="directory not found"):
        resolve_dataset_file("locomo", tmp_path)
    (tmp_path / "locomo").mkdir()
    with pytest.raises(DatasetFileNotFoundError, match="No .json"):
        resolve_dataset_file("locomo", tmp_path)


def test_resolve_unknown_dataset(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset"):
        resolve_dataset_file("squad", tmp_path)


def test_load_dataset_dispatch(tmp_path):
    write_json(tmp_path / "locomo" / "locomo10.json", locomo_fixture())
    ds = load_dataset("locomo", tmp_path)
    assert len(ds) == 3

    write_json(tmp_path / "longmemeval" / "longmemeval_s.json", longmemeval_fixture())
    ds2 = load_dataset("longmemeval", tmp_path)
    assert len(ds2) == 2

    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("unknown_ds", tmp_path)


def test_to_eval_dataset_conversion(tmp_path):
    path = write_json(tmp_path / "locomo10.json", locomo_fixture())
    benchmark_ds = load_locomo(path)
    eval_ds = to_eval_dataset(benchmark_ds)

    assert len(eval_ds) == len(benchmark_ds)
    assert eval_ds.name == benchmark_ds.name
    assert all(isinstance(c, RetrievalEvalCase) for c in eval_ds)
    for bench, retr in zip(benchmark_ds, eval_ds):
        assert retr.query == bench.query
        assert retr.relevant_ids == bench.relevant_ids
        assert retr.case_id == bench.case_id
        assert retr.metadata["dataset"] == "locomo"
    assert not eval_ds.is_empty()
