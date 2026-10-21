from src.extraction.locomo_loader import (
    parse_locomo_date, iter_turns, extract_conversation,
)

RECORD = {
    "sample_id": "conv-test",
    "conversation": {
        "speaker_a": "Caroline", "speaker_b": "Melanie",
        "session_2_date_time": "9:00 am on 10 May, 2023",
        "session_2": [{"speaker": "Melanie", "dia_id": "D2:1", "text": "I ran a race."}],
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [
            {"speaker": "Caroline", "dia_id": "D1:1", "text": "Hi Mel!"},
            {"speaker": "Melanie", "dia_id": "D1:2", "text": "I painted a sunrise.",
             "blip_caption": "a painting of a sunrise"},
        ],
    },
}


def test_parse_date():
    assert parse_locomo_date("1:56 pm on 8 May, 2023") == "2023-05-08T13:56:00"
    assert parse_locomo_date("weird text") == "weird text"
    assert parse_locomo_date(None) is None


def test_turns_in_session_order_with_metadata():
    turns = list(iter_turns(RECORD))
    assert [m["source"] for _, m in turns] == ["D1:1", "D1:2", "D2:1"]
    assert turns[1][1]["timestamp"] == "2023-05-08T13:56:00"
    assert "sunrise" in turns[1][0] and "shared an image" in turns[1][0]


def test_extraction_is_cached(tmp_path):
    calls = []

    def fake_llm(prompt, system=None, **kw):
        calls.append(1)
        return '["A fact."]'

    first = extract_conversation(RECORD, llm=fake_llm, cache_dir=str(tmp_path))
    n_calls = len(calls)
    second = extract_conversation(RECORD, llm=fake_llm, cache_dir=str(tmp_path))

    assert n_calls == 3                 # one LLM call per turn
    assert len(calls) == n_calls        # second run: all from cache
    assert first == second
    assert first[0]["source"] == "D1:1"
    assert first[0]["id"] == "D1:1__0"