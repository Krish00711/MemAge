from src.extraction.extractor import extract_candidates

META = {"speaker": "Maya", "timestamp": "2023-03-05T10:00:00", "source": "D1:3"}


def test_returns_candidate_items():
    fake = lambda prompt, system=None, **kw: '["Maya adopted a dog named Biscuit."]'
    items = extract_candidates("I adopted a dog named Biscuit!", META, llm=fake)
    assert len(items) == 1
    assert items[0] == {
        "id": "D1:3__0",
        "text": "Maya adopted a dog named Biscuit.",
        "timestamp": "2023-03-05T10:00:00",
        "source": "D1:3",
        "embedding": None,
    }


def test_handles_code_fences_and_chatter():
    fake = lambda prompt, system=None, **kw: 'Sure!\n```json\n["Fact one.", "Fact two."]\n```'
    items = extract_candidates("hello", META, llm=fake)
    assert [i["text"] for i in items] == ["Fact one.", "Fact two."]


def test_small_talk_returns_empty():
    fake = lambda prompt, system=None, **kw: "[]"
    assert extract_candidates("hi!", META, llm=fake) == []


def test_retries_once_then_gives_up():
    calls = []

    def bad(prompt, system=None, **kw):
        calls.append(1)
        return "I cannot do that"

    assert extract_candidates("hello", META, llm=bad) == []
    assert len(calls) == 2