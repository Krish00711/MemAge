from src.store.memory_store import TieredMemoryStore


def make_item(i, text):
    return {"id": i, "text": text, "timestamp": "2026-01-01T00:00:00",
            "source": "session_1_turn_1", "embedding": None}


def test_add_query_update_delete(tmp_path):
    store = TieredMemoryStore(persist_dir=str(tmp_path))

    store.add(make_item("a", "User lives in Chennai"))
    store.add(make_item("b", "User likes playing cricket"))

    hits = store.query("Where does the user live?", k=1)
    assert hits[0]["id"] == "a"

    assert store.update("a", "User lives in Bangalore")
    assert store.get("a")["text"] == "User lives in Bangalore"

    assert store.delete("b")
    assert store.get("b") is None
    assert store.count()["STM"] == 1