from src.store.memory_store import TieredMemoryStore


def make_item(i, text, ts="2026-01-01T00:00:00"):
    return {"id": i, "text": text, "timestamp": ts,
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


def test_overflow_moves_oldest_to_mtm(tmp_path):
    store = TieredMemoryStore(persist_dir=str(tmp_path), stm_capacity=2)
    for n in range(3):
        store.add(make_item(f"m{n}", f"fact number {n}", ts=f"2026-01-0{n + 1}T00:00:00"))
    assert store.count() == {"STM": 2, "MTM": 1, "LTM": 0}
    assert store.get("m0")["tier"] == "MTM"


def test_frequent_access_promotes_to_ltm(tmp_path):
    store = TieredMemoryStore(persist_dir=str(tmp_path), stm_capacity=1,
                              ltm_access_threshold=2)
    store.add(make_item("a", "User lives in Chennai", ts="2026-01-01T00:00:00"))
    store.add(make_item("b", "User likes cricket", ts="2026-01-02T00:00:00"))
    assert store.get("a")["tier"] == "MTM"      # pushed out of STM

    store.query("Where does the user live?", k=1, tier="MTM")
    store.query("Where does the user live?", k=1, tier="MTM")
    assert store.get("a")["tier"] == "LTM"      # retrieved twice -> promoted