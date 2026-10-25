"""Offline tests for TieredMemoryStore dependency injection.

Verify that:
  - memory_store module imports without sentence-transformers installed.
  - TieredMemoryStore can be constructed with an injected fake embedder
    without importing sentence-transformers.
  - The injected embedder is actually used (not bypassed).
  - Passing embedder=None without sentence-transformers raises ImportError
    with a clear message, not a silent failure.
  - All store methods work end-to-end with a fake embedder + fake chromadb.

No real ChromaDB, no real embeddings, no API keys required.

Run: python -m pytest tests/test_store_injection.py -q
"""
from __future__ import annotations

import sys
import types
import importlib
import unittest.mock as mock

import pytest


# ---------------------------------------------------------------------------
# Helpers — fake chromadb and fake embedder
# ---------------------------------------------------------------------------

def _fake_chromadb_module():
    """Build a minimal stub of the chromadb module."""
    chromadb = types.ModuleType("chromadb")

    class _FakeCollection:
        def __init__(self):
            self._docs: dict = {}  # id -> {doc, embedding, metadata}

        def add(self, ids, documents, embeddings, metadatas):
            for i, (doc_id, doc, emb, meta) in enumerate(
                zip(ids, documents, embeddings, metadatas)
            ):
                self._docs[doc_id] = {"doc": doc, "embedding": emb, "meta": meta}

        def get(self, ids=None, include=None):
            if ids:
                hits = [(k, v) for k, v in self._docs.items() if k in ids]
            else:
                hits = list(self._docs.items())
            return {
                "ids": [h[0] for h in hits],
                "documents": [h[1]["doc"] for h in hits],
                "metadatas": [h[1]["meta"] for h in hits],
                "embeddings": [h[1]["embedding"] for h in hits],
            }

        def update(self, ids, documents=None, embeddings=None, metadatas=None):
            for i, doc_id in enumerate(ids):
                if doc_id in self._docs:
                    if documents:
                        self._docs[doc_id]["doc"] = documents[i]
                    if embeddings:
                        self._docs[doc_id]["embedding"] = embeddings[i]
                    if metadatas:
                        self._docs[doc_id]["meta"] = metadatas[i]

        def delete(self, ids):
            for doc_id in ids:
                self._docs.pop(doc_id, None)

        def count(self):
            return len(self._docs)

        def query(self, query_embeddings, n_results):
            # Return all items sorted by fake distance (0.0 for all)
            items = list(self._docs.items())[:n_results]
            return {
                "ids": [[i for i, _ in items]],
                "documents": [[v["doc"] for _, v in items]],
                "metadatas": [[v["meta"] for _, v in items]],
                "distances": [[0.1 * (idx + 1) for idx in range(len(items))]],
            }

    class _FakeClient:
        def __init__(self, path=None):
            self._collections: dict[str, _FakeCollection] = {}

        def get_or_create_collection(self, name, metadata=None):
            if name not in self._collections:
                self._collections[name] = _FakeCollection()
            return self._collections[name]

    chromadb.PersistentClient = _FakeClient
    return chromadb


class _FakeEmbedder:
    """Deterministic fake embedder: encodes text as a fixed-dim float list."""

    CALLS: list[str] = []

    def encode(self, text: str) -> list[float]:
        self.CALLS.append(text)
        # 4-dim vector derived from text length — deterministic, not random
        n = len(text)
        return [float(n % 10) / 10, float((n * 3) % 10) / 10,
                float((n * 7) % 10) / 10, float((n * 13) % 10) / 10]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_chromadb(monkeypatch):
    """Inject the fake chromadb into sys.modules for the duration of the test."""
    fake = _fake_chromadb_module()
    monkeypatch.setitem(sys.modules, "chromadb", fake)
    # Reload memory_store so it picks up the patched chromadb
    if "src.store.memory_store" in sys.modules:
        del sys.modules["src.store.memory_store"]
    yield fake
    if "src.store.memory_store" in sys.modules:
        del sys.modules["src.store.memory_store"]


@pytest.fixture
def store(patched_chromadb):
    """A TieredMemoryStore with fake chromadb + fake embedder, no real deps."""
    from src.store.memory_store import TieredMemoryStore
    embedder = _FakeEmbedder()
    embedder.CALLS.clear()
    return TieredMemoryStore(persist_dir="/tmp/fake", embedder=embedder)


def _item(item_id="t__0", text="Maya lives in Chennai",
          ts="2026-01-01T00:00:00", source="t"):
    return {"id": item_id, "text": text, "timestamp": ts,
            "source": source, "embedding": None}


# ---------------------------------------------------------------------------
# Import-level tests (no chromadb needed at module level)
# ---------------------------------------------------------------------------

class TestModuleImport:
    def test_module_imports_without_sentence_transformers(self, monkeypatch):
        """memory_store.py must import cleanly even when sentence-transformers absent."""
        # Remove sentence_transformers from sys.modules if present
        monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
        # Also block the import
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # type: ignore

        fake_chroma = _fake_chromadb_module()
        monkeypatch.setitem(sys.modules, "chromadb", fake_chroma)
        if "src.store.memory_store" in sys.modules:
            del sys.modules["src.store.memory_store"]

        # Should not raise ImportError at import time
        mod = importlib.import_module("src.store.memory_store")
        assert hasattr(mod, "TieredMemoryStore")

    def test_no_sentence_transformers_in_module_globals(self):
        """SentenceTransformer must not appear in module-level namespace."""
        import src.store.memory_store as m
        assert not hasattr(m, "SentenceTransformer"), (
            "SentenceTransformer was imported at module level — "
            "this prevents injection-only use without the package."
        )


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_injected_embedder_accepted(self, patched_chromadb):
        from src.store.memory_store import TieredMemoryStore
        embedder = _FakeEmbedder()
        store = TieredMemoryStore(persist_dir="/tmp/fake", embedder=embedder)
        assert store._embedder is embedder

    def test_injected_embedder_used_not_sentence_transformers(self, store):
        """Calling add() must invoke the injected embedder, not SentenceTransformer."""
        store._embedder.CALLS.clear()
        store.add(_item())
        assert len(store._embedder.CALLS) >= 1
        assert "Maya lives in Chennai" in store._embedder.CALLS

    def test_no_embedder_without_sentence_transformers_raises_import_error(
        self, patched_chromadb, monkeypatch
    ):
        """embedder=None without sentence-transformers or fastembed must raise ImportError."""
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # type: ignore
        monkeypatch.setitem(sys.modules, "fastembed", None)  # type: ignore
        # Also block embed_adapter so the fallback chain fully exhausts
        monkeypatch.setitem(sys.modules, "src.store.embed_adapter", None)  # type: ignore
        if "src.store.memory_store" in sys.modules:
            del sys.modules["src.store.memory_store"]

        from src.store.memory_store import TieredMemoryStore
        with pytest.raises(ImportError, match="fastembed"):
            TieredMemoryStore(persist_dir="/tmp/fake")

    def test_import_error_message_mentions_inject(self, patched_chromadb, monkeypatch):
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # type: ignore
        monkeypatch.setitem(sys.modules, "fastembed", None)  # type: ignore
        monkeypatch.setitem(sys.modules, "src.store.embed_adapter", None)  # type: ignore
        if "src.store.memory_store" in sys.modules:
            del sys.modules["src.store.memory_store"]
        from src.store.memory_store import TieredMemoryStore
        with pytest.raises(ImportError, match="embedder"):
            TieredMemoryStore(persist_dir="/tmp/fake")


# ---------------------------------------------------------------------------
# Functional tests with fake embedder + fake chromadb
# ---------------------------------------------------------------------------

class TestStoreWithFakeEmbedder:
    def test_add_returns_stored_dict(self, store):
        result = store.add(_item())
        assert result["id"] == "t__0"
        assert result["text"] == "Maya lives in Chennai"
        assert isinstance(result["embedding"], list)
        assert len(result["embedding"]) == 4  # fake embedder dim

    def test_get_retrieves_added_item(self, store):
        store.add(_item())
        got = store.get("t__0")
        assert got is not None
        assert got["text"] == "Maya lives in Chennai"
        assert got["tier"] == "STM"

    def test_get_missing_returns_none(self, store):
        assert store.get("nonexistent") is None

    def test_count_increments_on_add(self, store):
        before = sum(store.count().values())
        store.add(_item())
        after = sum(store.count().values())
        assert after == before + 1

    def test_count_returns_all_tiers(self, store):
        counts = store.count()
        assert set(counts.keys()) == {"STM", "MTM", "LTM"}

    def test_delete_removes_item(self, store):
        store.add(_item())
        ok = store.delete("t__0")
        assert ok is True
        assert store.get("t__0") is None

    def test_delete_missing_returns_false(self, store):
        assert store.delete("nope") is False

    def test_update_replaces_text(self, store):
        store.add(_item())
        ok = store.update("t__0", "Maya lives in Bangalore")
        assert ok is True
        got = store.get("t__0")
        assert got["text"] == "Maya lives in Bangalore"

    def test_update_missing_returns_false(self, store):
        assert store.update("nope", "text") is False

    def test_query_returns_list_of_dicts(self, store):
        store.add(_item("a", "fact one"))
        store.add(_item("b", "fact two"))
        results = store.query("fact", k=2)
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)
        assert all("id" in r and "text" in r and "tier" in r for r in results)

    def test_query_respects_k(self, store):
        for i in range(5):
            store.add(_item(f"x{i}", f"fact {i}"))
        results = store.query("fact", k=2)
        assert len(results) <= 2

    def test_tier_filter_all(self, store):
        store.add(_item())
        results = store.query("Maya", k=5, tier="all")
        assert len(results) >= 1

    def test_embedder_called_on_update(self, store):
        store.add(_item())
        store._embedder.CALLS.clear()
        store.update("t__0", "new text here")
        assert "new text here" in store._embedder.CALLS

    def test_embedder_called_on_query(self, store):
        store.add(_item())
        store._embedder.CALLS.clear()
        store.query("search text", k=1)
        assert "search text" in store._embedder.CALLS


# ---------------------------------------------------------------------------
# Fake embedder protocol test
# ---------------------------------------------------------------------------

class TestFakeEmbedderProtocol:
    """Document the minimum interface a real lightweight embedder must satisfy."""

    def test_encode_returns_list_of_floats(self):
        e = _FakeEmbedder()
        result = e.encode("hello world")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)
        assert len(result) > 0

    def test_fastembed_interface_matches(self):
        """fastembed.TextEmbedding also has encode(); document the adapter needed.

        fastembed returns a generator of np.ndarray, so a one-line adapter is
        required:
            class FastEmbedAdapter:
                def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
                    from fastembed import TextEmbedding
                    self._model = TextEmbedding(model_name)
                def encode(self, text: str) -> list[float]:
                    return list(next(self._model.embed([text])).tolist())

        This test documents the requirement; it does not import fastembed.
        """
        # The protocol: encode(str) -> list[float]
        class _MockFastEmbed:
            def encode(self, text: str) -> list[float]:
                return [0.1, 0.2, 0.3, 0.4]
        e = _MockFastEmbed()
        result = e.encode("any text")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)
