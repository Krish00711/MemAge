import sys
import builtins
import types
import pytest

from src.store.memory_store import TieredMemoryStore


def test_fallback_no_sentence_no_fastembed(monkeypatch):
    """When neither sentence_transformers nor fastembed are importable, init raises ImportError."""
    # Block all three layers of the fallback chain
    monkeypatch.setitem(sys.modules, 'sentence_transformers', None)  # type: ignore
    monkeypatch.setitem(sys.modules, 'fastembed', None)              # type: ignore
    monkeypatch.setitem(sys.modules, 'src.store.embed_adapter', None)  # type: ignore
    # Also reload memory_store so it sees the patched sys.modules
    sys.modules.pop('src.store.memory_store', None)
    from src.store.memory_store import TieredMemoryStore
    with pytest.raises(ImportError) as exc:
        TieredMemoryStore(persist_dir=':memory:')
    assert 'fastembed' in str(exc.value) or 'embedder' in str(exc.value)


def test_fallback_fastembed_available(monkeypatch):
    """When fastembed is available, TieredMemoryStore should use FastEmbedAdapter."""
    # Mock sentence_transformers import to raise ImportError
    original_import = builtins.__import__
    def selective_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'sentence_transformers':
            raise ImportError
        return original_import(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, '__import__', selective_import)

    # Build a fake fastembed module — TextEmbedding.embed() returns a generator of
    # objects that have a .tolist() method (mimicking numpy ndarray interface).
    fake_fastembed = types.ModuleType('fastembed')

    class _FakeVec:
        """A plain-Python stand-in for numpy ndarray (has .tolist() method)."""
        def __init__(self, data):
            self._data = data
        def tolist(self):
            return list(self._data)

    class DummyTextEmbedding:
        DIM = 384
        def __init__(self, model_name):
            self.model_name = model_name
        def embed(self, texts):
            for _ in texts:
                yield _FakeVec([0.1] * self.DIM)

    fake_fastembed.TextEmbedding = DummyTextEmbedding
    monkeypatch.setitem(sys.modules, 'fastembed', fake_fastembed)

    # Clear embed_adapter and memory_store caches so they re-import with our mock
    sys.modules.pop('src.store.embed_adapter', None)
    sys.modules.pop('src.store.memory_store', None)

    from src.store.embed_adapter import FastEmbedAdapter
    from src.store.memory_store import TieredMemoryStore as _Store
    store = _Store(persist_dir=':memory:')
    assert isinstance(store._embedder, FastEmbedAdapter)
    vec = store._embedder.encode('test')
    assert isinstance(vec, list) and len(vec) == DummyTextEmbedding.DIM
    assert all(isinstance(v, float) for v in vec)
