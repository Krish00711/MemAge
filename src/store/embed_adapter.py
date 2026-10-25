"""Lightweight embedding adapter for MemAge real mode.

FastEmbedAdapter wraps ``fastembed.TextEmbedding`` (BAAI/bge-small-en-v1.5,
~130 MB) into the simple ``encode(text: str) -> list[float]`` protocol that
TieredMemoryStore expects.

fastembed's native ``embed()`` returns a generator of numpy arrays, so we
consume the first (and only) item and convert it to a plain Python list.

Usage (in TieredMemoryStore or server startup):
    from src.store.embed_adapter import FastEmbedAdapter
    store = TieredMemoryStore(embedder=FastEmbedAdapter())

No API key required.  Downloads the model on first call to encode().
"""
from __future__ import annotations

__all__ = ["FastEmbedAdapter"]


class FastEmbedAdapter:
    """Wraps ``fastembed.TextEmbedding`` as an ``encode(text) -> list[float]`` object.

    Parameters
    ----------
    model_name:
        Any fastembed-supported model.  Default is the small BGE model
        (~130 MB on first download, stored in ~/.cache/fastembed).
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from fastembed import TextEmbedding  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "fastembed is not installed. "
                "Run: pip install fastembed"
            ) from exc
        self._model = TextEmbedding(model_name)

    def encode(self, text: str) -> list[float]:
        """Embed a single string and return a plain Python list of floats.

        fastembed.TextEmbedding.embed() returns a generator of numpy arrays;
        we take the first element and call .tolist() on it.
        """
        vec = next(self._model.embed([text]))
        # vec is a numpy ndarray; .tolist() gives list[float]
        return vec.tolist()
