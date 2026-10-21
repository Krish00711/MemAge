"""Tiered memory store (STM / MTM / LTM) backed by ChromaDB.

Owner: Aryan. Interface contract: docs/interfaces.md
"""
from __future__ import annotations

import chromadb
from sentence_transformers import SentenceTransformer

TIERS = ("STM", "MTM", "LTM")


class TieredMemoryStore:
    def __init__(self, persist_dir: str = "data/chroma", embedder=None):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embedder = embedder or SentenceTransformer("all-MiniLM-L6-v2")
        # One Chroma collection per tier
        self._cols = {
            tier: self._client.get_or_create_collection(
                name=tier.lower(), metadata={"hnsw:space": "cosine"}
            )
            for tier in TIERS
        }

    # ---------- helpers ----------
    def _embed(self, text: str) -> list[float]:
        return self._embedder.encode(text).tolist()

    def _find(self, item_id: str):
        """Return (tier, record) for an id, or (None, None) if not stored."""
        for tier, col in self._cols.items():
            res = col.get(ids=[item_id])
            if res["ids"]:
                return tier, res
        return None, None

    @staticmethod
    def _to_item(item_id, text, meta, tier, distance=None) -> dict:
        return {
            "id": item_id,
            "text": text,
            "timestamp": meta.get("timestamp"),
            "source": meta.get("source"),
            "embedding": None,   # not returned, to keep results light
            "tier": tier,        # extra field: tell the team (see interfaces.md)
            "distance": distance,
        }

    # ---------- write operations ----------
    def add(self, item: dict, tier: str = "STM") -> dict:
        """Store a CandidateMemoryItem. The store fills in the embedding."""
        if tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}")
        embedding = self._embed(item["text"])
        self._cols[tier].add(
            ids=[item["id"]],
            documents=[item["text"]],
            embeddings=[embedding],
            metadatas=[{
                "timestamp": item.get("timestamp") or "",
                "source": item.get("source") or "",
            }],
        )
        stored = dict(item)
        stored["embedding"] = embedding
        return stored

    def update(self, item_id: str, new_text: str, timestamp: str | None = None) -> bool:
        """Replace the text of an existing memory (re-embeds it)."""
        tier, res = self._find(item_id)
        if tier is None:
            return False
        meta = res["metadatas"][0]
        if timestamp:
            meta["timestamp"] = timestamp
        self._cols[tier].update(
            ids=[item_id],
            documents=[new_text],
            embeddings=[self._embed(new_text)],
            metadatas=[meta],
        )
        return True

    def delete(self, item_id: str) -> bool:
        tier, _ = self._find(item_id)
        if tier is None:
            return False
        self._cols[tier].delete(ids=[item_id])
        return True

    # ---------- read operations ----------
    def get(self, item_id: str) -> dict | None:
        tier, res = self._find(item_id)
        if tier is None:
            return None
        return self._to_item(item_id, res["documents"][0], res["metadatas"][0], tier)

    def query(self, text: str, k: int = 5, tier: str = "all") -> list[dict]:
        """Return the k most similar memories (interface for src/retrieval/)."""
        tiers = TIERS if tier == "all" else (tier,)
        query_emb = self._embed(text)
        hits = []
        for t in tiers:
            col = self._cols[t]
            n = min(k, col.count())
            if n == 0:
                continue
            res = col.query(query_embeddings=[query_emb], n_results=n)
            for i, item_id in enumerate(res["ids"][0]):
                hits.append(self._to_item(
                    item_id, res["documents"][0][i], res["metadatas"][0][i],
                    t, res["distances"][0][i],
                ))
        hits.sort(key=lambda h: h["distance"])   # smaller distance = more similar
        return hits[:k]

    def count(self) -> dict:
        return {t: c.count() for t, c in self._cols.items()}