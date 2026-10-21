"""Tiered memory store (STM / MTM / LTM) backed by ChromaDB.

Owner: Aryan. Interface contract: docs/interfaces.md
"""
from __future__ import annotations

import chromadb
from sentence_transformers import SentenceTransformer

TIERS = ("STM", "MTM", "LTM")

# Tier rules: tune these during benchmarking
STM_CAPACITY = 20           # newest memories live here
MTM_CAPACITY = 100
LTM_ACCESS_THRESHOLD = 3    # retrieved this many times -> promote MTM -> LTM


class TieredMemoryStore:
    def __init__(self, persist_dir: str = "data/chroma", embedder=None,
                 stm_capacity: int = STM_CAPACITY,
                 mtm_capacity: int = MTM_CAPACITY,
                 ltm_access_threshold: int = LTM_ACCESS_THRESHOLD):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embedder = embedder or SentenceTransformer("all-MiniLM-L6-v2")
        self.stm_capacity = stm_capacity
        self.mtm_capacity = mtm_capacity
        self.ltm_access_threshold = ltm_access_threshold
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
            "access_count": meta.get("access_count", 0),
        }

    def _oldest(self, tier: str, n: int) -> list[str]:
        """IDs of the n oldest memories in a tier (ISO timestamps sort correctly)."""
        res = self._cols[tier].get(include=["metadatas"])
        pairs = sorted(zip(res["ids"], res["metadatas"]),
                       key=lambda p: p[1].get("timestamp", ""))
        return [item_id for item_id, _ in pairs[:n]]

    def _bump_access(self, hit: dict) -> None:
        """Count one retrieval of this memory."""
        col = self._cols[hit["tier"]]
        meta = col.get(ids=[hit["id"]])["metadatas"][0]
        meta["access_count"] = meta.get("access_count", 0) + 1
        col.update(ids=[hit["id"]], metadatas=[meta])
        hit["access_count"] = meta["access_count"]

    # ---------- tier logic ----------
    def move(self, item_id: str, to_tier: str) -> bool:
        """Move a memory to another tier, keeping its embedding."""
        if to_tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}")
        for tier, col in self._cols.items():
            res = col.get(ids=[item_id],
                          include=["documents", "embeddings", "metadatas"])
            if not res["ids"]:
                continue
            if tier == to_tier:
                return True
            self._cols[to_tier].add(
                ids=[item_id],
                documents=res["documents"],
                embeddings=[[float(x) for x in res["embeddings"][0]]],
                metadatas=res["metadatas"],
            )
            col.delete(ids=[item_id])
            return True
        return False

    def rebalance(self) -> None:
        """Apply the tier rules. Called automatically after add() and query()."""
        # 1. Frequently retrieved MTM memories go to LTM
        res = self._cols["MTM"].get(include=["metadatas"])
        for item_id, meta in zip(res["ids"], res["metadatas"]):
            if meta.get("access_count", 0) >= self.ltm_access_threshold:
                self.move(item_id, "LTM")
        # 2. Overflow cascades down: STM -> MTM -> LTM (oldest first)
        for src, dst, cap in (("STM", "MTM", self.stm_capacity),
                              ("MTM", "LTM", self.mtm_capacity)):
            extra = self._cols[src].count() - cap
            if extra > 0:
                for item_id in self._oldest(src, extra):
                    self.move(item_id, dst)

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
                "access_count": 0,
            }],
        )
        self.rebalance()
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
        top = hits[:k]
        for hit in top:
            self._bump_access(hit)
        self.rebalance()
        return top

    def count(self) -> dict:
        return {t: c.count() for t, c in self._cols.items()}