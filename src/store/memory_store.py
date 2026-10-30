"""Tiered memory store (STM / MTM / LTM) backed by ChromaDB.

Owner: Aryan. Interface contract: docs/interfaces.md

Embedding note
--------------
By default the store uses ``sentence-transformers`` (all-MiniLM-L6-v2).
On disk-constrained machines use the bundled FastEmbedAdapter instead:

    from src.store.embed_adapter import FastEmbedAdapter
    store = TieredMemoryStore(embedder=FastEmbedAdapter())

Or inject any object with an ``encode(text: str) -> Iterable[float]`` method.
``sentence-transformers`` is NOT imported at module level so that the store
can be imported and injected with a custom embedder without it being installed.
"""
from __future__ import annotations
# chromadb will be imported lazily in TieredMemoryStore.__init__

TIERS = ("STM", "MTM", "LTM")

# Simple in‑memory collection used when chromadb is unavailable.
class _InMemoryCollection:
    def __init__(self):
        self._items: dict[str, dict] = {}

    def add(self, ids, documents, embeddings=None, metadatas=None):
        for i, _id in enumerate(ids):
            self._items[_id] = {
                "document": documents[i] if documents else "",
                "embedding": embeddings[i] if embeddings else None,
                "metadata": metadatas[i] if metadatas else {},
            }

    def delete(self, ids):
        for _id in ids:
            self._items.pop(_id, None)

    def update(self, ids, documents=None, embeddings=None, metadatas=None):
        for i, _id in enumerate(ids):
            if _id not in self._items:
                continue
            if documents:
                self._items[_id]["document"] = documents[i]
            if embeddings:
                self._items[_id]["embedding"] = embeddings[i]
            if metadatas:
                self._items[_id]["metadata"] = metadatas[i]

    def get(self, ids=None, include=None):
        # Chroma semantics: ids optional (None -> whole collection), and the
        # default include is ["metadatas", "documents"].
        if include is None:
            include = ["metadatas", "documents"]
        if ids is None:
            selected = list(self._items.items())
        else:
            selected = [(i, self._items[i]) for i in ids if i in self._items]
        result = {"ids": [i for i, _ in selected]}
        if "documents" in include:
            result["documents"] = [item["document"] for _, item in selected]
        if "embeddings" in include:
            result["embeddings"] = [item["embedding"] for _, item in selected]
        if "metadatas" in include:
            result["metadatas"] = [dict(item["metadata"]) for _, item in selected]
        return result

    def query(self, query_embeddings, n_results):
        query_vec = query_embeddings[0]
        distances = []
        for _id, item in self._items.items():
            emb = item.get("embedding")
            if emb is None:
                continue
            dist = sum((a - b) ** 2 for a, b in zip(emb, query_vec))
            distances.append((dist, _id))
        distances.sort(key=lambda x: x[0])
        top = distances[:n_results]
        ids = [i for _, i in top]
        docs = [self._items[i]["document"] for i in ids]
        embs = [self._items[i]["embedding"] for i in ids]
        metas = [dict(self._items[i]["metadata"]) for i in ids]
        return {"ids": [ids], "documents": [docs], "embeddings": [embs],
                "metadatas": [metas], "distances": [[d for d, _ in top]]}

    def count(self):
        return len(self._items)

# Tier rules: tune these during benchmarking
STM_CAPACITY = 20           # newest memories live here
MTM_CAPACITY = 100
LTM_ACCESS_THRESHOLD = 3    # retrieved this many times -> promote MTM -> LTM


class TieredMemoryStore:
    def __init__(self, persist_dir: str = "data/chroma", embedder=None,
                 stm_capacity: int = STM_CAPACITY,
                 mtm_capacity: int = MTM_CAPACITY,
                 ltm_access_threshold: int = LTM_ACCESS_THRESHOLD):
        # Lazy import of chromadb; if unavailable, use an in‑memory fallback
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=persist_dir)
        except ImportError:
            # No chromadb installed – use a simple in‑memory store
            self._client = None
            self._cols = {tier: _InMemoryCollection() for tier in TIERS}
        # Embedder handling
        if embedder is None:
            # Try sentence-transformers first
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import]
                embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                # Fallback: try fastembed via the adapter (proper import, not sys.modules hack)
                try:
                    from src.store.embed_adapter import FastEmbedAdapter  # type: ignore[import]
                    embedder = FastEmbedAdapter()
                except ImportError:
                    raise ImportError(
                        "Neither sentence-transformers nor fastembed is installed and no embedder was injected. "
                        "Install fastembed (pip install fastembed) or pass an embedder object with an "
                        "encode(text: str) -> list[float] method to TieredMemoryStore()."
                    )
        self._embedder = embedder
        self.stm_capacity = stm_capacity
        self.mtm_capacity = mtm_capacity
        self.ltm_access_threshold = ltm_access_threshold
        # Initialize collections based on backend availability
        if self._client is not None:
            self._cols = {
                tier: self._client.get_or_create_collection(
                    name=tier.lower(), metadata={"hnsw:space": "cosine"}
                )
                for tier in TIERS
            }
        else:
            # In‑memory fallback collections already set
            pass

    # ---------- helpers ----------
    def _embed(self, text: str) -> list[float]:
        result = self._embedder.encode(text)
        # SentenceTransformer returns a numpy array (.tolist() method exists).
        # An injected lightweight embedder may return a plain list directly.
        if hasattr(result, "tolist"):
            return result.tolist()
        return list(result)

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

    def list_all(self, tier: str = "all", q: str = "", limit: int = 1000) -> list[dict]:
        """Return stored memories WITHOUT a similarity query (UI listing).

        Unlike :meth:`query` this is read-only: it does not bump access
        counts, does not rebalance tiers and does not need an embedding —
        it walks each tier collection wholesale so the web UI can show
        every stored item (query() would only ever return the k nearest
        neighbours of some probe text).

        Args:
            tier: ``"all"`` (default) or a single tier name.
            q: Optional case-insensitive substring filter on the text.
            limit: Maximum number of items returned (newest first).
        """
        if tier != "all" and tier not in TIERS:
            raise ValueError(f"tier must be one of 'all' or {TIERS}")
        tiers = TIERS if tier == "all" else (tier,)
        items: list[dict] = []
        needle = (q or "").lower()
        for t in tiers:
            res = self._cols[t].get(include=["documents", "metadatas"])
            for item_id, text, meta in zip(res["ids"], res["documents"], res["metadatas"]):
                if needle and needle not in (text or "").lower():
                    continue
                items.append(self._to_item(item_id, text, meta, t))
        items.sort(key=lambda i: str(i.get("timestamp") or ""), reverse=True)
        return items[:limit]

    def count(self) -> dict:
        return {t: c.count() for t, c in self._cols.items()}