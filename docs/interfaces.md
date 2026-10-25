# Interfaces (fill this in together, before writing implementation code)

Whoever changes an interface here should ping the other two — this file is the contract that lets three people build in parallel.

## CandidateMemoryItem

```python
{
  "id": str,            # "<source>__<n>", e.g. "D1:3__0" (n = index of the fact within that turn)
  "text": str,          # the extracted fact/statement
  "timestamp": str,     # ISO 8601 (falls back to the raw date text if it cannot be parsed)
  "source": str,        # the dialogue turn id; for LoCoMo this is the dia_id, e.g. "D1:3"
  "embedding": list[float] | None,  # filled in by src/store/, not src/extraction/
}
```

Items returned by the store (`get`, `query`) carry three extra fields: `tier` ("STM" | "MTM" | "LTM"), `distance` (cosine distance, smaller = more similar; `None` for `get`) and `access_count` (how many times it has been retrieved). `embedding` is `None` in query results.

## src/extraction/  →  src/controller/

`extract_candidates(turn: str, metadata: dict, llm=None) -> list[CandidateMemoryItem]`

- `metadata` keys (all optional): `speaker`, `timestamp` (ISO 8601), `source`.
- `llm` defaults to `src.llm.client.call_llm`; tests pass a fake function.
- Returns `[]` for small talk or if the LLM reply cannot be parsed after 2 attempts.
- Because `source` is the LoCoMo `dia_id`, a retrieved item is relevant to a QA question when `item["source"]` is in that question's `evidence` list.

Dataset helpers in `src/extraction/locomo_loader.py`: `load_locomo()`, `extract_conversation(record, llm, cache_dir, limit)`, `ingest_conversation(record, store, ...)`. Extraction results are cached in `data/cache/` (gitignored).

## src/controller/  →  src/store/

`decide(candidate: CandidateMemoryItem, existing_matches: list[CandidateMemoryItem]) -> Decision`

```python
Decision = {
  "action": "ADD" | "UPDATE" | "DELETE" | "NOOP",
  "target_id": str | None,   # which existing memory this applies to, if UPDATE/DELETE
  "scores": {"importance": float, "temporal_validity": float, "contradiction": float, "storage_pressure": float},
}
```

Store methods the controller can call to carry out a decision: `add(item, tier="STM")`, `update(id, new_text, timestamp=None)`, `delete(id)`. `existing_matches` comes from `query(candidate["text"], k, "all")`. `count()` returns `{"STM": n, "MTM": n, "LTM": n}` for the storage-pressure signal. (Open question: do we want a single `apply_decision(candidate, decision)` method on the store?)

## src/store/  →  src/retrieval/

`query(text: str, k: int, tier: "STM" | "MTM" | "LTM" | "all") -> list[CandidateMemoryItem]`

Results are sorted by distance (most similar first). Each call counts as one access for every returned memory.

Other `TieredMemoryStore` methods: `get(id)`, `count()`, `move(id, tier)`, `rebalance()`.

Tier rules (constants at the top of `src/store/memory_store.py`): STM holds the newest 20 memories; overflow moves the oldest to MTM (cap 100), then to LTM. An MTM memory retrieved 3 times is promoted to LTM.

## src/llm/client.py  (used by extraction, controller, and response generation)

`call_llm(prompt: str, system: str = None, **kwargs) -> str`

## src/retrieval/  →  agent response step

`retrieve_context(query: str, token_budget: int) -> str`  (already formatted, ready to drop into the final prompt)

---
Update the sections above as soon as an interface is decided — do not let this file go stale.