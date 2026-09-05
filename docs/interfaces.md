# Interfaces (fill this in together, before writing implementation code)

Whoever changes an interface here should ping the other two — this file is the contract that lets three people build in parallel.

## CandidateMemoryItem

```python
{
  "id": str,
  "text": str,          # the extracted fact/statement
  "timestamp": str,     # ISO 8601
  "source": str,        # e.g. "session_12_turn_4"
  "embedding": list[float] | None,  # filled in by src/store/, not src/extraction/
}
```

## src/extraction/  →  src/controller/

`extract_candidates(turn: str, metadata: dict) -> list[CandidateMemoryItem]`

## src/controller/  →  src/store/

`decide(candidate: CandidateMemoryItem, existing_matches: list[CandidateMemoryItem]) -> Decision`

```python
Decision = {
  "action": "ADD" | "UPDATE" | "DELETE" | "NOOP",
  "target_id": str | None,   # which existing memory this applies to, if UPDATE/DELETE
  "scores": {"importance": float, "temporal_validity": float, "contradiction": float, "storage_pressure": float},
}
```

## src/store/  →  src/retrieval/

`query(text: str, k: int, tier: "STM" | "MTM" | "LTM" | "all") -> list[CandidateMemoryItem]`

## src/llm/client.py  (used by extraction, controller, and response generation)

`call_llm(prompt: str, system: str = None, **kwargs) -> str`

## src/retrieval/  →  agent response step

`retrieve_context(query: str, token_budget: int) -> str`  (already formatted, ready to drop into the final prompt)

---
Update the sections above as soon as an interface is decided — do not let this file go stale.
