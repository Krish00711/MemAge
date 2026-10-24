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

`retrieve_ranked(query: str, token_budget: int) -> tuple[str, list[str]]`

- Same pipeline as `retrieve_context` (Stage-1 retrieval → Stage-2 rerank → token-budget selection → format), returning `(formatted_context, selected_ids)`.
- `selected_ids` are the `id` fields of the memories that **survived token-budget selection**, in final display order — 1:1 and in-order with the `[Memory N]` blocks of `formatted_context`. Selected items lacking a string `id` still appear in the context but are omitted from `selected_ids`.
- `retrieve_context` is a thin wrapper that returns only the first element, so its behavior is unchanged.
- Empty result is `("", [])`.

## src/eval/  (benchmark wiring — retrieval IDs)

`make_retrieval_fn(retrieval_context, token_budget) -> Callable[[str], RetrievalOutcome]`

- Wraps any object exposing `retrieve_ranked(...)` into a `BenchmarkRunner.retrieval_fn` that returns `RetrievalOutcome(context=..., retrieved_ids=..., tokens=...)`.
- `BenchmarkRunner` additionally normalizes a raw `retrieval_fn` result of the form `(context_str, retrieved_ids_list)` — i.e. `retrieval_fn=lambda q: ctx.retrieve_ranked(q, budget)` works directly.
- With `retrieved_ids` present, `BenchmarkResult` fills `retrieval_recall` / `retrieval_precision` / `retrieval_rr` and the `BenchmarkReport` retrieval aggregates.

---
Update the sections above as soon as an interface is decided — do not let this file go stale.
