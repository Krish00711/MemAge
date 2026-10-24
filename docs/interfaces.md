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

## Strategy contract  (src/eval/ → `BenchmarkRunner`)

```python
RetrievalOutcome = {
  "context": str,                    # formatted context (may be empty)
  "retrieved_ids": list[str] | None, # ordered IDs; None => retrieval metrics are None
  "tokens": int | None,              # computed from context when omitted
}
```

`retrieval_fn: (query: str) ->` one of `RetrievalOutcome` | `str` (context only) | `dict` | `(context, retrieved_ids)` — `BenchmarkRunner` normalizes all four. `answer_fn: (query: str, context: str) -> str`.

Baselines in `src/eval/baselines.py` are callables of exactly this shape:

- `FullHistoryBaseline(history)` / `full_history_retrieval_fn(history)` — whole history as context, `retrieved_ids=None`.
- `MemoryOSBaseline(entries, k=...)` / `memoryos_retrieval_fn(...)` — tiered STM/LTM approximation, top-`k` lexical selection, no budget.
- `MemoryR1Baseline(entries, token_budget=...)` / `memory_r1_retrieval_fn(...)` — global lexical rank + `select_by_token_budget`.
- `HMemBaseline(entries, token_budget=...)` / `hmem_retrieval_fn(...)` — three recency tiers (LTM/MTM/STM), coarse-to-fine ordering + budget.

The three checklist baselines (`memoryos`, `memory_r1`, `hmem`) are **project-level approximations, not paper reproductions** (see `src/eval/baselines.py` module docstring).

## src/eval/comparison.py  (multi-strategy runs)

`compare_strategies(dataset, strategies, answer_fn, k=5) -> ComparisonResult`

- `dataset`: `BenchmarkDataset` or sequence of `BenchmarkCase`.
- `strategies`: `list[StrategySpec]` with unique non-empty `name`s; each gets its own `BenchmarkRunner` over a fresh copy of the cases (results isolated, input order preserved).
- `StrategySpec(name, retrieval_fn, answer_fn=None)` — per-strategy `answer_fn` falls back to the shared one.
- `ComparisonResult(dataset_name, k, num_cases, results)` with `.names` / `.get(name)`; each `StrategyResult(name, report)` wraps the unmodified `BenchmarkReport`.
- `format_comparison_table(result) -> str` renders the aligned text table (`None` metrics as `-`).

## src/eval/ablation.py  (ablation experiments)

`run_ablation(dataset, configs, strategy_builder, answer_fn, k=5, ablation="") -> AblationResult`

- `AblationConfig(kind, label, params)`; constructors `stage_configs()` (fixed `coarse_only` vs `two_stage` pair), `k_configs(ks)`, `budget_configs(budgets)` (caller-supplied positive unique ints; labels `k=<v>` / `token_budget=<v>`).
- `strategy_builder: (AblationConfig) -> StrategySpec`; the config `label` overrides the spec name as the result key.
- `k` (the metric cutoff) stays fixed across configs — an ablation varies exactly one factor.
- `AblationResult(ablation, configs, comparison)` with `.labels` / `.get(label)`; the nested `ComparisonResult` holds one `BenchmarkReport` per config.
- `PassthroughReranker` satisfies the `RetrievalContext` rerank contract by returning candidates unchanged (Stage-2 bypass for `coarse_only`).

## src/eval/serialization.py  (result export)

```python
to_json(result, indent=2) -> str
write_json(result, path, indent=2) -> Path
to_csv(result) -> str
write_csv(result, path) -> Path
```

- `result` must be a `BenchmarkReport`, `ComparisonResult`, or `AblationResult` (`TypeError` otherwise); serialization is `dataclasses.asdict()` — no metric recomputation.
- JSON preserves the full structure; `None` aggregates are `null`. CSV is one row per strategy/configuration with header `name,kind,mean_f1,mean_bleu1,mean_recall_at_k,mean_precision_at_k,mrr,mean_tokens,mean_latency_ms,num_queries`; `None` cells are empty (never `0.0`). Output is deterministic (`\n` line endings).
- Parent directories are not created; `OSError`/`ValueError` propagate to the caller (the CLI maps write failures to exit code 2).

## src/eval/adapters.py  (real dataset loading)

`load_dataset(name, data_dir=None) -> BenchmarkDataset` — `name` is `locomo` or `longmemeval`; files resolve as `<data-dir>/<dataset>/*.json` (default: repo `data/`; see `data/README.md`). Also exposes `load_locomo`, `load_longmemeval`, `resolve_dataset_file`, `to_eval_dataset`.

- Raises `DatasetFileNotFoundError` (files absent) or `DatasetFormatError` (malformed); no network access, no downloads.

---
Update the sections above as soon as an interface is decided — do not let this file go stale.
