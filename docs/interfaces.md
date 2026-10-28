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

Two case helpers in `src/eval/baselines.py` convert a `BenchmarkCase` to baseline input: `memory_entries_from_case(case) -> List[dict]` gives one entry per turn with the dataset turn ID preserved (`locomo` `dia_id`; `longmemeval` flattened per session as `session_id`) — this is what the entry-based baselines receive; `history_from_case(case) -> str` is the formatted full-history string (`format_full_history` over the same entries) — used only by the `full_history` baseline. Both live-dataset formats are supported (synthetic cases have neither `turns` nor `sessions` and yield empty history / entries).

### The real `none` strategy (our system) on real datasets

Built by `_build_strategy(baseline, token_budget, data_dir, embedder=None)` in `src/eval/run_benchmark.py` → `_build_none_strategy(...)`, per case:

`case → memory_entries_from_case → _store_items_from_entries → fresh TieredMemoryStore → RetrievalContext(CoarseRetriever → LLMReranker(call_llm) → token budget → format_context) → make_retrieval_fn`

- `_store_items_from_entries(entries)` produces one store item per **dataset memory id** (LoCoMo `dia_id` per turn; LongMemEval `session_id` with that session's turns joined — the store's primary key and the `relevant_ids` unit are the same, so Recall@K/P@K/MRR stay id-compatible). Text/timestamp/source follow `_normalize_entry` precedence; entries are never mutated; id-less/empty-text entries are skipped.
- **`none` semantics: no selective-memory controller filtering.** Every case memory is stored as-is (no extraction LLM, no controller ADD/UPDATE/DELETE decisions); Stage-1 receives *all* of the case's memories (`k = item count`); answering is `call_llm` (provider-agnostic). It is never the synthetic demo's `_StaticStore`/`_LexicalReranker` — those exist only for `--dataset synthetic`.
- **Case isolation:** each `memory_prepare(case)` replaces the strategy state wholesale with a new store; with chromadb installed the store persists to a unique temp dir per case (previous case's dir removed), without chromadb the store's per-instance in-memory fallback applies. Prepare failures reset state, print to stderr, and never serve the previous case.
- `embedder=None` uses the store's default chain (`sentence-transformers` → `fastembed`); if neither exists (and nothing was injected) `_build_strategy` returns `None` → CLI exit 2 with the dependency named. Tests inject a deterministic embedder via the `embedder=` parameter.
- The returned `retrieval_fn` and `memory_prepare` carry a diagnostic `.state` dict: `store` / `ctx` / `retrieval_fn` / `persist_dir` / `thresholds` / `error`.
- `_build_none_strategy` doubles as the **real-dataset ablation builder** (one factor varied per config, all other behavior identical): `stage` (`"two_stage"` → `LLMReranker`, `"coarse_only"` → `PassthroughReranker`), `pipe_k` (Stage-1 candidate count; the metric cutoff stays on `BenchmarkRunner`), and `thresholds` (when not `None`, each memory is ingested through `src.controller.controller.process(candidate, store, llm=call_llm, thresholds=...)` instead of a blind `store.add` — **signal-ablation scope**: thresholds change what is stored and therefore retrieval/QA metrics; when `None` the controller is never invoked, preserving the `none` baseline). `build_real_ablation_spec(AblationConfig, ...)` maps `stage_configs` / `k_configs` / `budget_configs` / `signal_configs` params onto those arguments (`params["token_budget"]` overrides the builder default).

## src/eval/comparison.py  (multi-strategy runs)

`compare_strategies(dataset, strategies, answer_fn, k=5) -> ComparisonResult`

- `dataset`: `BenchmarkDataset` or sequence of `BenchmarkCase`.
- `strategies`: `list[StrategySpec]` with unique non-empty `name`s; each gets its own `BenchmarkRunner` over a fresh copy of the cases (results isolated, input order preserved).
- `StrategySpec(name, retrieval_fn, answer_fn=None, memory_prepare_fn=None)` — per-strategy `answer_fn` falls back to the shared one; `memory_prepare_fn` (if set) is called with the case before `retrieval_fn` and is forwarded to `BenchmarkRunner` (used by real-dataset strategies for per-case state, e.g. an isolated store). Both must be callable-or-`None` (`TypeError` otherwise).
- `ComparisonResult(dataset_name, k, num_cases, results)` with `.names` / `.get(name)`; each `StrategyResult(name, report)` wraps the unmodified `BenchmarkReport`.
- `format_comparison_table(result) -> str` renders the aligned text table (`None` metrics as `-`).

## src/eval/ablation.py  (ablation experiments)

`run_ablation(dataset, configs, strategy_builder, answer_fn, k=5, ablation="") -> AblationResult`

- `AblationConfig(kind, label, params)`; constructors `stage_configs()` (fixed `coarse_only` vs `two_stage` pair), `k_configs(ks)`, `budget_configs(budgets)` (caller-supplied positive unique ints; labels `k=<v>` / `token_budget=<v>`).
- `strategy_builder: (AblationConfig) -> StrategySpec`; the config `label` overrides the spec name as the result key.
- `k` (the metric cutoff) stays fixed across configs — an ablation varies exactly one factor.
- `AblationResult(ablation, configs, comparison)` with `.labels` / `.get(label)`; the nested `ComparisonResult` holds one `BenchmarkReport` per config.
- `PassthroughReranker` satisfies the `RetrievalContext` rerank contract by returning candidates unchanged (Stage-2 bypass for `coarse_only`).
- The builder's returned `memory_prepare_fn` is preserved when `run_ablation` re-labels specs with config labels.

### CLI dispatch (`src/eval/run_benchmark.py`)

`--compare` / `--ablate` dispatch on `--dataset`: `synthetic` keeps the existing synthetic paths (`run_synthetic_comparison` / `run_synthetic_ablation`); `locomo` / `longmemeval` load via `_load_real_dataset(dataset, data_dir, sample)` and run `run_dataset_comparison` (→ `build_real_comparison_specs(names, token_budget, data_dir, embedder)`: one `StrategySpec` per name from `_build_strategy`, all with `answer_fn=None` so one shared `_build_answer_fn(call_llm)` is used — `compare_strategies` → `Comparison ({dataset}, k={k})` header + `format_comparison_table`) or `run_dataset_ablation` (→ `build_real_ablation_spec` per config; a probe of `configs[0]` reports a dependency error before the run). Missing dataset files/dependencies → exit 2 (`_get_unavailable_message`); usage errors (unknown/empty/duplicate names, `--compare`+`--ablate`) → exit 1. Real-dataset signal ablations apply thresholds at ingestion via the controller (scope above); synthetic signal ablations gate the demo memories through the real `src.controller.controller.decide` policy.

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