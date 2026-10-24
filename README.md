# Selective Memory Agent

Long-term memory management for LLM-based agents — a Selective Memory Controller that jointly decides **ADD / UPDATE / DELETE / NOOP** for every candidate memory based on semantic importance, temporal validity, contradiction detection, and storage pressure.

Built for BCSE306L (Artificial Intelligence), VIT — DA1/DA2 mini-project.

## Architecture

```
Interaction Stream → Fact/Attribute Extraction → Candidate Memory Item
        → Selective Memory Controller (4 joint signals) → Decision
        → Tiered Memory Store (STM → MTM → LTM) → Two-Stage Retrieval
        → Agent Response → Evaluation Feedback (loops back to Controller)
```

Full architecture, literature review, and problem statement: see `docs/architecture.md` (exported from the DA1 report).

## Team & Module Ownership

| Module | Owner | Folder |
|---|---|---|
| Fact/Attribute Extraction, Tiered Memory Store (STM/MTM/LTM, ChromaDB) | Person A | `src/extraction/`, `src/store/` |
| Two-Stage Retrieval, Benchmarking & Baselines (MemoryOS, Memory-R1, H-MEM), Metrics | Person B | `src/retrieval/`, `src/eval/` |
| Selective Memory Controller (4 scoring signals + joint decision policy), LLM client | Krish | `src/controller/`, `src/llm/` |

Each module owner is responsible for their folder's code, tests, and docstrings. Cross-module interfaces (function signatures, data schemas) should be agreed in `docs/interfaces.md` **before** writing code, so all three modules can be built in parallel without blocking each other.

### Person B — retrieval & evaluation (what is implemented)

Both folders are implemented and tested end to end:

- **`src/retrieval/` — two-stage retrieval.** `RetrievalContext` runs coarse
  retrieval (`CoarseRetriever` over any store exposing
  `query(text, k, tier)`) → Stage-2 rerank (`rerank(query, candidates)`,
  e.g. `LLMReranker`; the synthetic demo uses a deterministic lexical
  scorer) → greedy token-budget selection → context formatting. The public
  contract is `retrieve_context(query, token_budget) -> str` and
  `retrieve_ranked(query, token_budget) -> (formatted_context, selected_ids)`.
- **`src/eval/` — benchmarking framework.** `BenchmarkRunner` scores any
  retrieval strategy on any `BenchmarkDataset` with answer metrics (F1,
  BLEU-1), retrieval metrics (Recall@K, Precision@K, MRR), and efficiency
  (mean latency, mean tokens). Includes:
  - dataset adapters for LoCoMo and LongMemEval (`src/eval/adapters.py`,
    file resolution `<data-dir>/<dataset>/*.json`);
  - five runnable strategies in `src/eval/baselines.py` (see below);
  - baseline comparison (`src/eval/comparison.py`, `--compare`);
  - ablation experiment infrastructure (`src/eval/ablation.py`, `--ablate`):
    retrieval stage, Stage-1 `k`, and token budget;
  - deterministic JSON/CSV result export (`src/eval/serialization.py`,
    `--export-json` / `--export-csv`).

All code in these folders has unit tests under `tests/` (run them with
`python -m pytest -q` from the repository root).

## Setup

```bash
git clone <your-repo-url>
cd selective-memory-agent
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then fill in your API key(s)
```

### LLM API (free tier)

This project calls an LLM for fact extraction, contradiction detection, and (optionally) response generation. Pick one and set the matching key in `.env`:

- **Gemini** (Google AI Studio, free tier): set `GEMINI_API_KEY`
- **Groq** (free tier, fast open-weight models): set `GROQ_API_KEY`

`src/llm/client.py` should expose one function, e.g. `call_llm(prompt: str, **kwargs) -> str`, so the rest of the codebase never cares which provider is behind it. Whoever builds `src/controller/` and whoever builds `src/extraction/` should both call through this one function.

### Vector store

ChromaDB runs embedded (no separate server needed) and persists to `data/chroma/` (gitignored — do not commit the database itself, only the code that builds it).

## Datasets

- [LoCoMo](https://github.com/snap-research/locomo) — primary benchmark
- LongMemEval — secondary/generalization benchmark

Raw data is gitignored (see `data/README.md` for exact file layout and
download commands — nothing is downloaded by project code). Place the files
at:

```
data/
├── locomo/
│   └── locomo10.json
└── longmemeval/
    └── longmemeval_s.json        # or longmemeval_m.json / longmemeval_oracle.json
```

The benchmark resolves them as `<data-dir>/<dataset>/*.json` (`--data-dir`
defaults to the repository `data/` folder).

**Status of real-dataset runs:** the adapters parse LoCoMo/LongMemEval into
benchmark cases, but *executing* a strategy on them is currently blocked —
building the retrieval/answer strategy requires `src/store` (tiered memory
store) and `src/llm` (`call_llm` for answer generation), which are owned by
teammates and not available yet; `--compare` / `--ablate` likewise run on
synthetic data only until that integration lands. The CLI reports this with
exit code 2 and a message naming the missing module. Everything else
(single-strategy synthetic runs, comparison, ablation, exports) works today.

## Running

The benchmark CLI is `python -m src.eval.run_benchmark`. Exit codes:
**0** success, **1** usage/format error, **2** unavailable (missing dataset
files or blocked store/LLM integration). `--help` lists all flags.

### Single strategy (synthetic dataset — always works, no network/API keys)

```bash
python -m src.eval.run_benchmark --dataset synthetic --baseline none
python -m src.eval.run_benchmark --dataset synthetic --baseline memoryos --k 3
python -m src.eval.run_benchmark --dataset synthetic --baseline memory_r1 --token-budget 64
```

`--dataset {synthetic,locomo,longmemeval}` (default `synthetic`);
`--baseline {none,full_history,memoryos,memory_r1,hmem}` (default `none`);
`--k` (metric cutoff, default 5); `--token-budget` (default 512).

The five strategies:

| Name | What it is |
|---|---|
| `none` | Our system — real two-stage retrieval pipeline |
| `full_history` | Full-context upper bound (no retrieval; retrieval metrics are `-`) |
| `memoryos` | MemoryOS-style tiered baseline (**project-level approximation, not a paper reproduction**) |
| `memory_r1` | Memory-R1-style budgeted baseline (**project-level approximation, not a paper reproduction**) |
| `hmem` | H-MEM-style hierarchical baseline (**project-level approximation, not a paper reproduction**) |

### Baseline comparison (synthetic only)

```bash
python -m src.eval.run_benchmark --dataset synthetic --compare none,full_history,memoryos,memory_r1,hmem
```

Prints one metrics table (F1, BLEU-1, Recall@K, P@K, MRR, tokens, latency)
with one row per strategy. `--compare` is mutually exclusive with `--ablate`
and requires `--dataset synthetic` (real datasets return exit 2 for now).

### Ablation experiments (synthetic only)

```bash
python -m src.eval.run_benchmark --dataset synthetic --ablate stage            # coarse-only vs two-stage
python -m src.eval.run_benchmark --dataset synthetic --ablate k:1,3,5          # Stage-1 candidate counts
python -m src.eval.run_benchmark --dataset synthetic --ablate budget:12,64,512 # token budgets
```

One family per run (the metric cutoff `--k` stays fixed across configs so
each ablation varies exactly one factor).

### JSON / CSV export

```bash
python -m src.eval.run_benchmark --dataset synthetic --compare none,full_history \
    --export-json results.json --export-csv results.csv
python -m src.eval.run_benchmark --dataset synthetic --ablate k:1,3,5 \
    --export-json ablation.json
```

`--export-json PATH` / `--export-csv PATH` serialize the structured
`ComparisonResult` / `AblationResult` (full JSON including per-case scores;
flat CSV with one row per strategy/configuration). They require `--compare`
or `--ablate`. Programmatic access: `to_json` / `write_json` / `to_csv` /
`write_csv` from `src.eval.serialization`. Parent directories must already
exist (write failures exit 2).

### Real datasets (LoCoMo / LongMemEval)

```bash
python -m src.eval.run_benchmark --dataset locomo --baseline none --data-dir data
```

With the dataset files in place (see **Datasets**), this loads and parses
LoCoMo, then stops with exit 2 until the store/LLM/controller integration
above is available; if the files are missing it exits 2 immediately with
the expected path.

## Branching & PRs

See `CONTRIBUTING.md`.
