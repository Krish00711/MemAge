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

Download scripts belong in `src/extraction/` or a `scripts/` folder; raw data itself is gitignored (see `data/README.md`).

## Running

```bash
python -m src.eval.run_benchmark --dataset locomo --baseline none   # our system
python -m src.eval.run_benchmark --dataset locomo --baseline memoryos
```
(Exact CLI to be finalized once `src/eval/` is built — update this section once it's real.)

## Branching & PRs

See `CONTRIBUTING.md`.
