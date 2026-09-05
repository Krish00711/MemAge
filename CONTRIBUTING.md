# Contributing

Three people, three mostly-independent modules — the goal is to work in parallel without stepping on each other.

## Branch strategy

- `main` — always working, always demoable. Never commit directly to it.
- One long-lived branch per person per module:
  - `feature/extraction-store-<name>` (Person A)
  - `feature/retrieval-eval-<name>` (Person B)
  - `feature/controller-krish` (Krish)
- Small fixes/experiments: branch off your own feature branch, e.g. `feature/controller-krish-fix-scoring`.

## Workflow

1. Agree on interfaces first (function signatures, the shape of a "candidate memory item" dict/object, what the Controller returns) — write these in `docs/interfaces.md` before coding, so nobody blocks anybody else.
2. Work on your own branch. Commit early and often — small commits with clear messages beat one giant commit.
3. Commit message format: `<module>: <what changed>` — e.g. `controller: add temporal-validity scoring function`.
4. When a module (or a meaningful chunk of it) is ready, open a Pull Request into `main`.
5. **At least one other teammate reviews before merging** — even a quick skim. This is what catches interface mismatches early.
6. Pull `main` into your branch regularly (`git pull origin main` while on your branch, or rebase) so merges stay small.

## Interface contract (fill in as you build)

This is the most important file in the repo for a 3-person split — it's the thing that lets all three of you build simultaneously without integration hell at the end.

```
docs/interfaces.md should define, at minimum:

- CandidateMemoryItem: the exact fields (text, timestamp, source, embedding?, id)
- What src/extraction/ hands to src/controller/
- What src/controller/ hands to src/store/ (the decision + item)
- What src/store/ exposes to src/retrieval/ (query interface)
- What src/retrieval/ hands back to the agent-response step
- The call_llm() signature everyone uses (src/llm/client.py)
```

Update this file the moment an interface changes — a stale interface doc is worse than no doc.

## Issues

Use GitHub Issues for anything bigger than a single commit: one issue per scoring signal, one per baseline to reproduce, one per metric to implement. Label with `extraction`, `controller`, `retrieval`, `eval`, or `docs` so it's clear whose queue it's in.
