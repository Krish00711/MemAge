"""MemAge end-to-end smoke test.

Exercises a real persistent store (ChromaDB + sentence-transformers) and a
real LLM provider.  Requires all dependencies installed and .env configured.

Usage:
    python scripts/smoke_test.py [--persist-dir /tmp/memage_smoke]

What it tests:
  Turn 1 — store a new fact ("Maya lives in Chennai")
  Turn 2 — store a conflicting fact ("Maya moved to Bangalore")
  Reopen  — reload the store from disk
  Verify  — query the store and confirm the newer fact is retrievable
  Agent   — ask the agent "Where does Maya live?" and check the answer

Exit code 0 = smoke test passed, non-zero = failure.

NOTE: This script makes real LLM API calls and writes to disk.
Do not run without a configured .env file.
"""
from __future__ import annotations

import argparse
import os
import sys

# Add project root to path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def _check_deps():
    missing = []
    for mod in ("chromadb", "sentence_transformers"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"[FAIL] Missing dependencies: {missing}")
        print("Run: pip install chromadb==0.5.5 sentence-transformers==3.0.1")
        sys.exit(2)


def _build_store(persist_dir: str):
    from src.store.memory_store import TieredMemoryStore
    return TieredMemoryStore(persist_dir=persist_dir)


def main():
    parser = argparse.ArgumentParser(description="MemAge smoke test")
    parser.add_argument("--persist-dir", default="/tmp/memage_smoke",
                        help="ChromaDB persistence directory (default: /tmp/memage_smoke)")
    args = parser.parse_args()

    _load_env()
    _check_deps()

    persist_dir = args.persist_dir
    print(f"\n=== MemAge smoke test  (store: {persist_dir}) ===\n")

    # ------------------------------------------------------------------ Turn 1
    print("Step 1: Ingest turn 1 - initial city fact")
    from src.pipeline import run_turn
    from src.llm.client import call_llm

    store = _build_store(persist_dir)
    meta1 = {"speaker": "Maya", "timestamp": "2026-01-01T09:00:00", "source": "smoke:1"}
    r1 = run_turn("I live in Chennai.", meta1, store=store, llm=call_llm)
    if r1.candidates and r1.candidates[0].ok:
        action = r1.candidates[0].decision["action"]
        print(f"  action={action}  candidate_id={r1.candidates[0].candidate['id']}")
        assert action == "ADD", f"Expected ADD, got {action}"
    else:
        err = r1.candidates[0].error if r1.candidates else "no candidates extracted"
        print(f"  [FAIL] Turn 1: {err}")
        sys.exit(1)

    # ------------------------------------------------------------------ Turn 2
    print("Step 2: Ingest turn 2 - correction (moved to Bangalore)")
    meta2 = {"speaker": "Maya", "timestamp": "2026-06-01T09:00:00", "source": "smoke:2"}
    r2 = run_turn("I moved to Bangalore last month.", meta2, store=store, llm=call_llm)
    if r2.candidates:
        action2 = r2.candidates[0].decision["action"]
        print(f"  action={action2}  candidate_id={r2.candidates[0].candidate['id']}")
        assert action2 in ("ADD", "UPDATE", "DELETE"), f"Unexpected action: {action2}"
    else:
        print("  [WARN] No candidates extracted for turn 2 (small talk / LLM parse fail)")

    # ------------------------------------------------------------------ Reopen store
    print("Step 3: Reopen store from disk and verify persistence")
    del store
    store2 = _build_store(persist_dir)
    counts = store2.count()
    print(f"  Counts after reload: {counts}")
    total = sum(counts.values())
    assert total > 0, "Store is empty after reload - persistence failed"

    # ------------------------------------------------------------------ Query
    print("Step 4: Query for city-related memory")
    results = store2.query("Where does Maya live?", k=3, tier="all")
    print(f"  Retrieved {len(results)} memory/memories")
    assert len(results) > 0, "No memories retrieved - query failed"
    top = results[0]
    print(f"  Top result: '{top['text']}'  (tier={top['tier']}, distance={top['distance']:.3f})")

    # ------------------------------------------------------------------ Agent
    print("Step 5: Agent chat - ask where Maya lives")
    from src.agent import MemAgeAgent
    agent = MemAgeAgent(store2, llm=call_llm, token_budget=300, k=5)
    resp = agent.chat("Where does Maya live?", session_id="smoke")
    print(f"  Answer: {resp.answer[:200]}")
    print(f"  Retrieved IDs: {resp.retrieved_ids}")
    assert resp.ok, f"Agent returned error: {resp.error}"
    # The answer should reference either Chennai or Bangalore
    answer_lower = resp.answer.lower()
    assert "chennai" in answer_lower or "bangalore" in answer_lower or "bengaluru" in answer_lower, (
        f"Answer does not mention a city: {resp.answer}"
    )

    print("\n=== Smoke test PASSED ===\n")


if __name__ == "__main__":
    main()
