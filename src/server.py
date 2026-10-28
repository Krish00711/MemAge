"""MemAge web server — zero-dependency HTTP API + static file server.

Uses only Python stdlib (http.server, json, threading).
Requires the store/LLM to be configured via .env before starting.

Usage:
    python -m src.server [--port 8000] [--persist-dir data/chroma] [--demo]

    --demo   Use a small in-memory fake store with sample data so the UI
             can be explored without an API key or ChromaDB.

Endpoints:
    GET  /api/health               — provider, store counts, ok flag
    POST /api/chat                 — {message, session_id} -> AgentResponse
    GET  /api/memories             — list all memories (all tiers)
    GET  /api/memories?tier=STM    — filter by tier
    GET  /api/memories?q=<text>    — search by text substring
    DELETE /api/memories/<id>      — delete one memory
    DELETE /api/memories           — reset all (requires confirm=true query param)
    GET  /api/timeline             — ingestion event log (in-memory, resets on restart)
    GET  /                         — serve static/index.html
    GET  /static/<file>            — serve static assets
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Timeline log — in-memory, reset on server restart
# ---------------------------------------------------------------------------
_TIMELINE: list[dict] = []
_TIMELINE_LOCK = threading.Lock()


def _log_event(action: str, candidate: dict, decision: dict) -> None:
    with _TIMELINE_LOCK:
        _TIMELINE.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "candidate_id": candidate.get("id"),
            "candidate_text": candidate.get("text", "")[:120],
            "scores": decision.get("scores", {}),
            "target_id": decision.get("target_id"),
        })


# ---------------------------------------------------------------------------
# Store + agent singletons (set in main())
# ---------------------------------------------------------------------------
_store: Any = None
_agent: Any = None


# ---------------------------------------------------------------------------
# Demo store (no ChromaDB, no embeddings — for offline exploration)
# ---------------------------------------------------------------------------

class _DemoStore:
    """Fully in-memory store with deterministic keyword search."""

    _SAMPLE = [
        {"id": "demo__0", "text": "Maya prefers tea over coffee.",
         "timestamp": "2026-01-01T09:00:00", "source": "demo", "embedding": None,
         "tier": "STM", "distance": 0.0, "access_count": 2},
        {"id": "demo__1", "text": "Maya lives in Chennai.",
         "timestamp": "2026-01-01T09:01:00", "source": "demo", "embedding": None,
         "tier": "STM", "distance": 0.0, "access_count": 1},
        {"id": "demo__2", "text": "Maya is a software engineer.",
         "timestamp": "2025-06-15T10:00:00", "source": "demo", "embedding": None,
         "tier": "MTM", "distance": 0.0, "access_count": 0},
        {"id": "demo__3", "text": "Maya adopted a dog named Biscuit in March 2023.",
         "timestamp": "2024-03-01T00:00:00", "source": "demo", "embedding": None,
         "tier": "LTM", "distance": 0.0, "access_count": 5},
    ]

    def __init__(self) -> None:
        self._items: dict[str, dict] = {i["id"]: dict(i) for i in self._SAMPLE}
        self._next = len(self._items)
        self.stm_capacity = 20
        self.mtm_capacity = 100

    def query(self, text: str, k: int = 10, tier: str = "all") -> list[dict]:
        words = set(text.lower().split())
        results = []
        for item in self._items.values():
            if tier != "all" and item.get("tier") != tier:
                continue
            overlap = len(words & set(item["text"].lower().split()))
            results.append((overlap, item))
        results.sort(key=lambda x: -x[0])
        return [dict(i) for _, i in results[:k]]

    def get(self, item_id: str) -> dict | None:
        item = self._items.get(item_id)
        return dict(item) if item else None

    def count(self) -> dict:
        counts = {"STM": 0, "MTM": 0, "LTM": 0}
        for item in self._items.values():
            t = item.get("tier", "STM")
            counts[t] = counts.get(t, 0) + 1
        return counts

    def add(self, item: dict, tier: str = "STM") -> dict:
        stored = dict(item)
        stored["tier"] = tier
        stored["distance"] = 0.0
        stored["access_count"] = 0
        self._items[item["id"]] = stored
        return stored

    def update(self, item_id: str, new_text: str, timestamp: str | None = None) -> bool:
        if item_id not in self._items:
            return False
        self._items[item_id]["text"] = new_text
        if timestamp:
            self._items[item_id]["timestamp"] = timestamp
        return True

    def delete(self, item_id: str) -> bool:
        if item_id not in self._items:
            return False
        del self._items[item_id]
        return True

    def list_all(self, tier: str = "all", q: str = "") -> list[dict]:
        results = []
        for item in self._items.values():
            if tier != "all" and item.get("tier") != tier:
                continue
            if q and q.lower() not in item["text"].lower():
                continue
            results.append(dict(item))
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results

    def reset(self) -> None:
        self._items.clear()


# ---------------------------------------------------------------------------
# Demo LLM (no API key needed)
# ---------------------------------------------------------------------------

def _demo_llm(prompt: str, system: str | None = None, **_kw) -> str:
    """Deterministic stub used in demo mode.

    Extracts plausible facts from statements, scores importance as 0.8,
    detects low contradiction, and answers from context.  No API calls.
    """
    sys_lower = (system or "").lower()

    # ── Extraction ──────────────────────────────────────────────────────
    # The extractor passes the dialogue turn in the last "Message: ..." line.
    if "memories from conversations" in sys_lower:
        import re as _re
        # Pull the Message: line out of the prompt
        m = _re.search(r"Message:\s*(.+)", prompt)
        if not m:
            return "[]"
        text = m.group(1).strip()
        # Heuristic: questions and greetings produce no facts
        if text.endswith("?") or text.lower().startswith(("hi", "hello", "hey", "thanks")):
            return "[]"
        # Speaker name for personalisation
        sp_m = _re.search(r"Speaker:\s*(\w+)", prompt)
        speaker = sp_m.group(1) if sp_m else "User"
        # Map common patterns to self-contained facts
        fact = text
        for pat, repl in [
            (r"(?i)^I('m| am) (.+)", f"{speaker} is \\2"),
            (r"(?i)^I live in (.+)", f"{speaker} lives in \\1"),
            (r"(?i)^I moved to (.+)", f"{speaker} moved to \\1"),
            (r"(?i)^I (like|love|prefer|enjoy) (.+)", f"{speaker} likes \\2"),
            (r"(?i)^I work as (.+)", f"{speaker} works as \\1"),
            (r"(?i)^My name is (.+)", f"{speaker}'s name is \\1"),
            (r"(?i)^I ('ve|have) (.+)", f"{speaker} has \\2"),
        ]:
            import re as _re2
            sub = _re2.sub(pat, repl, text)
            if sub != text:
                fact = sub
                break
        # Strip trailing punctuation for cleanliness
        fact = fact.rstrip(".,!").strip()
        if fact:
            import json as _j
            return _j.dumps([fact])
        return "[]"

    # ── Importance scoring ───────────────────────────────────────────────
    if "importance scorer" in sys_lower:
        return '{"importance": 0.8}'

    # ── Contradiction detection ──────────────────────────────────────────
    if "contradiction detector" in sys_lower:
        # Check for key conflict words
        conflicting = any(w in prompt.lower() for w in
                         ("no longer", "moved", "changed", "now ", "actually", "wrong", "incorrect"))
        score = 0.75 if conflicting else 0.05
        return f'{{"contradiction": {score}}}'

    # ── LLM Reranker ────────────────────────────────────────────────────
    if "relevance reranker" in sys_lower:
        import re as _re3, json as _j3
        ids = _re3.findall(r"- ID: (\S+)", prompt)
        if ids:
            rankings = [{"id": i, "score": round(0.9 - 0.05 * n, 3)} for n, i in enumerate(ids)]
            return _j3.dumps({"rankings": rankings})
        return '{"rankings": []}'

    # ── Answer generation ────────────────────────────────────────────────
    if "helpful assistant" in sys_lower:
        if "no relevant memories" in prompt:
            return "I don't have any stored information about that yet."
        lines = [l for l in prompt.split("\n") if l.startswith("Content:")]
        if lines:
            facts = "; ".join(l.replace("Content:", "").strip() for l in lines[:2])
            return f"Based on what I remember: {facts}."
        return "I'm not sure — no relevant memories found."

    return "[]"


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _json(obj: Any) -> bytes:
    return json.dumps(obj, default=str).encode()


def _err(msg: str, code: int = 400) -> tuple[bytes, int]:
    return _json({"error": msg}), code


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence default access log
        pass

    def _send(self, body: bytes, code: int = 200,
              content_type: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj: Any, code: int = 200) -> None:
        self._send(_json(obj), code)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except Exception:
            return {}

    def _parse_qs(self) -> dict:
        parsed = urllib.parse.urlparse(self.path)
        return dict(urllib.parse.parse_qsl(parsed.query))

    def _path(self) -> str:
        return urllib.parse.urlparse(self.path).path

    # --- OPTIONS (CORS preflight) ---
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # --- GET ---
    def do_GET(self):
        path = self._path()

        # Static files
        if path == "/" or path == "/index.html":
            return self._serve_static("index.html")
        if path.startswith("/static/"):
            return self._serve_static(path[len("/static/"):])

        if path == "/api/health":
            return self._handle_health()
        if path == "/api/memories":
            return self._handle_list_memories()
        if path == "/api/timeline":
            return self._handle_timeline()

        mem_prefix = "/api/memories/"
        if path.startswith(mem_prefix):
            return self._handle_get_memory(path[len(mem_prefix):])

        self._send_json({"error": "not found"}, 404)

    # --- POST ---
    def do_POST(self):
        path = self._path()
        if path == "/api/chat":
            return self._handle_chat()
        self._send_json({"error": "not found"}, 404)

    # --- DELETE ---
    def do_DELETE(self):
        path = self._path()
        qs = self._parse_qs()

        if path == "/api/memories":
            if qs.get("confirm") != "true":
                return self._send_json(
                    {"error": "add ?confirm=true to reset all memories"}, 400
                )
            return self._handle_reset()

        mem_prefix = "/api/memories/"
        if path.startswith(mem_prefix):
            return self._handle_delete_memory(path[len(mem_prefix):])

        self._send_json({"error": "not found"}, 404)

    # --- endpoint implementations ---

    def _serve_static(self, filename: str) -> None:
        static_dir = Path(__file__).parent.parent / "static"
        fp = static_dir / filename
        if not fp.is_file():
            return self._send_json({"error": f"{filename} not found"}, 404)
        ct = "text/html" if filename.endswith(".html") else \
             "text/css" if filename.endswith(".css") else \
             "application/javascript" if filename.endswith(".js") else \
             "application/octet-stream"
        self._send(fp.read_bytes(), content_type=ct)

    def _handle_health(self) -> None:
        counts = _store.count()
        is_demo = isinstance(_store, _DemoStore)
        provider = "demo" if is_demo else os.environ.get("LLM_PROVIDER", "(not set)")
        self._send_json({
            "ok": True,
            "provider": provider,
            "is_demo": is_demo,
            "counts": counts,
            "stm_capacity": getattr(_store, "stm_capacity", 20),
            "mtm_capacity": getattr(_store, "mtm_capacity", 100),
        })

    def _handle_list_memories(self) -> None:
        qs = self._parse_qs()
        tier = qs.get("tier", "all")
        q = qs.get("q", "")
        if hasattr(_store, "list_all"):
            items = _store.list_all(tier=tier, q=q)
        else:
            # Real store: use query with broad search
            try:
                items = _store.query(q or "the", k=200, tier=tier)
                if q:
                    items = [i for i in items if q.lower() in i.get("text", "").lower()]
            except Exception as exc:
                return self._send_json({"error": str(exc)}, 500)
        self._send_json({"items": items, "total": len(items)})

    def _handle_get_memory(self, item_id: str) -> None:
        item = _store.get(item_id)
        if item is None:
            return self._send_json({"error": "not found"}, 404)
        self._send_json(item)

    def _handle_delete_memory(self, item_id: str) -> None:
        ok = _store.delete(item_id)
        if not ok:
            return self._send_json({"error": "not found"}, 404)
        _log_event("DELETE", {"id": item_id, "text": ""}, {"scores": {}, "target_id": item_id})
        self._send_json({"deleted": item_id})

    def _handle_reset(self) -> None:
        if hasattr(_store, "reset"):
            _store.reset()
        else:
            # Real store: delete all from all tiers
            for tier in ("STM", "MTM", "LTM"):
                try:
                    items = _store.query("the", k=500, tier=tier)
                    for item in items:
                        _store.delete(item["id"])
                except Exception:
                    pass
        with _TIMELINE_LOCK:
            _TIMELINE.clear()
        self._send_json({"reset": True})

    def _handle_timeline(self) -> None:
        with _TIMELINE_LOCK:
            self._send_json({"events": list(_TIMELINE)})

    def _handle_chat(self) -> None:
        body = self._read_body()
        message = body.get("message", "").strip()
        session_id = body.get("session_id", "web")
        if not message:
            return self._send_json({"error": "message is required"}, 400)

        resp = _agent.chat(message, session_id=session_id)

        # Log ingestion events to timeline
        if resp.turn_result:
            for cr in resp.turn_result.candidates:
                if cr.decision["action"] != "NOOP":
                    _log_event(cr.decision["action"], cr.candidate, cr.decision)

        self._send_json({
            "answer": resp.answer,
            "context": resp.context,
            "retrieved_ids": resp.retrieved_ids,
            "turn_result": _serialise_turn_result(resp.turn_result),
            "ok": resp.ok,
            "error": resp.error,
        })


def _serialise_turn_result(tr) -> dict | None:
    if tr is None:
        return None
    return {
        "candidates": [
            {
                "id": cr.candidate.get("id"),
                "text": cr.candidate.get("text", "")[:120],
                "decision": cr.decision,
                "ok": cr.ok,
                "error": cr.error,
            }
            for cr in tr.candidates
        ],
        "added": tr.added,
        "updated": tr.updated,
        "deleted": tr.deleted,
        "noops": tr.noops,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="MemAge web server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--persist-dir", default="data/chroma")
    parser.add_argument("--demo", action="store_true",
                        help="Use in-memory demo store (no ChromaDB/API key needed)")
    parser.add_argument("--token-budget", type=int, default=500)
    args = parser.parse_args(argv)

    global _store, _agent

    if args.demo:
        _store = _DemoStore()
        llm = _demo_llm
        print("Running in DEMO mode (fake store + stub LLM)")
    else:
        # Load .env before touching the store or LLM so API keys and
        # CHROMA_PERSIST_DIR are available at startup, not deferred to
        # the first request.
        try:
            from dotenv import load_dotenv
            load_dotenv(override=False)
        except ImportError:
            pass  # rely on shell environment

        # Validate LLM provider key early so the error is visible at startup.
        provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
        if not provider:
            print("[WARN] LLM_PROVIDER not set - defaulting to gemini. "
                  "Copy .env.example to .env and set your key.", file=sys.stderr)
        elif provider == "gemini" and not os.environ.get("GEMINI_API_KEY", "").strip():
            print("[WARN] GEMINI_API_KEY not set - chat will fail. "
                  "Set it in .env or the environment.", file=sys.stderr)
        elif provider == "groq" and not os.environ.get("GROQ_API_KEY", "").strip():
            print("[WARN] GROQ_API_KEY not set - chat will fail. "
                  "Set it in .env or the environment.", file=sys.stderr)
        elif provider == "mistral" and not os.environ.get("MISTRAL_API_KEY", "").strip():
            print("[WARN] MISTRAL_API_KEY not set - chat will fail. "
                  "Set it in .env or the environment.", file=sys.stderr)

        # Prefer CHROMA_PERSIST_DIR env var over CLI default
        persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "").strip() or args.persist_dir
        try:
            from src.store.memory_store import TieredMemoryStore
            _store = TieredMemoryStore(persist_dir=persist_dir)
            llm = None  # resolved lazily via call_llm() which re-reads env each call
            print(f"Store: {persist_dir} | counts: {_store.count()}")
        except ImportError as exc:
            msg = str(exc)
            if "chromadb" in msg:
                print(f"[ERROR] chromadb not installed: {exc}", file=sys.stderr)
                print("  Install: pip install chromadb==0.5.5", file=sys.stderr)
            elif "fastembed" in msg or "sentence-transformers" in msg or "embedder" in msg:
                print(f"[ERROR] Embedding backend not installed: {exc}", file=sys.stderr)
                print("  Install (lightweight): pip install fastembed", file=sys.stderr)
                print("  Install (full):        pip install sentence-transformers==3.0.1", file=sys.stderr)
            else:
                print(f"[ERROR] Missing dependency: {exc}", file=sys.stderr)
            print("Run with --demo for offline exploration, or install missing packages.", file=sys.stderr)
            sys.exit(1)

    from src.agent import MemAgeAgent
    _agent = MemAgeAgent(_store, llm=llm, token_budget=args.token_budget)

    server = HTTPServer(("0.0.0.0", args.port), _Handler)
    print(f"MemAge server -> http://localhost:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
