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
import re
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

# Session-scoped live statistics (reset when the process restarts).  Every
# number below is incremented from a REAL request — nothing is estimated.
_STATS: dict = {
    "chats": 0,
    "retrieved_memories": 0,
    "latency_ms_total": 0.0,
    "tokens_total": 0,
    "llm_calls": 0,
    "decisions": {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NOOP": 0},
    "started": datetime.now(timezone.utc).isoformat(),
}


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
        if path == "/api/benchmarks":
            return self._handle_benchmarks()

        mem_prefix = "/api/memories/"
        if path.startswith(mem_prefix):
            return self._handle_get_memory(path[len(mem_prefix):])

        self._send_json({"error": "not found"}, 404)

    # --- POST ---
    def do_POST(self):
        path = self._path()
        if path == "/api/chat":
            return self._handle_chat()
        if path == "/api/benchmarks/run":
            return self._handle_benchmark_run()
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
        if is_demo:
            model = "deterministic-stub (offline demo)"
        else:
            model = os.environ.get(f"{provider.strip().upper()}_MODEL", "").strip() or "(default)"
        with _TIMELINE_LOCK:
            timeline_count = len(_TIMELINE)
        self._send_json({
            "ok": True,
            "provider": provider,
            "model": model,
            "is_demo": is_demo,
            "counts": counts,
            "total": sum(counts.values()) if isinstance(counts, dict) else 0,
            "stm_capacity": getattr(_store, "stm_capacity", 20),
            "mtm_capacity": getattr(_store, "mtm_capacity", 100),
            "stats": {
                **_STATS,
                "timeline_events": timeline_count,
                "avg_latency_ms": (
                    round(_STATS["latency_ms_total"] / _STATS["chats"], 1)
                    if _STATS["chats"] else 0
                ),
            },
            "agent": {
                "token_budget": getattr(_agent, "_token_budget", None),
                "k": getattr(_agent, "_k", None),
            },
            "datasets": _datasets_available(),
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
            # Real store: delete every stored item across all tiers.
            if hasattr(_store, "list_all"):
                for item in _store.list_all():
                    try:
                        _store.delete(item["id"])
                    except Exception:
                        pass
            else:
                for tier in ("STM", "MTM", "LTM"):
                    try:
                        for item in _store.query("the", k=500, tier=tier):
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

        # Exact timing + real provider token usage for this request.
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
        try:
            from src.llm import client as _llm_client
            _llm_client.usage_reset()
        except Exception:
            _llm_client = None
        t0 = time.perf_counter()
        resp = _agent.chat(message, session_id=session_id)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        if _llm_client is not None:
            usage = _llm_client.usage_snapshot()

        # Log ingestion events to timeline + count every controller decision
        if resp.turn_result:
            for cr in resp.turn_result.candidates:
                action = cr.decision.get("action")
                if action in _STATS["decisions"]:
                    _STATS["decisions"][action] += 1
                if action != "NOOP":
                    _log_event(action, cr.candidate, cr.decision)

        _STATS["chats"] += 1
        _STATS["retrieved_memories"] += len(resp.retrieved_ids or [])
        _STATS["latency_ms_total"] += latency_ms
        _STATS["tokens_total"] += usage.get("total_tokens", 0)
        _STATS["llm_calls"] += usage.get("calls", 0)

        self._send_json({
            "answer": resp.answer,
            "context": resp.context,
            "retrieved_ids": resp.retrieved_ids,
            "retrieval_trace": resp.retrieval_trace,
            "turn_result": _serialise_turn_result(resp.turn_result),
            "ok": resp.ok,
            "error": resp.error,
            "latency_ms": latency_ms,
            "usage": usage,
        })


# --- benchmark artifacts & background jobs -----------------------------

    def _handle_benchmarks(self) -> None:
        """List exported benchmark artifacts + in-flight run jobs."""
        self._send_json({
            "experiments": _load_experiments(),
            "jobs": _jobs_list(),
            "datasets": _datasets_available(),
            "strategies": _strategy_catalog(),
        })

    def _handle_benchmark_run(self) -> None:
        """Start ONE background benchmark/ablation run (never concurrent).

        The repository CLI (``python -m src.eval.run_benchmark``) is reused
        verbatim — the API only validates input, builds the argv and exports
        into ``results/``.  Nothing runs unless a client explicitly calls
        this endpoint.
        """
        body = self._read_body()
        err, argv, name = _build_benchmark_argv(body)
        if err:
            return self._send_json({"error": err}, 400)

        job, conflict = _start_job(name, argv, body)
        if conflict:
            return self._send_json({"error": conflict, "job": job}, 409)
        self._send_json({"job": job}, 202)


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
# Benchmark artifacts (results/) — read-only exposure for the UI
# ---------------------------------------------------------------------------

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
_DATASETS = ("synthetic", "locomo", "longmemeval")
_STRATEGY_ORDER = ("none", "full_history", "memoryos", "memory_r1", "hmem")
_SIGNAL_NAMES = ("importance", "temporal", "contradiction", "storage")
_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _datasets_available() -> dict:
    """Which benchmark datasets can actually load from the repo data dir."""
    out: dict = {"synthetic": True}
    try:
        from src.eval.adapters import resolve_dataset_file
    except Exception:
        return {ds: False for ds in _DATASETS}
    for ds in ("locomo", "longmemeval"):
        try:
            resolve_dataset_file(ds, None)
            out[ds] = True
        except Exception:
            out[ds] = False
    return out


def _strategy_catalog() -> list:
    """Canonical strategy names + descriptions (from the benchmark CLI)."""
    try:
        from src.eval.run_benchmark import SUPPORTED_BASELINES
        return [{"name": n, "description": SUPPORTED_BASELINES[n]}
                for n in _STRATEGY_ORDER if n in SUPPORTED_BASELINES]
    except Exception:
        return [{"name": n, "description": ""} for n in _STRATEGY_ORDER]


def _row_metrics(report: dict) -> dict:
    """Project a BenchmarkReport dict down to its aggregate metrics."""
    if not isinstance(report, dict):
        return {}
    return {key: report.get(key) for key in (
        "mean_f1", "mean_bleu1", "mean_recall_at_k", "mean_precision_at_k",
        "mrr", "mean_tokens", "mean_latency_ms", "num_queries",
    )}


def _strategy_rows(results: list, kinds: dict | None = None) -> list:
    """"results" list -> flat rows of {name, kind, metrics...}."""
    rows = []
    for r in results or []:
        if not isinstance(r, dict):
            continue
        name = r.get("name", "?")
        rows.append({
            "name": name,
            "kind": (kinds or {}).get(name, ""),
            **_row_metrics(r.get("report")),
        })
    return rows


def _load_experiments() -> list:
    """Parse every exported ``results/*.json`` into a UI-ready record.

    Recognises the two shapes the benchmark CLI exports:
    ``ComparisonResult`` ({dataset_name,k,num_cases,results}) and
    ``AblationResult`` ({ablation,configs,comparison}).  Files that match
    neither schema are skipped (never guessed at).
    """
    experiments: list = []
    if not _RESULTS_DIR.is_dir():
        return experiments
    for fp in sorted(_RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue

        record: dict = {
            "file": fp.name,
            "modified": datetime.fromtimestamp(
                fp.stat().st_mtime, timezone.utc
            ).isoformat(),
        }
        if isinstance(data.get("comparison"), dict) and "ablation" in data:
            comp = data["comparison"]
            kinds = {c.get("label"): c.get("kind", "")
                     for c in data.get("configs", []) if isinstance(c, dict)}
            record.update(
                type="ablation",
                ablation=data.get("ablation", ""),
                dataset_name=comp.get("dataset_name"),
                k=comp.get("k"),
                num_cases=comp.get("num_cases"),
                configs=list(kinds.keys()),
                rows=_strategy_rows(comp.get("results"), kinds),
            )
        elif "dataset_name" in data and isinstance(data.get("results"), list):
            record.update(
                type="comparison",
                ablation="",
                dataset_name=data.get("dataset_name"),
                k=data.get("k"),
                num_cases=data.get("num_cases"),
                rows=_strategy_rows(data.get("results")),
            )
        else:
            continue  # unrelated JSON — do not fabricate a record
        experiments.append(record)
    return experiments


# ---------------------------------------------------------------------------
# Background benchmark jobs (one at a time, quota-safe)
# ---------------------------------------------------------------------------

_JOBS: dict = {}
_JOBS_LOCK = threading.Lock()
_MAX_SAMPLE = 20  # hard cap: a "small smoke run" cannot be turned into a full


def _jobs_list() -> list:
    with _JOBS_LOCK:
        return sorted(_JOBS.values(), key=lambda j: j["started"], reverse=True)


def _any_job_running() -> dict | None:
    with _JOBS_LOCK:
        for job in _JOBS.values():
            if job["status"] == "running":
                return job
    return None


def _build_benchmark_argv(body: dict) -> tuple:
    """Validate a run request.

    Returns ``(error, argv, name)`` — *error* is ``None`` on success.
    Only whitelisted values are accepted; nothing from the request reaches
    the shell (the CLI runs in-process, argv is a list, never a string).
    """
    dataset = str(body.get("dataset", "")).strip()
    kind = str(body.get("kind", "")).strip()
    if dataset not in _DATASETS:
        return f"dataset must be one of {list(_DATASETS)}", None, None
    if dataset != "synthetic" and not _datasets_available().get(dataset):
        return f"dataset '{dataset}' is not available locally (see data/README.md)", None, None
    if kind not in ("compare", "ablate"):
        return "kind must be 'compare' or 'ablate'", None, None

    argv = ["--dataset", dataset]
    name_parts = [dataset]

    if kind == "compare":
        strategies = body.get("strategies") or []
        if not isinstance(strategies, list) or not strategies:
            return "strategies must be a non-empty list", None, None
        names = [str(s).strip() for s in strategies]
        bad = [n for n in names if n not in _STRATEGY_ORDER]
        if bad:
            return f"unknown strategies {bad}; supported: {list(_STRATEGY_ORDER)}", None, None
        if len(set(names)) != len(names):
            return "duplicate strategy names", None, None
        argv += ["--compare", ",".join(names)]
        name_parts += ["compare", "-".join(names)]
    else:
        spec = body.get("ablation") or {}
        akind = str(spec.get("kind", "")).strip()
        if akind == "stage":
            ablate_arg = "stage"
            name_parts += ["ablate", "stage"]
        elif akind in ("k", "budget"):
            values = spec.get("values") or []
            if not isinstance(values, list) or not values:
                return f"ablation '{akind}' needs a non-empty values list", None, None
            try:
                ints = [int(v) for v in values]
            except (TypeError, ValueError):
                return "ablation values must be integers", None, None
            if any(v <= 0 for v in ints):
                return "ablation values must be > 0", None, None
            ablate_arg = f"{akind}:" + ",".join(str(v) for v in ints)
            name_parts += ["ablate", akind]
        elif akind == "signal":
            sig = str(spec.get("signal", "")).strip()
            if sig not in _SIGNAL_NAMES:
                return f"signal must be one of {list(_SIGNAL_NAMES)}", None, None
            ablate_arg = f"signal:{sig}"
            name_parts += ["ablate", sig]
        else:
            return ("ablation.kind must be one of stage, k, budget, signal",
                    None, None)
        argv += ["--ablate", ablate_arg]

    sample = body.get("sample")
    if sample not in (None, "", 0):
        try:
            sample = int(sample)
        except (TypeError, ValueError):
            return "sample must be an integer", None, None
        if sample <= 0 or sample > _MAX_SAMPLE:
            return f"sample must be between 1 and {_MAX_SAMPLE}", None, None
        argv += ["--sample", str(sample)]
        name_parts.append(f"n{sample}")

    for flag, key in (("--k", "k"), ("--token-budget", "token_budget")):
        val = body.get(key)
        if val not in (None, "", 0):
            try:
                val = int(val)
            except (TypeError, ValueError):
                return f"{key} must be an integer", None, None
            if val <= 0:
                return f"{key} must be > 0", None, None
            argv += [flag, str(val)]

    raw_name = str(body.get("name", "")).strip() or "_".join(name_parts)
    name = _NAME_RE.sub("_", raw_name).strip("._-")[:80] or "run"
    return None, argv, name


def _start_job(name: str, argv: list, body: dict) -> tuple:
    """Create + launch one background job.  Returns ``(job, conflict)``."""
    running = _any_job_running()
    if running:
        return running, (
            "a benchmark job is already running — wait for it to finish "
            "(quota-safe single-concurrency limit)"
        )

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _RESULTS_DIR / f"{name}.json"
    csv_path = _RESULTS_DIR / f"{name}.csv"
    job_id = f"{int(time.time())}-{name}"
    job = {
        "id": job_id,
        "name": name,
        "dataset": str(body.get("dataset", "")),
        "kind": str(body.get("kind", "")),
        "argv": argv + ["--export-json", str(json_path),
                        "--export-csv", str(csv_path)],
        "sample": body.get("sample"),
        "status": "running",
        "started": datetime.now(timezone.utc).isoformat(),
        "finished": None,
        "exit_code": None,
        "error": None,
        "log": "",
    }
    with _JOBS_LOCK:
        _JOBS[job_id] = job

    def _runner() -> None:
        import contextlib
        import io as _io
        buf = _io.StringIO()
        code: int | None = None
        try:
            from src.eval.run_benchmark import main as _bench_main
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                code = _bench_main(job["argv"])
        except Exception as exc:  # provider/dependency failures, not crashes
            buf.write(f"\n[job crashed] {type(exc).__name__}: {exc}\n")
            job["error"] = f"{type(exc).__name__}: {exc}"
            job["status"] = "failed"
        finally:
            job["log"] = buf.getvalue()[-6000:]
            job["exit_code"] = code
            if job["status"] == "running":
                job["status"] = "done" if code == 0 else "failed"
                if code not in (0, None) and not job["error"]:
                    job["error"] = f"benchmark exited with code {code}"
            job["finished"] = datetime.now(timezone.utc).isoformat()

    thread = threading.Thread(target=_runner, daemon=True, name=f"bench-{name}")
    thread.start()
    return job, None


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
