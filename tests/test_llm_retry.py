"""Hermetic tests for the Groq transient rate-limit retry in src/llm/client.py.

No network, no real API key: the provider SDK is stubbed via sys.modules and
``time.sleep`` is patched, so these tests never consume Groq quota.
"""
from __future__ import annotations

import sys
import types

import pytest


def test_rate_limit_delay_parses_provider_hint():
    from src.llm.client import _rate_limit_delay

    msg = (
        "Error code: 429 - {'error': {'message': 'Rate limit reached ... "
        "Limit 8000, Used 5598, Requested 2645. Please try again in 1.8225s. "
        "Need more tokens?', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
    )
    delay = _rate_limit_delay(msg, attempt=0)
    assert delay is not None
    assert 1.8 <= delay <= 3.0  # hint + small margin


def test_rate_limit_delay_backoff_without_hint():
    from src.llm.client import _rate_limit_delay

    msg = "Error code: 429 - rate_limit_exceeded"
    assert _rate_limit_delay(msg, attempt=0) == pytest.approx(2.0)
    assert _rate_limit_delay(msg, attempt=1) == pytest.approx(4.0)
    assert _rate_limit_delay(msg, attempt=10) == pytest.approx(30.0)  # capped


def test_rate_limit_delay_ignores_other_errors():
    from src.llm.client import _rate_limit_delay

    assert _rate_limit_delay("Error code: 401 - Invalid API Key", 0) is None
    assert _rate_limit_delay("Error code: 500 - server error", 0) is None
    assert _rate_limit_delay("connection reset", 0) is None


def test_call_groq_retries_rate_limit_then_succeeds(monkeypatch):
    """A 429 with a retry hint is retried (no real sleep) and succeeds."""
    from src.llm import client

    calls = {"n": 0}
    sleeps: list[float] = []

    class _FakeCompletions:
        def create(self, **_kwargs):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise RuntimeError(
                    "Error code: 429 - rate_limit_exceeded. "
                    "Please try again in 0.01s."
                )
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(content="ok")
                )]
            )

    fake_mod = types.ModuleType("groq")
    fake_mod.Groq = lambda api_key=None: types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=_FakeCompletions())
    )
    monkeypatch.setitem(sys.modules, "groq", fake_mod)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(client, "_load_env", lambda: None)
    monkeypatch.setattr(client.time, "sleep", sleeps.append)

    out = client.call_llm("hello")
    assert out == "ok"
    assert calls["n"] == 3  # 2 rejected attempts + 1 success
    assert len(sleeps) == 2  # slept between attempts, no real waiting


def test_call_groq_non_rate_limit_error_not_retried(monkeypatch):
    """Non-429 errors surface immediately (exactly one attempt)."""
    from src.llm import client

    calls = {"n": 0}

    class _FakeCompletions:
        def create(self, **_kwargs):
            calls["n"] += 1
            raise RuntimeError("Error code: 401 - Invalid API Key")

    fake_mod = types.ModuleType("groq")
    fake_mod.Groq = lambda api_key=None: types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=_FakeCompletions())
    )
    monkeypatch.setitem(sys.modules, "groq", fake_mod)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(client, "_load_env", lambda: None)
    monkeypatch.setattr(client.time, "sleep", lambda s: pytest.fail(
        "non-rate-limit errors must not sleep/retry"
    ))

    with pytest.raises(RuntimeError, match="401"):
        client.call_llm("hello")
    assert calls["n"] == 1


def test_call_groq_gives_up_after_max_retries(monkeypatch):
    """Persistent 429s fail with RuntimeError after the bounded attempts."""
    from src.llm import client

    calls = {"n": 0}

    class _FakeCompletions:
        def create(self, **_kwargs):
            calls["n"] += 1
            raise RuntimeError(
                "Error code: 429 - rate_limit_exceeded. "
                "Please try again in 0.01s."
            )

    fake_mod = types.ModuleType("groq")
    fake_mod.Groq = lambda api_key=None: types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=_FakeCompletions())
    )
    monkeypatch.setitem(sys.modules, "groq", fake_mod)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(client, "_load_env", lambda: None)
    monkeypatch.setattr(client.time, "sleep", lambda s: None)

    with pytest.raises(RuntimeError, match="429"):
        client.call_llm("hello")
    assert calls["n"] == 1 + client._MAX_RATE_RETRIES
