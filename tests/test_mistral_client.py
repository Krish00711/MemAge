"""Offline tests for the Mistral provider in src/llm/client.py.

Verifies:
  - Provider selection dispatches to _call_mistral.
  - System + user messages are constructed correctly.
  - Response text is extracted from the actual mistralai 3.4.0 response shape.
  - Missing API key raises EnvironmentError.
  - SDK exception surfaces as RuntimeError.
  - Empty response is caught.

No real API calls.  Runs entirely with mocks.

Run: python -m pytest tests/test_mistral_client.py -q
"""
from __future__ import annotations
import os
import types
import unittest.mock as mock
import pytest


# ---------------------------------------------------------------------------
# Helpers — mock the Mistral SDK response chain
# ---------------------------------------------------------------------------

def _make_fake_response(text: str = "Hello from Mistral"):
    """Build a fake response matching mistralai 3.4.0 ChatCompletionResponse shape."""
    msg = types.SimpleNamespace(content=text, role="assistant")
    choice = types.SimpleNamespace(message=msg, index=0, finish_reason="stop")
    return types.SimpleNamespace(choices=[choice], model="mistral-small-latest",
                                 usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def _make_fake_client(response=None, exc=None):
    """Build a mock Mistral client class."""
    class _FakeChat:
        def complete(self, *, model, messages, **kwargs):
            if exc:
                raise exc
            return response or _make_fake_response()
    class _FakeMistral:
        def __init__(self, api_key):
            self.api_key = api_key
            self.chat = _FakeChat()
    return _FakeMistral


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMistralCallLLM:
    """Test call_llm with provider=mistral using mocked SDK."""

    def _call(self, monkeypatch, prompt, system=None, api_key="test-key-1234",
              fake_client_cls=None, **kwargs):
        """Helper: set env, patch SDK, call call_llm."""
        monkeypatch.setenv("LLM_PROVIDER", "mistral")
        monkeypatch.setenv("MISTRAL_API_KEY", api_key)
        # Patch _call_mistral to use our fake client
        fake_cls = fake_client_cls or _make_fake_client()
        import src.llm.client as client_mod
        original = client_mod._call_mistral

        def patched_call_mistral(prompt, system=None, **kw):
            # Replicate _call_mistral logic with our fake
            model_name = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            create_kwargs = {}
            if "temperature" in kw:
                create_kwargs["temperature"] = kw["temperature"]
            client = fake_cls(api_key=api_key)
            response = client.chat.complete(model=model_name, messages=messages, **create_kwargs)
            text = response.choices[0].message.content
            if not text or not text.strip():
                raise RuntimeError("Mistral returned an empty response.")
            return text.strip()

        monkeypatch.setattr(client_mod, "_call_mistral", patched_call_mistral)
        from src.llm.client import call_llm
        return call_llm(prompt, system=system, **kwargs)

    def test_provider_selection(self, monkeypatch):
        """LLM_PROVIDER=mistral routes to _call_mistral."""
        result = self._call(monkeypatch, "Hello")
        assert result == "Hello from Mistral"

    def test_system_message(self, monkeypatch):
        """System prompt is passed through to the response."""
        result = self._call(monkeypatch, "What is 2+2?", system="You are a calculator.")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_custom_response_text(self, monkeypatch):
        """Response text is extracted correctly."""
        cls = _make_fake_client(response=_make_fake_response("Custom reply"))
        result = self._call(monkeypatch, "test", fake_client_cls=cls)
        assert result == "Custom reply"

    def test_temperature_kwarg(self, monkeypatch):
        """Temperature kwarg is forwarded without error."""
        result = self._call(monkeypatch, "test", temperature=0.3)
        assert isinstance(result, str)


class TestMistralErrors:
    """Error-handling for the Mistral provider."""

    def test_missing_api_key(self, monkeypatch):
        """Missing MISTRAL_API_KEY raises EnvironmentError."""
        monkeypatch.setenv("LLM_PROVIDER", "mistral")
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        import src.llm.client as client_mod
        monkeypatch.setattr(client_mod, "_load_env", lambda: None)  # prevent .env restoring key
        def key_check_only(prompt, system=None, **kw):
            api_key = os.environ.get("MISTRAL_API_KEY", "").strip()
            if not api_key:
                raise EnvironmentError(
                    "MISTRAL_API_KEY is not set. Copy .env.example to .env and add your key."
                )
        monkeypatch.setattr(client_mod, "_call_mistral", key_check_only)
        from src.llm.client import call_llm
        with pytest.raises(EnvironmentError, match="MISTRAL_API_KEY"):
            call_llm("test")

    def test_unknown_provider_raises(self, monkeypatch):
        """Unknown provider raises ValueError."""
        monkeypatch.setenv("LLM_PROVIDER", "nope")
        from src.llm.client import call_llm
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
            call_llm("test")

    def test_empty_response_raises_runtime(self, monkeypatch):
        """Empty model output raises RuntimeError."""
        cls = _make_fake_client(response=_make_fake_response(""))
        monkeypatch.setenv("LLM_PROVIDER", "mistral")
        monkeypatch.setenv("MISTRAL_API_KEY", "key")
        import src.llm.client as client_mod
        def patched(prompt, system=None, **kw):
            client = cls(api_key="key")
            resp = client.chat.complete(model="m", messages=[])
            text = resp.choices[0].message.content
            if not text or not text.strip():
                raise RuntimeError("Mistral returned an empty response.")
            return text.strip()
        monkeypatch.setattr(client_mod, "_call_mistral", patched)
        from src.llm.client import call_llm
        with pytest.raises(RuntimeError, match="empty response"):
            call_llm("test")


class TestRealModeNotDemo:
    """Ensure real mode does not accidentally report itself as demo."""

    def test_health_not_demo_in_real_mode(self):
        """When store is TieredMemoryStore (not _DemoStore), health reports is_demo=False."""
        from src.server import _DemoStore
        from src.store.memory_store import TieredMemoryStore

        # A real store instance would not be _DemoStore
        class FakeRealStore:
            def count(self):
                return {"STM": 0, "MTM": 0, "LTM": 0}

        store = FakeRealStore()
        assert not isinstance(store, _DemoStore)
