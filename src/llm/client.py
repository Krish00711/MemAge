"""Provider-agnostic LLM client for MemAge.

Exposes one function used by every module that needs an LLM:

    call_llm(prompt: str, system: str = None, **kwargs) -> str

Provider is selected via the LLM_PROVIDER environment variable
("gemini", "groq", or "mistral").  API keys are read from GEMINI_API_KEY /
GROQ_API_KEY / MISTRAL_API_KEY.  No SDK is imported at module level so that
controller and extraction code can be imported and unit-tested without any
provider package installed.

Owner: Krish.  Interface contract: docs/interfaces.md
"""
from __future__ import annotations

import os
import re
import time

# ---------------------------------------------------------------------------
# Default model names — override via env if needed
# ---------------------------------------------------------------------------
_DEFAULT_GEMINI_MODEL  = "gemini-1.5-flash"
# llama-3.1-8b-instant was shut down by Groq on 2026-08-16; gpt-oss-20b is the
# recommended replacement.
_DEFAULT_GROQ_MODEL    = "openai/gpt-oss-20b"
_DEFAULT_MISTRAL_MODEL = "mistral-small-latest"


class LLMRateLimitError(RuntimeError):
    """Raised when a provider rate limit cannot be recovered by a short wait.

    Subclasses RuntimeError so existing ``except RuntimeError`` code keeps
    working.  Benchmark scripts can catch this specifically to save progress
    and stop cleanly (e.g. when the Groq daily token cap is hit) instead of
    recording an error string as a model answer.
    """


def _load_env() -> None:
    """Load .env if python-dotenv is available (silently skips if not)."""
    try:
        from dotenv import load_dotenv  # type: ignore[import]
        load_dotenv(override=False)
    except ImportError:
        pass  # dotenv not installed; rely on shell environment


def call_llm(prompt: str, system: str | None = None, **kwargs) -> str:
    """Call the configured LLM provider and return the response text.

    Parameters
    ----------
    prompt:
        User-facing prompt string.
    system:
        Optional system instruction.  Gemini receives it prepended to the
        prompt; Groq/Mistral receive it as a ``{"role": "system", ...}``
        message.
    **kwargs:
        Forwarded to the underlying SDK call.  Supported: ``temperature`` (all
        providers) and ``max_tokens`` / ``max_completion_tokens`` (Groq).
        Unknown kwargs are silently ignored.

    Returns
    -------
    str
        The model's text response, stripped of leading/trailing whitespace.

    Raises
    ------
    EnvironmentError
        If the required API key is missing from the environment.
    ImportError
        If the required provider SDK is not installed.
    LLMRateLimitError
        If a rate limit cannot be recovered with a short wait (Groq).
    RuntimeError
        If the provider call fails or returns an empty response.
    """
    _load_env()

    provider = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()

    if provider == "gemini":
        return _call_gemini(prompt, system, **kwargs)
    elif provider == "groq":
        return _call_groq(prompt, system, **kwargs)
    elif provider == "mistral":
        return _call_mistral(prompt, system, **kwargs)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. Set LLM_PROVIDER=gemini, groq, or mistral."
        )


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def _call_gemini(prompt: str, system: str | None, **kwargs) -> str:
    try:
        import google.generativeai as genai  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "google-generativeai is not installed. "
            "Run: pip install google-generativeai==0.7.2"
        ) from exc

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )

    genai.configure(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL", _DEFAULT_GEMINI_MODEL)

    # Build the prompt: if a system instruction is provided, prepend it.
    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    # google-generativeai 0.7.x: GenerativeModel.generate_content()
    model = genai.GenerativeModel(model_name)

    # Pass only kwargs the SDK accepts; temperature is the common one.
    gen_kwargs: dict = {}
    if "temperature" in kwargs:
        gen_kwargs["generation_config"] = {"temperature": kwargs["temperature"]}

    try:
        response = model.generate_content(full_prompt, **gen_kwargs)
    except Exception as exc:
        raise RuntimeError(f"Gemini API call failed: {exc}") from exc

    text = ""
    try:
        text = response.text
    except Exception:
        # response.text can raise if safety filters blocked the response
        try:
            text = response.candidates[0].content.parts[0].text
        except Exception:
            pass

    if not text or not text.strip():
        raise RuntimeError("Gemini returned an empty response.")

    return text.strip()


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------

# Transient rate-limit retry.  Groq's free tier has tight per-minute token
# limits (TPM) and per-day limits (TPD/RPD).  A rejected 429 consumes no quota,
# so a bounded wait turns a transient TPM blip into a success.  If the
# provider asks us to wait longer than _MAX_WAIT_SECONDS (typically the daily
# cap), we raise LLMRateLimitError so the caller can checkpoint and stop.

_MAX_RATE_RETRIES = int(os.environ.get("LLM_MAX_RATE_RETRIES", "6"))
_MAX_WAIT_SECONDS = float(os.environ.get("LLM_MAX_WAIT_SECONDS", "90"))

# Matches hints such as "try again in 2.5s", "7m12.96s", "1h2m3s", "850ms".
_RETRY_HINT_RE = re.compile(
    r"try again in\s+((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)", re.IGNORECASE
)
_DURATION_PART_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)", re.IGNORECASE)
_UNIT_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


def _parse_duration(text: str) -> float:
    """Convert a duration like '7m12.96s' into seconds."""
    return sum(
        float(value) * _UNIT_SECONDS[unit.lower()]
        for value, unit in _DURATION_PART_RE.findall(text)
    )


def _is_rate_limit(message: str) -> bool:
    low = message.lower()
    return "429" in message or "rate_limit" in low or "rate limit" in low


def _rate_limit_delay(message: str, attempt: int) -> float | None:
    """Seconds to wait for a rate-limit error, else ``None`` if not one."""
    if not _is_rate_limit(message):
        return None
    match = _RETRY_HINT_RE.search(message)
    if match:
        # Honour the provider's own hint, with a small safety margin.
        return _parse_duration(match.group(1)) + 0.5
    # No hint: exponential backoff (2s, 4s, 8s, ... capped at 30s).
    return min(2.0 * (2 ** attempt), 30.0)


def _call_groq(prompt: str, system: str | None, **kwargs) -> str:
    try:
        from groq import Groq  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "groq is not installed. "
            "Run: pip install -U groq"
        ) from exc

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )

    model_name = os.environ.get("GROQ_MODEL", _DEFAULT_GROQ_MODEL).strip()

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    create_kwargs: dict = {}
    if "temperature" in kwargs:
        create_kwargs["temperature"] = kwargs["temperature"]

    # Groq's parameter is max_completion_tokens; accept the common alias too.
    max_tokens = kwargs.get("max_completion_tokens", kwargs.get("max_tokens"))
    if max_tokens:
        create_kwargs["max_completion_tokens"] = max_tokens

    # gpt-oss models are reasoning models: keep reasoning short so that
    # extraction / controller calls stay fast and don't burn the token budget.
    if model_name.startswith("openai/gpt-oss"):
        create_kwargs["extra_body"] = {
            "reasoning_effort": os.environ.get("GROQ_REASONING_EFFORT", "low")
        }

    client = Groq(api_key=api_key)
    completion = None
    for attempt in range(1 + _MAX_RATE_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                **create_kwargs,
            )
            break
        except Exception as exc:
            delay = _rate_limit_delay(str(exc), attempt)
            if delay is None:
                # Not a rate-limit error: never retry.
                raise RuntimeError(f"Groq API call failed: {exc}") from exc
            if delay > _MAX_WAIT_SECONDS or attempt == _MAX_RATE_RETRIES:
                raise LLMRateLimitError(
                    f"Groq rate limit not recoverable (wait={delay:.0f}s, "
                    f"attempt={attempt + 1}): {exc}"
                ) from exc
            time.sleep(delay)
    if completion is None:  # pragma: no cover - loop always sets or raises
        raise RuntimeError("Groq API call failed: no completion produced.")

    try:
        text = completion.choices[0].message.content
    except (IndexError, AttributeError) as exc:
        raise RuntimeError(f"Groq returned an unexpected response structure: {exc}") from exc

    if not text or not text.strip():
        raise RuntimeError(
            "Groq returned an empty response (if using a reasoning model, the "
            "token budget may have been consumed by reasoning; raise max_tokens "
            "or lower GROQ_REASONING_EFFORT)."
        )

    return text.strip()


# ---------------------------------------------------------------------------
# Mistral
# ---------------------------------------------------------------------------

def _call_mistral(prompt: str, system: str | None, **kwargs) -> str:
    try:
        # mistralai 3.x exposes Mistral from mistralai.client
        from mistralai.client import Mistral  # type: ignore[import]
    except (ImportError, ModuleNotFoundError):
        try:
            # Older 1.x layout or test stubs inject at top-level
            from mistralai import Mistral  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "mistralai is not installed. "
                "Run: pip install 'mistralai>=1.0.0'"
            ) from exc

    api_key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "MISTRAL_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )

    model_name = os.environ.get("MISTRAL_MODEL", _DEFAULT_MISTRAL_MODEL)

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    create_kwargs: dict = {}
    if "temperature" in kwargs:
        create_kwargs["temperature"] = kwargs["temperature"]

    client = Mistral(api_key=api_key)
    try:
        response = client.chat.complete(
            model=model_name,
            messages=messages,
            **create_kwargs,
        )
    except Exception as exc:
        raise RuntimeError(f"Mistral API call failed: {exc}") from exc

    try:
        text = response.choices[0].message.content
    except (IndexError, AttributeError) as exc:
        raise RuntimeError(
            f"Mistral returned an unexpected response structure: {exc}"
        ) from exc

    if not text or not text.strip():
        raise RuntimeError("Mistral returned an empty response.")

    return text.strip()