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

# ---------------------------------------------------------------------------
# Default model names — override via env if needed
# ---------------------------------------------------------------------------
_DEFAULT_GEMINI_MODEL  = "gemini-1.5-flash"
_DEFAULT_GROQ_MODEL    = "llama3-8b-8192"
_DEFAULT_MISTRAL_MODEL = "mistral-small-latest"


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
        Optional system instruction.  Gemini receives it as the first
        ``"user"`` turn prefixed with a newline separator; Groq receives it
        as a ``{"role": "system", ...}`` message.
    **kwargs:
        Forwarded to the underlying SDK call (e.g. ``temperature``).
        Unknown kwargs are silently ignored if the SDK does not accept them.

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

def _call_groq(prompt: str, system: str | None, **kwargs) -> str:
    try:
        from groq import Groq  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "groq is not installed. "
            "Run: pip install groq==0.9.0"
        ) from exc

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )

    model_name = os.environ.get("GROQ_MODEL", _DEFAULT_GROQ_MODEL)

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # Forward temperature if supplied
    create_kwargs: dict = {}
    if "temperature" in kwargs:
        create_kwargs["temperature"] = kwargs["temperature"]

    client = Groq(api_key=api_key)
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            **create_kwargs,
        )
    except Exception as exc:
        raise RuntimeError(f"Groq API call failed: {exc}") from exc

    try:
        text = completion.choices[0].message.content
    except (IndexError, AttributeError) as exc:
        raise RuntimeError(f"Groq returned an unexpected response structure: {exc}") from exc

    if not text or not text.strip():
        raise RuntimeError("Groq returned an empty response.")

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
