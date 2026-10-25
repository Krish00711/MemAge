"""MemAge configuration validator.

Run from the project root:

    python -m src.check_config

Reports missing environment variables and package availability without
printing secret values or making network requests.
"""
from __future__ import annotations

import os
import sys


def _try_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _env_present(key: str) -> bool:
    return bool(os.environ.get(key, "").strip())


def _masked(key: str) -> str:
    """Return a safe representation: shows prefix only, never the full value."""
    val = os.environ.get(key, "")
    if not val:
        return "(not set)"
    return val[:4] + "****"


def check() -> int:  # returns 0 = ok, 1 = problems found
    # Load .env if available
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass

    ok = True
    print("=== MemAge configuration check ===\n")

    # ------------------------------------------------------------------ LLM
    print("-- LLM provider --")
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if not provider:
        print("  [MISSING] LLM_PROVIDER  (set to 'gemini', 'groq', or 'mistral')")
        ok = False
    else:
        print(f"  [OK]      LLM_PROVIDER = {provider}")

    if provider in ("gemini", ""):
        if _env_present("GEMINI_API_KEY"):
            print(f"  [OK]      GEMINI_API_KEY = {_masked('GEMINI_API_KEY')}")
        else:
            print("  [MISSING] GEMINI_API_KEY")
            if provider == "gemini":
                ok = False
        sdk_ok = _try_import("google.generativeai")
        print(f"  {'[OK]' if sdk_ok else '[MISSING]'}     google-generativeai SDK")
        if not sdk_ok:
            print("            pip install google-generativeai==0.7.2")
            if provider == "gemini":
                ok = False

    if provider in ("groq", ""):
        if _env_present("GROQ_API_KEY"):
            print(f"  [OK]      GROQ_API_KEY = {_masked('GROQ_API_KEY')}")
        else:
            print("  [MISSING] GROQ_API_KEY")
            if provider == "groq":
                ok = False
        sdk_ok = _try_import("groq")
        print(f"  {'[OK]' if sdk_ok else '[MISSING]'}     groq SDK")
        if not sdk_ok:
            print("            pip install groq==0.9.0")
            if provider == "groq":
                ok = False

    if provider in ("mistral", ""):
        if _env_present("MISTRAL_API_KEY"):
            print(f"  [OK]      MISTRAL_API_KEY = {_masked('MISTRAL_API_KEY')}")
        else:
            print("  [MISSING] MISTRAL_API_KEY")
            if provider == "mistral":
                ok = False
        sdk_ok = _try_import("mistralai")
        print(f"  {'[OK]' if sdk_ok else '[MISSING]'}     mistralai SDK")
        if not sdk_ok:
            print("            pip install 'mistralai>=1.0.0'")
            if provider == "mistral":
                ok = False

    # ------------------------------------------------------------------ Store
    print("\n-- Vector store --")
    chroma_dir = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma")
    print(f"  CHROMA_PERSIST_DIR = {chroma_dir}")
    chroma_ok = _try_import("chromadb")
    fe_ok = _try_import("fastembed")
    st_ok = _try_import("sentence_transformers")
    print(f"  {'[OK]' if chroma_ok else '[MISSING]'}     chromadb")
    if not chroma_ok:
        print("            pip install chromadb==0.5.5")
        ok = False
    embed_ok = fe_ok or st_ok
    if fe_ok:
        print("  [OK]      fastembed (lightweight ONNX embeddings)")
    elif st_ok:
        print("  [OK]      sentence-transformers")
    else:
        print("  [MISSING] Embedding backend (fastembed or sentence-transformers)")
        print("            Recommended: pip install fastembed==0.3.6  (~30 MB, ONNX, no PyTorch)")
        print("            Alternative: pip install sentence-transformers==3.0.1  (~1.5 GB with PyTorch)")
        ok = False

    # ------------------------------------------------------------------ Disk
    print("\n-- Disk space --")
    try:
        stat = os.statvfs(".")
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        print(f"  Free space: {free_gb:.1f} GB")
        if free_gb < 1.0:
            print("  [WARNING] Less than 1 GB free — sentence-transformers model may not fit.")
            print("            Consider fastembed (pip install fastembed==0.3.6, ~30 MB).")
            ok = False
        elif free_gb < 0.5:
            print("  [ERROR] Less than 500 MB free — installation will likely fail.")
            ok = False
    except Exception:
        print("  (could not determine disk space)")

    # ------------------------------------------------------------------ Core deps
    print("\n-- Core dependencies --")
    for pkg, imp in [("python-dotenv", "dotenv"), ("numpy", "numpy"),
                     ("pandas", "pandas")]:
        avail = _try_import(imp)
        print(f"  {'[OK]' if avail else '[MISSING]'}     {pkg}")

    # ------------------------------------------------------------------ Summary
    print()
    if ok:
        print("All required settings are present.")
    else:
        print("One or more required settings are missing — see above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(check())
