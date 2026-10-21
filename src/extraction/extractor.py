"""Fact extraction: dialogue turn -> list of CandidateMemoryItem dicts.

Owner: Aryan. Interface contract: docs/interfaces.md
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

SYSTEM_PROMPT = (
    "You extract long-term memories from conversations. "
    "Reply with ONLY a JSON array of strings, nothing else."
)

INSTRUCTIONS = """Extract the facts worth remembering long-term from the message below: \
personal details, preferences, plans, events, relationships, opinions and decisions.

Rules:
- Each fact is ONE short, self-contained sentence that makes sense without the conversation.
- Use the speaker's name instead of "I", "he" or "she".
- If a date is given, turn relative times ("yesterday", "last week") into absolute dates when you can.
- Ignore greetings, small talk, and questions that carry no information.
- If nothing is worth remembering, return [].

Example output: ["Maya adopted a dog named Biscuit in March 2023."]
"""


def _build_prompt(turn: str, metadata: dict) -> str:
    speaker = metadata.get("speaker", "Unknown")
    date = metadata.get("timestamp", "unknown")
    return f"{INSTRUCTIONS}\nSpeaker: {speaker}\nDate: {date}\nMessage: {turn}\n"


def _parse_facts(raw: str) -> list[str]:
    """Pull a JSON array of strings out of the model's reply. Raises ValueError."""
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        raise ValueError("no JSON array in reply")
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(str(e))
    if not isinstance(data, list):
        raise ValueError("reply is not a list")
    return [f.strip() for f in data if isinstance(f, str) and f.strip()]


def extract_candidates(turn: str, metadata: dict | None = None,
                       llm=None, max_attempts: int = 2) -> list[dict]:
    """Turn one dialogue turn into CandidateMemoryItem dicts.

    metadata keys (all optional): speaker, timestamp (ISO 8601), source
    (e.g. "D1:3"). The store fills in `embedding`, so it is None here.
    """
    metadata = metadata or {}
    if not turn or not turn.strip():
        return []
    if llm is None:
        from src.llm.client import call_llm as llm  # Krish's module

    prompt = _build_prompt(turn, metadata)
    facts: list[str] = []
    for _ in range(max_attempts):
        try:
            facts = _parse_facts(llm(prompt, system=SYSTEM_PROMPT))
            break
        except ValueError:
            continue  # malformed reply -> try again

    source = metadata.get("source", "unknown")
    timestamp = metadata.get("timestamp") or datetime.now(timezone.utc).isoformat()
    return [
        {
            "id": f"{source}__{i}",   # traceable back to the dialogue turn
            "text": fact,
            "timestamp": timestamp,
            "source": source,
            "embedding": None,
        }
        for i, fact in enumerate(facts)
    ]