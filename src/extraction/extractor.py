"""Fact extraction: dialogue turn -> list of CandidateMemoryItem dicts.

Owner: Aryan. Interface contract: docs/interfaces.md
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

# Bump this whenever the prompt changes: cached extractions made with an older
# prompt are then ignored automatically (the loader puts it in the cache name).
PROMPT_VERSION = "v4"

# Hard cap per dialogue turn: a single chat message rarely holds more than a
# couple of durable facts, so extra output is almost always filler.
MAX_FACTS_PER_TURN = 3

SYSTEM_PROMPT = (
    "You extract long-term memories from conversations. "
    "Reply with ONLY a JSON array of strings, nothing else."
)

INSTRUCTIONS = """Extract the facts worth remembering long-term from the MESSAGE below.

Keep: personal details, events (with dates), plans, preferences, relationships,
decisions, hobbies, places, numbers and names.

Rules:
- Extract ONLY what the MESSAGE itself says. Never extract or repeat facts that appear only in the earlier messages.
- Each fact is ONE short, self-contained sentence that makes sense without the conversation.
- Record each fact ONCE. Never restate the same idea in different words, and never split one idea into two sentences.
- Return at most 3 facts. Prefer concrete facts (who, what, when, where) over feelings.
- Name the person: use the speaker's name, never "I", "we", "our", "he" or "she".
- Convert relative times ("yesterday", "last year", "next month") into absolute dates or years using the conversation date.
- Do not attach the conversation date to a plan or fact unless the message itself gives a time.
- Do NOT record questions, thanks, greetings, reactions, compliments, agreement or encouragement ("thanks for the compliment", "asks if it is hers", "sounds great", "you'd be great at it").
- Do NOT record vague feelings or opinions with no specific detail ("loves the painting", "thinks the group is helpful").
- Ignore descriptions of shared images unless the message says something specific about who or what is in them.
- If nothing is worth remembering, return [].

Example output: ["Maya adopted a dog named Biscuit in March 2023."]
"""


def _build_prompt(turn: str, metadata: dict) -> str:
    speaker = metadata.get("speaker", "Unknown")
    date = metadata.get("timestamp", "unknown")
    context = metadata.get("context") or []
    earlier = "\n".join(context) if context else "(none)"
    return (
        f"{INSTRUCTIONS}\n"
        f"Conversation date: {date}\n"
        f"Earlier messages (context only, do NOT extract from them):\n{earlier}\n\n"
        f"Speaker: {speaker}\n"
        f"MESSAGE: {turn}\n"
    )


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
    (e.g. "D1:3"), context (list of the previous messages, for reference only).
    The store fills in `embedding`, so it is None here.
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
    facts = facts[:MAX_FACTS_PER_TURN]

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