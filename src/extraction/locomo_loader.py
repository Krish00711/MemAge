"""LoCoMo loader: dialogue turns -> extracted CandidateMemoryItems -> store.

Owner: Aryan. Raw data lives in data/ (gitignored, never committed).
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from src.extraction.extractor import PROMPT_VERSION, extract_candidates

try:
    from src.llm.client import LLMRateLimitError
except ImportError:  # older client.py without this class
    class LLMRateLimitError(RuntimeError):
        pass


def load_locomo(path: str = "data/locomo10.json") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_locomo_date(raw: str | None) -> str | None:
    """'1:56 pm on 8 May, 2023' -> '2023-05-08T13:56:00'. Falls back to raw text."""
    if not raw:
        return None
    try:
        return datetime.strptime(raw.strip(), "%I:%M %p on %d %B, %Y").isoformat()
    except ValueError:
        return raw


def iter_turns(record: dict):
    """Yield (turn_text, metadata) for every turn, in conversation order."""
    conv = record["conversation"]
    sessions = sorted(
        (int(m.group(1)), key)
        for key in conv
        if (m := re.fullmatch(r"session_(\d+)", key))
    )
    for num, key in sessions:
        timestamp = parse_locomo_date(conv.get(f"{key}_date_time"))
        history: list[str] = []                  # restarts every session
        for turn in conv[key]:
            text = turn.get("text", "")
            if turn.get("blip_caption"):          # some turns share an image
                text += f" [shared an image: {turn['blip_caption']}]"
            speaker = turn.get("speaker", "Unknown")
            yield text, {
                "speaker": speaker,
                "timestamp": timestamp,
                "source": turn["dia_id"],         # e.g. "D1:3" = qa 'evidence' id
                "context": history[-2:],          # previous two messages
            }
            history.append(f"{speaker}: {text}")


def extract_conversation(record: dict, llm=None,
                         cache_dir: str = "data/cache",
                         limit: int | None = None) -> list[dict]:
    """Run the extractor over a whole conversation, caching results on disk.

    If the LLM provider's rate limit cannot be recovered, progress is saved
    and the error is raised; run the same command again later to resume.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = Path(cache_dir) / f"extraction_{record['sample_id']}_{PROMPT_VERSION}.json"
    cache: dict = {}
    if cache_file.exists():
        cache = json.loads(cache_file.read_text(encoding="utf-8"))

    def save() -> None:
        cache_file.write_text(json.dumps(cache), encoding="utf-8")

    items: list[dict] = []
    for n, (text, meta) in enumerate(iter_turns(record)):
        if limit is not None and n >= limit:
            break
        key = meta["source"]
        if key not in cache:
            try:
                cache[key] = extract_candidates(text, meta, llm=llm)
            except LLMRateLimitError:
                save()
                print(f"Rate limit reached at turn {key}; progress saved "
                      f"({len(cache)} turns cached). Run again later to resume.")
                raise
            if n % 20 == 0:                        # save progress as we go
                save()
        items.extend(cache[key])
    save()
    return items


def ingest_conversation(record: dict, store, llm=None,
                        cache_dir: str = "data/cache",
                        limit: int | None = None) -> int:
    """Extract a conversation and add every memory to `store`. Returns count."""
    items = extract_conversation(record, llm=llm, cache_dir=cache_dir, limit=limit)
    for item in items:
        store.add(item)
    return len(items)