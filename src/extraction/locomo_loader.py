"""LoCoMo loader: dialogue turns -> extracted CandidateMemoryItems -> store.

Owner: Aryan. Raw data lives in data/ (gitignored, never committed).
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from src.extraction.extractor import extract_candidates


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
        for turn in conv[key]:
            text = turn.get("text", "")
            if turn.get("blip_caption"):          # some turns share an image
                text += f" [shared an image: {turn['blip_caption']}]"
            yield text, {
                "speaker": turn.get("speaker", "Unknown"),
                "timestamp": timestamp,
                "source": turn["dia_id"],         # e.g. "D1:3" = qa 'evidence' id
            }


def extract_conversation(record: dict, llm=None,
                         cache_dir: str = "data/cache",
                         limit: int | None = None) -> list[dict]:
    """Run the extractor over a whole conversation, caching results on disk."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = Path(cache_dir) / f"extraction_{record['sample_id']}.json"
    cache: dict = {}
    if cache_file.exists():
        cache = json.loads(cache_file.read_text(encoding="utf-8"))

    items: list[dict] = []
    for n, (text, meta) in enumerate(iter_turns(record)):
        if limit is not None and n >= limit:
            break
        key = meta["source"]
        if key not in cache:
            cache[key] = extract_candidates(text, meta, llm=llm)
            if n % 20 == 0:                        # save progress as we go
                cache_file.write_text(json.dumps(cache), encoding="utf-8")
        items.extend(cache[key])
    cache_file.write_text(json.dumps(cache), encoding="utf-8")
    return items


def ingest_conversation(record: dict, store, llm=None,
                        cache_dir: str = "data/cache",
                        limit: int | None = None) -> int:
    """Extract a conversation and add every memory to `store`. Returns count."""
    items = extract_conversation(record, llm=llm, cache_dir=cache_dir, limit=limit)
    for item in items:
        store.add(item)
    return len(items)