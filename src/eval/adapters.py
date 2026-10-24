"""Dataset adapters: LoCoMo and LongMemEval → EvalDataset / BenchmarkDataset.

The raw dataset files are NOT part of this repository (see ``data/README.md``).
The schemas below are taken from the official dataset documentation — they are
NOT invented:

LoCoMo — https://github.com/snap-research/locomo (``data/locomo10.json``)
    Top level: JSON array of samples::

        {
          "sample_id": "sample_1",
          "conversation": {
            "speaker_a": "...", "speaker_b": "...",
            "session_1_date_time": "...",
            "session_1": [
              {"speaker": "...", "dia_id": "D1:1", "text": "..."}
            ],
            "session_2": [...], "session_2_date_time": "..."
          },
          "qa": [
            {"question": "...", "answer": "...",
             "evidence": ["D1:3", ...], "category": <label>}
          ]
        }

    * ``qa[].evidence`` lists the ``dia_id`` values of the turns containing
      the answer → mapped to ``BenchmarkCase.relevant_ids``.
    * Adversarial QA entries may omit ``evidence`` (→ ``relevant_ids=[]``,
      handled safely by metrics) and provide ``adversarial_answer`` instead
      of ``answer``.
    * Turn content may be absent for image-only turns; ``blip_caption`` is
      used as fallback text, turns with neither are skipped.

LongMemEval — https://github.com/xiaowu0162/LongMemEval
    (``longmemeval_s.json`` / ``_m`` / ``_oracle`` / HF ``*_cleaned.json``)
    Top level: JSON array of question records::

        {
          "question_id": "...",
          "question_type": "single-session-user" | ... | "multi-session",
          "question": "...", "answer": "...", "question_date": "...",
          "haystack_session_ids": ["s1", ...],
          "haystack_dates": ["...", ...],
          "haystack_sessions": [
            [{"role": "user"|"assistant", "content": "...",
              "has_answer": true}, ...],
            ...
          ],
          "answer_session_ids": ["s1", ...]
        }

    * ``haystack_session_ids`` / ``haystack_dates`` / ``haystack_sessions``
      are parallel arrays, zipped by index.
    * ``answer_session_ids`` (evidence sessions) →
      ``BenchmarkCase.relevant_ids``. An empty list is valid (abstention
      questions, ``question_id`` ending in ``_abs``).

Error policy
------------
* :class:`DatasetFileNotFoundError` — the file/directory does not exist.
* :class:`DatasetFormatError` (a ``ValueError``) — unparseable JSON, root not
  a JSON array, a top-level entry that is not an object, a LoCoMo sample with
  a malformed ``conversation``/``qa`` container, or zero valid cases after
  skipping malformed individual records.
* Individual malformed QA/question records are skipped with a warning; the
  rest of the dataset still loads.

Expected locations (configurable via ``--data-dir`` on the CLI):
    data/locomo/locomo10.json
    data/longmemeval/longmemeval_s.json   (or _m / _oracle / *_cleaned.json)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.eval.dataset import BenchmarkCase, BenchmarkDataset, EvalDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatasetFormatError(ValueError):
    """Raised when a dataset file exists but does not match the schema."""


class DatasetFileNotFoundError(FileNotFoundError):
    """Raised when the expected dataset file/directory cannot be found."""


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

# Preference order when several files are present in the dataset directory.
_PREFERRED_FILES: Dict[str, List[str]] = {
    "locomo": ["locomo10.json"],
    "longmemeval": [
        "longmemeval_s.json",
        "longmemeval_s_cleaned.json",
        "longmemeval_m.json",
        "longmemeval_m_cleaned.json",
        "longmemeval_oracle.json",
    ],
}

ADAPTERS = ("locomo", "longmemeval")


def default_data_dir() -> Path:
    """Repository ``data/`` directory (regardless of CWD)."""
    return Path(__file__).resolve().parent.parent.parent / "data"


def _resolve_data_dir(data_dir: Optional[Union[str, Path]]) -> Path:
    return Path(data_dir) if data_dir is not None else default_data_dir()


def resolve_dataset_file(
    name: str, data_dir: Optional[Union[str, Path]] = None
) -> Path:
    """Resolve the JSON file for dataset *name* under *data_dir*.

    Looks in ``<data_dir>/<name>/`` — preferring the documented filename
    (see ``_PREFERRED_FILES``), then any ``*.json`` (top level, then
    recursive).

    Raises:
        ValueError: If *name* is not a supported adapter.
        DatasetFileNotFoundError: If no directory or JSON file is found.
    """
    if name not in _PREFERRED_FILES:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: {sorted(_PREFERRED_FILES)}"
        )
    base = _resolve_data_dir(data_dir) / name
    if not base.is_dir():
        raise DatasetFileNotFoundError(
            f"Dataset directory not found: {base}. Expected "
            f"<data-dir>/{name}/ containing {_PREFERRED_FILES[name]} - "
            "see data/README.md for download instructions."
        )
    for preferred in _PREFERRED_FILES[name]:
        candidate = base / preferred
        if candidate.is_file():
            return candidate
    candidates = sorted(p for p in base.glob("*.json") if p.is_file())
    if not candidates:
        candidates = sorted(p for p in base.rglob("*.json") if p.is_file())
    if not candidates:
        raise DatasetFileNotFoundError(
            f"No .json file found in {base}. Expected one of "
            f"{_PREFERRED_FILES[name]} - see data/README.md."
        )
    return candidates[0]


def _read_json_array(path: Union[str, Path]) -> list:
    """Read *path* and require a JSON array at the root."""
    p = Path(path)
    if not p.is_file():
        raise DatasetFileNotFoundError(f"Dataset file not found: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, ValueError) as exc:
        raise DatasetFormatError(f"Could not parse JSON in {p}: {exc}") from exc
    if not isinstance(raw, list):
        raise DatasetFormatError(
            f"Expected a JSON array at the root of {p}, got {type(raw).__name__}"
        )
    return raw


# ---------------------------------------------------------------------------
# LoCoMo
# ---------------------------------------------------------------------------

_SESSION_KEY_RE = re.compile(r"^session_(\d+)$")


def _locomo_turns(conversation: dict) -> List[dict]:
    """Flatten a LoCoMo ``conversation`` into ordered turn records.

    Sessions are ordered numerically (session_1, session_2, ...). Each turn
    keeps its ``dia_id`` (the unit referenced by ``qa[].evidence``) and the
    session timestamp. Turns with neither ``text`` nor ``blip_caption`` are
    skipped, as are non-object turns.
    """
    session_keys = sorted(
        (k for k in conversation if _SESSION_KEY_RE.match(k)),
        key=lambda k: int(_SESSION_KEY_RE.match(k).group(1)),  # type: ignore[union-attr]
    )
    turns: List[dict] = []
    for key in session_keys:
        raw_turns = conversation[key]
        if not isinstance(raw_turns, list):
            logger.warning("LoCoMo %s is not a list; skipping session", key)
            continue
        timestamp = conversation.get(f"{key}_date_time")
        for turn in raw_turns:
            if not isinstance(turn, dict):
                continue
            text = turn.get("text") or turn.get("blip_caption")
            if not isinstance(text, str) or not text.strip():
                continue
            speaker = turn.get("speaker")
            turns.append(
                {
                    "id": str(turn.get("dia_id", "")),
                    "text": text.strip(),
                    "speaker": str(speaker) if speaker is not None else "",
                    "session": key,
                    "timestamp": str(timestamp) if timestamp is not None else "",
                }
            )
    return turns


def load_locomo(
    path: Union[str, Path], *, name: str = "locomo"
) -> BenchmarkDataset:
    """Load LoCoMo ``locomo10.json`` into a :class:`BenchmarkDataset`.

    Each ``qa`` entry becomes one :class:`BenchmarkCase`:

    * ``query`` ← ``question``
    * ``reference_answer`` ← ``answer`` (falls back to ``adversarial_answer``)
    * ``relevant_ids`` ← ``evidence`` (dia_ids; ``[]`` when absent)
    * ``case_id`` ← ``"<sample_id>::qa<i>"``, ``conversation_id`` ← ``sample_id``
    * ``metadata`` ← ``dataset``, ``sample_id``, ``category``,
      ``adversarial`` flag and the conversation ``turns`` (raw material for
      future memory preparation).

    Individual malformed QA entries are skipped with a warning; malformed
    sample containers (``conversation``/``qa``) and an unusable file raise
    :class:`DatasetFormatError`.
    """
    raw = _read_json_array(path)
    cases: List[BenchmarkCase] = []
    skipped = 0

    for sample_idx, sample in enumerate(raw):
        if not isinstance(sample, dict):
            raise DatasetFormatError(
                f"LoCoMo sample[{sample_idx}] must be a JSON object, "
                f"got {type(sample).__name__}"
            )
        conversation = sample.get("conversation")
        qa = sample.get("qa")
        if not isinstance(conversation, dict):
            raise DatasetFormatError(
                f"LoCoMo sample[{sample_idx}] missing/invalid 'conversation' "
                "(expected object)"
            )
        if not isinstance(qa, list):
            raise DatasetFormatError(
                f"LoCoMo sample[{sample_idx}] missing/invalid 'qa' "
                "(expected array)"
            )
        sample_id = str(sample.get("sample_id") or f"sample_{sample_idx}")
        turns = _locomo_turns(conversation)

        for qa_idx, entry in enumerate(qa):
            if not isinstance(entry, dict):
                skipped += 1
                logger.warning(
                    "LoCoMo %s qa[%d] is not an object; skipping",
                    sample_id,
                    qa_idx,
                )
                continue
            question = entry.get("question")
            if not isinstance(question, str) or not question.strip():
                skipped += 1
                logger.warning(
                    "LoCoMo %s qa[%d] missing 'question'; skipping",
                    sample_id,
                    qa_idx,
                )
                continue
            answer = entry.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                answer = entry.get("adversarial_answer")
            if not isinstance(answer, str) or not answer.strip():
                skipped += 1
                logger.warning(
                    "LoCoMo %s qa[%d] missing 'answer'/'adversarial_answer'; "
                    "skipping",
                    sample_id,
                    qa_idx,
                )
                continue
            evidence = entry.get("evidence", [])
            if not isinstance(evidence, list):
                skipped += 1
                logger.warning(
                    "LoCoMo %s qa[%d] 'evidence' is not a list; skipping",
                    sample_id,
                    qa_idx,
                )
                continue
            relevant_ids = [
                str(e) for e in evidence if e is not None and str(e).strip()
            ]
            cases.append(
                BenchmarkCase(
                    query=question.strip(),
                    reference_answer=answer.strip(),
                    relevant_ids=relevant_ids,
                    case_id=f"{sample_id}::qa{qa_idx}",
                    metadata={
                        "dataset": "locomo",
                        "sample_id": sample_id,
                        "category": entry.get("category"),
                        "adversarial": "evidence" not in entry,
                        "turns": turns,
                    },
                    conversation_id=sample_id,
                )
            )

    if not cases:
        raise DatasetFormatError(
            f"No valid LoCoMo QA cases found in {path} "
            f"({skipped} record(s) skipped as malformed)"
        )
    if skipped:
        logger.warning("LoCoMo %s: skipped %d malformed QA record(s)", path, skipped)
    return BenchmarkDataset(cases=cases, name=name)


# ---------------------------------------------------------------------------
# LongMemEval
# ---------------------------------------------------------------------------


def _longmemeval_sessions(row: dict) -> List[dict]:
    """Zip ``haystack_session_ids`` / ``haystack_dates`` / ``haystack_sessions``.

    Parallel arrays are zipped by index; missing ids/dates fall back to
    ``session_<i>`` / ``None``. Malformed sessions or turns are skipped with
    a warning. Turn-level ``has_answer`` labels are preserved for future
    turn-level recall evaluation.
    """
    haystack = row.get("haystack_sessions")
    ids = row.get("haystack_session_ids")
    dates = row.get("haystack_dates")
    sessions: List[dict] = []
    for idx, raw_turns in enumerate(haystack):
        if not isinstance(raw_turns, list):
            logger.warning(
                "LongMemEval %s session[%d] is not a list; skipping",
                row.get("question_id"),
                idx,
            )
            continue
        sid = (
            str(ids[idx])
            if isinstance(ids, list) and idx < len(ids)
            else f"session_{idx}"
        )
        date = (
            str(dates[idx])
            if isinstance(dates, list) and idx < len(dates)
            else None
        )
        turns: List[dict] = []
        for turn in raw_turns:
            if not isinstance(turn, dict):
                continue
            content = turn.get("content")
            if not isinstance(content, str):
                continue
            role = turn.get("role")
            record: dict = {
                "role": str(role) if role is not None else "",
                "content": content,
            }
            if turn.get("has_answer") is True:
                record["has_answer"] = True
            turns.append(record)
        if not turns:
            logger.warning(
                "LongMemEval %s session '%s' has no valid turns; skipping",
                row.get("question_id"),
                sid,
            )
            continue
        sessions.append({"session_id": sid, "date": date, "turns": turns})
    return sessions


def load_longmemeval(
    path: Union[str, Path], *, name: str = "longmemeval"
) -> BenchmarkDataset:
    """Load a LongMemEval release file into a :class:`BenchmarkDataset`.

    Each question record becomes one :class:`BenchmarkCase`:

    * ``query`` ← ``question``, ``reference_answer`` ← ``answer``
    * ``relevant_ids`` ← ``answer_session_ids`` (evidence sessions;
      ``[]`` only when the key is present and empty, e.g. abstention)
    * ``case_id`` ← ``question_id``
    * ``metadata`` ← ``dataset``, ``question_type``, ``question_date`` and
      the zipped ``sessions`` (raw material for future memory preparation).

    Records missing ``question`` / ``answer`` / ``haystack_sessions`` /
    ``answer_session_ids`` are skipped with a warning; an unusable file
    raises :class:`DatasetFormatError`.
    """
    raw = _read_json_array(path)
    cases: List[BenchmarkCase] = []
    skipped = 0

    for idx, row in enumerate(raw):
        if not isinstance(row, dict):
            raise DatasetFormatError(
                f"LongMemEval record[{idx}] must be a JSON object, "
                f"got {type(row).__name__}"
            )

        def _skip(reason: str) -> None:
            nonlocal skipped
            skipped += 1
            logger.warning(
                "LongMemEval record[%d] (%s): %s; skipping",
                idx,
                row.get("question_id", "?"),
                reason,
            )

        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            _skip("missing 'question'")
            continue
        answer = row.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            _skip("missing 'answer'")
            continue
        if not isinstance(row.get("haystack_sessions"), list):
            _skip("missing/invalid 'haystack_sessions'")
            continue
        answer_ids = row.get("answer_session_ids")
        if not isinstance(answer_ids, list):
            _skip("missing/invalid 'answer_session_ids'")
            continue
        sessions = _longmemeval_sessions(row)
        if not sessions:
            _skip("no usable haystack sessions")
            continue

        question_id = str(row.get("question_id") or f"longmemeval_{idx}")
        cases.append(
            BenchmarkCase(
                query=question.strip(),
                reference_answer=answer.strip(),
                relevant_ids=[str(a) for a in answer_ids if a is not None],
                case_id=question_id,
                metadata={
                    "dataset": "longmemeval",
                    "question_type": row.get("question_type"),
                    "question_date": row.get("question_date"),
                    "sessions": sessions,
                },
                conversation_id=None,
            )
        )

    if not cases:
        raise DatasetFormatError(
            f"No valid LongMemEval cases found in {path} "
            f"({skipped} record(s) skipped as malformed)"
        )
    if skipped:
        logger.warning(
            "LongMemEval %s: skipped %d malformed record(s)", path, skipped
        )
    return BenchmarkDataset(cases=cases, name=name)


# ---------------------------------------------------------------------------
# Dispatch / conversion
# ---------------------------------------------------------------------------

_LOADERS = {"locomo": load_locomo, "longmemeval": load_longmemeval}


def load_dataset(
    name: str, data_dir: Optional[Union[str, Path]] = None
) -> BenchmarkDataset:
    """Load dataset *name* from *data_dir* (default: repo ``data/``).

    Raises:
        ValueError: Unknown dataset name.
        DatasetFileNotFoundError: Files not present locally.
        DatasetFormatError: Present but malformed.
    """
    if name not in _LOADERS:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: {sorted(_LOADERS)}"
        )
    path = resolve_dataset_file(name, data_dir)
    return _LOADERS[name](path)


def to_eval_dataset(benchmark_dataset: BenchmarkDataset) -> EvalDataset:
    """Convert a QA-level :class:`BenchmarkDataset` to retrieval-level cases.

    Uses the existing ``BenchmarkCase.to_retrieval_case()`` contract (drops
    ``reference_answer``; keeps ``query`` / ``relevant_ids`` / metadata).
    """
    return EvalDataset(
        cases=[case.to_retrieval_case() for case in benchmark_dataset],
        name=benchmark_dataset.name,
    )
