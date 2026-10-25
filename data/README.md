Raw datasets (LoCoMo, LongMemEval) and the ChromaDB persistence directory go here.
This folder's contents (except this file) are gitignored — do not commit datasets or the vector DB itself.

The benchmark CLI (`python -m src.eval.run_benchmark --data-dir data`) resolves
dataset files as `<data-dir>/<dataset>/*.json` (adapters in `src/eval/adapters.py`).

## Expected file layout (not shipped with the repo — download manually)

```
data/
├── locomo/
│   └── locomo10.json          # primary benchmark (10 conversations)
├── longmemeval/
│   └── longmemeval_s.json     # or longmemeval_m.json / longmemeval_oracle.json
└── chroma/                    # ChromaDB persistence (gitignored)
```

## Download instructions

- LoCoMo: https://github.com/snap-research/locomo — dataset lives at
  `data/locomo10.json` in that repository (git-lfs), e.g.:

  ```
  git clone https://github.com/snap-research/locomo /tmp/locomo-repo
  # then copy /tmp/locomo-repo/data/locomo10.json -> data/locomo/locomo10.json
  ```

- LongMemEval: https://github.com/xiaowu0162/LongMemEval — files are hosted on
  Hugging Face (`xiaowu0162/longmemeval-cleaned`), e.g.:

  ```
  mkdir -p data/longmemeval
  # from that HF dataset repo:
  #   longmemeval_s.json / longmemeval_s_cleaned.json  (~40 sessions, ~115k tokens)
  #   longmemeval_m.json / longmemeval_m_cleaned.json  (~500 sessions)
  #   longmemeval_oracle.json                          (evidence sessions only)
  ```

No download is performed automatically by the project code (offline-only).
