"""Tests for src.eval.run_benchmark CLI boundary."""

import sys

import pytest

from src.eval.run_benchmark import (
    SUPPORTED_BASELINES,
    SUPPORTED_DATASETS,
    _dataset_available,
    _get_unavailable_message,
    build_parser,
    main,
    parse_args,
)


def test_cli_argument_parsing():
    args = parse_args(["--dataset", "synthetic", "--baseline", "none"])
    assert args.dataset == "synthetic"
    assert args.baseline == "none"

    args2 = parse_args(["--dataset", "locomo", "--baseline", "full_history"])
    assert args2.dataset == "locomo"
    assert args2.baseline == "full_history"

    args3 = parse_args(["--dataset", "synthetic", "--baseline", "none", "--k", "10"])
    assert args3.k == 10


def test_cli_defaults():
    args = parse_args([])
    assert args.dataset == "synthetic"
    assert args.baseline == "none"


def test_unavailable_dataset_behavior(tmp_path):
    # locomo data absent from an empty data dir -> unavailable (hermetic:
    # the repo's data/ folder may contain the real datasets)
    msg = _get_unavailable_message("locomo", "none", str(tmp_path))
    assert msg is not None
    assert "not available locally" in msg or "requires" in msg

    # Unknown dataset
    msg2 = _get_unavailable_message("unknown_xyz", "none")
    assert "Unknown dataset" in msg2

    # Unknown baseline
    msg3 = _get_unavailable_message("synthetic", "unknown_baseline")
    assert "Unknown baseline" in msg3


def test_no_baselines_are_stubs_anymore():
    # every checklist baseline is implemented now (all approximations)
    from src.eval.run_benchmark import NOT_YET_IMPLEMENTED_BASELINES

    for name in SUPPORTED_BASELINES:
        assert _get_unavailable_message("synthetic", name) is None
    assert NOT_YET_IMPLEMENTED_BASELINES == set()


def test_dataset_available_synthetic(tmp_path):
    assert _dataset_available("synthetic") is True
    # locomo not present in an empty data dir (hermetic: the repo's data/
    # folder may contain the real datasets)
    assert _dataset_available("locomo", str(tmp_path)) is False


def test_main_synthetic_success(capsys):
    code = main(["--dataset", "synthetic", "--baseline", "none"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Synthetic benchmark" in out
    assert "Mean F1" in out


def test_main_unavailable_dataset_returns_error(capsys, tmp_path):
    code = main(["--dataset", "locomo", "--baseline", "none",
                 "--data-dir", str(tmp_path)])
    assert code == 2
    err = capsys.readouterr().err
    assert "not available locally" in err or "Error" in err


def test_main_hmem_baseline_runs(capsys):
    # hmem is implemented now: it must run, not return the stub error
    code = main(["--dataset", "synthetic", "--baseline", "hmem"])
    assert code == 0
    assert "baseline=hmem" in capsys.readouterr().out


def test_cli_supported_lists():
    assert "synthetic" in SUPPORTED_DATASETS
    assert "locomo" in SUPPORTED_DATASETS
    assert "none" in SUPPORTED_BASELINES
    assert "full_history" in SUPPORTED_BASELINES
    assert "memoryos" in SUPPORTED_BASELINES


# ---------------------------------------------------------------------------
# --token-budget propagation
# ---------------------------------------------------------------------------


def test_token_budget_argument_parsed():
    from src.eval.run_benchmark import DEFAULT_TOKEN_BUDGET

    args = parse_args([])
    assert args.token_budget == DEFAULT_TOKEN_BUDGET

    args2 = parse_args(["--token-budget", "123"])
    assert args2.token_budget == 123

    # non-positive budget is a usage error (argparse exits with code 2)
    with pytest.raises(SystemExit):
        parse_args(["--token-budget", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--token-budget", "not-a-number"])


def test_token_budget_propagated_to_retrieval_pipeline(monkeypatch):
    import src.eval.run_benchmark as rb

    original = rb.make_retrieval_fn
    seen = []

    def spy(retrieval_context, token_budget):
        seen.append(token_budget)
        return original(retrieval_context, token_budget)

    monkeypatch.setattr(rb, "make_retrieval_fn", spy)

    code = rb.main(["--dataset", "synthetic", "--token-budget", "77"])
    assert code == 0
    assert seen == [77]  # exactly one pipeline build, with the CLI value


def test_token_budget_affects_synthetic_output(capsys):
    import src.eval.run_benchmark as rb

    # Default budget: both synthetic memories selected
    assert rb.main(["--dataset", "synthetic"]) == 0
    default_out = capsys.readouterr().out
    default_tokens = float(
        [l for l in default_out.splitlines() if "Mean tokens" in l][0].split(":")[1]
    )

    # Tight budget: fewer memories survive the pipeline's budget selection
    assert rb.main(["--dataset", "synthetic", "--token-budget", "12"]) == 0
    tight_out = capsys.readouterr().out
    tight_tokens = float(
        [l for l in tight_out.splitlines() if "Mean tokens" in l][0].split(":")[1]
    )

    assert tight_tokens <= 12
    assert tight_tokens < default_tokens


# ---------------------------------------------------------------------------
# --data-dir / real dataset handling
# ---------------------------------------------------------------------------


def _write_locomo(tmp_path, n_cases=1):
    import json

    fixture = [
        {
            "sample_id": "sample_1",
            "conversation": {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "hello"}],
                "session_1_date_time": "t",
            },
            "qa": [
                {
                    "question": f"Question {i}?",
                    "answer": f"Answer {i}",
                    "evidence": ["D1:1"],
                }
                for i in range(n_cases)
            ],
        }
    ]
    d = tmp_path / "locomo"
    d.mkdir(parents=True, exist_ok=True)
    (d / "locomo10.json").write_text(json.dumps(fixture), encoding="utf-8")


def test_data_dir_option_respected(tmp_path, capsys, monkeypatch):
    # without files: unavailable at an empty data dir (hermetic: the repo's
    # data/ folder may contain the real datasets)
    assert _dataset_available("locomo", str(tmp_path)) is False

    _write_locomo(tmp_path, n_cases=3)
    assert _dataset_available("locomo", str(tmp_path)) is True

    # The real `none` strategy executes on real data: inject a deterministic
    # embedder + fake LLM so the test stays offline (the store, pipeline and
    # call_llm wiring under test are the real implementations).
    from tests.test_real_none_baseline import (
        _FakeLLM,
        _HashEmbedder,
        _patch_llm,
        _patch_store_embedder,
    )

    _patch_llm(monkeypatch, _FakeLLM(answer="Answer 1"))
    _patch_store_embedder(monkeypatch, _HashEmbedder())

    code = main(
        ["--dataset", "locomo", "--baseline", "none", "--data-dir", str(tmp_path)]
    )
    captured = capsys.readouterr()
    # dataset loaded and parsed through the CLI, then benchmarked end-to-end
    assert "Loaded locomo: 3 cases" in captured.out
    assert code == 0
    assert "Mean F1" in captured.out
    assert "MRR" in captured.out  # retrieval metrics are non-degenerate


def test_real_dataset_without_files_still_reports_missing(capsys, tmp_path):
    code = main(["--dataset", "longmemeval", "--baseline", "none",
                 "--data-dir", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 2
    assert "not available locally" in captured.err


def test_dataset_available_uses_data_dir(tmp_path):
    assert _dataset_available("longmemeval", str(tmp_path)) is False
    d = tmp_path / "longmemeval"
    d.mkdir()
    (d / "longmemeval_s.json").write_text("[]", encoding="utf-8")
    assert _dataset_available("longmemeval", str(tmp_path)) is True


# ---------------------------------------------------------------------------
# --baseline full_history (synthetic)
# ---------------------------------------------------------------------------


def _mean_tokens(out):
    lines = [l for l in out.splitlines() if "Mean tokens" in l]
    assert lines, "expected a Mean tokens line"
    return float(lines[0].split(":")[1])


def test_cli_full_history_baseline_synthetic(capsys):
    code = main(["--dataset", "synthetic", "--baseline", "full_history"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Synthetic benchmark (synthetic_demo, baseline=full_history)" in out
    assert "Mean F1" in out
    # no ranking -> retrieval metrics are not reported for this baseline
    assert "Mean Recall" not in out
    assert "MRR" not in out
    # token usage still measured
    assert _mean_tokens(out) > 0


def test_cli_full_history_ignores_token_budget(capsys):
    # Tight budget that would exclude everything from the retrieval pipeline
    code = main(
        ["--dataset", "synthetic", "--baseline", "full_history", "--token-budget", "5"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert _mean_tokens(out) > 5  # full history returned regardless of budget


def test_cli_default_baseline_unchanged(capsys):
    # baseline none behavior is preserved (pipeline + budget still applies)
    code = main(["--dataset", "synthetic", "--token-budget", "12"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Synthetic benchmark (synthetic_demo" in out
    assert _mean_tokens(out) <= 12
