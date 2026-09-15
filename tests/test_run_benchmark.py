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


def test_unavailable_dataset_behavior():
    # locomo data not present locally -> unavailable
    msg = _get_unavailable_message("locomo", "none")
    assert msg is not None
    assert "not available locally" in msg or "requires" in msg

    # Unknown dataset
    msg2 = _get_unavailable_message("unknown_xyz", "none")
    assert "Unknown dataset" in msg2

    # Unknown baseline
    msg3 = _get_unavailable_message("synthetic", "unknown_baseline")
    assert "Unknown baseline" in msg3


def test_not_yet_implemented_baseline():
    msg = _get_unavailable_message("synthetic", "memoryos")
    assert msg is not None
    assert "not yet implemented" in msg
    assert "Phase 7" in msg

    msg2 = _get_unavailable_message("synthetic", "hmem")
    assert "not yet implemented" in msg2


def test_dataset_available_synthetic():
    assert _dataset_available("synthetic") is True
    # locomo not present (data folder empty)
    assert _dataset_available("locomo") is False


def test_main_synthetic_success(capsys):
    code = main(["--dataset", "synthetic", "--baseline", "none"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Synthetic benchmark" in out
    assert "Mean F1" in out


def test_main_unavailable_dataset_returns_error(capsys):
    code = main(["--dataset", "locomo", "--baseline", "none"])
    assert code == 2
    err = capsys.readouterr().err
    assert "not available locally" in err or "Error" in err


def test_main_not_implemented_baseline(capsys):
    code = main(["--dataset", "synthetic", "--baseline", "memoryos"])
    assert code == 2
    err = capsys.readouterr().err
    assert "not yet implemented" in err


def test_cli_supported_lists():
    assert "synthetic" in SUPPORTED_DATASETS
    assert "locomo" in SUPPORTED_DATASETS
    assert "none" in SUPPORTED_BASELINES
    assert "full_history" in SUPPORTED_BASELINES
    assert "memoryos" in SUPPORTED_BASELINES
