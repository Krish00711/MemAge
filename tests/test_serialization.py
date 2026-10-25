"""Tests for src.eval.serialization — JSON/CSV export of benchmark results.

Covers: JSON structure/round-trip, CSV headers and rows, None metric
handling, comparison/ablation serialization, deterministic output, CLI
export, invalid paths/arguments, and regression of existing CLI modes.
No new metrics; no metric recalculation.
"""

import csv
import io
import json
from dataclasses import asdict

import pytest

from src.eval.ablation import budget_configs, k_configs, run_ablation, stage_configs
from src.eval.benchmark import BenchmarkReport, BenchmarkRunner
from src.eval.comparison import compare_strategies
from src.eval.dataset import BenchmarkCase, BenchmarkDataset
from src.eval.run_benchmark import (
    _synthetic_answer,
    build_synthetic_ablation_spec,
    build_synthetic_strategies,
    main,
)
from src.eval.serialization import (
    CSV_HEADER,
    to_csv,
    to_json,
    write_csv,
    write_json,
)


def _dataset():
    cases = [
        BenchmarkCase(
            query="What is the user's preference?",
            reference_answer="User prefers dark mode",
            relevant_ids=["mem_001"],
            case_id="syn_001",
        ),
        BenchmarkCase(
            query="Where does the user live?",
            reference_answer="User lives in Berlin",
            relevant_ids=["mem_002"],
            case_id="syn_002",
        ),
    ]
    return BenchmarkDataset(cases=cases, name="synthetic_demo")


def _comparison(names=("none", "full_history", "memoryos", "memory_r1", "hmem"), k=5):
    return compare_strategies(
        _dataset(), build_synthetic_strategies(list(names), k=k), _synthetic_answer, k=k
    )


def _ablation(configs=None, ablation="k", k=5):
    return run_ablation(
        _dataset(),
        configs if configs is not None else k_configs([1, 2]),
        lambda cfg: build_synthetic_ablation_spec(cfg),
        _synthetic_answer,
        k=k,
        ablation=ablation,
    )


def _report():
    runner = BenchmarkRunner(
        retrieval_fn=build_synthetic_strategies(["none"], k=5)[0].retrieval_fn,
        answer_fn=_synthetic_answer,
        k=5,
    )
    return runner.run_dataset(_dataset())


# ---------------------------------------------------------------------------
# JSON structure / round-trip
# ---------------------------------------------------------------------------


def test_report_json_structure():
    report = _report()
    data = json.loads(to_json(report))
    assert data == asdict(report)
    for key in (
        "num_queries", "mean_f1", "mean_bleu1", "mean_latency_ms",
        "mean_tokens", "mean_recall_at_k", "mean_precision_at_k", "mrr",
        "per_case",
    ):
        assert key in data
    assert data["num_queries"] == 2
    assert isinstance(data["per_case"], list) and len(data["per_case"]) == 2
    assert data["per_case"][0]["case_id"] == "syn_001"


def test_comparison_json_structure():
    result = _comparison(("none", "full_history"))
    data = json.loads(to_json(result))
    assert data == asdict(result)
    assert data["dataset_name"] == "synthetic_demo"
    assert data["k"] == 5
    assert data["num_cases"] == 2
    assert [r["name"] for r in data["results"]] == ["none", "full_history"]
    rep = data["results"][0]["report"]
    assert rep["mean_f1"] == pytest.approx(1.0)
    assert "per_case" in rep


def test_ablation_json_structure():
    result = _ablation(stage_configs(), ablation="retrieval_stage")
    data = json.loads(to_json(result))
    assert data == asdict(result)
    assert data["ablation"] == "retrieval_stage"
    assert [c["label"] for c in data["configs"]] == [
        "stage=coarse_only", "stage=two_stage",
    ]
    assert data["configs"][0]["params"] == {"stage": "coarse_only"}
    assert data["comparison"]["dataset_name"] == "synthetic_demo"
    assert [r["name"] for r in data["comparison"]["results"]] == [
        "stage=coarse_only", "stage=two_stage",
    ]


def test_json_round_trip_all_types():
    for result in (_report(), _comparison(), _ablation()):
        # round trip: parse back and compare against the canonical asdict
        parsed = json.loads(to_json(result))
        assert parsed == asdict(result)
        # and re-serializing the parsed structure is byte-identical
        assert json.dumps(parsed, indent=2, ensure_ascii=True) == to_json(result)


# ---------------------------------------------------------------------------
# None metric handling (no fake zeros)
# ---------------------------------------------------------------------------


def test_none_metrics_are_null_in_json():
    result = _comparison(("full_history",))
    data = json.loads(to_json(result))
    rep = data["results"][0]["report"]
    assert rep["mean_recall_at_k"] is None
    assert rep["mean_precision_at_k"] is None
    assert rep["mrr"] is None
    assert "0.0" not in (rep["mean_recall_at_k"], rep["mrr"])


def test_none_metrics_are_empty_cells_in_csv():
    result = _comparison(("full_history", "none"))
    rows = list(csv.reader(io.StringIO(to_csv(result))))
    header, data_rows = rows[0], rows[1:]
    assert header == CSV_HEADER
    by_name = {r[0]: r for r in data_rows}
    fh = by_name["full_history"]
    none_row = by_name["none"]
    idx_recall = CSV_HEADER.index("mean_recall_at_k")
    idx_mrr = CSV_HEADER.index("mrr")
    idx_prec = CSV_HEADER.index("mean_precision_at_k")
    # full_history: empty cells (not "0.0" / "None")
    for i in (idx_recall, idx_prec, idx_mrr):
        assert fh[i] == ""
        assert fh[i] != "0.0" and fh[i] != "None"
    # ranking strategy: real values
    assert float(none_row[idx_recall]) == pytest.approx(1.0)
    assert float(none_row[idx_mrr]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CSV headers and rows
# ---------------------------------------------------------------------------


def test_csv_header_exact():
    for result in (_report(), _comparison(), _ablation()):
        header = to_csv(result).splitlines()[0]
        assert header == ",".join(CSV_HEADER)


def test_csv_comparison_rows():
    result = _comparison(("none", "memoryos", "hmem"))
    rows = list(csv.reader(io.StringIO(to_csv(result))))
    header, data_rows = rows[0], rows[1:]
    assert len(data_rows) == 3  # one row per strategy
    names = [r[0] for r in data_rows]
    assert names == ["none", "memoryos", "hmem"]
    assert all(r[1] == "" for r in data_rows)  # kind empty for comparison
    for row in data_rows:
        assert len(row) == len(header) == len(CSV_HEADER)
        assert float(row[CSV_HEADER.index("mean_f1")]) == pytest.approx(1.0)
        assert float(row[CSV_HEADER.index("mean_recall_at_k")]) == pytest.approx(1.0)
        assert int(row[CSV_HEADER.index("num_queries")]) == 2
        assert float(row[CSV_HEADER.index("mean_tokens")]) > 0


def test_csv_ablation_rows():
    result = _ablation(k_configs([1, 2]), ablation="k")
    rows = list(csv.reader(io.StringIO(to_csv(result))))
    header, data_rows = rows[0], rows[1:]
    assert len(data_rows) == 2
    assert [r[0] for r in data_rows] == ["k=1", "k=2"]      # config labels
    assert [r[1] for r in data_rows] == ["k", "k"]          # config kind
    f1s = [float(r[CSV_HEADER.index("mean_f1")]) for r in data_rows]
    assert f1s[0] == pytest.approx(0.5) and f1s[1] == pytest.approx(1.0)


def test_csv_bare_report_single_row():
    rows = list(csv.reader(io.StringIO(to_csv(_report()))))
    assert len(rows) == 2  # header + 1 row
    assert rows[1][0] == "" and rows[1][1] == ""
    assert float(rows[1][CSV_HEADER.index("mean_f1")]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_output_same_object():
    for result in (_report(), _comparison(), _ablation()):
        assert to_json(result) == to_json(result)
        assert to_csv(result) == to_csv(result)


def test_deterministic_output_across_runs_excluding_latency():
    # two independent comparison runs -> identical JSON once wall-clock
    # latency (nondeterministic input) is normalized away
    def normalize(res):
        data = json.loads(to_json(res))
        data["results"] = data.get("results", data.get("comparison", {}).get("results", []))
        for r in data["results"]:
            r["report"]["mean_latency_ms"] = None
            for pc in r["report"]["per_case"]:
                pc["latency_ms"] = None
        return json.dumps(data, indent=2)

    a = _comparison()
    b = _comparison()
    assert normalize(a) == normalize(b)
    # CSV identical too after removing the latency column
    def strip_latency(res):
        rows = list(csv.reader(io.StringIO(to_csv(res))))
        i = CSV_HEADER.index("mean_latency_ms")
        return [[c for j, c in enumerate(row) if j != i] for row in rows]

    assert strip_latency(a) == strip_latency(b)


# ---------------------------------------------------------------------------
# write_json / write_csv (programmatic)
# ---------------------------------------------------------------------------


def test_write_json_and_csv_round_trip(tmp_path):
    result = _comparison(("none", "full_history"))
    jp = write_json(result, tmp_path / "out.json")
    cp = write_csv(result, tmp_path / "out.csv")
    assert jp.exists() and cp.exists()
    assert json.loads(jp.read_text(encoding="utf-8")) == asdict(result)
    parsed = list(csv.reader(io.StringIO(cp.read_text(encoding="utf-8"))))
    assert parsed[0] == CSV_HEADER
    assert [r[0] for r in parsed[1:]] == ["none", "full_history"]


def test_write_accepts_str_path(tmp_path):
    result = _ablation()
    p = write_json(result, str(tmp_path / "ablation.json"))
    assert isinstance(p, type(tmp_path))
    assert json.loads(p.read_text(encoding="utf-8"))["ablation"] == "k"


# ---------------------------------------------------------------------------
# Invalid arguments / paths
# ---------------------------------------------------------------------------


def test_unsupported_result_type_rejected():
    for bad in ({"a": 1}, "report", 42, None, stage_configs()):
        with pytest.raises(TypeError):
            to_json(bad)
        with pytest.raises(TypeError):
            to_csv(bad)


def test_invalid_output_paths(tmp_path):
    result = _comparison()
    with pytest.raises(OSError):  # parent directory does not exist (not auto-created)
        write_json(result, tmp_path / "missing_dir" / "out.json")
    with pytest.raises(OSError):  # target is a directory
        write_csv(result, tmp_path)
    with pytest.raises((OSError, ValueError)):  # embedded NUL
        write_json(result, "bad\x00path.json")


# ---------------------------------------------------------------------------
# CLI export
# ---------------------------------------------------------------------------


def test_cli_compare_export_json(tmp_path, capsys):
    out_json = tmp_path / "compare.json"
    code = main(
        ["--dataset", "synthetic", "--compare", "none,full_history",
         "--export-json", str(out_json)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Comparison (synthetic_demo, k=5)" in out  # table still printed
    assert f"Exported JSON: {out_json}" in out
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["dataset_name"] == "synthetic_demo"
    assert [r["name"] for r in data["results"]] == ["none", "full_history"]


def test_cli_compare_export_csv(tmp_path, capsys):
    out_csv = tmp_path / "compare.csv"
    code = main(
        ["--dataset", "synthetic", "--compare", "none,memoryos,hmem",
         "--export-csv", str(out_csv)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert f"Exported CSV: {out_csv}" in out
    rows = list(csv.reader(io.StringIO(out_csv.read_text(encoding="utf-8"))))
    assert rows[0] == CSV_HEADER
    assert [r[0] for r in rows[1:]] == ["none", "memoryos", "hmem"]


def test_cli_ablate_export_both_formats(tmp_path, capsys):
    out_json = tmp_path / "ablation.json"
    out_csv = tmp_path / "ablation.csv"
    code = main(
        ["--dataset", "synthetic", "--ablate", "stage",
         "--export-json", str(out_json), "--export-csv", str(out_csv)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Exported JSON" in out and "Exported CSV" in out
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["ablation"] == "retrieval_stage"
    rows = list(csv.reader(io.StringIO(out_csv.read_text(encoding="utf-8"))))
    assert [r[0] for r in rows[1:]] == ["stage=coarse_only", "stage=two_stage"]
    assert [r[1] for r in rows[1:]] == ["retrieval_stage", "retrieval_stage"]


def test_cli_export_requires_structured_mode(capsys):
    # single-strategy mode: export flags rejected
    assert main(["--dataset", "synthetic", "--export-json", "x.json"]) == 1
    assert "require --compare or --ablate" in capsys.readouterr().err
    assert main(["--dataset", "synthetic", "--export-csv", "x.csv"]) == 1


def test_cli_export_failure_returns_2(tmp_path, capsys):
    bad = tmp_path / "no_such_dir" / "out.json"
    code = main(
        ["--dataset", "synthetic", "--compare", "none", "--export-json", str(bad)]
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "could not export" in err
    assert not bad.exists()


# ---------------------------------------------------------------------------
# Regression: existing CLI modes unchanged when flags omitted
# ---------------------------------------------------------------------------


def test_regression_modes_without_export_flags(capsys, tmp_path, monkeypatch):
    # run from a temp cwd so any accidental file writes would be visible/absent
    import os

    monkeypatch.chdir(tmp_path)
    # single mode unchanged
    assert main(["--dataset", "synthetic"]) == 0
    out0 = capsys.readouterr().out
    assert "Synthetic benchmark (synthetic_demo" in out0
    assert "Exported" not in out0
    # compare mode unchanged
    assert main(["--dataset", "synthetic", "--compare", "none,hmem"]) == 0
    out1 = capsys.readouterr().out
    assert "Comparison (synthetic_demo, k=5): 2 cases, 2 strategies" in out1
    assert "Exported" not in out1
    # ablate mode unchanged
    assert main(["--dataset", "synthetic", "--ablate", "k:1,2"]) == 0
    out2 = capsys.readouterr().out
    assert "Ablation k (synthetic_demo, metric_k=5)" in out2
    assert "Exported" not in out2
    # no files were created
    assert list(tmp_path.iterdir()) == []


def test_regression_single_baseline_and_errors(capsys):
    assert main(["--dataset", "synthetic", "--baseline", "memoryos"]) == 0
    assert "Synthetic benchmark (synthetic_demo, baseline=memoryos)" in (
        capsys.readouterr().out
    )
    # export + both modes still mutually exclusive
    assert main(["--dataset", "synthetic", "--compare", "none", "--ablate", "stage"]) == 1
    assert "only one" in capsys.readouterr().err
