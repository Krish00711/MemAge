"""Runtime console output must be encodable on cp1252 Windows consoles.

Regression tests for the crash class where non-ASCII symbols (e.g. "->")
in print()/sys.exit() literals raised UnicodeEncodeError when stdout or
stderr was redirected through the Windows cp1252 code page.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _assert_cp1252(text: str, label: str) -> None:
    try:
        text.encode("cp1252")
    except UnicodeEncodeError as exc:
        raise AssertionError(f"{label} is not cp1252-encodable: {exc}") from exc


def _is_console_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "print"
    if isinstance(func, ast.Attribute):
        if func.attr == "exit":  # sys.exit("...") prints the message to stderr
            return True
        if func.attr == "write" and isinstance(func.value, ast.Attribute):
            return func.value.attr in ("stdout", "stderr")
    return False


def test_console_literals_are_ascii():
    offenders = []
    for rel_dir in ("src", "scripts"):
        for path in sorted((ROOT / rel_dir).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not _is_console_call(node):
                    continue
                for sub in ast.walk(node):
                    if (
                        isinstance(sub, ast.Constant)
                        and isinstance(sub.value, str)
                        and not sub.value.isascii()
                    ):
                        offenders.append((path.relative_to(ROOT), sub.lineno, sub.value))
    assert not offenders, "non-ASCII literals in console output:\n" + "\n".join(
        f"{p}:{lineno}: {lit!r}" for p, lineno, lit in offenders
    )


def test_run_benchmark_output_dir_cp1252(tmp_path, capsys):
    from src.eval.run_benchmark import main

    code = main(
        [
            "--dataset",
            "synthetic",
            "--baseline",
            "none",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "Saved predictions ->" in captured.out
    assert "Saved results" in captured.out
    _assert_cp1252(captured.out, "run_benchmark stdout")
    _assert_cp1252(captured.err, "run_benchmark stderr")


def test_run_benchmark_missing_dataset_error_cp1252(capsys, tmp_path):
    from src.eval.run_benchmark import main

    # empty data dir: the missing-dataset error path stays testable now
    # that the repo ships real datasets under data/
    code = main(["--dataset", "locomo", "--baseline", "none",
                 "--data-dir", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 2
    _assert_cp1252(captured.out, "run_benchmark stdout (missing dataset)")
    _assert_cp1252(captured.err, "run_benchmark stderr (missing dataset)")


def test_check_config_cp1252(capsys, monkeypatch):
    import sys as _sys

    import src.check_config as check_config

    # Neutralize .env loading so output does not depend on local secrets.
    monkeypatch.setitem(_sys.modules, "dotenv", None)
    code = check_config.check()
    captured = capsys.readouterr()
    assert code in (0, 1)
    _assert_cp1252(captured.out, "check_config stdout")
    _assert_cp1252(captured.err, "check_config stderr")


def test_server_demo_banner_cp1252(capsys, monkeypatch):
    import src.server as server_mod

    class _StubServer:
        def __init__(self, addr, handler):
            self.addr = addr
            self.handler = handler

        def serve_forever(self):
            return None

    monkeypatch.setattr(server_mod, "HTTPServer", _StubServer)
    # Snapshot module globals so they are restored after the test.
    monkeypatch.setattr(server_mod, "_store", server_mod._store)
    monkeypatch.setattr(server_mod, "_agent", server_mod._agent)

    server_mod.main(["--demo", "--port", "0"])
    captured = capsys.readouterr()
    assert "Running in DEMO mode" in captured.out
    assert "MemAge server ->" in captured.out
    _assert_cp1252(captured.out, "server stdout")
    _assert_cp1252(captured.err, "server stderr")
