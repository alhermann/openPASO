"""A blind cell must not read another cell's work through host tools."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The harness tests need the LangGraph extras, which live in a
# separate virtualenv (.venv-lg). Without this guard a run from
# the corpus interpreter aborts COLLECTION, so the whole suite
# reports nothing rather than skipping these few modules.
pytest.importorskip("langchain_core")

from langgraph_eval import agent as A  # noqa: E402
import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent))
import backend_probe  # noqa: E402


def _tool(tools, name):
    return next(tool for tool in tools if tool.name == name)


def test_mcp_pins_mounted_backend_runtimes_before_masking_home(tmp_path):
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    paths = {
        "FENICS_PYTHON": home / "miniconda3/envs/fenics/bin/python",
        "DUNE_PYTHON": home / "miniconda3/envs/dune-py313/bin/python",
        "FEBIO_BINARY": home / "FEBio/bin/febio4",
        "DEAL_II_DIR": home / "dealii/build",
        "SPARTA_BINARY": workspace / "sparta/src/spa_serial",
        "SPARTA_ROOT": workspace / "sparta",
        "SPARTA_DATA_DIR": workspace / "sparta/data",
    }
    for key, path in paths.items():
        if key in {"DEAL_II_DIR", "SPARTA_ROOT", "SPARTA_DATA_DIR"}:
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("runtime")

    env = {"FENICS_PYTHON": "/explicit/fenics/python"}
    A._pin_backend_runtime_env(env, home=home, workspace=workspace)

    assert env["FENICS_PYTHON"] == "/explicit/fenics/python"
    for key, path in paths.items():
        if key != "FENICS_PYTHON":
            assert env[key] == str(path)


def test_shell_cannot_read_an_adjacent_cell(tmp_path):
    campaign = tmp_path / "campaign"
    work = campaign / "runs" / "cell_a" / "work"
    sibling = campaign / "runs" / "cell_b" / "work"
    work.mkdir(parents=True)
    sibling.mkdir(parents=True)
    marker = "CROSS_CELL_MARKER_91f226"
    secret = sibling / "solution.txt"
    secret.write_text(marker)

    run_bash = A._bash_tool_for(work)
    runtime_python = (
        str(backend_probe.openpaso_python()))
    pair_params = json.loads((
        ROOT / "benchmarks/coupling_pairs/fourc_kratos_cht/params.json"
    ).read_text())
    assert pair_params["kratos_python"] == runtime_python
    out = run_bash.invoke({
        "command": (
            f"{runtime_python} -c 'import KratosMultiphysics; "
            "print(\"runtime-ok\")'; "
            f"cat {secret} 2>&1 || true"
        )
    })

    assert "runtime-ok" in out
    assert marker not in out


def test_read_file_refuses_an_adjacent_cell(tmp_path):
    campaign = tmp_path / "campaign"
    work = campaign / "runs" / "cell_a" / "work"
    sibling = campaign / "runs" / "cell_b" / "work"
    work.mkdir(parents=True)
    sibling.mkdir(parents=True)
    marker = "CROSS_CELL_MARKER_5c37a8"
    secret = sibling / "solution.txt"
    secret.write_text(marker)

    read_file = _tool(A._read_write_tools_for(work), "read_file")
    out = read_file.invoke({"path": str(secret)})

    assert marker not in out
    assert "outside" in out.lower() and "refused" in out.lower()


def test_dune_baseline_compiler_script_is_relocated(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    script = baseline / "python/dune/generated/buildScript.sh"
    script.parent.mkdir(parents=True)
    script.write_text(
        "DUNE_CXX_COMPILER_LAUNCHER=/baseline/dune-py/compiler_launcher.sh\n"
        "/usr/bin/c++ -c /baseline/dune-py/python/dune/generated/$1.cc\n")
    (baseline / "compiler_launcher.sh").write_text(
        "/baseline/dune-py/dune-compiler_launcher.sh $@\n")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("OPENPASO_DUNE_CACHE_BASELINE", str(baseline))

    scratch = A.sandbox_scratch_for(work)
    try:
        copied = scratch / "dune-cache/dune-py/python/dune/generated/buildScript.sh"
        text = copied.read_text()
        assert "/baseline/dune-py" not in text
        assert "/tmp/dune-cache/dune-py/compiler_launcher.sh" in text
        assert "/tmp/dune-cache/dune-py/python/dune/generated/$1.cc" in text
        nested = (scratch / "dune-cache/dune-py/compiler_launcher.sh").read_text()
        assert "/baseline/dune-py" not in nested
        assert "/tmp/dune-cache/dune-py/dune-compiler_launcher.sh" in nested
    finally:
        A.cleanup_sandbox_scratch(work)


def test_shell_sees_neither_credentials_nor_host_processes(tmp_path,
                                                           monkeypatch):
    marker = "OPENROUTER_SECRET_MARKER_1f86d3"
    monkeypatch.setenv("OPENROUTER_API_KEY", marker)
    sensitive_names = (
        "GIT_ASKPASS",
        "VSCODE_GIT_ASKPASS_EXTRA_ARGS",
        "VSCODE_GIT_ASKPASS_MAIN",
        "VSCODE_GIT_ASKPASS_NODE",
        "SSH_AUTH_SOCK",
        "XAUTHORITY",
    )
    for name in sensitive_names:
        monkeypatch.setenv(name, marker)
    run_bash = A._bash_tool_for(tmp_path)
    out = run_bash.invoke({
        "command": (
            "env; printf 'PID_COUNT='; "
            "find /proc -maxdepth 1 -regex '/proc/[0-9]+' | wc -l"
        )
    })
    assert marker not in out
    assert "OPENROUTER_API_KEY" not in out
    assert all(name not in out for name in sensitive_names)
    count = int(out.split("PID_COUNT=", 1)[1].splitlines()[0])
    assert count < 10, out