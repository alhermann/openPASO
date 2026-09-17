"""A relative history path must still produce the history file, in the task directory.

MEASURED 2026-09-17, C3 on the current build. The agent called
`couple_levels(..., history_dir='.')`. couple_levels joined that into the relative
path `residual_level<k>.csv`, and couple() refused it -- "history_path must be
absolute" -- so the driver wrote NO convergence history while its own reply said
`"all_levels_converged": true`. The agent then wrote the residual files itself,
from a different normalisation, and its level-3 file ended at 1.53e-06 against
the prescribed 1e-6. The grader correctly read the evidence as contradicted and
the cell, otherwise complete, scored nothing.

It is not one agent's slip: across the recorded runs 25 couple() calls passed a
relative `history_path` and one couple_levels call a relative `history_dir`. Every
one of them silently left the evidence unwritten.

The absolute-only rule existed for a sound reason -- a relative path resolves
against the SERVER's working directory, which the agent cannot see. The fix keeps
that reason and removes the trap: a relative path is resolved against the task's
working directory (OPENPASO_CELL_WORKDIR), or, when none is set, beside the
participants -- the same default couple_levels already uses. A path that escapes
the task directory is still refused.
"""
import asyncio
import inspect
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TOY = '''import json, os
from pathlib import Path
cfg = json.loads(Path("config.json").read_text()) if Path("config.json").is_file() else {}
cfg.update(json.loads(os.environ.get("OPENPASO_CONFIG_JSON") or "{}"))
imp = json.loads(Path("imports.json").read_text()) if Path("imports.json").is_file() else {}
other = 0.0
for d in imp.values():
    other = float((d.get("values") or [0.0])[0])
n = int(cfg.get("nx", 1))
mine = COEF * other + OFFSET + 0.001 / n
print(f"NDOF = {n}")
Path("exports.json").write_text(json.dumps({"field_name": "u", "n_points": 1, "coordinates": [[0.5, 0.0]],
    "values": [mine], "normal_fluxes": [0.0]}))
'''


class _CollectingMCP:
    def __init__(self):
        self.tools = {}

    def tool(self, *a, **k):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


@pytest.fixture(scope="module")
def tools():
    os.chdir(ROOT)
    from core.registry import load_all_backends
    load_all_backends()
    from tools.consolidated import register_consolidated_tools
    mcp = _CollectingMCP()
    register_consolidated_tools(mcp)
    return mcp.tools


def _call(fn, **kw):
    r = fn(**kw)
    if inspect.iscoroutine(r):
        r = asyncio.run(r)
    return json.loads(str(r))


def _pair(task_dir):
    for name, coef, off in (("A", 0.6, 1.0), ("B", 0.7, 0.2)):
        d = task_dir / f"side_{name}"
        d.mkdir(parents=True)
        (d / "part.py").write_text(TOY.replace("COEF", str(coef)).replace("OFFSET", str(off)))
    return [{"name": "A", "command": [sys.executable, "part.py"],
             "work_dir": str(task_dir / "side_A"), "imports_from": ["B"]},
            {"name": "B", "command": [sys.executable, "part.py"],
             "work_dir": str(task_dir / "side_B"), "imports_from": ["A"]}]


def _elsewhere(tmp_path_factory, monkeypatch):
    # The server's own cwd is NOT the task directory; a relative path must not
    # land here.
    other = tmp_path_factory.mktemp("server_cwd")
    monkeypatch.chdir(other)
    return other


def test_couple_writes_a_relative_history_path_into_the_task_directory(
        tmp_path, tools, monkeypatch, tmp_path_factory):
    other = _elsewhere(tmp_path_factory, monkeypatch)
    monkeypatch.setenv("OPENPASO_CELL_WORKDIR", str(tmp_path))
    out = _call(tools["couple"], participants=json.dumps(_pair(tmp_path)), max_iter=80,
                tol=1e-9, critic_approved=True, probe=False,
                history_path="residual_level1.csv")
    assert out.get("converged"), {k: out.get(k) for k in ("error", "history_file")}
    assert (tmp_path / "residual_level1.csv").is_file(), (
        "a relative history_path left no history file in the task directory: "
        f"{out.get('history_file')}")
    assert not (other / "residual_level1.csv").exists(), "it landed in the server's cwd"
    hf = out.get("history_file") or {}
    assert "error" not in hf, hf


def test_without_a_task_directory_it_lands_beside_the_participants(
        tmp_path, tools, monkeypatch, tmp_path_factory):
    _elsewhere(tmp_path_factory, monkeypatch)
    monkeypatch.delenv("OPENPASO_CELL_WORKDIR", raising=False)
    out = _call(tools["couple"], participants=json.dumps(_pair(tmp_path)), max_iter=80,
                tol=1e-9, critic_approved=True, probe=False,
                history_path="residual_level1.csv")
    assert (tmp_path / "residual_level1.csv").is_file(), out.get("history_file")


def test_a_relative_path_that_escapes_the_task_directory_is_still_refused(
        tmp_path, tools, monkeypatch, tmp_path_factory):
    _elsewhere(tmp_path_factory, monkeypatch)
    task = tmp_path / "task"
    task.mkdir()
    monkeypatch.setenv("OPENPASO_CELL_WORKDIR", str(task))
    out = _call(tools["couple"], participants=json.dumps(_pair(task)), max_iter=80,
                tol=1e-9, critic_approved=True, probe=False,
                history_path="../escaped_level1.csv")
    assert not (tmp_path / "escaped_level1.csv").exists()
    assert "outside" in str((out.get("history_file") or {}).get("error", "")), out.get("history_file")


def test_couple_levels_honours_a_relative_history_dir(
        tmp_path, tools, monkeypatch, tmp_path_factory):
    other = _elsewhere(tmp_path_factory, monkeypatch)
    monkeypatch.setenv("OPENPASO_CELL_WORKDIR", str(tmp_path))
    levels = [{"level": 1, "A": {"nx": 4}, "B": {"nx": 5}},
              {"level": 2, "A": {"nx": 8}, "B": {"nx": 10}}]
    out = _call(tools["couple_levels"], participants=json.dumps(_pair(tmp_path)),
                levels=json.dumps(levels), critic_approved=True, max_iter=80, tol=1e-9,
                probe=False, history_dir=".", history_pattern="residual_level{k}.csv")
    assert out.get("all_levels_converged"), out
    for k in (1, 2):
        assert (tmp_path / f"residual_level{k}.csv").is_file(), (
            f"level {k}: history_dir='.' wrote no history "
            f"({[x.get('history_file') for x in out.get('levels', [])]})")
        assert not (other / f"residual_level{k}.csv").exists()
