"""`all_levels_converged: true` over a coupling that never exchanged anything.

MEASURED 2026-09-17, C3 seed 8868 on the then-current build. One couple_levels
call came back

    "all_levels_converged": true,
    "levels": [{"level": 1, "converged": true, "iterations": 2, "residual": 0.0}, ...]
    "next_step": "EVERY REQUESTED LEVEL CONVERGED. For EACH level k and EACH
                  side: write the task's per-level field file ..."

and the residual history it wrote holds ONE row per level, the last value 0.0 at
every level. The agent did what the headline told it and handed in three levels;
the grader read the evidence as contradicted and the cell scored nothing.

couple() itself knew better. Its own reply for such a level carries
`IS NOT A COUPLED RESULT YET: the iteration stopped after N step(s) ... Do not
write this level's deliverables from this run` -- the guard added in 4c6da31d
after an identical case. couple_levels copied that sentence into
`what_to_fix_next` and then computed its summary from `converged` alone, so the
reply contradicted itself and the louder half was wrong.

A fixed point reached at once is what two sides that exchange nothing produce:
they cannot disagree, so the residual is zero at the first step. That is the
shape of a null coupling, and it is the one thing a coupled result may not be.

These tests run a real two-participant coupling through the registered tools.
The null case is built the way the recorded one arose: a participant whose
export does not depend on its imports.
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

# A REAL participant: its export depends on what the partner sent.
LIVE = '''import json, os
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
Path("exports.json").write_text(json.dumps({"field_name": "u", "n_points": 1,
    "coordinates": [[0.5, 0.0]], "values": [mine], "normal_fluxes": [0.0]}))
'''

# A NULL participant: it ignores imports.json entirely, so the pair reaches its
# "fixed point" at the first step without ever transmitting anything.
NULL = '''import json, os
from pathlib import Path
cfg = json.loads(Path("config.json").read_text()) if Path("config.json").is_file() else {}
cfg.update(json.loads(os.environ.get("OPENPASO_CONFIG_JSON") or "{}"))
n = int(cfg.get("nx", 1))
print(f"NDOF = {n}")
Path("exports.json").write_text(json.dumps({"field_name": "u", "n_points": 1,
    "coordinates": [[0.5, 0.0]], "values": [CONST], "normal_fluxes": [0.0]}))
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


def _pair(task_dir, body):
    specs = []
    for name, coef, off, const in (("A", 0.6, 1.0, 0.25), ("B", 0.7, 0.2, 0.5)):
        d = task_dir / f"side_{name}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "part.py").write_text(body.replace("COEF", str(coef))
                                       .replace("OFFSET", str(off))
                                       .replace("CONST", str(const)))
        specs.append({"name": name, "command": [sys.executable, "part.py"],
                      "work_dir": str(d), "imports_from": ["B" if name == "A" else "A"]})
    return specs


LEVELS = json.dumps([{"level": 1, "A": {"nx": 4}, "B": {"nx": 4}},
                     {"level": 2, "A": {"nx": 8}, "B": {"nx": 8}}])


def _levels(tmp_path, tools, body):
    return _call(tools["couple_levels"], participants=json.dumps(_pair(tmp_path, body)),
                 levels=LEVELS, max_iter=60, tol=1e-9, critic_approved=True, probe=False,
                 history_dir=str(tmp_path), history_pattern="residual_level{k}.csv")


def test_a_null_exchange_is_not_reported_as_every_level_converged(tmp_path, tools):
    out = _levels(tmp_path, tools, NULL)
    assert out.get("all_levels_converged") is not True, (
        "a pair whose exports ignore imports.json reached its fixed point at "
        "the first step and was still reported as every level converged:\n"
        + json.dumps(out, indent=2)[:1500])


def test_the_null_level_says_what_is_wrong_with_it(tmp_path, tools):
    out = _levels(tmp_path, tools, NULL)
    first = (out.get("levels") or [{}])[0]
    blob = json.dumps(first)
    assert "NOT A COUPLED RESULT" in blob or "not a coupled result" in blob.lower(), (
        "the level says nothing about the null exchange: " + blob[:900])


def test_the_next_step_does_not_ask_for_deliverables_from_a_null_level(tmp_path, tools):
    out = _levels(tmp_path, tools, NULL)
    nxt = str(out.get("next_step", ""))
    assert "EVERY REQUESTED LEVEL CONVERGED" not in nxt, nxt[:600]
    assert "write" not in nxt.lower() or "do not" in nxt.lower(), (
        "the next step still tells the agent to write this level's "
        "deliverables: " + nxt[:600])


def test_a_real_coupling_is_still_reported_as_converged(tmp_path, tools):
    """The guard may not cost a genuine coupling its verdict."""
    out = _levels(tmp_path, tools, LIVE)
    assert out.get("all_levels_converged") is True, json.dumps(out, indent=2)[:1500]
    assert "EVERY REQUESTED LEVEL CONVERGED" in str(out.get("next_step", ""))
    for lvl in out["levels"]:
        assert lvl.get("converged") is True, lvl
