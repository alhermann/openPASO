"""couple_levels runs every prescribed level in one call: it merges each level's
keys into each side's config.json, warm-starts level k+1 from level k, writes
residual_level<k>.csv per level and keeps each side's console per level."""
import asyncio
import inspect
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TOY = '''import json
from pathlib import Path
import os
cfg = json.loads(Path("config.json").read_text()) if Path("config.json").is_file() else {}
cfg.update(json.loads(os.environ.get("OASIS_CONFIG_JSON") or "{}"))
imp = json.loads(Path("imports.json").read_text()) if Path("imports.json").is_file() else {}
other = 0.0
for d in imp.values():
    other = float((d.get("values") or [0.0])[0])
n = int(cfg.get("nx", 1))
mine = COEF * other + OFFSET + 0.001 / n
print(f"toy console level {cfg.get('level')} nx {n}: NDOF = {n}")
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


def test_three_levels_in_one_call(tmp_path, tools, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OASIS_CELL_WORKDIR", str(tmp_path))
    for name, coef, off in (("A", 0.6, 1.0), ("B", 0.7, 0.2)):
        d = tmp_path / f"side_{name}"
        d.mkdir()
        (d / "part.py").write_text(TOY.replace("COEF", str(coef)).replace("OFFSET", str(off)))
        (d / "config.json").write_text(json.dumps({"k": 1.0}))
    parts = [{"name": "A", "command": [sys.executable, "part.py"], "work_dir": str(tmp_path / "side_A"), "imports_from": ["B"]},
             {"name": "B", "command": [sys.executable, "part.py"], "work_dir": str(tmp_path / "side_B"), "imports_from": ["A"]}]
    levels = [{"level": 1, "A": {"nx": 4}, "B": {"nx": 5}}, {"level": 2, "A": {"nx": 8}, "B": {"nx": 10}},
              {"level": 3, "A": {"nx": 16}, "B": {"nx": 20}}]
    out = _call(tools["couple_levels"], participants=json.dumps(parts), levels=json.dumps(levels),
                critic_approved=True, max_iter=80, tol=1e-9, probe=False, history_pattern="residual_level{k}.csv")
    assert out["all_levels_converged"], out
    assert out["levels_run"] == 3 and [x["level"] for x in out["levels"]] == [1, 2, 3]
    for k in (1, 2, 3):
        assert (tmp_path / f"residual_level{k}.csv").is_file()
        for side in ("A", "B"):
            log = tmp_path / f"side_{side}" / f"participant_output_level{k}.log"
            assert log.is_file() and f"toy console level {k}" in log.read_text()
    # OASiS wrote NOTHING into the participants' own files: config.json is untouched
    cfg = json.loads((tmp_path / "side_A" / "config.json").read_text())
    assert cfg == {"k": 1.0}, cfg
    # levels 2 and 3 warm-started (fewer iterations than level 1, which started cold)
    assert out["levels"][1]["iterations"] <= out["levels"][0]["iterations"]
    assert "EVERY REQUESTED LEVEL CONVERGED" in out["next_step"]


def test_a_level_that_fails_stops_the_sequence(tmp_path, tools, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OASIS_CELL_WORKDIR", str(tmp_path))
    for name in ("A", "B"):
        d = tmp_path / f"side_{name}"
        d.mkdir()
        (d / "part.py").write_text("import json, sys, os\nfrom pathlib import Path\ncfg = json.loads(os.environ.get('OASIS_CONFIG_JSON') or '{}')\n"
                                   "if int(cfg.get('nx', 1)) > 4:\n    sys.exit(3)\n"
                                   "Path('exports.json').write_text(json.dumps({'field_name': 'u', 'n_points': 1, 'coordinates': [[0.5, 0.0]], 'values': [1.0], 'normal_fluxes': [0.0]}))\n")
    parts = [{"name": "A", "command": [sys.executable, "part.py"], "work_dir": str(tmp_path / "side_A"), "imports_from": ["B"]},
             {"name": "B", "command": [sys.executable, "part.py"], "work_dir": str(tmp_path / "side_B"), "imports_from": ["A"]}]
    levels = [{"level": 1, "A": {"nx": 4}, "B": {"nx": 4}}, {"level": 2, "A": {"nx": 8}, "B": {"nx": 8}}]
    out = _call(tools["couple_levels"], participants=json.dumps(parts), levels=json.dumps(levels),
                critic_approved=True, max_iter=20, tol=1e-9, probe=False)
    assert (tmp_path / "coupling_history_level1.csv").is_file()      # the neutral default name
    assert not out["all_levels_converged"] and out["levels_run"] == 2
    assert out["levels"][0]["converged"] and not out["levels"][1]["converged"]
    assert "LEVEL 2 DID NOT CONVERGE" in out["next_step"]
