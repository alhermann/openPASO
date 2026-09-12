"""couple(): a run that 'converges' at iteration 2 because both participants re-export unchanged data is
led as NOT A COUPLED RESULT YET, never as 'LEVEL k CONVERGED, write the deliverables'. Measured in a
round-39 cell: stale exports.json on both sides, residual 2e-13 at iteration 2, funnel 'unresponsive',
lead 'CONVERGED'."""
from __future__ import annotations
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEAF = '''import json
from pathlib import Path
Path("exports.json").write_text(json.dumps({"field_name": "u", "n_points": 1, "coordinates": [[0.5, 0.0]],
                                             "values": [{V}], "normal_fluxes": [0.0]}))
print("NDOF = 7")
'''
ALIVE = '''import json
from pathlib import Path
imp = json.loads(Path("imports.json").read_text()) if Path("imports.json").is_file() else {}
other = 0.0
for d in imp.values():
    other = float((d.get("values") or [0.0])[0])
Path("exports.json").write_text(json.dumps({"field_name": "u", "n_points": 1, "coordinates": [[0.5, 0.0]],
                                             "values": [{A} + 0.5 * other], "normal_fluxes": [0.0]}))
print("NDOF = 7")
'''


def _couple():
    from mcp.server.fastmcp import FastMCP
    from core.registry import load_all_backends
    load_all_backends()
    from tools.consolidated import register_consolidated_tools
    mcp = FastMCP("probe"); tools = {}
    orig = mcp.tool
    def tool(*a, **k):
        dec = orig(*a, **k)
        def wrap(fn):
            tools[fn.__name__] = fn
            return dec(fn)
        return wrap
    mcp.tool = tool
    register_consolidated_tools(mcp)
    return tools["couple"]


def _run(tmp_path, a_body, b_body):
    import asyncio, inspect
    (tmp_path / "side_A").mkdir(); (tmp_path / "side_B").mkdir()
    (tmp_path / "side_A" / "part.py").write_text(a_body)
    (tmp_path / "side_B" / "part.py").write_text(b_body)
    parts = [{"name": "A", "command": [sys.executable, "part.py"], "work_dir": str(tmp_path / "side_A"), "imports_from": ["B"]},
             {"name": "B", "command": [sys.executable, "part.py"], "work_dir": str(tmp_path / "side_B"), "imports_from": ["A"]}]
    r = _couple()(participants=json.dumps(parts), max_iter=40, tol=1e-8, critic_approved=True, probe=False, iface_level=1)
    if inspect.isawaitable(r):
        r = asyncio.run(r)
    return json.loads(r)


def test_two_deaf_participants_are_not_led_as_a_converged_level(tmp_path):
    d = _run(tmp_path, DEAF.replace("{V}", "1.0"), DEAF.replace("{V}", "2.0"))
    lead = d["what_to_fix_next"]
    assert d["converged"] is True and d["iterations"] <= 2, (d["converged"], d["iterations"])
    assert lead.startswith("LEVEL 1 IS NOT A COUPLED RESULT YET"), lead[:300]
    assert "CONVERGED. FIRST" not in lead
    assert "byte-identical" in lead and "imports.json" in lead


def test_a_live_pair_keeps_the_converged_lead(tmp_path):
    d = _run(tmp_path, ALIVE.replace("{A}", "1.0"), ALIVE.replace("{A}", "2.0"))
    assert d["converged"] is True and d["iterations"] >= 3
    assert d["what_to_fix_next"].startswith("LEVEL 1 CONVERGED. FIRST"), d["what_to_fix_next"][:200]


def test_a_dump_with_the_agents_own_suffix_is_not_reported_missing(tmp_path):
    """Round 44, C2 7163: the participants wrote field_level1_A.csv / interface_level1_A.csv (side suffix) and the
    exact-name check said the dumps were missing six times."""
    d = _run(tmp_path, ALIVE.replace("{A}", "1.0").replace('print("NDOF = 7")', 'print("NDOF = 7")\nPath("field_level1_A.csv").write_text("x,y,u\\n0,0,1\\n")\nPath("interface_level1_A.csv").write_text("x,y,u,qn\\n0,0,1,0\\n")'),
             ALIVE.replace("{A}", "2.0").replace('print("NDOF = 7")', 'print("NDOF = 7")\nPath("field_level1_B.csv").write_text("x,y,u\\n0,0,1\\n")\nPath("interface_level1_B.csv").write_text("x,y,u,qn\\n0,0,1,0\\n")'))
    lead = d["what_to_fix_next"]
    assert "PER-LEVEL DUMPS ARE MISSING" not in lead, lead[:400]
    assert lead.startswith("LEVEL 1 CONVERGED. FIRST"), lead[:200]
