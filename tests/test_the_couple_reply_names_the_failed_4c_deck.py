"""When a 4C participant exits non-zero inside couple(), the lead finding carries what its directory
holds -- 4C's own stop line and the deck's defects with the closest real section names -- instead of an
empty stderr tail and 'fix that script'. Measured on the parent's most-called tool (couple: 4 calls per
run against 0 check_input)."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FOURC = Path("/home/user/4C/build/4C")

FAILING_4C_SIDE = '''import subprocess, sys
from pathlib import Path
deck = ('PROBLEM TYPE:\\n  PROBLEMTYPE: "Scalar_Transport"\\nSCALAR TRANSPORT DYNAMIC:\\n  CALCFLUX_BOUNDARY: "diffusive"\\n'
        'THERMO FLUX CALC LINE CONDITIONS:\\n  - E: 2\\nDESIGN LINE DIRICH CONDITIONS:\\n  - E: 1\\n    NUMDOF: 1\\n'
        'DLINE-NODE TOPOLOGY:\\n  - "NODE 1 DLINE 1"\\n')
Path("slab.4C.yaml").write_text(deck)
subprocess.run("stdbuf -oL -eL /home/user/4C/build/4C slab.4C.yaml out > slab.4C.yaml.log 2>&1", shell=True)
print("NDOF = 9")
sys.exit(1)
'''
QUIET_SIDE = '''import json
from pathlib import Path
Path("exports.json").write_text(json.dumps({"field_name": "thermoelastic", "n_points": 3,
    "coordinates": [[0.8, 0.0], [0.8, 0.5], [0.8, 1.0]], "values": [[0.0, 0.0, 0.0]] * 3, "normal_fluxes": [[0.0, 0.0, 0.0]] * 3}))
print("NDOF = 12")
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


def test_the_lead_finding_carries_4cs_stop_and_the_decks_defects(tmp_path):
    if not FOURC.is_file():
        pytest.skip("4C binary not on this host")
    import asyncio, inspect
    (tmp_path / "side_A").mkdir(); (tmp_path / "side_B").mkdir()
    (tmp_path / "side_A" / "part.py").write_text(FAILING_4C_SIDE)
    (tmp_path / "side_B" / "part.py").write_text(QUIET_SIDE)
    parts = [{"name": "A", "command": [sys.executable, "part.py"], "work_dir": str(tmp_path / "side_A"), "imports_from": ["B"]},
             {"name": "B", "command": [sys.executable, "part.py"], "work_dir": str(tmp_path / "side_B"), "imports_from": ["A"]}]
    r = _couple()(participants=json.dumps(parts), max_iter=3, tol=1e-6, critic_approved=True, probe=False)
    if inspect.isawaitable(r):
        r = asyncio.run(r)
    lead = json.loads(r)["what_to_fix_next"]
    assert "PARTICIPANT A EXITED NON-ZERO" in lead
    assert "ON DISK IN side_A" in lead, lead
    assert "THERMO FLUX CALC LINE CONDITIONS" in lead and "SCATRA FLUX CALC LINE CONDITIONS" in lead, lead   # defect + closest real name
    assert "not a valid section name" in lead or "4C stopped" in lead, lead                                    # 4C's own words
    assert "Fix the deck as named" in lead
