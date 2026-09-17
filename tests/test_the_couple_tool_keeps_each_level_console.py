"""The coupling tool keeps each side's captured solver console per level.

Measured on four development cells: participant_output.log holds only the latest
level's console, the agents wrote every level's run log from it at the end, and
every level carried the finest level's DOF count. When the call names a level
(iface_level, or a history_path like residual_level<k>.csv) the same console is
also kept as participant_output_level<k>.log next to exports.json."""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TOY = '''import json, sys
from pathlib import Path
imp = json.loads(Path("imports.json").read_text()) if Path("imports.json").is_file() else {}
other = 0.0
for d in imp.values():
    v = d.get("values") or [0.0]
    other = float(v[0])
mine = {A}
print("toy solver console: NDOF = 7")
print("NDOF = 7")            # the canonical run-log line every served participant prints
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
def couple_tool():
    os.chdir(ROOT)
    from core.registry import load_all_backends
    load_all_backends()
    from tools.consolidated import register_consolidated_tools
    mcp = _CollectingMCP()
    register_consolidated_tools(mcp)
    return mcp.tools["couple"]


def test_each_side_keeps_its_console_for_the_named_level(tmp_path, couple_tool, monkeypatch):
    import asyncio, inspect
    monkeypatch.chdir(tmp_path)
    for name, expr in (("A", "0.3 * other + 1.0"), ("B", "0.5 * other + 0.2")):
        d = tmp_path / f"side_{name}"
        d.mkdir()
        (d / "part.py").write_text(TOY.replace("{A}", expr))
    parts = [{"name": "A", "command": [sys.executable, "part.py"], "work_dir": str(tmp_path / "side_A"), "imports_from": ["B"]},
             {"name": "B", "command": [sys.executable, "part.py"], "work_dir": str(tmp_path / "side_B"), "imports_from": ["A"]}]
    r = couple_tool(participants=json.dumps(parts), max_iter=60, tol=1e-8, critic_approved=True, iface_level=2, probe=False)
    if inspect.iscoroutine(r):
        r = asyncio.run(r)
    out = json.loads(str(r))
    assert out.get("converged"), {k: out.get(k) for k in ("error", "reason", "why", "history", "verdict")}
    # the converged reply leads with this level's run logs FIRST (measured: left to the end they were
    # lost in all but 2-4 cells of 9), then names the one-call mesh sequence with {k} kept literal
    # (round 25: six proven couplings ran couple() per level and the wall cut them)
    lead = out.get("what_to_fix_next") or ""
    assert "FIRST, WRITE THIS LEVEL'S RUN LOGS" in lead, lead[:600]
    assert "couple_levels(" in lead and "{k}" in lead, lead[:600]
    assert lead.index("FIRST, WRITE THIS LEVEL'S RUN LOGS") < lead.index("couple_levels("), lead[:600]
    # the participants list is echoed verbatim (the worker sees only the brief; a placeholder got bare names)
    assert "participants='[{\"name\": \"A\"" in lead and str(tmp_path / "side_A") in lead, lead[:1200]
    assert "<the same list" not in lead
    assert "MEASURED COST: this level's iteration took" in lead and "4x and 16x the cells" in lead     # the next levels' cost, in seconds
    assert (out.get("relaxation") or {}).get("mode") == "aitken", out.get("relaxation")      # a single-field exchange keeps Aitken
    for name in ("A", "B"):
        per_level = tmp_path / f"side_{name}" / "participant_output_level2.log"
        assert per_level.is_file(), f"side {name}: no per-level console copy"
        assert "toy solver console: NDOF = 7" in per_level.read_text()
    assert not (tmp_path / "side_A" / "participant_output_level1.log").exists()


TOY3 = '''import json
from pathlib import Path
imp = json.loads(Path("imports.json").read_text()) if Path("imports.json").is_file() else {}
other = [0.0, 0.0, 0.0]
for d in imp.values():
    v = d.get("values") or [[0.0, 0.0, 0.0]]
    other = [float(x) for x in v[0]]
mine = [COEF * other[0] + 1.0, COEF * other[1] + 0.01, COEF * other[2] + 0.001]
print("toy solver console: NDOF = 21")
Path("exports.json").write_text(json.dumps({"field_name": "thermoelastic", "n_points": 1, "coordinates": [[0.5, 0.0]],
    "values": [mine], "normal_fluxes": [[0.0, 0.0, 0.0]]}))
'''


def test_a_three_component_exchange_does_not_crash_the_tool(tmp_path, couple_tool, monkeypatch):
    """Measured 2026-09-11: every couple() call of the thermo-elastic cells died with
    "float() argument must be a string or a real number, not 'list'" in the tool's own
    export summary, after the driver had run the coupling; one cell spent twelve calls on it."""
    import asyncio, inspect
    monkeypatch.chdir(tmp_path)
    for name, coef in (("A", 0.3), ("B", 0.5)):
        d = tmp_path / f"side_{name}"
        d.mkdir()
        (d / "part.py").write_text(TOY3.replace("COEF", str(coef)))
    parts = [{"name": "A", "command": [sys.executable, "part.py"], "work_dir": str(tmp_path / "side_A"), "imports_from": ["B"]},
             {"name": "B", "command": [sys.executable, "part.py"], "work_dir": str(tmp_path / "side_B"), "imports_from": ["A"]}]
    r = couple_tool(participants=json.dumps(parts), max_iter=60, tol=1e-8, critic_approved=True, probe=False)
    if inspect.iscoroutine(r):
        r = asyncio.run(r)
    out = json.loads(str(r))
    assert "float()" not in str(out.get("error")), out.get("error")
    assert out.get("converged"), {k: out.get(k) for k in ("error", "history")}
    # a three-component exchange resolves accelerator='auto' to Anderson mixing (measured: 25 vs 55 iterations)
    assert (out.get("relaxation") or {}).get("mode") == "anderson", out.get("relaxation")
    assert out["exports"]["A"]["first_values"][0] == [pytest.approx(v) for v in out["exports"]["A"]["first_values"][0]]
    assert len(out["exports"]["A"]["first_values"][0]) == 3


def test_an_unchanged_mesh_between_levels_leads_the_reply(tmp_path, couple_tool, monkeypatch):
    """Round 27: a cell coupled three levels on the same mesh (NDOF 693 -> 693 -> 693) and was graded as no
    refinement. The tool compares this level's NDOF line with the previous level's captured console and
    says so first."""
    import asyncio, inspect
    monkeypatch.chdir(tmp_path)
    for name, expr in (("A", "0.3 * other + 1.0"), ("B", "0.5 * other + 0.2")):
        d = tmp_path / f"side_{name}"
        d.mkdir()
        (d / "part.py").write_text(TOY.replace("{A}", expr))
    parts = [{"name": "A", "command": [sys.executable, "part.py"], "work_dir": str(tmp_path / "side_A"), "imports_from": ["B"]},
             {"name": "B", "command": [sys.executable, "part.py"], "work_dir": str(tmp_path / "side_B"), "imports_from": ["A"]}]
    for lvl in (1, 2):
        r = couple_tool(participants=json.dumps(parts), max_iter=60, tol=1e-8, critic_approved=True, iface_level=lvl, probe=False)
        if inspect.iscoroutine(r):
            r = asyncio.run(r)
        out = json.loads(str(r))
        assert out.get("converged"), out.get("error")
    lead = out.get("what_to_fix_next") or ""
    assert lead.startswith("MESH UNCHANGED FROM LEVEL 1"), lead[:300]
    assert "A (NDOF 7 at both levels)" in lead and "B (NDOF 7 at both levels)" in lead


def test_a_wrong_level_run_log_on_disk_leads_the_next_couple_reply(tmp_path, couple_tool, monkeypatch):
    """Rounds 29-31: run logs copied from another level were named by an audit nobody called; couple()
    is called several times a run, so the deliverable findings ride on it."""
    import asyncio, inspect
    monkeypatch.chdir(tmp_path)
    for name, expr in (("A", "0.3 * other + 1.0"), ("B", "0.5 * other + 0.2")):
        d = tmp_path / f"side_{name}"
        d.mkdir()
        (d / "part.py").write_text(TOY.replace("{A}", expr))
    parts = [{"name": "A", "command": [sys.executable, "part.py"], "work_dir": str(tmp_path / "side_A"), "imports_from": ["B"]},
             {"name": "B", "command": [sys.executable, "part.py"], "work_dir": str(tmp_path / "side_B"), "imports_from": ["A"]}]
    r = couple_tool(participants=json.dumps(parts), max_iter=60, tol=1e-8, critic_approved=True, iface_level=1, probe=False)
    if inspect.iscoroutine(r):
        r = asyncio.run(r)
    assert json.loads(str(r)).get("converged")
    # the agent's level-1 run log for side A carries another level's NDOF
    (tmp_path / "run_level1_A.log").write_text("toy solver console\nNDOF = 999\n")
    r = couple_tool(participants=json.dumps(parts), max_iter=60, tol=1e-8, critic_approved=True, iface_level=2, probe=False)
    if inspect.iscoroutine(r):
        r = asyncio.run(r)
    out = json.loads(str(r))
    lead = out.get("what_to_fix_next") or ""
    assert lead.startswith("YOUR DELIVERABLES ON DISK HAVE DEFECTS"), lead[:200]
    assert "RUN LOG FROM THE WRONG LEVEL" in lead and "run_level1_A.log carries NDOF 999" in lead, lead[:700]
