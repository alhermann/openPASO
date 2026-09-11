"""check_input: the setup checks as a standalone gate for a deck the agent runs itself. Names every
defect in one call (section names against the installed binary's grammar, condition ids against the
topology, the measured TSI defects); writes nothing; a clean deck gets no finding."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


def _tools():
    from mcp.server.fastmcp import FastMCP
    from core.registry import load_all_backends
    load_all_backends()
    import tools.consolidated as C
    mcp = FastMCP("probe"); tools = {}
    orig = mcp.tool
    def tool(*a, **k):
        dec = orig(*a, **k)
        def wrap(fn):
            tools[fn.__name__] = fn
            return dec(fn)
        return wrap
    mcp.tool = tool
    C.register_consolidated_tools(mcp)
    return tools


BAD = ('PROBLEM TYPE:\n  PROBLEMTYPE: "Thermo_Structure_Interaction"\nIO/RUNTIME VTK OUTPUT/THERMO:\n  OUTPUT_THERMO: true\n'
       'SOLIDSCATRA ELEMENTS:\n  - "1 SOLIDSCATRA HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM linear TYPE Undefined"\n'
       'MATERIALS:\n  - MAT: 1\n    MAT_Struct_ThermoStVenantK:\n      YOUNGNUM: 1\n'
       'DESIGN VOL NEUMANN CONDITIONS:\n  - E: 1\n    NUMDOF: 3\nDVOLUME-NODE TOPOLOGY:\n  - "NODE 1 DVOLUME 1"\n')


def test_a_defective_4c_deck_gets_every_defect_named_in_one_call(tmp_path):
    if not Path("/home/alexander/4C/build/4C").is_file():
        pytest.skip("4C binary not on this host")
    t = _tools()
    (tmp_path / "deck_U.4C.yaml").write_text(BAD)
    out = t["check_input"](solver="fourc", input_path=str(tmp_path / "deck_U.4C.yaml"))
    assert out.startswith("CHECK_INPUT (fourc, deck_U.4C.yaml)")
    assert "IO/RUNTIME VTK OUTPUT/THERMO" in out and "THERMAL DYNAMIC/RUNTIME VTK OUTPUT" in out
    assert "SOLIDSCATRA ELEMENTS" in out and "ELEMENT TYPE" in out and "STRUCTURE ELEMENTS" in out
    assert "DVOLUME" in out and "CLONING MATERIAL MAP" in out
    assert "DESIGN VOL NEUMANN CONDITIONS names E id(s) 1" in out
    assert sorted(p.name for p in tmp_path.iterdir()) == ["deck_U.4C.yaml"]        # writes nothing


def test_a_clean_scalar_transport_deck_gets_no_finding(tmp_path):
    if not Path("/home/alexander/4C/build/4C").is_file():
        pytest.skip("4C binary not on this host")
    t = _tools()
    clean = ('PROBLEM TYPE:\n  PROBLEMTYPE: "Scalar_Transport"\nSCALAR TRANSPORT DYNAMIC:\n  CALCFLUX_BOUNDARY: "diffusive"\n'
             'IO/RUNTIME VTK OUTPUT:\n  INTERVAL_STEPS: 1\n'
             'MATERIALS:\n  - MAT: 1\n    MAT_scatra:\n      DIFFUSIVITY: 1.0\n'
             'SCATRA FLUX CALC LINE CONDITIONS:\n  - E: 2\nDESIGN LINE DIRICH CONDITIONS:\n  - E: 1\n    NUMDOF: 1\n'
             'DLINE-NODE TOPOLOGY:\n  - "NODE 1 DLINE 1"\n  - "NODE 2 DLINE 2"\n')
    out = t["check_input"](solver="fourc", input_content=clean)
    assert "no defect named" in out, out


def test_the_instructions_and_the_ladder_name_the_gate():
    from core.instructions import INSTRUCTIONS
    assert "check_input(solver, input_path)" in INSTRUCTIONS
    assert "writes nothing into your directories" in INSTRUCTIONS      # couple_levels no longer claims to write config.json
    src = (ROOT / "src" / "tools" / "result_audit.py").read_text()
    assert "check_input(solver='fourc', input_path=<the deck>)" in src


def test_the_measured_first_attempt_traps_ride_with_the_4c_coupling_door():
    """The twelve-deck trap list (one topology section per kind, Scalar_Transport for the flux, the exact
    runtime output section names, element types vs sections, the material's list-valued YOUNG, the THERMO
    Dirichlet family, counter-clockwise nodes) is a served fact, measured the way every other backend's
    facts were, and it points at check_input."""
    t = _tools()
    # the session's first reply carries the orchestrator lead and the contract for the PARENT; the
    # WORKER's own door call (the second, pointer-mode reply) leads with the deciding facts
    first = t["knowledge"](topic="coupling", solver="fourc", physics="thermoelastic")
    reply = t["knowledge"](topic="coupling", solver="fourc", physics="thermoelastic")
    reply = reply if isinstance(reply, str) else str(reply)
    assert "WHAT DECIDES THIS RUN" in reply[:800]
    assert "FIRST-ATTEMPT DECK TRAPS" in reply and "ONE topology section per kind" in reply
    assert "check_input(solver='fourc'" in reply and "counter-clockwise" in reply
    assert "partner_values" in reply and "EXPORT SELF-CHECK" in reply      # the contract rides with the facts


def test_the_parents_first_4c_reply_keeps_the_must_read_whole():
    """Measured: the 31k thermo-elastic block pushed part B (rho budget, interface-file rule, measured
    history) past the 48k cap in the parent's first reply. Now the first reply keeps the contract's
    prose plus a pointer and part B whole; the worker's own (second) call leads with the full block."""
    t = _tools()
    first = t["knowledge"](topic="coupling", solver="fourc", physics="thermoelastic")
    first = first if isinstance(first, str) else str(first)
    assert "THE COUPLING HISTORY IS MEASURED" in first and "rho" in first        # part B's tail survives
    assert "IS NOT REPEATED IN THIS FIRST REPLY" in first and "partner_values" not in first
    second = t["knowledge"](topic="coupling", solver="fourc", physics="thermoelastic")
    second = second if isinstance(second, str) else str(second)
    assert "partner_values" in second and "EXPORT SELF-CHECK" in second and "WHAT DECIDES THIS RUN" in second[:800]
