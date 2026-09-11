"""Every audit and couple() reply names the next unmet step of a coupled run, as a sub-agent brief.

Measured: a small model that sees the whole coupled job at once gives up (8 of 9 runs in one honest
round, most with 25 minutes left); given one step whose end is a file it can see, it keeps the served
contract (6 of 6 single-step trials against 0 of 45 whole-problem runs). The ladder is computed from the
agent's own files only and knows no task's naming.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PART = 'import json\nimp = json.load(open("imports.json"))\n# exports.json LAST: the driver takes its existence as proof of success.\njson.dump({"values": [1.0], "normal_fluxes": [2.0]}, open("exports.json", "w"))\n'
SCRATCH = 'import json\nimp = json.load(open("imports.json"))\njson.dump({"values": [1.0], "normal_fluxes": [2.0]}, open("exports.json", "w"))\n'


def _w(d: Path, name: str, text: str) -> None:
    d.mkdir(parents=True, exist_ok=True); (d / name).write_text(text)


def test_no_participants_means_no_ladder(tmp_path):
    from tools.result_audit import coupled_ladder
    _w(tmp_path, "solution_level1.csv", "x,y,u\n0,0,1\n")
    assert coupled_ladder(tmp_path) is None


def test_step_1_when_one_participant_exists(tmp_path):
    from tools.result_audit import coupled_ladder
    _w(tmp_path / "side_A", "participant_A.py", PART)
    r = coupled_ladder(tmp_path)
    assert r["step"] == 1 and "SECOND PARTICIPANT" in r["text"] and "spawn_subagent(role='worker'" in r["text"]


def test_step_2_until_both_export_and_drivers_are_not_participants(tmp_path):
    from tools.result_audit import coupled_ladder
    _w(tmp_path / "side_A", "participant_A.py", PART)
    _w(tmp_path / "side_B", "participant_B.py", PART)
    _w(tmp_path, "run_coupling.py", PART)             # a driver mentions both files too
    _w(tmp_path / "side_A", "exports.json", json.dumps({"values": [1.0], "normal_fluxes": [2.0]}))
    r = coupled_ladder(tmp_path)
    assert r["step"] == 2 and "side_B/participant_B.py" in r["text"] and "run_coupling" not in r["text"]


def test_step_3_couple_level_1_then_4_deliverables_then_6_complete(tmp_path):
    from tools.result_audit import coupled_ladder
    for s in ("A", "B"):
        _w(tmp_path / f"side_{s}", f"participant_{s}.py", PART)
        _w(tmp_path / f"side_{s}", "exports.json", json.dumps({"values": [1.0], "normal_fluxes": [2.0]}))
    assert coupled_ladder(tmp_path)["step"] == 3
    _w(tmp_path, "residual_level1.csv", "iteration,interface_residual\n1,0.5\n2,0.1\n3,0.01\n")
    r = coupled_ladder(tmp_path)
    assert r["step"] == 4 and "LEVEL 1" in r["text"]
    for s in ("A", "B"):
        _w(tmp_path, f"solution_level1_{s}.csv", "x,y,u\n0,0,1\n")
        _w(tmp_path, f"interface_level1_{s}.csv", "x,y,u,qn\n0,0,1,2\n")
    r = coupled_ladder(tmp_path)
    assert r["step"] == 5 and "RUN LOGS" in r["text"]


def test_the_audit_reply_carries_the_ladder(tmp_path):
    from tools.result_audit import audit
    _w(tmp_path / "side_A", "participant_A.py", PART)
    r = audit(str(tmp_path))
    assert r.get("next_step") and "LADDER STEP 1" in r["next_step"]
    assert "LADDER STEP 1" in (r.get("what_to_fix_next") or "")


def test_a_participant_written_from_scratch_is_sent_back_to_the_served_contract(tmp_path):
    """Every served contract carries one of four lines; a script with none of them
    was not copied from the door (measured: zero interface loads, unresponsive
    sides). Before any level converges, restoring the contract is the step."""
    from tools.result_audit import coupled_ladder
    _w(tmp_path / "side_A", "participant_A.py", PART)
    _w(tmp_path / "side_B", "participant_B.py", SCRATCH)
    _w(tmp_path / "side_A", "exports.json", json.dumps({"values": [1.0], "normal_fluxes": [2.0]}))
    r = coupled_ladder(tmp_path)
    assert r["step"] == 1 and "RESTORE THE SERVED CONTRACT" in r["text"] and "side_B/participant_B.py" in r["text"]
    assert "physics='thermoelastic'" in r["brief"] and "EXPORT SELF-CHECK" in r["brief"]
    # once a level has converged the scripts evidently work: the gate steps aside
    _w(tmp_path, "residual_level1.csv", "iteration,interface_residual\n1,0.5\n2,0.1\n3,0.01\n")
    _w(tmp_path / "side_B", "exports.json", json.dumps({"values": [1.0], "normal_fluxes": [2.0]}))
    assert "RESTORE THE SERVED CONTRACT" not in coupled_ladder(tmp_path)["text"]


def test_the_next_level_brief_names_couple_levels_with_a_literal_k(tmp_path):
    """Measured in round 25: the brief rendered {k} as the level number ("with 1") and no cell used the
    one-call mesh sequence; the next action now leads and the placeholder is literal."""
    from tools.result_audit import coupled_ladder
    for s in ("A", "B"):
        _w(tmp_path / f"side_{s}", f"participant_{s}.py", PART)
        _w(tmp_path / f"side_{s}", "exports.json", json.dumps({"values": [1.0], "normal_fluxes": [2.0]}))
        _w(tmp_path, f"solution_level1_{s}.csv", "x,y,u\n0,0,1\n")
        _w(tmp_path, f"interface_level1_{s}.csv", "x,y,u,qn\n0,0,1,2\n")
        _w(tmp_path, f"run_level1_{s}.csv", "x\n")
    _w(tmp_path, "residual_level1.csv", "iteration,interface_residual\n1,0.5\n2,0.1\n3,0.01\n")
    r = coupled_ladder(tmp_path)
    txt = r["text"]
    assert "couple_levels(" in txt.split("HAND THIS STEP")[0]          # the next action leads the step line
    assert "{k}" in txt and "with 1" not in txt
