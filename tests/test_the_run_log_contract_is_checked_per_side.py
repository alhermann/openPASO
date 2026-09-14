"""The run-log contract is graded per SIDE, so the audit has to check it per side.

Measured on a round-48 cell: both codes PROVEN, the coupling PROVEN, three levels of fields, and it
lost anyway -- "side A: no NDOF contract line for level(s) [2, 3]", the same for side B. The audit
asked only whether ANY log at a level carried a dof count, and level 1's logs did, so it stayed silent
at the moment it mattered. A gate weaker than the contract it guards is worth nothing.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _three_levels(w: Path, dof_at: set, sides=("A", "B")):
    for k in (1, 2, 3):
        for side in sides:
            tag = f"_{side}" if side else ""
            (w / f"solution_level{k}{tag}.csv").write_text("x,y,u\n0.1,0.1,1.0\n0.2,0.2,2.0\n")
            body = "solver console line\nsome iterations\n"
            if (k, side) in dof_at:
                body += "NDOF = 425\n"
            (w / f"run_level{k}{tag}.log").write_text(body)
        (w / f"residual_level{k}.csv").write_text("iteration,interface_residual\n1,1e-3\n2,1e-7\n")
    (w / "RESULT.txt").write_text("LEVELS = 3\nINTERFACE_RESIDUAL = 1e-7\nMESH_INDEPENDENCE = CONVERGED\n")


def _audit_text(w: Path) -> str:
    from tools.result_audit import audit
    return json.dumps(audit(str(w)))


def test_a_dof_line_at_level_one_no_longer_covers_the_others(tmp_path):
    _three_levels(tmp_path, dof_at={(1, "A"), (1, "B")})
    txt = _audit_text(tmp_path)
    assert "NO DOF-COUNT LINE" in txt
    for missing in ("2A", "2B", "3A", "3B"):
        assert missing in txt, f"the audit does not name {missing}"


def test_one_side_carrying_the_line_does_not_cover_the_other(tmp_path):
    _three_levels(tmp_path, dof_at={(k, "A") for k in (1, 2, 3)})
    txt = _audit_text(tmp_path)
    assert "NO DOF-COUNT LINE" in txt
    assert "1B" in txt and "2B" in txt and "3B" in txt
    assert "1A" not in txt.split("NO DOF-COUNT LINE")[1][:200]


def test_a_complete_set_is_not_flagged(tmp_path):
    _three_levels(tmp_path, dof_at={(k, s) for k in (1, 2, 3) for s in ("A", "B")})
    assert "NO DOF-COUNT LINE" not in _audit_text(tmp_path)


def test_a_single_code_run_is_still_judged_per_level(tmp_path):
    """No side suffix: the old per-level rule is the right one there."""
    _three_levels(tmp_path, dof_at={(1, "")}, sides=("",))
    txt = _audit_text(tmp_path)
    assert "NO DOF-COUNT LINE" in txt and "level(s)" in txt
