"""A dead flux column must not silence the check on the field beside it.

THE HOLE. `contract_findings` compares the agent's reported interface residual
with its own two interface files. It used to skip a level outright when every
flux entry on both sides was exactly zero -- so a field that disagreed by 100%
between the sides went unreported next to a residual of 8e-7, because a
DIFFERENT column happened to carry nothing. The same guard shape (a check whose
precondition excludes the degenerate case it exists to catch) had already been
found and fixed four times elsewhere; this test feeds the degenerate input
itself, which no earlier test did.

ADMISSION, old vs new over every coupled run with work files (1373): 0 of 35
CORRECT changed; 3 newly named, all ungraded, each reporting a residual of 0 or
8e-7 while its two interface fields differ by 38%, 99% and 100% and both flux
columns are identically zero.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tools.result_audit import contract_findings  # noqa: E402

YS = [i / 10 for i in range(6)]


def _write(work: Path, u_a, u_b, q_a, q_b, residual="7.97e-07"):
    for side, u, q in (("A", u_a, q_a), ("B", u_b, q_b)):
        rows = ["x,y,u,qn"] + [f"0.625,{y},{uu},{qq}" for y, uu, qq in zip(YS, u, q)]
        (work / f"interface_level1_{side}.csv").write_text("\n".join(rows) + "\n")
    (work / "RESULT.txt").write_text(f"INTERFACE_RESIDUAL = {residual}\n")


def _residual_findings(work):
    return [f for f in contract_findings(work) if f.get("sequence") == "interface residual"]


def test_a_field_mismatch_is_named_when_both_flux_columns_are_zero(tmp_path):
    _write(tmp_path, u_a=[1.0] * 6, u_b=[0.0] * 6, q_a=[0.0] * 6, q_b=[0.0] * 6)
    found = _residual_findings(tmp_path)
    assert found, "a 100% field mismatch beside a dead flux column went unreported"
    text = found[0]["finding"]
    assert "identically zero" in text and "fail to cancel by 0%" not in text


def test_agreeing_sides_with_a_dead_flux_column_stay_silent(tmp_path):
    _write(tmp_path, u_a=[1.0] * 6, u_b=[1.0] * 6, q_a=[0.0] * 6, q_b=[0.0] * 6)
    assert not _residual_findings(tmp_path)


def test_a_live_flux_that_fails_to_cancel_keeps_its_wording(tmp_path):
    _write(tmp_path, u_a=[1.0] * 6, u_b=[1.0] * 6, q_a=[1.0] * 6, q_b=[1.0] * 6)
    found = _residual_findings(tmp_path)
    assert found and "fail to cancel by 200%" in found[0]["finding"]
