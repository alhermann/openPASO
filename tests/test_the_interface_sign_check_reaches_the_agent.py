"""The interface sign check must be callable BY THE AGENT, and must fire.

It lived only inside the grader. That is the eighth time in this repo that a
mechanism existed, was instrumented, and did not reach the case it was built
for -- and the ninth was this test's own subject: the first version of the tool
unpacked read_interface_csv wrongly and abstained on every side with
"TypeError: type tuple doesn't define __round__", which looks exactly like an
honest NOT_ASSESSED.

So this test asserts three things, in order of what actually goes wrong:
  1. the tool is registered, so an agent can call it at all;
  2. it FIRES on the submission it was written for;
  3. it stays QUIET on the one that is correct.

Fixtures are two real submissions, both graded with the answers sealed:
  C2_27b_MCP_seed303  COMPLETED_UNPHYSICAL, order 1.2434. Both prescribed
      codes genuinely ran, the coupling genuinely iterated over three levels,
      the interface field matched to 0.000e+00 -- and one side reported its
      flux with the INWARD normal, giving an implied coefficient of -250.8
      where +200 was right, and a relative flux jump that GREW under
      refinement: 8.139e-01, 9.066e-01, 9.530e-01.
  the 4C+Kratos reference  CORRECT, order 1.9796. Implied coefficients +0.98
      to +1.30 on the k=1 side and +200.4 to +206.7 on the k=200 side, which
      is the check recovering both conductivities from the submission alone.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WRONG = ROOT / "campaign3_blind" / "runs" / "C2_27b_MCP_seed303" / "work"
# A C2 submission graded CORRECT, kept outside the repository. Point at it to run the
# checks that need it; they skip otherwise.
RIGHT = Path(__import__("os").environ.get("OPENPASO_C2_CORRECT_SUBMISSION", "/nonexistent"))


def _tool():
    from tools.consolidated import register_consolidated_tools

    class Cap:
        def __init__(self):
            self.fns = {}

        def tool(self, *a, **k):
            def deco(fn):
                self.fns[fn.__name__] = fn
                return fn
            return deco

    cap = Cap()
    register_consolidated_tools(cap)
    return cap.fns


def _coefficients(root: Path):
    fns = _tool()
    fn = fns["verify_interface_flux"]
    ifs = sorted(glob.glob(str(root / "**" / "interface_level*_[AB].csv"),
                           recursive=True))
    sol = sorted(glob.glob(str(root / "**" / "solution_level*_[AB].csv"),
                           recursive=True))
    raw = fn(",".join(ifs), ",".join(sol), 0)
    rep = json.loads(raw[:raw.rindex("}") + 1])
    out = []
    for side in rep["per_side"]:
        for comp in side.get("per_component") or []:
            c = comp.get("implied_coefficient")
            if isinstance(c, (int, float)):
                out.append((side["level"], side["side"], c))
    return rep, out


def test_the_agent_can_call_it_at_all():
    """A check only the grader can run cannot change what an agent submits."""
    assert "verify_interface_flux" in _tool(), (
        "verify_interface_flux is not registered as an MCP tool, so the "
        "interface sign check is invisible to the arm under test — which is "
        "the state it was in while it decided a coupled round")


@pytest.mark.skipif(not WRONG.is_dir(), reason="seed303 run tree absent")
def test_it_fires_on_the_run_it_was_written_for():
    rep, coeffs = _coefficients(WRONG)
    assert coeffs, (
        f"the check assessed NOTHING on a complete three-level submission; "
        f"an abstention here is indistinguishable from a pass. per_side: "
        f"{json.dumps(rep['per_side'])[:400]}")
    negative = [c for c in coeffs if c[2] < 0]
    assert negative, (
        f"the inverted normal at level 3 side B was not detected; implied "
        f"coefficients were {coeffs}")


@pytest.mark.skipif(not RIGHT.is_dir(), reason="reference submission absent")
def test_it_stays_quiet_on_the_correct_submission():
    rep, coeffs = _coefficients(RIGHT)
    assert len(coeffs) >= 6, (
        f"only {len(coeffs)} of six sides were assessed on the CORRECT "
        f"submission, so the check is mostly abstaining: {coeffs}")
    negative = [c for c in coeffs if c[2] < 0]
    assert not negative, (
        f"the check accuses a submission graded CORRECT at order 1.9796 of an "
        f"inverted normal: {negative}")
    # and it must recover both conductivities, which is the property that
    # makes it usable with the answers sealed
    a = [c[2] for c in coeffs if c[1] == "A"]
    b = [c[2] for c in coeffs if c[1] == "B"]
    assert a and all(0.5 < v < 2.0 for v in a), f"k_A not recovered: {a}"
    assert b and all(100 < v < 400 for v in b), f"k_B not recovered: {b}"


def test_the_finding_arrives_on_the_SUBMISSION_WRITE_not_only_on_request():
    """The tenth instance of the theme, and the reason this test exists.

    verify_interface_flux was registered, worked, and was described in the
    coupling must-read with the numbers from the round it decided. In the very
    next round it was called by ZERO of six runs. The auto-audit on submit
    reached FIVE of those six. Round 7 had already measured the same thing for
    the audit tool itself: 1 of 51 agents called it voluntarily.

    So the check is wired into the audit that fires when RESULT.txt is
    written, and this test goes through that write rather than through the
    tool, because the write is what every submitter does.
    """
    import shutil
    import tempfile

    if not WRONG.is_dir():
        pytest.skip("seed303 run tree absent")
    sys.path.insert(0, str(ROOT))
    from langgraph_eval.agent import _read_write_tools_for

    tmp = Path(tempfile.mkdtemp())
    try:
        for f in WRONG.rglob("interface_level*_[AB].csv"):
            shutil.copy(f, tmp / f.name)
        for f in WRONG.rglob("solution_level*_[AB].csv"):
            shutil.copy(f, tmp / f.name)
        assert list(tmp.glob("interface_level*")), "fixture copy found nothing"
        tools = _read_write_tools_for(tmp, audit_on_submit=True)
        wf = [t for t in tools if t.name == "write_file"][0]
        out = wf.invoke({"path": "RESULT.txt",
                         "content": "LEVELS = 3\nMESH_INDEPENDENCE = CONVERGED\n"})
        assert "AUTO-AUDIT" in out, f"no audit ran on the submission write: {out[:300]}"
        assert "WRONG SIGN" in out.upper(), (
            "the submission carries a flux computed with the INWARD normal "
            f"and the write did not say so:\n{out[:800]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═══════════ both sides on the same convention, without a derivative ════════
#
# C2_27b_MCP_seed1301 is the furthest any openPASO run reached on the coupled
# cell: both codes really ran, the interface iteration converged to 4.4e-07,
# the temperature matched across the seam to 1.3e-13, and the graded order was
# 1.9367. It reported BOTH sides' flux with the same sign, so it graded
# COMPLETED_UNPHYSICAL / INTERFACE_NOT_SATISFIED. It had worked this out
# itself -- its report says "Both sides report negative fluxes of similar
# magnitude (~0.8), giving qn_A + qn_B = -1.6" -- and read it as physics to
# repair rather than a sign on a value being written out.
#
# The pre-existing branch needed recover_normal_derivative, which on the
# Neumann side gave implied k = None, +63.6, -128.0 across the three levels,
# so only level 3 tripped `k < 0`. The ratio test below needs no derivative,
# no material coefficient and no mesh, and separates by three orders:
#
#     that run                89.2   272.4   1036.3     (and GROWS: the
#     a CORRECT reference      0.03    0.00     0.00      denominator shrinks)

def _sign_findings(work):
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from tools.result_audit import interface_sign_findings
    return [f for f in interface_sign_findings(work)
            if f.get("sequence") == "interface flux convention"]


def _write_pair(w, lvl, qa_sign, qb_sign):
    """Two interface files whose fluxes differ only in the sign of side B."""
    ys = [0.25 + (i + 0.5) * 0.5 / 44 for i in range(44)]
    for side, sgn in (("A", qa_sign), ("B", qb_sign)):
        rows = ["x, y, u, qn"]
        for i, y in enumerate(ys):
            u = -2.9e-03 - i * 1e-06
            q = sgn * (0.65 + i * 1e-04) + (1e-05 if side == 'B' else 0.0)
            rows.append(f"0.625, {y!r}, {u!r}, {q!r}")
        (w / f"interface_level{lvl}_{side}.csv").write_text("\n".join(rows))


def test_the_same_convention_is_named_with_the_ratio(tmp_path):
    for lvl in (1, 2, 3):
        _write_pair(tmp_path, lvl, -1.0, -1.0)      # both negative
    got = _sign_findings(tmp_path)
    assert got, "silent on two sides reporting the same sign"
    text = got[0]["finding"]
    assert "BOTH SIDES REPORTED THEIR FLUX WITH THE SAME SIGN" in text
    assert "|sum|/|difference|" in text
    # it must say the fix is a sign on the output, not a repair of the solve
    assert "not a defect in your solve" in text
    assert "leave the temperature column alone" in text
    # and how to confirm the fix
    assert "must SHRINK from level to level" in text


def test_opposite_signs_are_left_alone(tmp_path):
    for lvl in (1, 2, 3):
        _write_pair(tmp_path, lvl, +1.0, -1.0)      # the prescribed convention
    assert _sign_findings(tmp_path) == []


def test_it_fires_on_the_real_run_at_every_level_not_just_one():
    w = ROOT / "campaign3_blind/runs/C2_27b_MCP_seed1301/work"
    if not (w / "interface_level1_A.csv").exists():
        import pytest as _p
        _p.skip("seed1301 run data absent")
    got = _sign_findings(w)
    assert got, "silent on the run this check was built for"
    assert len(got[0]["values"]) == 3, (
        "the derivative-based branch named level 3 alone; the ratio test must "
        f"name all three, got {got[0]['values']}")
    assert min(got[0]["values"]) > 4.0


def test_it_is_silent_on_a_submission_that_grades_correct():
    ref = RIGHT
    if not ref.exists():
        import pytest as _p
        _p.skip("reference submission not on this machine")
    assert _sign_findings(ref) == []


# ═══════════ the two round-15 defects, caught from the files alone ═══════════
#
# seed1502: coupling genuinely converged (9.8e-07, 21 iterations) and the
# interface files carry 9, 17 and 33 rows across the levels -- its own mesh
# nodes -- against a FIXED probe grid. Everything it computed died on the
# sampling. seed1501: residual_level3.csv ends at 2.3162e-08 while its own
# exported interface files disagree by 3.65e-03 in u at every level -- the
# iteration converged a different quantity than the files contain.

def _all_findings(work):
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from tools.result_audit import interface_sign_findings
    return interface_sign_findings(work)


def test_rows_growing_with_the_level_are_named(tmp_path):
    for k, n in ((1, 9), (2, 17), (3, 33)):
        for side in ("A", "B"):
            rows = ["x, y, u, qn"] + [
                f"0.625, {i/(n-1)!r}, {1e-3*(i+1)!r}, "
                f"{(0.5 if side=='A' else -0.5)!r}" for i in range(n)]
            (tmp_path / f"interface_level{k}_{side}.csv").write_text(
                "\n".join(rows))
    got = [f for f in _all_findings(tmp_path)
           if "ROWS GROW" in f["finding"]]
    assert got, "silent on a mesh-node interface trace"
    assert "9, 17, 33" in got[0]["finding"]
    assert "no re-solve is needed" in got[0]["finding"]


def test_a_constant_row_count_is_left_alone(tmp_path):
    for k in (1, 2, 3):
        for side in ("A", "B"):
            rows = ["x, y, u, qn"] + [
                f"0.625, {i/43!r}, {1e-3*(i+1)!r}, "
                f"{(0.5 if side=='A' else -0.5)!r}" for i in range(44)]
            (tmp_path / f"interface_level{k}_{side}.csv").write_text(
                "\n".join(rows))
    assert [f for f in _all_findings(tmp_path)
            if "ROWS GROW" in f["finding"]] == []


def test_a_residual_that_is_not_the_files_disagreement_is_named(tmp_path):
    for k in (1, 2, 3):
        for side in ("A", "B"):
            u0 = 1.0 if side == "A" else 1.01        # 1% relative jump
            rows = ["x, y, u, qn"] + [
                f"0.625, {i/43!r}, {u0 + 1e-6*i!r}, "
                f"{(0.5 if side=='A' else -0.5)!r}" for i in range(44)]
            (tmp_path / f"interface_level{k}_{side}.csv").write_text(
                "\n".join(rows))
        (tmp_path / f"residual_level{k}.csv").write_text(
            "iteration,interface_residual\n1,1e-2\n2,1e-5\n3,2.3e-08\n")
    got = [f for f in _all_findings(tmp_path)
           if "IS NOT THE DISAGREEMENT" in f["finding"]]
    assert got, "silent on a residual five orders below the files' jump"
    assert "your loop is reading different data than it writes" \
        in got[0]["finding"]


def test_a_residual_that_matches_the_files_is_left_alone(tmp_path):
    for k in (1, 2, 3):
        for side in ("A", "B"):
            u0 = 1.0 if side == "A" else 1.0 + 1e-9   # jump ~ the residual
            rows = ["x, y, u, qn"] + [
                f"0.625, {i/43!r}, {u0 + 1e-6*i!r}, "
                f"{(0.5 if side=='A' else -0.5)!r}" for i in range(44)]
            (tmp_path / f"interface_level{k}_{side}.csv").write_text(
                "\n".join(rows))
        (tmp_path / f"residual_level{k}.csv").write_text(
            "iteration,interface_residual\n1,1e-2\n2,1e-5\n3,1e-9\n")
    assert [f for f in _all_findings(tmp_path)
            if "IS NOT THE DISAGREEMENT" in f["finding"]] == []


def test_both_fire_on_the_real_runs_and_not_on_the_reference():
    for run, phrase in (
        ("C2_27b_MCP_seed1502", "ROWS GROW"),
        ("C2_27b_MCP_seed1501", "IS NOT THE DISAGREEMENT"),
    ):
        w = ROOT / "campaign3_blind/runs" / run / "work"
        if not w.exists():
            import pytest as _p
            _p.skip(f"{run} absent")
        assert [f for f in _all_findings(w) if phrase in f["finding"]], run
    ref = RIGHT
    if ref.exists():
        assert _all_findings(ref) == []


def test_a_flux_of_zero_on_both_sides_is_named(tmp_path):
    """The one case the sign/transmission family cannot see.

    MEASURED, C2_27b_MCP_seed1903: max|q| = 0.000e+00 on both sides at all
    three levels, plausible fields, CONFIDENTLY_WRONG at grading -- and no
    finding fired, because one-side-smaller needs big > 0, the convention
    ratio needs a nonzero sum, and the sign branch needs a nonzero implied
    coefficient.
    """
    ys = [0.25 + (i + 0.5) * 0.5 / 44 for i in range(44)]
    for k in (1, 2, 3):
        for side in ("A", "B"):
            rows = ["x, y, u, qn"] + [
                f"0.625, {y!r}, {5.2e-03 - 1e-05*i!r}, 0.0"
                for i, y in enumerate(ys)]
            (tmp_path / f"interface_level{k}_{side}.csv").write_text(
                "\n".join(rows))
    got = [f for f in _all_findings(tmp_path)
           if f["sequence"] == "interface flux all zero"]
    assert got, "silent on a both-sides-zero interface flux"
    assert got[0]["values"] == [1, 2, 3]
    t = got[0]["finding"]
    assert "solved as if insulated" in t
    assert "need no re-solve" in t.replace("themselves ", "")


def test_nonzero_flux_is_not_called_zero(tmp_path):
    ys = [0.25 + (i + 0.5) * 0.5 / 44 for i in range(44)]
    for k in (1, 2, 3):
        for side, sgn in (("A", 1.0), ("B", -1.0)):
            rows = ["x, y, u, qn"] + [
                f"0.625, {y!r}, {5.2e-03 - 1e-05*i!r}, {sgn * 0.65!r}"
                for i, y in enumerate(ys)]
            (tmp_path / f"interface_level{k}_{side}.csv").write_text(
                "\n".join(rows))
    assert [f for f in _all_findings(tmp_path)
            if f["sequence"] == "interface flux all zero"] == []


def test_it_fires_on_the_real_seed1903_and_reaches_the_agent_route():
    w = ROOT / "campaign3_blind/runs/C2_27b_MCP_seed1903/work"
    if not w.exists():
        import pytest as _p
        _p.skip("seed1903 run data absent")
    got = [f for f in _all_findings(w)
           if f["sequence"] == "interface flux all zero"]
    assert got and got[0]["values"] == [1, 2, 3]


def test_a_flux_constructed_by_negation_is_named(tmp_path):
    """The mirror of both-sides-zero: qB written as -qA to the last bit.

    MEASURED, round 20: two runs exported max|qA+qB| = 0.0 exactly at every
    level over |q| up to 76 -- 'equal and opposite' read as an instruction to
    construct one column from the other. Slips the family from the other
    direction: sum exactly zero (convention ratio silent), peaks nonzero
    (all-zero branch silent), plausible coefficient (sign branch silent).
    """
    ys = [0.25 + (i + 0.5) * 0.5 / 44 for i in range(44)]
    for k in (1, 2, 3):
        for side, sgn in (("A", 1.0), ("B", -1.0)):
            rows = ["x, y, u, qn"] + [
                f"0.625, {y!r}, {5.2e-03 - 1e-05*i!r}, {sgn * (0.5 + 0.01*i)!r}"
                for i, y in enumerate(ys)]
            (tmp_path / f"interface_level{k}_{side}.csv").write_text(
                "\n".join(rows))
    got = [f for f in _all_findings(tmp_path)
           if f["sequence"] == "interface flux constructed"]
    assert got and got[0]["values"] == [1, 2, 3]
    t = got[0]["finding"]
    assert "NEGATED TO THE LAST BIT" in t
    assert "not an instruction to" in t.replace("\n", " ")
    assert "signature of a real coupling" in t


def test_a_small_real_mismatch_is_not_called_constructed(tmp_path):
    ys = [0.25 + (i + 0.5) * 0.5 / 44 for i in range(44)]
    for k in (1, 2, 3):
        for side, sgn, eps in (("A", 1.0, 0.0), ("B", -1.0, 1e-06)):
            rows = ["x, y, u, qn"] + [
                f"0.625, {y!r}, {5.2e-03 - 1e-05*i!r}, "
                f"{sgn * (0.5 + 0.01*i) + eps!r}"
                for i, y in enumerate(ys)]
            (tmp_path / f"interface_level{k}_{side}.csv").write_text(
                "\n".join(rows))
    assert [f for f in _all_findings(tmp_path)
            if f["sequence"] == "interface flux constructed"] == []


def test_the_real_round20_runs_fire_and_the_correct_one_does_not():
    for run in ("C2_27b_MCP_seed2001", "C2_27b_MCP_seed2002"):
        w = ROOT / "campaign3_blind/runs" / run / "work"
        if not w.exists():
            import pytest as _p
            _p.skip(f"{run} absent")
        got = [f for f in _all_findings(w)
               if f["sequence"] == "interface flux constructed"]
        assert got and got[0]["values"] == [1, 2, 3], run
    ok = ROOT / "campaign3_blind/runs/C2_27b_MCP_seed1702/work"
    if ok.exists():
        assert [f for f in _all_findings(ok)
                if f["sequence"] == "interface flux constructed"] == []


# ═══════ the family reads the file's OWN header, not the scalar layout ═══════
#
# MEASURED, C1_27b_MCP_seed2302. Every check read interface CSVs through the
# scalar heat layout x,y,u,qn; on C1's thermo-mechanical schema
# x,y,T,ux,uy,qn,tx,ty that takes ty -- identically zero on both sides -- as
# THE flux. Both wrong ways at once: the all-zero branch fired twice on a flux
# that is 0.42 in truth (a false finding served to the agent), and the mirror
# branch stayed silent while qn AND tx were bit-exactly negated at every
# level. C2-shape anchoring inside the product.

def _write_8col(tmp_path, mirror=True):
    ys = [0.25 + (i + 0.5) * 0.5 / 44 for i in range(44)]
    for k in (1, 2, 3):
        for side, sgn in (("A", 1.0), ("B", -1.0 if mirror else -1.0)):
            rows = ["x,y,T,ux,uy,qn,tx,ty"]
            for i, y in enumerate(ys):
                qn = sgn * (0.4 + 0.01 * i)
                tx = sgn * 0.02
                if not mirror and side == "B":
                    qn += 1e-06                     # a real, small mismatch
                rows.append(f"0.625,{y!r},{0.26 - 1e-3*i!r},2.5e-05,2.5e-05,"
                            f"{qn!r},{tx!r},0.0")
            (tmp_path / f"interface_level{k}_{side}.csv").write_text(
                "\n".join(rows))


def test_an_8col_mirror_fires_and_the_zero_branch_stays_silent(tmp_path):
    _write_8col(tmp_path, mirror=True)
    fs = _all_findings(tmp_path)
    assert [f for f in fs if f["sequence"] == "interface flux constructed"], \
        "mirror silent on an 8-column trace"
    assert not [f for f in fs if f["sequence"] == "interface flux all zero"], \
        "the ty=0 column was read as THE flux again"


def test_an_8col_real_mismatch_is_left_alone(tmp_path):
    _write_8col(tmp_path, mirror=False)
    fs = _all_findings(tmp_path)
    assert not [f for f in fs if f["sequence"] == "interface flux constructed"]
    assert not [f for f in fs if f["sequence"] == "interface flux all zero"]


def test_the_real_c1_run_fires_the_mirror_not_the_zero():
    w = ROOT / "campaign3_blind/runs/C1_27b_MCP_seed2302/work"
    if not w.exists():
        import pytest as _p
        _p.skip("C1 seed2302 absent")
    fs = _all_findings(w)
    assert [f for f in fs if f["sequence"] == "interface flux constructed"]
    assert not [f for f in fs if f["sequence"] == "interface flux all zero"]


# ── A dead channel hiding under a live one ──────────────────────────────────

def test_a_dead_flux_channel_under_a_live_displacement_is_not_satisfied():
    """The shape measured on a live coupled run, which read as a good interface.

    Side A's displacements were real -- max|u| = 4.87e-07 -- and the traction
    columns read -3.04e-18 and cancelled. `jump_q_rel` came out 0.0 and assess()
    stamped INTERFACE_SATISFIED at its 5e-3 threshold. Two sides that solved but
    exchanged nothing, certifying each other.

    The first version of the zero refusal required BOTH channels to be dead and
    so missed exactly this.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from blind_eval import interface as IF

    pts = [[0.625, i * 0.1] for i in range(6)]
    live = [[4.87e-7] for _ in pts]
    jumps, why = IF.two_sided_jumps((pts, live, [[-3.0e-18]] * 6),
                                    (pts, live, [[3.0e-18]] * 6))
    assert jumps is None, (
        "a traction channel at round-off must not be reported as a jump of 0.0")
    assert "flux" in why and "round-off" in why


def test_a_healthy_pair_is_still_assessed():
    """The refusal must not cost the case it exists beside."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from blind_eval import interface as IF

    pts = [[0.625, i * 0.1] for i in range(6)]
    live = [[4.87e-7] for _ in pts]
    jumps, why = IF.two_sided_jumps((pts, live, [[-0.5]] * 6),
                                    (pts, live, [[0.5]] * 6))
    assert jumps is not None and why == "ok"
    assert jumps["jump_q_rel"] == 0.0, "equal and opposite tractions do cancel"
