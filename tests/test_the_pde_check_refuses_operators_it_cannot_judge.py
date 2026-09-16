"""A check that answers about an operator it does not implement is not weak.

It is a source of wrong answers pointed at the arm under test.

MEASURED, by handing real submissions to the unguarded tool:

    FC2 (linear elasticity, Lame lambda/mu)
        INCONSISTENT — "your field is converging to something that is not the
        solution of the stated problem", residual 1.197e+297
    SK2 (the biharmonic equation, lap(lap(u)) = f)
        CONSISTENT — "Your field satisfies the equation you were given",
        rate 2.09, residual 5.190e+293

The first tells an agent to discard work that may be right. The second BLESSES
a field on evidence that does not exist, which is worse. Of the campaign's 16
single-code cells, 13 lie outside the implemented form — elasticity, Stokes,
Navier-Stokes, biharmonic, transient heat, a nonlinear a(u), a variable a(x,y)
— and `verify_pde_consistency` is advertised in the universal core, which now
reaches 100% of knowledge calls. FE1, whose coefficient depends on u, was
observed calling it during round 9.

So the tool now requires the equation and refuses anything but the scalar
second-order diffusion form, and the underlying check refuses a residual too
large to be a discretisation's.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _tool():
    """The registered MCP tool, as an agent reaches it."""
    from mcp.server.fastmcp import FastMCP
    from tools import consolidated
    mcp = FastMCP("t")
    consolidated.register(mcp) if hasattr(consolidated, "register") else None
    for name in ("verify_pde_consistency",):
        fn = getattr(consolidated, name, None)
        if fn is not None:
            return fn
    pytest.skip("verify_pde_consistency is not reachable as a plain function")


def _levels(tmp: Path, n=8, scale=1.0):
    """u = scale * x*(1-x)*y*(1-y) on the prescribed midpoint grid."""
    files = []
    for lvl, m in enumerate((22, 44), start=1):
        p = tmp / f"solution_level{lvl}.csv"
        rows = ["x,y,u"]
        for i in range(m):
            for j in range(m):
                x = (i + 0.5) / m
                y = (j + 0.5) / m
                rows.append(f"{x},{y},{scale * x*(1-x)*y*(1-y)}")
        p.write_text("\n".join(rows))
        files.append(str(p))
    return ",".join(files)


def test_a_field_that_satisfies_the_identity_exactly_is_not_condemned():
    """Found by writing the test below, and worth its own name.

    The decay test asks whether the residual FALLS. It has no answer when the
    residual is already zero: an exact field gives 0.000e+00 at every level,
    `last < first/3` is false, and the verdict came out INCONSISTENT with the
    explanation "the weak residual is FLAT: 0.000e+00 -> 0.000e+00" — the one
    field that could not be more right, told it converges to the wrong
    solution.
    """
    import math

    from tools.pde_consistency import check_levels
    lv = {}
    for lvl, m in ((1, 22), (2, 44)):
        lv[lvl] = [[(i + 0.5) / m, (j + 0.5) / m,
                    math.sin(math.pi * (i + 0.5) / m)
                    * math.sin(math.pi * (j + 0.5) / m)]
                   for i in range(m) for j in range(m)]
    out = check_levels(lv, "2*pi**2*sin(pi*x)*sin(pi*y)", 1.0,
                       [(0.0, 1.0), (0.0, 1.0)])
    assert out.verdict == "CONSISTENT", out.explanation
    # THE ROUTE CHANGED WITH THE TEST FUNCTION; THE CONTRACT DID NOT.
    # Under the old v (a product of half-period sines) this exact field made
    # the midpoint quadrature exact too, so the residual was 0 and the
    # round-off branch answered. v is now a product of sin^2 -- needed so BOTH
    # boundary terms of the weak identity vanish, which is what lets the check
    # answer a side whose boundary carries the partner's data -- and with it
    # the quadrature carries its own O(h^2) error. The residual is therefore
    # small, nonzero, and FALLS at the order midpoint quadrature should give.
    # Either route is the check declining to condemn a field that is right,
    # which is what this test exists for.
    assert ("round-off" in out.explanation
            or "falls" in out.explanation), out.explanation
    if "round-off" in out.explanation:
        # it must not overclaim: round-off also describes an interpolated
        # exact field, which proves nothing about a solver having run
        assert "not that a solver ran" in out.explanation
    else:
        assert 1.7 <= (out.rate or 0.0) <= 2.3, (
            f"an exact field should show the quadrature's own order, got {out.rate}")


def test_the_guard_exists_in_the_served_tool():
    """What the AGENT ends up with: the refusal, and what it points to."""
    src = (ROOT / "src" / "tools" / "consolidated.py").read_text()
    # THE GUARD LIVES IN THE SHARED BODY. verify_pde_consistency is now a thin
    # wrapper so that couple() can run the same check rather than a second
    # implementation of it; the operator guard and its refusals moved with the
    # body. Slicing from the tool alone stopped seeing them.
    j = src.index("def _pde_consistency_body(")
    body = src[j:src.index("async def audit_results", j)]
    i = src.index("def verify_pde_consistency(")
    assert "equation: str = \"\"" in src[i:i + 400], (
        "the tool must take the equation; it cannot tell from the numbers "
        "alone whether its own operator applies")
    assert "REFUSED: pass `equation=`" in body
    assert "REFUSED: this check implements -div(K grad u) = f" in body
    assert "_OP = _re.compile(" in body, (
        "the operator is matched by SHAPE — a whitelist of spellings refused "
        "seven cells whose operator this check does implement")
    # it must name the alternative rather than leaving the agent stuck
    assert body.count("audit_results(work_dir=") >= 2, (
        "a refusal that does not say what to use instead costs the run an "
        "action and gives it nothing")
    # the measured evidence must travel with the guard
    assert "2.09" in body and "biharmonic" in body


def test_an_absurd_residual_is_not_reported_as_a_verdict():
    from tools.pde_consistency import check_levels
    # a field of enormous values against a small source: the identity's two
    # sides are not comparable, which is exactly the elasticity/biharmonic case
    # The field must VANISH ON THE BOUNDARY, or the boundary-trace guard fires
    # first and this test stops isolating the residual-magnitude guard it is
    # named for. x(1-x)y(1-y) does; a bare x*y does not.
    lv = {}
    for lvl, m in ((1, 22), (2, 44)):
        rows = []
        for i in range(m):
            for j in range(m):
                x, y = (i + 0.5) / m, (j + 0.5) / m
                rows.append([x, y, 1e150 * x * (1 - x) * y * (1 - y)])
        lv[lvl] = rows
    out = check_levels(lv, "1.0", 1.0, [(0.0, 1.0), (0.0, 1.0)])
    assert out.verdict == "NOT_APPLICABLE", (
        f"a residual outside anything a discretisation produces must not "
        f"become a CONSISTENT or INCONSISTENT verdict; got {out.verdict}")
    assert "REFUSED" in " ".join(
        r.detail for r in out.levels), out.explanation


def test_a_genuine_scalar_diffusion_case_still_passes():
    """The guard must not close the door on the three cells it does serve."""
    from tools.pde_consistency import check_levels
    import math
    # u = sin(pi x) sin(pi y) solves -lap u = 2 pi^2 sin(pi x) sin(pi y)
    lv = {}
    for lvl, m in ((1, 22), (2, 44)):
        rows = []
        for i in range(m):
            for j in range(m):
                x, y = (i + 0.5) / m, (j + 0.5) / m
                rows.append([x, y, math.sin(math.pi * x) * math.sin(math.pi * y)])
        lv[lvl] = rows
    out = check_levels(lv, "2*pi**2*sin(pi*x)*sin(pi*y)", 1.0,
                       [(0.0, 1.0), (0.0, 1.0)])
    assert out.verdict == "CONSISTENT", out.explanation


def test_the_guard_decides_every_cell_in_the_campaign_correctly():
    """All 47 cells, by their own equation strings, not by hand-picked cases.

    The guard went through three wrong versions before this passed, each caught
    only by running it against the real strings:

      * exact-match on a list of spellings refused `-div(k grad u) = f in each
        subdomain` (C2, C8, C10, D1-D6) — the implemented operator applied per
        side;
      * it refused KR2, whose line is `-lap(u) = f  (steady diffusion, unit
        conductivity)`, because of the trailing gloss;
      * matching the operator's shape refused `-div(K grad T) = f` (C1, C6)
        because the field is called T, and then refused KR2 again because on the
        `lap(u)` branch the coefficient group is None rather than "".

    A whitelist of spellings cannot survive contact with 47 task authors. This
    test is the thing that has to stay.
    """
    import glob
    import json
    import re as _re

    src = (ROOT / "src" / "tools" / "consolidated.py").read_text()
    i = src.index("_OP = _re.compile(")
    pat = src[i:src.index(")\n", src.index('r"^-', i))]
    # rebuild the guard's decision from the SHIPPED source, so this test cannot
    # drift away from what an agent actually meets
    rx = _re.compile("".join(_re.findall(r'r"([^"]*)"', pat)))
    qi = src.index("_QUAL = (")
    quals = tuple(s.strip().strip('"') for s in
                  src[qi + 9:src.index(")", qi)].replace("\n", "").split(","))

    def decide(eq):
        raw = _re.sub(r"\s*\([^()]*\)\s*$", "", str(eq)).strip()
        e = "".join(raw.split()).lower()
        if not e:
            return False
        m = rx.match(e)
        return bool(m and (m.group(1) or "") in ("", "k", "a")
                    and (m.group(4) or "") in quals)

    accepted, refused = [], []
    specs = sorted(glob.glob(str(ROOT / "campaign3_blind" / "problems"
                                 / "*" / "spec_public.json")))
    if not specs:
        pytest.skip("the campaign's graded problem set is not in this checkout")
    assert len(specs) >= 40, f"only {len(specs)} cells found"
    for f in specs:
        d = json.loads(Path(f).read_text())
        (accepted if decide(d.get("equation")) else refused).append(d["id"])

    # the operator IS -div(K grad u) = f here, single-code and per-subdomain
    MUST_ACCEPT = {"NG1", "DU1", "KR2", "B1", "C2", "C8", "C10", "C6",
                   "D1", "D2", "D3", "D6"}
    # elasticity, Stokes, Navier-Stokes, biharmonic, transient, nonlinear a(u),
    # variable a(x,y), piecewise-constant k, advection, DSMC, FSI, multiphysics
    MUST_REFUSE = {"FC2", "FE2", "SK1", "SK2", "NG2", "FE1", "DL1", "FC1",
                   "DL2", "KR1", "DU2", "FB1", "FB2", "C5", "D5", "C4", "D8",
                   "C3", "D7", "C7", "C9", "C11", "C12", "D4", "C13", "C14",
                   "SP1", "SP2", "C1", "B2", "B3"}
    assert MUST_ACCEPT <= set(accepted), (
        f"the check is valid on these and would be refused: "
        f"{sorted(MUST_ACCEPT - set(accepted))}")
    assert MUST_REFUSE <= set(refused), (
        f"the check would answer about operators it does not implement: "
        f"{sorted(MUST_REFUSE - set(refused))}")
