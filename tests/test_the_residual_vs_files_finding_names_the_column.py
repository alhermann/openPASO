"""When the converged residual and the delivered interface files disagree, the finding names WHICH column
differs and whether one side's column is a copy of another column of the same file. Measured (round 42,
C1 7072): T disagreed by a factor 40 while ux and uy agreed to 1e-6, because side A's T column was its
own uy column; the parent read 'field jump 6.2e+01' as a physics error and stopped with 13 minutes left."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write(work: Path):
    ys = [0.1 * i for i in range(1, 10)]
    T = [0.05 + 0.01 * y for y in ys]; ux = [1e-4 * y for y in ys]; uy = [-1e-3 * (1 - y) for y in ys]
    hdr = "x,y,T,ux,uy,qn,tx,ty\n"
    a = hdr + "".join(f"0.625,{y},{uy_},{ux_},{uy_},0.1,0.2,0.3\n" for y, ux_, uy_ in zip(ys, ux, uy))       # A: T column = uy column
    b = hdr + "".join(f"0.625,{y},{T_},{ux_},{uy_},-0.1,-0.2,-0.3\n" for y, T_, ux_, uy_ in zip(ys, T, ux, uy))
    (work / "interface_level1_A.csv").write_text(a); (work / "interface_level1_B.csv").write_text(b)
    (work / "residual_level1.csv").write_text("iteration,residual\n" + "".join(f"{i},{0.5 * 0.3 ** i:.3e}\n" for i in range(1, 15)))
    (work / "solution_level1_A.csv").write_text("x,y,T,ux,uy\n0.1,0.1,0.01,0.0,0.0\n")
    (work / "solution_level1_B.csv").write_text("x,y,T,ux,uy\n1.0,0.1,0.01,0.0,0.0\n")


def test_the_column_and_the_copy_are_named(tmp_path):
    from tools.result_audit import audit
    _write(tmp_path)
    fs = [f for f in audit(tmp_path).get("findings", []) if "IS NOT THE DISAGREEMENT" in str(f.get("finding", ""))]
    assert fs, "the residual-vs-files finding did not fire"
    text = fs[0]["finding"]
    assert "COLUMN BY COLUMN:" in text and "The disagreement is in T alone while ux, uy agree" in text, text
    assert "In side A's file the T column equals its uy column at every row" in text, text
    assert "fix the column index in the script that writes this file, not the coupling" in text
