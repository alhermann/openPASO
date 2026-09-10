"""OASiS hands the agent node data, and 1 in 7 OASiS coupled runs submits it.

MEASURED across every coupled run in the tree: 11 submissions wrote an interface
file containing a coordinate the task explicitly excludes, and 10 of the 11 are
OASiS-arm runs — 13.7% of OASiS coupled runs against 1.2% of bare ones. Nine of
them wrote exactly 9 rows, which is the level-1 mesh nodes at h = 1/8.

The cause is the coupling contract itself. Each participant writes exports.json
with `coordinates`, `values` and `normal_fluxes` at its interface NODES, because
that is what the exchange needs; the deliverable then asks for a fixed list of
points that are deliberately NOT nodes, and copying the file across is one line.
C4_27b_MCP_seed5's submitted interface_level1_A.csv is exports.json verbatim,
down to the float noise -2.6927850894701087e-18 where a node sits at y = 0.

The bare arm cannot make this mistake, because it has no such file. So this is
a harm OASiS causes, and the warning belongs in the reply that accompanies the
data — the corpus already says it ~700 lines away in a different payload, which
is not where the agent is when it decides.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SRC = (ROOT / "src" / "tools" / "consolidated.py").read_text()


def _couple_body() -> str:
    i = SRC.index("async def couple(participants: str")
    j = SRC.index("async def couple_precice(", i)
    return SRC[i:j]


def test_couple_returns_the_warning():
    body = _couple_body()
    assert 'result["reporting"]' in body, (
        "couple()'s reply does not carry the reporting rule; the agent reads "
        "this reply while holding the node data"
    )
    assert "THE POINTS EXCHANGED ABOVE ARE NOT THE POINTS YOU REPORT" in body


def test_it_names_the_file_that_gets_copied():
    body = _couple_body()
    assert "exports.json" in body.split('result["reporting"]')[1][:1200]
    assert "per-level interface file" in body


def test_it_says_to_interpolate_with_each_sides_own_material():
    seg = _couple_body().split('result["reporting"]')[1][:1400]
    assert "interpolate" in seg
    assert "own material" in seg and "outward normal" in seg


def test_it_warns_against_completing_the_list_with_the_ends():
    seg = _couple_body().split('result["reporting"]')[1][:1600]
    assert "excluded the interface ENDS on purpose" in seg


def test_the_warning_is_on_the_success_path_not_only_on_errors():
    """A run that COUPLED is exactly the one that has tempting data."""
    body = _couple_body()
    idx_warn = body.index('result["reporting"]')
    idx_final = body.index("return json.dumps(result, indent=2)", idx_warn)
    assert idx_final > idx_warn, (
        "the warning must be attached before the successful return"
    )


def test_the_measured_submission_really_is_a_copy_of_the_exports():
    """The premise, re-measured rather than trusted."""
    import csv
    import json
    run = ROOT / "campaign3_blind" / "runs" / "C4_27b_MCP_seed5" / "work"
    exp = run / "level1_A" / "exports.json"
    sub = next(iter(run.rglob("interface_level1_A.csv")), None)
    if not (exp.is_file() and sub):
        import pytest
        pytest.skip("C4_27b_MCP_seed5 not present")
    coords = json.loads(exp.read_text())["coordinates"]
    rows = []
    with sub.open() as fh:
        for r in csv.reader(fh):
            try:
                rows.append((float(r[0]), float(r[1])))
            except (ValueError, IndexError):
                continue
    assert len(rows) == len(coords)
    for (x, y), c in zip(rows, coords):
        assert abs(x - c[0]) < 1e-12 and abs(y - c[1]) < 1e-12, (
            "the submission is no longer a copy of exports.json; re-check the "
            "premise of this test before trusting its conclusion"
        )
