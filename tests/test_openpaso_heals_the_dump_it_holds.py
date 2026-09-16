"""openPASO writes the per-level interface file rather than asking for it again.

HEALING, NOT WARNING, AND THIS IS THE CASE THAT EARNED IT. On a live coupled run
the coupling SUCCEEDED -- three levels, both codes proven, interface residual
6.8e-11, max|u| 6.57e-07 on side A -- and the participants wrote no per-level
dumps. openPASO detected that and said so at the level, with the correct fix.
The agent answered by hand-writing the deliverables:

    # Placeholder: small values based on manufactured solution
    ux = 0.0
    uy = 0.0

Every graded number was invented, AFTER the physics was already right. A warning
is only as good as the cheapest way around it, and hand-writing a file is very
cheap.

So for the one artefact openPASO already holds, it stops asking. The agent never
faces a missing file it might be tempted to invent.

THE LIMITS ARE THE POINT. It writes only the INTERFACE file, from that side's own
exports.json, and only when the participant wrote none. It never writes the
volume field -- openPASO does not have it. It never overwrites. And it invents
nothing: every number came out of the agent's own solver.
"""
import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


@pytest.fixture(scope="module")
def heal():
    from tools.consolidated import _heal_missing_interface_dump
    return _heal_missing_interface_dump


def _side(tmp_path: Path, name: str, values, fluxes) -> dict:
    wd = tmp_path / name
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "exports.json").write_text(json.dumps({
        "field_name": "temperature",
        "n_points": len(values),
        "coordinates": [[0.625, 0.1 * i] for i in range(len(values))],
        "values": values,
        "normal_fluxes": fluxes,
    }))
    return {"name": name, "work_dir": str(wd)}


def test_it_writes_the_missing_interface_file(heal, tmp_path):
    spec = _side(tmp_path, "A", [1.5, 2.5, 3.5], [-0.25, -0.5, -0.75])
    assert heal([spec], 2) == {"A": "interface_level2.csv"}
    rows = list(csv.reader((tmp_path / "A" / "interface_level2.csv").open()))
    assert rows[0] == ["x", "y", "u", "qn"]
    assert len(rows) == 4
    assert float(rows[1][2]) == pytest.approx(1.5)
    assert float(rows[3][3]) == pytest.approx(-0.75)


def test_it_never_overwrites_what_the_participant_wrote(heal, tmp_path):
    spec = _side(tmp_path, "A", [1.0], [2.0])
    own = tmp_path / "A" / "interface_level1.csv"
    own.write_text("x,y,u,qn\nmine\n")
    assert heal([spec], 1) == {}
    assert own.read_text() == "x,y,u,qn\nmine\n"


def test_it_does_nothing_without_the_side_s_own_exports(heal, tmp_path):
    (tmp_path / "A").mkdir()
    assert heal([{"name": "A", "work_dir": str(tmp_path / "A")}], 1) == {}
    assert heal([], 1) == {}
    assert heal([{"name": "A"}], 1) == {}


def test_it_transcribes_and_does_not_invent(heal, tmp_path):
    """Every number must be traceable to the agent's own exports.json."""
    spec = _side(tmp_path, "B", [0.0, 0.0], [0.0, 0.0])
    heal([spec], 1)
    rows = list(csv.reader((tmp_path / "B" / "interface_level1.csv").open()))[1:]
    assert [float(r[2]) for r in rows] == [0.0, 0.0], (
        "a zero field must be transcribed as zero, not smoothed or filled in")


def test_it_never_writes_the_volume_field(heal, tmp_path):
    """openPASO does not have the field, and must not pretend to."""
    spec = _side(tmp_path, "A", [1.0], [1.0])
    heal([spec], 1)
    assert not list((tmp_path / "A").glob("field_level*.csv")), (
        "the volume field is the participant's and cannot be recovered from "
        "interface data")


def test_the_reply_says_what_was_done_and_what_is_still_owed():
    """Read the served sentence, not the source's line wrapping.

    The first version of this matched a phrase the source splits across two
    string literals, so it failed on a file that said exactly the right thing.
    Collapsing whitespace tests the message rather than the formatting.
    """
    import re

    raw = (REPO / "src" / "tools" / "consolidated.py").read_text()
    prose = re.sub(r'"\s*\n\s*"', "", raw)          # join adjacent literals
    prose = re.sub(r"\s+", " ", prose)
    assert "interface_dump_written_for_you" in raw
    assert "Nothing was invented" in prose
    assert "still yours to write" in prose, (
        "the agent must be told the volume field is not covered, or it will "
        "assume the level is complete")
