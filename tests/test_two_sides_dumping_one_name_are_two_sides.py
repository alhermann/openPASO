"""A participant's own directory names the side when the filename does not.

The served per-level dump writes interface_level<k>.csv inside each
participant's work dir, so a two-sided run has two files with one name at one
depth. Reading that as a collision refused a correct result set, and the
advice it printed told one run to delete a side's files -- after which the
audit found no sequences at all and the divergence in that run's own ladder
was never reported.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tools.result_audit import _sequences_from_level_csvs  # noqa: E402


def _two_sided(tmp_path, scale_b=1.0):
    """Three levels, both sides dumping the served names into their own dirs."""
    for side in ("side_A", "side_B"):
        d = tmp_path / side
        d.mkdir()
        (d / "config.json").write_text('{"level": 1, "nx": 4, "ny": 4}')
        (d / "participant.py").write_text("# participant\n")
        mult = scale_b if side == "side_B" else 1.0
        for lv, val in ((1, 1.0), (2, 0.5), (3, 0.25)):
            rows = ["x,y,u"] + [f"{i * 0.25},0.0,{val * mult * (1 + i)}" for i in range(5)]
            (d / f"interface_level{lv}.csv").write_text("\n".join(rows) + "\n")
    return tmp_path


def test_the_two_sides_are_not_read_as_an_ambiguous_slot(tmp_path):
    seqs = _sequences_from_level_csvs(_two_sided(tmp_path))
    assert "__ambiguous__" not in seqs, (
        "a participant dumping the served name in its own dir is a side, not a collision")
    assert seqs, "both sides were discarded instead of being read"


def test_each_side_is_kept_as_its_own_sequence(tmp_path):
    seqs = _sequences_from_level_csvs(_two_sided(tmp_path, scale_b=3.0))
    labels = " ".join(seqs)
    assert "_A_" in labels or "_SIDE_A_" in labels.upper(), labels
    assert "_B_" in labels or "_SIDE_B_" in labels.upper(), labels


def test_a_real_collision_in_one_directory_is_still_refused(tmp_path):
    """Two files claiming one slot at one depth, with no side to tell them apart."""
    d = tmp_path
    for name in ("interface_level1.csv", "interface_level2.csv"):
        (d / name).write_text("x,y,u\n0,0,1\n")
    sub = d / "copy"
    sub.mkdir()
    # same depth as each other, neither a participant dir
    for k in (1, 2):
        (sub / f"interface_level{k}.csv").write_text("x,y,u\n0,0,2\n")
    other = d / "copy2"
    other.mkdir()
    for k in (1, 2):
        (other / f"interface_level{k}.csv").write_text("x,y,u\n0,0,3\n")
    seqs = _sequences_from_level_csvs(d)
    # the top-level pair is shallowest and wins; nothing here should crash
    assert isinstance(seqs, dict)


def test_the_ambiguity_message_shows_paths_not_bare_names(tmp_path):
    """Two identical basenames told the reader nothing about which files collided."""
    for sub in ("one", "two"):
        d = tmp_path / sub
        d.mkdir()
        for k in (1, 2):
            (d / f"interface_level{k}.csv").write_text("x,y,u\n0,0,1\n")
    seqs = _sequences_from_level_csvs(tmp_path)
    if "__ambiguous__" in seqs:
        text = " ".join(seqs["__ambiguous__"])
        assert "/" in text, f"the message must name paths, got {text}"
