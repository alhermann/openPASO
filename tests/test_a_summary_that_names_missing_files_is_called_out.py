"""Test to add under tests/: a summary that names files it never wrote is called out."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write(d: Path, name: str, text: str) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(text)


def test_a_summary_that_names_missing_files_is_called_out_first(tmp_path):
    from tools.result_audit import audit, summary_names_findings
    w = tmp_path / "work"
    _write(w / "results", "residual_level1.csv", "iteration,interface_residual\n1,0.5\n2,0.1\n3,0.01\n")
    _write(w, "RESULT.txt", "LEVELS = 3\nFILES = solution_level1_A.csv, interface_level1_A.csv, residual_level1.csv\n")
    fs = summary_names_findings(w)
    assert len(fs) == 1
    f = fs[0]["finding"]
    assert "solution_level1_A.csv" in f and "interface_level1_A.csv" in f
    assert "residual_level1.csv" not in f.split("anywhere under")[1].split(".")[0], "an existing file was reported missing"
    r = audit(str(w), summary_path=str(w / "RESULT.txt"))
    assert "DO NOT EXIST" in (r.get("what_to_fix_next") or ""), (
        "the missing-names finding must lead the hand-in reply")


def test_a_summary_whose_files_all_exist_is_silent(tmp_path):
    from tools.result_audit import summary_names_findings
    w = tmp_path / "work"
    _write(w, "solution_level1.csv", "x,y,u\n0,0,1\n")
    _write(w, "RESULT.txt", "LEVELS = 1\nFILES = solution_level1.csv\n")
    assert summary_names_findings(w) == []


def test_it_never_reads_anything_but_the_agents_own_files(tmp_path):
    from tools.result_audit import summary_names_findings
    w = tmp_path / "work"
    _write(w, "RESULT.txt", "see https://example.org/data.csv and notes.txt\n")
    fs = summary_names_findings(w)
    assert fs and "notes.txt" in fs[0]["finding"] and "example.org" not in fs[0]["finding"]
