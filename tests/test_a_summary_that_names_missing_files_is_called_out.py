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


def test_a_coupling_that_ran_and_wrote_no_field_is_told_so(tmp_path):
    from tools.result_audit import audit, missing_fields_findings
    w = tmp_path / "work"
    for k in (1, 2, 3):
        _write(w / "results", f"residual_level{k}.csv",
               "iteration,interface_residual\n1,0.5\n2,0.1\n3,0.01\n4,0.001\n")
        _write(w / "side_A", f"run_level{k}_A.log", "DUNE-INFO: solving\nNDOF = 100\n")
        _write(w / "side_B", f"run_level{k}_B.log", "4C output\nNDOF = 120\n")
    fs = missing_fields_findings(w)
    assert len(fs) == 1 and "NO PER-LEVEL FIELD FILE" in fs[0]["finding"]
    r = audit(str(w))
    assert any("NO PER-LEVEL FIELD FILE" in f.get("finding", "") for f in r.get("findings", [])), (
        "the audit must name the missing fields on a coupling that ran")
    lead = r.get("what_to_fix_next") or ""
    # only a crashed side, invented names or a log that is not the solver's
    # own output may outrank it (the fixture logs are hand-written, so the
    # identity check is allowed to lead here)
    assert "NO PER-LEVEL FIELD FILE" in lead or "NO LINE ANY SOLVER EMITS" in lead
    assert r.get("sequences_found", 0) == 0     # nothing to verify yet, and it says so


def test_no_finding_once_a_field_file_exists(tmp_path):
    from tools.result_audit import missing_fields_findings
    w = tmp_path / "work"
    _write(w, "residual_level1.csv", "iteration,interface_residual\n1,0.5\n2,0.1\n3,0.01\n")
    _write(w, "solution_level1_A.csv", "x,y,u\n0,0,1\n")
    assert missing_fields_findings(w) == []
