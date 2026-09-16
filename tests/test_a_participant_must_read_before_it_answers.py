"""One side of a partitioned iteration has to depend on the other.

A participant that exports without importing is a fixed point at the first
step, so the run reads as a coupling that converged immediately rather than
one that never started -- and it makes the partner look unresponsive.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tools.participant_lint import export_without_import  # noqa: E402

SERVED = sorted((ROOT / "data" / "coupling_participants").glob("*.py"))


@pytest.mark.parametrize("path", SERVED, ids=lambda p: p.name)
def test_every_served_contract_reads_before_it_answers(path):
    assert export_without_import(path.read_text()) == "", (
        f"{path.name} is served and must not be flagged by its own lint")


def test_an_export_with_no_import_is_named():
    src = ('import json\n'
           'Q = compute()\n'
           'json.dump({"normal_fluxes": Q}, open("exports.json", "w"))\n')
    msg = export_without_import(src)
    assert "never reads imports.json" in msg
    assert "fixed point" in msg, "the message must explain why it looks converged"


def test_reading_the_file_directly_counts():
    src = ('import json\n'
           'imp = json.loads(open("imports.json").read() or "{}")\n'
           'json.dump({"values": v}, open("exports.json", "w"))\n')
    assert export_without_import(src) == ""


def test_the_served_helper_counts_too():
    src = ('imp = read_imports()\n'
           'with open("exports.json", "w") as f:\n'
           '    f.write("{}")\n')
    assert export_without_import(src) == ""


def test_a_script_that_exports_nothing_is_not_a_participant():
    assert export_without_import("print('hello')\n") == ""
