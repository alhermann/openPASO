"""The second participant is half the work and got four words.

MEASURED over the graded coupled record. 151 cells end with EXACTLY ONE side
exporting, and in every single one the silent side has no exports.json at all --
it never ran to completion. That is the largest identifiable failure group in
the campaign, and it is not a knowledge gap about any one code: every backend
appears on the silent side (4C 42, dune 38, fenics 25, skfem 19, kratos 19),
including codes that work on the live side of other cells. In 122 of the 151 the
script for the silent code EXISTS -- written, never made to run.

It is positional. Side A survives 117 times against side B's 27, and the first
script written is side A's in 457 of 458 cells that write both.

The served brief is why. `YOUR FIRST SUB-AGENT, NOW` spells side A out in
twenty-odd lines -- which knowledge call, which physics selector, copy the
contract unchanged, fill only the marked hole, write config.json and a synthetic
imports.json, use that code's own interpreter, run until exports.json appears
with finite values, report DONE or the exact error -- and then says, in full:

    Then the same for side B.

Every specific that makes the side-A brief work is absent from the side-B one,
including the two the record says decide it: the physics selector that leads the
reply with the right contract variant, and "that code's OWN interpreter", which
is a different binary for the second code in most pairs.

This test pins that the second side's brief carries the same load-bearing
specifics as the first. It does not pin wording.
"""
import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _served_strings():
    """Every string literal in openPASO that an agent can receive."""
    out = []
    for f in list((ROOT / "src/tools").glob("*.py")) + [ROOT / "src/core/instructions.py"]:
        try:
            tree = ast.parse(f.read_text())
        except (OSError, SyntaxError):
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                out.append(n.value)
    return out


def _first_subagent_block():
    for s in _served_strings():
        if "YOUR FIRST SUB-AGENT" in s:
            return s
    raise AssertionError("the first-sub-agent brief is gone — re-point this test")


def _side_b_brief(block):
    """The text from where side B is introduced to the end of that passage."""
    m = re.search(r"[^\n]*side B[^\n]*", block)
    assert m, "the brief never mentions side B at all"
    return block[m.start():m.start() + 2000]


# The specifics the side-A brief carries. Each is load-bearing: the knowledge
# call is how the contract arrives, the physics selector decides WHICH contract
# variant leads the reply, the interpreter is a different binary per code, and
# exports.json is the check that ends the step.
_LOAD_BEARING = {
    "the knowledge call": r"knowledge\(",
    "the physics selector": r"physics=",
    "that code's own interpreter": r"interpreter",
    "the exports.json check": r"exports\.json",
}


def test_the_second_side_is_not_just_told_to_do_the_same_again():
    brief = _side_b_brief(_first_subagent_block())
    missing = [name for name, pat in _LOAD_BEARING.items()
               if not re.search(pat, brief, re.I)]
    assert not missing, (
        "the side-B brief omits what makes the side-A brief work: "
        + ", ".join(missing)
        + ". 151 cells end with one side exporting and the other never "
          "running; the silent side is side B 117 times out of 144.")


def test_the_second_side_brief_is_not_a_single_sentence():
    brief = _side_b_brief(_first_subagent_block())
    # "Then the same for side B." is 25 characters. The side-A brief is ~1200.
    assert len(brief.split("\n\n")[0]) > 200, (
        "the second participant is half the work and gets one sentence")


def test_the_first_side_brief_still_carries_its_own_specifics():
    # Guard against fixing B by gutting A.
    block = _first_subagent_block()
    head = block[:block.index("side B")] if "side B" in block else block
    for name, pat in _LOAD_BEARING.items():
        assert re.search(pat, head, re.I), f"the side-A brief lost {name}"
