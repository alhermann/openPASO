"""Cutting the solve out must not leave the agent guessing what filled the hole.

Option B removes the finite-element solve from every served participant, which
is right: the solve is the agent's work. But the surviving code still indexes
`iface_dofs`, assembles `a` and writes `uh`, and nothing told the agent which
names the hole had been defining, or how many.

Measured on the served skfem participant before this: 22 names are used and
never defined anywhere in the payload, and not one is named as the reader's
responsibility. The consequence shows in the runs — 73% of the coupled openPASO
runs that gave up never got both sides to exchange interface data even once,
dying in a write-run-error-rewrite loop on the participant script, 205
rewrites and 51 in-place patches inside the final fifteen tool calls alone.

The contract is DERIVED from the same markers that perform the elision, so it
cannot drift from what was actually cut. It leaks no physics: these are
variable names the surviving code already mentions. What it removes is the
reverse-engineering step, not the solving.

These tests ask what the agent RECEIVES, not what the function returns.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tools.coupling_knowledge import (  # noqa: E402
    _SOLVE_BEGIN,
    _elide_solve,
    _reconstruction_contract,
)

PARTICIPANTS = sorted(
    (REPO / "data" / "coupling_participants").glob("participant_*.py"))

MARKER = "WHAT YOUR SOLVE MUST LEAVE BEHIND"


def _with_elision():
    return [p for p in PARTICIPANTS if _SOLVE_BEGIN in p.read_text()]


def test_there_are_participants_to_check():
    assert _with_elision(), (
        "no participant carries a solve-elision marker; this test is not "
        "guarding anything")


@pytest.mark.parametrize("path", _with_elision(), ids=lambda p: p.name)
def test_every_elided_participant_states_its_contract(path):
    served = _elide_solve(path.read_text())
    assert MARKER in served, (
        f"{path.name} has its solve cut out and never says which names the "
        f"cut removed; the agent must reverse-engineer them from code with a "
        f"hole in it")


@pytest.mark.parametrize("path", _with_elision(), ids=lambda p: p.name)
def test_the_contract_names_exactly_what_was_cut(path):
    original = path.read_text()
    served = _elide_solve(original)
    listed = set(re.findall(r"^#     (\w+)$", served, re.M))
    computed = set(_reconstruction_contract(served, original))
    assert listed == computed, (
        f"{path.name}: the printed contract {sorted(listed)} does not match "
        f"the names the elision actually removed {sorted(computed)}")
    assert listed, f"{path.name}: the contract is empty"


@pytest.mark.parametrize("path", _with_elision(), ids=lambda p: p.name)
def test_the_contract_serves_no_solve(path):
    """It may name variables. It may not hand back the solve."""
    served = _elide_solve(path.read_text())
    block = served[served.index(MARKER):]
    forbidden = ("solve(", "assemble(", "LinearProblem", "spsolve",
                 "factorized(", "splu(")
    leaked = [f for f in forbidden if f in block]
    assert not leaked, (
        f"{path.name}: the contract block contains {leaked} — it is supposed "
        f"to name what the hole defined, not refill it")


@pytest.mark.parametrize("path", _with_elision(), ids=lambda p: p.name)
def test_the_solve_is_still_gone(path):
    """The contract must not have quietly undone the elision."""
    original = path.read_text()
    served = _elide_solve(original)
    assert _SOLVE_BEGIN not in served
    # ONLY the marked regions, not everything after the first marker — the
    # code between the end of one region and the start of the next is meant
    # to survive, and including it compares the survivor against itself.
    from tools.coupling_knowledge import _SOLVE_END

    cut, i = [], 0
    while True:
        a = original.find(_SOLVE_BEGIN, i)
        if a < 0:
            break
        b = original.find(_SOLVE_END, a)
        cut.append(original[a:b if b > 0 else len(original)])
        if b < 0:
            break
        i = b + len(_SOLVE_END)
    bodies = [ln.strip() for ln in "\n".join(cut).splitlines()
              if ln.strip() and not ln.strip().startswith("#")]
    # A line that ALSO appears outside the marked regions is surviving code
    # that happens to be repeated inside them — e.g. the same mesh-unpacking
    # line used by both halves. Only a line found nowhere but inside the cut
    # is evidence the elision failed.
    outside = set()
    rest, i = original, 0
    kept = []
    while True:
        a = rest.find(_SOLVE_BEGIN, i)
        if a < 0:
            kept.append(rest[i:])
            break
        b = rest.find(_SOLVE_END, a)
        kept.append(rest[i:a])
        if b < 0:
            break
        i = b + len(_SOLVE_END)
    for ln in "\n".join(kept).splitlines():
        if ln.strip():
            outside.add(ln.strip())
    leaked = [ln for ln in bodies
              if len(ln) > 25 and ln in served and ln not in outside]
    assert not leaked, (
        f"{path.name}: {len(leaked)} substantive line(s) of the elided solve "
        f"are back in the served text, e.g. {leaked[:2]}")


def test_the_contract_survives_the_real_serving_door():
    """Not the helper — the payload an agent actually gets."""
    import core.registry as R
    from tools.consolidated import _get_coupling_knowledge

    R.load_all_backends()
    text = str(_get_coupling_knowledge(solver="fenics"))
    assert MARKER in text, (
        "the contract is produced by _elide_solve but does not survive into "
        "the payload knowledge(topic='coupling', solver='fenics') returns — "
        "which is the only thing the agent reads")
