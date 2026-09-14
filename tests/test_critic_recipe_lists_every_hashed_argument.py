"""The served critic-review recipe must name every argument the digest hashes.

`couple` binds a critic review to a setup by hashing the call's arguments
through `_coupling_setup_text`. The served instructions show the agent which
arguments to put in `submit_critic_review(coupling_args=...)`. If the two lists
disagree, an agent that copies the recipe verbatim and changes NOTHING is told:

    a critic review exists for this solver but NOT for this setup:
    the input changed after it was reviewed

They did not change the input. The recipe was short.

That is what happened: the digest covers participants, max_iter, tol,
accelerator, theta, monolithic and probe, and the served example listed only
the first five. An engineer following the documented procedure hit it three
times in a row before finding the cause by reading consolidated.py. In the
campaign the coupled openPASO arm ends HONEST_INCOMPLETE 63% of the time at a
median 30% of budget, and a spurious NOT VERIFIED is exactly the kind of thing
that stops a run that was going fine.

This test compares the two lists rather than trusting either, so adding a
parameter to the digest fails here until the recipe is updated.
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CONSOLIDATED = REPO / "src" / "tools" / "consolidated.py"
KNOWLEDGE = REPO / "src" / "tools" / "coupling_knowledge.py"

# Arguments of _coupling_setup_text that are NOT part of a `couple` call's
# identity — they belong to the other caller (couple_precice) or are keying
# material rather than user-supplied setup.
NOT_COUPLE_ARGS = {"data", "exchanges"}


def _digest_args() -> set:
    """The parameters `couple` feeds into the setup digest."""
    src = CONSOLIDATED.read_text()
    # the call inside `couple`, i.e. the one passing monolithic/probe
    m = re.search(r"setup_text=_coupling_setup_text\((.*?)\)\)", src, re.S)
    assert m, "could not find couple's _coupling_setup_text call"
    return set(re.findall(r"(\w+)\s*=", m.group(1)))


def _recipe_args() -> set:
    """The keys the served submit_critic_review example tells the agent to pass."""
    src = KNOWLEDGE.read_text()
    # Anchor on the coupling_args JSON itself. A plain prose mention of
    # submit_critic_review appears earlier in the file and carries no keys, so
    # matching on the call name alone finds the wrong block and reports every
    # argument as missing.
    m = re.search(r"coupling_args='\{\{(.*?)\}\}'", src, re.S)
    assert m, "could not find the served coupling_args example"
    return set(re.findall(r'"(\w+)"\s*:', m.group(1)))


def test_the_recipe_lists_every_argument_the_digest_hashes():
    digest, recipe = _digest_args(), _recipe_args()
    missing = digest - recipe - NOT_COUPLE_ARGS
    assert not missing, (
        f"the served critic-review recipe omits {sorted(missing)}, which "
        f"`couple` hashes into the setup digest. An agent that copies the "
        f"recipe verbatim gets NOT VERIFIED with 'the input changed after it "
        f"was reviewed', having changed nothing.")


def test_the_recipe_does_not_invent_arguments():
    """A key the digest does not hash would be equally confusing."""
    digest, recipe = _digest_args(), _recipe_args()
    extra = recipe - digest
    assert not extra, (
        f"the recipe tells the agent to pass {sorted(extra)}, which the "
        f"digest does not hash")


def test_the_recipe_warns_about_the_defaulted_arguments():
    """The two that bite are the ones with defaults, so say so."""
    src = KNOWLEDGE.read_text()
    m = re.search(r"submit_critic_review\(solver=\"couple\".*?Then couple",
                  src, re.S)
    assert m, "recipe block not found"
    block = m.group(0)
    assert "default" in block.lower(), (
        "the recipe does not warn that arguments left at their defaults must "
        "still be declared, which is the whole failure mode")


def test_a_converged_run_is_never_told_it_may_not_have_converged():
    """The verdict text must not hand a working run a reason to stop.

    The reason string was a disjunction — "did not converge, or failed one of
    openPASO's silent-wrong checks" — emitted whenever the checks failed, even
    with converged=True in the same payload.
    """
    src = CONSOLIDATED.read_text()
    bad = "the coupling did not converge, or failed one of openPASO"
    occurrences = [
        ln for i, ln in enumerate(src.splitlines())
        if bad in ln and not ln.lstrip().startswith("#")
    ]
    assert not occurrences, (
        "a converged coupling can still be told it did not converge:\n  "
        + "\n  ".join(occurrences))


def test_the_converged_but_checked_branch_says_it_is_still_a_result():
    """Section 3b's rule must appear where the agent reads the verdict."""
    src = CONSOLIDATED.read_text()
    assert "converged result with a caveat" in src or \
           "NOT a failed run" in src, (
        "the branch that fires on a converged run with a failed downstream "
        "check does not tell the agent the result still counts; that rule "
        "lives ~700 lines away in a different payload")
