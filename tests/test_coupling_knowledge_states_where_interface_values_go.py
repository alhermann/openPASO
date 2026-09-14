"""A coupled run can prove every execution and still be unusable.

MEASURED on C2_27b_MCP_seed73. The grader found 4C PROVEN (10 files of
structured solver output), Kratos PROVEN (6 files), the coupling PROVEN and not
forged, and a residual falling 0.309 -> 8.2e-07 over 20 iterations with a
healthy wandering ratio. It was still scored MALFORMED_SUBMISSION, for exactly
one reason: interface_level<k>_A.csv held the agent's own interface MESH NODES
(7 rows at level 1, 15 at level 2, 23 at level 3) instead of the fixed list of
points the task names, which is the same list at every level.

The same runs wrote the SOLUTION file correctly — 1936 prescribed probe rows at
every level. So the agents had understood "fixed probe points" for one file and
not for the other, and the served coupling knowledge said nothing about it: zero
occurrences of "interface probe", "probe point", "mesh node" or "interpolat" in
a 2,746-character must-read.

An order of convergence compares the SAME quantity across levels. Node sets move
and multiply under refinement, so there is no common point to compare and no
partial credit for an iteration that converged beautifully.

The rule is stated in general terms on purpose: no point count, no coordinate,
no interface location. Knowledge that carries a problem's dimensions teaches the
benchmark instead of the method, and the campaign's own rule is that a primitive
must hold for any coupled problem.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class _CollectingMCP:
    def __init__(self):
        self.tools: dict[str, object] = {}

    def tool(self, *a, **kw):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


@pytest.fixture(scope="module")
def knowledge_tool():
    os.chdir(ROOT)
    from core.registry import load_all_backends
    load_all_backends()
    from tools.consolidated import register_consolidated_tools
    mcp = _CollectingMCP()
    register_consolidated_tools(mcp)
    return mcp.tools["knowledge"]


@pytest.fixture(scope="module")
def coupling_payload(knowledge_tool):
    return knowledge_tool(topic="coupling")


def test_the_rule_is_served_at_all(coupling_payload):
    assert "INTERFACE FILE IS WRITTEN AT THE POINTS THE TASK LISTS" in coupling_payload, (
        "the coupling knowledge does not tell the agent where interface values "
        "go — the one defect that cost a fully PROVEN run its grade"
    )


def test_it_says_not_the_mesh_nodes(coupling_payload):
    low = coupling_payload.lower()
    assert "mesh nodes" in low
    assert "nearest node" in low, (
        "the rule must say to interpolate rather than take the nearest node; "
        "nearest-node evaluation is a measured order-1 failure mode"
    )


def test_it_survives_the_truncation(knowledge_tool):
    """The rule must REACH THE AGENT, for every solver, however the text grows.

    This used to assert the rule's POSITION: that it sat inside the first
    _COUPLING_HEAD_LIMIT characters. That was a proxy for the thing that
    matters, and the proxy went stale -- the rule drifted to character 32,322
    of a 28,000-character head budget and the test failed, while the rule was
    in fact still reaching every agent. It lives in the must-read block, which
    is appended whole AFTER the head, so the head cut never touches it.

    Asserting the outcome instead is strictly stronger: it holds for all nine
    solvers, and it keeps holding wherever in the corpus the rule ends up.

    The must-read is served once per session by design -- the first coupling
    call gets it whole, later calls get a pointer -- so each solver is asked as
    the first call of its own session, which is how an agent meets it.
    """
    from tools import consolidated as C

    rule = "INTERFACE FILE IS WRITTEN AT THE POINTS THE TASK LISTS"
    missing = []
    for solver in ("fourc", "fenics", "dealii", "ngsolve", "skfem",
                   "kratos", "dune", "febio", "sparta"):
        C._MUST_READ_STATE["served"] = False
        served = knowledge_tool(topic="coupling", solver=solver)
        if rule not in served:
            missing.append(f"{solver} ({len(served):,} chars)")
    assert not missing, (
        "the rule that decides whether a converged coupled run counts does not "
        "reach the agent for: " + ", ".join(missing))


def test_adding_it_did_not_evict_the_rest_of_the_must_read(coupling_payload):
    """The must-read is prepended whole, so growing it shrinks the corpus head.

    Everything the must-read exists to say has to still arrive.
    """
    for mark in ("couple(participants=",              # how to run the iteration
                 "history_path",                      # where the history goes
                 "DETECTED and read as invented",  # the forgery rules
                 "iterations   11     37    66",      # the rho budget table
                 "SWAP WHICH SIDE IS DIRICHLET"):     # the high-rho remedy
        assert mark in coupling_payload, (
            f"{mark!r} was pushed out of the coupling payload; the interface "
            f"rule must not be added by evicting another must-read item"
        )


def test_the_rule_carries_no_problem_dimensions(coupling_payload):
    """General primitive, not this benchmark's numbers.

    Anchoring knowledge to a drawn problem's coordinates would teach the
    benchmark. The rule must work for any interface, any point count.
    """
    start = coupling_payload.find("INTERFACE FILE IS WRITTEN AT THE POINTS")
    section = coupling_payload[start:start + 2600]
    banned = {
        "5/8": "the C2 interface location",
        "0.625": "the C2 interface location",
        "22 points": "the C2 interface point count",
        "j = 11": "the C2 index range",
        "1936": "the C2 solution probe count",
        "(0.25, 0.75)": "the C2 graded band",
    }
    found = [f"{tok} ({why})" for tok, why in banned.items() if tok in section]
    assert not found, (
        "the interface rule quotes this problem's own dimensions: "
        + "; ".join(found)
        + " — that anchors the knowledge to the drawn instance"
    )


def test_the_measured_counts_are_illustrative_not_prescriptive(coupling_payload):
    """7/15/23 appear as evidence of the failure, not as an instruction.

    They are what a wrong submission contained. Keep them attached to the
    measurement so no reader mistakes them for the required shape.
    """
    start = coupling_payload.find("INTERFACE FILE IS WRITTEN AT THE POINTS")
    section = coupling_payload[start:start + 2600]
    if "7 rows" in section:
        assert "what was written" in section, (
            "the row counts must stay labelled as what a REJECTED submission "
            "wrote, or they read as the required number of rows"
        )


def test_a_coupled_request_by_any_route_gets_it(knowledge_tool):
    """Agents reach coupling knowledge by more than one call shape.

    The must-read (which carries the interface rule) is served ONCE per
    session -- measured, a small model was handed five copies in seven calls
    and drowned -- so the FIRST coupled reply carries the rule and every later
    one carries the pointer that names how to get it back."""
    from tools import consolidated as _C
    _C._MUST_READ_STATE["served"] = False
    first = knowledge_tool(topic="coupling")
    assert "INTERFACE FILE IS WRITTEN AT THE POINTS" in first
    missing = []
    for kwargs in ({"topic": "coupling", "solver": "fourc"},
                   {"topic": "coupling", "signal": "interface residual will not fall"},
                   {"topic": "conjugate_heat_transfer"}):
        out = knowledge_tool(**kwargs)
        if not isinstance(out, str):
            continue
        if ("INTERFACE FILE IS WRITTEN AT THE POINTS" not in out
                and "signal='must-read'" not in out):
            missing.append(kwargs)
    assert not missing, (
        f"these coupling requests come back with neither the interface rule "
        f"nor the pointer to it: {missing}"
    )
    again = knowledge_tool(topic="coupling", signal="must-read")
    assert "INTERFACE FILE IS WRITTEN AT THE POINTS" in again, (
        "signal='must-read' must bring the must-read back")
