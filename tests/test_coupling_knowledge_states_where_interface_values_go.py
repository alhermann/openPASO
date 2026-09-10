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


def test_it_survives_the_truncation(coupling_payload):
    """The coupling payload is cut at a budget; this must be above the cut."""
    from tools.consolidated import _COUPLING_HEAD_LIMIT
    at = coupling_payload.find("INTERFACE FILE IS WRITTEN AT THE POINTS")
    assert 0 <= at < _COUPLING_HEAD_LIMIT, (
        f"the rule sits at character {at}, at or beyond the "
        f"{_COUPLING_HEAD_LIMIT}-character head limit, so a truncated payload "
        f"drops the very thing it was added for"
    )


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
    """Agents reach coupling knowledge by more than one call shape."""
    missing = []
    for kwargs in ({"topic": "coupling"},
                   {"topic": "coupling", "solver": "fourc"},
                   {"topic": "coupling", "signal": "interface residual will not fall"},
                   {"topic": "conjugate_heat_transfer"}):
        out = knowledge_tool(**kwargs)
        if not isinstance(out, str):
            continue
        if "INTERFACE FILE IS WRITTEN AT THE POINTS" not in out:
            missing.append(kwargs)
    assert not missing, (
        f"these coupling requests come back without the interface rule: {missing}"
    )
