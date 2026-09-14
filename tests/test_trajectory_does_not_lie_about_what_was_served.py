"""The trajectory must not read as if openPASO served less than it did.

The trajectory is the only record of what an agent was told, and it is what
the development loop greps to answer "did the primitive reach the agent". It
truncated tool results at 600 characters with no marker, so a payload of
30,099 characters was written as ~600 and a grep for anything past that
returned nothing -- indistinguishable from never having been served.

That produced a wrong finding on 2026-08-30: FB2_27b_MCP_seed15 was reported
as never having consulted a newly added primitive, because the primitive's
distinctive strings were absent from its trajectory. The run had called
prepare_simulation(febio, viscoelasticity) and been served the whole payload.
The conclusion came from the truncation, not from the run.

Same shape as the other defects this campaign keeps finding: the mechanism
(a trajectory log for post-hoc analysis) exists, is instrumented, and does not
reach the case it was built for (recording what was actually served).

These tests ask what a READER of the log can conclude, not what the writer
does.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUN_BLIND = REPO / "campaign3_blind" / "run_blind.py"


def _tracer_class():
    """Load the trajectory writer without importing the whole runner.

    run_blind.py pulls in langchain at import time, which is not present in
    every environment the suite runs in, so the class is compiled out of the
    source instead.
    """
    lines = RUN_BLIND.read_text().splitlines()
    first = next(i for i, ln in enumerate(lines)
                 if ln.startswith("    def __init__(self, path):"))
    head = max(i for i, ln in enumerate(lines[:first])
               if ln.startswith("class "))
    # take the class and nothing after it: stop at the first line that is
    # neither blank nor indented
    end = head + 1
    while end < len(lines) and (not lines[end].strip()
                                or lines[end].startswith((" ", "\t"))):
        end += 1
    body = "\n".join(lines[head:end])
    # The class inherits langchain's BaseCallbackHandler, which need not be
    # installed to test what the writer puts in the file. Any base with the
    # two hooks as no-ops will do.
    class _Base:
        def on_tool_start(self, *a, **k): pass
        def on_tool_end(self, *a, **k): pass

    import json as _json
    import os as _os
    import time as _time
    ns: dict = {"BaseCallbackHandler": _Base, "sys": sys, "os": _os,
                "time": _time, "json": _json, "Path": Path}
    exec(compile(body, "tracer", "exec"), ns)
    name = lines[head].split("class ", 1)[1].split("(")[0].split(":")[0].strip()
    return ns[name]


@pytest.fixture
def tracer(tmp_path):
    cls = _tracer_class()
    return cls(tmp_path / "trajectory.txt"), tmp_path / "trajectory.txt"


def test_a_truncated_result_says_so(tracer):
    """A reader must be able to tell a stub from a complete record."""
    t, path = tracer
    payload = "PRIMITIVE_MARKER " + ("x" * 40000)
    t.on_tool_end(payload)
    text = path.read_text()
    assert "TRUNCATED BY THE LOGGER" in text, (
        "the log dropped 40k characters without saying so; a grep that finds "
        "nothing cannot be told apart from a payload that was never served")
    assert str(len(payload)) in text, "the served size is not recorded"


def test_the_marker_warns_against_the_wrong_inference(tracer):
    """The exact inference that produced a false finding."""
    t, path = tracer
    t.on_tool_end("y" * 5000)
    assert "NOT evidence it was not served" in path.read_text()


def test_a_short_result_is_recorded_whole_and_unmarked(tracer):
    """No marker where nothing was dropped, or the marker means nothing."""
    t, path = tracer
    t.on_tool_end("a short result")
    text = path.read_text()
    assert "a short result" in text
    assert "TRUNCATED" not in text


def test_the_served_size_is_always_recorded(tracer):
    """served_bytes lets a reader check the log against the tool's output."""
    t, path = tracer
    t.on_tool_end("z" * 123)
    assert "served_bytes=123" in path.read_text()


def test_tool_calls_are_marked_too(tracer):
    """The call side carries arguments and truncates on the same rule."""
    t, path = tracer
    t.on_tool_start({"name": "prepare_simulation"}, "q" * 5000)
    text = path.read_text()
    assert "prepare_simulation" in text
    assert "TRUNCATED BY THE LOGGER" in text


def test_the_cut_still_bounds_the_file(tracer):
    """The fix must not turn the trajectory into a multi-megabyte dump."""
    t, path = tracer
    for _ in range(20):
        t.on_tool_end("w" * 50000)
    assert path.stat().st_size < 40_000, (
        f"{path.stat().st_size} bytes for 20 results: the truncation stopped "
        "bounding the file, and a full round would fill the disk")
