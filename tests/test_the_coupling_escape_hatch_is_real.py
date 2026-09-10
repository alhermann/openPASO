"""OASiS promised the truncated remainder on request. It did not deliver it.

The coupling payload is capped, and the notice at the cut tells the agent, in
as many words: "The rest is available on request -- ask
knowledge(topic='coupling', solver='...', signal='<what you are stuck on>')".
The module comment behind the cap says the same: "Nothing is deleted: every
section is still reachable by asking for it, which is what the `signal`
argument is for."

MEASURED, both false. `knowledge(topic='coupling', solver='kratos')` is cut at
the head limit, and asking again with a signal returned a DIFFERENT, shorter
payload of failure-table entries — 11,529 characters carrying 5 lines the base
did not already have. The material that had been cut was in neither reply. For
solver='kratos' the casualty is the participant script itself, severed
mid-definition at `def build_model()` — on the side a coupled task is most
likely to give Kratos. The consistent flux-recovery formula, which decides the
graded interface order, was cut away too and equally unreachable.

An unkept promise is worse than an honest cap: the agent spends a call, thinks
it has asked for the missing piece, and reads a reply that does not contain it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class _CollectingMCP:
    def __init__(self):
        self.tools = {}

    def tool(self, *a, **kw):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


@pytest.fixture(scope="module")
def knowledge_tool():
    import os
    os.chdir(ROOT)
    from core.registry import load_all_backends
    load_all_backends()
    from tools.consolidated import register_consolidated_tools
    mcp = _CollectingMCP()
    register_consolidated_tools(mcp)
    return mcp.tools["knowledge"]


# (signal, a string that lives PAST the head cut and must come back)
CASES = [
    ("the kratos participant script is cut off, I need build_model",
     "def build_model", "kratos"),
    ("my interface flux will not balance and the order is 1.1",
     "q = -r/w", ""),
    ("the residual is rising, my iteration diverges",
     "DIVERGES", ""),
]


@pytest.mark.parametrize("signal,marker,solver", CASES)
def test_a_signalled_request_returns_what_the_cut_left_out(
        signal, marker, solver, knowledge_tool):
    kw = {"topic": "coupling", "signal": signal}
    if solver:
        kw["solver"] = solver
    out = knowledge_tool(**kw)
    assert marker in out, (
        f"signal={signal!r} did not bring back {marker!r}, which sits past the "
        f"head cut — the notice's promise is still unkept"
    )


def test_the_continuation_says_how_much_it_brought_back(knowledge_tool):
    out = knowledge_tool(topic="coupling", signal="flux recovery")
    assert "WHAT THE TRUNCATED REPLY LEFT OUT" in out
    assert "remaining characters" in out, (
        "the agent should be able to see how much of the cut material this "
        "reply covered, so it knows whether to ask again"
    )


def test_the_continuation_survives_the_front_loader(knowledge_tool):
    """Appending before the cut fed it straight back into the trimmer."""
    src = (ROOT / "src" / "tools" / "consolidated.py").read_text()
    assert "_append_coupling_continuation(\n            _front_load_coupling(" in src, (
        "the continuation must be appended AFTER truncation; appending before "
        "it made the requested material the first thing dropped"
    )


def test_rare_words_outrank_common_ones(knowledge_tool):
    """Counting raw hits buried the section the agent actually named.

    Measured: the section carrying `def build_model()` scored 2 while three
    generic sections scored 4 on "element", "script" and "need", filled the
    budget and pushed it out.
    """
    out = knowledge_tool(topic="coupling", solver="kratos",
                         signal="build_model")
    assert "def build_model" in out, (
        "a one-word signal naming a distinctive symbol must find its section"
    )


def test_an_unsignalled_request_is_unchanged(knowledge_tool):
    """The cap exists for a reason: only an explicit ask reopens it."""
    from tools.consolidated import _COUPLING_HEAD_LIMIT
    plain = knowledge_tool(topic="coupling", solver="kratos")
    assert "WHAT THE TRUNCATED REPLY LEFT OUT" not in plain, (
        "an ordinary coupling request must not carry the continuation, or the "
        "head limit stops doing its job"
    )
    assert "THIS PAYLOAD IS TRUNCATED HERE" in plain
    # Bound the payload by what it is MADE OF rather than by a round number:
    # the truncated head, the universal core that rides on every reply, and the
    # truncation notice. Stated this way the check keeps working when the core
    # is edited, and still fails if the head limit stops being enforced.
    from tools.knowledge import _UNIVERSAL_CORE
    from tools.consolidated import _deciding_block
    NOTICE = 1500
    # a named code's deciding facts ride along after the lead (measured: runs
    # that never called prepare_simulation hand-rolled the code's API from
    # memory); they are bounded by their own length, not by the head cap
    facts = len(_deciding_block("kratos", "coupled side"))
    assert len(plain) <= _COUPLING_HEAD_LIMIT + len(_UNIVERSAL_CORE) + NOTICE + facts, (
        f"the unsignalled coupling reply is {len(plain)} characters against a "
        f"head limit of {_COUPLING_HEAD_LIMIT}; the cap is not being enforced"
    )


@pytest.mark.parametrize("solver,role,filename", [
    ("fourc", "", "participant_fourc.py"),
    ("kratos", ":neumann", "participant_kratos_neumann.py"),
])
def test_participant_escape_hatch_reconstructs_the_elided_contract(
        solver, role, filename, knowledge_tool):
    """The parts route exists for clients that truncate long replies. It used
    to read the file raw and hand over the SHA-256 of the complete tested
    program in the last part -- a solver hand-over door beside the elided one.
    The parts must reassemble to exactly what the elided route serves, and to
    nothing more."""
    import re
    from tools.coupling_knowledge import _serve_participant, _SOLVE_BEGIN, _SOLVE_END

    path = ROOT / "data" / "coupling_participants" / filename
    source = path.read_text()
    chunks = []
    final = ""
    for part in range(1, 10):
        out = knowledge_tool(
            topic="coupling", solver=solver,
            signal=f"participant{role}:part{part}")
        marker = f"participant CONTRACT (solve elided): part {part} of"
        start = out.index(marker)
        match = re.search(r"```python\n(.*?)```", out[start:], re.S)
        assert match, f"{solver} part {part} has no fenced source"
        chunks.append(match.group(1))
        assert "THIS PAYLOAD IS TRUNCATED HERE" not in out
        if "FINAL PART" in out:
            final = out
            break

    joined = "".join(chunks)
    assert joined == _serve_participant(path)
    assert joined != source, "the parts reassemble to the complete tested file"
    # every marked SOLVE region of the source is absent from the reassembly
    i = 0
    while True:
        a = source.find(_SOLVE_BEGIN, i)
        if a < 0:
            break
        b = source.find(_SOLVE_END, a)
        body = source[a + len(_SOLVE_BEGIN):b].strip()
        assert body and body not in joined, f"{filename}: a SOLVE region came back"
        i = b + len(_SOLVE_END)
    assert "exact tested file" not in final and "complete tested" not in final


def test_unsignalled_backend_reply_names_complete_script_route(knowledge_tool):
    out = knowledge_tool(topic="coupling", solver="kratos")
    assert "signal='participant:part1'" in out
    assert "signal='participant:neumann:part1'" in out


def test_the_continuation_is_bounded(knowledge_tool):
    """Reopening the cap must not dump the whole corpus back."""
    from tools.consolidated import _COUPLING_CONTINUATION_LIMIT
    out = knowledge_tool(topic="coupling", solver="kratos",
                         signal="build_model element construction script flux "
                                "residual theta accelerator interface normal")
    assert len(out) < 60_000, (
        f"a signalled reply grew to {len(out)} characters; the budget that "
        f"makes the agent run out of ACTIONS is what the cap protects"
    )
    assert _COUPLING_CONTINUATION_LIMIT <= 20_000
