"""A time-dependent, vector or 3-D coupled task gets ITS contract first. Measured 2026-09-13: with
physics='transient_heat' the fenics coupling reply was byte-identical to the physics-less one (the steady
scalar contract led, the transient block sat past every cut); kratos with 'heat_3d' and dune the same; the
thermo-elastic block alone was promoted. The same reply now (1) leads with the variant the physics names,
(2) follows it through the parts door and the pushed contract, (3) tells the physics-less caller which word
to pass. Also: the deal.II door no longer names a shipped C++ solver (there is none), a FEBio or SPARTA side
gets its deck grammar whole, and the NGSolve/FEniCSx source knob says how a NumPy function enters the form."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _first_block(text: str) -> str:
    i = text.find("```python")
    j = text.find("```", i + 9)
    return text[i:j] if i >= 0 else ""


def _replies(solver: str, physics: str = "", signal: str = ""):
    from core.registry import load_all_backends
    load_all_backends()
    from tools import consolidated as C
    C._MUST_READ_STATE["served"] = False          # the first reply of a session ...
    first = C._get_coupling_knowledge(solver, signal, physics)
    pointer = C._get_coupling_knowledge(solver, signal, physics)   # ... and the worker's pointer-mode reply
    return first, pointer


class _StubMCP:
    def __init__(self):
        self.tools = {}

    def tool(self, *a, **k):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


def _knowledge():
    from core.registry import load_all_backends
    load_all_backends()
    from tools.consolidated import register_consolidated_tools
    mcp = _StubMCP()
    register_consolidated_tools(mcp)
    return mcp.tools["knowledge"]


CASES = [
    ("fenics", "transient two-material heat conduction", "TRANSIENT participant", "transient"),
    ("dealii", "transient_heat", "TRANSIENT participant", "transient"),
    ("skfem", "linear_elasticity", "VECTOR participant", "elasticity"),
    ("ngsolve", "two-material linear elasticity, plane strain", "VECTOR participant", "elasticity"),
    ("febio", "elasticity", "VECTOR participant", "elasticity"),
    ("kratos", "heat_3d", "3-D scalar", "3d"),
    ("dune", "diffusion_3d", "3-D scalar", "3d"),
]


@pytest.mark.parametrize("solver,physics,marker,word", CASES)
def test_the_pointer_reply_leads_with_the_variant_contract(solver, physics, marker, word):
    _first, pointer = _replies(solver, physics)
    block = _first_block(pointer)
    assert marker in block, f"{solver}/{physics}: the first served block is not the variant:\n{block[:600]}"
    assert "imports.json" in block and "exports.json" in block
    assert "WHICH CONTRACT BELOW IS YOURS" not in pointer      # the notice is for the physics-less call


@pytest.mark.parametrize("solver,physics,marker,word", CASES)
def test_the_first_reply_carries_the_variant_or_names_its_physics_word(solver, physics, marker, word):
    first, _pointer = _replies(solver, physics)
    block = _first_block(first)
    if block:
        assert marker in block, f"{solver}/{physics}: the first reply's block is not the variant"
    else:   # too long to repeat next to the must-read: the note must send the worker with the word
        assert f"physics='{word}'" in first, first[:1500]


@pytest.mark.parametrize("solver,physics,marker,word", CASES)
def test_the_pushed_contract_follows_the_physics(solver, physics, marker, word):
    from core.registry import load_all_backends
    load_all_backends()
    from tools import consolidated as C
    pushed = C._coupling_participant_script(solver, physics)
    assert marker in pushed, f"{solver}/{physics}: the pushed contract is not the variant:\n{pushed[:500]}"


def test_the_parts_door_follows_the_physics():
    from core.registry import load_all_backends
    load_all_backends()
    from tools import consolidated as C
    part = C._get_coupling_knowledge("fenics", "participant:part1", "transient heat conduction")
    assert "TRANSIENT" in part, part[:800]
    part = C._get_coupling_knowledge("skfem", "participant:part1", "linear_elasticity")
    assert "VECTOR" in part, part[:800]


def test_the_physics_less_call_is_told_which_word_to_pass():
    first, pointer = _replies("fenics", "")
    for out in (first, pointer):
        assert "WHICH CONTRACT BELOW IS YOURS" in out
        for word in ("physics='transient'", "physics='elasticity'", "physics='thermoelastic'"):
            assert word in out, word
    assert "TRANSIENT participant" not in _first_block(pointer)   # the scalar contract still leads there


@pytest.mark.parametrize("physics,expected", [
    ("thermoelastic", "thermoelastic"), ("thermo_structure_interaction", "thermoelastic"),
    ("linear_elasticity", "elastic"), ("two-material linear elasticity, plane strain", "elastic"),
    ("transient two-material heat conduction", "transient"), ("heat_transient", "transient"),
    ("heat_3d", "3d"), ("3d", "3d"), ("diffusion_3d", "3d"),
    ("fluid_structure_interaction", ""), ("fsi", ""), ("heat", ""), ("conjugate_heat_transfer", ""), ("", ""),
])
def test_the_physics_word_maps_to_the_shipped_variant(physics, expected):
    from tools import consolidated as C
    assert C._variant_for(physics) == expected


def test_the_dealii_door_names_no_shipped_cpp_solver():
    first, pointer = _replies("dealii", "")
    for out in (first, pointer):
        head = out[:out.find("```python")] if "```python" in out else out
        assert "YOURS TO WRITE" in head, "the C++-is-yours statement must precede the contract"
        assert "PRINTED IN FULL" not in out
        assert "heat_iface_dealii.cc" not in out and "(heat_iface_dealii)" not in out
        assert "DEALII_EXE" in out
    for name in ("participant_dealii.py", "participant_dealii_elastic.py", "participant_dealii_transient.py"):
        text = (ROOT / "data" / "coupling_participants" / name).read_text()
        assert "THAT YOU\nWRITE AND BUILD YOURSELF" in text, name
        assert ".cc)" not in text.split("OASiS DOES NOT SERVE THIS")[0], name   # nothing served names a .cc


@pytest.mark.parametrize("solver,module,const,physics", [
    ("febio", "backends.febio.deck_grammar", "FEBIO_DECK_GRAMMAR", ""),
    ("febio", "backends.febio.deck_grammar", "FEBIO_DECK_GRAMMAR", "elasticity"),
    ("sparta", "backends.sparta.deck_grammar", "SPARTA_INPUT_GRAMMAR", ""),
])
def test_a_binary_side_gets_its_deck_grammar_whole(solver, module, const, physics):
    from tools.consolidated import _KNOWLEDGE_REPLY_LIMIT, _UNIVERSAL_CORE
    grammar = getattr(__import__(module, fromlist=[const]), const)
    K = _knowledge()
    for _ in range(2):    # first and pointer reply
        out = K(topic="coupling", solver=solver, physics=physics) if physics else K(topic="coupling", solver=solver)
        assert grammar in out, f"{solver}: the deck grammar is cut or missing"
        assert len(out) <= _KNOWLEDGE_REPLY_LIMIT + len(grammar) + len(_UNIVERSAL_CORE) + 4000


def test_the_source_knob_says_how_a_numpy_function_enters_the_form():
    _first, ng = _replies("ngsolve", "")
    block = _first_block(ng)
    assert "CoefficientFunction(F_SRC) is a TypeError" in block
    assert "GridFunction" in block and "v.point for v in mesh.vertices" in block
    _first, fx = _replies("fenics", "")
    block = _first_block(fx)
    assert "interpolate" in block and "SpatialCoordinate" in block
    _first, ngv = _replies("ngsolve", "linear_elasticity")
    assert "CoefficientFunction(B_SRC) is a TypeError" in _first_block(ngv)


WHOLE_BLOCK_CASES = [("dune", "3d"), ("kratos", "3d"), ("fenics", "transient"), ("dealii", "transient"),
                     ("skfem", "elasticity"), ("ngsolve", "elasticity"), ("febio", "elasticity"),
                     ("fenics", ""), ("fourc", "thermoelastic"), ("sparta", "")]


@pytest.mark.parametrize("solver,physics", WHOLE_BLOCK_CASES)
def test_the_first_handshake_block_arrives_whole_through_the_tool(solver, physics):
    """The reply an agent actually receives must not end inside the contract it is told to copy.

    Measured 2026-09-14 on knowledge(topic='coupling', solver='dune', physics='3d'): the served text
    ended with an opening ```python and NO closing fence -- the 3-D participant was cut mid-block by the
    last cap every reply passes through, after the front-loader had kept it whole. All three C10 cells
    of round 49 failed on that exact side, and none of them ever ran it."""
    import re
    K = _knowledge()
    for _ in range(2):          # the session's first reply, then a worker's own call
        out = K(topic="coupling", solver=solver, physics=physics) if physics else K(topic="coupling", solver=solver)
    m = re.search(r"```python\n(.*?)```", out, re.S)
    assert m, f"{solver}/{physics or 'base'}: no complete fenced block in a {len(out)}-character reply"
    block = m.group(1)
    assert "imports.json" in block and "exports.json" in block, "the first block is not the handshake"
    # A LATER section may still be cut -- that is what the cap is for -- but then the reply has to say
    # so. What must never happen is the reply ending inside the block the agent was told to copy.
    if out.count("```") % 2:
        assert "THIS PAYLOAD IS TRUNCATED HERE" in out or "THIS REPLY IS CUT AT" in out, \
            "the reply ends inside a code block and never says it was cut"
        assert out.rindex("```") > out.index(m.group(1)), "the FIRST block is the one that was cut"


@pytest.mark.parametrize("solver", ["fourc", "fenics", "skfem"])
def test_a_fluid_structure_task_reaches_the_fsi_contracts(solver):
    """They were reachable only by passing the physics as a SOLVER name.

    Measured 2026-09-14: knowledge(topic='coupling', solver='fsi') served the fluid and structure
    participants, while solver='fourc' or solver='fenics' -- what an agent coupling those two codes
    actually calls -- served the scalar heat contract with nothing saying an FSI contract exists. One
    development problem is exactly a 4C structure against a FEniCSx fluid."""
    K = _knowledge()
    K(topic="coupling", solver=solver, physics="fsi")            # the session's first reply
    out = K(topic="coupling", solver=solver, physics="fsi")      # what the WORKER's own call returns
    assert "THE FLUID PARTICIPANT" in out and "THE STRUCTURE PARTICIPANT" in out
    assert "YOU ASKED FOR A FLUID-STRUCTURE COUPLING" in out
    long_word = K(topic="coupling", solver=solver, physics="fluid_structure_interaction")
    assert "THE FLUID PARTICIPANT" in long_word
    # and the physics-less reply has to say the word exists
    plain = K(topic="coupling", solver=solver)
    assert "physics='fsi'" in plain, "the physics-less reply never names the FSI word"


def test_the_fsi_reply_names_the_participant_that_runs_under_the_asked_code():
    K = _knowledge()
    K(topic="coupling", solver="fourc", physics="fsi")
    out = K(topic="coupling", solver="fourc", physics="fsi")
    assert "participant_fsi_solid_fourc.py" in out
    # and the reply LEADS with the participant that runs under the code that asked
    import re
    m = re.search(r"```python\n(.*?)```", out, re.S)
    assert m and "4C STRUCTURE participant" in m.group(1)[:400], \
        "the 4C side is still handed another code's structure contract first"
    out2 = K(topic="coupling", solver="skfem", physics="fsi")
    assert "participant_fsi_solid_skfem.py" in out2
