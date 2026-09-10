"""Guards for the coupling knowledge an agent is actually served.

The failure this file exists to prevent, measured on the live tools before the
rewrite: the whole coupling corpus for all nine backends was smaller than a
tenth of one backend's pitfall payload, named two backends out of nine, varied
not at all with `solver=`, and documented the DEPRECATED tool's enum while the
general tool's contract lived only inside that tool's own docstring.

Every assertion here is structural — a key that must be documented, a payload
that must differ per backend, a script that must compile. Nothing pins a number
produced by a run, because a knowledge test that pins a measurement makes the
measurement part of the shipped corpus.
"""
from __future__ import annotations

import ast
import inspect
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tools.coupling_knowledge import (            # noqa: E402
    _BACKEND_ORDER, _PARTICIPANT_DIR, coupling_core, coupling_knowledge,
    coupling_sides_table, precice_knowledge,
)


def _core() -> str:
    return coupling_core()


# ── the contract must be IN the knowledge, not only in a docstring ───────

@pytest.mark.parametrize("token", [
    "imports.json", "exports.json", "InterfaceData", "coordinates",
    "normal_fluxes", "imports_from", "work_dir", "field_name", "n_points",
])
def test_couple_contract_terms_are_documented(token):
    """The `couple` contract used to exist ONLY in the tool's docstring: the
    knowledge contained zero occurrences of any of these."""
    assert token in _core(), f"'{token}' missing from knowledge(topic='coupling')"


def test_knowledge_names_the_general_tool_not_the_deprecated_enum():
    core = _core()
    assert "couple(" in core
    # the legacy enum values must not be presented as the way to couple
    for legacy in ("heat_dd", "poisson_dd", "tsi_dd", "poisson_dd_study"):
        assert legacy not in core, (
            f"knowledge still advertises the deprecated coupled_solve enum "
            f"value '{legacy}'")


def test_knowledge_does_not_point_at_private_generators():
    """It used to instruct the agent to write 'a new subdomain script generator
    (analogous to `_fenics_heat_subdomain_script`)' — a private function that no
    agent can call."""
    core = _core()
    assert "_fenics_heat_subdomain_script" not in core
    assert "_generate_domain_b_input" not in core


def test_theta_guidance_matches_a_real_parameter():
    """Two knowledge sections used to give theta-selection tables while the tool
    had no theta parameter at all."""
    from tools import consolidated
    src = inspect.getsource(consolidated)
    sig = re.search(r"async def couple\((.*?)\) -> str:", src, re.S).group(1)
    assert "theta" in sig, "couple() has no theta parameter"
    assert "theta" in _core(), "knowledge does not mention theta"


def test_sign_convention_states_both_quantities():
    core = _core()
    assert "apply the same number, export opposite numbers" in core.lower()


# ── per-backend payloads must actually differ ────────────────────────────

def test_every_backend_has_a_named_payload_entry():
    core = _core()
    for name in _BACKEND_ORDER:
        assert f"solver='{name}'" in core, (
            f"knowledge(topic='coupling') never tells the agent to ask for "
            f"solver='{name}'")


def test_coupling_payload_varies_by_backend():
    """The measured bug: byte-identical output for every solver."""
    seen = {}
    for name in _BACKEND_ORDER:
        text = coupling_knowledge(name)
        assert text != coupling_core(), f"solver='{name}' returned the core payload"
        assert text not in seen.values(), (
            f"solver='{name}' is byte-identical to solver='{seen and [k for k,v in seen.items() if v==text][0]}'")
        seen[name] = text


def test_precice_payload_varies_by_backend():
    seen = {}
    for name in _BACKEND_ORDER:
        text = precice_knowledge(name)
        assert text not in seen.values(), (
            f"precice solver='{name}' is byte-identical to another backend")
        seen[name] = text


def test_unknown_solver_is_not_a_dead_end():
    out = coupling_knowledge("no-such-backend")
    assert "no-such-backend" in out
    # still hands over the general contract rather than nothing
    assert "imports.json" in out


# ── the shipped participant scripts must be real, runnable files ─────────

def _script_backends():
    return [n for n in _BACKEND_ORDER
            if (_PARTICIPANT_DIR / f"participant_{n}.py").is_file()]


def test_at_least_the_flagship_participants_ship():
    for name in ("fenics", "fourc"):
        assert (_PARTICIPANT_DIR / f"participant_{name}.py").is_file(), (
            f"participant script for {name} is missing")


@pytest.mark.parametrize("name", _script_backends())
def test_participant_script_parses_and_is_complete(name):
    p = _PARTICIPANT_DIR / f"participant_{name}.py"
    text = p.read_text()
    ast.parse(text)                                   # syntactically runnable
    assert "imports.json" in text and "exports.json" in text
    assert "EDIT THIS BLOCK" in text, (
        "the script must mark the block a user edits, or a weak model has to "
        "work out which constants are problem data")
    assert "PLACEHOLDER" in text, (
        "the edit block must say its numbers are placeholders — shipping a "
        "ready-made benchmark configuration is the anchoring the project bans")
    assert re.search(r"^PARTNER\b", text, re.M), f"{name}: no PARTNER constant"
    # a participant that can take both roles must expose which one it is
    if "SIDE ==" in text:
        assert re.search(r"^SIDE\b", text, re.M), f"{name}: no SIDE constant"


def _solve_bodies(text: str) -> list[str]:
    """The marked SOLVE regions of a participant file, body text only."""
    from tools.coupling_knowledge import _SOLVE_BEGIN, _SOLVE_END
    out, i = [], 0
    while True:
        a = text.find(_SOLVE_BEGIN, i)
        if a < 0:
            return out
        b = text.find(_SOLVE_END, a)
        body = text[a + len(_SOLVE_BEGIN):b if b >= 0 else len(text)]
        if body.strip():
            out.append(body)
        if b < 0:
            return out
        i = b + len(_SOLVE_END)


@pytest.mark.parametrize("name", _script_backends())
def test_served_payload_is_the_elided_contract_of_the_tested_participant(name):
    """Serving and execution use ONE source file, and serving elides its SOLVE.

    What the agent receives is the tested file minus every marked SOLVE
    region -- the handshake, the sign convention, the flux recovery and the
    exports schema survive; the mesh, form, material, source and solve do not.
    """
    from tools.coupling_knowledge import _script, _serve_participant
    served = coupling_knowledge(name)
    path = _PARTICIPANT_DIR / f"participant_{name}.py"
    text = path.read_text()
    assert "```python" in served

    from tools.coupling_knowledge import lean_view
    excerpt = _script(name)
    assert excerpt == _serve_participant(path)
    # the payload carries the LEAN view of the elided contract (comment blocks
    # thinned, every code line kept) -- the copyable form; the annotated text
    # is behind the parts door
    assert lean_view(excerpt) in served, (
        f"solver='{name}': the payload does not contain the lean view of what "
        f"_script() returns -- the served path and the tested file have diverged")
    code = [l for l in excerpt.splitlines() if l.strip() and not l.strip().startswith("#")]
    assert all(l in served for l in code), f"solver='{name}': a code line was dropped"
    assert excerpt != text, f"solver='{name}': nothing was elided"
    bodies = _solve_bodies(text)
    assert bodies, f"{path.name} has no marked SOLVE region"
    for body in bodies:
        assert body.strip() not in served, (
            f"solver='{name}': a marked SOLVE region reached the agent")
    assert "EDIT THIS BLOCK" in excerpt and "PLACEHOLDER" in excerpt
    assert "imports.json" in excerpt and "exports.json" in excerpt


# ── EVERY file a served payload reaches, not a list of nine names ────────
#
# This section replaced a test parametrised on `_BACKEND_ORDER`. That test
# looked at NINE files while THIRTY sat in the directory, and its own docstring
# already said the right thing — "a file with no markers is served whole, that
# is a gap, not a licence" — about files it never opened.
#
# What it missed, measured on the live tool: `knowledge(topic='coupling',
# solver='fsi')` returned 33 kB carrying TWO COMPLETE SOLVERS verbatim, a
# Taylor-Hood incompressible Navier-Stokes fluid with a harmonic ALE lift and a
# scikit-fem solid, under the heading "copy verbatim, edit the marked block
# only". `fsi` is not a backend name, so no parameter of that test ever named
# those files and no assertion in the suite ever read them.
#
# A NAME LIST CANNOT NOTICE A FILE IT WAS NEVER TOLD ABOUT. So the enumeration
# is taken from the tool instead: build every payload a caller can ask for and
# record which participant files were actually read while doing it. Add a
# backend, a physics topic or an alias and its scripts enter this test by being
# served, which is the only condition that matters.

_SERVED_DOORS = ([""] + _BACKEND_ORDER
                 + ["fsi", "fluid-structure", "fluid_structure", "tsi"])


def _open_every_door():
    """(files reached, payloads produced) over every served coupling door.

    The spy sits on `Path.read_text` rather than on `_script`, because
    `_script` was never the only door: four helpers — the 3-D, transient, role
    and vector blocks — read their variant file directly, and the FSI payload
    reads two more through `_script` under names that are not backend names.
    Recording the READ is the only way to enumerate that does not have to be
    kept in step with the module by hand.
    """
    import pathlib
    seen: list[str] = []
    payloads: list[str] = []
    original = pathlib.Path.read_text

    def spy(self, *args, **kwargs):
        if self.parent == _PARTICIPANT_DIR and self.name not in seen:
            seen.append(self.name)
        return original(self, *args, **kwargs)

    pathlib.Path.read_text = spy
    try:
        for door in _SERVED_DOORS:
            payloads.append(coupling_knowledge(door))
            payloads.append(precice_knowledge(door))
    finally:
        pathlib.Path.read_text = original
    return sorted(seen), payloads


_REACHED, _SERVED_PAYLOADS = _open_every_door()


def test_the_door_sweep_reaches_more_than_the_backend_names():
    """A guard on the guard: if the spy stops seeing reads, every test below
    silently parametrises on nothing and passes without checking anything."""
    assert _REACHED, (
        "no participant file was read while building any served payload — the "
        "read spy in _open_every_door() has stopped working, and every marker "
        "test below is now vacuous")
    assert "participant_fenics.py" in _REACHED
    assert "participant_fsi_fluid_fenics.py" in _REACHED, (
        "the FSI door is no longer reaching its fluid participant; either the "
        "payload changed or 'fsi' has been dropped from _SERVED_DOORS")
    assert len(_REACHED) > len(_BACKEND_ORDER), (
        f"the sweep found only {len(_REACHED)} files. The whole point is that "
        f"more files are served than there are backend names — a count at or "
        f"below {len(_BACKEND_ORDER)} means the doors are not all open")


@pytest.mark.parametrize("fname", _REACHED)
def test_every_reached_participant_is_served_with_its_solve_elided(fname):
    """No door hands the agent a marked SOLVE region, and none hands over the
    file verbatim; the contract around the hole is what gets served."""
    text = (_PARTICIPANT_DIR / fname).read_text()
    served_all = "\n".join(_SERVED_PAYLOADS)
    assert text not in served_all, (
        f"{fname}: the tested participant file is served verbatim, solve "
        "included -- a door has bypassed the elision")
    bodies = _solve_bodies(text)
    assert bodies, f"{fname} has no marked SOLVE region"
    for body in bodies:
        assert body.strip() not in served_all, (
            f"{fname}: a marked SOLVE region reached the agent")


def test_the_parts_door_serves_the_same_elided_contract():
    """`signal='participant[:role]:partN'` exists for clients that truncate
    long replies. It used to read the file raw and hand the SHA-256 of the
    complete tested program over in the last part -- a solver hand-over door
    beside the elided one. The parts must reassemble to the ELIDED text."""
    from tools.coupling_knowledge import _serve_participant
    for name in _script_backends():
        for role in ("", ":neumann", ":elastic", ":transient", ":3d"):
            suffix = role.replace(":", "_")
            path = _PARTICIPANT_DIR / f"participant_{name}{suffix}.py"
            if not path.is_file():
                continue
            parts, k = [], 1
            while True:
                out = coupling_knowledge(name, f"participant{role}:part{k}")
                m = re.search(r"```python\n(.*?)```", out, re.S)
                assert m, f"{name}{role} part {k}: no fenced block: {out[:120]}"
                parts.append(m.group(1))
                if "FINAL PART" in out:
                    break
                k += 1
                assert k < 40, f"{name}{role}: no FINAL PART after {k} parts"
            joined = "".join(parts)
            assert joined == _serve_participant(path), (
                f"{name}{role}: the parts do not reassemble to the elided "
                "contract")
            for body in _solve_bodies(path.read_text()):
                assert body.strip() not in joined, (
                    f"{name}{role}: the parts door served a SOLVE region")
            assert "exact tested file" not in out and "complete tested" not in out


def test_every_shipped_participant_keeps_balanced_region_markers():
    """The markers are what the elision cuts on, so every file needs them."""
    from tools.coupling_knowledge import _SOLVE_BEGIN, _SOLVE_END
    missing = []
    for p in sorted(_PARTICIPANT_DIR.glob("participant_*.py")):
        text = p.read_text()
        if _SOLVE_BEGIN not in text:
            missing.append(p.name)
        else:
            assert text.count(_SOLVE_BEGIN) == text.count(_SOLVE_END), (
                f"{p.name} has unbalanced solve markers")
    assert not missing, f"participant files missing solve-region markers: {missing}"


# ── nothing machine-specific may be served ───────────────────────────────

_HOST_PATH = re.compile(r"(?<![\w/])(?:/home/|/Users/|/opt/|/usr/(?:local/)?(?:bin|lib))")


@pytest.mark.parametrize("name", [""] + _BACKEND_ORDER)
def test_no_absolute_host_paths_in_served_coupling_knowledge(name):
    for text in (coupling_knowledge(name), precice_knowledge(name)):
        hits = _HOST_PATH.findall(text)
        assert not hits, (
            f"solver='{name}': served knowledge carries a path from the build "
            f"host ({hits[:3]}); it is wrong on every other machine. Tell the "
            f"agent to read the path from discover(query='list') instead.")


def test_discover_has_a_coupling_branch():
    """There was no path to the coupling knowledge unless you already knew the
    topic string existed."""
    from tools import consolidated
    src = inspect.getsource(consolidated)
    assert 'query == "coupling"' in src
    doc = re.search(r"def discover\(.*?\"\"\"(.*?)\"\"\"", src, re.S).group(1)
    assert "coupling" in doc, (
        "discover's docstring does not advertise query='coupling'")
    assert "knowledge(topic='coupling'" in src, (
        "discover(query='coupling') must name the follow-up knowledge call")


# ── the launch section must be SPECIALISED where the participant is a wrapper ──

# backend -> the constant its shipped script keeps the solver binary in
_WRAPPER_BACKENDS = {"fourc": "FOURC_BIN", "febio": "FEBIO",
                     "sparta": "SPARTA", "dealii": "DEALII_EXE"}


@pytest.mark.parametrize("name,const", sorted(_WRAPPER_BACKENDS.items()))
def test_wrapper_backends_say_the_command_runs_the_wrapper(name, const):
    """For these four the participant is a Python WRAPPER around a binary, so
    `command` must run PYTHON and the binary goes into a constant in the script.

    The measured bug this pins: the specialised step 4 was applied with
    `_LAUNCH_PY.replace(<literal>, ...)` and ALL FOUR literals were stale, so
    every one of these backends served the generic "get the interpreter or
    binary from discover(query='list')" — and what discover prints for them IS
    the binary. `str.replace` cannot fail, so the corpus said nothing while
    telling a weak model to put a solver binary where an interpreter goes.
    """
    served = coupling_knowledge(name)
    assert "runs the WRAPPER" in served, (
        f"solver='{name}': the launch section does not say that `command` runs "
        f"the wrapper rather than the {name} binary")
    assert const in served, (
        f"solver='{name}': the launch section never names `{const}`, the "
        f"constant the binary path actually goes into")


@pytest.mark.parametrize("name", _BACKEND_ORDER)
def test_launch_section_has_no_unsubstituted_field(name):
    """A named field left in the served text is a hard stop for a weak model."""
    for text in (coupling_knowledge(name), coupling_core()):
        assert "{INTERP}" not in text, f"solver='{name}': unsubstituted {{INTERP}}"
        assert "{RIGHT}" not in text, f"solver='{name}': unsubstituted {{RIGHT}}"


@pytest.mark.parametrize("name", _BACKEND_ORDER)
def test_served_edit_block_only_names_constants_the_script_defines(name):
    """Step 2 hands over a "complementary side" edit block. Every constant in it
    must EXIST in the script served in the same payload.

    The measured bug: the CONDUCTION right-side block was embedded in all nine
    payloads. FEBio's script is elasticity (no K, F_SRC, T_OUTER, T_INIT),
    Kratos's has no SIDE/IFACE_X/Y0,Y1/NX,NY/F_SRC/Q_INIT, and SPARTA's has none
    of nine of them — so three of nine backends were told to set constants that
    do not exist in the file they had just been given. For SPARTA the block also
    said `SIDE="neumann"` while the same payload says the Neumann role is
    IMPOSSIBLE. Applying the served block to the served FEBio script was
    confirmed to fail on its first key.
    """
    def assigned(text):
        """Names bound by a module-level assignment, incl. `X0, X1 = a, b`."""
        out = set()
        for m in re.finditer(r"^([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*=[^=]", text, re.M):
            out |= {c.strip() for c in m.group(1).split(",")}
        return out

    served = coupling_knowledge(name)
    script = (_PARTICIPANT_DIR / f"participant_{name}.py").read_text()
    defined = assigned(script)
    blocks = re.findall(r"```python\n(.*?)```", served, re.S)
    # the edit blocks are the short fenced blocks; the participant is the long one
    for blk in [b for b in blocks if b != script and len(b) < 2000]:
        missing = sorted(assigned(blk) - defined)
        assert not missing, (
            f"solver='{name}': the served edit block sets {missing}, which the "
            f"served participant script never defines")


def test_sparta_flux_claim_is_not_overstated():
    """The payload said SPARTA "CANNOT take the Neumann role" because
    "`fix surf/temp` derives one from SPARTA's OWN computed flux, not from an
    imported one". The second half is factually wrong, checked in the SPARTA
    source on this install:

      * fix_surf_temp.cpp accepts `f_<fixID>` and the ONLY requirement is
        `per_surf_flag != 0` — it never checks where the flux came from. SPARTA's
        own doc/fix_surf_temp.txt says "Note that SPARTA does not check that the
        specified compute/fix calculates an energy flux."
      * fix_ave_surf.cpp accepts `s_<name>` (CUSTOM, arg[i][0] == 's') and sets
        `per_surf_flag = 1`, and its CUSTOM branch reads `surf->edvec[...]`.
      * `custom surf ... file` resets custom per-surf values from a FILE.

    So an imported flux does reach `fix surf/temp`, which converts it via
    `T = (q/(sigma*emisurf))^(1/4)` (fix_surf_temp.cpp: `pow(prefactor*qw,0.25)`
    with `prefactor = 1/(emi*SB_SI)`). That is not a Neumann BC and carries real
    caveats, but "cannot" removed a capability that exists. A wrong "cannot" is
    as costly as a wrong "can".
    """
    served = coupling_knowledge("sparta")
    assert "CANNOT take the Neumann role" not in served, (
        "the flat 'cannot' is wrong: an imported flux can reach fix surf/temp")
    assert "not from an imported one" not in served, (
        "fix surf/temp does not care where the flux came from")
    # the honest version must still be unmistakable about the absence of a real
    # flux BC, and must name the indirect route rather than hiding it
    assert "NO native flux boundary condition" in served
    assert "fix surf/temp" in served and "Stefan-Boltzmann" in served
    assert "custom surf" in served and "fix ave/surf" in served
    assert "NOT been run here" in served, (
        "a source-derived route must be marked unproven, not implied to be tested")


def test_sparta_is_not_handed_a_neumann_edit_block():
    """SPARTA's own payload says the Neumann role is impossible; the launch
    section must not simultaneously tell the agent to make a neumann copy."""
    served = coupling_knowledge("sparta")
    assert 'SIDE      = "neumann"' not in served, (
        "the SPARTA payload serves a neumann edit block while stating that the "
        "Neumann role cannot be done at all")
    assert "DIRICHLET-side" in served or "Dirichlet-side" in served


def test_pure_python_backends_do_not_claim_a_wrapper():
    """The complement: a backend whose participant IS the script must not tell
    the agent to look for a binary constant that does not exist in it."""
    for name in ("fenics", "ngsolve", "skfem", "dune"):
        assert "runs the WRAPPER" not in coupling_knowledge(name), (
            f"solver='{name}': participant is a plain script, not a wrapper")


# ── through the TOOL, which is the only path an agent has ────────────────
#
# Everything above calls the payload functions directly. That is how a whole
# surface stayed broken while the suite was green: `knowledge(topic='precice',
# solver=...)` reached `get_precice_knowledge()`, which took NO solver argument
# while its caller passed one, so EVERY preCICE call — core included — returned
#   "⚠ `get_precice_knowledge()` raised: `TypeError: ... takes 0 positional
#    arguments but 1 was given`"
# a 151-character error string in place of an 11 kB payload, for all nine
# backends. Measured through the tool, the per-backend corpus was 117,632
# characters rather than the 217,492 the functions produce. Direct-call tests
# cannot see this; only driving the tool can.

def _knowledge_tool():
    """The registered `knowledge` tool function, as an agent would reach it."""
    from tools.consolidated import register_consolidated_tools

    captured = {}

    class _Recorder:
        def tool(self, *a, **k):
            def deco(fn):
                captured[fn.__name__] = fn
                return fn
            return deco

    register_consolidated_tools(_Recorder())
    return captured["knowledge"]


_ERROR_MARK = "⚠ `"


@pytest.mark.parametrize("topic", ["coupling", "precice"])
@pytest.mark.parametrize("solver", [""] + _BACKEND_ORDER)
def test_knowledge_tool_serves_a_real_payload(topic, solver):
    out = _knowledge_tool()(topic=topic, solver=solver)
    assert _ERROR_MARK not in out, (
        f"knowledge(topic={topic!r}, solver={solver!r}) served an ERROR STRING "
        f"instead of a payload: {out[:200]}")
    assert len(out) > 3000, (
        f"knowledge(topic={topic!r}, solver={solver!r}) served only {len(out)} "
        f"characters — that is not the payload, it is a stub or an error")


@pytest.mark.parametrize("topic", ["coupling", "precice"])
def test_knowledge_tool_output_matches_the_payload_function(topic):
    """The tool must serve what the tested function returns, not a second copy
    and not a legacy inline string that drifted away from it."""
    # WHAT "MATCHES" MEANS, now that the tool adds two things uniformly.
    #
    # The rule this protects is that the served text is not a SECOND COPY that
    # drifted from the tested function. It used to be checked by exact
    # equality, which also forbade any uniform addition — and the tool now
    # makes two: it front-loads a must-read and truncates at a budget, and it
    # appends the universal core that every knowledge reply carries.
    #
    # So the check becomes: strip what the tool is known to add, and whatever
    # is left must be a PREFIX of the tested function's output. A drifted copy
    # fails that immediately; a truncated faithful copy passes.
    from tools.knowledge import _UNIVERSAL_CORE, _UNIVERSAL
    from tools.consolidated import _COUPLING_MUST_READ, _MUST_READ_POINTER
    fn = coupling_knowledge if topic == "coupling" else precice_knowledge
    tool = _knowledge_tool()
    for solver in [""] + _BACKEND_ORDER:
        served, expected = tool(topic=topic, solver=solver), fn(solver)
        body = served
        for suffix in (_UNIVERSAL_CORE, _UNIVERSAL):
            if body.endswith(suffix):
                body = body[: -len(suffix)]
        cut = body.find("THIS PAYLOAD IS TRUNCATED HERE")
        if cut != -1:
            # the notice is preceded by a rule of box-drawing characters, so
            # strip those too or the "prefix" carries a line the source has not
            body = body[:body.rfind("\n", 0, cut)].rstrip("\u2500\n ")
        # the must-read leads the FIRST coupled reply of a session; every
        # later one leads with the pointer to it -- strip whichever it is
        for lead in (_COUPLING_MUST_READ, _MUST_READ_POINTER):
            if body.startswith(lead):
                body = body[len(lead):]
                break
        body = body.lstrip("\n")
        assert expected.startswith(body.rstrip()) or body.strip() in expected, (
            f"knowledge(topic={topic!r}, solver={solver!r}) is not a faithful "
            f"prefix of {fn.__name__}({solver!r}) — the served path and the "
            f"tested path have diverged")


def test_the_accelerator_advice_carries_its_measured_crossover():
    """The accelerator advice has been wrong in BOTH directions, and this pins
    the shape that is neither.

    It first told the agent that "aitken" — the tool's DEFAULT — "is not the safe
    choice here" and to use "constant" FIRST. A sweep replaced that with a blanket
    endorsement instead, and the blanket was measured on a driver that no longer
    exists: it was run on the PER-PARTICIPANT Aitken theta that
    feature/coupling-robustness replaced with one global theta, on an analytic
    stand-in for the participants rather than the participants, and its harness
    was never committed, so nothing could notice when it stopped being true.

    Re-measured on the shipped driver and the shipped participants over rho in
    {1/4, 1/2, 1, 2, 4, 6, 9} x theta in {0.1,...,1.0}, 70 settings, both
    accelerators, the same max_iter=300 and tol=1e-4 (scripts/sweep_accelerators.py):
    the truth has a CROSSOVER in it. Up to rho = 2 Aitken solved all 40 settings
    including theta = 1.0, where a constant theta reaches 1.1e+44. At rho = 6 and
    rho = 9 it solved none of the 20, and at rho = 6, theta = 1/(1+rho) = 0.143 it
    DIVERGES to 1.0e+04 where the same theta with "constant" converges in 130
    iterations.

    So both blanket statements are wrong advice, in opposite regimes, and the
    knowledge has to carry the number that separates them.
    """
    core = _core()
    first_try = core.split("| Symptom")[1].split("residual falls steadily")[0]
    assert 'keep accelerator="aitken"' in first_try, (
        "the first-try rows must still keep the adaptive DEFAULT when rho is "
        "unknown — that is the regime it was measured to win in")
    assert "USE THIS FIRST" not in core, (
        "'constant' must not be advertised as the blanket first choice")
    low = core.lower()
    assert "default and you should normally keep" in low, (
        "the knowledge must say plainly that the default accelerator is the "
        "normal choice")
    # ...and it must say WHERE that stops, with the ratio in it. A blanket
    # endorsement is what went stale last time.
    assert "rho >= 4" in core or "rho = 4" in core, (
        "the accelerator advice must name the measured ratio at which the "
        "default stops being the right choice")
    assert "scripts/sweep_accelerators.py" in core, (
        "a measured claim must name the harness that measures it, or it decays "
        "into a quotation nobody can re-run")


def test_iteration_sizing_is_a_worked_formula_not_a_lookup_table():
    """The knowledge tells the agent to size max_iter from log(tol)/log(1-theta).

    It used to follow that with six pre-evaluated values — "about 27 iterations
    at theta=0.5, 52 at theta=0.3, 83 at theta=0.2; at tol=1e-6, about
    20 / 39 / 62" — in the same breath as telling the reader to "evaluate the
    expression for your own theta and tol instead of reusing a number". The
    numbers were arithmetic, not measurements, so no contamination gate saw
    them; the objection is different. A table of six reusable figures is a
    lookup table, and the one thing an agent graded on a relaxation study will
    do with a lookup table is quote it as a result. One worked example teaches
    the method; six teach the answers.

    So: exactly one worked evaluation, it must be arithmetically right, and it
    must say in words that it is a value of the formula rather than something
    observed.
    """
    import math
    core = _core()
    assert "SIZING GUIDE" in core or "not a lower bound" in core, (
        "the figure was stated as a bound the iteration needs 'AT LEAST', and "
        "runs beat it, because the residual is normalised by the field "
        "magnitude — a good relative starting point needs FEWER iterations")
    assert "not something anybody observed" in core, (
        "the worked figure must be marked as a value of the formula; without "
        "that it reads as a measured iteration count and will be quoted as one")
    worked = re.findall(
        r"theta=([\d.]+), tol=(1e-\d+), d0=1\s*->\s*log\(1e-\d+\)/log\("
        r"[\d.]+\)\s*~\s*(\d+) iterations", core)
    assert len(worked) == 1, (
        f"expected exactly ONE worked evaluation of the sizing formula, found "
        f"{len(worked)}; a list of them is a lookup table")
    th, tol, got = worked[0]
    want = math.log(float(tol)) / math.log(1 - float(th))
    assert abs(int(got) - want) <= 1.5, (
        f"the worked figure for theta={th} at tol={tol} is quoted as {got}, "
        f"but log({tol})/log(1-{th}) = {want:.1f}")
    # And no second table sneaking back in.
    assert not re.search(r"about \d+ / \d+ / \d+", core), (
        "a slash-separated run of iteration counts is the lookup-table shape "
        "this test exists to keep out")


def test_sides_table_does_not_overstate_what_converged_here():
    """The blanket sentence under the table claimed every "yes" was a coupling
    that ran on THIS install and CONVERGED. Two of the nine rows contradicted it
    in their own text: Kratos says "in a separate Kratos install, not OASiS's own
    interpreter here", and SPARTA says the residual "cannot beat the Monte-Carlo
    noise" — i.e. `couple` reported FAILURE. The rows were honest; the summary
    over them was not, and it is the headline `discover('coupling')` serves.
    """
    table = coupling_sides_table()
    assert "Every \"yes\" above means" in table or "Every UNSTARRED" in table, (
        "the table needs a summary sentence stating what a yes is evidence of")
    # The rule, whichever way a cell is marked: an UNSTARRED yes may not sit in
    # a row whose own text takes it back. Both stars were removed once real
    # couplings backed them (pair_kratos_skfem for Kratos on Dirichlet,
    # tests/test_coupling_pair_fourc_kratos.py for Kratos on Neumann,
    # stochastic_noise_floor_makes_dsmc_gradable for SPARTA), and this is what
    # stops the next star being removed without the run behind it.
    RETRACTIONS = ("NOT reproducible", "not reproducible", "reported FAILURE",
                   "cannot beat", "could not be run", "was not run here")
    for line in table.splitlines():
        if not line.startswith("|") or "yes" not in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3 or cells[0] in ("Backend",):
            continue
        unstarred = [c for c in cells[1:3] if c.startswith("yes")
                     and not c.startswith("yes*")]
        if not unstarred:
            continue
        for phrase in RETRACTIONS:
            assert phrase not in line, (
                f"the {cells[0]} row carries an UNSTARRED yes while its own "
                f"text says {phrase!r}. Either star the cell or delete the "
                f"retraction — the summary sentence is the headline "
                f"discover('coupling') serves and it must not be contradicted "
                f"by the row underneath it.")


def test_sides_table_agrees_with_the_per_backend_payloads():
    """Both are served surfaces; they must not disagree about what was coupled.
    The table said DUNE and deal.II were "coupled to FEniCSx" while their own
    payloads said FEniCSx AND each other, in all four role/position combinations.
    """
    table = coupling_sides_table()
    for label, name, partner in (("DUNE-fem", "dune", "deal.II"),
                                 ("deal.II", "dealii", "DUNE-fem")):
        row = next(r for r in table.splitlines() if r.startswith(f"| {label}"))
        assert partner in row, (
            f"the table omits that {label} was coupled to {partner}, which "
            f"knowledge(topic='coupling', solver='{name}') states")


def test_sides_table_covers_every_backend():
    table = coupling_sides_table()
    for label in ("FEniCSx", "NGSolve", "scikit-fem", "DUNE", "deal.II",
                  "4C", "FEBio", "Kratos", "SPARTA"):
        assert label in table, f"{label} is absent from the side table"


def test_complete_templates_contain_no_task_result_artifacts():
    """Runnable generic templates are capability, not a solved benchmark."""
    from tools.coupling_knowledge import coupling_knowledge as _ck
    for solver in _BACKEND_ORDER:
        payload = _ck(solver)
        assert "EDIT THIS BLOCK" in payload
        assert "PLACEHOLDER" in payload
        for artifact in ("solution_level1_A.csv", "INTERFACE_RESIDUAL =",
                         "MESH_INDEPENDENCE = CONVERGED"):
            assert artifact not in payload, (
                f"solver={solver!r}: generic template contains task result "
                f"artifact {artifact!r}")


def test_the_interface_contract_is_still_served_everywhere():
    """Cutting the solver must not cut the contract with it."""
    from tools.coupling_knowledge import coupling_knowledge as _ck
    for solver in ("fenics", "ngsolve", "skfem", "dune", "dealii",
                   "fourc", "febio", "kratos"):
        text = _ck(solver)
        for need in ("imports.json", "exports.json"):
            assert need in text, f"solver={solver!r} lost {need!r}"
