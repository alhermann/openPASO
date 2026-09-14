"""A template for a named code must DRIVE that code, not merely disclose that
it doesn't.

This is the stronger sibling of test_no_undisclosed_surrogates, and it exists
because that test passed the whole time the defect was live. Its rule is "if a
template does not use its own backend, its docstring must say so", and the
worst offender said so: kratos/generators/heat.py's first line read

    Heat conduction - Kratos (manual assembly)

which is honest, disclosed, and still the wrong artefact. Six Kratos generators
emitted a numpy/scipy assembly with no `import KratosMultiphysics` anywhere in
them: heat (steady and transient), poisson, linear_elasticity,
structural_dynamics, cosimulation, shape_optimization.

WHAT IT COST, measured, because this is the fifth time a mechanism existed and
did not reach the case it was built for. The blind campaign's only CORRECT
coupled run, C2_27b_MCP_seed15, was graded on second-order numbers and credited
to the two codes its task prescribed. Its participants contain 0 references to
4C, 0 to Kratos and 40 hand-rolled-assembly markers across 11 files, and they
are a structural match to the heat.py template -- the same node_map[(i,j)]
loop, the same two-triangles-per-quad split [n1,n2,n4]/[n2,n3,n4], the same
`area = 0.5*abs(...)` and the same b/c gradient rows. The agent did not invent
a solver. openPASO served one, the agent used what it was served, and the
attribution gate then recorded per_code_attribution=UNPROVEN and graded the
numbers anyway. Disclosure in a docstring cannot fix that: a submission is
attributable or it is not, and a task that names a code is failed by an
artefact that cannot run it.

So the rule here is about the ARTEFACT THE AGENT ENDS UP WITH: every emitted
template must carry its backend's marker, unless it is one of the entries that
is explicitly not a runnable input at all (umbrella groupings and 4C reference
stubs that decline to fabricate a mesh they cannot know -- declining is
honesty, and this guard must not accuse it).
"""
from __future__ import annotations

import ast
import functools
import re

import pytest

# The same markers the sibling test uses, so the two cannot drift apart on what
# "drives its own backend" means.
_BACKEND_MARKERS = {
    "fenics": ("dolfinx", "ufl"),
    "dune": ("dune",),
    "ngsolve": ("ngsolve", "netgen"),
    "skfem": ("skfem",),
    "kratos": ("KratosMultiphysics",),
    "febio": ("febio_spec", "<febio_spec"),
    "fourc": ("PROBLEM TYPE", "PROBLEMTYPE"),
    "sparta": ("create_box", "species", "global "),
}

# NOT a runnable input, and says so. These are allowed to carry no marker.
_NOT_RUNNABLE = re.compile(
    r"not\s+(?:a\s+)?(?:\w+\s+){0,3}runnable"
    r"|reference stub|meta-reference|umbrella|catalog placeholder"
    r"|must supply|user must provide", re.I)

# A solver written in the template instead of called.
_HAND_ROLLED = re.compile(r"\blil_matrix\b|\bspsolve\b|\bcsr_matrix\b"
                          r"|np\.linalg\.solve|scipy\.sparse\.linalg")


_IMPORT_FAILURES: list[str] = []


@functools.lru_cache(maxsize=1)
def _templates():
    """Every emitted template, read WITHOUT needing the solver installed.

    The first version of this went through core.registry, like the sibling
    test. In the interpreter the suite actually runs in, that enumerated 235
    templates across eight backends and EXACTLY ZERO Kratos ones, because
    `get_backend("kratos")` returns None wherever KratosMultiphysics is not
    importable — and it is not importable in the suite's environment, only in
    the one the campaign hands to a Kratos run. A guard written to catch a
    Kratos defect that cannot see a single Kratos template is worse than no
    guard: it reports success. That is the same shape as every earlier
    instance in this repo — the mechanism existed, was instrumented, and did
    not reach the case it was built for.

    So the generator MODULES are imported directly and their template
    functions called. They emit text; none of them needs the solver present.
    """
    import importlib
    import inspect
    import pathlib

    out = []
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "backends"
    for gen in sorted(root.glob("*/generators/*.py")):
        backend = gen.parent.parent.name
        if gen.name == "__init__.py":
            continue
        try:
            mod = importlib.import_module(
                f"backends.{backend}.generators.{gen.stem}")
        except Exception as exc:
            # NOT swallowed. A silent `continue` here is how a whole backend
            # went missing: one SyntaxError in linear_elasticity.py broke
            # kratos/generators/__init__, every kratos module failed to
            # import, and the enumeration quietly reported 3 templates
            # instead of 13 while all the assertions below passed.
            _IMPORT_FAILURES.append(f"{backend}/{gen.stem}: "
                                    f"{type(exc).__name__}: {exc}")
            continue
        for name, fn in vars(mod).items():
            if not (name.startswith("_") and callable(fn)):
                continue
            if inspect.getmodule(fn) is not mod:
                continue                       # imported helper, not a template
            try:
                sig = inspect.signature(fn)
            except (TypeError, ValueError):
                continue
            if len(sig.parameters) != 1:
                continue
            try:
                text = fn({})
            except Exception:
                continue                       # generator failures: other tests
            if isinstance(text, str) and len(text) > 200:
                out.append((backend, f"{gen.stem}:{name}", text))
    # The module walk finds module-level `_name(params)` templates. Backends
    # whose generators are CLASSES (4C: ThermoGenerator and friends) expose
    # nothing that shape, so the registry route is added rather than replaced.
    try:
        from core.registry import get_backend, list_backends, load_all_backends
        load_all_backends()
        for entry in list_backends():
            be = get_backend(entry["name"])
            if not be:
                continue
            for phys in be.supported_physics():
                for var in list(phys.template_variants):
                    try:
                        txt = be.generate_input(phys.name, var, {})
                    except Exception:
                        continue
                    if isinstance(txt, str) and len(txt) > 200:
                        out.append((entry["name"],
                                    f"registry:{phys.name}/{var}", txt))
    except Exception:
        pass
    return tuple(out)


def test_the_enumeration_is_not_empty_for_any_backend_that_ships_generators():
    """The guard on the guard. Written because the registry route silently
    excluded the entire backend this file exists for."""
    import pathlib
    from collections import Counter

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "backends"
    shipped = {p.parent.parent.name for p in root.glob("*/generators/*.py")
               if p.name != "__init__.py"}
    seen = Counter(b for b, _, _ in _templates())
    missing = sorted(b for b in shipped if not seen.get(b))
    assert not missing, (
        f"these backends ship generator modules but contributed NO template "
        f"to the enumeration, so every assertion below is vacuous for them: "
        f"{missing}\nenumerated: {dict(seen)}")
    assert not _IMPORT_FAILURES, (
        "generator modules that FAILED TO IMPORT, so their templates were "
        "never judged:\n  " + "\n  ".join(_IMPORT_FAILURES))
    assert seen.get("kratos", 0) >= 10, (
        f"only {seen.get('kratos', 0)} kratos templates enumerated; this file "
        f"exists because six kratos generators emitted a scipy solver, and "
        f"the registry route saw zero of them")


def _header(text: str) -> str:
    try:
        doc = ast.get_docstring(ast.parse(text))
        if doc:
            return doc
    except (SyntaxError, ValueError):
        pass
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#") or not s:
            if s.strip("#= \t-*_"):
                lines.append(s)
            if len(lines) > 25:
                break
        elif lines:
            break
    return "\n".join(lines) or text[:400]


def test_no_template_hand_rolls_a_solver_for_a_code_it_names():
    """The defect itself: a script that solves the problem with scipy while
    being the answer to "how do I solve this with <code>".

    Judged on the emitted text, not on the generator source, because the
    emitted text is what the agent runs and hands in.
    """
    offenders = []
    for backend, name, text in _templates():
        markers = _BACKEND_MARKERS.get(backend)
        if not markers:
            continue
        if any(m in text for m in markers):
            continue                            # it drives its backend
        if not _HAND_ROLLED.search(text):
            continue                            # not a solver at all
        offenders.append(f"{backend}/{name}")
    assert not offenders, (
        "these templates assemble and solve the problem themselves while "
        "standing in for a named code, so nothing produced from them can be "
        "attributed to that code:\n  " + "\n  ".join(offenders)
        + "\n\nA docstring saying 'manual assembly' does NOT make this "
          "acceptable: the sibling test test_no_undisclosed_surrogates "
          "accepted exactly that wording, and the campaign's one CORRECT "
          "coupled run was produced by a template it passed. Emit a script "
          "that calls the code. An independent assembly is a CHECK on the "
          "solver's answer and a fatal substitute for it.")


def test_every_runnable_template_carries_its_backend_marker():
    """Wider than the test above: a template need not hand-roll a solver to be
    unattributable — one that merely prints a summary would also be. Anything
    presented as runnable must show its backend.
    """
    offenders = []
    for backend, name, text in _templates():
        markers = _BACKEND_MARKERS.get(backend)
        if not markers or any(m in text for m in markers):
            continue
        if _NOT_RUNNABLE.search(_header(text)):
            continue                            # declared non-runnable stub
        offenders.append(f"{backend}/{name}: "
                         f"{_header(text).splitlines()[0][:80]!r}")
    assert not offenders, (
        "these templates are presented as runnable and carry no sign of "
        "their own backend:\n  " + "\n  ".join(offenders)
        + "\n\nEither call the code, or say plainly in the header that the "
          "entry is not a runnable input.")


def test_the_kratos_templates_that_were_the_defect_now_drive_kratos():
    """Named, so a revert cannot pass quietly.

    Reads the generator modules directly, NOT the registry — the registry
    cannot see the kratos backend wherever KratosMultiphysics is absent, which
    is precisely the suite's own environment.

    Each was verified by EXECUTION when it was repaired, and the numbers are
    recorded here so a change that keeps the import but breaks the physics is
    still visible in the diff:

        poisson        -lap u = 1 on the unit square -> max u = 7.3446e-02
                       against the exact 7.36713e-02 (0.31% at h = 1/16)
        heat/transient EulerianDiffusion2D3N, midpoint T = 0.0835 -> 11.86 ->
                       35.99 -> 46.88 at steps 1/5/15/30, approaching the
                       steady-state 50 from below. LaplacianElement2D3N has NO
                       capacity term, so a transient built on it returns the
                       steady answer at every step and the history is flat.
        elasticity     40x4 cantilever, E = 1e5, nu = 0.3, tip shear 1 ->
                       max|u| = 3.9010e-02 against Euler-Bernoulli 4.0e-02
        dynamics       Bossak, tip uy -5.3486e-04 -> -8.1485e-02 over 20 steps
        cosimulation   both fields solved by Kratos in real subprocesses
        shape_opt      two descent steps, J 3.3155e-02 -> 3.3127e-02, every
                       design evaluation a Kratos subprocess

    linear_elasticity:_elasticity_nonlinear_kratos is deliberately NOT in this
    list. The real Kratos route for it does not converge on this install --
    max iterations on all five load steps, displacement ratio stuck near
    1.11e-02, reporting tip uy -2.2917e+00 where the verified linear answer on
    the same mesh and load is 3.3040e-02 -- so it ships as a declared
    non-runnable reference entry instead of a template that returns an
    unconverged iterate.
    """
    FAMILIES = {"poisson", "heat", "linear_elasticity", "structural_dynamics",
                "cosimulation", "shape_optimization"}
    checked = []
    for backend, name, text in _templates():
        if backend != "kratos":
            continue
        if name.split(":")[0] not in FAMILIES:
            continue
        if _NOT_RUNNABLE.search(_header(text)):
            continue
        checked.append(name)
        assert "KratosMultiphysics" in text, (
            f"kratos/{name} does not import KratosMultiphysics — this is the "
            f"exact defect that produced the campaign's fake coupled pass")
        assert not _HAND_ROLLED.search(text), (
            f"kratos/{name} still contains a hand-rolled solver")
    assert len(checked) >= 6, (
        f"only {len(checked)} runnable templates from the six repaired "
        f"families were reachable ({checked}); a rename would make this test "
        f"vacuous, which is how the original defect survived a guard that "
        f"already existed")


def test_the_guard_fires_on_the_original_defect_and_spares_the_honest_cases():
    """Calibration. A guard nobody has seen fire is not a guard.

    The first string is the shape of the real defect, taken from what
    kratos/generators/heat.py emitted before the repair: a disclosed manual
    assembly standing in for Kratos. The sibling test accepted it for months.
    The rest must NOT be accused -- a guard that flagged the 4C reference
    stubs, or the umbrella groupings, would be deleted inside a week.
    """
    defect = ('"""Heat conduction - Kratos (manual assembly)"""\n'
              "import numpy as np\n"
              "from scipy.sparse import lil_matrix\n"
              "from scipy.sparse.linalg import spsolve\n"
              + "K = lil_matrix((n, n))\n" * 60)
    assert _HAND_ROLLED.search(defect), (
        "the hand-rolled-solver pattern no longer matches the defect this "
        "file was written for")
    assert "KratosMultiphysics" not in defect
    assert not _NOT_RUNNABLE.search(_header(defect)), (
        "'(manual assembly)' must NOT count as declaring the entry "
        "non-runnable -- that is exactly the wording the defect hid behind")

    for honest in (
        "# 4C reference stub: ssi / monolithic_elch_3d\n"
        "# Not a runnable input - the user must supply the mesh\n" + "#\n" * 60,
        "# 4C umbrella / meta-reference physics: 'thermal'\n"
        "# This is NOT a runnable 4C input.\n" + "#\n" * 60,
        '"""Poisson - Kratos"""\nimport KratosMultiphysics as KM\n'
        + "mp.CreateNewElement('LaplacianElement2D3N', ...)\n" * 60,
    ):
        spared = (_NOT_RUNNABLE.search(_header(honest))
                  or "KratosMultiphysics" in honest)
        assert spared, f"the guard would accuse an honest entry: {honest[:70]!r}"
