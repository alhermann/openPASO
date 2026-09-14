"""Name, at write time, the API calls a participant script is about to die on.

WHY THIS EXISTS, MEASURED. Round 49 (the first honest round on problems other than C1-C3) was lost to
wall clock, not to knowledge: every one of the nine cells hit the 45-minute limit at 35 to 73 seconds
per action, bought 37 to 75 actions, and exactly one reached couple() at all. The action sink is the
write-run-error-rewrite loop on the participant script -- and when all 58 saved step-trial fills were
re-executed and graded the way the tasks grade, 46 of them never wrote an export and EVERY ONE of those
died on an invented API call, not on the physics.

Each call below was measured on THIS install, and each finding names the error the run will print and
the call that works. The same facts are served through the coupling door; this is the moment they are
most useful, which is before the run that would have taught them.

WHAT THIS IS NOT. It does not judge the mesh, the weak form, the material, the source or the solve --
those are the agent's and OASiS has no business dictating them. It reads the file and writes nothing.
"""
from __future__ import annotations

import re

# (backend, pattern, what the run prints, the call that works)
_TRAPS: tuple[tuple[str, str, str, str], ...] = (
    # ── scikit-fem ────────────────────────────────────────────────────────
    ("skfem", r"\.dofs\s*\(",
     "TypeError: 'Dofs' object is not callable",
     "basis.dofs is an ATTRIBUTE; select with basis.get_dofs(facets=<facet indices>) or "
     "basis.get_dofs(lambda x: np.isclose(x[0], IFACE_X)), then .flatten()"),
    ("skfem", r"\.assemble\s*\(\s*basis\s*=",
     "TypeError: BilinearForm._assemble() missing 1 required positional argument: 'ubasis'",
     "a form takes its basis POSITIONALLY: laplace.assemble(basis) or asm(laplace, basis)"),
    ("skfem", r"\bw\.y\b",
     "AttributeError: Attribute 'y' not found in 'w'",
     "inside a form the coordinates are w.x[0] and w.x[1]; there is no w.y"),
    ("skfem", r"\binit_rect\s*\(",
     "AttributeError: type object 'MeshTri1' has no attribute 'init_rect'",
     "a rectangle is MeshTri.init_tensor(np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1))"),
    ("skfem", r"\bdoforder\s*=",
     "TypeError: CellBasis.__init__() got an unexpected keyword argument 'doforder'",
     "Basis(mesh, element) takes no doforder keyword"),
    ("skfem", r"\bfind_dofs\s*\(",
     "AttributeError: 'FacetBasis' object has no attribute 'find_dofs'",
     "every basis has get_dofs, not find_dofs"),
    ("skfem", r"\bmesh\.f\b(?!acets|2)",
     "AttributeError: 'MeshTri1' object has no attribute 'f'",
     "the facet node table is mesh.facets (2 x n_facets); mesh.p, mesh.t, mesh.t2f and mesh.f2t are the others"),
    # ── NGSolve ───────────────────────────────────────────────────────────
    ("ngsolve", r"\.AddVertex\s*\(",
     "AttributeError: 'SplineGeometry' object has no attribute 'AddVertex'",
     "a rectangle is geo.AddRectangle((X0, Y0), (X1, Y1), bcs=(bottom, right, top, left)); "
     "single points are AddPoint(x, y) / AppendPoint(x, y), with SEPARATE coordinates"),
    ("ngsolve", r"\.AddRect\s*\(",
     "AttributeError: 'SplineGeometry' object has no attribute 'AddRect'",
     "the call is AddRectangle, spelled in full"),
    ("ngsolve", r"\.Faces\s*\(\s*\)|\.Vertices\s*\(\s*\)|\.Edges\s*\(\s*\)",
     "AttributeError: 'Mesh' object has no attribute 'Faces' (or 'Vertices')",
     "the mesh iterators are LOWERCASE properties: mesh.vertices, mesh.faces, mesh.edges"),
    ("ngsolve", r"\bfes\.Dofs\s*\(",
     "AttributeError: 'H1' object has no attribute 'Dofs'",
     "a vertex's dof is fes.GetDofNrs(NodeId(VERTEX, v.nr))[0]; the free set is fes.FreeDofs()"),
    ("ngsolve", r"\.vertexnr\b|NodeId\([^)]*\)\.index\b|\bv\.index\b",
     "AttributeError: the node object has no 'vertexnr' / 'index'",
     "a MeshNode carries .nr (its number) and .point (its coordinates)"),
    ("ngsolve", r"\bimport\s*\([^)]*\binverse\b|,\s*inverse\s*[,)]\s*$",
     "ImportError: cannot import name 'inverse' from 'ngsolve'",
     "`inverse` is a KEYWORD of Inverse: a.mat.Inverse(fes.FreeDofs(), inverse='sparsecholesky')"),
    ("ngsolve", r"\.vec\.vec\b",
     "AttributeError: 'BaseVector' object has no attribute 'vec'",
     "a LinearForm's vector is f.vec and it is already the BaseVector"),
    ("ngsolve", r"\.vec\s*\[[^]]*FreeDofs",
     "TypeError: __getitem__(): incompatible function arguments (a BitArray is not an index)",
     "take numbers out with v.FV().NumPy() or np.array(v), and assign through v.data"),
    ("ngsolve", r"CoefficientFunction\s*\(\s*(lambda|[A-Z_]+SRC)|(?<![A-Za-z_])CF\s*\(\s*lambda",
     "TypeError: incompatible constructor arguments (a Python function is not a CoefficientFunction)",
     "sample the function at the vertices into a P1 GridFunction and integrate that; a GridFunction "
     "IS a CoefficientFunction"),
    # ── FEniCSx ───────────────────────────────────────────────────────────
    ("fenics", r"\.subset_dofs\s*\(",
     "AttributeError: 'FunctionSpace' object has no attribute 'subset_dofs'",
     "fem.locate_dofs_topological(V, fdim, facets) or fem.locate_dofs_geometrical(V, marker)"),
    ("fenics", r"ufl\.FiniteElement\s*\(",
     "AttributeError: module 'ufl' has no attribute 'FiniteElement'",
     "fem.functionspace(mesh, ('Lagrange', 1)), or basix.ufl.element(...)"),
    ("fenics", r"fem\.VectorFunctionSpace\s*\(",
     "AttributeError: module 'dolfinx.fem' has no attribute 'VectorFunctionSpace'",
     "fem.functionspace(mesh, ('Lagrange', 1, (mesh.geometry.dim,)))"),
    ("fenics", r"\.geometric_dimension\s*\(",
     "AttributeError: a UFL argument has no geometric_dimension()",
     "take the dimension from the mesh: mesh.geometry.dim"),
    # ── DUNE-fem ──────────────────────────────────────────────────────────
    ("dune", r"ufl\.Eq\s*\(",
     "ImportError / AttributeError: ufl.Eq no longer exists",
     "write the equation with Python's ==: galerkin([a == L, dbc], solver='cg'); ufl.eq is only a "
     "conditional's test"),
    ("dune", r"DirichletBC\([^)]*\bmarker\s*=",
     "TypeError: DirichletBC() got an unexpected keyword argument 'marker'",
     "the signature is DirichletBC(space, value, subDomain=None)"),
    ("dune", r"\binfo\.converged\b",
     "AttributeError: 'dict' object has no attribute 'converged'",
     "scheme.solve returns a DICT: read info['converged']"),
    ("dune", r"\.geometry\.point\b",
     "AttributeError: the Geometry object has no attribute 'point'",
     "a vertex's coordinates are vertex.geometry.center (or .corner(0))"),
    ("dune", r"\bspace\.dim\b|\.space\.dim\b",
     "AttributeError: the space has no attribute 'dim'",
     "a discrete space counts its dofs with space.size (or len(space))"),
    ("dune", r"\bdofCoordinates\s*\(",
     "AttributeError: the space has no attribute 'dofCoordinates'",
     "dof coordinates come from the grid: iterate gridView.vertices and read v.geometry.center, "
     "numbering with gridView.indexSet.index(v)"),
    # ── deal.II (C++, but the trap is what the build dies on) ─────────────
    ("dealii", r"\bFEEvaluation\s*<",
     "a page of template errors inside the deal.II headers (synchronous_iterator.h), not in your file",
     "assemble with FEValues<2> fe_values(fe, quadrature, update_flags), not FEEvaluation -- that is "
     "the matrix-free class and has a different contract"),
    # ── Kratos ────────────────────────────────────────────────────────────
    ("kratos", r"LaplacianElement2D4N",
     "Kratos error: LaplacianElement2D4N is not registered",
     "2-D conduction is P1 TRIANGLES: LaplacianElement2D3N (3-D is LaplacianElement3D4N)"),
)

_IMPORT_MARKERS = {
    "skfem": ("from skfem", "import skfem"),
    "ngsolve": ("from ngsolve", "import ngsolve", "from netgen"),
    "fenics": ("from dolfinx", "import dolfinx"),
    "dune": ("from dune", "import dune"),
    "kratos": ("import KratosMultiphysics", "from KratosMultiphysics"),
    "dealii": ("#include <deal.II/", "dealii::", "using namespace dealii"),
}


def looks_like_participant(text: str) -> bool:
    """A coupling participant: it speaks the driver's handshake."""
    return "imports.json" in text and "exports.json" in text


def backends_in(text: str) -> list:
    return [b for b, marks in _IMPORT_MARKERS.items() if any(m in text for m in marks)]


def _strip_strings_and_comments(text: str) -> str:
    """Docstrings and comments quote the WRONG calls on purpose -- every served contract does, and so
    does this module. Linting them would name a defect that is not there, and a gate that names an
    absent defect costs the agent an action for nothing (measured, honest rounds)."""
    out = re.sub(r'"""(?:.|\n)*?"""', "", text)
    out = re.sub(r"'''(?:.|\n)*?'''", "", out)
    out = re.sub(r"#[^\n]*", "", out)
    return out


_MODULE_USE = (
    ("dolfinx", r"\bdolfinx\.[A-Za-z_]", "import dolfinx",
     "NameError: name 'dolfinx' is not defined",
     "`from dolfinx import fem, mesh` does NOT bind the module: add a bare `import dolfinx` when you "
     "call dolfinx.log.set_log_level(...)"),
    ("ngsolve", r"\bngsolve\.[A-Za-z_]", "import ngsolve",
     "NameError: name 'ngsolve' is not defined",
     "`from ngsolve import ...` does NOT bind the module: add a bare `import ngsolve` when you set "
     "ngsolve.ngsglobals.msg_level"),
    ("logging", r"\blogging\.[A-Za-z_]", "import logging",
     "NameError: name 'logging' is not defined",
     "add `import logging` before logging.basicConfig(level=logging.INFO)"),
)


def _module_findings(body: str, source: str) -> list:
    """A module used by name that the script never imported.

    Measured 2026-09-14: a FEniCSx worker followed both the task and the served facts, wrote
    dolfinx.log.set_log_level(...) at the top of the contract, and died with NameError -- the served
    contracts did `from dolfinx import fem` and never bound the module. They bind it now; a script
    written from scratch still may not."""
    # AN IMPORT LINE IS NOT A USE. `from dolfinx.fem.petsc import LinearProblem` contains
    # "dolfinx." and binds nothing by that name -- scanning it flagged four served contracts that
    # are correct (measured while writing this).
    body = "\n".join(l for l in body.split("\n")
                     if not re.match(r"\s*(from|import)\s", l))
    out = []
    for mod, use, imp, error, fix in _MODULE_USE:
        if not re.search(use, body):
            continue
        if re.search(rf"^\s*{imp}\b", source, re.M) or re.search(rf"^\s*import\s+[^\n]*\b{mod}\s+as\b", source, re.M):
            continue
        out.append(f"{mod}: this script calls `{mod}....` but never imports the module -- "
                   f"the run stops with {error}. {fix}. Measured on this install.")
    return out


def undefined_names(text: str) -> list:
    """Names the script USES at module level and never defines anywhere.

    This is the leave-behind contract, checked. The served contract names every variable its surviving
    lines still use -- iface_dofs, y_if, a, b_vol and the rest -- and says plainly that the hole has to
    define them. Measured 2026-09-14 on the DUNE 3-D step trial: a worker's script called
    resample_plane(imp, "normal_fluxes", Q_INIT, pts) with no `pts` anywhere, and the run died with
    UnboundLocalError after the JIT had already spent minutes compiling. A name that is never assigned
    cannot be found by any amount of running.

    Deliberately conservative: module level only, and a name bound ANYWHERE in the file counts as
    defined. It reports what cannot work, not what looks unusual.
    """
    import ast
    import builtins
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return [f"this file does not parse: {e.__class__.__name__} at line {e.lineno} -- {e.msg}"]
    bound = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            bound.add(getattr(node, "name", ""))
            a = getattr(node, "args", None)
            if a is not None:
                for arg in list(a.args) + list(a.posonlyargs) + list(a.kwonlyargs):
                    bound.add(arg.arg)
                for extra in (a.vararg, a.kwarg):
                    if extra is not None:
                        bound.add(extra.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                bound.add((al.asname or al.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.Global):
            bound.update(node.names)
    # module-level loads only: function bodies may legitimately read module names defined later
    missing = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load) and sub.id not in bound:
                missing.setdefault(sub.id, sub.lineno)
    return [f"`{n}` is used at line {ln} and never defined in this file -- the run stops with "
            f"NameError/UnboundLocalError there. The served contract lists the names its surviving "
            f"lines need; every one of them has to come out of your solve."
            for n, ln in sorted(missing.items(), key=lambda kv: kv[1])]


# The error text a run prints -> the call that works. Same measurements as _TRAPS, keyed the other
# way round: a run that already failed should not cost a second run to diagnose.
_ERROR_FIXES: tuple = (
    ("'Dofs' object is not callable",
     "scikit-fem: basis.dofs is an ATTRIBUTE. Select with basis.get_dofs(facets=...) or "
     "basis.get_dofs(lambda x: ...), then .flatten()"),
    ("missing 1 required positional argument: 'ubasis'",
     "scikit-fem: a form takes its basis POSITIONALLY -- laplace.assemble(basis), or asm(laplace, basis)"),
    ("Attribute 'y' not found in 'w'",
     "scikit-fem: inside a form the coordinates are w.x[0] and w.x[1]; there is no w.y"),
    ("has no attribute 'init_rect'",
     "scikit-fem: a rectangle is MeshTri.init_tensor(np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1))"),
    ("unexpected keyword argument 'doforder'",
     "scikit-fem: Basis(mesh, element) takes no doforder keyword"),
    ("has no attribute 'find_dofs'",
     "scikit-fem: every basis has get_dofs, not find_dofs"),
    ("'MeshTri1' object has no attribute 'f'",
     "scikit-fem: the facet node table is mesh.facets; mesh.p, mesh.t, mesh.t2f and mesh.f2t are the others"),
    ("has no attribute 'AddVertex'",
     "NGSolve: a rectangle is geo.AddRectangle((X0, Y0), (X1, Y1), bcs=(bottom, right, top, left)); "
     "single points are AddPoint(x, y) with SEPARATE coordinates"),
    ("has no attribute 'AddRect'",
     "NGSolve: the call is AddRectangle, spelled in full"),
    ("object has no attribute 'Faces'",
     "NGSolve: the mesh iterators are lowercase properties -- mesh.vertices, mesh.faces, mesh.edges"),
    ("object has no attribute 'Vertices'",
     "NGSolve: the mesh iterators are lowercase properties -- mesh.vertices, mesh.faces, mesh.edges"),
    ("MeshNode' object has no attribute",
     "NGSolve: a MeshNode carries .nr and .point; a vertex's dof is fes.GetDofNrs(NodeId(VERTEX, v.nr))[0]"),
    ("cannot import name 'inverse' from 'ngsolve'",
     "NGSolve: `inverse` is a KEYWORD of Inverse, not an import -- "
     "a.mat.Inverse(fes.FreeDofs(), inverse='sparsecholesky')"),
    ("BaseVector' object has no attribute 'vec'",
     "NGSolve: a LinearForm's vector IS f.vec; assign through .data and read numbers with FV().NumPy()"),
    ("must not have TrialFunction",
     "NGSolve: a LinearForm carries only the test function; every term with the trial function belongs "
     "in the BilinearForm"),
    ("has no attribute 'subset_dofs'",
     "FEniCSx: fem.locate_dofs_topological(V, fdim, facets) or fem.locate_dofs_geometrical(V, marker)"),
    ("has no attribute 'FiniteElement'",
     "FEniCSx: build spaces with fem.functionspace(mesh, ('Lagrange', 1))"),
    ("has no attribute 'VectorFunctionSpace'",
     "FEniCSx: fem.functionspace(mesh, ('Lagrange', 1, (mesh.geometry.dim,)))"),
    ("missing 1 required keyword-only argument: 'petsc_options_prefix'",
     "FEniCSx: LinearProblem requires petsc_options_prefix='<any name>' on this install (measured "
     "exactly this message)"),
    ("'LinearProblem' object has no attribute '_solver'",
     "FEniCSx: the solver is the PUBLIC p.solver; touching p._solver raises"),
    ("has no attribute 'geometry.point'",
     "DUNE-fem: a vertex's coordinates are vertex.geometry.center (or .corner(0))"),
    ("'Geometry' object has no attribute 'point'",
     "DUNE-fem: a vertex's coordinates are vertex.geometry.center (or .corner(0))"),
    ("'Form' object has no attribute 'copy'",
     "DUNE-fem: a UFL form is immutable -- build the second one by writing the expression again"),
    ("has no attribute 'converged'",
     "DUNE-fem: scheme.solve returns a DICT -- read info['converged']"),
    ("has no attribute 'dim'",
     "DUNE-fem: a discrete space counts its dofs with space.size (or len(space))"),
    ("has no attribute 'dofCoordinates'",
     "DUNE-fem: dof coordinates come from the grid -- gridView.vertices with v.geometry.center, "
     "numbered by gridView.indexSet.index(v)"),
    ("synchronous_iterator.h",
     "deal.II: that wall of template errors is a constructor mistake in YOUR file, usually "
     "FEEvaluation where FEValues belongs"),
    ("request for member \u2018unsubscribe\u2019",
     "deal.II: read the FIRST error and check your constructor -- FEValues<2>(fe, quadrature, "
     "update_flags), in that order"),
    ("is not registered",
     "Kratos: 2-D conduction is P1 TRIANGLES (LaplacianElement2D3N); 3-D is LaplacianElement3D4N"),
)


def findings_from_output(output: str) -> list:
    """What a run already told you, with the call that works.

    A failed run has bought the diagnosis; spending a second run to learn it is the loop that ate
    round 49. Names at most three, because an agent acts on the first."""
    if not isinstance(output, str) or not output.strip():
        return []
    out, seen = [], set()
    for needle, fix in _ERROR_FIXES:
        if needle in output and fix not in seen:
            seen.add(fix)
            out.append(f"the run printed `{needle}` -- {fix}. Measured on this install.")
        if len(out) >= 3:
            break
    return out


def participant_findings(text: str) -> list:
    """Measured API traps present in this script, named with the error and the working call."""
    if not isinstance(text, str) or not text.strip():
        return []
    codes = backends_in(text)
    if not codes:
        return []
    body = _strip_strings_and_comments(text)
    out, seen = _module_findings(body, text) + undefined_names(text), set()
    for backend, pattern, error, fix in _TRAPS:
        if backend not in codes or (backend, pattern) in seen:
            continue
        m = re.search(pattern, body, re.M)
        if not m:
            continue
        seen.add((backend, pattern))
        line = body[:m.start()].count("\n") + 1
        out.append(f"{backend}: `{m.group(0).strip()}` (near line {line} of the stripped source) -- "
                   f"this run will stop with {error}. {fix}. Measured on this install.")
    return out
