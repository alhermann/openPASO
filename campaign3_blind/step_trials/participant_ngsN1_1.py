"""NGSolve participant for the OASiS `couple` driver.

Steady heat conduction  -div(k grad T) = f  on one rectangular subdomain.
CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.
"""
import json
from pathlib import Path

import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import (VERTEX, BilinearForm, GridFunction, H1, LinearForm, Mesh,
                     NodeId, TaskManager, ds, dx, grad)

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side; the payload that served
SIDE      = "neumann"     # this copy takes the OTHER side
PARTNER   = "left"        # the name you gave the first participant
X0, X1    = 0.6, 1.4      # the OTHER subdomain: starts where the first ends
Y0, Y1    = 0.0, 1.0      # same y-extent as the partner
IFACE_X   = 0.6           # SAME interface coordinate as the partner
K         = 5.0           # this subdomain's own material


def F_SRC(x, y):
    """Volumetric source, as a function of position.

    Returns zero as shipped, which is a PLACEHOLDER like every number above
    and is almost never what your problem wants. THIS KNOB USED TO BE A SCALAR
    CONSTANT, AND A CONSTANT CANNOT REPRESENT A SOURCE THAT VARIES WITH
    POSITION: the source of a manufactured solution is a POLYNOMIAL in x and y,
    and no single number is that polynomial. Left at zero the temperature is
    harmonic, the outer Dirichlet values are the only data left in the problem,
    and the answer degenerates to the 1-D profile between them — the interface
    flux is one constant along the whole interface, and it is identically zero
    when the two subdomains carry the same outer value. The coupling will
    converge beautifully to that, and it is not the problem you were given.

    If your problem states a source, or gives you a manufactured solution whose
    source term you derived, put it here. `x` and `y` are NumPy arrays, so
    build the answer with NumPy and return ONE array of the same shape (write
    `0.0 * x + c` for a genuine constant, never a bare `c`):

        # -div(K grad T) for the manufactured T = x**3 * y**2
        return -K * (6.0 * x * y**2 + 2.0 * x**3)
    """
    return K * np.pi**2 * (1.4 - x) * np.sin(np.pi * y)


T_OUTER   = 0.0           # Dirichlet on ITS outer boundary (here x = 1.4)
NX, NY    = 8, 10         # its own mesh — deliberately NOT the partner's
T_INIT    = 310.0
Q_INIT    = 0.0
# ─────────────────────────────────────────────────────────────────────────

MAXH  = min((X1 - X0) / NX, (Y1 - Y0) / NY)   # netgen's scalar mesh size
ORDER = 1                                     # H1 order (nodal == vertex dofs)

ON_RIGHT = abs(IFACE_X - X1) < abs(IFACE_X - X0)   # interface is this side's x-max?
OUTER_X = X0 if ON_RIGHT else X1
S = 1.0 if ON_RIGHT else -1.0              # outward normal at interface = S * e_x
TOL = 1e-9 * max(X1 - X0, Y1 - Y0)


def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return d.get(PARTNER) or None


def sample(imp, key, fallback, y):
    """Interpolate the partner's samples onto this participant's y-coordinates."""
    if not imp or not imp.get("coordinates"):
        return np.full(len(y), float(fallback))
    ys = np.array([c[1] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key, []), float).ravel()
    if vs.size != ys.size:
        return np.full(len(y), float(fallback))
    o = np.argsort(ys)
    return np.interp(y, ys[o], vs[o])


imp = read_imports()

# ─────────────────────────────────────────────────────────────────────────
# THE SOLVE ITSELF IS YOURS AND IS NOT SERVED HERE.
#
# Build the mesh, the function space, the weak form and the linear solve for
# ordinary finite-element work and OASiS has no business dictating it.
#
# What OASiS does document — because you cannot guess it and it is what the
# interface check compares against — is everything AROUND the solve: the
# imports/exports handshake above, the interface sign convention, and the flux
# recovery below. Those are this tool's own interface, not your method.
# ─────────────────────────────────────────────────────────────────────────

# Build geometry and mesh
geom = SplineGeometry()
geom.AddRectangle((X0, Y0), (X1, Y1), bcs=("bottom", "interface", "top", "outer"))
mesh = Mesh(geom.GenerateMesh(maxh=MAXH))

# Function space
fes = H1(mesh, order=ORDER, dirichlet="outer")
u = GridFunction(fes)
u.Set(0.0)

# Trial/test functions
v = fes.TestFunction()

# Bilinear form
a = BilinearForm(fes)
a += K * grad(u) * grad(v) * dx

# Linear form (volume load)
f = LinearForm(fes)
f += F_SRC(mesh.x, mesh.y) * v * dx

# Apply partner's flux as Neumann load on interface (NEUMANN side)
if SIDE == "neumann":
    y_if = np.array([vtx[1] for vtx in mesh.Vertices() if abs(vtx[0] - IFACE_X) < TOL])
    q_if = sample(imp, "normal_fluxes", Q_INIT, y_if)
    gfun = GridFunction(fes)
    gfun.vec[:] = 0.0
    iface_dofs = [int(NodeId(VERTEX, i)) for i, vtx in enumerate(mesh.Vertices())
                  if abs(vtx[0] - IFACE_X) < TOL]
    for d, q in zip(iface_dofs, q_if):
        gfun.vec[int(d)] = float(q)
    f += gfun * v * ds("interface")

# Assemble
a.Assemble()
f.Assemble()

# Solve
with TaskManager():
    u.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec

# Identify interface and outer DOFs
iface_dofs = [int(NodeId(VERTEX, i)) for i, vtx in enumerate(mesh.Vertices())
              if abs(vtx[0] - IFACE_X) < TOL]
outer_dofs = [int(NodeId(VERTEX, i)) for i, vtx in enumerate(mesh.Vertices())
              if abs(vtx[0] - OUTER_X) < TOL]
iface_v = [i for i, d in enumerate(iface_dofs)]
y_if = np.array([vtx[1] for vtx in mesh.Vertices() if abs(vtx[0] - IFACE_X) < TOL])

# ─────────────────────────────────────────────────────────────────────────
# THE SOLVE ITSELF IS YOURS AND IS NOT SERVED HERE.
#
# Build the mesh, the function space, the weak form and the linear solve for
# ordinary finite-element work and OASiS has no business dictating it.
#
# What OASiS does document — because you cannot guess it and it is what the
# interface check compares against — is everything AROUND the solve: the
# imports/exports handshake above, the interface sign convention, and the flux
# recovery below. Those are this tool's own interface, not your method.
# ─────────────────────────────────────────────────────────────────────────

if SIDE == "dirichlet":
    T_if = sample(imp, "values", T_INIT, y_if)
    for d, t in zip(iface_dofs, T_if):
        u.vec[int(d)] = float(t)
else:
    q_if = sample(imp, "normal_fluxes", Q_INIT, y_if)
    gfun = GridFunction(fes)               # P1 trace of the partner's samples
    gfun.vec[:] = 0.0
    for d, q in zip(iface_dofs, q_if):
        gfun.vec[int(d)] = float(q)
    f += gfun * v * ds("interface")        # APPLY the partner's number unchanged

with TaskManager():
    # Outward normal flux density q = -(k grad T).n on the interface.
    #
    # WHY NOT AN L2 PROJECTION OF THE STRESS. That is what this file used to do
    # on the Neumann side: project -(sigma(u_h) . n_own) over the whole
    # subdomain and sample it at the interface. The gradient of a P1 solution —
    # and therefore the stress — is only O(h) accurate ON the boundary; the
    # superconvergence points are interior, and the boundary trace is exactly
    # what the coupling reads.
    #
    # THE CONSISTENT (REACTION) TRACTION. From
    #     a(u,v) - (f,v) = int_dOmega (sigma(u).n).v ds = -int_Gamma q_out.v ds
    # (the second equality is this file's sign convention) it follows that for
    # each interface basis function phi_i:
    #     r_i = (A u_h)_i - (f_vol)_i = -int_Gamma q_out phi_i ds
    # so the consistent nodal flux is q_i = -r_i / w_i where w_i = int_Gamma phi_i ds.
    #
    rvec = f.vec.CreateVector()
    rvec.data = a.mat * u.vec - f.vec    # r = A u_h - b_vol, no bc
    fw = LinearForm(fes)
    fw += v * ds("interface")                  # w_i = int_Gamma phi_i ds
    fw.Assemble()

    r_if = np.array([rvec[int(d)] for d in iface_dofs], float)
    w_if = np.array([fw.vec[int(d)] for d in iface_dofs], float)
    Q = np.zeros(len(iface_dofs))
    ok = np.abs(w_if) > 1e-14
    Q[ok] = -r_if[ok] / w_if[ok]

    # An interface node that ALSO lies on the outer Dirichlet boundary
    # carries the OUTER reaction as well, so its residual is not this
    # side's consistent flux alone. Interpolate from its good neighbors.
    suspect = np.isin(iface_dofs, outer_dofs) | ~ok
    good = np.where(~suspect)[0]
    if len(good):
        for i in np.where(suspect)[0]:
            Q[i] = Q[good[np.argmin(np.abs(good - i))]]

# ── EXPORT SELF-CHECK ─ keep this block. It stops the three exports that look
#    fine and are worthless: a non-finite field; a Neumann side whose imported
#    load never entered the assembled system (it returns the no-load answer and
#    converges to the wrong thing); or a Dirichlet side that copies the partner
#    array instead of recovering from its own solve.
_chk_vals = np.asarray([u.vec[int(d)] for d in iface_dofs], float).ravel()
_chk_flux = np.asarray(Q, float).ravel()
if not (np.isfinite(_chk_vals).all() and np.isfinite(_chk_flux).all()):
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface values or fluxes; "
                     "the solve did not produce a usable field, so nothing was "
                     "exported")
_chk_imp = (json.loads(Path("imports.json").read_text() or "{}")
            if Path("imports.json").is_file() else {})
_chk_qin = (np.concatenate([np.asarray(_d.get("normal_fluxes") or [], float).ravel()
                            for _d in _chk_imp.values()])
            if _chk_imp else np.zeros(0))
if SIDE == "neumann" and _chk_qin.size and np.abs(_chk_qin).max() > 0 \
        and np.abs(_chk_flux).max() < 1e-9 * np.abs(_chk_qin).max():
    raise SystemExit("EXPORT SELF-CHECK: the recovered interface flux is ~0 "
                     "against a nonzero imported flux: the imported load never "
                     "entered the assembled system (the facet term / boundary "
                     "condition that integrates it is missing). Fix the "
                     "application; do not couple on")
# (Dirichlet role only: a Neumann side's consistent recovery of a CONSTANT
#  applied flux can legitimately reproduce it to the last bit.)
if SIDE == "dirichlet" and _chk_qin.shape == _chk_flux.shape and _chk_flux.size \
        and np.array_equal(_chk_flux, -_chk_qin):
    raise SystemExit("EXPORT SELF-CHECK: the exported flux is the partner's "
                     "array negated, bit for bit: a copy, not a recovery from "
                     "this side's own assembled system")

Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(iface_v)),
    "coordinates": [[float(IFACE_X), float(yy)] for yy in y_if],
    "values": [float(u.vec[int(d)]) for d in iface_dofs],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))

# ── WHAT YOUR SOLVE MUST LEAVE BEHIND ─────────────────────────
# The code above and below the elided block uses these names. Your
# block has to define every one of them, or the rest will not run:
#
#     a
#     d
#     f
#     f_vol
#     fes
#     gfu
#     i
#     iface_dofs
#     iface_v
#     outer_dofs
#     v
#     y_if
#
# That is the whole contract. Read the surviving lines to see the
# shape each one has to have -- they are already indexed, assembled
# or written out there. OASiS does not serve the solve itself, but
# it will not make you guess which variables the hole was filling.
