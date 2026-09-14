"""NGSolve participant for the OASiS `couple` driver.

Steady heat conduction  -div(k grad T) = f  on one rectangular subdomain.
CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.
"""
import json
from pathlib import Path

import ngsolve                # the MODULE, so ngsolve.ngsglobals.msg_level = 3 resolves:
                              # a run log that must carry this code's own output needs it,
                              # and `from ngsolve import ...` alone leaves `ngsolve` undefined
import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import (VERTEX, BilinearForm, GridFunction, H1, LinearForm, Mesh,
                     NodeId, TaskManager, ds, dx, grad)

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
SIDE      = "dirichlet"   # "dirichlet" | "neumann"
PARTNER   = "right"       # name of the partner participant in couple(...)
X0, X1    = 0.0, 0.6      # this subdomain
Y0, Y1    = 0.0, 1.0
IFACE_X   = 0.6           # shared interface (must be X0 or X1)
K         = 1.0           # conductivity


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

    HOW THIS ENTERS NGSolve (your solve below has to do it): NGSolve's symbolic
    `x`, `y` are CoefficientFunctions and carry NO NumPy ufuncs, so do NOT call
    this function on them (np.sin(x) raises; np.zeros_like(x) silently returns
    a 0-d object array and the source collapses to a constant), and do NOT wrap
    the function -- CoefficientFunction(F_SRC) is a TypeError ("incompatible
    constructor arguments", measured). Sample it at the mesh vertices instead,
    np.array([v.point for v in mesh.vertices]), into a P1 GridFunction on your
    space (gf.vec.FV().NumPy()[vertex_dofs] = F_SRC(vx, vy)); a GridFunction IS
    a CoefficientFunction and integrates as gf * v * dx -- the P1 interpolant
    of the source, quadrature error O(h^2), the order of the discretisation.
    """
    return np.pi**2 * x * np.sin(np.pi * y)
T_OUTER   = 0.0           # Dirichlet value on the NON-interface x-boundary
NX, NY    = 6, 10         # this subdomain's own mesh (netgen maxh derived below)
T_INIT    = 0.0            # iteration-1 fallback interface temperature
Q_INIT    = 0.0           # iteration-1 fallback interface flux
# ─────────────────────────────────────────────────────────────────────────

MAXH  = min((X1 - X0) / NX, (Y1 - Y0) / NY)   # netgen's scalar mesh size
ORDER = 1                                     # H1 order (nodal == vertex dofs)

ON_RIGHT = abs(IFACE_X - X1) < abs(IFACE_X - X0)   # interface is this side's x-max?
OUTER_X = X0 if ON_RIGHT else X1
S = 1.0 if ON_RIGHT else -1.0              # outward normal at interface = S * e_x
TOL = 1e-9 * max(X1 - X0, Y1 - Y0)

ngsolve.ngsglobals.msg_level = 3


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
# recovery.

# Build geometry and mesh
geo = SplineGeometry()
geo.AddRectangle((X0, Y0), (X1, Y1), bcs=('bottom', 'interface', 'top', 'outer'))
mesh = Mesh(geo.GenerateMesh(maxh=MAXH))

# Function space
fes = H1(mesh, order=ORDER, dirichlet='outer')

# Trial/test functions
u, v = fes.TnT()

# Bilinear form: k*grad(u)*grad(v)
a = BilinearForm(fes)
a += K * grad(u) * grad(v) * dx

# Linear form: source term
f = LinearForm(fes)

# Sample source at vertices and interpolate into a GridFunction
vertices = list(mesh.vertices)
vx = np.array([v.point[0] for v in vertices])
vy = np.array([v.point[1] for v in vertices])
src_vals = F_SRC(vx, vy)

# Get vertex DOFs
vertex_dofs = [fes.GetDofNrs(NodeId(VERTEX, i))[0] for i in range(len(vertices))]

# Create source GridFunction
gf_src = GridFunction(fes)
gf_src.vec[:] = 0.0
for dof_idx, val in zip(vertex_dofs, src_vals):
    gf_src.vec[dof_idx] = val

# Add source to linear form
f += gf_src * v * dx

# Assemble forms
a.Assemble()
f.Assemble()

# Solution vector
gfu = GridFunction(fes)

# Identify interface nodes and their DOFs
iface_elems = [el for el in mesh.Elements() if 'interface' in el.boundaries]
iface_nodes = set()
for el in iface_elems:
    for nv in el.vertices:
        iface_nodes.add(nv.index)

iface_v = sorted(list(iface_nodes))
iface_dofs = [fes.GetDofNrs(NodeId(VERTEX, iv))[0] for iv in iface_v]

# Identify outer boundary nodes (corners where interface meets outer)
outer_elems = [el for el in mesh.Elements() if 'outer' in el.boundaries]
outer_nodes = set()
for el in outer_elems:
    for nv in el.vertices:
        outer_nodes.add(nv.index)

outer_dofs = [fes.GetDofNrs(NodeId(VERTEX, ov))[0] for ov in sorted(list(outer_nodes))]

# Extract y-coordinates of interface nodes
y_if = np.array([mesh.vertices[iv].point[1] for iv in iface_v])

if SIDE == "dirichlet":
    T_if = sample(imp, "values", T_INIT, y_if)
    for d, t in zip(iface_dofs, T_if):
        gfu.vec[int(d)] = float(t)
else:
    q_if = sample(imp, "normal_fluxes", Q_INIT, y_if)
    gfun = GridFunction(fes)               # P1 trace of the partner's samples
    gfun.vec[:] = 0.0
    for d, q in zip(iface_dofs, q_if):
        gfun.vec[int(d)] = float(q)
    f += gfun * v * ds("interface")        # APPLY the partner's number unchanged

with TaskManager():
    # Solve the system
    f_vol = LinearForm(fes)
    f_vol += gf_src * v * dx
    f_vol.Assemble()

    gfu.vec[:] = 0.0
    
    # Apply Dirichlet BCs (outer boundary already handled by space definition)
    # Set interface values for Dirichlet side
    if SIDE == "dirichlet":
        for d, t in zip(iface_dofs, T_if):
            gfu.vec[int(d)] = float(t)
    
    # Free DOFs are those not on outer boundary
    free_dofs = fes.FreeDofs()
    
    # Solve
    a.mat.Inverse(free_dofs, inverse='sparsecholesky')
    rhs = f.vec.CreateVector()
    rhs.data = f.vec
    
    # Zero out RHS entries for constrained DOFs
    for d in iface_dofs:
        rhs[int(d)] = 0.0
    
    # Compute solution
    gfu.vec.data = a.mat.Inverse(free_dofs, inverse='sparsecholesky') * rhs

    # Outward normal flux density q = -(k grad T).n on the interface.
    rvec = f.vec.CreateVector()
    rvec.data = a.mat * gfu.vec - f_vol.vec    # r = A u_h - b_vol, no bc
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
    suspect = np.isin(iface_dofs, outer_dofs) | ~ok
    good = np.where(~suspect)[0]
    if len(good):
        for i in np.where(suspect)[0]:
            Q[i] = Q[good[np.argmin(np.abs(good - i))]]

# ── EXPORT SELF-CHECK ─ keep this block. It stops the three exports that look
#    fine and are worthless: a non-finite field; a Neumann side whose imported
#    load never entered the assembled system (it returns the no-load answer and
_chk_vals = np.asarray([gfu.vec[int(d)] for d in iface_dofs], float).ravel()
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
    "values": [float(gfu.vec[int(d)]) for d in iface_dofs],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))

print("Side A completed successfully")
print(f"Interface points: {len(iface_v)}")
print(f"Max interface value: {max(abs(v) for v in _chk_vals):.6e}")
print(f"Max interface flux: {max(abs(q) for q in _chk_flux):.6e}")
