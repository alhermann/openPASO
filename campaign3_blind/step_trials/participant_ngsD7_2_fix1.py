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
#    As shipped this is the LEFT / Dirichlet side; the payload that served
SIDE      = "dirichlet"   # "dirichlet" | "neumann"
PARTNER   = "right"       # name of the partner participant in couple(...)
X0, X1    = 0.0, 0.6      # this subdomain
Y0, Y1    = 0.0, 1.0
IFACE_AXIS = "x"          # WHICH straight line the interface is: "x" -> the line x = IFACE_X
                          # (the subdomains sit side by side) | "y" -> the line y = IFACE_X
                          # (they are stacked). Everything below follows from it.
IFACE_X   = 0.6           # shared interface (X0/X1 for axis "x", Y0/Y1 for axis "y")
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
T_INIT    = 0.0           # iteration-1 fallback interface temperature
Q_INIT    = 0.0           # iteration-1 fallback interface flux
# ─────────────────────────────────────────────────────────────────────────

MAXH  = min((X1 - X0) / NX, (Y1 - Y0) / NY)   # netgen's scalar mesh size
ORDER = 1                                     # H1 order (nodal == vertex dofs)

AX = 0 if IFACE_AXIS == "x" else 1         # the coordinate the interface FIXES
AL = 1 - AX                                # the coordinate that RUNS ALONG it
LO, HI = (X0, X1) if AX == 0 else (Y0, Y1)         # this subdomain, across the interface
ALO, AHI = (Y0, Y1) if AX == 0 else (X0, X1)       # this subdomain, along it
ON_RIGHT = abs(IFACE_X - HI) < abs(IFACE_X - LO)   # interface at this side's MAX of that axis?
OUTER_X = LO if ON_RIGHT else HI           # the opposite face, on the same axis
S = 1.0 if ON_RIGHT else -1.0              # outward normal at interface = S * e_AX
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
    ys = np.array([c[AL] for c in imp["coordinates"]], float)   # the coordinate ALONG the interface
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
# ... (explanation continues in the annotated block)
# ordinary finite-element work and OASiS has no business dictating it.
#
# What OASiS does document — because you cannot guess it and it is what the
# interface check compares against — is everything AROUND the solve: the
# imports/exports handshake above, the interface sign convention, and the flux
# ... (explanation continues in the annotated block)

# Build the mesh
geo = SplineGeometry()
geo.AddRectangle((X0, Y0), (X1, Y1), bcs=('bottom', 'right', 'top', 'left'))
mesh = Mesh(geo.GenerateMesh(maxh=MAXH))

# Function space
fes = H1(mesh, order=ORDER)
u, v = fes.TnT()

# Identify interface and outer boundary dofs
pts = np.array([v.point for v in mesh.vertices])
vdof = np.array([fes.GetDofNrs(NodeId(VERTEX, v.nr))[0] for v in mesh.vertices])
iface = np.where(np.abs(pts[:, AX] - IFACE_X) < TOL)[0]
iface_v = iface
iface_dofs = vdof[iface]
y_if = pts[iface, AL]

# Outer boundary dofs (corners at interface meet outer BC, so exclude them)
outer_mask = np.zeros(len(pts), dtype=bool)
if AX == 0:  # interface at x = IFACE_X, outer is x = 0
    outer_mask |= np.abs(pts[:, 0] - X0) < TOL
    outer_mask |= np.abs(pts[:, 1] - Y0) < TOL
    outer_mask |= np.abs(pts[:, 1] - Y1) < TOL
else:  # interface at y = IFACE_X, outer is y = 0 and y = 1
    outer_mask |= np.abs(pts[:, 1] - Y0) < TOL
    outer_mask |= np.abs(pts[:, 1] - Y1) < TOL
    outer_mask |= np.abs(pts[:, 0] - X0) < TOL
interface_mask = np.abs(pts[:, AX] - IFACE_X) < TOL  # exclude interface corners
outer_mask &= ~interface_mask
outer_dofs = vdof[outer_mask]

# Bilinear form: a(u,v) = int(k grad u grad v) dx
a = BilinearForm(fes)
a += K * grad(u) * grad(v) * dx
a.Assemble()

# Linear form: f(v) = int(source * v) dx
# Sample source at vertices and interpolate
source_gf = GridFunction(fes)
vx = np.array([v.point[0] for v in mesh.vertices])
vy = np.array([v.point[1] for v in mesh.vertices])
source_gf.vec.FV().NumPy()[vdof] = F_SRC(vx, vy)

f_vol = LinearForm(fes)
f_vol += source_gf * v * dx
f_vol.Assemble()

# Solution grid function
gfu = GridFunction(fes)
gfu.Set(0)

# Apply the partner's interface values (Dirichlet side reads values from partner)
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
    f = LinearForm(fes)
    f += gfun * v * ds("right")        # APPLY the partner's number unchanged
    f.Assemble()

with TaskManager():
    # Solve the system
    # Free dofs are those not on Dirichlet boundaries (exclude interface for Dirichlet side)
    all_dofs = fes.GetDofNrs()
    if SIDE == "dirichlet":
        fixed_dofs = np.concatenate([outer_dofs, iface_dofs])
    else:
        fixed_dofs = outer_dofs
    free_dofs = fes.FreeDofs()
    
    # Set outer Dirichlet values
    for d in outer_dofs:
        gfu.vec[int(d)] = T_OUTER
    
    # Build the system matrix and RHS
    A = a.mat
    b = f_vol.vec.CreateVector()
    b.data = f_vol.vec
    
    # Solve for free dofs
    rhs_vec = f_vol.vec.CreateVector()
    rhs_vec.data = f_vol.vec
    gfu.vec.data[:] = 0.0
    gfu.vec.data[free_dofs] = A.Inverse(free_dofs, inverse='sparsecholesky') * rhs_vec
    
    # Apply Dirichlet values at interface (for Dirichlet side)
    if SIDE == "dirichlet":
        for d, t in zip(iface_dofs, T_if):
            gfu.vec[int(d)] = float(t)
    
    # Outward normal flux density q = -(k grad T).n on the interface.
    #
    # ... (explanation continues in the annotated block)
    rvec = f_vol.vec.CreateVector()
    rvec.data = a.mat * gfu.vec - f_vol.vec    # r = A u_h - b_vol, no bc
    fw = LinearForm(fes)
    fw += v * ds("right")                  # w_i = int_Gamma phi_i ds
    fw.Assemble()

    r_if = np.array([rvec[int(d)] for d in iface_dofs], float)
    w_if = np.array([fw.vec[int(d)] for d in iface_dofs], float)
    Q = np.zeros(len(iface_dofs))
    ok = np.abs(w_if) > 1e-14
    Q[ok] = -r_if[ok] / w_if[ok]

    # An interface node that ALSO lies on the outer Dirichlet boundary
    # carries the OUTER reaction as well, so its residual is not this
    # ... (explanation continues in the annotated block)
    suspect = np.isin(iface_dofs, outer_dofs) | ~ok
    good = np.where(~suspect)[0]
    if len(good):
        for i in np.where(suspect)[0]:
            Q[i] = Q[good[np.argmin(np.abs(good - i))]]
# ── EXPORT SELF-CHECK ─ keep this block. It stops the three exports that look
#    fine and are worthless: a non-finite field; a Neumann side whose imported
#    load never entered the assembled system (it returns the no-load answer and
# ... (explanation continues in the annotated block)
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
    "coordinates": [([float(IFACE_X), float(yy)] if AX == 0 else [float(yy), float(IFACE_X)])
                    for yy in y_if],
    "values": [float(gfu.vec[int(d)]) for d in iface_dofs],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))

print("Side A participant completed successfully")
