"""scikit-fem participant for subdomain B (Dirichlet side) - coupled elasticity."""
import json
from pathlib import Path
import numpy as np
from skfem import (Basis, BilinearForm, ElementTriP1, ElementVector,
                   FacetBasis, LinearForm, MeshTri, asm, condense, solve)
from skfem.helpers import ddot, sym_grad, trace

# Problem parameters
SIDE = "dirichlet"  # B is Dirichlet side
PARTNER = "A"
X0, X1 = 0.0, 1.0
Y0, Y1 = 0.625, 1.5
IFACE_Y = 0.625

# Material: lambda=480, mu=240
LAMBDA = 480.0
MU = 240.0

LEVEL = 1
if LEVEL == 1:
    NX, NY = 8, 7
elif LEVEL == 2:
    NX, NY = 16, 14
else:
    NX, NY = 32, 28

UI_X, UI_Y = 0.0, 0.0
TI_X, TI_Y = 0.0, 0.0
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

def sample(imp, key, fallback, x):
    fb = np.asarray(fallback, float).ravel()
    if not imp or not imp.get("coordinates"):
        return np.tile(fb, (len(x), 1))
    xs = np.array([c[0] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key) or [], float)
    if vs.ndim == 1:
        vs = vs.reshape(-1, 1)
    if vs.shape[0] != xs.size or vs.shape[1] != fb.size:
        return np.tile(fb, (len(x), 1))
    o = np.argsort(xs)
    return np.column_stack([np.interp(x, xs[o], vs[o, c])
                            for c in range(vs.shape[1])])

imp = read_imports()

# Build mesh
x = np.linspace(X0, X1, NX + 1)
y = np.linspace(Y0, Y1, NY + 1)
m = MeshTri.init_tensor(x, y)

# Vector element
e = ElementVector(ElementTriP1())
basis = Basis(m, e)
nd = basis.nodal_dofs  # shape (2, n_nodes)

# Elasticity bilinear form using built-in model
from skfem.models.elasticity import linear_elasticity
A = linear_elasticity(LAMBDA, MU).assemble(basis)

# Volume load - zero for now
@LinearForm
def body_force(v, w):
    return 0.0 * v[0] + 0.0 * v[1]

b_vol = asm(body_force, basis)

# Find interface nodes (at y = IFACE_Y)
all_iface_n = np.where(np.abs(m.p[1] - IFACE_Y) < TOL)[0]
all_iface_n.sort()

# Interface x-coordinates
x_if = m.p[0, all_iface_n]

# Outer boundary nodes (top|left|right, bottom is interface)
outer_n = np.where(
    (np.abs(m.p[1] - Y1) < TOL) | 
    (np.abs(m.p[0] - X0) < TOL) | 
    (np.abs(m.p[0] - X1) < TOL)
)[0]

# Facet basis for interface
fbasis = FacetBasis(m, ElementTriP1())
fbi = FacetBasis(m, e)

# Apply boundary conditions
sol = basis.zeros()

if SIDE == "dirichlet":
    # Import displacement
    u_if = sample(imp, "values", (UI_X, UI_Y), x_if)
    
    # Apply to interface nodes (excluding corners)
    for i, n in enumerate(all_iface_n):
        if np.abs(m.p[0, n] - X0) < TOL or np.abs(m.p[0, n] - X1) < TOL:
            continue  # corners are on outer boundary
        sol[nd[0, n]] = u_if[i, 0]
        sol[nd[1, n]] = u_if[i, 1]
    
    # Dirichlet DOFs: outer boundary + interface
    iface_dofs = np.unique([nd[0, n] for n in all_iface_n] + [nd[1, n] for n in all_iface_n])
    outer_dofs = np.unique([nd[0, n] for n in outer_n] + [nd[1, n] for n in outer_n])
    D = np.unique(np.concatenate([outer_dofs, iface_dofs]))
else:
    # Neumann side: import traction
    t_if = sample(imp, "normal_fluxes", (TI_X, TI_Y), x_if)
    
    @LinearForm
    def traction(v, w):
        return w['t'][0] * v[0] + w['t'][1] * v[1]
    
    gnod = basis.zeros()
    for i, n in enumerate(all_iface_n):
        gnod[nd[0, n]] = t_if[i, 0]
        gnod[nd[1, n]] = t_if[i, 1]
    
    b = b_vol + asm(traction, fbi, t=fbi.interpolate(gnod))
    
    # Dirichlet DOFs: outer boundary only
    D = np.unique([nd[0, n] for n in outer_n] + [nd[1, n] for n in outer_n])

# Solve
if SIDE == "dirichlet":
    sol = solve(*condense(A, b_vol, D=D))
else:
    sol = solve(*condense(A, b, D=D))

# Export displacement at interface
u_export = []
for n in all_iface_n:
    ux = float(sol[nd[0, n]])
    uy = float(sol[nd[1, n]])
    u_export.append([ux, uy])

# Compute traction export q_out = -(sigma . n_own)
r = A @ sol - b_vol

# Weight form for nodal integration
@LinearForm
def unit_load(v, w):
    return 1.0 * v[0] + 1.0 * v[1]

wgt = asm(unit_load, fbi)

idx = np.column_stack([nd[0, all_iface_n], nd[1, all_iface_n]])
wi = wgt[idx]

Q = np.zeros_like(wi)
ok = np.abs(wi) > 1e-14
Q[ok] = -r[idx][ok] / wi[ok]

# Handle corners
suspect = np.zeros(len(all_iface_n), bool)
for i, n in enumerate(all_iface_n):
    if np.abs(m.p[0, n] - X0) < TOL or np.abs(m.p[0, n] - X1) < TOL:
        suspect[i] = True
    elif not ok[i].all():
        suspect[i] = True

good = np.where(~suspect)[0]
if len(good) > 0:
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]

exports = {
    "field_name": "displacement",
    "n_points": len(all_iface_n),
    "coordinates": [[float(m.p[0, n]), float(m.p[1, n])] for n in all_iface_n],
    "values": u_export,
    "normal_fluxes": [[float(Q[i, 0]), float(Q[i, 1])] for i in range(len(all_iface_n))]
}
Path("exports.json").write_text(json.dumps(exports, indent=2))

ndof = A.shape[0]
with open(f"run_level{LEVEL}_B.log", "w") as f:
    f.write(f"NDOF = {ndof}\n")

print(f"Subdomain B (Dirichlet): NDOF={ndof}, interface points={len(all_iface_n)}")
