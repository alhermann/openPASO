"""DUNE-fem participant for subdomain B (Neumann side)."""
import json
from pathlib import Path
import numpy as np
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.fem.operator import galerkin as operator_galerkin
from dune.ufl import DirichletBC, Constant
from ufl import TrialFunction, TestFunction, SpatialCoordinate, conditional, dot, ds, dx, grad, lt

PARTNER = "A"
X0, X1 = 0.625, 1.5
Y0, Y1 = 0.0, 1.0
Z0, Z1 = 0.0, 1.0
K = 4.0
Q_INIT = 0.0


def source_term(x, y, z):
    return (-15*x**3*y**3/32 + 35*x**3*y**2/32 - 45*x**3*y*z**2/32 + 45*x**3*y*z/32 
            - 5*x**3*y/8 + 35*x**3*z**2/32 - 35*x**3*z/32 + 875*x**2*y**3/384 
            + 635*x**2*y**2/128 + 875*x**2*y*z**2/128 - 875*x**2*y*z/128 
            - 695*x**2*y/96 + 635*x**2*z**2/128 - 635*x**2*z/128 
            - 45*x*y**3*z**2/32 + 45*x*y**3*z/32 + 4585*x*y**3/2048 
            + 105*x*y**2*z**2/32 - 105*x*y**2*z/32 - 2805*x*y**2/2048 
            + 9915*x*y*z**2/2048 - 9915*x*y*z/2048 - 445*x*y/512 
            - 2805*x*z**2/2048 + 2805*x*z/2048 + 875*y**3*z**2/384 
            - 875*y**3*z/384 - 28275*y**3/4096 + 635*y**2*z**2/128 
            - 635*y**2*z/128 - 52425*y**2/4096 - 343435*y*z**2/12288 
            + 343435*y*z/12288 + 20175*y/1024 - 52425*z**2/4096 + 52425*z/4096)


def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text() or "{}").get(PARTNER)
    except:
        return None


def resample(imp, key, fallback, pts):
    n = len(pts)
    if imp is None or "coordinates" not in imp:
        return np.full(n, float(fallback))
    coords = np.asarray(imp["coordinates"], float)
    values = np.asarray(imp.get(key, []), float).ravel()
    if len(values) != len(coords):
        return np.full(n, float(fallback))
    src_yz = coords[:, 1:3]
    try:
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        interp = LinearNDInterpolator(src_yz, values)
        result = interp(pts)
        nan_mask = ~np.isfinite(result)
        if np.any(nan_mask):
            nn = NearestNDInterpolator(src_yz, values)
            result[nan_mask] = nn(pts[nan_mask])
        return result
    except:
        dist = ((pts[:, None, :] - src_yz[None, :, :]) ** 2).sum(-1)
        return values[np.argmin(dist, axis=1)]


config = json.loads(Path("level_config.json").read_text()) if Path("level_config.json").exists() else {}
NX_GLOBAL = config.get("nx_global", 12)
level = config.get("level", 1)

nxB = int(round(NX_GLOBAL * 7 / 12))
nyB = int(round(NX_GLOBAL * 1 / 12))
nzB = int(round(NX_GLOBAL * 1 / 12))
nxB, nyB, nzB = max(nxB, 2), max(nyB, 2), max(nzB, 2)

print(f"[B] Mesh: {nxB}x{nyB}x{nzB}, level={level}")

gridView = structuredGrid([X0, Y0, Z0], [X1, Y1, Z1], [nxB, nyB, nzB])
space = lagrange(gridView, order=1)

x = SpatialCoordinate(space)
u = TrialFunction(space)
v = TestFunction(space)

a_form = K * dot(grad(u), grad(v)) * dx

ffun = space.interpolate(0, name="source")
xd = space.interpolate(x[0]).as_numpy
yd = space.interpolate(x[1]).as_numpy
zd = space.interpolate(x[2]).as_numpy
ffun.as_numpy[:] = source_term(xd, yd, zd)

b_vol = ffun * v * dx

gflx = space.interpolate(0, name="iface_flux")

tol = 1e-8
iface_mask = np.abs(xd - X0) < tol
iface_dofs = np.where(iface_mask)[0]

corner_mask = np.zeros(len(iface_dofs), dtype=bool)
for idx, dof in enumerate(iface_dofs):
    y, z = yd[dof], zd[dof]
    if (np.abs(y - Y0) < tol or np.abs(y - Y1) < tol or 
        np.abs(z - Z0) < tol or np.abs(z - Z1) < tol):
        corner_mask[idx] = True

interior_iface_dofs = iface_dofs[~corner_mask]
pts_yz = np.array([[yd[d], zd[d]] for d in interior_iface_dofs])

print(f"[B] Found {len(iface_dofs)} interface dofs, {len(interior_iface_dofs)} interior")

imp = read_imports()
q_in = resample(imp, "normal_fluxes", Q_INIT, pts_yz)
gflx.as_numpy[:] = 0.0
gflx.as_numpy[interior_iface_dofs] = q_in

eps = 1e-8
b_form = b_vol + conditional(lt(abs(x[0] - X0), eps), gflx * v, 0.0) * ds

bc_where = (lt(abs(x[0] - X1), eps) | 
            lt(abs(x[1] - Y0), eps) | 
            lt(abs(x[1] - Y1), eps) | 
            lt(abs(x[2] - Z0), eps) | 
            lt(abs(x[2] - Z1), eps))

dbc = DirichletBC(space, 0.0, bc_where)

scheme = galerkin([a_form == b_form, dbc], solver="cg")

uh = space.interpolate(0, name="temperature")
scheme.solve(target=uh)

T_vals = uh.as_numpy[iface_dofs]

op_free = operator_galerkin([a_form == b_vol])
rfun = space.interpolate(0, name="residual")
op_free(uh, rfun)
r = np.array(rfun.as_numpy)

from dune.fem import assemble
wfun = assemble(conditional(lt(abs(x[0] - X0), eps), v, 0.0) * ds)
w = np.array(wfun.as_numpy)[iface_dofs]

Q = np.zeros(len(iface_dofs))
ok = np.abs(w) > 1e-14
Q[ok] = -r[iface_dofs][ok] / w[ok]

suspect = corner_mask
good = np.where(~suspect)[0]
if len(good) > 0:
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]

iface_coords = [[X0, float(yd[d]), float(zd[d])] for d in iface_dofs]
sorted_pairs = sorted(zip(iface_coords, range(len(iface_dofs))), key=lambda p: (p[0][1], p[0][2]))
iface_coords = [p[0] for p in sorted_pairs]
reorder = [p[1] for p in sorted_pairs]
T_sorted = T_vals[reorder]
Q_sorted = Q[reorder]

exports = {
    "field_name": "temperature",
    "n_points": len(iface_dofs),
    "coordinates": iface_coords,
    "values": [float(t) for t in T_sorted],
    "normal_fluxes": [float(q) for q in Q_sorted]
}
Path("exports.json").write_text(json.dumps(exports, indent=2))

ndof = len(uh.as_numpy)
Path(f"run_level{level}_B.log").write_text(f"NDOF = {ndof}\n")

print(f"[DUNE B] {ndof} DOFs, {len(iface_dofs)} iface nodes, T=[{T_sorted.min():.4f},{T_sorted.max():.4f}]")
