"""4C path-walk participant: scalar transport (conduction), either role, 2-D.

4C is a compiled YAML-in / VTU-out code with no Python API, so a participant is a
WRAPPER: write the deck from imports.json, run the binary, read the VTU back,
write exports.json.

TWO THINGS THE SHIPPED WRAPPER CANNOT DO, both needed here.

1. A SPATIALLY VARYING SOURCE. ``participant_fourc.py`` takes a constant
   ``F_SRC``. A manufactured problem has a source that varies over the domain,
   and it enters here as a SYMBOLIC_FUNCTION_OF_SPACE_TIME on a volume Neumann
   condition, which 4C evaluates at the quadrature points -- so the source is
   integrated exactly rather than lumped onto nodes.

2. AN EXACT NODAL INTERFACE DATUM. The shipped wrapper fits the imported
   interface samples with a degree-3 least-squares polynomial, because a 4C
   boundary condition takes a symbolic expression and not a table. On the
   DIRICHLET side that fit error goes straight into the interface value and
   therefore into the coupled answer. Here the Dirichlet side instead gives each
   interface node its OWN design point and its own condition, which is exact and
   needs no fit at all. The Neumann side keeps a polynomial, at a degree high
   enough to represent the trace (which is a low-degree polynomial in the
   manufactured family), and the wrapper PRINTS the fit residual so the
   approximation is visible rather than assumed.

FLUX EXPORT. 4C's `flux_domain` is the L2 projection of -D grad(phi), which is
only O(h) accurate ON the boundary -- the superconvergence points are interior.
There is no access to 4C's own reaction, so the Dirichlet side exports that
projection and the walk MEASURES what order it delivers rather than assuming one.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wcommon as W                                              # noqa: E402

cfg = W.load_cfg()
dim, axis, xi = cfg["dim"], cfg["axis"], cfg["xi"]
ext, n = cfg["extent"], cfg["n"]
if dim != 2 or cfg["physics"] != "scalar":
    sys.exit("the 4C walk participant serves 2-D scalar transport only")
K = np.asarray(cfg["K"], float)
if not np.allclose(K, K[0, 0] * np.eye(2)):
    sys.exit("MAT_scatra takes a SCALAR diffusivity; this cell carries a tensor")
kval = float(K[0, 0])
free_axis = 1 - axis
S = W.outward_sign(ext, axis, xi)
FIT_DEG = int(cfg.get("fit_degree", 8))
BIN = cfg.get("fourc_bin", "/home/alexander/4C/build/4C")
LD = cfg.get("fourc_ld", "/opt/4C-dependencies/lib")

# ── mesh: QUAD4 tensor grid ───────────────────────────────────────────
ax = [np.linspace(lo, hi, k + 1) for (lo, hi), k in zip(ext, n)]
nid = np.zeros((len(ax[0]), len(ax[1])), int)
nodes, P = [], []
c = 1
for j in range(len(ax[1])):
    for i in range(len(ax[0])):
        nodes.append(f"NODE {c} COORD {ax[0][i]:.15f} {ax[1][j]:.15f} 0.0")
        nid[i, j] = c
        P.append((ax[0][i], ax[1][j]))
        c += 1
P = np.array(P, float)
elems = []
for j in range(n[1]):
    for i in range(n[0]):
        elems.append(f"{len(elems) + 1} TRANSP QUAD4 {nid[i, j]} "
                     f"{nid[i + 1, j]} {nid[i + 1, j + 1]} {nid[i, j + 1]} "
                     f"MAT 1 TYPE Std")

ifm = np.abs(P[:, axis] - xi) < 1e-9
outer = np.zeros(len(P), bool)
for a in range(2):
    for val in ext[a]:
        if a == axis and abs(val - xi) < 1e-9:
            continue
        outer |= np.abs(P[:, a] - val) < 1e-9
inode = np.where(ifm)[0]
inode = inode[np.argsort(P[inode, free_axis])]
ipts = P[inode]

imp = W.read_imports(cfg["partner"])
src = cfg["source"].replace("**", "^")

blocks, funcs = [], [f'FUNCT1:\n  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{src}"\n']
point_dirich, dnode_top = [], []
fit_note = ""

if cfg["side"] == "dirichlet":
    g = W.sample(imp, "values", ipts, 0.0, 1, [free_axis]).ravel()
    # ONE DESIGN POINT PER INTERFACE NODE: exact nodal data, no polynomial fit.
    # The two ENDS are left out: they lie on an outer face as well as on the
    # interface and carry the outer datum on BOTH sides of the split.
    d = 1
    for i, val in zip(inode, g):
        if outer[i]:
            continue
        point_dirich.append(f"  - E: {d}\n    NUMDOF: 1\n    ONOFF: [1]\n"
                            f"    VAL: [{val:.17g}]\n    FUNCT: [0]\n")
        dnode_top.append(f'  - "NODE {i + 1} DNODE {d}"\n')
        d += 1
else:
    q = W.sample(imp, "normal_fluxes", ipts, 0.0, 1, [free_axis]).ravel()
    s = ipts[:, free_axis]
    if float(np.ptp(q)) < 1e-300:
        expr, resid = f"{float(q[0]):.17g}", 0.0
    else:
        deg = int(min(FIT_DEG, len(s) - 1))
        co = np.polyfit(s, q, deg)
        resid = float(np.max(np.abs(np.polyval(co, s) - q))
                      / max(np.max(np.abs(q)), 1e-300))
        v = ["x", "y"][free_axis]
        expr = " + ".join(f"({a:.17g})*{v}^{deg - i}" if deg - i else
                          f"({a:.17g})" for i, a in enumerate(co))
    fit_note = f"neumann_profile_fit_rel_residual = {resid:.3e}"
    funcs.append(f'FUNCT2:\n  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{expr}"\n')
    # APPLY THE PARTNER'S NUMBER UNCHANGED: the two participants export with
    # respect to their own outward normals, so this is the natural term here.
    blocks.append("DESIGN LINE NEUMANN CONDITIONS:\n  - E: 2\n    NUMDOF: 1\n"
                  "    ONOFF: [1]\n    VAL: [1.0]\n    FUNCT: [2]\n")

deck = f"""TITLE:
  - "openPASO balanced-set path walk (4C scalar transport)"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "Stationary"
  SOLVERTYPE: "linear_full"
  VELOCITYFIELD: "zero"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
  CALCFLUX_DOMAIN: "diffusive"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  OUTPUT_DATA_FORMAT: "ascii"
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: {kval}
"""
deck += "".join(funcs)
deck += ("DESIGN LINE DIRICH CONDITIONS:\n  - E: 1\n    NUMDOF: 1\n"
         "    ONOFF: [1]\n    VAL: [0.0]\n    FUNCT: [0]\n")
if point_dirich:
    deck += "DESIGN POINT DIRICH CONDITIONS:\n" + "".join(point_dirich)
deck += "".join(blocks)
# The volume source. In 2-D the "SURF" IS the domain, so a surface Neumann
# condition on a scalar-transport problem is the volumetric source term, and
# FUNCT makes it vary in space.
deck += ("DESIGN SURF NEUMANN CONDITIONS:\n  - E: 1\n    NUMDOF: 1\n"
         "    ONOFF: [1]\n    VAL: [1.0]\n    FUNCT: [1]\n")
deck += "NODE COORDS:\n" + "".join(f'  - "{s}"\n' for s in nodes)
deck += "TRANSPORT ELEMENTS:\n" + "".join(f'  - "{s}"\n' for s in elems)
deck += "DLINE-NODE TOPOLOGY:\n"
deck += "".join(f'  - "NODE {i + 1} DLINE 1"\n' for i in np.where(outer)[0])
deck += "".join(f'  - "NODE {i + 1} DLINE 2"\n' for i in inode)
if dnode_top:
    deck += "DNODE-NODE TOPOLOGY:\n" + "".join(dnode_top)
deck += "DSURF-NODE TOPOLOGY:\n" + "".join(
    f'  - "NODE {i + 1} DSURFACE 1"\n' for i in range(len(P)))
Path("input.4C.yaml").write_text(deck)

env = dict(os.environ)
env["LD_LIBRARY_PATH"] = LD + os.pathsep + env.get("LD_LIBRARY_PATH", "")
# stdbuf is not optional with 4C: without line buffering an MPI_Abort eats the
# error message and the failure arrives as an empty log.
r = subprocess.run(["stdbuf", "-oL", "-eL", BIN, "input.4C.yaml", "out"],
                   capture_output=True, text=True, env=env, timeout=3600)
Path("fourc_stdout.log").write_text(r.stdout[-200000:] + "\n" + r.stderr[-20000:])

vtus = sorted(Path("out-vtk-files").glob("scatra-*-0.vtu"))
if not vtus:
    sys.exit(f"4C produced no VTU (rc={r.returncode}). "
             f"stdout tail:\n{r.stdout[-3000:]}")


def _step(p):
    # scatra-00001-0.vtu -> 1. The TRAILING number is the MPI RANK, not the
    # step: matching the last number returns the initial condition, silently.
    m = re.match(r"scatra-(\d+)-\d+\.vtu$", p.name)
    return int(m.group(1)) if m else -1


import meshio                                                    # noqa: E402
m = meshio.read(str(max(vtus, key=_step)))
pts = np.asarray(m.points)[:, :2]
phi = np.asarray(m.point_data["phi_1"]).ravel()
flux = m.point_data.get("flux_domain_phi_1")
if flux is None:
    sys.exit('no flux field in the 4C VTU - set CALCFLUX_DOMAIN: "diffusive"')
flux = np.asarray(flux)

# A 4C VTU repeats every node once per element (QUAD4 -> up to 4 copies).
# Collapse by coordinate or every array is several times too long.
key = {}
for a, p in enumerate(pts):
    key.setdefault(tuple(np.round(p, 9)), []).append(a)
T = np.zeros(len(P))
QD = np.zeros(len(P))
for i, p in enumerate(P):
    rows = key.get(tuple(np.round(p, 9)))
    if not rows:
        sys.exit(f"4C VTU has no node at {p}")
    T[i] = float(np.mean(phi[rows]))
    QD[i] = float(np.mean(flux[rows][:, axis]))

# ── the interface flux ────────────────────────────────────────────────
#
# WHY NOT 4C's OWN `flux_domain`. It is the L2 projection of -D grad(phi), and
# the gradient of a bilinear solution is only O(h) accurate ON the boundary --
# the superconvergence points are interior -- while the boundary trace is exactly
# what the coupling reads. MEASURED on this cell with that projection: the
# coupling converged at every level and the field order was 1.758 then 1.593,
# drifting DOWN. The pass band is [1.6, 2.4], so a CORRECT run grades
# CONFIDENTLY_WRONG at the third level. path_readiness.json recorded the drift
# and recorded that whether it crosses the band edge on a specific instance was
# "not established"; it is established here, and it does.
#
# THE CONSISTENT (REACTION) FLUX, WITHOUT ACCESS TO 4C's REACTION. 4C exposes no
# residual, so the Q1 system is re-assembled here in numpy on 4C's OWN mesh, with
# 4C's own material and source, and evaluated at 4C's OWN solution:
#     int_Gamma qn phi_i ds = -(A u - b)_i
# with A and b carrying no boundary condition. This is post-processing of 4C's
# answer, not a second solve -- nothing is solved and u is not touched.
#
# It is only the reaction if the re-assembly IS 4C's discretisation, so that is
# CHECKED rather than assumed: on every interior node the discrete equation holds
# and the residual must vanish. The check is printed and written to the log, so a
# quadrature or material mismatch shows up as a number instead of as a quietly
# wrong flux.
def _consistent_flux():
    import scipy.sparse as sp
    import scipy.sparse.linalg  # noqa: F401
    hx = (ext[0][1] - ext[0][0]) / n[0]
    hy = (ext[1][1] - ext[1][0]) / n[1]
    gp = [-1 / np.sqrt(3.0), 1 / np.sqrt(3.0)]
    # The STIFFNESS needs only 2x2 (exact for bilinear on a rectangle), but the
    # LOAD is the integral of a high-degree polynomial source against the shape
    # functions, and 2x2 is not exact for it. A quadrature that differs from 4C's
    # makes the re-assembled residual differ from 4C's reaction by the difference
    # of the two rules -- measured 4.3e-03 rising to 3.0e-02 of the interface
    # scale with 2x2. A 5-point Gauss rule integrates degree 9 exactly, which
    # covers every source in this family, so the load is then EXACT and any
    # remaining mismatch is 4C's own quadrature error rather than a modelling
    # difference. The interior residual is reported either way.
    gl = np.array([-0.9061798459386640, -0.5384693101056831, 0.0,
                   0.5384693101056831, 0.9061798459386640])
    wl = np.array([0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                   0.4786286704993665, 0.2369268850561891])
    fsrc = W.make_fun(cfg["source"], 2)
    Ke = np.zeros((4, 4))
    for a in gp:
        for b in gp:
            # bilinear shape functions on [-1,1]^2, nodes (0,0),(1,0),(1,1),(0,1)
            dNa = 0.25 * np.array([[-(1 - b), (1 - b), (1 + b), -(1 + b)],
                                   [-(1 - a), -(1 + a), (1 + a), (1 - a)]])
            J = np.diag([hx / 2, hy / 2])
            B = np.linalg.solve(J, dNa)
            Ke += kval * (B.T @ B) * np.linalg.det(J)
    rows, cols, vals, bvec = [], [], [], np.zeros(len(P))
    for j in range(n[1]):
        for i in range(n[0]):
            g = [nid[i, j] - 1, nid[i + 1, j] - 1,
                 nid[i + 1, j + 1] - 1, nid[i, j + 1] - 1]
            for r_ in range(4):
                for c_ in range(4):
                    rows.append(g[r_]); cols.append(g[c_]); vals.append(Ke[r_, c_])
            for a, wa in zip(gl, wl):
                for b, wb in zip(gl, wl):
                    N = 0.25 * np.array([(1 - a) * (1 - b), (1 + a) * (1 - b),
                                         (1 + a) * (1 + b), (1 - a) * (1 + b)])
                    xg = ax[0][i] + hx * (a + 1) / 2
                    yg = ax[1][j] + hy * (b + 1) / 2
                    bvec[g] += (N * float(fsrc(xg, yg)) * wa * wb
                                * (hx * hy / 4))
    A = sp.coo_matrix((vals, (rows, cols)), shape=(len(P), len(P))).tocsr()
    res = A @ T - bvec
    free = ~(outer | ifm)
    scale = max(float(np.max(np.abs(res[ifm]))), 1e-300)
    interior = float(np.max(np.abs(res[free]))) / scale
    w = np.full(len(inode), hy if axis == 0 else hx)
    w[0] = w[-1] = 0.5 * w[0]
    q = -res[inode] / w
    bad = outer[inode]
    good = np.where(~bad)[0]
    for i in np.where(bad)[0]:
        q[i] = q[good[np.argmin(np.abs(good - i))]]
    return q, interior


if cfg["side"] == "dirichlet":
    Q, interior_res = _consistent_flux()
    fit_note += (f"  reassembly_interior_residual_rel = {interior_res:.3e}"
                 if not fit_note else
                 f"\nreassembly_interior_residual_rel = {interior_res:.3e}")
    if interior_res > 1e-6:
        print(f"[4C] WARNING: the re-assembled operator is not 4C's: interior "
              f"residual {interior_res:.3e} of the interface scale. The "
              f"exported flux is NOT the discrete reaction.", file=sys.stderr)
else:
    # The Dirichlet partner reads this side's VALUES, not its flux, so the
    # projection is harmless here; the reaction formula must not be used on free
    # dofs, where the discrete equations hold and it would export zero.
    Q = S * QD[inode]

W.write_nodes("nodes.csv", P, T.reshape(-1, 1))
W.write_log(cfg, len(P),
            f"number of nodes = {len(P)}\nnumber of elements = {len(elems)}\n"
            + fit_note)
print(f"[4C {cfg['sidename']} {cfg['side']}] NDOF={len(P)} "
      f"iface_n={len(inode)} T=[{T.min():.6g},{T.max():.6g}] "
      f"q=[{Q.min():.6g},{Q.max():.6g}] {fit_note}")
W.write_exports(ipts, T[inode].reshape(-1, 1), Q.reshape(-1, 1), "temperature")
