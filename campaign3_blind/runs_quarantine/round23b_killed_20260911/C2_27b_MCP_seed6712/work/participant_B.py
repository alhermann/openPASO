#!/usr/bin/env python3
"""Kratos Multiphysics participant for OASiS couple driver - NEUMANN side (subdomain B).

Subdomain B: (0.625, 1.5) x (0, 1), k=200
Interface at x=0.625 (left edge of subdomain B)
Receives outward normal flux from partner (4C), applies as Neumann BC
Exports interface temperature values
"""
import json
import sys
from pathlib import Path
import numpy as np

# Import Kratos
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication

# ============================================================================
# CONFIGURATION
# ============================================================================
CFG = json.loads(Path("config.json").read_text())
NX, NY = CFG["nx"], CFG["ny"]
LEVEL = CFG.get("level", 1)

# Subdomain B geometry
X0, X1 = 0.625, 1.5
Y0, Y1 = 0.0, 1.0
K = 200.0  # conductivity

# Interface is at left edge (x = 0.625)
IFACE_X = 0.625
ON_MAX_X = False  # interface is at x-min, not x-max
OUTER_X = X1  # outer Dirichlet boundary is at x = 1.5

# Partner name in imports.json
PARTNER = "A"

# Source term for subdomain B (Python function)
def F_SRC(x, y):
    """Source term f(x,y) for subdomain B."""
    return (-3*x**3*y/20000 + x**3/12500 - 6167*x**2*y/80000 + 13967*x**2/150000 
            - 3*x*y**3/20000 + 3*x*y**2/12500 - 45948751*x*y/6400000 + 4904887*x/480000 
            - 6167*y**3/240000 + 13967*y**2/150000 + 139208181*y/12800000 - 994403/64000)

T_OUTER = 0.0  # Dirichlet value on outer boundary (x = 1.5)
Q_INIT = 0.0   # fallback interface flux on iteration 1

# ============================================================================
# PARTNER INTERFACE DATA (handshake)
# ============================================================================
def read_imports():
    """Read imports.json and return partner's data."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        imp = json.loads(p.read_text() or "{}")
        return imp.get(PARTNER) or None
    except json.JSONDecodeError:
        return None

def sample(imp, key, fallback, y_coords):
    """Map partner's samples onto this participant's interface points."""
    if not imp or not imp.get("coordinates"):
        return np.full(len(y_coords), float(fallback))
    ys = np.array([c[1] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key) or [], float).ravel()
    if vs.size != ys.size:
        return np.full(len(y_coords), float(fallback))
    o = np.argsort(ys)
    return np.interp(y_coords, ys[o], vs[o])

# ============================================================================
# BUILD MODEL PART
# ============================================================================
hx = (X1 - X0) / NX
hy = (Y1 - Y0) / NY

# Create model and model part
model = KM.Model()
mp = model.CreateModelPart('thermal')

# Set domain size
mp.ProcessInfo[KM.DOMAIN_SIZE] = 2

# Create settings
settings = KM.ConvectionDiffusionSettings()
settings.SetUnknownVariable(KM.TEMPERATURE)
settings.SetDiffusionVariable(KM.CONDUCTIVITY)
settings.SetVolumeSourceVariable(KM.HEAT_FLUX)
settings.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, settings)

# Add nodal solution step variables BEFORE creating nodes
mp.AddNodalSolutionStepVariable(KM.TEMPERATURE)
mp.AddNodalSolutionStepVariable(KM.CONDUCTIVITY)
mp.AddNodalSolutionStepVariable(KM.HEAT_FLUX)
mp.AddNodalSolutionStepVariable(KM.FACE_HEAT_FLUX)
mp.AddNodalSolutionStepVariable(KM.REACTION_FLUX)

# Set buffer size
mp.SetBufferSize(1)

# Create properties (placeholder)
props = mp.CreateNewProperties(1)

# Generate nodes
nid = {}  # (ix, iy) -> node id (1-based)
node_id = 1
for j in range(NY + 1):
    y = Y0 + j * hy
    for i in range(NX + 1):
        x = X0 + i * hx
        n = mp.CreateNewNode(node_id, x, y, 0.0)
        nid[(i, j)] = node_id
        
        # Set conductivity (nodal)
        n.SetSolutionStepValue(KM.CONDUCTIVITY, K)
        
        # Set source term
        n.SetSolutionStepValue(KM.HEAT_FLUX, F_SRC(x, y))
        
        # Initialize FACE_HEAT_FLUX to 0
        n.SetSolutionStepValue(KM.FACE_HEAT_FLUX, 0.0)
        
        node_id += 1

N_NODES = len(mp.Nodes)

# Identify interface and outer boundary nodes
# Interface is at x = X0 = 0.625 (left edge, i = 0)
# Outer Dirichlet is at x = X1 = 1.5 (right edge, i = NX)
i_if = 0      # interface column index
i_out = NX    # outer Dirichlet column index

# Interface nodes (exclude corners which are on outer boundary)
interface_node_ids = [nid[(i_if, j)] for j in range(1, NY)]  # interior only

# Outer boundary nodes (u=0): x=X1 (right), y=Y0 (bottom), y=Y1 (top)
outer_nodes = []
for j in range(NY + 1):
    outer_nodes.append(nid[(i_out, j)])  # right edge x=X1
for i in range(i_if + 1, i_out + 1):
    outer_nodes.append(nid[(i, 0)])       # bottom edge y=Y0
    outer_nodes.append(nid[(i, NY)])      # top edge y=Y1

# Apply outer Dirichlet conditions (u=0)
for nid_val in outer_nodes:
    n = mp.Nodes[nid_val]
    n.SetSolutionStepValue(KM.TEMPERATURE, T_OUTER)
    n.Fix(KM.TEMPERATURE)

# ============================================================================
# ADD DOFs
# ============================================================================
KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)

# ============================================================================
# CREATE ELEMENTS
# ============================================================================
elem_id = 1
for j in range(NY):
    for i in range(NX):
        n_bl = nid[(i, j)]
        n_br = nid[(i+1, j)]
        n_tr = nid[(i+1, j+1)]
        n_tl = nid[(i, j+1)]
        
        # Split quad into two triangles
        mp.CreateNewElement('LaplacianElement2D3N', elem_id, [n_bl, n_br, n_tr], props)
        elem_id += 1
        mp.CreateNewElement('LaplacianElement2D3N', elem_id, [n_bl, n_tr, n_tl], props)
        elem_id += 1

# ============================================================================
# APPLY IMPORTED FLUX AS NEUMANN BC ON INTERFACE
# ============================================================================
# Read imported flux from partner
q_in = sample(read_imports(), "normal_fluxes", Q_INIT, np.array([Y0 + j * hy for j in range(NY + 1)]))

# Set FACE_HEAT_FLUX on interface nodes
# Note: FACE_HEAT_FLUX is INWARD normal flux, but we receive OUTWARD from partner
# The partner's outward flux is our inward load, so apply unchanged
y_if = np.array([Y0 + j * hy for j in range(NY + 1)])
for j in range(NY + 1):
    n = mp.Nodes[nid[(i_if, j)]]
    n.SetSolutionStepValue(KM.FACE_HEAT_FLUX, float(q_in[j]))

# Create FluxCondition2D2N on interface edges (REQUIRED for flux to be integrated)
cond_id = 1
for j in range(NY):
    n1 = nid[(i_if, j)]
    n2 = nid[(i_if, j+1)]
    mp.CreateNewCondition("FluxCondition2D2N", cond_id, [n1, n2], props)
    cond_id += 1

# ============================================================================
# SOLVE
# ============================================================================
scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
builder = KM.ResidualBasedBlockBuilderAndSolver(KM.SkylineLUFactorizationSolver())
strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder, True, False, False, False)

strategy.Initialize()
strategy.Solve()

# ============================================================================
# READ INTERFACE SOLUTION
# ============================================================================
# Get interface node coordinates
y_if_vals = [Y0 + j * hy for j in range(NY + 1)]

# Read temperature at interface nodes
iface_nodes = [mp.Nodes[nid[(i_if, j)]] for j in range(NY + 1)]
T = np.array([n.GetSolutionStepValue(KM.TEMPERATURE) for n in iface_nodes])

# ============================================================================
# RECOVER OUTWARD NORMAL FLUX
# ============================================================================
# For Neumann side, recover flux from assembled conditions
info = mp.ProcessInfo
rhs_i = np.zeros(N_NODES + 1)
w_i = np.zeros(N_NODES + 1)

for cond in mp.Conditions:
    vec = KM.Vector()
    cond.CalculateRightHandSide(vec, info)
    nds = cond.GetNodes()
    p0, p1 = nds[0], nds[1]
    seg = ((p1.X - p0.X) ** 2 + (p1.Y - p0.Y)**2) ** 0.5
    for nd in nds:
        rhs_i[nd.Id] += float(vec[0])  # First component
        w_i[nd.Id] += 0.5 * seg

ids_if = [nid[(i_if, j)] for j in range(NY + 1)]
r_if = np.array([rhs_i[i] for i in ids_if])
wq = np.array([w_i[i] for i in ids_if])

# Outward normal at left interface is (-1, 0)
# q_outward = -r/w (consistent with sign convention)
Q = np.where(np.abs(wq) > 1e-14, -r_if / np.maximum(np.abs(wq), 1e-300) * np.sign(np.where(wq == 0, 1.0, wq)), 0.0)

# Exclude corners from flux (they have mixed reactions)
Q[[0, -1]] = 0.0

# ============================================================================
# CONSERVATION CHECK
# ============================================================================
load_iface = float(hy * (0.5 * q_in[0] + q_in[1:-1].sum() + 0.5 * q_in[-1]))
load_vol = 0.0
for el in mp.Elements:
    nds = el.GetNodes()
    x = [n.X for n in nds]
    y = [n.Y for n in nds]
    det = (x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0])
    load_vol += (0.5 * abs(det)) * sum(n.GetSolutionStepValue(KM.HEAT_FLUX) for n in nds) / 3.0

react = sum(mp.Nodes[nid[(i_out, j)]].GetSolutionStepValue(KM.REACTION_FLUX) for j in range(NY + 1))
imb_abs = abs(react + load_vol + load_iface)
scale = max(abs(react), abs(load_vol), abs(load_iface))

if scale <= 1e-10 * K * max(1.0, abs(T_OUTER)) * (Y1 - Y0):
    bal = f"balance trivial (no heat flow yet, |imbalance|={imb_abs:.3e})"
else:
    bal = f"balance |sum(reactions)+vol+iface| = {imb_abs:.3e} abs / {imb_abs/scale:.3e} rel"

print(f"[kratos neumann] interface n={len(T)} q_applied=[{q_in.min():.6g},{q_in.max():.6g}] T=[{T.min():.6g},{T.max():.6g}] {bal}")

# ============================================================================
# WRITE EXPORTS.JSON
# ============================================================================
# NEUMANN side: export values (temperature trace) and normal_fluxes
coords = [[float(n.X), float(n.Y)] for n in iface_nodes]

exports = {
    "field_name": "temperature",
    "n_points": len(coords),
    "coordinates": coords,
    "values": [float(t) for t in T],
    "normal_fluxes": [float(q) for q in Q]
}

json.dump(exports, open("exports.json", "w"), indent=2)

# ============================================================================
# WRITE PER-LEVEL FILES
# ============================================================================
# field_level{LEVEL}_B.csv
with open(f"field_level{LEVEL}_B.csv", "w") as f:
    f.write("x,y,u\n")
    for n in mp.Nodes:
        u_val = n.GetSolutionStepValue(KM.TEMPERATURE)
        f.write(f"{float(n.X):.11e},{float(n.Y):.11e},{float(u_val):.11e}\n")

# interface_level{LEVEL}_B.csv
with open(f"interface_level{LEVEL}_B.csv", "w") as f:
    f.write("x,y,u,qn\n")
    for n, t, q in zip(iface_nodes, T, Q):
        f.write(f"{float(n.X):.11e},{float(n.Y):.11e},{float(t):.11e},{float(q):.11e}\n")

# ============================================================================
# PRINT NDOF
# ============================================================================
print(f"NDOF = {N_NODES}")
