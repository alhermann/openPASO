"""Kratos as NEUMANN side - fixed version."""
import json
from pathlib import Path
import numpy as np
import sys

import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication

CFG = json.loads(Path("config.json").read_text())
NX, NY = CFG["nx"], CFG["ny"]
X0, X1, Y0, Y1 = CFG["x0"], CFG["x1"], CFG["y0"], CFG["y1"]
KV = CFG["k"]
LEVEL = CFG.get("level", 1)

def source(x, y):
    return (-3*x**3*y/20000 + x**3/12500 - 6167*x**2*y/80000 + 13967*x**2/150000 
            - 3*x*y**3/20000 + 3*x*y**2/12500 - 45948751*x*y/6400000 + 4904887*x/480000 
            - 6167*y**3/240000 + 13967*y**2/150000 + 139208181*y/12800000 - 994403/640000)

IFACE_X = X0  # Interface on left

imp_data = {}
if Path("imports.json").is_file():
    imp_data = json.loads(Path("imports.json").read_text() or "{}")

def get_imported_flux(y_coord):
    if not imp_data or "A" not in imp_data:
        return 0.0
    d = imp_data["A"]
    coords = d.get("coordinates", [])
    fluxes = d.get("normal_fluxes", [])
    if not coords or not fluxes:
        return 0.0
    ys = [c[1] for c in coords]
    ys_sorted = sorted(zip(ys, fluxes))
    xs, qs = zip(*ys_sorted)
    t = max(min(y_coord, xs[-1]), xs[0])
    for i in range(len(xs)-1):
        if xs[i] <= t <= xs[i+1]:
            w = 0.0 if xs[i+1] == xs[i] else (t - xs[i]) / (xs[i+1] - xs[i])
            return qs[i] + w * (qs[i+1] - qs[i])
    return qs[-1]

hx = (X1 - X0) / NX
hy = (Y1 - Y0) / NY
nodes = []
node_id = {}
nid = 0
for j in range(NY + 1):
    for i in range(NX + 1):
        x = X0 + i * hx
        y = Y0 + j * hy
        nodes.append((x, y))
        node_id[(i, j)] = nid
        nid += 1

tris = []
for j in range(NY):
    for i in range(NX):
        a, b, c, d = node_id[(i, j)], node_id[(i+1, j)], node_id[(i+1, j+1)], node_id[(i, j+1)]
        tris.append([a, b, c])
        tris.append([a, c, d])

model = KM.Model()
mp = model.CreateModelPart("thermal")
mp.ProcessInfo[KM.DOMAIN_SIZE] = 2

for v in (KM.TEMPERATURE, KM.CONDUCTIVITY, KM.HEAT_FLUX, KM.FACE_HEAT_FLUX, 
          KM.REACTION_FLUX, KM.DENSITY, KM.SPECIFIC_HEAT, KM.VELOCITY, KM.MESH_VELOCITY):
    mp.AddNodalSolutionStepVariable(v)

settings = KM.ConvectionDiffusionSettings()
settings.SetUnknownVariable(KM.TEMPERATURE)
settings.SetDiffusionVariable(KM.CONDUCTIVITY)
settings.SetVolumeSourceVariable(KM.HEAT_FLUX)
settings.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
settings.SetDensityVariable(KM.DENSITY)
settings.SetSpecificHeatVariable(KM.SPECIFIC_HEAT)
settings.SetVelocityVariable(KM.VELOCITY)
settings.SetMeshVelocityVariable(KM.MESH_VELOCITY)
settings.SetReactionVariable(KM.REACTION_FLUX)
mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, settings)

mp.SetBufferSize(1)
props = mp.CreateNewProperties(1)

for idx, (px, py) in enumerate(nodes):
    n = mp.CreateNewNode(idx + 1, float(px), float(py), 0.0)
    n.SetSolutionStepValue(KM.CONDUCTIVITY, KV)
    n.SetSolutionStepValue(KM.HEAT_FLUX, float(source(px, py)))
    n.SetSolutionStepValue(KM.DENSITY, 1.0)
    n.SetSolutionStepValue(KM.SPECIFIC_HEAT, 1.0)

for el_idx, el in enumerate(tris):
    mp.CreateNewElement("LaplacianElement2D3N", el_idx + 1,
                        [int(v) + 1 for v in el], props)

KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)

# Outer boundary Dirichlet (right, top, bottom) - u=0
for j in range(NY + 1):
    nd = mp.GetNode(node_id[(NX, j)] + 1)
    nd.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
    nd.Fix(KM.TEMPERATURE)

for i in range(NX + 1):
    nd = mp.GetNode(node_id[(i, NY)] + 1)
    nd.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
    nd.Fix(KM.TEMPERATURE)
    nd = mp.GetNode(node_id[(i, 0)] + 1)
    nd.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
    nd.Fix(KM.TEMPERATURE)

# Interface nodes (left edge, interior only)
interface_nodes_ij = [(0, j) for j in range(1, NY)]
interface_node_ids = [node_id[n] for n in interface_nodes_ij]

# Apply imported flux
for ij, nid_val in zip(interface_nodes_ij, interface_node_ids):
    y_coord = Y0 + ij[1] * hy
    q_in = get_imported_flux(y_coord)
    nd = mp.GetNode(nid_val + 1)
    nd.SetSolutionStepValue(KM.FACE_HEAT_FLUX, float(q_in))

# Create FluxCondition on interface edges
cond_id = 1
for j in range(NY):
    n1 = node_id[(0, j)] + 1
    n2 = node_id[(0, j+1)] + 1
    mp.CreateNewCondition("FluxCondition2D2N", cond_id, [n1, n2], props)
    cond_id += 1

lin = KM.LinearSolverFactory().Create(
    KM.Parameters('{"solver_type":"skyline_lu_factorization"}'))
scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
bas = KM.ResidualBasedBlockBuilderAndSolver(lin)
strat = KM.ResidualBasedLinearStrategy(mp, scheme, bas, True, False, False, False)
strat.SetEchoLevel(1)
mp.ProcessInfo[KM.DELTA_TIME] = 1.0
mp.CloneTimeStep(1.0)
strat.Initialize()
strat.Solve()

u_at_nodes = [mp.GetNode(i + 1).GetSolutionStepValue(KM.TEMPERATURE) for i in range(len(nodes))]

# Interface values
y_if = [Y0 + j * hy for j in range(NY + 1)]
T_if = [mp.GetNode(node_id[(0, j)] + 1).GetSolutionStepValue(KM.TEMPERATURE) for j in range(NY + 1)]

# Compute outward flux (outward normal on left is (-1,0))
# q = -k * du/dn = -k * (-du/dx) = k * du/dx
# Use forward difference
q_own = []
for j in range(1, NY):  # Interior only
    u_if = u_at_nodes[node_id[(0, j)]]
    u_right = u_at_nodes[node_id[(1, j)]]
    du_dx = (u_right - u_if) / hx
    q = KV * du_dx  # Outward flux (normal is -x direction)
    q_own.append(q)

co = [[IFACE_X, Y0 + j * hy] for j in range(1, NY)]
vals_if = [T_if[j] for j in range(1, NY)]
q_own_if = q_own

json.dump({
    "field_name": "temperature",
    "coordinates": co,
    "values": vals_if,
    "normal_fluxes": q_own_if,
    "n_points": len(co)
}, open("exports.json", "w"), indent=2)

with open(f"field_level{LEVEL}.csv", "w") as f:
    f.write("x,y,u\n")
    for (x, y), u in zip(nodes, u_at_nodes):
        f.write(f"{x:.11e},{y:.11e},{u:.11e}\n")

with open(f"interface_level{LEVEL}.csv", "w") as f:
    f.write("x,y,u,qn\n")
    for (x, y), u, q in zip(co, vals_if, q_own_if):
        f.write(f"{x:.11e},{y:.11e},{u:.11e},{q:.11e}\n")

print(f"NDOF = {len(nodes)}")
print(f"Kratos done: max|u|={max(abs(u) for u in u_at_nodes):.6e}")
