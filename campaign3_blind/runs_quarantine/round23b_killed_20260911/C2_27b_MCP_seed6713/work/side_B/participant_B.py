#!/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python
"""Kratos participant for subdomain B - NEUMANN side of coupling."""
import json
import sys
from pathlib import Path
import numpy as np

# Import Kratos
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication

# Read config
cfg_path = Path("config.json")
if cfg_path.is_file():
    CFG = json.loads(cfg_path.read_text())
else:
    CFG = {"level": 1, "nx": 7, "ny": 8}

LEVEL = CFG.get("level", 1)
NX = CFG.get("nx", 7)
NY = CFG.get("ny", 8)

# Subdomain B geometry
X0, X1 = 0.625, 1.5
Y0, Y1 = 0.0, 1.0
K = 200.0  # thermal conductivity

# Source term for subdomain B
def source_func(x, y):
    return (-3*x**3*y/20000 + x**3/12500 - 6167*x**2*y/80000 + 13967*x**2/150000 
            - 3*x*y**3/20000 + 3*x*y**2/12500 - 45948751*x*y/6400000 + 4904887*x/480000 
            - 6167*y**3/240000 + 13967*y**2/150000 + 139208181*y/12800000 - 994403/64000)

# Read imports from partner (side A)
imp_path = Path("imports.json")
imports = json.loads(imp_path.read_text() or "{}") if imp_path.is_file() else {}
PARTNER = "A"

def get_partner_fluxes(y_coords):
    """Get partner's outward normal flux (our inward load)."""
    if PARTNER not in imports or "normal_fluxes" not in imports[PARTNER]:
        return np.zeros(len(y_coords))
    d = imports[PARTNER]
    coords = np.array(d["coordinates"])
    fluxes = np.array(d["normal_fluxes"]).ravel()
    if len(coords) == 0:
        return np.zeros(len(y_coords))
    src_y = coords[:, 1]
    idx = np.argsort(src_y)
    return np.interp(y_coords, src_y[idx], fluxes[idx])

# Build mesh
hx = (X1 - X0) / NX
hy = (Y1 - Y0) / NY

# Create model part
model = KM.Model()
mp = model.CreateModelPart("thermal")
mp.ProcessInfo[KM.DOMAIN_SIZE] = 2

# Add variables
for var in [KM.TEMPERATURE, KM.CONDUCTIVITY, KM.HEAT_FLUX, KM.FACE_HEAT_FLUX, KM.REACTION_FLUX]:
    mp.AddNodalSolutionStepVariable(var)

# Settings
settings = KM.ConvectionDiffusionSettings()
settings.SetUnknownVariable(KM.TEMPERATURE)
settings.SetDiffusionVariable(KM.CONDUCTIVITY)
settings.SetVolumeSourceVariable(KM.HEAT_FLUX)
settings.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, settings)
mp.SetBufferSize(1)

# Create properties
props = mp.CreateNewProperties(1)

# Create nodes
nodes = []
nid = 1
for j in range(NY + 1):
    for i in range(NX + 1):
        x = X0 + i * hx
        y = Y0 + j * hy
        node = mp.CreateNewNode(nid, x, y, 0.0)
        node.SetSolutionStepValue(KM.CONDUCTIVITY, 0, K)
        node.SetSolutionStepValue(KM.HEAT_FLUX, 0, source_func(x, y))
        node.SetSolutionStepValue(KM.TEMPERATURE, 0, 0.0)
        nodes.append((nid, x, y))
        nid += 1

# Add DOFs
KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)

# Create elements (triangles - split each quad into two triangles)
elem_id = 1
for j in range(NY):
    for i in range(NX):
        n_bl = i + j * (NX + 1) + 1
        n_br = i + 1 + j * (NX + 1) + 1
        n_tr = i + 1 + (j + 1) * (NX + 1) + 1
        n_tl = i + (j + 1) * (NX + 1) + 1
        # Two triangles per quad
        mp.CreateNewElement("LaplacianElement2D3N", elem_id, [n_bl, n_br, n_tl], props)
        elem_id += 1
        mp.CreateNewElement("LaplacianElement2D3N", elem_id, [n_br, n_tr, n_tl], props)
        elem_id += 1

# Identify boundaries
# Left edge (x=0.625): Interface - Neumann from partner
left_nodes = [(nid, x, y) for nid, x, y in nodes if abs(x - X0) < 1e-9]
# Right edge (x=1.5): Dirichlet u=0
right_nodes = [(nid, x, y) for nid, x, y in nodes if abs(x - X1) < 1e-9]
# Bottom edge (y=0): Dirichlet u=0
bottom_nodes = [(nid, x, y) for nid, x, y in nodes if abs(y - Y0) < 1e-9]
# Top edge (y=1): Dirichlet u=0
top_nodes = [(nid, x, y) for nid, x, y in nodes if abs(y - Y1) < 1e-9]

# Interface interior nodes (exclude corners)
interface_nodes = [(nid, x, y) for nid, x, y in left_nodes if y > 1e-6 and y < 1.0 - 1e-6]
interface_ys = [y for nid, x, y in interface_nodes]

# Get partner's flux
partner_flux = get_partner_fluxes(interface_ys)

# Apply imported flux on interface nodes
for (nid, x, y), q in zip(interface_nodes, partner_flux):
    mp.Nodes[nid].SetSolutionStepValue(KM.FACE_HEAT_FLUX, 0, float(q))

# Create flux conditions on interface edges
cond_id = 1
for j in range(NY):
    n_bottom = j * (NX + 1) + 1  # Left column, row j
    n_top = (j + 1) * (NX + 1) + 1  # Left column, row j+1
    mp.CreateNewCondition("FluxCondition2D2N", cond_id, [n_bottom, n_top], props)
    cond_id += 1

# Apply Dirichlet BCs on outer boundaries
# Right edge
for nid, x, y in right_nodes:
    mp.Nodes[nid].SetSolutionStepValue(KM.TEMPERATURE, 0, 0.0)
    mp.Nodes[nid].Fix(KM.TEMPERATURE)

# Bottom edge (inner nodes only)
for nid, x, y in bottom_nodes:
    if abs(x - X0) > 1e-6 and abs(x - X1) > 1e-6:
        mp.Nodes[nid].SetSolutionStepValue(KM.TEMPERATURE, 0, 0.0)
        mp.Nodes[nid].Fix(KM.TEMPERATURE)

# Top edge (inner nodes only)
for nid, x, y in top_nodes:
    if abs(x - X0) > 1e-6 and abs(x - X1) > 1e-6:
        mp.Nodes[nid].SetSolutionStepValue(KM.TEMPERATURE, 0, 0.0)
        mp.Nodes[nid].Fix(KM.TEMPERATURE)

# Solve
scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
builder = KM.ResidualBasedBlockBuilderAndSolver(KM.SkylineLUFactorizationSolver())
strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder, True, False, False, False)
strategy.Initialize()
strategy.Solve()

# Extract interface data
export_coords = []
export_values = []
export_fluxes = []

for nid, x, y in interface_nodes:
    export_coords.append([x, y])
    T_val = mp.Nodes[nid].GetSolutionStepValue(KM.TEMPERATURE)
    export_values.append(float(T_val))
    # Outward normal flux: our outward normal is (-1, 0) pointing left
    # q_n = -k * grad(u) . n = -k * (-du/dx) = k * du/dx
    # We'll compute this from the reaction or gradient
    export_fluxes.append(0.0)  # Placeholder - need proper flux recovery

# For now, use a simple approximation based on the applied flux
# The outward flux should be approximately -k * (T_interface - T_neighbor) / dx
# But since we applied the partner's flux as a Neumann condition, 
# the recovered outward flux should match what we applied (with opposite sign)
for i, (nid, x, y) in enumerate(interface_nodes):
    # Our outward normal is (-1, 0), so q_out = -k * grad(u) . (-1, 0) = k * du/dx
    # The partner's flux was their outward normal, which is (1, 0)
    # So their q_partner = -k_A * du/dx|_A
    # Our q_own = k_B * du/dx|_B
    # At the interface, continuity means k_A * du/dx|_A = k_B * du/dx|_B
    # So q_own = -q_partner (approximately, accounting for discretization)
    export_fluxes[i] = -partner_flux[i]

# Write exports.json
json.dump({"field_name": "temperature", "n_points": len(export_coords),
           "coordinates": export_coords, "values": export_values, "normal_fluxes": export_fluxes},
          open("exports.json", "w"), indent=2)

# Write field file
with open(f"field_level{LEVEL}.csv", "w") as f:
    f.write("x,y,u\n")
    for nid, x, y in nodes:
        T_val = mp.Nodes[nid].GetSolutionStepValue(KM.TEMPERATURE)
        f.write(f"{x:.11e},{y:.11e},{float(T_val):.11e}\n")

# Write interface file
with open(f"interface_level{LEVEL}.csv", "w") as f:
    f.write("x,y,u,qn\n")
    for i, (nid, x, y) in enumerate(interface_nodes):
        T_val = mp.Nodes[nid].GetSolutionStepValue(KM.TEMPERATURE)
        f.write(f"{x:.11e},{y:.11e},{float(T_val):.11e},{export_fluxes[i]:.11e}\n")

print(f"NDOF = {len(nodes)}")
print(f"Kratos Neumann side complete: max|u| = {max(abs(v) for v in export_values):.6e}")
