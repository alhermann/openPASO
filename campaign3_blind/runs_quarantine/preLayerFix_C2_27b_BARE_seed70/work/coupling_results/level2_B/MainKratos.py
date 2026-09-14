#!/usr/bin/env python3
"""Kratos solver for subdomain B - Neumann side of coupling"""
import json
from pathlib import Path
import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication

# Problem parameters
L_A = 0.625
L_B = 0.875
H = 1.0
K_B = 200.0
nx = 14
ny = 16

# Incoming flux from partner (interpolated to our interface nodes)
q_in_values = [np.float64(-0.001329241311675792), np.float64(-0.0006804392665828319), np.float64(-0.0008727472111592577), np.float64(-0.0010887002553250672), np.float64(-0.0012603735142633406), np.float64(-0.0013859315559946904), np.float64(-0.0014711311267418545), np.float64(-0.0015205146371190352), np.float64(-0.001536694186497033), np.float64(-0.0015205102556362865), np.float64(-0.0014711226682909695), np.float64(-0.0013859233436601158), np.float64(-0.0012603942588224736), np.float64(-0.0010889131771920748), np.float64(-0.0008741128983590866), np.float64(-0.0006885297818793169), np.float64(-0.001392554058691984)]

model = KM.Model()
mp = model.CreateModelPart("thermal")
mp.ProcessInfo[KM.DOMAIN_SIZE] = 2

# Setup ConvectionDiffusion settings
settings = KM.ConvectionDiffusionSettings()
settings.SetUnknownVariable(KM.TEMPERATURE)
settings.SetDiffusionVariable(KM.CONDUCTIVITY)
settings.SetVolumeSourceVariable(KM.HEAT_FLUX)
settings.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, settings)

for v in (KM.TEMPERATURE, KM.CONDUCTIVITY, KM.HEAT_FLUX, KM.FACE_HEAT_FLUX):
    mp.AddNodalSolutionStepVariable(v)

mp.SetBufferSize(1)

props = mp.CreateNewProperties(1)

# Create nodes
nid = {}
cnt = 1
for j in range(ny + 1):
    for i in range(nx + 1):
        x = L_A + L_B * i / nx
        y = H * j / ny
        mp.CreateNewNode(cnt, x, y, 0.0)
        nid[(i, j)] = cnt
        cnt += 1

# Create elements (triangles)
eid = 1
for j in range(ny):
    for i in range(nx):
        a, b, c, d = nid[(i, j)], nid[(i + 1, j)], nid[(i + 1, j + 1)], nid[(i, j + 1)]
        mp.CreateNewElement("LaplacianElement2D3N", eid, [a, b, d], props); eid += 1
        mp.CreateNewElement("LaplacianElement2D3N", eid, [b, c, d], props); eid += 1

# Create line conditions for Neumann BC on left edge
cid = 1
for j in range(ny):
    mp.CreateNewCondition("FluxCondition2D2N", cid,
                          [nid[(0, j)], nid[(0, j + 1)]], props)
    cid += 1

# Set material properties and sources (constant source f = 1.0)
for node in mp.Nodes:
    node.SetSolutionStepValue(KM.CONDUCTIVITY, K_B)
    node.SetSolutionStepValue(KM.HEAT_FLUX, 1.0)  # Constant source term
    node.SetSolutionStepValue(KM.FACE_HEAT_FLUX, 0.0)

# Apply Neumann BC on left edge (interface) - incoming flux from A
for j in range(ny + 1):
    mp.Nodes[nid[(0, j)]].SetSolutionStepValue(KM.FACE_HEAT_FLUX, float(q_in_values[j]))

# Apply Dirichlet BC on outer boundaries (u=0)
# Right edge (x = L_A + L_B)
for j in range(ny + 1):
    n = mp.Nodes[nid[(nx, j)]]
    n.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
    n.Fix(KM.TEMPERATURE)

# Bottom edge (y = 0) - skip corners
for i in range(1, nx):
    n = mp.Nodes[nid[(i, 0)]]
    n.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
    n.Fix(KM.TEMPERATURE)

# Top edge (y = H) - skip corners
for i in range(1, nx):
    n = mp.Nodes[nid[(i, ny)]]
    n.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
    n.Fix(KM.TEMPERATURE)

# Add DOFs
KM.VariableUtils().AddDof(KM.TEMPERATURE, mp)

# Solve
scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
linear_solver = KM.SkylineLUFactorizationSolver()
builder = KM.ResidualBasedBlockBuilderAndSolver(linear_solver)
strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder, False, False, False, False)
strategy.Initialize()
strategy.Solve()

# Extract full solution for probe point evaluation
all_nodes = []
all_temps = []
for node in mp.Nodes:
    all_nodes.append([node.X, node.Y])
    all_temps.append(node.GetSolutionStepValue(KM.TEMPERATURE))

all_nodes = np.array(all_nodes)
all_temps = np.array(all_temps)

# Save full solution to file
np.savez("solution_full.npz", nodes=all_nodes, temps=all_temps)

# Extract interface data
dx = L_B / nx
T_if = np.array([mp.Nodes[nid[(0, j)]].GetSolutionStepValue(KM.TEMPERATURE)
                 for j in range(ny + 1)])
T_near = np.array([mp.Nodes[nid[(1, j)]].GetSolutionStepValue(KM.TEMPERATURE)
                   for j in range(ny + 1)])

# Outward normal flux (pointing out of B, i.e., -x direction)
# q_out = -k * grad(u) . n = -k * du/dx * (-1) = k * du/dx
q_out = K_B * (T_near - T_if) / dx

# Export results
export = {
    "field_name": "temperature",
    "n_points": len(T_if),
    "coordinates": [[L_A, float(H * j / ny)] for j in range(ny + 1)],
    "values": [float(v) for v in T_if],
    "normal_fluxes": [float(v) for v in q_out],
}

Path("exports.json").write_text(json.dumps(export, indent=2))

print(f"Kratos slab B: T_if(mean)={T_if.mean():.6f}, q_in(mean)={np.mean(q_in_values):.6f}")
