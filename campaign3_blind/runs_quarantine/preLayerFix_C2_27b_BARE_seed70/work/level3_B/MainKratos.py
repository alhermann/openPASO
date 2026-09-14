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
nx = 28
ny = 32

# Incoming flux from partner (interpolated to our interface nodes)
q_in_values = [np.float64(-0.001343763311175584), np.float64(-0.0004944898175773846), np.float64(-0.0005860110140872448), np.float64(-0.0007492029461509918), np.float64(-0.000900778704279786), np.float64(-0.0010315316171245863), np.float64(-0.0011433247976389994), np.float64(-0.0012388039180360137), np.float64(-0.0013201315652256358), np.float64(-0.0013889628067657953), np.float64(-0.0014465625336268165), np.float64(-0.0014939027742662814), np.float64(-0.00153172980249296), np.float64(-0.0015606091142162225), np.float64(-0.0015809555004449877), np.float64(-0.0015930529378104033), np.float64(-0.0015970671782519381), np.float64(-0.0015930527212233182), np.float64(-0.00158095504623142), np.float64(-0.0015606083779651347), np.float64(-0.0015317287113180266), np.float64(-0.0014939012180650371), np.float64(-0.0014465603515539834), np.float64(-0.0013889597688061285), np.float64(-0.0013201273564586967), np.float64(-0.0012387981731206699), np.float64(-0.0011433174319582708), np.float64(-0.0010315247100643934), np.float64(-0.0009007866654609257), np.float64(-0.0007493076328650367), np.float64(-0.0005866937381778336), np.float64(-0.0004985419431111082), np.float64(-0.001375479139359328)]

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
