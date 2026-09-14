#!/usr/bin/env python3
"""Solve subdomain A using scikit-fem"""
import numpy as np
import skfem
from skfem import *
from skfem.models.poisson import laplace, mass
import json
import sys

# Parameters
INTERFACE_X = 5.0 / 8.0
L_A = 0.625
H = 1.0
K_A = 1.0
LEVEL = int(sys.argv[1]) if len(sys.argv) > 1 else 8
WORK_DIR = sys.argv[2] if len(sys.argv) > 2 else "."

# Create mesh
m = skfem.MeshTri.init_rectangle([0, L_A], [0, H], refinements=int(np.log2(LEVEL)))
print(f"Created mesh with {m.p.shape[1]} vertices")

# Create basis
e = ElementTriP1()
b = Basis(m, e, intelements=True)

# Define source term
def f_A(x):
    x0, x1 = x[0], x[1]
    return (-12*x0**3*x1/5 + 14*x0**3/5 - 8979*x0**2*x1/2000 - 5147*x0**2/12000 
            - 12*x0*x1**3/5 + 42*x0*x1**2/5 - 1779*x0*x1/800 - 1007*x0/1200 
            - 2993*x1**3/2000 - 5147*x1**2/12000 + 4621*x1/2400)

# Assemble system
A = asm(K_A * laplace, b)
F = asm(lambda x: f_A(x), b)

# Apply Dirichlet BC on all boundaries
D = m.boundary_nodes()
A, F = cond(A, F, D)

# Solve
x = solve(A, F)

print(f"Solved with {len(x)} DOFs")

# Interpolate solution
u = b.interpolate(x)

# Output solution values at probe points
results = []
for i_y in range(44):
    for i_x in range(44):
        px = 0 + (i_x + 0.5) * L_A / 44
        py = 0 + (i_y + 0.5) * H / 44
        results.append({"x": px, "y": py, "u": float(u(px, py))})

with open(WORK_DIR + "/solution_A.json", "w") as f:
    json.dump(results, f)

# Interface points - compute flux
interface_results = []
for j in range(11, 33):
    px = INTERFACE_X
    py = (j + 0.5) / 44
    
    # Get value at interface
    u_val = float(u(px, py))
    
    # Compute gradient (approximate)
    h = 1e-6
    u_plus = u(px + h, py)
    grad_u_x = (u_plus - u_val) / h
    
    qn = -K_A * grad_u_x  # Outward normal from A is (+1, 0)
    interface_results.append({"x": px, "y": py, "u": u_val, "qn": qn})

with open(WORK_DIR + "/interface_A.json", "w") as f:
    json.dump(interface_results, f)

print("scikit-fem solved subdomain A successfully")
print(f"NDOF = {len(x)}")
