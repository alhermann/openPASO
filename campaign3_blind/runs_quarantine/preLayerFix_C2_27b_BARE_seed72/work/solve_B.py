#!/usr/bin/env python3
"""Solve subdomain B using NGSolve"""
from ngsolve import *
import netgen.csg as csg
import json
import math
import sys

# Parameters
INTERFACE_X = 5.0 / 8.0
L_B = 0.875
H = 1.0
K_B = 200.0
LEVEL = int(sys.argv[1]) if len(sys.argv) > 1 else 8
WORK_DIR = sys.argv[2] if len(sys.argv) > 2 else "."

# Create geometry and mesh
geo = csg.CSGeometry()
box = csg.OrthoBrick(csg.Pnt(INTERFACE_X, 0, 0), csg.Pnt(1.5, H, 0))
geo.Add(box)
ngmesh = geo.GenerateMesh(maxh=L_B/LEVEL)
mesh = Mesh(ngmesh)

print(f"Created mesh with {len(mesh.vertices)} vertices")

# Create finite element space
fes = H1(mesh, order=1)

# Define bilinear and linear forms
u, v = fes.TnT()
a = BilinearForm(fes)
a += K_B * grad(u) * grad(v) * dx

# Source term using lambda function
def f_B(x):
    x0, x1 = x[0], x[1]
    return (-3*x0**3*x1/50000 + 7*x0**3/100000 - 8967*x0**2*x1/200000 + 28769*x0**2/1200000 
            - 3*x0*x1**3/50000 + 21*x0*x1**2/100000 - 2938983*x0*x1/640000 + 7203409*x0/3840000 
            - 2989*x1**3/200000 + 28769*x1**2/1200000 + 26803463*x1/3840000 - 1468421/512000)

f_form = LinearForm(fes)
f_form += f_B * v * dx

# Boundary conditions - outer boundaries except left (interface) are Dirichlet
SetBC(fes, dirichlet="bottom|top|right")
# Left boundary has natural (Neumann) BC

# Solve
c = fes.UpdateMatrix(a)
b = fes.UpdateVector(f_form)
x = fes.vec()

prec = Preconditioner(c, type="jacobi")
krylov = KrylovSolver(c, "cg", prec)
krylov.Mult(b, x)

fes.SetNodalValues(x)

print(f"Solved with {len(fes)} DOFs")

# Output solution values at probe points
results = []
for i_y in range(44):
    for i_x in range(44):
        px = INTERFACE_X + (i_x + 0.5) * L_B / 44
        py = 0 + (i_y + 0.5) * H / 44
        results.append({"x": px, "y": py, "u": float(fes(Point(px, py)))})

with open(WORK_DIR + "/solution_B.json", "w") as f:
    json.dump(results, f)

# Interface points
interface_results = []
for j in range(11, 33):
    px = INTERFACE_X
    py = (j + 0.5) / 44
    u_val = float(fes(Point(px, py)))
    grad_u = fes.Diff(Point(px, py))
    qn = -K_B * (-grad_u.x)  # Outward normal from B is (-1, 0)
    interface_results.append({"x": px, "y": py, "u": u_val, "qn": qn})

with open(WORK_DIR + "/interface_B.json", "w") as f:
    json.dump(interface_results, f)

print("NGSolve solved subdomain B successfully")
print(f"NDOF = {len(fes)}")
