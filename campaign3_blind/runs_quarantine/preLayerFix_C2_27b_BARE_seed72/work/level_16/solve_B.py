#!/usr/bin/env python3
from ngsolve import *
import netgen.csg as csg
import json
import math

INTERFACE_X = 0.625
L_B = 0.875
H = 1.0
K_B = 200.0
LEVEL = 16
WORK_DIR = "level_16"

# Create geometry and mesh
geo = csg.CSGeometry()
box = csg.OrthoBrick(csg.Pnt(INTERFACE_X, 0, 0), csg.Pnt(1.5, H, 0))
geo.Add(box)
ngmesh = geo.GenerateMesh(maxh=L_B/LEVEL)
mesh = Mesh(ngmesh)

# Create finite element space
fes = H1(mesh, order=1)

# Define bilinear and linear forms
u, v = fes.TnT()
a = BilinearForm(fes)
a += K_B * grad(u) * grad(v) * dx

# Source term using symbolic coefficient function
f_func = SymbolicCoefficientFunction("-3*x[0]**3*x[1]/50000 + 7*x[0]**3/100000 - 8967*x[0]**2*x[1]/200000 + 28769*x[0]**2/1200000 - 3*x[0]*x[1]**3/50000 + 21*x[0]*x[1]**2/100000 - 2938983*x[0]*x[1]/640000 + 7203409*x[0]/3840000 - 2989*x[1]**3/200000 + 28769*x[1]**2/1200000 + 26803463*x[1]/3840000 - 1468421/512000", fes.VSpace().tdim)
f_form = LinearForm(fes)
f_form += f_func * v * dx

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

# Output solution values at probe points
probe_points_B = []
for i_y in range(44):
    for i_x in range(44):
        px = INTERFACE_X + (i_x + 0.5) * L_B / 44
        py = 0 + (i_y + 0.5) * H / 44
        probe_points_B.append((px, py))

results = []
for px, py in probe_points_B:
    results.append({"x": px, "y": py, "u": float(fes(Point(px, py)))})

with open(WORK_DIR + "/solution_B.json", "w") as f:
    json.dump(results, f)

# Interface points
interface_points = []
for j in range(11, 33):
    px = INTERFACE_X
    py = (j + 0.5) / 44
    interface_points.append((px, py))

interface_results = []
for px, py in interface_points:
    u_val = float(fes(Point(px, py)))
    grad_u = fes.Diff(Point(px, py))
    qn = -K_B * (-grad_u.x)  # Outward normal from B is (-1, 0)
    interface_results.append({"x": px, "y": py, "u": u_val, "qn": qn})

with open(WORK_DIR + "/interface_B.json", "w") as f:
    json.dump(interface_results, f)

print("NGSolve solved subdomain B: " + str(len(fes)) + " DOFs")
print("NDOF = " + str(len(fes)))
