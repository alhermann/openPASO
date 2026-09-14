#!/usr/bin/env python3
from ngsolve import *
import netgen.csg as csg
import json
import math

INTERFACE_X = 0.625
L_A = 0.625
H = 1.0
K_A = 1.0
LEVEL = 16
WORK_DIR = "level_16"

# Create geometry and mesh
geo = csg.CSGeometry()
box = csg.OrthoBrick(csg.Pnt(0, 0, 0), csg.Pnt(L_A, H, 0))
geo.Add(box)
ngmesh = geo.GenerateMesh(maxh=L_A/LEVEL)
mesh = Mesh(ngmesh)

# Create finite element space
fes = H1(mesh, order=1)

# Define bilinear and linear forms
u, v = fes.TnT()
a = BilinearForm(fes)
a += K_A * grad(u) * grad(v) * dx

# Source term using symbolic coefficient function
f_func = SymbolicCoefficientFunction("-12*x[0]**3*x[1]/5 + 14*x[0]**3/5 - 8979*x[0]**2*x[1]/2000 - 5147*x[0]**2/12000 - 12*x[0]*x[1]**3/5 + 42*x[0]*x[1]**2/5 - 1779*x[0]*x[1]/800 - 1007*x[0]/1200 - 2993*x[1]**3/2000 - 5147*x[1]**2/12000 + 4621*x[1]/2400", fes.VSpace().tdim)
f_form = LinearForm(fes)
f_form += f_func * v * dx

# Boundary conditions - all outer boundaries are Dirichlet u=0
SetBC(fes, dirichlet="bottom|top|left|right")

# Solve
c = fes.UpdateMatrix(a)
b = fes.UpdateVector(f_form)
x = fes.vec()

prec = Preconditioner(c, type="jacobi")
krylov = KrylovSolver(c, "cg", prec)
krylov.Mult(b, x)

fes.SetNodalValues(x)

# Output solution values at probe points
probe_points_A = []
for i_y in range(44):
    for i_x in range(44):
        px = 0 + (i_x + 0.5) * L_A / 44
        py = 0 + (i_y + 0.5) * H / 44
        probe_points_A.append((px, py))

results = []
for px, py in probe_points_A:
    results.append({"x": px, "y": py, "u": float(fes(Point(px, py)))})

with open(WORK_DIR + "/solution_A.json", "w") as f:
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
    qn = -K_A * grad_u.x  # Outward normal from A is (+1, 0)
    interface_results.append({"x": px, "y": py, "u": u_val, "qn": qn})

with open(WORK_DIR + "/interface_A.json", "w") as f:
    json.dump(interface_results, f)

print("NGSolve solved subdomain A: " + str(len(fes)) + " DOFs")
print("NDOF = " + str(len(fes)))
