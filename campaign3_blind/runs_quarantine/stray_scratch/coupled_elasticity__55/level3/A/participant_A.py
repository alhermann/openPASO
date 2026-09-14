#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import *

NX = 32
NY_A = 20
Y_INTERFACE = 0.625
X_MAX = 1.0
Y_A_MAX = 0.625
LAMBDA_A = 480.0
MU_A = 1200.0
LEVEL_IDX = 3

class SourceA(CoefficientFunction):
    def Ldef(self):
        x, y = self.x[0], self.x[1]
        fx = -63*x**4*y/3125 + 99*x**4/5000 + 168*x**3*y/15625 - 33*x**3/3125 
              + 216*x**2*y**2/625 - 1557*x**2*y/3125 - 99*x**2/5000 
              - 288*x*y**2/3125 + 1992*x*y/15625 + 33*x/3125 
              - 36*y**2/625 + 54*y/625
        fy = -108*x**5/15625 + 72*x**4/15625 - 18*x**3*y**2/625 + 153*x**3*y/1250 
              - 279*x**3/3125 + 36*x**2*y**2/3125 - 153*x**2*y/3125 + 486*x**2/15625 
              + 9*x*y**2/625 - 153*x*y/2500 + 63*x/1250 
              - 12*y**2/3125 + 51*y/3125 - 42/3125
        return (fx, fy)

geo = SplineGeometry()
pnts = [(0, 0), (X_MAX, 0), (X_MAX, Y_A_MAX), (0, Y_A_MAX)]
p = [geo.AddPoint(*pnt) for pnt in pnts]
geo.Append(["line", p[0], p[1]], leftdomain=1, rightdomain=0, bc="bottom")
geo.Append(["line", p[1], p[2]], leftdomain=1, rightdomain=0, bc="right")
geo.Append(["line", p[2], p[3]], leftdomain=1, rightdomain=0, bc="interface")
geo.Append(["line", p[3], p[0]], leftdomain=1, rightdomain=0, bc="left")

mesh = Mesh(geo.GenerateMesh(maxh=X_MAX/NX))

fes = VectorH1(mesh, order=1)
u, v = fes.TnT()

# Build node-to-dof mapping
node_to_dofs = {}
for vd in mesh.vertices:
    dofs = fes.GetDofNrs(NodeId(VERTEX, vd.nr))
    node_to_dofs[vd.nr] = dofs

# Get DOFs by boundary name using mesh.Boundaries()
outer_dofs = fes.GetDofs(mesh.Boundaries("bottom|left|right"))
iface_dofs_all = fes.GetDofs(mesh.Boundaries("interface"))

corner_node_nums = []
for vd in mesh.vertices:
    pt = vd.point
    if abs(pt[1] - Y_INTERFACE) < 1e-10:
        if abs(pt[0]) < 1e-10 or abs(pt[0] - X_MAX) < 1e-10:
            corner_node_nums.append(vd.nr)

iface_dofs = []
for dof in iface_dofs_all:
    found = False
    for node_nr, dofs in node_to_dofs.items():
        if dof in dofs:
            if node_nr not in corner_node_nums:
                iface_dofs.append(dof)
            found = True
            break

all_outer_dofs = set(outer_dofs)
for corner_node in corner_node_nums:
    all_outer_dofs.update(node_to_dofs[corner_node])
all_outer_dofs = list(all_outer_dofs)

def Strain(u):
    return 0.5 * (Grad(u) + Grad(u).trans)

def Stress(u):
    return 2 * MU_A * Strain(u) + LAMBDA_A * Trace(Strain(u)) * Id(2)

a = BilinearForm(fes)
a += InnerProduct(Stress(u), Strain(v)) * dx
a.Assemble()

f_vol = LinearForm(fes)
source = SourceA(dim=2)
f_vol += source * v * dx
f_vol.Assemble()

try:
    with open("imports.json", "r") as fp:
        imports = json.load(fp)
    has_imports = True
except:
    has_imports = False
    imports = dict()

iface_dofs_ordered = sorted(iface_dofs)
iface_coords_x = []
iface_node_map = {}
for dof in iface_dofs_ordered:
    for node_nr, dofs in node_to_dofs.items():
        if dof in dofs:
            pt = mesh.vertices[node_nr].point
            iface_coords_x.append(pt[0])
            iface_node_map[dof] = (pt[0], pt[1], node_nr)
            break
iface_coords_x = np.array(iface_coords_x)

rhs_vec = f_vol.vec.dup()

if has_imports and "B" in imports:
    imported_data = imports["B"]
    imported_coords = np.array(imported_data["coordinates"])
    imported_traction = np.array(imported_data["normal_fluxes"])
    
    traction_x_interp = np.interp(iface_coords_x, imported_coords[:, 0], imported_traction[:, 0])
    traction_y_interp = np.interp(iface_coords_x, imported_coords[:, 0], imported_traction[:, 1])
    
    h_x = X_MAX / NX
    traction_weight = h_x / 2
    
    for i, dof in enumerate(iface_dofs_ordered):
        comp = fes.GetNDofLevel(dof) % 2
        if comp == 0:
            rhs_vec[dof] += traction_x_interp[i] * traction_weight
        else:
            rhs_vec[dof] += traction_y_interp[i] * traction_weight

gfu = GridFunction(fes)
mat = a.mat
inv = mat.Inverse(all_outer_dofs, inverse='sparsecholesky')
gfu.vec.data = inv * rhs_vec.vec

ndof = fes.ndof
logname = "run_level" + str(LEVEL_IDX) + "_A.log"
with open(logname, "w") as fp:
    fp.write("NDOF = " + str(ndof) + "\n")
    fp.write("Elements = " + str(mesh.ne) + "\n")
    fp.write("Interface DOFs = " + str(len(iface_dofs)) + "\n")

export_coords = []
export_values = []
export_fluxes = []

for dof in iface_dofs_ordered:
    x, y, node_nr = iface_node_map[dof]
    export_coords.append([x, y])
    
    ux_val = gfu.components[0](mesh(x, y))
    uy_val = gfu.components[1](mesh(x, y))
    export_values.append([ux_val, uy_val])
    
    grad_u = gfu.Grad()(mesh(x, y))
    eps_xx = grad_u[0][0]
    eps_yy = grad_u[1][1]
    eps_xy = 0.5 * (grad_u[0][1] + grad_u[1][0])
    
    sigma_xx = LAMBDA_A * (eps_xx + eps_yy) + 2 * MU_A * eps_xx
    sigma_yy = LAMBDA_A * (eps_xx + eps_yy) + 2 * MU_A * eps_yy
    sigma_xy = 2 * MU_A * eps_xy
    
    tx = -sigma_xy
    ty = -sigma_yy
    export_fluxes.append([tx, ty])

exports = {
    "field_name": "displacement",
    "coordinates": export_coords,
    "values": export_values,
    "normal_fluxes": export_fluxes,
    "n_points": len(export_coords)
}

with open("exports.json", "w") as fp:
    json.dump(exports, fp)

print("Subdomain A complete: " + str(ndof) + " DOFs, " + str(len(export_coords)) + " interface points")
