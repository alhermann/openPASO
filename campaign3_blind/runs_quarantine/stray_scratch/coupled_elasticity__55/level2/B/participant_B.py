#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from skfem import *
from skfem.models.elasticity import linear_elasticity

NX = 16
NY_B = 14
Y_INTERFACE = 0.625
Y_B_MIN = 0.625
Y_B_MAX = 1.5
X_MAX = 1.0
LAMBDA_B = 480.0
MU_B = 240.0
LEVEL_IDX = 2

def fx_B(x, y):
    return (1737*x**4*y/30625 - 9357*x**4/245000 - 4632*x**3*y/153125 + 3119*x**3/153125 
            + 396*x**2*y**2/875 - 57969*x**2*y/61250 + 86847*x**2/245000 
            - 528*x*y**2/4375 + 40962*x*y/153125 - 16034*x/153125 
            - 66*y**2/875 + 519*y/3500 - 369/7000)

def fy_B(x, y):
    return (2316*x**5/153125 - 1544*x**4/153125 + 1158*x**3*y**2/30625 + 9201*x**3*y/61250 
            - 53561*x**3/245000 - 2316*x**2*y**2/153125 - 9201*x**2*y/153125 + 59737*x**2/612500 
            - 579*x*y**2/30625 - 9201*x*y/122500 + 9477*x/98000 
            + 772*y**2/153125 + 3067*y/153125 - 3159/122500)

xs = np.linspace(0, X_MAX, NX + 1)
ys = np.linspace(Y_B_MIN, Y_B_MAX, NY_B + 1)
m = MeshQuad.init_tensor(xs, ys)

tol = 1e-10
m = m.with_boundaries({
    "bottom": lambda x: np.abs(x[1] - Y_B_MIN) < tol,
    "top": lambda x: np.abs(x[1] - Y_B_MAX) < tol,
    "left": lambda x: x[0] < tol,
    "right": lambda x: np.abs(x[0] - X_MAX) < tol,
})

e = ElementVector(ElementQuad1())
ib = Basis(m, e)

K = linear_elasticity(LAMBDA_B, MU_B).assemble(ib)

@LinearForm
def body_force(v, w):
    x, y = w['x'][0], w['x'][1]
    return fx_B(x, y) * v[0] + fy_B(x, y) * v[1]

f_vol = body_force.assemble(ib)

outer_dofs = ib.get_dofs(("top", "left", "right")).flatten()
iface_dofs_all = ib.get_dofs("bottom").flatten()

corner_nodes = []
for node in range(m.p.shape[1]):
    x, y = m.p[0, node], m.p[1, node]
    if abs(y - Y_B_MIN) < tol:
        if abs(x) < tol or abs(x - X_MAX) < tol:
            corner_nodes.append(node)

iface_dofs = []
for dof in iface_dofs_all:
    for comp in range(2):
        if dof in ib.nodal_dofs[comp, :]:
            node = np.where(ib.nodal_dofs[comp, :] == dof)[0][0]
            if node not in corner_nodes:
                if dof not in iface_dofs:
                    iface_dofs.append(dof)
            break

iface_dofs = sorted(iface_dofs)

iface_coords_x = []
iface_nodes = []
for dof in iface_dofs:
    for comp in range(2):
        if dof in ib.nodal_dofs[comp, :]:
            node = np.where(ib.nodal_dofs[comp, :] == dof)[0][0]
            iface_coords_x.append(m.p[0, node])
            iface_nodes.append(node)
            break
iface_coords_x = np.array(iface_coords_x)
iface_nodes = np.array(iface_nodes)

try:
    with open("imports.json", "r") as fp:
        imports = json.load(fp)
    has_imports = True
except:
    has_imports = False
    imports = dict()

u_D = np.zeros(ib.N)
dirichlet_dofs = outer_dofs.copy()

if has_imports and "A" in imports:
    imported_data = imports["A"]
    imported_coords = np.array(imported_data["coordinates"])
    imported_disp = np.array(imported_data["values"])
    
    disp_x_interp = np.interp(iface_coords_x, imported_coords[:, 0], imported_disp[:, 0])
    disp_y_interp = np.interp(iface_coords_x, imported_coords[:, 0], imported_disp[:, 1])
    
    dirichlet_dofs = np.concatenate([outer_dofs, iface_dofs])
    
    for i, dof in enumerate(iface_dofs):
        comp = dof % 2
        if comp == 0:
            u_D[dof] = disp_x_interp[i]
        else:
            u_D[dof] = disp_y_interp[i]

A_reduced, b_reduced, _, _ = condense(K, f_vol, D=dirichlet_dofs, xD=u_D)
u_reduced = solve(A_reduced, b_reduced)
u_full = np.zeros(ib.N)
free_dofs = np.setdiff1d(range(ib.N), dirichlet_dofs)
u_full[free_dofs] = u_reduced
u_full[dirichlet_dofs] = u_D[dirichlet_dofs]

ndof = K.shape[0]
logname = "run_level" + str(LEVEL_IDX) + "_B.log"
with open(logname, "w") as fp:
    fp.write("NDOF = " + str(ndof) + "\n")
    fp.write("Elements = " + str(m.ne) + "\n")
    fp.write("Interface DOFs = " + str(len(iface_dofs)) + "\n")

export_coords = []
export_values = []
export_fluxes = []

residual = K @ u_full - f_vol

for i, dof in enumerate(iface_dofs):
    for comp in range(2):
        if dof in ib.nodal_dofs[comp, :]:
            node = np.where(ib.nodal_dofs[comp, :] == dof)[0][0]
            x, y = m.p[0, node], m.p[1, node]
            break
    
    export_coords.append([x, y])
    
    ux_val = u_full[ib.nodal_dofs[0, node]]
    uy_val = u_full[ib.nodal_dofs[1, node]]
    export_values.append([ux_val, uy_val])
    
    found_elem = None
    for el_idx in range(m.t.shape[1]):
        el_nodes = m.t[:, el_idx]
        if node in el_nodes:
            if el_nodes[0] == node or el_nodes[1] == node:
                found_elem = el_idx
                break
    
    if found_elem is not None:
        el_nodes = m.t[:, found_elem]
        el_coords = m.p[:, el_nodes]
        el_u = u_full[ib.nodal_dofs[:, el_nodes]].reshape(2, 4).T
        
        xi_nodes = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]])
        dN_dxi = 0.25 * xi_nodes
        J = el_coords.T @ dN_dxi
        invJ = np.linalg.inv(J)
        
        du_dxi = el_u @ dN_dxi
        du_dx = du_dxi @ invJ
        
        eps_xx = du_dx[0, 0]
        eps_yy = du_dx[1, 1]
        eps_xy = 0.5 * (du_dx[0, 1] + du_dx[1, 0])
        
        sigma_xx = LAMBDA_B * (eps_xx + eps_yy) + 2 * MU_B * eps_xx
        sigma_yy = LAMBDA_B * (eps_xx + eps_yy) + 2 * MU_B * eps_yy
        sigma_xy = 2 * MU_B * eps_xy
        
        tx = sigma_xy
        ty = sigma_yy
    else:
        tx = 0.0
        ty = 0.0
    
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

print("Subdomain B complete: " + str(ndof) + " DOFs, " + str(len(export_coords)) + " interface points")
