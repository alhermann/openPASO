#!/usr/bin/env python3
import os, json, numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

L_A, H, k_A = 0.625, 1.0, 1.0
nx, ny = 5, 8
dx, dy = L_A/nx, H/ny
n_nodes_x, n_nodes = nx+1, (nx+1)*(ny+1)

def node_idx(i,j): return j*n_nodes_x + i

def source_A(x,y):
    return (-12*x**3*y/5 + 14*x**3/5 - 8979*x**2*y/2000 - 5147*x**2/12000 
            - 12*x*y**3/5 + 42*x*y**2/5 - 1779*x*y/800 - 1007*x/1200 
            - 2993*y**3/2000 - 5147*y**2/12000 + 4621*y/2400)

# Build mesh and assemble
elements = []
for j in range(ny):
    for i in range(nx):
        elements.append([node_idx(i,j), node_idx(i+1,j), node_idx(i,j+1)])
        elements.append([node_idx(i+1,j), node_idx(i+1,j+1), node_idx(i,j+1)])

K = lil_matrix((n_nodes, n_nodes))
F = np.zeros(n_nodes)

for elem in elements:
    coords = np.array([[nid%n_nodes_x*dx, nid//n_nodes_x*dy] for nid in elem])
    x1,y1,x2,y2,x3,y3 = coords[0,0],coords[0,1],coords[1,0],coords[1,1],coords[2,0],coords[2,1]
    area = 0.5*abs((x2-x1)*(y3-y1)-(x3-x1)*(y2-y1))
    if area < 1e-15: continue
    b = np.array([y2-y3, y3-y1, y1-y2])
    c = np.array([x3-x2, x1-x3, x2-x1])
    Ke = (k_A/(4*area))*(np.outer(b,b)+np.outer(c,c))
    xc,yc = coords[:,0].mean(), coords[:,1].mean()
    Fe = source_A(xc,yc)*area/3*np.ones(3)
    for a in range(3):
        for bb in range(3): K[elem[a],elem[bb]] += Ke[a,bb]
        F[elem[a]] += Fe[a]

K_csr = K.tocsr()

# Read imports
imports_file = os.path.join(os.getcwd(), 'imports.json')
interface_temp = None
if os.path.exists(imports_file):
    with open(imports_file) as f:
        imports = json.load(f)
    if 'B' in imports:
        interface_temp = np.array(imports['B']['values'])

# Dirichlet BCs
dirichlet_mask = np.zeros(n_nodes, dtype=bool)
dirichlet_values = np.zeros(n_nodes)
for i in range(nx+1):
    dirichlet_mask[node_idx(i,0)] = True
    dirichlet_mask[node_idx(i,ny)] = True
for j in range(ny+1):
    dirichlet_mask[node_idx(0,j)] = True
    nid = node_idx(nx,j)
    dirichlet_mask[nid] = True
    if interface_temp is not None and j < len(interface_temp):
        dirichlet_values[nid] = interface_temp[j]

interior = np.where(~dirichlet_mask)[0]
dirichlet = np.where(dirichlet_mask)[0]

if len(interior) > 0:
    K_ii = K_csr[interior[:,None], interior].tocsc()
    F_int = F[interior] - K_csr[interior[:,None], dirichlet].tocsc() @ dirichlet_values[dirichlet]
    u_int = spsolve(K_ii, F_int)
    u = np.zeros(n_nodes)
    u[interior] = u_int
    u[dirichlet] = dirichlet_values[dirichlet]
else:
    u = dirichlet_values.copy()

# Extract interface data
temps = np.array([u[j*n_nodes_x+nx] for j in range(ny+1)])
fluxes = np.zeros(ny+1)
for j in range(ny+1):
    du_dx = (u[j*n_nodes_x+nx] - u[j*n_nodes_x+nx-1])/dx
    fluxes[j] = -k_A * du_dx

exports = {
    "field_name": "temperature",
    "coordinates": [[L_A, j*dy] for j in range(ny+1)],
    "values": temps.tolist(),
    "normal_fluxes": fluxes.tolist(),
    "n_points": ny+1
}

with open(os.path.join(os.getcwd(), 'exports.json'), 'w') as f:
    json.dump(exports, f)

print(f"Participant A: {ny+1} interface points")
