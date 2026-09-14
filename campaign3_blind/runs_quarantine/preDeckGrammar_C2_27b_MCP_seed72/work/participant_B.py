#!/usr/bin/env python3
"""
Participant B (Kratos): Subdomain (0.625, 1.5) x (0, 1), k = 200
Role: NEUMANN side - receives flux from A, applies as Neumann BC on left edge
      Exports temperature at interface
"""
import json
import os
import sys
from pathlib import Path
import numpy as np

# ── PROBLEM PARAMETERS ─
SIDE      = "neumann"     # "neumann" (import flux, export T)
PARTNER   = "A"           # the partner's name in couple() call
X0, X1    = 0.625, 1.5    # subdomain B's x-extent
Y0, Y1    = 0.0, 1.0      # subdomain B's y-extent
IFACE_X   = 0.625         # interface at x = 0.625
K         = 200.0         # thermal conductivity in B

T_OUTER   = 0.0           # Dirichlet value on outer boundaries
T_INIT    = 0.0           # iteration-1 fallback interface temperature
Q_INIT    = 0.0           # iteration-1 fallback interface flux

# Fixed interface probe points (same as participant A)
INTERFACE_PROBES = [(0.625, (j + 0.5) / 44) for j in range(11, 33)]
INTERFACE_YS = [p[1] for p in INTERFACE_PROBES]

# Get mesh divisions from level.json
level_file = Path("level.json")
if level_file.exists():
    divisions = json.loads(level_file.read_text())["divisions"]
else:
    divisions = 8

NX = int((X1 - X0) * divisions)
NY = int((Y1 - Y0) * divisions)

print(f"[Kratos Participant B] NX={NX}, NY={NY}, divisions={divisions}")

# Source term function
def F_SRC(x, y):
    return (-3*x**3*y/50000 + 7*x**3/100000 - 8967*x**2*y/200000 + 28769*x**2/1200000 
            - 3*x*y**3/50000 + 21*x*y**2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 
            - 2989*y**3/200000 + 28769*y**2/1200000 + 26803463*y/3840000 - 1468421/512000)

# ── READ IMPORTS ─
imp_file = Path("imports.json")
if imp_file.is_file():
    try:
        imports = json.loads(imp_file.read_text())
        imp = imports.get(PARTNER)
    except:
        imp = None
else:
    imp = None

# Sample imported flux at FIXED interface points
if imp and imp.get("coordinates"):
    yy_imp = np.array([c[1] for c in imp["coordinates"]], float)
    vv_imp = np.asarray(imp.get("normal_fluxes", []), float).ravel()
    if len(vv_imp) > 0:
        iface_fluxes = np.interp(INTERFACE_YS, yy_imp, vv_imp)
    else:
        iface_fluxes = np.full(len(INTERFACE_YS), Q_INIT)
else:
    iface_fluxes = np.full(len(INTERFACE_YS), Q_INIT)

# ── GENERATE MESH ─
nodes = []
node_map = {}
nid = 1

for j in range(NY + 1):
    for i in range(NX + 1):
        x = X0 + i * (X1 - X0) / NX
        y = Y0 + j * (Y1 - Y0) / NY
        nodes.append((nid, x, y))
        node_map[(i, j)] = nid
        nid += 1

n_nodes = len(nodes)

# Elements
elements = []
eid = 1
for j in range(NY):
    for i in range(NX):
        n1 = node_map[(i, j)]
        n2 = node_map[(i+1, j)]
        n3 = node_map[(i+1, j+1)]
        n4 = node_map[(i, j+1)]
        elements.append((eid, n1, n2, n4))
        eid += 1
        elements.append((eid, n2, n3, n4))
        eid += 1

n_elements = len(elements)

# ── ASSEMBLE STIFFNESS MATRIX ─
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

K_mat = lil_matrix((n_nodes, n_nodes))

for eid, n1, n2, n4 in elements:
    idx = [n1-1, n2-1, n4-1]
    
    x = np.array([nodes[n-1][1] for n in [n1, n2, n4]])
    y = np.array([nodes[n-1][2] for n in [n1, n2, n4]])
    
    area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
    
    b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
    c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
    
    Ke = (K / (4.0 * area)) * (np.outer(b, b) + np.outer(c, c))
    
    for a in range(3):
        for bb in range(3):
            K_mat[idx[a], idx[bb]] += Ke[a, bb]

K_mat = K_mat.tocsr()

# ── ASSEMBLE LOAD VECTOR ─
F = np.zeros(n_nodes)

for eid, n1, n2, n4 in elements:
    idx = [n1-1, n2-1, n4-1]
    
    x = np.array([nodes[n-1][1] for n in [n1, n2, n4]])
    y = np.array([nodes[n-1][2] for n in [n1, n2, n4]])
    
    area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
    
    xc = (x[0] + x[1] + x[2]) / 3
    yc = (y[0] + y[1] + y[2]) / 3
    
    f_val = F_SRC(xc, yc)
    
    for i in range(3):
        F[idx[i]] += f_val * area / 3

# ── APPLY BOUNDARY CONDITIONS ─
tol = 1e-9
left_nodes = set()   # x = X0 (interface) - Neumann
right_nodes = set()  # x = X1 - Dirichlet u=0
bottom_nodes = set() # y = Y0 - Dirichlet u=0
top_nodes = set()    # y = Y1 - Dirichlet u=0

for nid, x, y in nodes:
    if abs(x - X0) < tol:
        left_nodes.add(nid-1)
    if abs(x - X1) < tol:
        right_nodes.add(nid-1)
    if abs(y - Y0) < tol:
        bottom_nodes.add(nid-1)
    if abs(y - Y1) < tol:
        top_nodes.add(nid-1)

# Apply Neumann BC on left (interface)
dy = (Y1 - Y0) / NY
for nid_idx in sorted(left_nodes):
    _, x, y = nodes[nid_idx + 1]
    q_in = np.interp(y, INTERFACE_YS, iface_fluxes)
    F[nid_idx] += q_in * dy

# Dirichlet nodes
dirichlet_nodes = right_nodes | bottom_nodes | top_nodes
all_nodes = set(range(n_nodes))
interior_nodes = all_nodes - dirichlet_nodes - left_nodes

# Solve
if len(interior_nodes) > 0:
    interior_list = sorted(interior_nodes)
    
    K_int = K_mat[np.ix_(interior_list, interior_list)]
    F_red = F[interior_list]
    
    u_interior = spsolve(K_int, F_red)
    
    u = np.zeros(n_nodes)
    u[interior_list] = u_interior
else:
    u = np.zeros(n_nodes)

# ── COMPUTE INTERFACE TEMPERATURE AT FIXED POINTS ─
iface_temps = np.zeros(len(INTERFACE_PROBES))

for i, (xi, yi) in enumerate(INTERFACE_PROBES):
    # Find node closest to this point
    min_dist = float('inf')
    best_u = 0.0
    for nid, x, y in nodes:
        dist = (x - xi)**2 + (y - yi)**2
        if dist < min_dist:
            min_dist = dist
            best_u = u[nid-1]
    iface_temps[i] = best_u

# ── WRITE EXPORTS ─
exports = {
    "field_name": "temperature",
    "n_points": len(INTERFACE_PROBES),
    "coordinates": [[float(p[0]), float(p[1])] for p in INTERFACE_PROBES],
    "values": [float(t) for t in iface_temps],
    "normal_fluxes": [float(q) for q in iface_fluxes]
}

(Path("exports.json")).write_text(json.dumps(exports, indent=2))

coords = np.array([[nodes[n-1][1], nodes[n-1][2]] for n in range(1, n_nodes+1)])
np.savez(Path("solution_B.npz"), 
         coords=coords, values=u, 
         nodes=np.array(nodes), iface_y=np.array(INTERFACE_YS), iface_T=iface_temps)

with open(Path("run.log"), "w") as f:
    f.write(f"[Kratos Participant B] Manual assembly solve completed\n")
    f.write(f"[Kratos Participant B] NDOF = {n_nodes}\n")
    f.write(f"[Kratos Participant B] Elements = {n_elements}\n")
    f.write(f"[Kratos Participant B] Temperature range: [{u.min():.6f}, {u.max():.6f}]\n")

print(f"[Kratos Participant B] Exported {len(INTERFACE_PROBES)} interface points")
print(f"[Kratos Participant B] Temperature range: [{u.min():.6f}, {u.max():.6f}]")
