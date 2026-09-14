#!/usr/bin/env python3
"""Kratos participant for OASiS couple driver - Subdomain B (NEUMANN side)

Subdomain B: x in [0.625, 1.5], y in [0, 1]
k = 200
Interface at x = 0.625 (left edge of this subdomain)
Outer BC: u = 0 on right edge (x=1.5), top/bottom edges
Interface BC: Neumann from partner's normal_fluxes (heat flux)
Export: temperature at interface

Uses manual assembly with scipy.sparse for heat conduction.

SIGN CONVENTION:
- Partner A exports qn^A = -K_A * grad(u)^A . n^A where n^A points OUT of A (+x direction)
- At the interface, n^B = -n^A (points OUT of B, i.e., -x direction)
- The weak form has: int_Omega K*grad(u)*grad(v) dx = int_Omega f*v dx + int_Gamma (K*grad(u).n)*v ds
- For domain B, the interface term is: int (K*grad(u).n^B)*v ds = int (-K*grad(u).n^A)*v ds
- Since A exports qn^A = -K_A*grad(u).n^A, we have K*grad(u).n^B = -qn^A
- So the Neumann contribution to RHS is: int (-qn^A)*v ds = -int qn^A*v ds
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Problem parameters
X0, X1 = 0.625, 1.5  # subdomain B extent
Y0, Y1 = 0.0, 1.0
IFACE_X = 0.625      # interface location (left edge)
OUTER_X = 1.5        # outer boundary (right edge)
K = 200.0            # thermal conductivity
PARTNER = "A"        # partner name in couple() call

# Mesh resolution - read from environment or use default
NX = int(os.environ.get('NX_B', '7'))
NY = int(os.environ.get('NY_B', '8'))

# Fallback values for iteration 1
Q_INIT = 0.0


def f_source(x, y):
    """Source term for subdomain B"""
    return (-3*x**3*y/50000 + 7*x**3/100000 - 8967*x**2*y/200000 + 28769*x**2/1200000 
            - 3*x*y**3/50000 + 21*x*y**2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 
            - 2989*y**3/200000 + 28769*y**2/1200000 + 26803463*y/3840000 - 1468421/512000)


def read_imports():
    """Read imports.json written by the coupling driver"""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
        return data.get(PARTNER)
    except (json.JSONDecodeError, KeyError):
        return None


def interpolate_imported_data(imp, ys, key, fallback):
    """Interpolate imported data to our interface nodes"""
    if imp is None or "coordinates" not in imp:
        return np.full(len(ys), float(fallback))
    
    coords = np.array(imp["coordinates"])
    vals = np.array(imp.get(key, [])).flatten()
    
    if len(vals) != len(coords):
        return np.full(len(ys), float(fallback))
    
    src_y = coords[:, 1]
    o = np.argsort(src_y)
    return np.interp(ys, src_y[o], vals[o])


def assemble_and_solve(nx, ny, imported_fluxes):
    """Assemble stiffness matrix and solve using manual assembly"""
    
    dx = (X1 - X0) / nx
    dy = (Y1 - Y0) / ny
    
    # Create node map and coordinates
    nid = 1
    node_map = {}
    coords = {}
    
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = X0 + i * dx
            y = Y0 + j * dy
            coords[nid] = (x, y)
            node_map[(i, j)] = nid
            nid += 1
    
    n_nodes = nid - 1
    
    # Create triangular elements (split each quad into 2 triangles)
    # Use counter-clockwise ordering for positive area
    elements = []
    for j in range(ny):
        for i in range(nx):
            n1 = node_map[(i, j)]      # bottom-left
            n2 = node_map[(i+1, j)]    # bottom-right
            n3 = node_map[(i+1, j+1)]  # top-right
            n4 = node_map[(i, j+1)]    # top-left
            elements.append((n1, n2, n4))  # Lower triangle (CCW)
            elements.append((n2, n3, n4))  # Upper triangle (CCW)
    
    # Assemble stiffness matrix K and load vector F
    # For -div(k grad u) = f, weak form is: int k*grad(u)*grad(v) = int f*v + boundary terms
    K_mat = lil_matrix((n_nodes, n_nodes))
    F_vec = np.zeros(n_nodes)
    
    for tri in elements:
        ids = [t - 1 for t in tri]  # Convert to 0-indexed
        x = np.array([coords[t][0] for t in tri])
        y = np.array([coords[t][1] for t in tri])
        
        # Triangle area (should be positive for CCW ordering)
        area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
        
        if area < 1e-15:
            continue
        
        # Gradient of shape functions (constant for linear triangle)
        b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
        c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
        
        # Element stiffness matrix: K_e = k * (b*b^T + c*c^T) / (4*A)
        Ke = (K / (4.0 * area)) * (np.outer(b, b) + np.outer(c, c))
        
        # Add to global matrix
        for a in range(3):
            for bb in range(3):
                K_mat[ids[a], ids[bb]] += Ke[a, bb]
        
        # Load vector: integrate f over element using centroid rule
        xc = (x[0] + x[1] + x[2]) / 3.0
        yc = (y[0] + y[1] + y[2]) / 3.0
        f_val = f_source(xc, yc)
        
        # Each node gets area/3 contribution
        for a in range(3):
            F_vec[ids[a]] += f_val * area / 3.0
    
    K_mat = K_mat.tocsr()
    
    # Identify Dirichlet nodes (outer boundary, NOT interface)
    dirichlet_nodes = set()
    
    # Right edge (x = OUTER_X) - Dirichlet u=0
    for j in range(ny + 1):
        dirichlet_nodes.add(node_map[(nx, j)] - 1)
    
    # Top edge (y = Y1) - Dirichlet u=0 (excluding corners already added)
    for i in range(1, nx):
        dirichlet_nodes.add(node_map[(i, ny)] - 1)
    
    # Bottom edge (y = Y0) - Dirichlet u=0 (excluding corners already added)
    for i in range(1, nx):
        dirichlet_nodes.add(node_map[(i, 0)] - 1)
    
    # Interface nodes (left edge, x = IFACE_X) are NOT Dirichlet - they have Neumann BC
    
    # Apply Neumann BC on interface: add flux contribution to load vector
    # SIGN: Partner A exports qn^A = -K_A*grad(u).n^A (n^A points +x, INTO domain B)
    # Domain B's outward normal n^B points -x (OUT of B)
    # Weak form: int_Gamma (K*grad(u).n^B)*v ds = int_Gamma (-qn^A)*v ds
    # So we ADD: -int qn^A * v ds to the RHS
    iface_y = np.linspace(Y0, Y1, ny + 1)
    
    for j in range(ny + 1):
        node_id = node_map[(0, j)] - 1  # 0-indexed
        flux_val = imported_fluxes[j]
        
        # Weight for trapezoidal integration
        if j == 0 or j == ny:
            weight = dy / 2
        else:
            weight = dy
        
        # NEGATIVE sign because n^B = -n^A
        F_vec[node_id] -= flux_val * weight
    
    # Solve with Dirichlet BCs
    interior_nodes = sorted(set(range(n_nodes)) - dirichlet_nodes)
    
    if len(interior_nodes) == 0:
        raise RuntimeError("No interior nodes - check mesh and BCs")
    
    # Extract submatrix for interior nodes
    K_int = K_mat[np.ix_(interior_nodes, interior_nodes)]
    
    # Compute RHS for interior nodes
    u_dirichlet = np.zeros(n_nodes)  # All Dirichlet values are 0
    F_int = F_vec[interior_nodes] - K_mat[np.ix_(interior_nodes, list(dirichlet_nodes))] @ u_dirichlet
    
    # Check for NaN/Inf
    if np.any(np.isnan(F_int)) or np.any(np.isinf(F_int)):
        raise RuntimeError("NaN or Inf in load vector")
    
    # Solve
    try:
        u_int = spsolve(K_int, F_int)
    except Exception as e:
        raise RuntimeError(f"Solve failed: {e}")
    
    # Check solution
    if np.any(np.isnan(u_int)) or np.any(np.isinf(u_int)):
        raise RuntimeError("NaN or Inf in solution")
    
    # Reconstruct full solution
    u = np.zeros(n_nodes)
    for idx, node_idx in enumerate(interior_nodes):
        u[node_idx] = u_int[idx]
    
    return u, node_map, coords


def main():
    print("Kratos Subdomain B (Neumann side) starting...", flush=True)
    
    # Read imported flux data
    imp = read_imports()
    
    # Get interface y-coordinates
    iface_y = np.linspace(Y0, Y1, NY + 1)
    
    # Interpolate imported flux to our interface nodes
    imported_fluxes = interpolate_imported_data(imp, iface_y, "normal_fluxes", Q_INIT)
    
    # Assemble and solve
    u, node_map, coords = assemble_and_solve(NX, NY, imported_fluxes)
    
    # Extract interface temperatures (left edge, i=0)
    T_if = []
    for j in range(NY + 1):
        node_id = node_map[(0, j)] - 1  # 0-indexed
        T_if.append(u[node_id])
    T_if = np.array(T_if)
    
    # Check solution quality
    if np.any(np.isnan(T_if)) or np.any(np.isinf(T_if)):
        print("WARNING: NaN or Inf in interface temperatures", file=sys.stderr)
    
    # Write exports.json
    exports = {
        "field_name": "temperature",
        "n_points": len(iface_y),
        "coordinates": [[float(IFACE_X), float(y)] for y in iface_y],
        "values": [float(t) for t in T_if]
        # Note: normal_fluxes is optional for Neumann side
    }
    
    Path("exports.json").write_text(json.dumps(exports, indent=2))
    
    print(f"[Kratos Subdomain B] Interface: n={len(iface_y)}, T=[{T_if.min():.6g},{T_if.max():.6g}]", flush=True)
    
    # Print NDOF for log file
    n_dof = (NX + 1) * (NY + 1)
    print(f"NDOF = {n_dof}", flush=True)
    
    # Also write solution to VTU for post-processing
    try:
        import meshio
        
        pts = np.array([[coords[i+1][0], coords[i+1][1], 0.0] for i in range(n_nodes)])
        
        elements_list = []
        for j in range(NY):
            for i in range(NX):
                n1 = node_map[(i, j)]
                n2 = node_map[(i+1, j)]
                n3 = node_map[(i+1, j+1)]
                n4 = node_map[(i, j+1)]
                elements_list.append([n1-1, n2-1, n4-1])
                elements_list.append([n1-1, n3-1, n4-1])
        
        cells = np.array(elements_list)
        mesh = meshio.Mesh(pts, [("triangle", cells)], point_data={"temperature": u})
        mesh.write("result.vtu")
        
        print("[Kratos Subdomain B] Wrote result.vtu", flush=True)
    except Exception as e:
        print(f"[Kratos Subdomain B] Could not write VTU: {e}", flush=True)


if __name__ == "__main__":
    main()
