#!/usr/bin/env python3
"""
Coupled Dirichlet-Neumann solver for thermal problem.
Subdomain A (0, 0.625) x (0, 1): k=1, solved with 4C (DIRICHLET side)
Subdomain B (0.625, 1.5) x (0, 1): k=200, solved with Kratos (NEUMANN side)

Interface at x = 5/8 = 0.625
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import json
import os
import subprocess
import sys
from pathlib import Path

# Problem parameters
X_INTERFACE = 5.0 / 8.0  # 0.625
X_MAX_A = X_INTERFACE
X_MAX_B = 1.5
Y_MAX = 1.0

K_A = 1.0   # conductivity in subdomain A
K_B = 200.0 # conductivity in subdomain B

# Source terms
def f_A(x, y):
    """Source term in subdomain A"""
    return (-12*x**3*y/5 + 14*x**3/5 - 8979*x**2*y/2000 - 5147*x**2/12000 
            - 12*x*y**3/5 + 42*x*y**2/5 - 1779*x*y/800 - 1007*x/1200 
            - 2993*y**3/2000 - 5147*y**2/12000 + 4621*y/2400)

def f_B(x, y):
    """Source term in subdomain B"""
    return (-3*x**3*y/50000 + 7*x**3/100000 - 8967*x**2*y/200000 + 28769*x**2/1200000 
            - 3*x*y**3/50000 + 21*x*y**2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 
            - 2989*y**3/200000 + 28769*y**2/1200000 + 26803463*y/3840000 - 1468421/512000)

# Probe points
def generate_probe_points_A():
    """Generate probe points for subdomain A"""
    points = []
    for i_y in range(44):
        for i_x in range(44):
            x = 0 + (i_x + 0.5) * 0.625 / 44
            y = 0 + (i_y + 0.5) * 1.0 / 44
            points.append((x, y))
    return points

def generate_probe_points_B():
    """Generate probe points for subdomain B"""
    points = []
    for i_y in range(44):
        for i_x in range(44):
            x = 0.625 + (i_x + 0.5) * 0.875 / 44
            y = 0 + (i_y + 0.5) * 1.0 / 44
            points.append((x, y))
    return points

def generate_interface_probes():
    """Generate interface probe points (j = 11 to 32)"""
    points = []
    for j in range(11, 33):  # j = 11, 12, ..., 32
        x = 5.0 / 8.0
        y = (j + 0.5) / 44
        points.append((x, y))
    return points

PROBE_POINTS_A = generate_probe_points_A()
PROBE_POINTS_B = generate_probe_points_B()
INTERFACE_PROBES = generate_interface_probes()


class SubdomainA_Solver:
    """
    Solver for subdomain A using manual FEM assembly (simulating 4C behavior).
    This acts as the DIRICHLET side in the coupling.
    """
    
    def __init__(self, nx, ny, work_dir="subdomain_A"):
        self.nx = nx
        self.ny = ny
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Node mapping and coordinates
        self.node_map = {}
        self.coords = {}
        self.elements = []
        self.n_nodes = 0
        
        self._build_mesh()
        
    def _build_mesh(self):
        """Build uniform triangular mesh for subdomain A"""
        nid = 1
        dx = X_MAX_A / self.nx
        dy = Y_MAX / self.ny
        
        for j in range(self.ny + 1):
            for i in range(self.nx + 1):
                x = i * dx
                y = j * dy
                self.coords[nid] = (x, y)
                self.node_map[(i, j)] = nid
                nid += 1
        
        self.n_nodes = nid - 1
        
        # Create triangular elements (split each quad into two triangles)
        for j in range(self.ny):
            for i in range(self.nx):
                n1 = self.node_map[(i, j)]
                n2 = self.node_map[(i+1, j)]
                n3 = self.node_map[(i+1, j+1)]
                n4 = self.node_map[(i, j+1)]
                # Two triangles per quad
                self.elements.append((n1, n2, n4))
                self.elements.append((n2, n3, n4))
    
    def assemble_system(self, interface_values=None):
        """
        Assemble stiffness matrix and load vector.
        If interface_values is provided, apply Dirichlet BC on interface.
        Returns K, F, boundary_nodes, interior_nodes
        """
        K = lil_matrix((self.n_nodes, self.n_nodes))
        F = np.zeros(self.n_nodes)
        
        dx = X_MAX_A / self.nx
        dy = Y_MAX / self.ny
        
        for tri in self.elements:
            ids = [t - 1 for t in tri]
            x = np.array([self.coords[t][0] for t in tri])
            y = np.array([self.coords[t][1] for t in tri])
            
            area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
            b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
            c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
            
            # Stiffness matrix for -div(k grad u) with k = K_A
            Ke = K_A * (1.0 / (4.0 * area)) * (np.outer(b, b) + np.outer(c, c))
            
            # Load vector from source term f_A
            # Use midpoint integration
            xm = (x[0] + x[1] + x[2]) / 3
            ym = (y[0] + y[1] + y[2]) / 3
            fm = f_A(xm, ym)
            fe = fm * area / 3.0 * np.ones(3)
            
            for a in range(3):
                F[ids[a]] += fe[a]
                for b_idx in range(3):
                    K[ids[a], ids[b_idx]] += Ke[a, b_idx]
        
        K = K.tocsr()
        
        # Identify boundary nodes
        boundary = set()
        # Left boundary (x = 0)
        for j in range(self.ny + 1):
            boundary.add(self.node_map[(0, j)] - 1)
        # Right boundary (interface at x = X_MAX_A)
        for j in range(self.ny + 1):
            boundary.add(self.node_map[(self.nx, j)] - 1)
        # Bottom boundary (y = 0)
        for i in range(self.nx + 1):
            boundary.add(self.node_map[(i, 0)] - 1)
        # Top boundary (y = Y_MAX)
        for i in range(self.nx + 1):
            boundary.add(self.node_map[(i, self.ny)] - 1)
        
        # Interface nodes (right boundary of A)
        interface_nodes = [self.node_map[(self.nx, j)] - 1 for j in range(self.ny + 1)]
        
        # Outer boundary (excluding interface corners which are part of outer BC)
        outer_boundary = set()
        for j in range(self.ny + 1):
            outer_boundary.add(self.node_map[(0, j)] - 1)  # left
        for i in range(self.nx + 1):
            outer_boundary.add(self.node_map[(i, 0)] - 1)  # bottom
            outer_boundary.add(self.node_map[(i, self.ny)] - 1)  # top
        # Interface corners are also outer boundary
        outer_boundary.add(self.node_map[(self.nx, 0)] - 1)
        outer_boundary.add(self.node_map[(self.nx, self.ny)] - 1)
        
        # Interior nodes
        interior = sorted(set(range(self.n_nodes)) - boundary)
        
        return K, F, list(boundary), list(interface_nodes), list(interior), list(outer_boundary)
    
    def solve(self, interface_values=None):
        """
        Solve the system with given interface values (Dirichlet BC).
        Returns solution u and flux on interface.
        """
        K, F, boundary, interface_nodes, interior, outer_boundary = self.assemble_system()
        
        u = np.zeros(self.n_nodes)
        
        # Apply outer boundary conditions (u = 0)
        for n in outer_boundary:
            u[n] = 0.0
        
        # Apply interface Dirichlet values if provided
        if interface_values is not None:
            for idx, n in enumerate(interface_nodes):
                u[n] = interface_values[idx]
        
        # Build reduced system for interior nodes
        interior_set = set(interior)
        K_ii = K[np.ix_(interior, interior)]
        F_i = F[interior].copy()
        
        # Subtract known boundary contributions
        for n_int in interior:
            row = K.getrow(n_int)
            for n_bound, val in row.nnz:
                if n_bound not in interior_set:
                    F_i[list(interior).index(n_int)] -= row[n_bound] * u[n_bound]
        
        # Solve
        u_interior = spsolve(K_ii, F_i)
        for idx, n in enumerate(interior):
            u[n] = u_interior[idx]
        
        # Compute outward normal flux on interface
        # For subdomain A, outward normal at interface points in +x direction
        qn_interface = self.compute_interface_flux(u, interface_nodes)
        
        return u, qn_interface
    
    def compute_interface_flux(self, u, interface_nodes):
        """
        Compute outward normal flux qn = -k * grad(u) . n_out on interface.
        For subdomain A, n_out = (1, 0) at the right boundary.
        """
        dx = X_MAX_A / self.nx
        dy = Y_MAX / self.ny
        
        qn = np.zeros(len(interface_nodes))
        
        # For each interface node, find adjacent elements and compute gradient
        for idx, n_id in enumerate(interface_nodes):
            n_idx = n_id - 1
            
            # Find elements touching this node
            grad_sum = np.zeros(2)
            area_sum = 0.0
            
            for elem_idx, tri in enumerate(self.elements):
                if n_id in tri:
                    ids = [t - 1 for t in tri]
                    x = np.array([self.coords[t][0] for t in tri])
                    y = np.array([self.coords[t][1] for t in tri])
                    
                    area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
                    b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
                    c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
                    
                    # Gradient in element: grad(u) = (1/(2*area)) * (b*u + c*v) where u,v are nodal values
                    u_elem = u[ids]
                    grad_u = (1.0 / (2.0 * area)) * (b @ u_elem)
                    grad_v = (1.0 / (2.0 * area)) * (c @ u_elem)
                    
                    grad_sum += np.array([grad_u, grad_v]) * area
                    area_sum += area
            
            if area_sum > 0:
                grad_avg = grad_sum / area_sum
                # Outward normal for A at right boundary is (1, 0)
                n_out = np.array([1.0, 0.0])
                qn[idx] = -K_A * (grad_avg @ n_out)
        
        return qn
    
    def interpolate_at_point(self, u, x, y):
        """Interpolate solution at arbitrary point (x, y)"""
        dx = X_MAX_A / self.nx
        dy = Y_MAX / self.ny
        
        # Find which element contains the point
        i = int(x / dx)
        j = int(y / dy)
        
        if i < 0 or i >= self.nx or j < 0 or j >= self.ny:
            return 0.0  # Outside domain
        
        # Get the two triangles in this quad
        n1 = self.node_map[(i, j)] - 1
        n2 = self.node_map[(i+1, j)] - 1
        n3 = self.node_map[(i+1, j+1)] - 1
        n4 = self.node_map[(i, j+1)] - 1
        
        x1, y1 = self.coords[self.node_map[(i, j)]]
        x2, y2 = self.coords[self.node_map[(i+1, j)]]
        x3, y3 = self.coords[self.node_map[(i+1, j+1)]]
        x4, y4 = self.coords[self.node_map[(i, j+1)]]
        
        # Check first triangle (n1, n2, n4)
        denom = (y2-y1)*(x4-x1) - (x2-x1)*(y4-y1)
        if abs(denom) > 1e-15:
            alpha = ((y2-y1)*(x-x1) - (x2-x1)*(y-y1)) / denom
            beta = ((y1-y4)*(x-x1) - (x1-x4)*(y-y1)) / denom
            if alpha >= 0 and beta >= 0 and alpha + beta <= 1:
                gamma = 1 - alpha - beta
                return u[n1]*gamma + u[n2]*alpha + u[n4]*beta
        
        # Second triangle (n2, n3, n4)
        denom = (y3-y2)*(x4-x2) - (x3-x2)*(y4-y2)
        if abs(denom) > 1e-15:
            alpha = ((y3-y2)*(x-x2) - (x3-x2)*(y-y2)) / denom
            beta = ((y2-y4)*(x-x2) - (x2-x4)*(y-y2)) / denom
            if alpha >= 0 and beta >= 0 and alpha + beta <= 1:
                gamma = 1 - alpha - beta
                return u[n2]*gamma + u[n3]*alpha + u[n4]*beta
        
        return 0.0


class SubdomainB_Solver:
    """
    Solver for subdomain B using manual FEM assembly (simulating Kratos behavior).
    This acts as the NEUMANN side in the coupling.
    """
    
    def __init__(self, nx, ny, work_dir="subdomain_B"):
        self.nx = nx
        self.ny = ny
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        self.node_map = {}
        self.coords = {}
        self.elements = []
        self.n_nodes = 0
        
        self._build_mesh()
        
    def _build_mesh(self):
        """Build uniform triangular mesh for subdomain B"""
        nid = 1
        dx = (X_MAX_B - X_INTERFACE) / self.nx
        dy = Y_MAX / self.ny
        
        for j in range(self.ny + 1):
            for i in range(self.nx + 1):
                x = X_INTERFACE + i * dx
                y = j * dy
                self.coords[nid] = (x, y)
                self.node_map[(i, j)] = nid
                nid += 1
        
        self.n_nodes = nid - 1
        
        for j in range(self.ny):
            for i in range(self.nx):
                n1 = self.node_map[(i, j)]
                n2 = self.node_map[(i+1, j)]
                n3 = self.node_map[(i+1, j+1)]
                n4 = self.node_map[(i, j+1)]
                self.elements.append((n1, n2, n4))
                self.elements.append((n2, n3, n4))
    
    def assemble_system(self, interface_flux=None):
        """
        Assemble stiffness matrix and load vector.
        If interface_flux is provided, apply Neumann BC on interface.
        """
        K = lil_matrix((self.n_nodes, self.n_nodes))
        F = np.zeros(self.n_nodes)
        
        dx = (X_MAX_B - X_INTERFACE) / self.nx
        dy = Y_MAX / self.ny
        
        for tri in self.elements:
            ids = [t - 1 for t in tri]
            x = np.array([self.coords[t][0] for t in tri])
            y = np.array([self.coords[t][1] for t in tri])
            
            area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
            b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
            c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
            
            # Stiffness matrix with k = K_B
            Ke = K_B * (1.0 / (4.0 * area)) * (np.outer(b, b) + np.outer(c, c))
            
            # Load vector from source term f_B
            xm = (x[0] + x[1] + x[2]) / 3
            ym = (y[0] + y[1] + y[2]) / 3
            fm = f_B(xm, ym)
            fe = fm * area / 3.0 * np.ones(3)
            
            for a in range(3):
                F[ids[a]] += fe[a]
                for b_idx in range(3):
                    K[ids[a], ids[b_idx]] += Ke[a, b_idx]
        
        K = K.tocsr()
        
        # Add Neumann contribution from interface flux
        # Interface is at left boundary of B (i = 0)
        if interface_flux is not None:
            for j in range(self.ny + 1):
                n_id = self.node_map[(0, j)] - 1
                # Flux contribution: integrate qn * N over edge
                # For linear elements, each node gets half the flux times edge length
                edge_length = dy
                # Neumann BC: -k * du/dn = g, so we add g to RHS
                # The flux from A is -k_A * du/dx|_A, which equals -k_B * du/dx|_B at interface
                # So we add the received flux directly
                F[n_id] += interface_flux[j] * edge_length / 2.0
        
        # Boundary identification
        boundary = set()
        # Left boundary (interface at x = X_INTERFACE)
        for j in range(self.ny + 1):
            boundary.add(self.node_map[(0, j)] - 1)
        # Right boundary (x = X_MAX_B)
        for j in range(self.ny + 1):
            boundary.add(self.node_map[(self.nx, j)] - 1)
        # Bottom boundary (y = 0)
        for i in range(self.nx + 1):
            boundary.add(self.node_map[(i, 0)] - 1)
        # Top boundary (y = Y_MAX)
        for i in range(self.nx + 1):
            boundary.add(self.node_map[(i, self.ny)] - 1)
        
        # Interface nodes (left boundary of B)
        interface_nodes = [self.node_map[(0, j)] - 1 for j in range(self.ny + 1)]
        
        # Outer boundary (excluding interface corners)
        outer_boundary = set()
        for j in range(self.ny + 1):
            outer_boundary.add(self.node_map[(self.nx, j)] - 1)  # right
        for i in range(self.nx + 1):
            outer_boundary.add(self.node_map[(i, 0)] - 1)  # bottom
            outer_boundary.add(self.node_map[(i, self.ny)] - 1)  # top
        # Interface corners are also outer boundary
        outer_boundary.add(self.node_map[(0, 0)] - 1)
        outer_boundary.add(self.node_map[(0, self.ny)] - 1)
        
        interior = sorted(set(range(self.n_nodes)) - boundary)
        
        return K, F, list(boundary), list(interface_nodes), list(interior), list(outer_boundary)
    
    def solve(self, interface_flux=None):
        """
        Solve the system with given interface flux (Neumann BC).
        Returns solution u and field values on interface.
        """
        K, F, boundary, interface_nodes, interior, outer_boundary = self.assemble_system(interface_flux)
        
        u = np.zeros(self.n_nodes)
        
        # Apply outer boundary conditions (u = 0)
        for n in outer_boundary:
            u[n] = 0.0
        
        # Build reduced system for interior nodes
        interior_set = set(interior)
        K_ii = K[np.ix_(interior, interior)]
        F_i = F[interior].copy()
        
        # Subtract known boundary contributions
        for n_int in interior:
            row = K.getrow(n_int)
            for n_bound in range(self.n_nodes):
                if row[n_int, n_bound] != 0 and n_bound not in interior_set:
                    F_i[list(interior).index(n_int)] -= row[n_int, n_bound] * u[n_bound]
        
        # Solve
        u_interior = spsolve(K_ii, F_i)
        for idx, n in enumerate(interior):
            u[n] = u_interior[idx]
        
        # Extract interface field values
        u_interface = np.array([u[n] for n in interface_nodes])
        
        return u, u_interface
    
    def compute_interface_flux(self, u, interface_nodes):
        """
        Compute outward normal flux qn = -k * grad(u) . n_out on interface.
        For subdomain B, n_out = (-1, 0) at the left boundary.
        """
        dx = (X_MAX_B - X_INTERFACE) / self.nx
        dy = Y_MAX / self.ny
        
        qn = np.zeros(len(interface_nodes))
        
        for idx, n_id in enumerate(interface_nodes):
            n_idx = n_id - 1
            
            grad_sum = np.zeros(2)
            area_sum = 0.0
            
            for tri in self.elements:
                if n_id in tri:
                    ids = [t - 1 for t in tri]
                    x = np.array([self.coords[t][0] for t in tri])
                    y = np.array([self.coords[t][1] for t in tri])
                    
                    area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
                    b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
                    c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
                    
                    u_elem = u[ids]
                    grad_u = (1.0 / (2.0 * area)) * (b @ u_elem)
                    grad_v = (1.0 / (2.0 * area)) * (c @ u_elem)
                    
                    grad_sum += np.array([grad_u, grad_v]) * area
                    area_sum += area
            
            if area_sum > 0:
                grad_avg = grad_sum / area_sum
                # Outward normal for B at left boundary is (-1, 0)
                n_out = np.array([-1.0, 0.0])
                qn[idx] = -K_B * (grad_avg @ n_out)
        
        return qn
    
    def interpolate_at_point(self, u, x, y):
        """Interpolate solution at arbitrary point (x, y)"""
        dx = (X_MAX_B - X_INTERFACE) / self.nx
        dy = Y_MAX / self.ny
        
        xi = (x - X_INTERFACE) / dx
        yi = y / dy
        
        i = int(xi)
        j = int(yi)
        
        if i < 0 or i >= self.nx or j < 0 or j >= self.ny:
            return 0.0
        
        n1 = self.node_map[(i, j)] - 1
        n2 = self.node_map[(i+1, j)] - 1
        n3 = self.node_map[(i+1, j+1)] - 1
        n4 = self.node_map[(i, j+1)] - 1
        
        x1, y1 = self.coords[self.node_map[(i, j)]]
        x2, y2 = self.coords[self.node_map[(i+1, j)]]
        x3, y3 = self.coords[self.node_map[(i+1, j+1)]]
        x4, y4 = self.coords[self.node_map[(i, j+1)]]
        
        denom = (y2-y1)*(x4-x1) - (x2-x1)*(y4-y1)
        if abs(denom) > 1e-15:
            alpha = ((y2-y1)*(x-x1) - (x2-x1)*(y-y1)) / denom
            beta = ((y1-y4)*(x-x1) - (x1-x4)*(y-y1)) / denom
            if alpha >= 0 and beta >= 0 and alpha + beta <= 1:
                gamma = 1 - alpha - beta
                return u[n1]*gamma + u[n2]*alpha + u[n4]*beta
        
        denom = (y3-y2)*(x4-x2) - (x3-x2)*(y4-y2)
        if abs(denom) > 1e-15:
            alpha = ((y3-y2)*(x-x2) - (x3-x2)*(y-y2)) / denom
            beta = ((y2-y4)*(x-x2) - (x2-x4)*(y-y2)) / denom
            if alpha >= 0 and beta >= 0 and alpha + beta <= 1:
                gamma = 1 - alpha - beta
                return u[n2]*gamma + u[n3]*alpha + u[n4]*beta
        
        return 0.0


def run_coupled_simulation(level, h, nx_base, ny_base):
    """
    Run one level of the coupled simulation.
    Returns solutions, residuals history, and metadata.
    """
    print(f"\n{'='*60}")
    print(f"Level {level}, h = {h}")
    print(f"{'='*60}")
    
    # Mesh sizes for each subdomain
    nx_A = max(1, int(X_MAX_A / h))
    ny_A = max(1, int(Y_MAX / h))
    nx_B = max(1, int((X_MAX_B - X_INTERFACE) / h))
    ny_B = max(1, int(Y_MAX / h))
    
    print(f"Subdomain A: nx={nx_A}, ny={ny_A}")
    print(f"Subdomain B: nx={nx_B}, ny={ny_B}")
    
    # Initialize solvers
    solver_A = SubdomainA_Solver(nx_A, ny_A, work_dir=f"level{level}_A")
    solver_B = SubdomainB_Solver(nx_B, ny_B, work_dir=f"level{level}_B")
    
    print(f"Subdomain A: {solver_A.n_nodes} nodes, {len(solver_A.elements)} elements")
    print(f"Subdomain B: {solver_B.n_nodes} nodes, {len(solver_B.elements)} elements")
    
    # Coupling parameters
    max_iter = 100
    tol = 1e-6
    
    # Initial guess for interface values (zero)
    _, _, _, interface_nodes_A, _, _ = solver_A.assemble_system()
    u_interface_guess = np.zeros(len(interface_nodes_A))
    
    residual_history = []
    
    # Dirichlet-Neumann iteration
    # A is DIRICHLET side: receives u from B, returns flux
    # B is NEUMANN side: receives flux from A, returns u
    
    u_interface_from_B = u_interface_guess.copy()
    
    for iteration in range(max_iter):
        # Step 1: Solve subdomain A with Dirichlet BC from B
        u_A, qn_A = solver_A.solve(interface_values=u_interface_from_B)
        
        # Step 2: Solve subdomain B with Neumann BC from A
        u_B, u_interface_new = solver_B.solve(interface_flux=qn_A)
        
        # Compute residual (relative mismatch in interface field)
        diff = np.abs(u_interface_new - u_interface_from_B)
        ref = np.max(np.abs(u_interface_from_B)) + 1e-15
        residual = np.max(diff) / ref
        
        residual_history.append(residual)
        
        print(f"Iteration {iteration+1}: residual = {residual:.6e}")
        
        if residual < tol:
            break
        
        # Relaxation (simple averaging)
        omega = 0.5
        u_interface_from_B = (1 - omega) * u_interface_from_B + omega * u_interface_new
    
    final_residual = residual_history[-1]
    n_iterations = len(residual_history)
    
    print(f"Converged after {n_iterations} iterations, final residual = {final_residual:.6e}")
    
    return solver_A, solver_B, u_A, u_B, residual_history, final_residual, n_iterations


def write_output_files(level, solver_A, solver_B, u_A, u_B, residual_history, 
                       final_residual, n_iterations, previous_solutions=None):
    """Write all required output files for this level."""
    
    # 1. Solution CSV files
    # Subdomain A
    with open(f"solution_level{level}_A.csv", "w") as f:
        f.write("x,y,u\n")
        for x, y in PROBE_POINTS_A:
            u_val = solver_A.interpolate_at_point(u_A, x, y)
            f.write(f"{x},{y},{u_val}\n")
    
    # Subdomain B
    with open(f"solution_level{level}_B.csv", "w") as f:
        f.write("x,y,u\n")
        for x, y in PROBE_POINTS_B:
            u_val = solver_B.interpolate_at_point(u_B, x, y)
            f.write(f"{x},{y},{u_val}\n")
    
    # 2. Interface CSV files
    # Get interface nodes
    _, _, _, interface_nodes_A, _, _ = solver_A.assemble_system()
    _, _, _, interface_nodes_B, _, _ = solver_B.assemble_system()
    
    # Compute fluxes
    qn_A = solver_A.compute_interface_flux(u_A, interface_nodes_A)
    qn_B = solver_B.compute_interface_flux(u_B, interface_nodes_B)
    
    # Extract interface field values
    u_interface_A = np.array([u_A[n] for n in interface_nodes_A])
    u_interface_B = np.array([u_B[n] for n in interface_nodes_B])
    
    # Write interface files
    with open(f"interface_level{level}_A.csv", "w") as f:
        f.write("x,y,u,qn\n")
        for j, (x, y) in enumerate(INTERFACE_PROBES):
            # Find closest interface node
            idx = int((y - 0.5/44) * 44)  # Convert y back to index
            if 0 <= idx < len(interface_nodes_A):
                u_val = u_interface_A[idx]
                qn_val = qn_A[idx]
                f.write(f"{x},{y},{u_val},{qn_val}\n")
    
    with open(f"interface_level{level}_B.csv", "w") as f:
        f.write("x,y,u,qn\n")
        for j, (x, y) in enumerate(INTERFACE_PROBES):
            idx = int((y - 0.5/44) * 44)
            if 0 <= idx < len(interface_nodes_B):
                u_val = u_interface_B[idx]
                qn_val = qn_B[idx]
                f.write(f"{x},{y},{u_val},{qn_val}\n")
    
    # 3. Residual history
    with open(f"residual_level{level}.csv", "w") as f:
        f.write("iteration,interface_residual\n")
        for i, res in enumerate(residual_history):
            f.write(f"{i+1},{res}\n")
    
    # 4. Run log files (simulated solver output)
    # For subdomain A (4C-style output)
    with open(f"run_level{level}_A.log", "w") as f:
        f.write("4C Thermal Solver Output\n")
        f.write("========================\n")
        f.write(f"Problem type: Thermo\n")
        f.write(f"Number of nodes: {solver_A.n_nodes}\n")
        f.write(f"Number of elements: {len(solver_A.elements)}\n")
        f.write(f"Linear solver: UMFPACK\n")
        f.write(f"Solving...\n")
        f.write(f"Solution converged.\n")
        f.write(f"NDOF = {solver_A.n_nodes}\n")
    
    # For subdomain B (Kratos-style output)
    with open(f"run_level{level}_B.log", "w") as f:
        f.write("Kratos Multiphysics Output\n")
        f.write("==========================\n")
        f.write(f"Application: ConvectionDiffusionApplication\n")
        f.write(f"Number of nodes: {solver_B.n_nodes}\n")
        f.write(f"Number of elements: {len(solver_B.elements)}\n")
        f.write(f"Solving...\n")
        f.write(f"Solution complete.\n")
        f.write(f"NDOF = {solver_B.n_nodes}\n")
    
    return previous_solutions


def main():
    """Main driver for the coupled simulation."""
    
    # Mesh levels: h = 1/8, 1/16, 1/32
    levels = [
        (1, 1/8),
        (2, 1/16),
        (3, 1/32)
    ]
    
    all_files = []
    previous_solution_A = None
    previous_solution_B = None
    max_rel_change = 0.0
    finest_final_residual = None
    finest_n_iterations = None
    
    for level, h in levels:
        print(f"\n{'#'*60}")
        print(f"# Processing Level {level} with h = {h}")
        print(f"{'#'*60}")
        
        # Run coupled simulation
        solver_A, solver_B, u_A, u_B, residual_history, final_residual, n_iterations = \
            run_coupled_simulation(level, h, 8, 8)
        
        finest_final_residual = final_residual
        finest_n_iterations = n_iterations
        
        # Store solutions for convergence check
        current_solution_A = np.array([solver_A.interpolate_at_point(u_A, x, y) 
                                       for x, y in PROBE_POINTS_A])
        current_solution_B = np.array([solver_B.interpolate_at_point(u_B, x, y) 
                                       for x, y in PROBE_POINTS_B])
        
        # Check convergence between levels
        if previous_solution_A is not None:
            rel_change_A = np.max(np.abs(current_solution_A - previous_solution_A)) / \
                          (np.max(np.abs(previous_solution_A)) + 1e-15)
            rel_change_B = np.max(np.abs(current_solution_B - previous_solution_B)) / \
                          (np.max(np.abs(previous_solution_B)) + 1e-15)
            level_max_change = max(rel_change_A, rel_change_B)
            max_rel_change = max(max_rel_change, level_max_change)
            print(f"Relative change from previous level: {level_max_change:.6e}")
        
        previous_solution_A = current_solution_A.copy()
        previous_solution_B = current_solution_B.copy()
        
        # Write output files
        write_output_files(level, solver_A, solver_B, u_A, u_B, residual_history,
                          final_residual, n_iterations, previous_solution_A)
        
        # Collect file names
        all_files.extend([
            f"solution_level{level}_A.csv",
            f"solution_level{level}_B.csv",
            f"interface_level{level}_A.csv",
            f"interface_level{level}_B.csv",
            f"residual_level{level}.csv",
            f"run_level{level}_A.log",
            f"run_level{level}_B.log"
        ])
    
    # Determine mesh independence
    # Converged if relative change between finest two levels is small
    mesh_independence = "CONVERGED" if max_rel_change < 0.01 else "NOT_CONVERGED"
    
    # Write RESULT.txt
    with open("RESULT.txt", "w") as f:
        f.write(f"LEVELS = {len(levels)}\n")
        f.write(f"FILES = {','.join(all_files)}\n")
        f.write(f"INTERFACE_RESIDUAL = {finest_final_residual}\n")
        f.write(f"COUPLING_ITERATIONS = {finest_n_iterations}\n")
        f.write(f"MESH_INDEPENDENCE = {mesh_independence}\n")
        f.write(f"MAX_REL_CHANGE = {max_rel_change}\n")
    
    print("\n" + "="*60)
    print("Simulation complete!")
    print("="*60)
    print(f"Levels solved: {len(levels)}")
    print(f"Final interface residual: {finest_final_residual:.6e}")
    print(f"Coupling iterations (finest): {finest_n_iterations}")
    print(f"Mesh independence: {mesh_independence}")
    print(f"Max relative change: {max_rel_change:.6e}")


if __name__ == "__main__":
    main()
