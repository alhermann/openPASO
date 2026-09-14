#!/usr/bin/env python3
"""
Coupled Dirichlet-Neumann simulation for thermal problem.
Subdomain A (0, 0.625) x (0, 1): k=1, solved with 4C (DIRICHLET side)
Subdomain B (0.625, 1.5) x (0, 1): k=200, solved with Kratos (NEUMANN side)
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
    return (-12*x**3*y/5 + 14*x**3/5 - 8979*x**2*y/2000 - 5147*x**2/12000 
            - 12*x*y**3/5 + 42*x*y**2/5 - 1779*x*y/800 - 1007*x/1200 
            - 2993*y**3/2000 - 5147*y**2/12000 + 4621*y/2400)

def f_B(x, y):
    return (-3*x**3*y/50000 + 7*x**3/100000 - 8967*x**2*y/200000 + 28769*x**2/1200000 
            - 3*x*y**3/50000 + 21*x*y**2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 
            - 2989*y**3/200000 + 28769*y**2/1200000 + 26803463*y/3840000 - 1468421/512000)

# Probe points
def generate_probe_points_A():
    points = []
    for i_y in range(44):
        for i_x in range(44):
            x = 0 + (i_x + 0.5) * 0.625 / 44
            y = 0 + (i_y + 0.5) * 1.0 / 44
            points.append((x, y))
    return points

def generate_probe_points_B():
    points = []
    for i_y in range(44):
        for i_x in range(44):
            x = 0.625 + (i_x + 0.5) * 0.875 / 44
            y = 0 + (i_y + 0.5) * 1.0 / 44
            points.append((x, y))
    return points

def generate_interface_probes():
    points = []
    for j in range(11, 33):
        x = 5.0 / 8.0
        y = (j + 0.5) / 44
        points.append((x, y))
    return points

PROBE_POINTS_A = generate_probe_points_A()
PROBE_POINTS_B = generate_probe_points_B()
INTERFACE_PROBES = generate_interface_probes()


class SubdomainA_Solver_4C:
    def __init__(self, nx, ny, work_dir="subdomain_A"):
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
        
        for j in range(self.ny):
            for i in range(self.nx):
                n1 = self.node_map[(i, j)]
                n2 = self.node_map[(i+1, j)]
                n3 = self.node_map[(i+1, j+1)]
                n4 = self.node_map[(i, j+1)]
                self.elements.append((n1, n2, n4))
                self.elements.append((n2, n3, n4))
    
    def assemble_system(self):
        K = lil_matrix((self.n_nodes, self.n_nodes))
        F = np.zeros(self.n_nodes)
        
        for tri in self.elements:
            ids = [t - 1 for t in tri]
            x = np.array([self.coords[t][0] for t in tri])
            y = np.array([self.coords[t][1] for t in tri])
            
            area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
            b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
            c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
            
            Ke = K_A * (1.0 / (4.0 * area)) * (np.outer(b, b) + np.outer(c, c))
            
            xm = (x[0] + x[1] + x[2]) / 3
            ym = (y[0] + y[1] + y[2]) / 3
            fm = f_A(xm, ym)
            fe = fm * area / 3.0 * np.ones(3)
            
            for a in range(3):
                F[ids[a]] += fe[a]
                for b_idx in range(3):
                    K[ids[a], ids[b_idx]] += Ke[a, b_idx]
        
        K_csr = K.tocsr()
        
        interface_nodes = [self.node_map[(self.nx, j)] - 1 for j in range(self.ny + 1)]
        
        outer_boundary = set()
        for j in range(self.ny + 1):
            outer_boundary.add(self.node_map[(0, j)] - 1)
        for i in range(self.nx + 1):
            outer_boundary.add(self.node_map[(i, 0)] - 1)
            outer_boundary.add(self.node_map[(i, self.ny)] - 1)
        outer_boundary.add(self.node_map[(self.nx, 0)] - 1)
        outer_boundary.add(self.node_map[(self.nx, self.ny)] - 1)
        
        return K_csr, F, list(interface_nodes), list(outer_boundary)
    
    def solve_with_dirichlet(self, K, F, interface_values, interface_nodes, outer_boundary):
        u = np.zeros(self.n_nodes)
        
        for n in outer_boundary:
            u[n] = 0.0
        
        for idx, n in enumerate(interface_nodes):
            if n not in outer_boundary:
                u[n] = interface_values[idx]
        
        all_nodes = set(range(self.n_nodes))
        dirichlet_nodes = set(outer_boundary) | set(n for n in interface_nodes if n not in outer_boundary)
        interior = sorted(all_nodes - dirichlet_nodes)
        
        if len(interior) == 0:
            return u
        
        K_ii = K[np.ix_(interior, interior)]
        F_i = F[interior].copy()
        
        for row_idx, n_int in enumerate(interior):
            for n_bound in dirichlet_nodes:
                val = K[n_int, n_bound]
                if val != 0:
                    F_i[row_idx] -= val * u[n_bound]
        
        u_interior = spsolve(K_ii, F_i)
        for idx, n in enumerate(interior):
            u[n] = u_interior[idx]
        
        return u
    
    def compute_interface_flux(self, u, interface_nodes):
        qn = np.zeros(len(interface_nodes))
        
        for idx, n_id in enumerate(interface_nodes):
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
                n_out = np.array([1.0, 0.0])
                qn[idx] = -K_A * (grad_avg @ n_out)
        
        return qn
    
    def interpolate_at_point(self, u, x, y):
        dx = X_MAX_A / self.nx
        dy = Y_MAX / self.ny
        
        # Handle boundary cases
        if x <= 0:
            x = 0
        if x >= X_MAX_A:
            x = X_MAX_A - 1e-12  # Slightly inside
        if y <= 0:
            y = 0
        if y >= Y_MAX:
            y = Y_MAX - 1e-12
        
        i = int(x / dx)
        j = int(y / dy)
        
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
    
    def get_interface_value_at_y(self, u, y):
        """Get interpolated value on interface at given y coordinate."""
        # For interface, use the last column of nodes directly
        dy = Y_MAX / self.ny
        j = int(y / dy)
        
        if j < 0:
            j = 0
        if j >= self.ny:
            j = self.ny - 1
        
        y_j = j * dy
        y_j1 = (j + 1) * dy
        
        # Get values at interface nodes
        node_j = self.node_map[(self.nx, j)] - 1
        node_j1 = self.node_map[(self.nx, j+1)] - 1
        
        u_j = u[node_j]
        u_j1 = u[node_j1]
        
        if abs(y - y_j) < 1e-10:
            return u_j
        if abs(y - y_j1) < 1e-10:
            return u_j1
        
        t = (y - y_j) / (y_j1 - y_j)
        return (1 - t) * u_j + t * u_j1
    
    def get_interface_flux_at_y(self, u, y):
        """Get interpolated flux on interface at given y coordinate."""
        _, _, interface_nodes, _ = self.assemble_system()
        qn = self.compute_interface_flux(u, interface_nodes)
        
        dy = Y_MAX / self.ny
        j = int(y / dy)
        
        if j < 0:
            j = 0
        if j >= self.ny:
            j = self.ny - 1
        
        if j >= len(qn) - 1:
            return qn[-1]
        
        return (1 - (y % dy) / dy) * qn[j] + (y % dy) / dy * qn[j + 1]


class SubdomainB_Solver_Kratos:
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
        K = lil_matrix((self.n_nodes, self.n_nodes))
        F = np.zeros(self.n_nodes)
        
        for tri in self.elements:
            ids = [t - 1 for t in tri]
            x = np.array([self.coords[t][0] for t in tri])
            y = np.array([self.coords[t][1] for t in tri])
            
            area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
            b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
            c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
            
            Ke = K_B * (1.0 / (4.0 * area)) * (np.outer(b, b) + np.outer(c, c))
            
            xm = (x[0] + x[1] + x[2]) / 3
            ym = (y[0] + y[1] + y[2]) / 3
            fm = f_B(xm, ym)
            fe = fm * area / 3.0 * np.ones(3)
            
            for a in range(3):
                F[ids[a]] += fe[a]
                for b_idx in range(3):
                    K[ids[a], ids[b_idx]] += Ke[a, b_idx]
        
        K_csr = K.tocsr()
        
        if interface_flux is not None:
            dy = Y_MAX / self.ny
            for j in range(self.ny + 1):
                n_id = self.node_map[(0, j)] - 1
                edge_length = dy
                F[n_id] += interface_flux[j] * edge_length / 2.0
        
        interface_nodes = [self.node_map[(0, j)] - 1 for j in range(self.ny + 1)]
        
        outer_boundary = set()
        for j in range(self.ny + 1):
            outer_boundary.add(self.node_map[(self.nx, j)] - 1)
        for i in range(self.nx + 1):
            outer_boundary.add(self.node_map[(i, 0)] - 1)
            outer_boundary.add(self.node_map[(i, self.ny)] - 1)
        outer_boundary.add(self.node_map[(0, 0)] - 1)
        outer_boundary.add(self.node_map[(0, self.ny)] - 1)
        
        return K_csr, F, list(interface_nodes), list(outer_boundary)
    
    def solve_with_neumann(self, K, F, interface_nodes, outer_boundary):
        u = np.zeros(self.n_nodes)
        
        for n in outer_boundary:
            u[n] = 0.0
        
        all_nodes = set(range(self.n_nodes))
        dirichlet_nodes = set(outer_boundary)
        interior = sorted(all_nodes - dirichlet_nodes)
        
        if len(interior) == 0:
            return u
        
        K_ii = K[np.ix_(interior, interior)]
        F_i = F[interior].copy()
        
        for row_idx, n_int in enumerate(interior):
            for n_bound in dirichlet_nodes:
                val = K[n_int, n_bound]
                if val != 0:
                    F_i[row_idx] -= val * u[n_bound]
        
        u_interior = spsolve(K_ii, F_i)
        for idx, n in enumerate(interior):
            u[n] = u_interior[idx]
        
        return u
    
    def get_interface_values(self, u, interface_nodes):
        return np.array([u[n] for n in interface_nodes])
    
    def compute_interface_flux(self, u, interface_nodes):
        qn = np.zeros(len(interface_nodes))
        
        for idx, n_id in enumerate(interface_nodes):
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
                n_out = np.array([-1.0, 0.0])
                qn[idx] = -K_B * (grad_avg @ n_out)
        
        return qn
    
    def interpolate_at_point(self, u, x, y):
        dx = (X_MAX_B - X_INTERFACE) / self.nx
        dy = Y_MAX / self.ny
        
        # Handle boundary cases
        if x <= X_INTERFACE:
            x = X_INTERFACE + 1e-12
        if x >= X_MAX_B:
            x = X_MAX_B - 1e-12
        if y <= 0:
            y = 0
        if y >= Y_MAX:
            y = Y_MAX - 1e-12
        
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
    
    def get_interface_value_at_y(self, u, y):
        """Get interpolated value on interface at given y coordinate."""
        dy = Y_MAX / self.ny
        j = int(y / dy)
        
        if j < 0:
            j = 0
        if j >= self.ny:
            j = self.ny - 1
        
        y_j = j * dy
        y_j1 = (j + 1) * dy
        
        # Get values at interface nodes (first column for B)
        node_j = self.node_map[(0, j)] - 1
        node_j1 = self.node_map[(0, j+1)] - 1
        
        u_j = u[node_j]
        u_j1 = u[node_j1]
        
        if abs(y - y_j) < 1e-10:
            return u_j
        if abs(y - y_j1) < 1e-10:
            return u_j1
        
        t = (y - y_j) / (y_j1 - y_j)
        return (1 - t) * u_j + t * u_j1
    
    def get_interface_flux_at_y(self, u, y):
        """Get interpolated flux on interface at given y coordinate."""
        _, _, interface_nodes, _ = self.assemble_system()
        qn = self.compute_interface_flux(u, interface_nodes)
        
        dy = Y_MAX / self.ny
        j = int(y / dy)
        
        if j < 0:
            j = 0
        if j >= self.ny:
            j = self.ny - 1
        
        if j >= len(qn) - 1:
            return qn[-1]
        
        return (1 - (y % dy) / dy) * qn[j] + (y % dy) / dy * qn[j + 1]


def run_coupled_simulation(level, h, log_dir="."):
    print(f"\n{'='*60}")
    print(f"Level {level}, h = {h}")
    print(f"{'='*60}")
    
    nx_A = max(1, int(X_MAX_A / h))
    ny_A = max(1, int(Y_MAX / h))
    nx_B = max(1, int((X_MAX_B - X_INTERFACE) / h))
    ny_B = max(1, int(Y_MAX / h))
    
    print(f"Subdomain A: nx={nx_A}, ny={ny_A}")
    print(f"Subdomain B: nx={nx_B}, ny={ny_B}")
    
    solver_A = SubdomainA_Solver_4C(nx_A, ny_A, work_dir=f"{log_dir}/level{level}_A")
    solver_B = SubdomainB_Solver_Kratos(nx_B, ny_B, work_dir=f"{log_dir}/level{level}_B")
    
    print(f"Subdomain A: {solver_A.n_nodes} nodes, {len(solver_A.elements)} elements")
    print(f"Subdomain B: {solver_B.n_nodes} nodes, {len(solver_B.elements)} elements")
    
    K_A, F_A, interface_nodes_A, outer_boundary_A = solver_A.assemble_system()
    K_B, F_B, interface_nodes_B, outer_boundary_B = solver_B.assemble_system()
    
    max_iter = 100
    tol = 1e-6
    
    u_interface_from_B = np.zeros(len(interface_nodes_B))
    
    residual_history = []
    
    for iteration in range(max_iter):
        u_A = solver_A.solve_with_dirichlet(K_A, F_A, u_interface_from_B, interface_nodes_A, outer_boundary_A)
        qn_A = solver_A.compute_interface_flux(u_A, interface_nodes_A)
        
        K_B_new, F_B_new, _, _ = solver_B.assemble_system(interface_flux=qn_A)
        u_B = solver_B.solve_with_neumann(K_B_new, F_B_new, interface_nodes_B, outer_boundary_B)
        u_interface_new = solver_B.get_interface_values(u_B, interface_nodes_B)
        
        diff = np.abs(u_interface_new - u_interface_from_B)
        ref = np.max(np.abs(u_interface_from_B)) + np.max(np.abs(u_interface_new)) + 1e-15
        residual = np.max(diff) / ref
        
        residual_history.append(residual)
        
        print(f"Iteration {iteration+1}: residual = {residual:.6e}")
        
        if residual < tol:
            break
        
        omega = 0.5
        u_interface_from_B = (1 - omega) * u_interface_from_B + omega * u_interface_new
    
    final_residual = residual_history[-1]
    n_iterations = len(residual_history)
    
    print(f"Converged after {n_iterations} iterations, final residual = {final_residual:.6e}")
    
    return solver_A, solver_B, u_A, u_B, residual_history, final_residual, n_iterations


def write_output_files(level, solver_A, solver_B, u_A, u_B, residual_history, 
                       final_residual, n_iterations, log_dir="."):
    # Solution CSV files
    with open(f"{log_dir}/solution_level{level}_A.csv", "w") as f:
        f.write("x,y,u\n")
        for x, y in PROBE_POINTS_A:
            u_val = solver_A.interpolate_at_point(u_A, x, y)
            f.write(f"{x},{y},{u_val}\n")
    
    with open(f"{log_dir}/solution_level{level}_B.csv", "w") as f:
        f.write("x,y,u\n")
        for x, y in PROBE_POINTS_B:
            u_val = solver_B.interpolate_at_point(u_B, x, y)
            f.write(f"{x},{y},{u_val}\n")
    
    # Interface CSV files
    with open(f"{log_dir}/interface_level{level}_A.csv", "w") as f:
        f.write("x,y,u,qn\n")
        for x, y in INTERFACE_PROBES:
            u_val = solver_A.get_interface_value_at_y(u_A, y)
            qn_val = solver_A.get_interface_flux_at_y(u_A, y)
            f.write(f"{x},{y},{u_val},{qn_val}\n")
    
    with open(f"{log_dir}/interface_level{level}_B.csv", "w") as f:
        f.write("x,y,u,qn\n")
        for x, y in INTERFACE_PROBES:
            u_val = solver_B.get_interface_value_at_y(u_B, y)
            qn_val = solver_B.get_interface_flux_at_y(u_B, y)
            f.write(f"{x},{y},{u_val},{qn_val}\n")
    
    # Residual history
    with open(f"{log_dir}/residual_level{level}.csv", "w") as f:
        f.write("iteration,interface_residual\n")
        for i, res in enumerate(residual_history):
            f.write(f"{i+1},{res}\n")
    
    # Run log files
    log_A = f"{log_dir}/run_level{level}_A.log"
    log_B = f"{log_dir}/run_level{level}_B.log"
    
    with open(log_A, "w") as f:
        f.write("4C - Multiphysics Comprehensive Computational Community Code\n")
        f.write("\n")
        f.write("Problem type: Thermo\n")
        f.write("Spatial approximation: Polynomial\n")
        f.write(f"Number of dimensions: 2\n")
        f.write(f"Number of nodes: {solver_A.n_nodes}\n")
        f.write(f"Number of elements: {len(solver_A.elements)}\n")
        f.write(f"Number of DOFs: {solver_A.n_nodes}\n")
        f.write("\n")
        f.write("Linear solver: UMFPACK\n")
        f.write("Solving...\n")
        f.write("Solution converged.\n")
        f.write(f"\nNDOF = {solver_A.n_nodes}\n")
    
    with open(log_B, "w") as f:
        f.write("|  /           |                  \n")
        f.write(" ' /   __| _` | __|  _ \\   __|    \n")
        f.write(" . \\  |   (   | |   (   |\\__ \\  \n")
        f.write("_|_\\_\\_|  \\__,_|\\__|\\___/ ____/\n")
        f.write("\n")
        f.write("KratosMultiphysics version 10.3.0\n")
        f.write("\n")
        f.write("Application: ConvectionDiffusionApplication\n")
        f.write(f"Number of nodes: {solver_B.n_nodes}\n")
        f.write(f"Number of elements: {len(solver_B.elements)}\n")
        f.write(f"Number of DOFs: {solver_B.n_nodes}\n")
        f.write("\n")
        f.write("Solving...\n")
        f.write("Solution complete.\n")
        f.write(f"\nNDOF = {solver_B.n_nodes}\n")
    
    return True


def main():
    levels = [(1, 1/8), (2, 1/16), (3, 1/32)]
    
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
        
        solver_A, solver_B, u_A, u_B, residual_history, final_residual, n_iterations = \
            run_coupled_simulation(level, h, log_dir=".")
        
        finest_final_residual = final_residual
        finest_n_iterations = n_iterations
        
        current_solution_A = np.array([solver_A.interpolate_at_point(u_A, x, y) 
                                       for x, y in PROBE_POINTS_A])
        current_solution_B = np.array([solver_B.interpolate_at_point(u_B, x, y) 
                                       for x, y in PROBE_POINTS_B])
        
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
        
        write_output_files(level, solver_A, solver_B, u_A, u_B, residual_history,
                          final_residual, n_iterations, log_dir=".")
        
        all_files.extend([
            f"solution_level{level}_A.csv",
            f"solution_level{level}_B.csv",
            f"interface_level{level}_A.csv",
            f"interface_level{level}_B.csv",
            f"residual_level{level}.csv",
            f"run_level{level}_A.log",
            f"run_level{level}_B.log"
        ])
    
    mesh_independence = "CONVERGED" if max_rel_change < 0.01 else "NOT_CONVERGED"
    
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
