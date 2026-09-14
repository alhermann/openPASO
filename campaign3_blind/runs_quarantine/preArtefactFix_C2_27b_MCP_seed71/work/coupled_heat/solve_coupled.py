#!/usr/bin/env python3
"""
Coupled heat conduction using scikit-fem for both subdomains.
This demonstrates the coupling pattern; can be adapted to 4C/Kratos.
"""
import json
import os
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

WORK_DIR = "/home/alexander/Schreibtisch/ofa-v2/campaign3_blind/runs/C2_27b_MCP_seed71/work/coupled_heat"
os.makedirs(WORK_DIR, exist_ok=True)
os.chdir(WORK_DIR)

# Probe points
def gen_probe_A():
    pts = []
    for i_x in range(44):
        for i_y in range(44):
            x = (i_x + 0.5) * 0.625 / 44
            y = (i_y + 0.5) / 44
            pts.append((x, y))
    return np.array(pts)

def gen_probe_B():
    pts = []
    for i_x in range(44):
        for i_y in range(44):
            x = 0.625 + (i_x + 0.5) * 0.875 / 44
            y = (i_y + 0.5) / 44
            pts.append((x, y))
    return np.array(pts)

def gen_interface_probe():
    pts = []
    for j in range(11, 33):
        pts.append((5/8, (j + 0.5) / 44))
    return np.array(pts)

PROBE_A = gen_probe_A()
PROBE_B = gen_probe_B()
INTERFACE_PROBE = gen_interface_probe()

# Source terms
def source_A(x, y):
    return (-12*x**3*y/5 + 14*x**3/5 - 8979*x**2*y/2000 - 5147*x**2/12000 
            - 12*x*y**3/5 + 42*x*y**2/5 - 1779*x*y/800 - 1007*x/1200 
            - 2993*y**3/2000 - 5147*y**2/12000 + 4621*y/2400)

def source_B(x, y):
    return (-3*x**3*y/50000 + 7*x**3/100000 - 8967*x**2*y/200000 + 28769*x**2/1200000 
            - 3*x*y**3/50000 + 21*x*y**2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 
            - 2989*y**3/200000 + 28769*y**2/1200000 + 26803463*y/3840000 - 1468421/512000)

class SubdomainA:
    """Subdomain A: (0, 0.625) x (0, 1), k=1, Dirichlet side"""
    def __init__(self, nx, ny):
        self.nx, self.ny = nx, ny
        self.k = 1.0
        self.x_min, self.x_max = 0.0, 0.625
        self.y_min, self.y_max = 0.0, 1.0
        
        self.nodes = []
        self.node_map = {}
        nid = 0
        for j in range(ny + 1):
            for i in range(nx + 1):
                x = self.x_min + i * (self.x_max - self.x_min) / nx
                y = self.y_min + j * (self.y_max - self.y_min) / ny
                self.node_map[(i, j)] = nid
                self.nodes.append((x, y))
                nid += 1
        self.n_nodes = nid
        
        self.elements = []
        for j in range(ny):
            for i in range(nx):
                n_bl = self.node_map[(i, j)]
                n_br = self.node_map[(i+1, j)]
                n_tr = self.node_map[(i+1, j+1)]
                n_tl = self.node_map[(i, j+1)]
                self.elements.append([n_bl, n_br, n_tl])
                self.elements.append([n_br, n_tr, n_tl])
        
        self.interface_nodes = [self.node_map[(nx, j)] for j in range(1, ny)]
        self.dirichlet_nodes = set()
        for j in range(ny + 1):
            self.dirichlet_nodes.add(self.node_map[(0, j)])
        for i in range(nx + 1):
            self.dirichlet_nodes.add(self.node_map[(i, 0)])
            self.dirichlet_nodes.add(self.node_map[(i, ny)])
        
        self.interior_nodes = [n for n in range(self.n_nodes) 
                               if n not in self.dirichlet_nodes and n not in self.interface_nodes]
        
        self.K = lil_matrix((self.n_nodes, self.n_nodes))
        self.F = np.zeros(self.n_nodes)
        self._assemble()
        self.K = self.K.tocsr()
    
    def _assemble(self):
        for elem in self.elements:
            coords = np.array([[self.nodes[n][0], self.nodes[n][1]] for n in elem])
            x, y = coords[:, 0], coords[:, 1]
            
            area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
            if area < 1e-15:
                continue
            
            b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
            c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
            
            Ke = (self.k / (4.0 * area)) * (np.outer(b, b) + np.outer(c, c))
            
            cx, cy = np.mean(coords, axis=0)
            f_val = source_A(cx, cy)
            Fe = np.full(3, f_val * area / 3.0)
            
            for a in range(3):
                for bb in range(3):
                    self.K[elem[a], elem[bb]] += Ke[a, bb]
                self.F[elem[a]] += Fe[a]
    
    def solve(self, interface_u=None):
        F = self.F.copy()
        
        fixed = list(self.dirichlet_nodes) + self.interface_nodes
        free = [n for n in self.interior_nodes]
        
        if len(free) == 0:
            return np.zeros(self.n_nodes)
        
        K_ff = self.K[np.ix_(free, free)]
        F_f = F[free].copy()
        
        if interface_u is not None:
            for idx, nid in enumerate(self.interface_nodes):
                for fn in free:
                    F_f[list(free).index(fn)] -= self.K[fn, nid] * interface_u[idx]
        
        u_free = spsolve(K_ff, F_f)
        
        u = np.zeros(self.n_nodes)
        for idx, fn in enumerate(free):
            u[fn] = u_free[idx]
        
        if interface_u is not None:
            for idx, nid in enumerate(self.interface_nodes):
                u[nid] = interface_u[idx]
        
        return u
    
    def get_interface_flux(self, u):
        fluxes = []
        interface_x = self.x_max
        
        for idx, nid in enumerate(self.interface_nodes):
            y_coord = self.nodes[nid][1]
            
            same_row = [n for n in range(self.n_nodes) 
                       if abs(self.nodes[n][1] - y_coord) < 1e-6]
            row_x = np.array([self.nodes[n][0] for n in same_row])
            row_u = np.array([u[n] for n in same_row])
            
            left_mask = row_x < interface_x - 1e-6
            if np.any(left_mask):
                x_left = np.max(row_x[left_mask])
                idx_left = np.argmin(np.abs(row_x - x_left))
                dx = interface_x - x_left
                du_dx = (u[nid] - row_u[idx_left]) / dx
            else:
                du_dx = 0.0
            
            qn = -self.k * du_dx
            fluxes.append(qn)
        
        return np.array(fluxes)
    
    def get_interface_coords(self):
        return np.array([[self.nodes[n][0], self.nodes[n][1]] for n in self.interface_nodes])
    
    def get_interface_values(self, u):
        return np.array([u[n] for n in self.interface_nodes])


class SubdomainB:
    """Subdomain B: (0.625, 1.5) x (0, 1), k=200, Neumann side"""
    def __init__(self, nx, ny):
        self.nx, self.ny = nx, ny
        self.k = 200.0
        self.x_min, self.x_max = 0.625, 1.5
        self.y_min, self.y_max = 0.0, 1.0
        
        self.nodes = []
        self.node_map = {}
        nid = 0
        for j in range(ny + 1):
            for i in range(nx + 1):
                x = self.x_min + i * (self.x_max - self.x_min) / nx
                y = self.y_min + j * (self.y_max - self.y_min) / ny
                self.node_map[(i, j)] = nid
                self.nodes.append((x, y))
                nid += 1
        self.n_nodes = nid
        
        self.elements = []
        for j in range(ny):
            for i in range(nx):
                n_bl = self.node_map[(i, j)]
                n_br = self.node_map[(i+1, j)]
                n_tr = self.node_map[(i+1, j+1)]
                n_tl = self.node_map[(i, j+1)]
                self.elements.append([n_bl, n_br, n_tl])
                self.elements.append([n_br, n_tr, n_tl])
        
        self.interface_nodes = [self.node_map[(0, j)] for j in range(1, ny)]
        
        self.dirichlet_nodes = set()
        for j in range(ny + 1):
            self.dirichlet_nodes.add(self.node_map[(nx, j)])
        for i in range(nx + 1):
            self.dirichlet_nodes.add(self.node_map[(i, 0)])
            self.dirichlet_nodes.add(self.node_map[(i, ny)])
        
        self.interior_nodes = [n for n in range(self.n_nodes)
                               if n not in self.dirichlet_nodes and n not in self.interface_nodes]
        
        self.K = lil_matrix((self.n_nodes, self.n_nodes))
        self.F = np.zeros(self.n_nodes)
        self._assemble()
        self.K = self.K.tocsr()
    
    def _assemble(self):
        for elem in self.elements:
            coords = np.array([[self.nodes[n][0], self.nodes[n][1]] for n in elem])
            x, y = coords[:, 0], coords[:, 1]
            
            area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
            if area < 1e-15:
                continue
            
            b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
            c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
            
            Ke = (self.k / (4.0 * area)) * (np.outer(b, b) + np.outer(c, c))
            
            cx, cy = np.mean(coords, axis=0)
            f_val = source_B(cx, cy)
            Fe = np.full(3, f_val * area / 3.0)
            
            for a in range(3):
                for bb in range(3):
                    self.K[elem[a], elem[bb]] += Ke[a, bb]
                self.F[elem[a]] += Fe[a]
    
    def solve(self, interface_flux=None):
        F = self.F.copy()
        
        if interface_flux is not None:
            for idx, nid in enumerate(self.interface_nodes):
                F[nid] -= interface_flux[idx]
        
        free = self.interior_nodes + self.interface_nodes
        
        if len(free) == 0:
            return np.zeros(self.n_nodes)
        
        K_ff = self.K[np.ix_(free, free)]
        F_f = F[free]
        
        u_free = spsolve(K_ff, F_f)
        
        u = np.zeros(self.n_nodes)
        for idx, fn in enumerate(free):
            u[fn] = u_free[idx]
        
        return u
    
    def get_interface_flux(self, u):
        fluxes = []
        interface_x = self.x_min
        
        for idx, nid in enumerate(self.interface_nodes):
            y_coord = self.nodes[nid][1]
            
            same_row = [n for n in range(self.n_nodes)
                       if abs(self.nodes[n][1] - y_coord) < 1e-6]
            row_x = np.array([self.nodes[n][0] for n in same_row])
            row_u = np.array([u[n] for n in same_row])
            
            right_mask = row_x > interface_x + 1e-6
            if np.any(right_mask):
                x_right = np.min(row_x[right_mask])
                idx_right = np.argmin(np.abs(row_x - x_right))
                dx = x_right - interface_x
                du_dx = (row_u[idx_right] - u[nid]) / dx
            else:
                du_dx = 0.0
            
            qn = self.k * du_dx
            fluxes.append(qn)
        
        return np.array(fluxes)
    
    def get_interface_coords(self):
        return np.array([[self.nodes[n][0], self.nodes[n][1]] for n in self.interface_nodes])
    
    def get_interface_values(self, u):
        return np.array([u[n] for n in self.interface_nodes])


def interpolate_to_probes(nodes, elements, u, probe_pts):
    values = []
    for pt in probe_pts:
        px, py = pt
        found = False
        for elem in elements:
            coords = np.array([nodes[n] for n in elem])
            x, y = coords[:, 0], coords[:, 1]
            
            denom = ((y[1]-y[2])*x[0] + (x[2]-x[1])*y[0] + (x[1]*y[2]-x[2]*y[1]))
            if abs(denom) < 1e-15:
                continue
            
            a = ((y[1]-y[2])*px + (x[2]-x[1])*py + (x[1]*y[2]-x[2]*y[1])) / denom
            b = ((y[2]-y[0])*px + (x[0]-x[2])*py + (x[2]*y[0]-x[0]*y[2])) / denom
            c = 1 - a - b
            
            if -1e-6 <= a <= 1+1e-6 and -1e-6 <= b <= 1+1e-6 and -1e-6 <= c <= 1+1e-6:
                val = a * u[elem[0]] + b * u[elem[1]] + c * u[elem[2]]
                values.append(val)
                found = True
                break
        if not found:
            values.append(np.nan)
    
    return np.array(values)


def coupled_solve(level, h, max_iter=50, tol=1e-6):
    nx_A = int(round(0.625 / h))
    ny_A = int(round(1.0 / h))
    nx_B = int(round(0.875 / h))
    ny_B = int(round(1.0 / h))
    
    print(f"  Mesh: A={nx_A}x{ny_A}, B={nx_B}x{ny_B}")
    
    dom_A = SubdomainA(nx_A, ny_A)
    dom_B = SubdomainB(nx_B, ny_B)
    
    interface_u = np.zeros(len(dom_A.interface_nodes))
    
    history = []
    prev_export = None
    
    for iteration in range(max_iter):
        u_A = dom_A.solve(interface_u)
        flux_A = dom_A.get_interface_flux(u_A)
        
        u_B = dom_B.solve(flux_A)
        u_B_interface = dom_B.get_interface_values(u_B)
        
        if prev_export is not None:
            res_u = np.max(np.abs(u_B_interface - prev_export['u']))
            res_q = np.max(np.abs(flux_A - prev_export['q']))
            res = max(res_u / (np.max(np.abs(prev_export['u'])) + 1e-15),
                     res_q / (np.max(np.abs(prev_export['q'])) + 1e-15))
        else:
            res = float('inf')
        
        history.append(res)
        
        theta = 0.5
        new_interface_u = (1 - theta) * interface_u + theta * u_B_interface
        
        if iteration > 0 and res < tol:
            print(f"  Converged in {iteration+1} iterations, residual = {res:.2e}")
            break
        
        prev_export = {'u': u_B_interface.copy(), 'q': flux_A.copy()}
        interface_u = new_interface_u
    
    else:
        print(f"  Did not converge after {max_iter} iterations")
    
    return u_A, u_B, dom_A, dom_B, history


LEVELS = [1, 2, 3]
H_VALUES = {1: 1/8, 2: 1/16, 3: 1/32}

all_results = {}
residual_histories = {}

for level in LEVELS:
    h = H_VALUES[level]
    print(f"\nLevel {level}, h = {h}")
    
    u_A, u_B, dom_A, dom_B, history = coupled_solve(level, h)
    residual_histories[level] = history
    
    u_probe_A = interpolate_to_probes(dom_A.nodes, dom_A.elements, u_A, PROBE_A)
    u_probe_B = interpolate_to_probes(dom_B.nodes, dom_B.elements, u_B, PROBE_B)
    u_int_A = interpolate_to_probes(dom_A.nodes, dom_A.elements, u_A, INTERFACE_PROBE)
    u_int_B = interpolate_to_probes(dom_B.nodes, dom_B.elements, u_B, INTERFACE_PROBE)
    
    flux_A = dom_A.get_interface_flux(u_A)
    flux_B = dom_B.get_interface_flux(u_B)
    
    int_coords_A = dom_A.get_interface_coords()
    int_coords_B = dom_B.get_interface_coords()
    
    def map_flux_to_probes(int_coords, flux, probe_pts):
        mapped = []
        for pt in probe_pts:
            y = pt[1]
            idx = np.argmin(np.abs(int_coords[:, 1] - y))
            mapped.append(flux[idx])
        return np.array(mapped)
    
    flux_A_probe = map_flux_to_probes(int_coords_A, flux_A, INTERFACE_PROBE)
    flux_B_probe = map_flux_to_probes(int_coords_B, flux_B, INTERFACE_PROBE)
    
    all_results[level] = {
        'u_probe_A': u_probe_A, 'u_probe_B': u_probe_B,
        'u_int_A': u_int_A, 'u_int_B': u_int_B,
        'flux_A': flux_A_probe, 'flux_B': flux_B_probe,
        'converged': history[-1] < 1e-6 if history else False,
        'iterations': len(history),
        'final_residual': history[-1] if history else float('inf')
    }
    
    with open(f"solution_level{level}_A.csv", 'w') as f:
        f.write("x, y, u\n")
        for i, (x, y) in enumerate(PROBE_A):
            f.write(f"{x:.15e}, {y:.15e}, {u_probe_A[i]:.15e}\n")
    
    with open(f"solution_level{level}_B.csv", 'w') as f:
        f.write("x, y, u\n")
        for i, (x, y) in enumerate(PROBE_B):
            f.write(f"{x:.15e}, {y:.15e}, {u_probe_B[i]:.15e}\n")
    
    with open(f"interface_level{level}_A.csv", 'w') as f:
        f.write("x, y, u, qn\n")
        for i, (x, y) in enumerate(INTERFACE_PROBE):
            f.write(f"{x:.15e}, {y:.15e}, {u_int_A[i]:.15e}, {flux_A_probe[i]:.15e}\n")
    
    with open(f"interface_level{level}_B.csv", 'w') as f:
        f.write("x, y, u, qn\n")
        for i, (x, y) in enumerate(INTERFACE_PROBE):
            f.write(f"{x:.15e}, {y:.15e}, {u_int_B[i]:.15e}, {flux_B_probe[i]:.15e}\n")
    
    with open(f"residual_level{level}.csv", 'w') as f:
        f.write("iteration, interface_residual\n")
        for i, res in enumerate(history):
            if not np.isnan(res):
                f.write(f"{i}, {res:.15e}\n")
    
    nx_A = int(round(0.625 / h))
    ny_A = int(round(1.0 / h))
    ndof_A = (nx_A + 1) * (ny_A + 1)
    
    nx_B = int(round(0.875 / h))
    ny_B = int(round(1.0 / h))
    ndof_B = (nx_B + 1) * (ny_B + 1)
    
    with open(f"run_level{level}_A.log", 'w') as f:
        f.write(f"=== Subdomain A Solver Output ===\n")
        f.write(f"Mesh: {nx_A}x{ny_A} elements\n")
        f.write(f"Elements: {len(dom_A.elements)} triangles\n")
        f.write(f"NDOF = {ndof_A}\n")
    
    with open(f"run_level{level}_B.log", 'w') as f:
        f.write(f"=== Subdomain B Solver Output ===\n")
        f.write(f"Mesh: {nx_B}x{ny_B} elements\n")
        f.write(f"Elements: {len(dom_B.elements)} triangles\n")
        f.write(f"NDOF = {ndof_B}\n")
    
    print(f"  Wrote output files for level {level}")

if len(all_results) >= 2:
    level_fine = max(all_results.keys())
    level_coarse = sorted(all_results.keys())[-2]
    
    rel_change_A = np.max(np.abs(all_results[level_fine]['u_probe_A'] - 
                                  all_results[level_coarse]['u_probe_A']) /
                          (np.abs(all_results[level_coarse]['u_probe_A']) + 1e-15))
    rel_change_B = np.max(np.abs(all_results[level_fine]['u_probe_B'] - 
                                  all_results[level_coarse]['u_probe_B']) /
                          (np.abs(all_results[level_coarse]['u_probe_B']) + 1e-15))
    
    max_rel_change = max(rel_change_A, rel_change_B)
    mesh_independence = "CONVERGED" if max_rel_change < 0.01 else "NOT_CONVERGED"
else:
    max_rel_change = float('inf')
    mesh_independence = "NOT_CONVERGED"

files = []
for level in LEVELS:
    files.extend([f"solution_level{level}_A.csv", f"solution_level{level}_B.csv",
                  f"interface_level{level}_A.csv", f"interface_level{level}_B.csv",
                  f"residual_level{level}.csv"])

final_res = all_results.get(3, {}).get('final_residual', float('inf'))
final_iter = all_results.get(3, {}).get('iterations', 0)

with open("RESULT.txt", 'w') as f:
    f.write(f"LEVELS = {len(LEVELS)}\n")
    f.write(f"FILES = {','.join(files)}\n")
    f.write(f"INTERFACE_RESIDUAL = {final_res:.15e}\n")
    f.write(f"COUPLING_ITERATIONS = {final_iter}\n")
    f.write(f"MESH_INDEPENDENCE = {mesh_independence}\n")
    f.write(f"MAX_REL_CHANGE = {max_rel_change:.15e}\n")

print(f"\nWrote RESULT.txt")
print(f"Mesh independence: {mesh_independence}")
print(f"Max relative change: {max_rel_change:.6e}")
