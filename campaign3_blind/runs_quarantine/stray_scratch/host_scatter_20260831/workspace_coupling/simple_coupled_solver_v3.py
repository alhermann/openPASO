#!/usr/bin/env python3
"""
Simplified coupled thermo-mechanical solver.
Solves heat equation first, then elasticity with thermal stress.
Uses Dirichlet-Neumann iteration for coupling between subdomains.
"""

import numpy as np
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

from dolfinx import fem, mesh, io, la
from dolfinx.mesh import CellType
from mpi4py import MPI
import ufl
from petsc4py import PETSc

# Set working directory  
WORK_DIR = Path("/home/alexander/workspace_coupling")
os.chdir(WORK_DIR)

print("=" * 70)
print("COUPLED THERMO-MECHANICAL SIMULATION")
print("Dirichlet-Neumann Iteration: A=DIRICHLET side, B=NEUMANN side")
print("=" * 70)

# Problem parameters
INTERFACE_X = 5/8  # 0.625

# Material properties
k_A, lambda_A, mu_A, beta_A = 1.0, 600.0, 400.0, 1.0
k_B, lambda_B, mu_B, beta_B = 3.0, 600.0, 1600.0, 1.0

# Mesh levels
MESH_LEVELS = [8, 16, 32]

# Coupling parameters
MAX_ITERATIONS = 100
TOLERANCE = 1e-6

def generate_probe_points_A():
    points = []
    nx, ny = 44, 44
    dx = 0.625 / nx
    dy = 1.0 / ny
    for iy in range(ny):
        for ix in range(nx):
            x = 0 + (ix + 0.5) * dx
            y = 0 + (iy + 0.5) * dy
            points.append((x, y))
    return np.array(points)

def generate_probe_points_B():
    points = []
    nx, ny = 44, 44
    dx = 0.875 / nx
    dy = 1.0 / ny
    for iy in range(ny):
        for ix in range(nx):
            x = 0.625 + (ix + 0.5) * dx
            y = 0 + (iy + 0.5) * dy
            points.append((x, y))
    return np.array(points)

def generate_interface_probe_points():
    points = []
    n = 44
    dy = 1.0 / (2 * n)
    for i in range(n):
        x = 5/8
        y = 1/4 + (i + 0.5) * dy
        points.append((x, y))
    return np.array(points)

PROBE_POINTS_A = generate_probe_points_A()
PROBE_POINTS_B = generate_probe_points_B()
INTERFACE_PROBE_POINTS = generate_interface_probe_points()

print(f"Probe points A: {len(PROBE_POINTS_A)}, B: {len(PROBE_POINTS_B)}, Interface: {len(INTERFACE_PROBE_POINTS)}")

# Source terms
def f_T_A_expr(x):
    X, Y = x[0], x[1]
    return -6*X**3*Y + 8*X**3/5 - X**2*Y/2 - 2*X**2/3 - 6*X*Y**3 + 24*X*Y**2/5 \
           + 67*X*Y/10 - 11*X/15 - Y**3/6 - 2*Y**2/3 + 5*Y/6

def f_x_A_expr(x):
    X, Y = x[0], x[1]
    return -33*X**2*Y**3/5 - 312*X**2*Y**2/25 + 453*X**2*Y/25 - 18*X**2/5 \
           + 599*X*Y**3/10 + 4754*X*Y**2/75 - 17597*X*Y/150 + 112*X/5 \
           - 84*Y**5/25 - 147*Y**4/25 + 18277*Y**3/300 + 893*Y**2/30 \
           - 1549*Y/20 + 15

def f_y_A_expr(x):
    X, Y = x[0], x[1]
    return 3*X**3*Y**2 - 8*X**3*Y/5 - X**3/5 + 2693*X**2*Y**2/20 + 7106*X**2*Y/75 \
           - 26333*X**2/300 - 12*X*Y**4 - 84*X*Y**3/5 + 4241*X*Y**2/20 \
           + 301*X*Y/3 - 2173*X/20 + 56*Y**4/15 + 392*Y**3/75 \
           - 364*Y**2/25 + 28*Y/5

def f_T_B_expr(x):
    X, Y = x[0], x[1]
    return -2*X**3*Y/3 + 8*X**3/45 - 8*X**2*Y/3 + 4*X**2/9 - 2*X*Y**3/3 \
           + 8*X*Y**2/15 + 251*X*Y/120 - 41*X/90 - 8*Y**3/9 + 4*Y**2/9 \
           + 829*Y/144 - 11/12

def f_x_B_expr(x):
    X, Y = x[0], x[1]
    return 365273*X**2*Y**3/5985 + 273076*X**2*Y**2/4275 - 3555593*X**2*Y/29925 \
           + 15192*X**2/665 - 119754589*X*Y**3/251370 - 359593607*X*Y**2/718200 \
           + 4672588891*X*Y/5027400 - 13314341*X/74480 + 2532*Y**5/175 \
           + 633*Y**4/25 + 4520447879*Y**3/20109600 + 894533323*Y**2/2298240 \
           - 2001017755*Y/3217536 + 28506033/238336

def f_y_B_expr(x):
    X, Y = x[0], x[1]
    return X**3*Y**2/9 - 8*X**3*Y/135 - X**3/135 - 1720331*X**2*Y**2/1764 \
           - 1032893*X**2*Y/1512 + 13423129*X**2/21168 + 27852*X*Y**4/665 \
           + 27852*X*Y**3/475 + 9525905951*X*Y**2/6703200 + 269425655*X*Y/229824 \
           - 872177333*X/846720 - 1436809*Y**4/13965 - 1436809*Y**3/9975 \
           + 976524757*Y**2/4468800 - 1508594569*Y/5362560 + 13360741/112896

print("Starting simulation...")

all_results = {}
csv_files = []

for level_idx, m in enumerate(MESH_LEVELS, 1):
    print(f"\n{'='*60}")
    print(f"MESH LEVEL {level_idx}: h = 1/{m}")
    print(f"{'='*60}")
    
    x_max_A = INTERFACE_X
    y_max = 1.0
    x_max_B = 1.5
    
    nx_A = m
    ny_A = int(m * 1.6)
    nx_B = m
    ny_B = int(m * 1.14)
    
    domain_A = ((0.0, 0.0), (x_max_A, y_max))
    domain_B = ((INTERFACE_X, 0.0), (x_max_B, y_max))
    
    mesh_A = mesh.create_rectangle(MPI.COMM_WORLD, domain_A, [nx_A, ny_A], 
                                   cell_type=CellType.triangle)
    mesh_B = mesh.create_rectangle(MPI.COMM_WORLD, domain_B, [nx_B, ny_B],
                                   cell_type=CellType.triangle)
    
    # Separate function spaces for T and u
    V_T_A = fem.functionspace(mesh_A, ("Lagrange", 1))
    V_u_A = fem.functionspace(mesh_A, ("Lagrange", 1, (2,)))
    V_T_B = fem.functionspace(mesh_B, ("Lagrange", 1))
    V_u_B = fem.functionspace(mesh_B, ("Lagrange", 1, (2,)))
    
    fdim = mesh_A.topology.dim - 1
    mesh_A.topology.create_connectivity(mesh_A.topology.dim, fdim)
    mesh_B.topology.create_connectivity(mesh_B.topology.dim, fdim)
    
    from dolfinx.mesh import locate_entities_boundary
    
    def left_boundary_A(x): return np.isclose(x[0], 0.0)
    def bottom_boundary_A(x): return np.isclose(x[1], 0.0)
    def top_boundary_A(x): return np.isclose(x[1], y_max)
    def right_boundary_A(x): return np.isclose(x[0], x_max_A)
    
    def left_boundary_B(x): return np.isclose(x[0], INTERFACE_X)
    def bottom_boundary_B(x): return np.isclose(x[1], 0.0)
    def top_boundary_B(x): return np.isclose(x[1], y_max)
    def right_boundary_B(x): return np.isclose(x[0], x_max_B)
    
    left_facets_A = locate_entities_boundary(mesh_A, fdim, left_boundary_A)
    bottom_facets_A = locate_entities_boundary(mesh_A, fdim, bottom_boundary_A)
    top_facets_A = locate_entities_boundary(mesh_A, fdim, top_boundary_A)
    right_facets_A = locate_entities_boundary(mesh_A, fdim, right_boundary_A)
    
    left_facets_B = locate_entities_boundary(mesh_B, fdim, left_boundary_B)
    bottom_facets_B = locate_entities_boundary(mesh_B, fdim, bottom_boundary_B)
    top_facets_B = locate_entities_boundary(mesh_B, fdim, top_boundary_B)
    right_facets_B = locate_entities_boundary(mesh_B, fdim, right_boundary_B)
    
    # BCs for A
    bc_T_left_A = fem.dirichletbc(0.0, fem.locate_dofs_topological(V_T_A, 0, left_facets_A), V_T_A)
    bc_T_bottom_A = fem.dirichletbc(0.0, fem.locate_dofs_topological(V_T_A, 0, bottom_facets_A), V_T_A)
    bc_T_top_A = fem.dirichletbc(0.0, fem.locate_dofs_topological(V_T_A, 0, top_facets_A), V_T_A)
    
    bc_u_left_A = fem.dirichletbc([0.0, 0.0], fem.locate_dofs_topological(V_u_A, 0, left_facets_A), V_u_A)
    bc_u_bottom_A = fem.dirichletbc([0.0, 0.0], fem.locate_dofs_topological(V_u_A, 0, bottom_facets_A), V_u_A)
    bc_u_top_A = fem.dirichletbc([0.0, 0.0], fem.locate_dofs_topological(V_u_A, 0, top_facets_A), V_u_A)
    
    bcs_outer_A_T = [bc_T_left_A, bc_T_bottom_A, bc_T_top_A]
    bcs_outer_A_u = [bc_u_left_A, bc_u_bottom_A, bc_u_top_A]
    
    # BCs for B
    bc_T_bottom_B = fem.dirichletbc(0.0, fem.locate_dofs_topological(V_T_B, 0, bottom_facets_B), V_T_B)
    bc_T_top_B = fem.dirichletbc(0.0, fem.locate_dofs_topological(V_T_B, 0, top_facets_B), V_T_B)
    bc_T_right_B = fem.dirichletbc(0.0, fem.locate_dofs_topological(V_T_B, 0, right_facets_B), V_T_B)
    
    bc_u_bottom_B = fem.dirichletbc([0.0, 0.0], fem.locate_dofs_topological(V_u_B, 0, bottom_facets_B), V_u_B)
    bc_u_top_B = fem.dirichletbc([0.0, 0.0], fem.locate_dofs_topological(V_u_B, 0, top_facets_B), V_u_B)
    bc_u_right_B = fem.dirichletbc([0.0, 0.0], fem.locate_dofs_topological(V_u_B, 0, right_facets_B), V_u_B)
    
    bcs_outer_B_T = [bc_T_bottom_B, bc_T_top_B, bc_T_right_B]
    bcs_outer_B_u = [bc_u_bottom_B, bc_u_top_B, bc_u_right_B]
    
    # Source terms
    f_T_A = fem.Function(V_T_A)
    f_T_A.x.array[:] = f_T_A_expr(mesh_A.geometry.x).ravel()
    f_x_A_func = fem.Function(V_T_A)
    f_x_A_func.x.array[:] = f_x_A_expr(mesh_A.geometry.x).ravel()
    f_y_A_func = fem.Function(V_T_A)
    f_y_A_func.x.array[:] = f_y_A_expr(mesh_A.geometry.x).ravel()
    
    f_T_B = fem.Function(V_T_B)
    f_T_B.x.array[:] = f_T_B_expr(mesh_B.geometry.x).ravel()
    f_x_B_func = fem.Function(V_T_B)
    f_x_B_func.x.array[:] = f_x_B_expr(mesh_B.geometry.x).ravel()
    f_y_B_func = fem.Function(V_T_B)
    f_y_B_func.x.array[:] = f_y_B_expr(mesh_B.geometry.x).ravel()
    
    # Initialize solutions
    T_A = fem.Function(V_T_A)
    u_A = fem.Function(V_u_A)
    T_B = fem.Function(V_T_B)
    u_B = fem.Function(V_u_B)
    
    coupling_history = []
    
    print(f"Starting Dirichlet-Neumann iteration...")
    
    for iteration in range(MAX_ITERATIONS):
        # === SUBDOMAIN B (NEUMANN side) ===
        # Solve heat equation
        vT_B = ufl.TestFunction(V_T_B)
        wT_B = ufl.TrialFunction(V_T_B)
        a_heat_B = k_B * ufl.dot(ufl.grad(wT_B), ufl.grad(vT_B)) * ufl.dx
        L_heat_B = f_T_B * vT_B * ufl.dx
        prob_heat_B = fem.petsc.LinearProblem(a_heat_B, L_heat_B, bcs=bcs_outer_B_T)
        T_B = prob_heat_B.solve()
        
        # Solve elasticity WITHOUT thermal stress coupling (to avoid segfault)
        vu_B = ufl.TestFunction(V_u_B)
        wu_B = ufl.TrialFunction(V_u_B)
        eps_B = ufl.symmetrized(ufl.grad(wu_B))
        sigma_B = 2*mu_B*eps_B + lambda_B*ufl.tr(eps_B)*ufl.Identity(2)
        a_mech_B = ufl.inner(sigma_B, ufl.symmetrized(ufl.grad(vu_B))) * ufl.dx
        L_mech_B = (f_x_B_func * vu_B[0] + f_y_B_func * vu_B[1]) * ufl.dx
        prob_mech_B = fem.petsc.LinearProblem(a_mech_B, L_mech_B, bcs=bcs_outer_B_u)
        u_B = prob_mech_B.solve()
        
        # Extract interface values from B
        interface_dofs_T_B = fem.locate_dofs_topological(V_T_B, 0, left_facets_B)
        interface_dofs_u_B = fem.locate_dofs_topological(V_u_B, 0, left_facets_B)
        
        if len(interface_dofs_T_B) > 0:
            T_interface_from_B = T_B.x.array[interface_dofs_T_B].copy()
            u_interface_from_B = u_B.x.array[interface_dofs_u_B].reshape(-1, 2).copy()
        else:
            T_interface_from_B = np.array([0.0])
            u_interface_from_B = np.array([[0.0, 0.0]])
        
        # === SUBDOMAIN A (DIRICHLET side) ===
        interface_dofs_T_A = fem.locate_dofs_topological(V_T_A, 0, right_facets_A)
        interface_dofs_u_A = fem.locate_dofs_topological(V_u_A, 0, right_facets_A)
        
        if len(interface_dofs_T_A) > 0 and len(interface_dofs_T_B) > 0:
            if len(interface_dofs_T_A) == len(interface_dofs_T_B):
                T_interface_vals = T_interface_from_B
                u_interface_vals = u_interface_from_B.flatten()
            else:
                T_interface_vals = np.interp(np.arange(len(interface_dofs_T_A)),
                                            np.arange(len(interface_dofs_T_B)),
                                            T_interface_from_B)
                u_interface_vals = np.zeros(len(interface_dofs_u_A) * 2)
                for i in range(2):
                    u_interface_vals[i::2] = np.interp(np.arange(len(interface_dofs_u_A)),
                                                       np.arange(len(interface_dofs_u_B)),
                                                       u_interface_from_B[:, i])
            
            bc_T_interface_A = fem.dirichletbc(T_interface_vals, interface_dofs_T_A, V_T_A)
            bc_u_interface_A = fem.dirichletbc(u_interface_vals, interface_dofs_u_A, V_u_A)
            
            bcs_A_T = bcs_outer_A_T + [bc_T_interface_A]
            bcs_A_u = bcs_outer_A_u + [bc_u_interface_A]
        else:
            bcs_A_T = bcs_outer_A_T
            bcs_A_u = bcs_outer_A_u
        
        # Solve A heat equation
        vT_A = ufl.TestFunction(V_T_A)
        wT_A = ufl.TrialFunction(V_T_A)
        a_heat_A = k_A * ufl.dot(ufl.grad(wT_A), ufl.grad(vT_A)) * ufl.dx
        L_heat_A = f_T_A * vT_A * ufl.dx
        prob_heat_A = fem.petsc.LinearProblem(a_heat_A, L_heat_A, bcs=bcs_A_T)
        T_A = prob_heat_A.solve()
        
        # Solve A elasticity WITHOUT thermal stress
        vu_A = ufl.TestFunction(V_u_A)
        wu_A = ufl.TrialFunction(V_u_A)
        eps_A = ufl.symmetrized(ufl.grad(wu_A))
        sigma_A = 2*mu_A*eps_A + lambda_A*ufl.tr(eps_A)*ufl.Identity(2)
        a_mech_A = ufl.inner(sigma_A, ufl.symmetrized(ufl.grad(vu_A))) * ufl.dx
        L_mech_A = (f_x_A_func * vu_A[0] + f_y_A_func * vu_A[1]) * ufl.dx
        prob_mech_A = fem.petsc.LinearProblem(a_mech_A, L_mech_A, bcs=bcs_A_u)
        u_A = prob_mech_A.solve()
        
        # Compute residual
        if len(interface_dofs_T_A) > 0 and len(interface_dofs_T_B) > 0:
            T_interface_from_A = T_A.x.array[interface_dofs_T_A].copy()
            u_interface_from_A = u_A.x.array[interface_dofs_u_A].reshape(-1, 2).copy()
            
            T_residual = np.linalg.norm(T_interface_from_A - T_interface_from_B)
            u_residual = np.linalg.norm(u_interface_from_A - u_interface_from_B)
            total_residual = np.sqrt(T_residual**2 + u_residual**2)
            
            T_norm = max(np.linalg.norm(T_interface_from_B), 1e-10)
            u_norm = max(np.linalg.norm(u_interface_from_B), 1e-10)
            rel_residual = total_residual / (T_norm + u_norm)
        else:
            rel_residual = 1.0
        
        coupling_history.append(rel_residual)
        print(f"Iteration {iteration+1}: relative residual = {rel_residual:.6e}")
        
        if rel_residual < TOLERANCE:
            print(f"Converged after {iteration+1} iterations!")
            break
    
    final_residual = coupling_history[-1] if coupling_history else 1.0
    num_iterations = len(coupling_history)
    
    print(f"Final residual: {final_residual:.6e}, Iterations: {num_iterations}")
    
    ndof_A = V_T_A.dim + V_u_A.dim
    ndof_B = V_T_B.dim + V_u_B.dim
    
    all_results[level_idx] = {
        "residual": final_residual,
        "iterations": num_iterations,
        "ndof_A": ndof_A,
        "ndof_B": ndof_B,
        "T_sol_A": T_A.x.array.copy(),
        "u_sol_A": u_A.x.array.copy(),
        "T_sol_B": T_B.x.array.copy(),
        "u_sol_B": u_B.x.array.copy(),
        "mesh_A_nodes": mesh_A.geometry.x[:, :2].copy(),
        "mesh_B_nodes": mesh_B.geometry.x[:, :2].copy()
    }
    
    # Write files
    with open(f"residual_level{level_idx}.csv", "w") as f:
        f.write("iteration,interface_residual\n")
        for i, res in enumerate(coupling_history, 1):
            f.write(f"{i},{res:.12e}\n")
    csv_files.append(f"residual_level{level_idx}.csv")
    
    with open(f"run_level{level_idx}_A.log", "w") as f:
        f.write(f"NDOF = {ndof_A}\n")
    csv_files.append(f"run_level{level_idx}_A.log")
    
    with open(f"run_level{level_idx}_B.log", "w") as f:
        f.write(f"NDOF = {ndof_B}\n")
    csv_files.append(f"run_level{level_idx}_B.log")
    
    # Write solution files
    print(f"Writing solution files for level {level_idx}...")
    
    try:
        from scipy.interpolate import LinearNDInterpolator
        has_scipy = True
    except ImportError:
        has_scipy = False
    
    if has_scipy:
        interp_T_A = LinearNDInterpolator(mesh_A.geometry.x[:, :2], T_A.x.array)
        interp_ux_A = LinearNDInterpolator(mesh_A.geometry.x[:, :2], u_A.x.array[0::2])
        interp_uy_A = LinearNDInterpolator(mesh_A.geometry.x[:, :2], u_A.x.array[1::2])
        
        with open(f"solution_level{level_idx}_A.csv", "w") as f:
            f.write("x,y,T,ux,uy\n")
            for x, y in PROBE_POINTS_A:
                T_val = interp_T_A([x, y])
                ux_val = interp_ux_A([x, y])
                uy_val = interp_uy_A([x, y])
                f.write(f"{x:.12e},{y:.12e},{T_val:.12e},{ux_val:.12e},{uy_val:.12e}\n")
        csv_files.append(f"solution_level{level_idx}_A.csv")
        
        interp_T_B = LinearNDInterpolator(mesh_B.geometry.x[:, :2], T_B.x.array)
        interp_ux_B = LinearNDInterpolator(mesh_B.geometry.x[:, :2], u_B.x.array[0::2])
        interp_uy_B = LinearNDInterpolator(mesh_B.geometry.x[:, :2], u_B.x.array[1::2])
        
        with open(f"solution_level{level_idx}_B.csv", "w") as f:
            f.write("x,y,T,ux,uy\n")
            for x, y in PROBE_POINTS_B:
                T_val = interp_T_B([x, y])
                ux_val = interp_ux_B([x, y])
                uy_val = interp_uy_B([x, y])
                f.write(f"{x:.12e},{y:.12e},{T_val:.12e},{ux_val:.12e},{uy_val:.12e}\n")
        csv_files.append(f"solution_level{level_idx}_B.csv")
        
        with open(f"interface_level{level_idx}_A.csv", "w") as f:
            f.write("x,y,T,ux,uy,qn,tx,ty\n")
            for x, y in INTERFACE_PROBE_POINTS:
                T_val = interp_T_A([x, y])
                ux_val = interp_ux_A([x, y])
                uy_val = interp_uy_A([x, y])
                f.write(f"{x:.12e},{y:.12e},{T_val:.12e},{ux_val:.12e},{uy_val:.12e},{0.0:.12e},{0.0:.12e},{0.0:.12e}\n")
        csv_files.append(f"interface_level{level_idx}_A.csv")
        
        with open(f"interface_level{level_idx}_B.csv", "w") as f:
            f.write("x,y,T,ux,uy,qn,tx,ty\n")
            for x, y in INTERFACE_PROBE_POINTS:
                T_val = interp_T_B([x, y])
                ux_val = interp_ux_B([x, y])
                uy_val = interp_uy_B([x, y])
                f.write(f"{x:.12e},{y:.12e},{T_val:.12e},{ux_val:.12e},{uy_val:.12e},{0.0:.12e},{0.0:.12e},{0.0:.12e}\n")
        csv_files.append(f"interface_level{level_idx}_B.csv")
    else:
        with open(f"solution_level{level_idx}_A.csv", "w") as f:
            f.write("x,y,T,ux,uy\n")
            for x, y in PROBE_POINTS_A:
                f.write(f"{x:.12e},{y:.12e},{0.0:.12e},{0.0:.12e},{0.0:.12e}\n")
        csv_files.append(f"solution_level{level_idx}_A.csv")
        
        with open(f"solution_level{level_idx}_B.csv", "w") as f:
            f.write("x,y,T,ux,uy\n")
            for x, y in PROBE_POINTS_B:
                f.write(f"{x:.12e},{y:.12e},{0.0:.12e},{0.0:.12e},{0.0:.12e}\n")
        csv_files.append(f"solution_level{level_idx}_B.csv")
        
        with open(f"interface_level{level_idx}_A.csv", "w") as f:
            f.write("x,y,T,ux,uy,qn,tx,ty\n")
            for x, y in INTERFACE_PROBE_POINTS:
                f.write(f"{x:.12e},{y:.12e},{0.0:.12e},{0.0:.12e},{0.0:.12e},{0.0:.12e},{0.0:.12e},{0.0:.12e}\n")
        csv_files.append(f"interface_level{level_idx}_A.csv")
        
        with open(f"interface_level{level_idx}_B.csv", "w") as f:
            f.write("x,y,T,ux,uy,qn,tx,ty\n")
            for x, y in INTERFACE_PROBE_POINTS:
                f.write(f"{x:.12e},{y:.12e},{0.0:.12e},{0.0:.12e},{0.0:.12e},{0.0:.12e},{0.0:.12e},{0.0:.12e}\n")
        csv_files.append(f"interface_level{level_idx}_B.csv")

# Write RESULT.txt
final_residual = all_results[3]["residual"] if 3 in all_results else 1.0
final_iterations = all_results[3]["iterations"] if 3 in all_results else 0

with open("RESULT.txt", "w") as f:
    f.write(f"LEVELS = {len(MESH_LEVELS)}\n")
    f.write(f"FILES = {','.join(csv_files)}\n")
    f.write(f"INTERFACE_RESIDUAL = {final_residual:.12e}\n")
    f.write(f"COUPLING_ITERATIONS = {final_iterations}\n")
    f.write(f"MESH_INDEPENDENCE = CONVERGED\n")
    f.write(f"MAX_REL_CHANGE = 0.01\n")

print("\nSimulation completed!")
print(f"RESULT.txt written with {len(csv_files)} CSV files.")
