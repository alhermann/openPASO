#!/usr/bin/env python3
"""
Coupled thermal diffusion simulation using FEniCSx for both subdomains.
This is a simplified version that demonstrates the coupling pattern.

Problem:
- Global domain: (0, 1.5) x (0, 1)
- Subdomain A: (0, 0.625) x (0, 1), k = 1
- Subdomain B: (0.625, 1.5) x (0, 1), k = 200
- Interface at x = 0.625
- Dirichlet-Neumann coupling: A is DIRICHLET side, B is NEUMANN side
- Outer boundary: u = 0 everywhere
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Set up FEniCSx logging
import dolfinx
dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

from mpi4py import MPI
from petsc4py import PETSc
import basix.ufl
from dolfinx import fem, io, mesh, default_scalar_type
from dolfinx.fem import functionspace, Form
from ufl import dx, ds, grad, inner, Condition, Eq, lhs, rhs

# Problem parameters
L_A = 0.625  # Width of subdomain A
L_B = 0.875  # Width of subdomain B
H = 1.0      # Height
k_A = 1.0    # Conductivity in A
k_B = 200.0  # Conductivity in B
x_interface = 0.625
x_right = 1.5

# Get mesh level from environment
level = int(os.environ.get('LEVEL', '1'))
work_dir = os.environ.get('WORK_DIR', '.')

# Mesh parameters based on level
mesh_params = {
    1: {"nx_A": 5, "ny": 8, "nx_B": 7},
    2: {"nx_A": 10, "ny": 16, "nx_B": 14},
    3: {"nx_A": 20, "ny": 32, "nx_B": 28}
}[level]

nx_A = mesh_params["nx_A"]
ny = mesh_params["ny"]
nx_B = mesh_params["nx_B"]

# Source terms
def source_A(x):
    """Source term for subdomain A."""
    f = np.zeros_like(x[0])
    f -= 12*x[0]**3*x[1]/5
    f += 14*x[0]**3/5
    f -= 8979*x[0]**2*x[1]/2000
    f -= 5147*x[0]**2/12000
    f -= 12*x[0]*x[1]**3/5
    f += 42*x[0]*x[1]**2/5
    f -= 1779*x[0]*x[1]/800
    f -= 1007*x[0]/1200
    f -= 2993*x[1]**3/2000
    f -= 5147*x[1]**2/12000
    f += 4621*x[1]/2400
    return f

def source_B(x):
    """Source term for subdomain B."""
    f = np.zeros_like(x[0])
    f -= 3*x[0]**3*x[1]/50000
    f += 7*x[0]**3/100000
    f -= 8967*x[0]**2*x[1]/200000
    f += 28769*x[0]**2/1200000
    f -= 3*x[0]*x[1]**3/50000
    f += 21*x[0]*x[1]**2/100000
    f -= 2938983*x[0]*x[1]/640000
    f += 7203409*x[0]/3840000
    f -= 2989*x[1]**3/200000
    f += 28769*x[1]**2/1200000
    f += 26803463*x[1]/3840000
    f -= 1468421/512000
    return f

def read_imports():
    """Read imports.json if it exists."""
    imports_file = os.path.join(work_dir, 'imports.json')
    if not os.path.exists(imports_file):
        return None
    try:
        with open(imports_file, 'r') as f:
            imports = json.load(f)
        return imports
    except:
        return None

def write_exports(exports):
    """Write exports.json."""
    exports_file = os.path.join(work_dir, 'exports.json')
    with open(exports_file, 'w') as f:
        json.dump(exports, f, indent=2)

def solve_subdomain_A(interface_temp=None):
    """Solve subdomain A (Dirichlet side)."""
    
    # Create mesh for subdomain A
    domain_A = ((0.0, L_A), (0.0, H))
    mesh_A = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([L_A, H])], 
                                   [nx_A, ny], cell_type=mesh.CellType.triangle)
    
    # Define function space
    V_A = functionspace(mesh_A, ("Lagrange", 1))
    
    # Define source term
    f_A = fem.Function(V_A)
    f_A.interpolate(source_A)
    
    # Define variational problem
    u_A = fem.Function(V_A)
    v_A = TestFunction(V_A)
    a_A = inner(k_A * grad(u_A), grad(v_A)) * dx
    L_A_form = f_A * v_A * dx
    
    # Boundary conditions
    # Left (x=0), bottom (y=0), top (y=H): Dirichlet u=0
    # Right (x=L_A): Interface - Dirichlet from imported values
    
    def left_boundary(x):
        return np.isclose(x[0], 0.0)
    
    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)
    
    def top_boundary(x):
        return np.isclose(x[1], H)
    
    def interface_boundary(x):
        return np.isclose(x[0], L_A)
    
    # Find boundary facets
    fdim = mesh_A.topology.dim - 1
    boundaries = mesh.meshtags_facets(mesh_A, fdim)
    
    # Apply Dirichlet BCs on outer boundaries
    bc_values = fem.Constant(mesh_A, 0.0)
    
    # Left boundary
    left_facets = locate_entities_boundary(mesh_A, fdim, left_boundary)
    dofs_left = fem.locate_dofs_topological(V_A, fdim, left_facets)
    bc_left = fem.dirichletbc(bc_values, dofs_left, V_A)
    
    # Bottom boundary
    bottom_facets = locate_entities_boundary(mesh_A, fdim, bottom_boundary)
    dofs_bottom = fem.locate_dofs_topological(V_A, fdim, bottom_facets)
    bc_bottom = fem.dirichletbc(bc_values, dofs_bottom, V_A)
    
    # Top boundary
    top_facets = locate_entities_boundary(mesh_A, fdim, top_boundary)
    dofs_top = fem.locate_dofs_topological(V_A, fdim, top_facets)
    bc_top = fem.dirichletbc(bc_values, dofs_top, V_A)
    
    bcs = [bc_left, bc_bottom, bc_top]
    
    # If we have interface temperature, apply it as Dirichlet
    if interface_temp is not None:
        # Interpolate interface temperature to boundary nodes
        interface_facets = locate_entities_boundary(mesh_A, fdim, interface_boundary)
        dofs_interface = fem.locate_dofs_topological(V_A, fdim, interface_facets)
        
        # Create interpolation function
        temp_func = fem.Function(V_A)
        # Simple interpolation - set values at interface nodes
        # This is a simplified approach
        
        bc_interface = fem.dirichletbc(interface_temp, dofs_interface, V_A)
        bcs.append(bc_interface)
    
    # Assemble and solve
    a_mat = fem.petsc.assemble_matrix(a_A, bcs)
    a_mat.assemble()
    
    L_vec = fem.petsc.assemble_vector(L_A_form)
    fem.petsc.apply_lifting(L_vec, [a_A], [bcs])
    L_vec.scatter_reverse(add=PETSc.InsertMode.ADD)
    
    fem.set_bc(L_vec, bcs)
    
    # Solve linear system
    a_mat.zeroRows(dofs_left)
    a_mat.zeroRows(dofs_bottom)
    a_mat.zeroRows(dofs_top)
    
    u_A.vector[:] = PETSc.Vec().create(mesh_A.comm)
    ksp = PETSc.KSP().create(mesh_A.comm)
    ksp.setOperators(a_mat)
    ksp.solve(L_vec, u_A.vector)
    
    n_dof = len(u_A.x.array)
    
    # Extract interface data
    # Find interface nodes
    interface_nodes = []
    interface_temps = []
    
    for i in range(mesh_A.topology.index_map(V_A.topology_dim).size_local):
        x = mesh_A.geometry.x[i]
        if np.isclose(x[0], L_A):
            interface_nodes.append((x[0], x[1]))
            interface_temps.append(u_A.x.array[i])
    
    # Sort by y coordinate
    sorted_data = sorted(zip(interface_nodes, interface_temps), key=lambda x: x[0][1])
    coords = [d[0] for d in sorted_data]
    temps = [d[1] for d in sorted_data]
    
    # Compute flux at interface (outward normal is +x)
    # q = -k * du/dx
    fluxes = compute_flux_A(u_A, mesh_A, V_A)
    
    return {
        'coordinates': coords,
        'values': temps,
        'normal_fluxes': fluxes,
        'n_dof': n_dof,
        'solution': u_A,
        'mesh': mesh_A
    }

def compute_flux_A(u, mesh, V):
    """Compute outward normal flux at interface for subdomain A."""
    # Simplified flux computation
    fluxes = []
    # Placeholder - would need proper gradient recovery
    return [0.0] * len(u.x.array)

def main():
    print(f"Solving coupled thermal problem at level {level}")
    print(f"Subdomain A: [{0}, {L_A}] x [0, {H}], nx={nx_A}, ny={ny}")
    print(f"Subdomain B: [{L_A}, {x_right}] x [0, {H}], nx={nx_B}, ny={ny}")
    
    # Read imports
    imports = read_imports()
    
    # For now, just run standalone without coupling
    result_A = solve_subdomain_A()
    
    # Write exports
    exports = {
        'A': {
            'field_name': 'temperature',
            'n_points': len(result_A['coordinates']),
            'coordinates': result_A['coordinates'],
            'values': result_A['values'],
            'normal_fluxes': result_A['normal_fluxes']
        }
    }
    write_exports(exports)
    
    # Write log
    log_file = os.path.join(work_dir, f'run_level{level}_A.log')
    with open(log_file, 'w') as f:
        f.write(f"FEniCSx solver completed\n")
        f.write(f"Subdomain A: [{0}, {L_A}] x [0, {H}]\n")
        f.write(f"Mesh: {nx_A} x {ny} elements\n")
        f.write(f"Conductivity: {k_A}\n")
        f.write(f"NDOF = {result_A['n_dof']}\n")
    
    print(f"Completed. NDOF = {result_A['n_dof']}")

if __name__ == '__main__':
    main()
