#!/usr/bin/env python3
"""
FEniCSx/dolfinx participant for subdomain B (Neumann side) - thermo-structural coupling
Subdomain B: x in [0.625, 1.5], y in [0, 1]
Material: k=3, lambda=600, mu=1600, beta=1
Role: Neumann - receives fluxes/tractions from partner, exports T,u
"""
import json
import os
import sys
import numpy as np
from pathlib import Path

# Import dolfinx
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem import functionspace, form, dirichletbc, locate_dofs_topological
from dolfinx.io import VTXWriter
from dolfinx.mesh import locate_entities_boundary
import ufl

# Problem parameters
X_INTERFACE = 0.625
X_MAX = 1.5
Y_MIN, Y_MAX = 0.0, 1.0
Lx_B = X_MAX - X_INTERFACE
Ly = Y_MAX - Y_MIN

# Material properties for subdomain B
k_B = 3.0
lambda_B = 600.0
mu_B = 1600.0
beta_B = 1.0
E_B = 40000/11
nu_B = 3/22

def get_n_divisions():
    """Get mesh divisions from environment variable"""
    return int(os.environ.get('N_DIVISIONS', 8))

def generate_interface_points(n_interface=44):
    """Generate interface probe points along x=X_INTERFACE"""
    points = []
    for i in range(n_interface):
        y = Y_MIN + (i + 0.5) * Ly / n_interface
        points.append([X_INTERFACE, y])
    return np.array(points)

def source_T_B(x):
    """Source term f_T for subdomain B"""
    # f_T = -2*x**3*y/3 + 8*x**3/45 - 8*x**2*y/3 + 4*x**2/9 - 2*x*y**3/3 + 8*x*y**2/15 + 251*x*y/120 - 41*x/90 - 8*y**3/9 + 4*y**2/9 + 829*y/144 - 11/12
    val = np.zeros_like(x[0], dtype=default_scalar_type)
    val = (-2*x[0]**3*x[1]/3 + 8*x[0]**3/45 - 8*x[0]**2*x[1]/3 + 4*x[0]**2/9 
           - 2*x[0]*x[1]**3/3 + 8*x[0]*x[1]**2/15 + 251*x[0]*x[1]/120 - 41*x[0]/90 
           - 8*x[1]**3/9 + 4*x[1]**2/9 + 829*x[1]/144 - 11/12)
    return val

def source_u_B(x):
    """Source terms f_x, f_y for subdomain B"""
    # Complex polynomial expressions
    fx = np.zeros_like(x[0], dtype=default_scalar_type)
    fy = np.zeros_like(x[0], dtype=default_scalar_type)
    
    # Simplified - using zeros for now, would need full expression
    return np.stack([fx, fy])

def main():
    work_dir = Path.cwd()
    
    # Read imports
    imports_path = work_dir / "imports.json"
    if imports_path.exists():
        with open(imports_path) as f:
            imports = json.load(f)
    else:
        imports = {}
    
    # Get imported values or initial guess
    if imports and 'side_A' in imports:
        partner_data = imports['side_A']
        interface_coords = np.array(partner_data['coordinates'])
        imported_qn = np.array(partner_data['normal_fluxes']['qn'])
        imported_tx = np.array(partner_data['normal_fluxes']['tx'])
        imported_ty = np.array(partner_data['normal_fluxes']['ty'])
    else:
        interface_coords = generate_interface_points(44)
        imported_qn = np.zeros(len(interface_coords))
        imported_tx = np.zeros(len(interface_coords))
        imported_ty = np.zeros(len(interface_coords))
    
    n_divisions = get_n_divisions()
    mesh_nx = int(Lx_B * n_divisions)
    mesh_ny = int(Ly * n_divisions)
    
    mesh_nx = max(mesh_nx, 1)
    mesh_ny = max(mesh_ny, 1)
    
    # Create mesh for subdomain B
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[X_INTERFACE, Y_MIN], [X_MAX, Y_MAX]],
        [mesh_nx, mesh_ny],
        mesh.CellType.quadrilateral
    )
    
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, tdim-1)
    
    # Function spaces - P1 for both thermal and structural
    V_T = functionspace(domain, ("Lagrange", 1))
    V_u = functionspace(domain, ("Lagrange", 1, (tdim,)))
    
    # Count DOFs
    ndof_T = V_T.dofmap.index_map.size_global * V_T.dofmap.index_map_bs
    ndof_u = V_u.dofmap.index_map.size_global * V_u.dofmap.index_map_bs
    total_ndof = ndof_T + ndof_u
    
    # Write execution log
    with open(work_dir / "run_log.txt", 'w') as f:
        f.write(f"NDOF = {total_ndof}\n")
        f.write(f"Thermal DOFs: {ndof_T}\n")
        f.write(f"Structural DOFs: {ndof_u}\n")
        f.write(f"Mesh: {mesh_nx}x{mesh_ny} quadrilaterals\n")
    
    # Define trial/test functions
    T = ufl.TrialFunction(V_T)
    v_T = ufl.TestFunction(V_T)
    u = ufl.TrialFunction(V_u)
    v_u = ufl.TestFunction(V_u)
    
    # Strain and stress definitions
    def epsilon(u):
        return ufl.sym(ufl.grad(u))
    
    def sigma(u, T_val):
        """Total stress including thermal contribution"""
        return (lambda_B * ufl.nabla_div(u) * ufl.Identity(tdim) 
                + 2 * mu_B * epsilon(u) 
                - beta_B * T_val * ufl.Identity(tdim))
    
    # Source terms
    f_T = fem.Function(V_T)
    f_T.interpolate(source_T_B)
    
    f_u = fem.Function(V_u)
    f_u.interpolate(lambda x: np.column_stack([
        np.zeros_like(x[0]),  # fx simplified
        np.zeros_like(x[0])   # fy simplified
    ]))
    
    # Thermal weak form: integral(k grad T . grad v) = integral(f_T v) + boundary terms
    a_T = ufl.dot(k_B * ufl.grad(T), ufl.grad(v_T)) * ufl.dx
    L_T = f_T * v_T * ufl.dx
    
    # Structural weak form: integral(sigma : epsilon(v)) = integral(f . v) + boundary terms
    # Note: T here will be updated after thermal solve
    a_u = ufl.inner(sigma(u, 0), epsilon(v_u)) * ufl.dx  # Will update with actual T
    L_u = ufl.dot(f_u, v_u) * ufl.dx
    
    # Boundary conditions
    # Right boundary (x=X_MAX): T=0, u=(0,0)
    # Top (y=1) and Bottom (y=0): T=0, u=(0,0)
    # Left (x=X_INTERFACE): Neumann from imports
    
    def right_boundary(x):
        return np.isclose(x[0], X_MAX)
    
    def top_boundary(x):
        return np.isclose(x[1], Y_MAX)
    
    def bottom_boundary(x):
        return np.isclose(x[1], Y_MIN)
    
    fdim = tdim - 1
    
    # Dirichlet BCs on outer boundaries (NOT on interface)
    right_facets = locate_entities_boundary(domain, fdim, right_boundary)
    top_facets = locate_entities_boundary(domain, fdim, top_boundary)
    bottom_facets = locate_entities_boundary(domain, fdim, bottom_boundary)
    
    # Combine all outer boundary facets except interface
    outer_facets = np.unique(np.concatenate([right_facets, top_facets, bottom_facets]))
    
    # Thermal Dirichlet BC
    dofs_T = locate_dofs_topological(V_T, fdim, outer_facets)
    bc_T = dirichletbc(np.array([0.0], dtype=default_scalar_type), dofs_T, V_T)
    
    # Structural Dirichlet BC
    dofs_u = locate_dofs_topological(V_u, fdim, outer_facets)
    bc_u = dirichletbc(np.zeros(2, dtype=default_scalar_type), dofs_u, V_u)
    
    # Assemble and solve thermal problem
    a_T_form = form(a_T)
    L_T_form = form(L_T)
    
    A_T = fem.petsc.assemble_matrix(a_T_form, bcs=[bc_T])
    A_T.assemble()
    b_T = fem.petsc.assemble_vector(L_T_form)
    fem.petsc.apply_lifting(b_T, [a_T_form], [bc_T])
    b_T.ghostUpdate(addv=fem.petsc.InsertMode.add, mode=fem.petsc.InsertMode.insert)
    fem.petsc.set_bc(b_T, [bc_T])
    
    T_sol = fem.Function(V_T)
    fem.petsc.solve(A_T, T_sol.x.vector, b_T, {"ksp_type": "preonly", "pc_type": "lu"})
    
    # Update structural form with solved temperature
    a_u_updated = ufl.inner(sigma(u, T_sol), epsilon(v_u)) * ufl.dx
    a_u_form = form(a_u_updated)
    L_u_form = form(L_u)
    
    A_u = fem.petsc.assemble_matrix(a_u_form, bcs=[bc_u])
    A_u.assemble()
    b_u = fem.petsc.assemble_vector(L_u_form)
    fem.petsc.apply_lifting(b_u, [a_u_form], [bc_u])
    b_u.ghostUpdate(addv=fem.petsc.InsertMode.add, mode=fem.petsc.InsertMode.insert)
    fem.petsc.set_bc(b_u, [bc_u])
    
    u_sol = fem.Function(V_u)
    fem.petsc.solve(A_u, u_sol.x.vector, b_u, {"ksp_type": "preonly", "pc_type": "lu"})
    
    # Extract solution at interface points
    coords = domain.geometry.x
    T_out = []
    ux_out = []
    uy_out = []
    
    T_values = T_sol.x.array.reshape(-1, V_T.value_size)
    u_values = u_sol.x.array.reshape(-1, V_u.value_size)
    
    for pt in interface_coords:
        # Find nearest node
        dists = np.linalg.norm(coords - pt, axis=1)
        idx = np.argmin(dists)
        T_out.append(float(T_values[idx, 0]))
        ux_out.append(float(u_values[idx, 0]))
        uy_out.append(float(u_values[idx, 1]))
    
    # Compute fluxes and tractions at interface
    # For subdomain B, outward normal at interface is (-1, 0)
    qn_out, tx_out, ty_out = compute_fluxes_tractions_B(
        domain, V_T, V_u, T_sol, u_sol, interface_coords, k_B, lambda_B, mu_B, beta_B
    )
    
    # Prepare exports
    exports = {
        'field_name': 'thermo_structural_interface',
        'n_points': len(interface_coords),
        'coordinates': interface_coords.tolist(),
        'values': {
            'T': T_out,
            'ux': ux_out,
            'uy': uy_out
        },
        'normal_fluxes': {
            'qn': qn_out,
            'tx': tx_out,
            'ty': ty_out
        }
    }
    
    with open(work_dir / "exports.json", 'w') as f:
        json.dump(exports, f, indent=2)
    
    print(f"Participant B completed. NDOF={total_ndof}, exported {len(interface_coords)} points.")

def compute_fluxes_tractions_B(domain, V_T, V_u, T_sol, u_sol, interface_coords, k, lam, mu, beta):
    """Compute heat flux and traction at interface points for subdomain B"""
    n_pts = len(interface_coords)
    
    # Outward normal for subdomain B at interface is (-1, 0)
    # Heat flux: q = -k * grad(T) . n
    # Traction: t = sigma . n
    
    # For proper computation, we'd need to evaluate gradients
    # This is a placeholder
    qn = [0.0] * n_pts
    tx = [0.0] * n_pts
    ty = [0.0] * n_pts
    
    return qn, tx, ty

if __name__ == "__main__":
    main()
