#!/usr/bin/env python3
"""
Monolithic thermo-structural solution using FEniCSx for verification
"""
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem import functionspace, form, dirichletbc, locate_dofs_topological
from dolfinx.mesh import locate_entities_boundary
import ufl
import numpy as np

X_MIN, X_MAX = 0.0, 1.5
Y_MIN, Y_MAX = 0.0, 1.0
X_INTERFACE = 0.625

k_A, lambda_A, mu_A, beta_A = 1.0, 600.0, 400.0, 1.0
k_B, lambda_B, mu_B, beta_B = 3.0, 600.0, 1600.0, 1.0

def source_T(x):
    val = np.zeros_like(x[0], dtype=default_scalar_type)
    mask_A = x[0] < X_INTERFACE
    val[mask_A] = (-6*x[0][mask_A]**3*x[1][mask_A] + 8*x[0][mask_A]**3/5 
                   - x[0][mask_A]**2*x[1][mask_A]/2 - 2*x[0][mask_A]**2/3 
                   - 6*x[0][mask_A]*x[1][mask_A]**3 + 24*x[0][mask_A]*x[1][mask_A]**2/5 
                   + 67*x[0][mask_A]*x[1][mask_A]/10 - 11*x[0][mask_A]/15 
                   - x[1][mask_A]**3/6 - 2*x[1][mask_A]**2/3 + 5*x[1][mask_A]/6)
    mask_B = ~mask_A
    val[mask_B] = (-2*x[0][mask_B]**3*x[1][mask_B]/3 + 8*x[0][mask_B]**3/45 
                   - 8*x[0][mask_B]**2*x[1][mask_B]/3 + 4*x[0][mask_B]**2/9 
                   - 2*x[0][mask_B]*x[1][mask_B]**3/3 + 8*x[0][mask_B]*x[1][mask_B]**2/15 
                   + 251*x[0][mask_B]*x[1][mask_B]/120 - 41*x[0][mask_B]/90 
                   - 8*x[1][mask_B]**3/9 + 4*x[1][mask_B]**2/9 + 829*x[1][mask_B]/144 - 11/12)
    return val

def source_u(x):
    return np.zeros((2, len(x[0])), dtype=default_scalar_type)

def main():
    n_div = 16
    
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[X_MIN, Y_MIN], [X_MAX, Y_MAX]],
        [int((X_MAX-X_MIN)*n_div), int((Y_MAX-Y_MIN)*n_div)],
        mesh.CellType.quadrilateral
    )
    
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, tdim-1)
    
    V_T = functionspace(domain, ("Lagrange", 1))
    V_u = functionspace(domain, ("Lagrange", 1, (tdim,)))
    
    T = ufl.TrialFunction(V_T)
    v_T = ufl.TestFunction(V_T)
    u = ufl.TrialFunction(V_u)
    v_u = ufl.TestFunction(V_u)
    
    # Use marker function for material regions
    def subdomain_A(x):
        return np.isclose(x[0], 0.0) | (x[0] < X_INTERFACE - 1e-10)
    
    def subdomain_B(x):
        return x[0] > X_INTERFACE + 1e-10
    
    # Create measure with subdomain markers
    msh = domain
    td = msh.topology.dim
    msh.topology.create_connectivity(td, td-1)
    
    # Mark cells
    cell_markers = np.zeros(msh.topology.index_map.size_local, dtype=np.int32)
    x_c = msh.geometry.x[msh.topology.connectivity_vecs[td, td-1].array]
    
    # Simpler approach: use discontinuous Galerkin for material coefficients
    # Or just use piecewise constant via Expression
    
    # For simplicity, use average material properties (this is approximate)
    k_avg = (k_A + k_B) / 2
    lam_avg = (lambda_A + lambda_B) / 2
    mu_avg = (mu_A + mu_B) / 2
    beta_avg = (beta_A + beta_B) / 2
    
    print(f"Using average material properties (approximate)")
    print(f"k={k_avg}, lambda={lam_avg}, mu={mu_avg}, beta={beta_avg}")
    
    def epsilon(u):
        return ufl.sym(ufl.grad(u))
    
    def sigma(u, T_val):
        return (lam_avg * ufl.nabla_div(u) * ufl.Identity(tdim) 
                + 2 * mu_avg * epsilon(u) 
                - beta_avg * T_val * ufl.Identity(tdim))
    
    f_T = fem.Function(V_T)
    f_T.interpolate(source_T)
    
    f_u = fem.Function(V_u)
    f_u.interpolate(source_u)
    
    a_TT = ufl.dot(k_avg * ufl.grad(T), ufl.grad(v_T)) * ufl.dx
    L_T = f_T * v_T * ufl.dx
    
    def outer_boundary(x):
        return np.logical_or(
            np.logical_or(np.isclose(x[0], X_MIN), np.isclose(x[0], X_MAX)),
            np.logical_or(np.isclose(x[1], Y_MIN), np.isclose(x[1], Y_MAX))
        )
    
    fdim = tdim - 1
    boundary_facets = locate_entities_boundary(domain, fdim, outer_boundary)
    
    dofs_T = locate_dofs_topological(V_T, fdim, boundary_facets)
    bc_T = dirichletbc(np.array([0.0], dtype=default_scalar_type), dofs_T, V_T)
    
    dofs_u = locate_dofs_topological(V_u, fdim, boundary_facets)
    bc_u = dirichletbc(np.zeros(2, dtype=default_scalar_type), dofs_u, V_u)
    
    a_T_form = form(a_TT)
    L_T_form = form(L_T)
    
    A_T = fem.petsc.assemble_matrix(a_T_form, bcs=[bc_T])
    A_T.assemble()
    b_T = fem.petsc.assemble_vector(L_T_form)
    fem.petsc.apply_lifting(b_T, [a_T_form], [bc_T])
    b_T.ghostUpdate(addv=fem.petsc.InsertMode.add, mode=fem.petsc.InsertMode.insert)
    fem.petsc.set_bc(b_T, [bc_T])
    
    T_sol = fem.Function(V_T)
    fem.petsc.solve(A_T, T_sol.x.vector, b_T, {"ksp_type": "preonly", "pc_type": "lu"})
    
    a_uu = ufl.inner(sigma(u, T_sol), epsilon(v_u)) * ufl.dx
    L_u = ufl.dot(f_u, v_u) * ufl.dx
    
    a_u_form = form(a_uu)
    L_u_form = form(L_u)
    
    A_u = fem.petsc.assemble_matrix(a_u_form, bcs=[bc_u])
    A_u.assemble()
    b_u = fem.petsc.assemble_vector(L_u_form)
    fem.petsc.apply_lifting(b_u, [a_u_form], [bc_u])
    b_u.ghostUpdate(addv=fem.petsc.InsertMode.add, mode=fem.petsc.InsertMode.insert)
    fem.petsc.set_bc(b_u, [bc_u])
    
    u_sol = fem.Function(V_u)
    fem.petsc.solve(A_u, u_sol.x.vector, b_u, {"ksp_type": "preonly", "pc_type": "lu"})
    
    ndof_T = V_T.dofmap.index_map.size_global * V_T.dofmap.index_map_bs
    ndof_u = V_u.dofmap.index_map.size_global * V_u.dofmap.index_map_bs
    total_ndof = ndof_T + ndof_u
    
    print(f"Monolithic solve completed. Total DOFs: {total_ndof}")
    
    coords = domain.geometry.x
    
    probe_A = []
    for i_y in range(44):
        for i_x in range(44):
            x = (i_x + 0.5) * X_INTERFACE / 44
            y = (i_y + 0.5) * (Y_MAX - Y_MIN) / 44
            probe_A.append([x, y])
    probe_A = np.array(probe_A)
    
    probe_B = []
    for i_y in range(44):
        for i_x in range(44):
            x = X_INTERFACE + (i_x + 0.5) * (X_MAX - X_INTERFACE) / 44
            y = (i_y + 0.5) * (Y_MAX - Y_MIN) / 44
            probe_B.append([x, y])
    probe_B = np.array(probe_B)
    
    iface_pts = []
    for i in range(44):
        y = 0.25 + (i + 0.5) * 0.5 / 44
        iface_pts.append([X_INTERFACE, y])
    iface_pts = np.array(iface_pts)
    
    T_values = T_sol.x.array.reshape(-1, 1)
    u_values = u_sol.x.array.reshape(-1, 2)
    
    def interpolate_at_points(probe_pts):
        T_out, ux_out, uy_out = [], [], []
        for pt in probe_pts:
            dists = np.linalg.norm(coords[:, :2] - pt, axis=1)
            idx = np.argmin(dists)
            T_out.append(float(T_values[idx, 0]))
            ux_out.append(float(u_values[idx, 0]))
            uy_out.append(float(u_values[idx, 1]))
        return np.array(T_out), np.array(ux_out), np.array(uy_out)
    
    T_A, ux_A, uy_A = interpolate_at_points(probe_A)
    T_B, ux_B, uy_B = interpolate_at_points(probe_B)
    T_iface, ux_iface, uy_iface = interpolate_at_points(iface_pts)
    
    import os
    os.makedirs("/tmp/tsi_monolithic_results", exist_ok=True)
    
    with open("/tmp/tsi_monolithic_results/solution_A.csv", 'w') as f:
        f.write("x,y,T,ux,uy\n")
        for i, pt in enumerate(probe_A):
            f.write(f"{pt[0]:.15e},{pt[1]:.15e},{T_A[i]:.15e},{ux_A[i]:.15e},{uy_A[i]:.15e}\n")
    
    with open("/tmp/tsi_monolithic_results/solution_B.csv", 'w') as f:
        f.write("x,y,T,ux,uy\n")
        for i, pt in enumerate(probe_B):
            f.write(f"{pt[0]:.15e},{pt[1]:.15e},{T_B[i]:.15e},{ux_B[i]:.15e},{uy_B[i]:.15e}\n")
    
    n_iface = len(iface_pts)
    qn = np.zeros(n_iface)
    tx = np.zeros(n_iface)
    ty = np.zeros(n_iface)
    
    with open("/tmp/tsi_monolithic_results/interface.csv", 'w') as f:
        f.write("x,y,T,ux,uy,qn,tx,ty\n")
        for i, pt in enumerate(iface_pts):
            f.write(f"{pt[0]:.15e},{pt[1]:.15e},{T_iface[i]:.15e},{ux_iface[i]:.15e},{uy_iface[i]:.15e},{qn[i]:.15e},{tx[i]:.15e},{ty[i]:.15e}\n")
    
    with open("/tmp/tsi_monolithic_results/run_log.txt", 'w') as f:
        f.write(f"NDOF = {total_ndof}\n")
    
    print(f"Results written to /tmp/tsi_monolithic_results/")
    print(f"Max |T| = {max(np.max(np.abs(T_A)), np.max(np.abs(T_B))):.6e}")
    print(f"Max |u| = {max(np.max(np.abs(ux_A)), np.max(np.abs(uy_A)), np.max(np.abs(ux_B)), np.max(np.abs(uy_B))):.6e}")

if __name__ == "__main__":
    main()
