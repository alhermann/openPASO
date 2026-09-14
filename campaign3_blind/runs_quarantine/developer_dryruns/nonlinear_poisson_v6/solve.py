"""
Nonlinear Poisson equation solver using FEniCSx/dolfinx
Problem: -div(a(u) grad u) = f on unit square
with a(u) = 1 + u^2/2
Homogeneous Dirichlet BCs on entire boundary
Uses Newton iteration for nonlinear solve
"""

from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
import ufl
import numpy as np


def source_term(x):
    """Source term f(x,y) as given in the problem statement."""
    dtype = default_scalar_type
    x0 = x[0]
    x1 = x[1]
    
    # The full polynomial expression (exactly as given in problem)
    f = (320*x0**9*x1**4 - 640*x0**9*x1**3 + 384*x0**9*x1**2 - 64*x0**9*x1 
         + 112*x0**8*x1**5 - 1008*x0**8*x1**4 + 1696*x0**8*x1**3 
         - 648*x0**8*x1**2/5 + 768*x0**8*x1/5
         + 7024*x0**7*x1**6/9 - 13116*x0**7*x1**5/5 + 16852*x0**7*x1**4/5 
         - 91496*x0**7*x1**3/45 + 14752*x0**7*x1**2/25 - 2112*x0**7*x1/25
         + 1348*x0**6*x1**7/9 - 51772*x0**6*x1**6/27 + 1132663*x0**6*x1**5/225 
         - 3243077*x0**6*x1**4/675 + 327088*x0**6*x1**3/225 
         + 14144*x0**6*x1**2/125 - 3584*x0**6*x1/125
         + 28*x0**5*x1**8/3 - 4808*x0**5*x1**7/15 + 343336*x0**5*x1**6/225 
         - 203819*x0**5*x1**5/75 + 90457*x0**5*x1**4/45 - 6304*x0**5*x1**3/15 
         - 12992*x0**5*x1**2/125 + 2112*x0**5*x1/125
         + 5*x0**4*x1**9/27 - 173*x0**4*x1**8/9 + 9229*x0**4*x1**7/45 
         - 244697*x0**4*x1**6/675 + 309*x0**4*x1**5/25 + 69931*x0**4*x1**4/225 
         - 28112*x0**4*x1**3/225 - 3488*x0**4*x1**2/125 + 768*x0**4*x1/125
         - 10*x0**3*x1**9/27 + 106*x0**3*x1**8/9 - 1318*x0**3*x1**7/45 
         - 25786*x0**3*x1**6/675 + 38573*x0**3*x1**5/225 - 114751*x0**3*x1**4/675 
         + 12584*x0**3*x1**3/225 - 224*x0**3*x1**2/125 + 64*x0**3*x1/125 
         + 8*x0**3
         + 2*x0**2*x1**9/9 - 26*x0**2*x1**8/15 - 382*x0**2*x1**7/75 
         + 12566*x0**2*x1**6/1125 + 4144*x0**2*x1**5/375 - 3488*x0**2*x1**4/125 
         + 1536*x0**2*x1**3/125 + 2*x0**2*x1 - 106*x0**2/15
         - x0*x1**9/27 - 7*x0*x1**8/45 + 11*x0*x1**7/225 + 2177*x0*x1**6/3375 
         - 44*x0*x1**5/375 - 112*x0*x1**4/125 + 64*x0*x1**3/125 
         + 24*x0*x1**2 - 26*x0*x1 - 14*x0/15
         + 2*x1**3/3 - 106*x1**2/15 + 32*x1/5)
    
    return np.full(x.shape[1], f, dtype=dtype)


def evaluate_at_points_simple(u, V, domain, probe_points):
    """Evaluate function u at arbitrary points using brute-force cell search."""
    n_points = len(probe_points)
    
    # Get dofmap and mesh coordinates  
    dofmap_list = V.dofmap.list
    mesh_coords = domain.geometry.x
    
    # Number of cells - index_map is a METHOD, not an attribute
    tdim = domain.topology.dim
    num_cells = domain.topology.index_map(tdim).size_local
    
    probe_values = np.zeros(n_points, dtype=default_scalar_type)
    
    for probe_idx in range(n_points):
        probe_pt = np.array([probe_points[probe_idx][0], probe_points[probe_idx][1]], dtype=default_scalar_type)
        found = False
        
        # Search through all cells to find containing cell
        for cell_idx in range(num_cells):
            cell_dofs = dofmap_list[cell_idx]
            vertex_coords = mesh_coords[cell_dofs]
            
            # Triangle vertices
            v0, v1, v2 = vertex_coords[0, :2], vertex_coords[1, :2], vertex_coords[2, :2]
            
            # Barycentric coordinates
            denom = ((v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1]))
            if abs(denom) < 1e-15:
                continue
            
            alpha = (((v1[1] - v2[1]) * (probe_pt[0] - v2[0]) + 
                      (v2[0] - v1[0]) * (probe_pt[1] - v2[1])) / denom)
            beta = (((v2[1] - v0[1]) * (probe_pt[0] - v2[0]) + 
                     (v0[0] - v2[0]) * (probe_pt[1] - v2[1])) / denom)
            gamma = 1.0 - alpha - beta
            
            # Check if point is inside triangle (all barycentric coords >= 0)
            if alpha >= -1e-10 and beta >= -1e-10 and gamma >= -1e-10:
                u_vertex = u.x.array[cell_dofs]
                probe_values[probe_idx] = alpha * u_vertex[0] + beta * u_vertex[1] + gamma * u_vertex[2]
                found = True
                break
        
        if not found:
            print(f"  Warning: Could not find cell for probe point {probe_idx}")
            probe_values[probe_idx] = 0.0
    
    return probe_values


def solve_nonlinear_poisson(N, level, output_dir="."):
    """Solve the nonlinear Poisson equation on an NxN triangular mesh."""
    print(f"\n{'='*60}")
    print(f"Solving level {level}: N = {N} elements per side")
    print(f"{'='*60}")
    
    # Create mesh
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.triangle)
    
    # Create function space - Lagrange P1 elements
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # Boundary condition: u = 0 on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)
    
    # Get number of degrees of freedom
    ndof = V.dofmap.index_map.size_global
    print(f"NDOF = {ndof}")
    
    # Define test function
    v = ufl.TestFunction(V)
    
    # Source term - interpolate into function space
    f = fem.Function(V)
    f.interpolate(source_term)
    
    # Create the solution Function
    u = fem.Function(V)
    
    # Coefficient a(u) = 1 + u^2/2
    a_u = 1.0 + u**2 / 2.0
    
    # Weak form: F = integral(a(u)*grad(u)*grad(v)) - integral(f*v)
    F = a_u * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx
    
    # Set up nonlinear problem with PETSc options
    petsc_options = {
        'snes_rtol': 1e-10,
        'snes_atol': 1e-12,
        'snes_max_it': 50,
        'snes_monitor': '',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps'
    }
    
    problem = NonlinearProblem(F, u, bcs=[bc],
                                petsc_options_prefix="nonlinear_",
                                petsc_options=petsc_options)
    
    # Initial guess: zero
    u.x.array[:] = 0.0
    
    # Solve
    print("Starting Newton iteration...")
    problem.solve()
    
    # Check convergence
    reason = problem.solver.getConvergedReason()
    num_iterations = problem.solver.getIterationNumber()
    converged = reason > 0
    
    print(f"Newton iterations: {num_iterations}")
    print(f"Converged: {converged}, Reason: {reason}")
    
    if not converged:
        print(f"WARNING: Solver did not converge! Reason code: {reason}")
    
    # Write run log
    log_filename = f"{output_dir}/run_level{level}.log"
    with open(log_filename, 'w') as logfile:
        logfile.write(f"NDOF = {ndof}\n")
        logfile.write(f"Newton iterations: {num_iterations}\n")
        logfile.write(f"Converged: {converged}\n")
        logfile.write(f"Convergence reason: {reason}\n")
        logfile.write(f"Mesh size: N={N}\n")
    
    print(f"Run log written to {log_filename}")
    
    # Generate probe points: 44x44 grid
    probe_points = []
    for i_x in range(44):
        for i_y in range(44):
            x_coord = (i_x + 0.5) / 44.0
            y_coord = (i_y + 0.5) / 44.0
            probe_points.append([x_coord, y_coord])
    
    print(f"Total probe points: {len(probe_points)}")
    
    # Evaluate solution at probe points
    probe_values = evaluate_at_points_simple(u, V, domain, probe_points)
    
    # Write CSV file with high precision
    csv_filename = f"{output_dir}/solution_level{level}.csv"
    with open(csv_filename, 'w') as csvfile:
        csvfile.write("x, y, u\n")
        for i, pt in enumerate(probe_points):
            csvfile.write(f"{pt[0]:.11e}, {pt[1]:.11e}, {probe_values[i]:.11e}\n")
    
    print(f"Solution CSV written to {csv_filename}")
    
    # Return statistics
    u_array = u.x.array
    return {
        'level': level,
        'N': N,
        'ndof': ndof,
        'iterations': num_iterations,
        'converged': converged,
        'reason': reason,
        'min_u': float(np.min(u_array)),
        'max_u': float(np.max(u_array)),
        'mean_u': float(np.mean(u_array)),
        'probe_min': float(np.min(probe_values)),
        'probe_max': float(np.max(probe_values))
    }


def main():
    """Main driver: solve on all mesh levels and assess convergence."""
    import os
    
    mesh_sizes = [8, 16, 32, 64]
    levels = [1, 2, 3, 4]
    
    output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    # Solve on each mesh level
    for level, N in zip(levels, mesh_sizes):
        try:
            result = solve_nonlinear_poisson(N, level, output_dir)
            results.append(result)
        except Exception as e:
            print(f"ERROR at level {level} (N={N}): {e}")
            import traceback
            traceback.print_exc()
            with open(f"{output_dir}/run_level{level}.log", 'w') as f:
                f.write(f"ERROR: {e}\n")
            break
    
    # Analyze convergence
    if len(results) >= 2:
        csv3_path = f"{output_dir}/solution_level3.csv"
        csv4_path = f"{output_dir}/solution_level4.csv"
        
        def read_csv_values(path):
            values = []
            with open(path, 'r') as f:
                f.readline()  # Skip header
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        values.append(float(parts[2]))
            return values
        
        vals3 = read_csv_values(csv3_path)
        vals4 = read_csv_values(csv4_path)
        
        rel_changes = []
        for v3, v4 in zip(vals3, vals4):
            if abs(v3) > 1e-15:
                rel_changes.append(abs(v4 - v3) / abs(v3))
            elif abs(v4) > 1e-15:
                rel_changes.append(1.0)
        
        max_rel_change = max(rel_changes) if rel_changes else 0.0
        converged = max_rel_change < 0.01
        convergence_status = "CONVERGED" if converged else "NOT_CONVERGED"
        
        print(f"\n{'='*60}")
        print("CONVERGENCE ANALYSIS")
        print(f"{'='*60}")
        print(f"Max relative change between N=32 and N=64: {max_rel_change:.6e}")
        print(f"Mesh independence: {convergence_status}")
    else:
        max_rel_change = 0.0
        convergence_status = "NOT_CONVERGED"
    
    # Write RESULT.txt
    csv_files = [f"solution_level{k}.csv" for k in range(1, len(results)+1)]
    
    with open(f"{output_dir}/RESULT.txt", 'w') as result_file:
        result_file.write(f"LEVELS = {len(results)}\n")
        result_file.write(f"FILES = {','.join(csv_files)}\n")
        result_file.write(f"MESH_INDEPENDENCE = {convergence_status}\n")
        result_file.write(f"MAX_REL_CHANGE = {max_rel_change:.11e}\n")
    
    print(f"\nResult summary written to {output_dir}/RESULT.txt")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "OK" if r['converged'] else "FAILED"
        print(f"Level {r['level']} (N={r['N']}): DOFs={r['ndof']}, "
              f"Iterations={r['iterations']}, Status={status}, "
              f"u_range=[{r['min_u']:.4e}, {r['max_u']:.4e}]")


if __name__ == "__main__":
    main()