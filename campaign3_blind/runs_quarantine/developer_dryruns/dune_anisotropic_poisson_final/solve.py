"""
Anisotropic Poisson problem on unit square using DUNE-fem
Equation: -div(K grad u) = f
K = [[5/2, 0], [0, 1]] (diagonal tensor)
u = 0 on entire boundary
Discretization: P1 continuous Lagrange on triangles
Mesh levels: N = 8, 16, 32, 64 cells per side
"""

import numpy as np
from dune.grid import cartesianDomain
from dune.alugrid import aluConformGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC, Constant
from ufl import TrialFunction, TestFunction, dot, grad, dx, as_tensor, SpatialCoordinate
import json
import os

# Problem parameters
K11 = 5.0/2.0  # K[0,0]
K22 = 1.0      # K[1,1]

def source_term(x):
    """f(x, y) = 9*x**3*y/2 - 17*x**3/6 + 15*x**2*y/2 + 41*x**2/6 + 
                45*x*y**3/4 - 85*x*y**2/4 - 2*x*y - 4*x + 
                25*y**3/4 + 205*y**2/12 - 70*y/3"""
    x_val = x[0]
    y_val = x[1]
    return (9*x_val**3*y_val/2 - 17*x_val**3/6 + 15*x_val**2*y_val/2 + 41*x_val**2/6 + 
            45*x_val*y_val**3/4 - 85*x_val*y_val**2/4 - 2*x_val*y_val - 4*x_val + 
            25*y_val**3/4 + 205*y_val**2/12 - 70*y_val/3)

def generate_probe_points():
    """Generate 1936 probe points: x = (i_x+0.5)/44, y = (i_y+0.5)/44
    for i_x, i_y = 0, ..., 43, ordered with last index varying fastest."""
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = (i_x + 0.5) / 44.0
            y = (i_y + 0.5) / 44.0
            points.append([x, y])
    return np.array(points)

def solve_level(N, level_idx):
    """Solve the problem for N cells per side."""
    print(f"\n{'='*60}")
    print(f"Level {level_idx}: N = {N} cells per side")
    print(f"{'='*60}")
    
    # Create triangular mesh using aluConformGrid
    # cartesianDomain creates a structured grid, aluConformGrid splits each cell into triangles
    domain = cartesianDomain([0, 0], [1, 1], [N, N])
    gridView = aluConformGrid(domain, dimgrid=2)
    
    num_elements = gridView.size(0)
    print(f"Number of triangular elements: {num_elements}")
    
    # Create P1 Lagrange space (continuous, order 1)
    space = lagrange(gridView, order=1)
    num_dofs = space.size
    print(f"Number of DOFs: {num_dofs}")
    
    # Define UFL forms
    u = TrialFunction(space)
    v = TestFunction(space)
    x = SpatialCoordinate(space)
    
    # Anisotropic diffusion: -div(K grad u) = f
    # K is diagonal: K = [[K11, 0], [0, K22]]
    # Weak form: integral(K * grad(u) . grad(v)) dx = integral(f * v) dx
    
    # Build the bilinear form with anisotropic K
    # grad(u) = [du/dx, du/dy]^T
    # K * grad(u) = [K11*du/dx, K22*du/dy]^T
    # K * grad(u) . grad(v) = K11*du/dx*dv/dx + K22*du/dy*dv/dy
    
    a = K11 * grad(u)[0] * grad(v)[0] * dx + K22 * grad(u)[1] * grad(v)[1] * dx
    
    # Source term
    b = source_term(x) * v * dx
    
    # Dirichlet BC: u = 0 on entire boundary
    dbc = DirichletBC(space, 0)
    
    # Build and solve
    scheme = galerkin([a == b, dbc], solver="cg")
    uh = space.interpolate(0, name="solution")
    info = scheme.solve(target=uh)
    
    print(f"Solver converged: {info['converged']}")
    print(f"Linear iterations: {info['linear_iterations']}")
    
    # Write run log
    with open(f"run_level{level_idx}.log", "w") as f:
        f.write(f"NDOF = {num_dofs}\n")
        f.write(f"Elements = {num_elements}\n")
        f.write(f"Converged = {info['converged']}\n")
        f.write(f"Linear iterations = {info['linear_iterations']}\n")
    
    # Generate probe points
    probe_points = generate_probe_points()
    print(f"Number of probe points: {len(probe_points)}")
    
    # Evaluate solution at probe points
    # Use pointSample from dune.fem.utility
    from dune.fem.utility import pointSample
    
    u_values = []
    for pt in probe_points:
        val = pointSample(uh, pt)
        u_values.append(float(val))
    
    # Write CSV file with high precision (at least 12 significant digits)
    csv_filename = f"solution_level{level_idx}.csv"
    with open(csv_filename, "w") as f:
        f.write("x, y, u\n")
        for i, pt in enumerate(probe_points):
            f.write(f"{pt[0]:.15e}, {pt[1]:.15e}, {u_values[i]:.15e}\n")
    
    print(f"Wrote {csv_filename}")
    
    # Return data for convergence check
    return {
        'level': level_idx,
        'N': N,
        'num_dofs': num_dofs,
        'num_elements': num_elements,
        'probe_values': np.array(u_values),
        'max_value': float(np.max(np.abs(u_values))),
        'mean_value': float(np.mean(np.abs(u_values)))
    }

def main():
    """Main driver: solve at all mesh levels and write RESULT.txt"""
    
    mesh_levels = [8, 16, 32, 64]
    results = []
    
    for idx, N in enumerate(mesh_levels, start=1):
        result = solve_level(N, idx)
        results.append(result)
    
    # Compute mesh independence
    # Compare finest two levels (32 and 64)
    finest = results[-1]
    second_finest = results[-2]
    
    u_fine = finest['probe_values']
    u_coarse = second_finest['probe_values']
    
    # Absolute difference
    diff = np.abs(u_fine - u_coarse)
    
    # For relative change, use a robust formula that handles near-zero values
    # rel_change = |u_fine - u_coarse| / max(|u_coarse|, eps)
    # where eps is a small threshold based on the solution scale
    
    # Scale reference: use the maximum absolute value across both solutions
    scale = max(np.max(np.abs(u_fine)), np.max(np.abs(u_coarse)))
    eps = 1e-12 * scale  # Small threshold relative to solution scale
    
    # Robust relative change at each point
    rel_changes = diff / np.maximum(np.abs(u_coarse), eps)
    
    max_rel_change = float(np.max(rel_changes))
    mean_rel_change = float(np.mean(rel_changes))
    
    # Also compute L2-based metrics for additional insight
    l2_diff = np.sqrt(np.mean(diff**2))
    l2_fine = np.sqrt(np.mean(u_fine**2))
    rel_l2_change = l2_diff / (l2_fine + 1e-30)
    
    print(f"\n{'='*60}")
    print("MESH INDEPENDENCE ANALYSIS")
    print(f"{'='*60}")
    print(f"Solution scale (max |u|): {scale:.6e}")
    print(f"L2 norm of fine solution: {l2_fine:.6e}")
    print(f"L2 norm of difference: {l2_diff:.6e}")
    print(f"Relative L2 change: {rel_l2_change:.6e}")
    print(f"Max relative change (robust): {max_rel_change:.6e}")
    print(f"Mean relative change: {mean_rel_change:.6e}")
    
    # Convergence criterion: max relative change < 1%
    tol = 0.01
    converged = max_rel_change < tol
    mesh_independence = "CONVERGED" if converged else "NOT_CONVERGED"
    
    print(f"MESH_INDEPENDENCE = {mesh_independence}")
    
    # Write RESULT.txt
    csv_files = ", ".join([f"solution_level{k}.csv" for k in range(1, len(results)+1)])
    
    with open("RESULT.txt", "w") as f:
        f.write(f"LEVELS = {len(results)}\n")
        f.write(f"FILES = {csv_files}\n")
        f.write(f"MESH_INDEPENDENCE = {mesh_independence}\n")
        f.write(f"MAX_REL_CHANGE = {max_rel_change:.15e}\n")
    
    print("\nWrote RESULT.txt")
    print("Done!")

if __name__ == "__main__":
    main()
