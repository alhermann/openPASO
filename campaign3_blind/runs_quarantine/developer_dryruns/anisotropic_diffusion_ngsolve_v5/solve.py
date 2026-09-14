"""
Anisotropic diffusion: -div(K grad u) = f on unit square
K = [[3, -1], [-1, 2]] (symmetric positive-definite)
u = 0 on entire boundary
Lagrange order 1 on triangles
Mesh levels: h = 1/8, 1/16, 1/32, 1/64
"""
from ngsolve import *
import numpy as np

# Probe points: 44x44 staggered grid
def generate_probe_points():
    """Generate 1936 probe points: x=(i_x+0.5)/44, y=(i_y+0.5)/44 for i_x,i_y=0..43
    Ordered with last index varying fastest (row-major: y varies slowest, x varies fastest)
    """
    points = []
    for i_y in range(44):
        for i_x in range(44):
            x = (i_x + 0.5) / 44.0
            y = (i_y + 0.5) / 44.0
            points.append((x, y))
    return points

PROBE_POINTS = generate_probe_points()
assert len(PROBE_POINTS) == 1936, f"Expected 1936 probe points, got {len(PROBE_POINTS)}"

# Mesh levels
MESH_LEVELS = [
    (1, 1/8),
    (2, 1/16),
    (3, 1/32),
    (4, 1/64)
]

def solve_level(level, h):
    """Solve the problem at one mesh level."""
    print(f"\n{'='*60}")
    print(f"Solving level {level} with h = {h}")
    print(f"{'='*60}")
    
    # Create mesh
    from netgen.geom2d import unit_square
    mesh = Mesh(unit_square.GenerateMesh(maxh=h))
    print(f"Mesh: {mesh.ne} elements, {mesh.nv} vertices")
    
    # FE space: H1 order 1 with Dirichlet on all boundaries
    fes = H1(mesh, order=1, dirichlet=".*")
    u, v = fes.TnT()
    
    # Diffusion tensor K = [[3, -1], [-1, 2]]
    # Bilinear form: int(K*grad(u).dot(grad(v))) dx
    # = int(3*ux*vx - 1*ux*vy - 1*uy*vx + 2*uy*vy) dx
    a = BilinearForm(fes)
    a += (3*grad(u)[0]*grad(v)[0] 
          - 1*grad(u)[0]*grad(v)[1] 
          - 1*grad(u)[1]*grad(v)[0] 
          + 2*grad(u)[1]*grad(v)[1]) * dx
    a.Assemble()
    
    # Linear form: integral of f*v
    # Use NGSolve's x and y as CoefficientFunctions
    x_cf = CoefficientFunction(x)
    y_cf = CoefficientFunction(y)
    
    # Source term f(x,y)
    f_expr = (36*x_cf**3*y_cf 
              - 20*x_cf**3/3 
              - 54*x_cf**2*y_cf**2 
              - 32*x_cf**2*y_cf/5 
              + 92*x_cf**2/15 
              + 54*x_cf*y_cf**3 
              - 18*x_cf*y_cf**2/5 
              - 448*x_cf*y_cf/15 
              - 32*x_cf/15 
              - 66*y_cf**3/5 
              + 2*y_cf**2 
              + 112*y_cf/15 
              + 8/3)
    
    F = LinearForm(fes)
    F += f_expr * v * dx
    F.Assemble()
    
    # Solve
    gfu = GridFunction(fes)
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse='sparsecholesky') * F.vec
    
    ndof = fes.ndof
    print(f"NDOF = {ndof}")
    
    # Evaluate at probe points
    results = []
    for (px, py) in PROBE_POINTS:
        val = gfu(mesh(px, py))
        results.append((px, py, val))
    
    # Write CSV file
    csv_filename = f"solution_level{level}.csv"
    with open(csv_filename, 'w') as csvfile:
        csvfile.write("x, y, u\n")
        for (px, py, val) in results:
            # Format with at least 12 significant digits
            csvfile.write(f"{px:.15e}, {py:.15e}, {val:.15e}\n")
    print(f"Wrote {csv_filename} with {len(results)} probe points")
    
    # Write run log
    log_filename = f"run_level{level}.log"
    with open(log_filename, 'w') as logfile:
        logfile.write(f"NDOF = {ndof}\n")
        logfile.write(f"h = {h}\n")
        logfile.write(f"Elements = {mesh.ne}\n")
        logfile.write(f"Vertices = {mesh.nv}\n")
    print(f"Wrote {log_filename}")
    
    # Return solution values for convergence analysis
    return [r[2] for r in results], ndof

def main():
    """Main driver: solve all levels and assess convergence."""
    all_solutions = {}
    all_ndofs = {}
    
    for level, h in MESH_LEVELS:
        sol_vals, ndof = solve_level(level, h)
        all_solutions[level] = sol_vals
        all_ndofs[level] = ndof
    
    # Assess mesh independence between finest two levels (3 and 4)
    sol_coarse = np.array(all_solutions[3])  # h=1/32
    sol_fine = np.array(all_solutions[4])    # h=1/64
    
    print(f"\n{'='*60}")
    print("CONVERGENCE ANALYSIS")
    print(f"{'='*60}")
    
    # Compute absolute changes
    abs_changes = np.abs(sol_fine - sol_coarse)
    print(f"Absolute change: min={abs_changes.min():.6e}, max={abs_changes.max():.6e}, mean={abs_changes.mean():.6e}")
    
    # L2 norm comparison
    l2_coarse = np.sqrt(np.mean(sol_coarse**2))
    l2_fine = np.sqrt(np.mean(sol_fine**2))
    l2_rel_change = abs(l2_fine - l2_coarse) / (l2_coarse + 1e-30)
    print(f"L2 norm level 3: {l2_coarse:.6e}")
    print(f"L2 norm level 4: {l2_fine:.6e}")
    print(f"L2-norm relative change: {l2_rel_change:.6e}")
    
    # Max norm comparison
    max_coarse = np.max(np.abs(sol_coarse))
    max_fine = np.max(np.abs(sol_fine))
    max_rel_change = abs(max_fine - max_coarse) / (max_coarse + 1e-30)
    print(f"Max |u| level 3: {max_coarse:.6e}")
    print(f"Max |u| level 4: {max_fine:.6e}")
    print(f"Max-norm relative change: {max_rel_change:.6e}")
    
    # For pointwise relative change, use a robust definition that handles near-zero values
    # Use: |u_fine - u_coarse| / max(|u_coarse|, |u_fine|, tol) where tol is based on solution scale
    ref_scale = max(l2_coarse, 1e-10)
    rel_changes = abs_changes / np.maximum(np.maximum(np.abs(sol_coarse), np.abs(sol_fine)), ref_scale * 1e-6)
    
    max_rel_change_pointwise = np.max(rel_changes)
    mean_rel_change_pointwise = np.mean(rel_changes)
    print(f"Pointwise relative change (robust): max={max_rel_change_pointwise:.6e}, mean={mean_rel_change_pointwise:.6e}")
    
    # Judge convergence based on multiple criteria
    # The solution is considered converged if:
    # 1. L2 norm change < 1%
    # 2. Max norm change < 1%
    # 3. Mean pointwise relative change < 5%
    tol_l2 = 0.01
    tol_max = 0.01
    tol_mean = 0.05
    
    converged_l2 = l2_rel_change < tol_l2
    converged_max = max_rel_change < tol_max
    converged_mean = mean_rel_change_pointwise < tol_mean
    
    converged = converged_l2 and converged_max and converged_mean
    
    print(f"\nConvergence criteria:")
    print(f"  L2 norm change < {tol_l2}: {'PASS' if converged_l2 else 'FAIL'} ({l2_rel_change:.6e})")
    print(f"  Max norm change < {tol_max}: {'PASS' if converged_max else 'FAIL'} ({max_rel_change:.6e})")
    print(f"  Mean pointwise change < {tol_mean}: {'PASS' if converged_mean else 'FAIL'} ({mean_rel_change_pointwise:.6e})")
    
    # Use the maximum of the three relative changes as MAX_REL_CHANGE
    max_rel_change_overall = max(l2_rel_change, max_rel_change, mean_rel_change_pointwise)
    
    print(f"\nMESH_INDEPENDENCE = {'CONVERGED' if converged else 'NOT_CONVERGED'}")
    
    # Write RESULT.txt
    csv_files = "solution_level1.csv, solution_level2.csv, solution_level3.csv, solution_level4.csv"
    with open("RESULT.txt", 'w') as result_file:
        result_file.write(f"LEVELS = 4\n")
        result_file.write(f"FILES = {csv_files}\n")
        result_file.write(f"MESH_INDEPENDENCE = {'CONVERGED' if converged else 'NOT_CONVERGED'}\n")
        result_file.write(f"MAX_REL_CHANGE = {max_rel_change_overall:.15e}\n")
    
    print("\nWrote RESULT.txt")
    print("Done!")

if __name__ == "__main__":
    main()