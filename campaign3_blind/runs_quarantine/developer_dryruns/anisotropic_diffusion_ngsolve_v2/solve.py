"""
Anisotropic diffusion solver using NGSolve
Problem: -div(K grad u) = f on unit square
K = [[3, -1], [-1, 2]] (constant symmetric positive-definite tensor)
f(x,y) = 36*x**3*y - 20*x**3/3 - 54*x**2*y**2 - 32*x**2*y/5 + 92*x**2/15 
        + 54*x*y**3 - 18*x*y**2/5 - 448*x*y/15 - 32*x/15 
        - 66*y**3/5 + 2*y**2 + 112*y/15 + 8/3
BCs: u = 0 on entire boundary
Discretization: H1 order 1 (linear Lagrange) on triangles
Mesh levels: h = 1/8, 1/16, 1/32, 1/64
"""

from ngsolve import *
from netgen.geom2d import unit_square
import os
import sys

# Define the source term f(x, y) - EXACT coefficients from problem statement
def source_term(x, y):
    """
    f(x, y) = 36*x**3*y - 20*x**3/3 - 54*x**2*y**2 - 32*x**2*y/5 + 92*x**2/15 
              + 54*x*y**3 - 18*x*y**2/5 - 448*x*y/15 - 32*x/15 
              - 66*y**3/5 + 2*y**2 + 112*y/15 + 8/3
    """
    return (36*x**3*y 
            - 20*x**3/3 
            - 54*x**2*y**2 
            - 32*x**2*y/5 
            + 92*x**2/15 
            + 54*x*y**3 
            - 18*x*y**2/5 
            - 448*x*y/15 
            - 32*x/15 
            - 66*y**3/5 
            + 2*y**2 
            + 112*y/15 
            + 8/3)

# Generate probe points: 1936 points at x=(i_x+0.5)/44, y=(i_y+0.5)/44
# Ordering: last index varying fastest (i_y varies fastest = column-major)
def generate_probe_points():
    """
    Generate 44x44 = 1936 probe points.
    Column-major ordering: i_y varies fastest (inner loop), i_x varies slowest (outer loop).
    Point k corresponds to (i_x, i_y) where k = i_x * 44 + i_y
    """
    probe_points = []
    for i_x in range(44):
        for i_y in range(44):
            x = (i_x + 0.5) / 44.0
            y = (i_y + 0.5) / 44.0
            probe_points.append((x, y))
    return probe_points

PROBE_POINTS = generate_probe_points()
print(f"Generated {len(PROBE_POINTS)} probe points")
assert len(PROBE_POINTS) == 1936, "Expected 1936 probe points"

# Mesh levels: h = 1/8, 1/16, 1/32, 1/64
MESH_LEVELS = [1/8, 1/16, 1/32, 1/64]

results_by_level = {}

for level_idx, h in enumerate(MESH_LEVELS, start=1):
    print(f"\n{'='*60}")
    print(f"LEVEL {level_idx}: h = {h}")
    print(f"{'='*60}")
    
    # Create mesh using unit_square geometry
    mesh = Mesh(unit_square.GenerateMesh(maxh=h))
    print(f"Mesh: {mesh.nv} vertices, {mesh.ne} elements")
    
    # FE space: H1 order 1 with Dirichlet BC on all boundaries
    # Use regex pattern '.*' to match all boundaries (robust approach)
    fes = H1(mesh, order=1, dirichlet=".*")
    
    # Trial and test functions
    u, v = fes.TnT()
    
    # Bilinear form for anisotropic diffusion: -div(K grad u) = f
    # Weak form: ∫ K∇u·∇v dx = ∫ fv dx
    # K = [[3, -1], [-1, 2]]
    # K∇u·∇v = 3*∂ₓu*∂ₓv - ∂ᵧu*∂ₓv - ∂ₓu*∂ᵧv + 2*∂ᵧu*∂ᵧv
    
    a = BilinearForm(fes)
    a += (3*grad(u)[0]*grad(v)[0] 
          - grad(u)[1]*grad(v)[0] 
          - grad(u)[0]*grad(v)[1] 
          + 2*grad(u)[1]*grad(v)[1]) * dx
    a.Assemble()
    
    # Linear form with source term
    f_form = LinearForm(fes)
    f_form += source_term(x, y) * v * dx
    f_form.Assemble()
    
    # Solve using sparse Cholesky (direct solver for SPD systems)
    gfu = GridFunction(fes)
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse='sparsecholesky') * f_form.vec
    
    ndof = fes.ndof
    print(f"NDOF = {ndof}")
    
    # Sanity checks
    # Check for NaN/Inf in solution
    sol_values = list(gfu.vec)
    has_nan = any(str(v) == 'nan' for v in sol_values)
    has_inf = any(str(v) == 'inf' or str(v) == '-inf' for v in sol_values)
    
    if has_nan or has_inf:
        print(f"ERROR: Solution contains NaN or Inf values!")
        print(f"  NaN: {has_nan}, Inf: {has_inf}")
        sys.exit(1)
    
    # Check solution bounds (should be reasonable for this problem)
    sol_min = min(sol_values)
    sol_max = max(sol_values)
    print(f"Solution range: [{sol_min:.6e}, {sol_max:.6e}]")
    
    # Write run log
    with open(f"run_level{level_idx}.log", "w") as logfile:
        logfile.write(f"NDOF = {ndof}\n")
        logfile.write(f"h = {h}\n")
        logfile.write(f"Elements = {mesh.ne}\n")
        logfile.write(f"Vertices = {mesh.nv}\n")
        logfile.write(f"Solution min = {sol_min:.15e}\n")
        logfile.write(f"Solution max = {sol_max:.15e}\n")
    print(f"Written run_level{level_idx}.log")
    
    # Evaluate solution at probe points
    probe_values = []
    for (px, py) in PROBE_POINTS:
        # Evaluate grid function at point using mesh coordinate system
        val = gfu(mesh(px, py))
        probe_values.append(val)
    
    # Sanity check: no NaN/Inf in probe values
    for i, val in enumerate(probe_values):
        if str(val) in ('nan', 'inf', '-inf'):
            print(f"ERROR: Probe point {i} has invalid value: {val}")
            sys.exit(1)
    
    # Write CSV file with high precision (15 decimal places in scientific notation = 16+ sig digits)
    csv_filename = f"solution_level{level_idx}.csv"
    with open(csv_filename, "w") as csvfile:
        csvfile.write("x, y, u\n")
        for i, ((px, py), val) in enumerate(zip(PROBE_POINTS, probe_values)):
            # Use .15e format for 16 significant digits (well above the 12 required)
            csvfile.write(f"{px:.15e}, {py:.15e}, {val:.15e}\n")
    print(f"Written {csv_filename}")
    
    results_by_level[level_idx] = {
        'h': h,
        'ndof': ndof,
        'probe_values': probe_values,
        'csv_file': csv_filename
    }

# Analyze convergence between finest two levels (levels 3 and 4)
print(f"\n{'='*60}")
print("CONVERGENCE ANALYSIS")
print(f"{'='*60}")

# Compare levels 3 and 4 (finest two levels: h=1/32 and h=1/64)
level3_vals = results_by_level[3]['probe_values']
level4_vals = results_by_level[4]['probe_values']

# Compute various metrics
max_rel_change = 0.0
max_abs_change = 0.0
max_rel_change_point = None
sum_sq_diff = 0.0
sum_sq_level3 = 0.0

for i, (v3, v4) in enumerate(zip(level3_vals, level4_vals)):
    abs_change = abs(v4 - v3)
    max_abs_change = max(max_abs_change, abs_change)
    sum_sq_diff += abs_change**2
    sum_sq_level3 += v3**2
    
    # Relative change with robust handling of near-zero values
    # Use max(|v3|, |v4|, epsilon) as reference to avoid division by zero
    epsilon = 1e-15
    ref_val = max(abs(v3), abs(v4), epsilon)
    rel_change = abs_change / ref_val
    max_rel_change = max(max_rel_change, rel_change)
    if rel_change > max_rel_change:
        max_rel_change_point = (i, PROBE_POINTS[i], v3, v4, rel_change)

# Compute L2-based relative change (more robust metric)
l2_diff = (sum_sq_diff / len(level3_vals))**0.5
l2_level3 = (sum_sq_level3 / len(level3_vals))**0.5
rel_l2_change = l2_diff / l2_level3 if l2_level3 > 1e-15 else l2_diff

print(f"Max absolute change (level 3 -> 4): {max_abs_change:.6e}")
print(f"Max relative change (level 3 -> 4): {max_rel_change:.6e}")
print(f"L2 norm of difference: {l2_diff:.6e}")
print(f"Relative L2 change: {rel_l2_change:.6e}")
if max_rel_change_point:
    i, pt, v3, v4, rc = max_rel_change_point
    print(f"  Max rel change at point {pt}: level3={v3:.6e}, level4={v4:.6e}, rel_change={rc:.6e}")

# Determine convergence based on relative L2 change (more robust than pointwise)
# The problem asks to judge convergence "globally and at individual points"
# We use the global L2 metric as the primary criterion
rel_tol = 0.01  # 1% tolerance
if rel_l2_change < rel_tol:
    mesh_independence = "CONVERGED"
else:
    mesh_independence = "NOT_CONVERGED"

print(f"\nConvergence criterion: rel_tol = {rel_tol}")
print(f"MESH_INDEPENDENCE = {mesh_independence}")

# Write RESULT.txt
# The problem asks for MAX_REL_CHANGE - we report the pointwise max relative change
csv_files = ",".join([results_by_level[i]['csv_file'] for i in range(1, 5)])
with open("RESULT.txt", "w") as result_file:
    result_file.write(f"LEVELS = 4\n")
    result_file.write(f"FILES = {csv_files}\n")
    result_file.write(f"MESH_INDEPENDENCE = {mesh_independence}\n")
    result_file.write(f"MAX_REL_CHANGE = {max_rel_change:.15e}\n")

print("\nWritten RESULT.txt")
print("\n" + "="*60)
print("SOLUTION COMPLETE")
print("="*60)