"""
Anisotropic diffusion: -div(K grad u) = f on unit square
K = [[3, -1], [-1, 2]] (symmetric positive definite)
Homogeneous Dirichlet BCs everywhere
H1 order=1 (linear Lagrange) on triangles
Mesh levels: h = 1/8, 1/16, 1/32, 1/64
"""
from ngsolve import *
import json
import os

# Define the anisotropic diffusion tensor K
# K = [[3, -1], [-1, 2]]
# Weak form: ∫ K*grad(u)·grad(v) dx = ∫(3*u_x*v_x - u_x*v_y - u_y*v_x + 2*u_y*v_y) dx

# Source term f(x,y) - polynomial of degree 3
def source_term(x, y):
    return (36*x**3*y - 20*x**3/3 - 54*x**2*y**2 - 32*x**2*y/5 
            + 92*x**2/15 + 54*x*y**3 - 18*x*y**2/5 - 448*x*y/15 
            - 32*x/15 - 66*y**3/5 + 2*y**2 + 112*y/15 + 8/3)

# Probe points: 1936 points
# x = (i_x+0.5)/44, y = (i_y+0.5)/44 for i_x, i_y = 0..43
# Ordered with last index (i_y) varying fastest = column-major order
# This means: for each i_x (outer), iterate all i_y (inner, varies fastest)
def generate_probe_points():
    probe_points = []
    for i_x in range(44):      # outer loop
        for i_y in range(44):  # inner loop (varies fastest)
            x = (i_x + 0.5) / 44.0
            y = (i_y + 0.5) / 44.0
            probe_points.append((x, y))
    return probe_points

probe_points = generate_probe_points()
print(f"Generated {len(probe_points)} probe points")

# Mesh levels: h = 1/8, 1/16, 1/32, 1/64
# For uniform mesh, maxh = h
mesh_levels = [1/8, 1/16, 1/32, 1/64]

results = {}
max_rel_change_overall = 0.0

for level_idx, h in enumerate(mesh_levels, start=1):
    print(f"\n{'='*60}")
    print(f"MESH LEVEL {level_idx}: h = {h}")
    print(f"{'='*60}")
    
    # Create mesh using NetGen's unit_square
    from netgen.geom2d import unit_square
    mesh = Mesh(unit_square.GenerateMesh(maxh=h))
    print(f"Mesh: {mesh.ne} elements, {mesh.nv} vertices")
    
    # FE space: H1 order=1 with homogeneous Dirichlet on ALL boundaries
    # Use lowercase boundary names as per NetGen convention
    fes = H1(mesh, order=1, dirichlet="left|right|top|bottom")
    
    # Verify Dirichlet BCs are applied correctly
    free_dofs_count = sum(1 for d in fes.FreeDofs())
    print(f"Total DOFs: {fes.ndof}, Free DOFs: {free_dofs_count}")
    
    # Trial and test functions
    u, v = fes.TnT()
    
    # Bilinear form: a(u,v) = ∫ K*grad(u)·grad(v) dx
    # K = [[3, -1], [-1, 2]]
    # grad(u) = [u_x, u_y]^T
    # K*grad(u) = [3*u_x - u_y, -u_x + 2*u_y]^T
    # K*grad(u)·grad(v) = (3*u_x - u_y)*v_x + (-u_x + 2*u_y)*v_y
    #                   = 3*u_x*v_x - u_y*v_x - u_x*v_y + 2*u_y*v_y
    # In NGSolve: grad(u)[0] is u_x, grad(u)[1] is u_y
    a = BilinearForm(fes)
    a += (3*grad(u)[0]*grad(v)[0] - grad(u)[1]*grad(v)[0] - grad(u)[0]*grad(v)[1] + 2*grad(u)[1]*grad(v)[1])*dx
    a.Assemble()
    
    # Linear form: f(v) = ∫ f(x,y)*v dx
    f_form = LinearForm(fes)
    f_form += source_term(x, y) * v * dx
    f_form.Assemble()
    
    # Solve
    gfu = GridFunction(fes)
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse='sparsecholesky') * f_form.vec
    
    # Write run log
    with open(f"run_level{level_idx}.log", "w") as logfile:
        logfile.write(f"NDOF = {fes.ndof}\n")
        logfile.write(f"h = {h}\n")
        logfile.write(f"Elements: {mesh.ne}\n")
        logfile.write(f"Vertices: {mesh.nv}\n")
        logfile.write(f"Free DOFs: {free_dofs_count}\n")
    print(f"Written run_level{level_idx}.log")
    
    # Evaluate solution at probe points
    # Correct syntax: gfu(x, y) directly
    u_values = []
    for (px, py) in probe_points:
        u_val = gfu(px, py)
        u_values.append(u_val)
    
    # Write CSV file with high precision (15 significant digits in scientific notation)
    csv_filename = f"solution_level{level_idx}.csv"
    with open(csv_filename, "w") as csvfile:
        csvfile.write("x,y,u\n")
        for i, ((px, py), u_val) in enumerate(zip(probe_points, u_values)):
            csvfile.write(f"{px:.15e},{py:.15e},{u_val:.15e}\n")
    print(f"Written {csv_filename}")
    
    # Store results for convergence analysis
    results[level_idx] = {
        'h': h,
        'ndof': fes.ndof,
        'u_values': u_values
    }
    
    # Compute relative change from previous level (if exists)
    if level_idx > 1:
        prev_u = results[level_idx-1]['u_values']
        rel_changes = []
        for i in range(len(probe_points)):
            if abs(prev_u[i]) > 1e-15:  # Avoid division by zero
                rel_change = abs(u_values[i] - prev_u[i]) / abs(prev_u[i])
                rel_changes.append(rel_change)
        
        max_rel_change = max(rel_changes) if rel_changes else 0.0
        avg_rel_change = sum(rel_changes) / len(rel_changes) if rel_changes else 0.0
        
        print(f"Max relative change from level {level_idx-1}: {max_rel_change:.6e}")
        print(f"Avg relative change from level {level_idx-1}: {avg_rel_change:.6e}")
        
        if level_idx == 4:
            max_rel_change_overall = max_rel_change

# Determine mesh independence
# CONVERGED if max relative change between finest two levels < 1% (0.01)
rel_tol = 0.01
converged = max_rel_change_overall < rel_tol
mesh_independence = "CONVERGED" if converged else "NOT_CONVERGED"

print(f"\n{'='*60}")
print(f"FINAL RESULTS")
print(f"{'='*60}")
print(f"Levels solved: {len(results)}")
print(f"Max relative change (levels 3->4): {max_rel_change_overall:.6e}")
print(f"Mesh independence: {mesh_independence}")

# Write RESULT.txt
csv_files = ", ".join([f"solution_level{k}.csv" for k in range(1, len(results)+1)])
with open("RESULT.txt", "w") as resultfile:
    resultfile.write(f"LEVELS = {len(results)}\n")
    resultfile.write(f"FILES = {csv_files}\n")
    resultfile.write(f"MESH_INDEPENDENCE = {mesh_independence}\n")
    resultfile.write(f"MAX_REL_CHANGE = {max_rel_change_overall:.15e}\n")

print("\nWritten RESULT.txt")
print("Simulation complete!")