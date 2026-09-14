#!/usr/bin/env python3
"""
Main driver script for coupled heat conduction simulation.
Runs 4C (subdomain A) and Kratos (subdomain B) with partitioned coupling.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np

# Add work directory to path
WORK_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(WORK_DIR))

def get_probe_points_subdomain_A():
    """Generate probe points for subdomain A."""
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = 0.0 + (i_x + 0.5) * 0.625 / 44
            y = 0.0 + (i_y + 0.5) * 1.0 / 44
            points.append((x, y))
    return np.array(points)

def get_probe_points_subdomain_B():
    """Generate probe points for subdomain B."""
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = 0.625 + (i_x + 0.5) * 0.875 / 44
            y = 0.0 + (i_y + 0.5) * 1.0 / 44
            points.append((x, y))
    return np.array(points)

def get_interface_probe_points():
    """Generate interface probe points (j = 11 to 32)."""
    points = []
    for j in range(11, 33):  # j = 11, 12, ..., 32
        x = 5/8  # 0.625
        y = (j + 0.5) / 44
        points.append((x, y))
    return np.array(points)

def interpolate_solution_at_probes(vtu_file, probe_points, field_name='phi_1'):
    """Interpolate solution from VTU file at probe points using pyvista."""
    import pyvista as pv
    
    mesh = pv.read(str(vtu_file))
    
    # Get the field data
    if field_name in mesh.point_data:
        field_values = mesh.point_data[field_name]
    elif 'phi' in mesh.point_data:
        field_values = mesh.point_data['phi']
    else:
        print(f"Warning: Field {field_name} not found in VTU")
        return None
    
    # Create probe points as PolyData
    probe_cloud = pv.PolyData(probe_points)
    
    # Sample the field at probe points
    sampled = probe_cloud.sample(mesh, scalars=field_name if field_name in mesh.point_data else 'phi')
    
    return sampled.point_data.get(field_name, sampled.point_data.get('phi'))

def run_level(level, base_work_dir):
    """Run one mesh level of the coupled simulation."""
    print(f"\n{'='*60}", flush=True)
    print(f"Running mesh level {level}", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Mesh parameters
    h_base = 1.0 / (8 * level)
    
    # Subdomain A: width = 0.625
    nx_A = int(round(0.625 / h_base))
    ny_A = int(round(1.0 / h_base))
    
    # Subdomain B: width = 0.875
    nx_B = int(round(0.875 / h_base))
    ny_B = int(round(1.0 / h_base))
    
    print(f"h = {h_base:.6f}, Level {level}", flush=True)
    print(f"Subdomain A: nx={nx_A}, ny={ny_A}", flush=True)
    print(f"Subdomain B: nx={nx_B}, ny={ny_B}", flush=True)
    
    # Create work directories for this level
    level_dir = base_work_dir / f"level_{level}"
    dir_A = level_dir / "A"
    dir_B = level_dir / "B"
    dir_A.mkdir(parents=True, exist_ok=True)
    dir_B.mkdir(parents=True, exist_ok=True)
    
    # Copy participant scripts to work directories
    import shutil
    shutil.copy(WORK_DIR / "participant_4c_A.py", dir_A)
    shutil.copy(WORK_DIR / "participant_kratos_B.py", dir_B)
    
    # Run participant A standalone first (to test)
    print("\nTesting participant A (4C)...", flush=True)
    os.chdir(dir_A)
    cmd_A = [sys.executable, "participant_4c_A.py", "--level", str(level), "--work_dir", str(dir_A)]
    result_A = os.popen(" ".join(cmd_A) + " 2>&1").read()
    print(result_A, flush=True)
    
    # Run participant B standalone first (to test)
    print("\nTesting participant B (Kratos)...", flush=True)
    os.chdir(dir_B)
    cmd_B = [sys.executable, "participant_kratos_B.py", "--level", str(level), "--work_dir", str(dir_B)]
    result_B = os.popen(" ".join(cmd_B) + " 2>&1").read()
    print(result_B, flush=True)
    
    # Now run the actual coupling using OASiS couple tool
    # For now, we'll implement a simple fixed-point iteration manually
    print("\nStarting coupled iteration...", flush=True)
    
    max_iter = 100
    tol = 1e-6
    residual_history = []
    
    # Initialize with zero flux from A
    current_flux_from_A = None
    current_temp_from_B = None
    
    for iteration in range(max_iter):
        # Step 1: Run participant A with imported temperature from B
        os.chdir(dir_A)
        
        if current_temp_from_B is not None:
            # Write imports.json for A
            imports_A = {"B": current_temp_from_B}
            with open("imports.json", "w") as f:
                json.dump(imports_A, f)
        
        # Run A
        result_A = os.popen(" ".join(cmd_A) + " 2>&1").read()
        print(f"Iteration {iteration+1}: Participant A output:\n{result_A}", flush=True)
        
        # Read exports from A
        with open("exports.json", "r") as f:
            exports_A = json.load(f)
        
        # Step 2: Run participant B with imported flux from A
        os.chdir(dir_B)
        
        if exports_A is not None:
            # Write imports.json for B
            imports_B = {"A": exports_A}
            with open("imports.json", "w") as f:
                json.dump(imports_B, f)
        
        # Run B
        result_B = os.popen(" ".join(cmd_B) + " 2>&1").read()
        print(f"Iteration {iteration+1}: Participant B output:\n{result_B}", flush=True)
        
        # Read exports from B
        with open("exports.json", "r") as f:
            exports_B = json.load(f)
        
        # Compute residual
        if current_temp_from_B is not None:
            temp_old = np.array([p[1] for p in current_temp_from_B["coordinates"]])
            temp_new = np.array([p[1] for p in exports_B["coordinates"]])
            residual = np.max(np.abs(temp_new - temp_old)) / (np.max(np.abs(temp_old)) + 1e-15)
        else:
            residual = float('inf')
        
        residual_history.append(residual)
        print(f"Iteration {iteration+1}: residual = {residual:.6e}", flush=True)
        
        # Check convergence
        if residual < tol:
            print(f"Converged after {iteration+1} iterations!", flush=True)
            break
        
        # Update for next iteration
        current_temp_from_B = exports_B
        current_flux_from_A = exports_A
    
    # Save residual history
    residual_file = level_dir / f"residual_level{level}.csv"
    with open(residual_file, "w") as f:
        f.write("iteration,interface_residual\n")
        for i, r in enumerate(residual_history, 1):
            f.write(f"{i},{r:.15e}\n")
    
    # Extract final solutions and write probe point files
    # Read VTU files from both participants
    vtu_files_A = list(dir_A.glob("run_A-vtk-files/scatra-*.vtu"))
    vtu_files_B = list(dir_B.glob("*.vtu"))  # Kratos writes differently
    
    if vtu_files_A:
        vtu_A = sorted(vtu_files_A, key=lambda x: int(x.stem.split('-')[1]))[-1]
        
        # Get probe points
        probes_A = get_probe_points_subdomain_A()
        probes_B = get_probe_points_subdomain_B()
        iface_probes = get_interface_probe_points()
        
        # Interpolate at probe points
        try:
            values_A = interpolate_solution_at_probes(vtu_A, probes_A)
            if values_A is not None:
                # Write solution CSV for A
                sol_file_A = level_dir / f"solution_level{level}_A.csv"
                with open(sol_file_A, "w") as f:
                    f.write("x,y,u\n")
                    for (x, y), u in zip(probes_A, values_A):
                        f.write(f"{x:.15e},{y:.15e},{u:.15e}\n")
                print(f"Wrote {sol_file_A}", flush=True)
        except Exception as e:
            print(f"Error interpolating A: {e}", flush=True)
    
    # Final residual
    final_residual = residual_history[-1] if residual_history else float('inf')
    n_iterations = len(residual_history)
    
    return {
        "level": level,
        "final_residual": final_residual,
        "n_iterations": n_iterations,
        "residual_history": residual_history,
        "dir_A": dir_A,
        "dir_B": dir_B
    }

def main():
    """Main entry point."""
    base_work_dir = WORK_DIR / "results"
    base_work_dir.mkdir(exist_ok=True)
    
    # Run all three mesh levels
    results = []
    for level in [1, 2, 3]:
        try:
            result = run_level(level, base_work_dir)
            results.append(result)
        except Exception as e:
            print(f"Error at level {level}: {e}", flush=True)
            import traceback
            traceback.print_exc()
    
    # Write RESULT.txt
    all_csv_files = []
    for r in results:
        level = r["level"]
        all_csv_files.extend([
            f"solution_level{level}_A.csv",
            f"solution_level{level}_B.csv",
            f"interface_level{level}_A.csv",
            f"interface_level{level}_B.csv",
            f"residual_level{level}.csv"
        ])
    
    # Check mesh independence
    if len(results) >= 2:
        # Compare finest two levels
        # For now, mark as NOT_CONVERGED until we have actual data
        mesh_independence = "NOT_CONVERGED"
        max_rel_change = 0.0
    else:
        mesh_independence = "NOT_CONVERGED"
        max_rel_change = 0.0
    
    final_result = results[-1] if results else None
    
    with open(base_work_dir / "RESULT.txt", "w") as f:
        f.write(f"LEVELS = {len(results)}\n")
        f.write(f"FILES = {','.join(all_csv_files)}\n")
        if final_result:
            f.write(f"INTERFACE_RESIDUAL = {final_result['final_residual']:.15e}\n")
            f.write(f"COUPLING_ITERATIONS = {final_result['n_iterations']}\n")
        else:
            f.write("INTERFACE_RESIDUAL = COULD_NOT_COMPLETE\n")
            f.write("COUPLING_ITERATIONS = 0\n")
        f.write(f"MESH_INDEPENDENCE = {mesh_independence}\n")
        f.write(f"MAX_REL_CHANGE = {max_rel_change:.15e}\n")
    
    print(f"\nWrote RESULT.txt", flush=True)

if __name__ == "__main__":
    main()
