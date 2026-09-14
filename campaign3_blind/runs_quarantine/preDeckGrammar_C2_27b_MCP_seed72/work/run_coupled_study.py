#!/usr/bin/env python3
"""
Run coupled heat conduction study for all three mesh levels.
Uses OASiS couple() tool to coordinate 4C (side A) and Kratos (side B).
"""

import os
import json
import numpy as np
from pathlib import Path
import shutil

# Base directory
BASE_DIR = Path("/home/alexander/Schreibtisch/ofa-v2/campaign3_blind/runs/C2_27b_MCP_seed72/work")

# Mesh levels: divisions = 8, 16, 32 (h = 1/8, 1/16, 1/32)
LEVELS = [8, 16, 32]

# Probe points
def generate_probe_points():
    L_A, L_B, H = 0.625, 0.875, 1.0
    
    # Subdomain A probes
    probes_A = []
    for i_y in range(44):
        for i_x in range(44):
            x = 0 + (i_x + 0.5) * L_A / 44
            y = 0 + (i_y + 0.5) * H / 44
            probes_A.append((x, y))
    
    # Subdomain B probes
    probes_B = []
    for i_y in range(44):
        for i_x in range(44):
            x = L_A + (i_x + 0.5) * L_B / 44
            y = 0 + (i_y + 0.5) * H / 44
            probes_B.append((x, y))
    
    return probes_A, probes_B

# Interface probe points: j = 11, ..., 32
def generate_interface_probes():
    interface_probes = []
    for j in range(11, 33):
        y = (j + 0.5) / 44
        interface_probes.append((0.625, y))
    return interface_probes

probes_A, probes_B = generate_probe_points()
interface_probes = generate_interface_probes()

print(f"Probes A: {len(probes_A)}, Probes B: {len(probes_B)}, Interface: {len(interface_probes)}")

# Store results
all_results = {}

for level_idx, divisions in enumerate(LEVELS, 1):
    print(f"\n{'='*60}")
    print(f"Running mesh level {level_idx}: divisions={divisions}")
    print(f"{'='*60}")
    
    # Create work directories
    dir_A = BASE_DIR / f"level{level_idx}_A"
    dir_B = BASE_DIR / f"level{level_idx}_B"
    
    for d in [dir_A, dir_B]:
        d.mkdir(exist_ok=True)
        # Copy participant scripts
        shutil.copy(BASE_DIR / "participant_A.py", d / "participant_A.py")
        shutil.copy(BASE_DIR / "participant_B.py", d / "participant_B.py")
        # Write level.json
        with open(d / "level.json", "w") as f:
            json.dump({"divisions": divisions}, f)
    
    # Test run each participant standalone first
    print("Testing participant A...")
    os.system(f"cd {dir_A} && python participant_A.py 2>&1 | tail -5")
    
    print("Testing participant B...")
    os.system(f"cd {dir_B} && python participant_B.py 2>&1 | tail -5")
    
    # Check if exports were created
    exp_A = dir_A / "exports.json"
    exp_B = dir_B / "exports.json"
    
    if not exp_A.exists() or not exp_B.exists():
        print(f"ERROR: exports.json not created for level {level_idx}")
        continue
    
    # Read exports to verify
    with open(exp_A) as f:
        data_A = json.load(f)
    with open(exp_B) as f:
        data_B = json.load(f)
    
    print(f"Participant A exported {data_A['n_points']} points")
    print(f"Participant B exported {data_B['n_points']} points")
    
    # Save solution data
    sol_A = np.load(dir_A / "solution_A.npz")
    sol_B = np.load(dir_B / "solution_B.npz")
    
    all_results[level_idx] = {
        "divisions": divisions,
        "coords_A": sol_A["coords"],
        "values_A": sol_A["values"],
        "coords_B": sol_B["coords"],
        "values_B": sol_B["values"],
        "iface_flux_A": data_A["normal_fluxes"],
        "iface_temp_B": data_B["values"],
        "iface_coords": data_A["coordinates"]
    }
    
    print(f"Level {level_idx} completed successfully")

print("\n" + "="*60)
print("All levels completed. Writing output files...")
print("="*60)

# This would be called from within OASiS context
# For now, just save intermediate results
with open(BASE_DIR / "intermediate_results.json", "w") as f:
    # Convert numpy arrays to lists for JSON serialization
    results_json = {}
    for k, v in all_results.items():
        results_json[str(k)] = {
            "divisions": v["divisions"],
            "n_points_A": len(v["values_A"]),
            "n_points_B": len(v["values_B"])
        }
    json.dump(results_json, f, indent=2)

print("Intermediate results saved.")
