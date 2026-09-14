#!/usr/bin/env python3
"""
Generate meshes for subdomains A and B using gmsh.
Creates .e files (Exodus format) that 4C can read.
"""

import os
import numpy as np
import subprocess
from pathlib import Path

WORK_DIR = Path("/home/alexander/workspace_coupling")
WORK_DIR.mkdir(exist_ok=True)
os.chdir(WORK_DIR)

def generate_gmsh_script(level, side):
    """Generate a GMSH script for the given mesh level and subdomain."""
    
    n_elem = level  # Number of elements in x direction
    
    if side == 'A':
        x_min, x_max = 0, 5/8
        y_min, y_max = 0, 1
        filename = f"mesh_A_level{level}.geo"
    else:
        x_min, x_max = 5/8, 1.5
        y_min, y_max = 0, 1
        filename = f"mesh_B_level{level}.geo"
    
    # Calculate number of elements in y direction to maintain aspect ratio
    if side == 'A':
        ny = int(n_elem * 1.6)
    else:
        ny = int(n_elem * 1.14)
    
    geo = f"""// Mesh for subdomain {side} at level {level}
SetFactory("OpenCASCADE");

// Define corners
Point(1) = {{{x_min}, {y_min}, 0, 1.0}};
Point(2) = {{{x_max}, {y_min}, 0, 1.0}};
Point(3) = {{{x_max}, {y_max}, 0, 1.0}};
Point(4) = {{{x_min}, {y_max}, 0, 1.0}};

// Define edges
Line(1) = {{1, 2}};
Line(2) = {{2, 3}};
Line(3) = {{3, 4}};
Line(4) = {{4, 1}};

// Create curve loop and plane surface
Curve Loop(1) = {{1, 2, 3, 4}};
Plane Surface(1) = {{1}};

// Physical groups for boundary conditions
// Bottom edge (y=0)
Physical Line("bottom") = {{1}};
// Right edge
Physical Line("right") = {{2}};
// Top edge (y=1)
Physical Line("top") = {{3}};
// Left edge
Physical Line("left") = {{4}};

// Physical volume for material
Physical Surface("material") = {{1}};

// Mesh size control
Mesh.CharacteristicLengthMin = {min((x_max-x_min)/n_elem, (y_max-y_min)/ny)};
Mesh.CharacteristicLengthMax = {max((x_max-x_min)/n_elem, (y_max-y_min)/ny)};

// Generate mesh
Mesh 2;
"""
    
    with open(filename, 'w') as f:
        f.write(geo)
    
    return filename

def run_gmsh(geo_file, output_prefix):
    """Run GMSH to generate the mesh."""
    cmd = [
        "gmsh", 
        "-2",  # 2D mesh
        "-format", "exo",  # Exodus format
        "-o", f"{output_prefix}.e",
        geo_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"GMSH stdout: {result.stdout}")
    print(f"GMSH stderr: {result.stderr}")
    return result.returncode == 0

# Test mesh generation
if __name__ == "__main__":
    for level in [8, 16, 32]:
        for side in ['A', 'B']:
            geo_file = generate_gmsh_script(level, side)
            output_prefix = f"mesh_{side}_level{level}"
            success = run_gmsh(geo_file, output_prefix)
            if success:
                print(f"Generated mesh for subdomain {side} at level {level}")
            else:
                print(f"Failed to generate mesh for subdomain {side} at level {level}")
