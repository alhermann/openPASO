#!/usr/bin/env python3
"""
Coupled heat conduction simulation using 4C (subdomain A) and Kratos (subdomain B).
Dirichlet-Neumann coupling: A receives T from B (Dirichlet side), B receives flux from A (Neumann side).
"""

import os
import json
import numpy as np
from pathlib import Path

# Problem parameters
INTERFACE_X = 5/8  # 0.625
DOMAIN_A_WIDTH = INTERFACE_X  # 0 to 0.625
DOMAIN_B_WIDTH = 1.5 - INTERFACE_X  # 0.625 to 1.5
DOMAIN_HEIGHT = 1.0

K_A = 1.0   # thermal conductivity in subdomain A
K_B = 200.0 # thermal conductivity in subdomain B

# Source terms (converted from Python ** to ^ for 4C)
SOURCE_A_PYTHON = "-12*x**3*y/5 + 14*x**3/5 - 8979*x**2*y/2000 - 5147*x**2/12000 - 12*x*y**3/5 + 42*x*y**2/5 - 1779*x*y/800 - 1007*x/1200 - 2993*y**3/2000 - 5147*y**2/12000 + 4621*y/2400"
SOURCE_B_PYTHON = "-3*x**3*y/50000 + 7*x**3/100000 - 8967*x**2*y/200000 + 28769*x**2/1200000 - 3*x*y**3/50000 + 21*x*y**2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 - 2989*y**3/200000 + 28769*y**2/1200000 + 26803463*y/3840000 - 1468421/512000"

# For 4C, convert ** to ^
SOURCE_A_4C = SOURCE_A_PYTHON.replace("**", "^")
SOURCE_B_4C = SOURCE_B_PYTHON.replace("**", "^")

# Mesh levels: h = 1/8, 1/16, 1/32
MESH_LEVELS = [8, 16, 32]  # divisions per unit length

# Probe points definition
def generate_probe_points_A():
    """Generate 1936 probe points for subdomain A"""
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = 0 + (i_x + 0.5) * DOMAIN_A_WIDTH / 44
            y = 0 + (i_y + 0.5) * DOMAIN_HEIGHT / 44
            points.append((x, y))
    return points

def generate_probe_points_B():
    """Generate 1936 probe points for subdomain B"""
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = INTERFACE_X + (i_x + 0.5) * DOMAIN_B_WIDTH / 44
            y = 0 + (i_y + 0.5) * DOMAIN_HEIGHT / 44
            points.append((x, y))
    return points

def generate_interface_probe_points():
    """Generate 22 interface probe points at x=5/8, y=(j+0.5)/44 for j=11..32"""
    points = []
    for j in range(11, 33):  # j = 11, 12, ..., 32
        x = INTERFACE_X
        y = (j + 0.5) / 44
        points.append((x, y))
    return points

PROBE_POINTS_A = generate_probe_points_A()
PROBE_POINTS_B = generate_probe_points_B()
INTERFACE_PROBE_POINTS = generate_interface_probe_points()

print(f"Probe points A: {len(PROBE_POINTS_A)}")
print(f"Probe points B: {len(PROBE_POINTS_B)}")
print(f"Interface probe points: {len(INTERFACE_PROBE_POINTS)}")

# Create work directory structure
WORK_DIR = Path("/tmp/coupled_heat_simulation")
WORK_DIR.mkdir(exist_ok=True)

for level_idx, n_divisions in enumerate(MESH_LEVELS):
    level_dir = WORK_DIR / f"level_{level_idx+1}"
    level_dir.mkdir(exist_ok=True)
    
    # Subdomain A work dir
    work_a = level_dir / "subdomain_A"
    work_a.mkdir(exist_ok=True)
    
    # Subdomain B work dir  
    work_b = level_dir / "subdomain_B"
    work_b.mkdir(exist_ok=True)
    
    print(f"Created directories for level {level_idx+1} (n={n_divisions})")

print("\nDirectory structure created successfully!")
