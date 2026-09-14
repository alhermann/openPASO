"""
Coupled thermal diffusion simulation using 4C (subdomain A) and Kratos (subdomain B).

Problem:
- Global domain: (0, 1.5) x (0, 1)
- Subdomain A: (0, 0.625) x (0, 1), k = 1
- Subdomain B: (0.625, 1.5) x (0, 1), k = 200
- Interface at x = 0.625
- Dirichlet-Neumann coupling: A is DIRICHLET side, B is NEUMANN side
- Outer boundary: u = 0 everywhere
- P1 elements in both codes

Source terms (polynomials):
Subdomain A: f_A(x,y) = -12*x^3*y/5 + 14*x^3/5 - 8979*x^2*y/2000 - 5147*x^2/12000 
                 - 12*x*y^3/5 + 42*x*y^2/5 - 1779*x*y/800 - 1007*x/1200 
                 - 2993*y^3/2000 - 5147*y^2/12000 + 4621*y/2400

Subdomain B: f_B(x,y) = -3*x^3*y/50000 + 7*x^3/100000 - 8967*x^2*y/200000 + 28769*x^2/1200000 
                 - 3*x*y^3/50000 + 21*x*y^2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 
                 - 2989*y^3/200000 + 28769*y^2/1200000 + 26803463*y/3840000 - 1468421/512000

Mesh levels: h = 1/8, 1/16, 1/32
- Level 1: nx_A = 5, ny = 8; nx_B = 7, ny = 8
- Level 2: nx_A = 10, ny = 16; nx_B = 14, ny = 16  
- Level 3: nx_A = 20, ny = 32; nx_B = 28, ny = 32
"""

import os
import json
import numpy as np
from pathlib import Path

# Problem parameters
L_A = 0.625  # Width of subdomain A
L_B = 0.875  # Width of subdomain B
H = 1.0      # Height
k_A = 1.0    # Conductivity in A
k_B = 200.0  # Conductivity in B
x_interface = 0.625

# Mesh levels
mesh_levels = [
    {"h": 1/8, "nx_A": 5, "ny": 8, "nx_B": 7},
    {"h": 1/16, "nx_A": 10, "ny": 16, "nx_B": 14},
    {"h": 1/32, "nx_A": 20, "ny": 32, "nx_B": 28}
]

# Probe points for solution evaluation
def generate_probe_points():
    """Generate probe points for both subdomains."""
    # Subdomain A: 44x44 grid, centered in each cell
    probe_A = []
    for i_y in range(44):
        for i_x in range(44):
            x = (i_x + 0.5) * L_A / 44
            y = (i_y + 0.5) * H / 44
            probe_A.append((x, y))
    
    # Subdomain B: 44x44 grid, centered in each cell
    probe_B = []
    for i_y in range(44):
        for i_x in range(44):
            x = L_A + (i_x + 0.5) * L_B / 44
            y = (i_y + 0.5) * H / 44
            probe_B.append((x, y))
    
    return probe_A, probe_B

# Interface probe points: j = 11 to 32 (22 points)
def generate_interface_probes():
    """Generate interface probe points (interior only, excluding corners)."""
    probes = []
    for j in range(11, 33):  # j = 11, 12, ..., 32
        y = (j + 0.5) / 44
        probes.append((x_interface, y))
    return probes

probe_A, probe_B = generate_probe_points()
interface_probes = generate_interface_probes()

print(f"Probe points A: {len(probe_A)}")
print(f"Probe points B: {len(probe_B)}")
print(f"Interface probes: {len(interface_probes)}")

# Save probe points for later use
with open("probe_A.json", "w") as f:
    json.dump([list(p) for p in probe_A], f)
with open("probe_B.json", "w") as f:
    json.dump([list(p) for p in probe_B], f)
with open("interface_probes.json", "w") as f:
    json.dump([list(p) for p in interface_probes], f)

print("Probe points saved.")
