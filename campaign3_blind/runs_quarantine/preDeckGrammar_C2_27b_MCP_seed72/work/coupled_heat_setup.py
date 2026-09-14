"""
Coupled heat conduction: 4C (side A) + Kratos (side B)
Dirichlet-Neumann iteration with A as Dirichlet side, B as Neumann side.

Subdomain A: (0, 0.625) x (0, 1), k = 1
Subdomain B: (0.625, 1.5) x (0, 1), k = 200
Interface at x = 0.625

Mesh levels: h = 1/8, 1/16, 1/32 (divisions = 8, 16, 32)
"""

import os
import json
import numpy as np
from pathlib import Path

# Problem parameters
L_A = 0.625  # width of subdomain A
L_B = 0.875  # width of subdomain B  
H = 1.0      # height
k_A = 1.0    # conductivity in A
k_B = 200.0  # conductivity in B
x_interface = 0.625

# Source terms (Python expressions - will convert to 4C syntax)
# Side A source term
f_A_py = "-12*x**3*y/5 + 14*x**3/5 - 8979*x**2*y/2000 - 5147*x**2/12000 - 12*x*y**3/5 + 42*x*y**2/5 - 1779*x*y/800 - 1007*x/1200 - 2993*y**3/2000 - 5147*y**2/12000 + 4621*y/2400"

# Side B source term  
f_B_py = "-3*x**3*y/50000 + 7*x**3/100000 - 8967*x**2*y/200000 + 28769*x**2/1200000 - 3*x*y**3/50000 + 21*x*y**2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 - 2989*y**3/200000 + 28769*y**2/1200000 + 26803463*y/3840000 - 1468421/512000"

def py_to_4c(expr):
    """Convert Python ** to 4C ^ for exponentiation"""
    return expr.replace("**", "^")

f_A_4c = py_to_4c(f_A_py)
f_B_4c = py_to_4c(f_B_py)

# Probe points definition
def generate_probe_points():
    """Generate probe points for both subdomains"""
    # Subdomain A: 44x44 grid
    probes_A = []
    for i_y in range(44):
        for i_x in range(44):
            x = 0 + (i_x + 0.5) * L_A / 44
            y = 0 + (i_y + 0.5) * H / 44
            probes_A.append((x, y))
    
    # Subdomain B: 44x44 grid
    probes_B = []
    for i_y in range(44):
        for i_x in range(44):
            x = L_A + (i_x + 0.5) * L_B / 44
            y = 0 + (i_y + 0.5) * H / 44
            probes_B.append((x, y))
    
    return probes_A, probes_B

# Interface probe points: j = 11, ..., 32 (22 points)
def generate_interface_probes():
    interface_probes = []
    for j in range(11, 33):  # 11 to 32 inclusive
        y = (j + 0.5) / 44
        interface_probes.append((x_interface, y))
    return interface_probes

probes_A, probes_B = generate_probe_points()
interface_probes = generate_interface_probes()

print(f"Probes A: {len(probes_A)}, Probes B: {len(probes_B)}, Interface: {len(interface_probes)}")

# Save probe points for later use
with open("probes_A.json", "w") as f:
    json.dump([list(p) for p in probes_A], f)
with open("probes_B.json", "w") as f:
    json.dump([list(p) for p in probes_B], f)
with open("interface_probes.json", "w") as f:
    json.dump([list(p) for p in interface_probes], f)

print("Probe points saved.")
