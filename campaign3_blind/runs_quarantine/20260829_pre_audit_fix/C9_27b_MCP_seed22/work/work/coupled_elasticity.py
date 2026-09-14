"""
Complete coupled elasticity simulation using NGSolve (subdomain A) and scikit-fem (subdomain B).
This script sets up and runs the coupling for all three mesh levels.
"""
import json
import os
import numpy as np
from pathlib import Path

# Working directory
WORK_DIR = Path("/home/alexander/Schreibtisch/ofa-v2/campaign3_blind/runs/C9_27b_MCP_seed22/work")

# Problem parameters
X0, X1 = 0.0, 1.0
Y_A = 0.625  # interface y-coordinate
Y_B = 1.5    # top of domain B

# Material properties
LAMBDA_A, MU_A = 480.0, 1200.0  # subdomain A
LAMBDA_B, MU_B = 480.0, 240.0   # subdomain B

# Mesh levels
LEVELS = [
    {"level": 1, "nx_a": 8, "ny_a": 5, "nx_b": 8, "ny_b": 7},   # h ~ 1/8
    {"level": 2, "nx_a": 16, "ny_a": 10, "nx_b": 16, "ny_b": 14}, # h ~ 1/16
    {"level": 3, "nx_a": 32, "ny_a": 20, "nx_b": 32, "ny_b": 28}, # h ~ 1/32
]

# Probe points for subdomain A
def generate_probe_points_A():
    """Generate 1936 probe points for subdomain A."""
    points = []
    for i_y in range(44):
        for i_x in range(44):
            x = (i_x + 0.5) / 44
            y = (i_y + 0.5) * 0.625 / 44
            points.append((x, y))
    return points

# Probe points for subdomain B
def generate_probe_points_B():
    """Generate 1936 probe points for subdomain B."""
    points = []
    for i_y in range(44):
        for i_x in range(44):
            x = (i_x + 0.5) / 44
            y = 0.625 + (i_y + 0.5) * 0.875 / 44
            points.append((x, y))
    return points

# Interface probe points (44 points)
def generate_interface_probes():
    """Generate 44 interface probe points."""
    points = []
    for i in range(44):
        x = 0.25 + (i + 0.5) * 0.5 / 44
        y = 5/8
        points.append((x, y))
    return points

print("Generating probe points...")
probes_A = generate_probe_points_A()
probes_B = generate_probe_points_B()
interface_probes = generate_interface_probes()
print(f"Probe points A: {len(probes_A)}, B: {len(probes_B)}, interface: {len(interface_probes)}")

# Source terms
def f_A(x, y):
    """Body force for subdomain A."""
    fx = (-63*x**4*y/3125 + 99*x**4/5000 + 168*x**3*y/15625 - 33*x**3/3125 
          + 216*x**2*y**2/625 - 1557*x**2*y/3125 - 99*x**2/5000 
          - 288*x*y**2/3125 + 1992*x*y/15625 + 33*x/3125 
          - 36*y**2/625 + 54*y/625)
    fy = (-108*x**5/15625 + 72*x**4/15625 - 18*x**3*y**2/625 + 153*x**3*y/1250 
          - 279*x**3/3125 + 36*x**2*y**2/3125 - 153*x**2*y/3125 + 486*x**2/15625 
          + 9*x*y**2/625 - 153*x*y/2500 + 63*x/1250 
          - 12*y**2/3125 + 51*y/3125 - 42/3125)
    return fx, fy

def f_B(x, y):
    """Body force for subdomain B."""
    fx = (1737*x**4*y/30625 - 9357*x**4/245000 - 4632*x**3*y/153125 + 3119*x**3/153125 
          + 396*x**2*y**2/875 - 57969*x**2*y/61250 + 86847*x**2/245000 
          - 528*x*y**2/4375 + 40962*x*y/153125 - 16034*x/153125 
          - 66*y**2/875 + 519*y/3500 - 369/7000)
    by = (2316*x**5/153125 - 1544*x**4/153125 + 1158*x**3*y**2/30625 + 9201*x**3*y/61250 
          - 53561*x**3/245000 - 2316*x**2*y**2/153125 - 9201*x**2*y/153125 + 59737*x**2/612500 
          - 579*x*y**2/30625 - 9201*x*y/122500 + 9477*x/98000 
          + 772*y**2/153125 + 3067*y/153125 - 3159/122500)
    return fx, by

print("Source terms defined.")
print("Ready to run coupling...")
