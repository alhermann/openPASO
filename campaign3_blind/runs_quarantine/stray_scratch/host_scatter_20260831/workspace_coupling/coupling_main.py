#!/usr/bin/env python3
"""
Coupled simulation of thermo-mechanical problem using 4C (subdomain A) and FEniCSx (subdomain B).
Dirichlet-Neumann iteration with A as DIRICHLET side and B as NEUMANN side.
"""

import numpy as np
import os
import sys
import subprocess
import shutil
from pathlib import Path

# Set up paths
WORK_DIR = Path("/home/alexander/workspace_coupling")
WORK_DIR.mkdir(exist_ok=True)
os.chdir(WORK_DIR)

# Problem parameters
INTERFACE_X = 5/8  # 0.625
DOMAIN_A = (0, INTERFACE_X, 0, 1)  # x range, y range
DOMAIN_B = (INTERFACE_X, 1.5, 0, 1)

# Material properties
# Subdomain A
k_A = 1
lambda_A = 600
mu_A = 400
beta_A = 1

# Subdomain B  
k_B = 3
lambda_B = 600
mu_B = 1600
beta_B = 1

# Mesh levels
MESH_LEVELS = [8, 16, 32]  # h = 1/8, 1/16, 1/32

# Probe points for subdomain A
def generate_probe_points_A():
    """Generate 1936 probe points for subdomain A."""
    points = []
    nx, ny = 44, 44
    dx = 0.625 / nx
    dy = 1.0 / ny
    for iy in range(ny):
        for ix in range(nx):
            x = 0 + (ix + 0.5) * dx
            y = 0 + (iy + 0.5) * dy
            points.append((x, y))
    return np.array(points)

# Probe points for subdomain B
def generate_probe_points_B():
    """Generate 1936 probe points for subdomain B."""
    points = []
    nx, ny = 44, 44
    dx = 0.875 / nx
    dy = 1.0 / ny
    for iy in range(ny):
        for ix in range(nx):
            x = 0.625 + (ix + 0.5) * dx
            y = 0 + (iy + 0.5) * dy
            points.append((x, y))
    return np.array(points)

# Interface probe points
def generate_interface_probe_points():
    """Generate 44 interface probe points at x = 5/8."""
    points = []
    n = 44
    dy = 1.0 / (2 * n)  # 1/2/44
    for i in range(n):
        x = 5/8
        y = 1/4 + (i + 0.5) * dy
        points.append((x, y))
    return np.array(points)

PROBE_POINTS_A = generate_probe_points_A()
PROBE_POINTS_B = generate_probe_points_B()
INTERFACE_PROBE_POINTS = generate_interface_probe_points()

print(f"Probe points A: {len(PROBE_POINTS_A)}")
print(f"Probe points B: {len(PROBE_POINTS_B)}")
print(f"Interface probe points: {len(INTERFACE_PROBE_POINTS)}")

# Source terms for subdomain A
def f_T_A(x, y):
    return (-6*x**3*y + 8*x**3/5 - x**2*y/2 - 2*x**2/3 - 6*x*y**3 + 24*x*y**2/5 
            + 67*x*y/10 - 11*x/15 - y**3/6 - 2*y**2/3 + 5*y/6)

def f_x_A(x, y):
    return (-33*x**2*y**3/5 - 312*x**2*y**2/25 + 453*x**2*y/25 - 18*x**2/5 
            + 599*x*y**3/10 + 4754*x*y**2/75 - 17597*x*y/150 + 112*x/5 
            - 84*y**5/25 - 147*y**4/25 + 18277*y**3/300 + 893*y**2/30 
            - 1549*y/20 + 15)

def f_y_A(x, y):
    return (3*x**3*y**2 - 8*x**3*y/5 - x**3/5 + 2693*x**2*y**2/20 + 7106*x**2*y/75 
            - 26333*x**2/300 - 12*x*y**4 - 84*x*y**3/5 + 4241*x*y**2/20 
            + 301*x*y/3 - 2173*x/20 + 56*y**4/15 + 392*y**3/75 
            - 364*y**2/25 + 28*y/5)

# Source terms for subdomain B
def f_T_B(x, y):
    return (-2*x**3*y/3 + 8*x**3/45 - 8*x**2*y/3 + 4*x**2/9 - 2*x*y**3/3 
            + 8*x*y**2/15 + 251*x*y/120 - 41*x/90 - 8*y**3/9 + 4*y**2/9 
            + 829*y/144 - 11/12)

def f_x_B(x, y):
    return (365273*x**2*y**3/5985 + 273076*x**2*y**2/4275 - 3555593*x**2*y/29925 
            + 15192*x**2/665 - 119754589*x*y**3/251370 - 359593607*x*y**2/718200 
            + 4672588891*x*y/5027400 - 13314341*x/74480 + 2532*y**5/175 
            + 633*y**4/25 + 4520447879*y**3/20109600 + 894533323*y**2/2298240 
            - 2001017755*y/3217536 + 28506033/238336)

def f_y_B(x, y):
    return (x**3*y**2/9 - 8*x**3*y/135 - x**3/135 - 1720331*x**2*y**2/1764 
            - 1032893*x**2*y/1512 + 13423129*x**2/21168 + 27852*x*y**4/665 
            + 27852*x*y**3/475 + 9525905951*x*y**2/6703200 + 269425655*x*y/229824 
            - 872177333*x/846720 - 1436809*y**4/13965 - 1436809*y**3/9975 
            + 976524757*y**2/4468800 - 1508594569*y/5362560 + 13360741/112896)

# Store source term values at probe points for later use
SOURCE_A = np.column_stack([f_T_A(*p), f_x_A(*p), f_y_A(*p) for p in PROBE_POINTS_A])
SOURCE_B = np.column_stack([f_T_B(*p), f_x_B(*p), f_y_B(*p) for p in PROBE_POINTS_B])

print("Source terms computed.")
