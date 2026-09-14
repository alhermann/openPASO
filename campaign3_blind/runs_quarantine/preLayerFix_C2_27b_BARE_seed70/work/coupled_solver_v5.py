#!/usr/bin/env python3
"""
Coupled 4C + Kratos solver for the two-subdomain heat conduction problem.

Subdomain A (4C): (0, 0.625) x (0, 1), k = 1
Subdomain B (Kratos): (0.625, 1.5) x (0, 1), k = 200

Coupling: Dirichlet-Neumann iteration
- Subdomain A is DIRICHLET side: receives interface T from B, returns flux
- Subdomain B is NEUMANN side: receives flux from A, returns interface T
"""

import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import meshio

# Configuration
FOURC_BINARY = "/home/alexander/4C/build/4C"
FOURC_LD_LIBRARY_PATH = "/opt/4C-dependencies/lib"
KRATOS_PYTHON = "/usr/bin/python3"

# Problem geometry
L_A = 0.625  # Width of subdomain A
L_B = 0.875  # Width of subdomain B  
H = 1.0      # Height
X_INTERFACE = L_A  # Interface at x = 5/8 = 0.625

# Material properties
K_A = 1.0    # Conductivity in A
K_B = 200.0  # Conductivity in B


def generate_probe_points_A():
    """Generate probe points for subdomain A"""
    points = []
    for i_y in range(44):
        for i_x in range(44):
            x = 0 + (i_x + 0.5) * L_A / 44
            y = 0 + (i_y + 0.5) * H / 44
            points.append((x, y))
    return points

def generate_probe_points_B():
    """Generate probe points for subdomain B"""
    points = []
    for i_y in range(44):
        for i_x in range(44):
            x = L_A + (i_x + 0.5) * L_B / 44
            y = 0 + (i_y + 0.5) * H / 44
            points.append((x, y))
    return points

def generate_interface_probe_points():
    """Generate interface probe points (j = 11 to 32)"""
    points = []
    for j in range(11, 33):
        x = X_INTERFACE
        y = (j + 0.5) / 44
        points.append((x, y))
    return points


class FourcSolver:
    """Solver for subdomain A using 4C"""
    
    def __init__(self, nx, ny, level, work_dir):
        self.nx = nx
        self.ny = ny
        self.level = level
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
    def write_deck(self, interface_temps):
        """Write 4C YAML input file with interface Dirichlet BC
        
        For varying interface temperatures, we create individual point conditions
        for each node on the interface.
        """
        nid = {}
        nodes = []
        cnt = 1
        for j in range(self.ny + 1):
            for i in range(self.nx + 1):
                nid[{{i}}, {{j}}] = cnt
                x = L_A * i / self.nx
                y = H * j / self.ny
                nodes.append((cnt, x, y))
                cnt += 1
        
        # Create DESIGN LINE DIRICH CONDITIONS for outer boundaries (uniform zero)
        yaml_content = f'''TITLE:
  - "Subdomain A: steady conduction, interface Dirichlet from partner"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo"
IO:
  VERBOSITY: "Standard"
IO/RUNTIME VTK OUTPUT:
  OUTPUT_DATA_FORMAT: ascii
THERMAL DYNAMIC:
  DYNAMICTYPE: Statics
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
THERMAL DYNAMIC/RUNTIME VTK OUTPUT:
  OUTPUT_THERMO: true
  TEMPERATURE: true
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Thermal_Solver"
MATERIALS:
  - MAT: 1
    MAT_Fourier:
      CAPA: 1.0
      CONDUCT:
        constant: [{K_A}]
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: 3
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
DLINE-NODE TOPOLOGY:
'''
        # Left boundary (x=0): dline 1
        for j in range(self.ny + 1):
            yaml_content += f'  - "NODE {nid[{{0}}, {{j}}]} DLINE 1"\n'
        # Bottom boundary (y=0): dline 2
        for i in range(self.nx + 1):
            yaml_content += f'  - "NODE {nid[{{i}}, {{0}}]} DLINE 2"\n'
        # Top boundary (y=H): dline 3
        for i in range(self.nx + 1):
            yaml_content += f'  - "NODE {nid[(i, self.ny)]} DLINE 3"\n'
        
        yaml_content += 'NODE COORDS:\n'
        for n, xx, yy in nodes:
            yaml_content += f'  - "NODE {n} COORD {xx:.14g} {yy:.14g} 0.0"\n'
        
        yaml_content += 'THERMO ELEMENTS:\n'
        eid = 1
        for j in range(self.ny):
            for i in range(self.nx):
                a, b, c, d = nid[{{i}}, {{j}}], nid[{{i + 1}}, {{j}}], nid[{{i + 1}}, {{j + 1}}], nid[{{i}}, {{j + 1}}]
                yaml_content += f'  - "{eid} THERMO QUAD4 {a} {b} {c} {d} MAT 1"\n'
                eid += 1
        
        # Add interface conditions as separate DESIGN POINT DIRICH CONDITIONS
        # Each node gets its own condition with its specific temperature
        yaml_content += '\nDESIGN POINT DIRICH CONDITIONS:\n'
        for j, t in enumerate(interface_temps):
            node_id = nid[(self.nx, j)]
            yaml_content += f'  - E: {j+1}\n'
            yaml_content += f'    NUMDOF: 1\n'
            yaml_content += f'    ONOFF: [1]\n'
            yaml_content += f'    VAL: [{t:.14g}]\n'
            yaml_content += f'    FUNCT: [0]\n'
        
        yaml_content += 'DNODE-NODE TOPOLOGY:\n'
        for j in range(len(interface_temps)):
            node_id = nid[(self.nx, j)]
            yaml_content += f'  - "NODE {node_id} DNODE {j+1}"\n'
        
        deck_path = self.work_dir / "slabA.4C.yaml"
        deck_path.write_text(yaml_content)
        
        return deck_path
    
    def run(self, interface_temps, log_file):
        """Run 4C solver"""
        deck_path = self.write_deck(interface_temps)
        
        env = os.environ.copy()
        env["DISPLAY"] = ""
        env["LD_LIBRARY_PATH"] = (FOURC_LD_LIBRARY_PATH + ":" + 
                                  env.get("LD_LIBRARY_PATH", "")).rstrip(":")
        
        cmd = ['mpirun', '-n', '1', FOURC_BINARY, 'slabA.4C.yaml', 'out']
        
        # Run and capture output to log file
        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                   env=env, cwd=self.work_dir, timeout=300)
        
        if result.returncode != 0:
            raise RuntimeError(f"4C exited with rc={result.returncode}")
        
        return self.extract_solution()
    
    def extract_solution(self):
        """Extract solution from 4C VTU output"""
        vtus = sorted(self.work_dir.glob("out-vtk-files/thermo-*-0.vtu"))
        if not vtus:
            raise RuntimeError("no 4C VTU output found")
        
        m = meshio.read(vtus[-1])
        pts = np.asarray(m.points)
        T = np.asarray(m.point_data["temperature"]).ravel()
        
        # Deduplicate nodes
        key = np.round(pts[:, :2], 10)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        Tu = np.zeros(len(uniq))
        for k in range(len(uniq)):
            Tu[k] = T[inv == k].mean()
        
        return uniq, Tu


class KratosSolver:
    """Solver for subdomain B using Kratos Multiphysics"""
    
    def __init__(self, nx, ny, level, work_dir):
        self.nx = nx
        self.ny = ny
        self.level = level
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
    def write_script(self, interface_fluxes_dict):
        """Write Kratos Python script"""
        
        # Interpolate interface fluxes to our grid
        y_if = np.array([H * j / self.ny for j in range(self.ny + 1)])
        
        # Read imported fluxes and interpolate
        if interface_fluxes_dict and "coordinates" in interface_fluxes_dict:
            src_coords = interface_fluxes_dict["coordinates"]
            src_q = interface_fluxes_dict["normal_fluxes"]
            src_y = np.array([p[1] for p in src_coords])
            src_q = np.array(src_q)
            order = np.argsort(src_y)
            q_in = np.interp(y_if, src_y[order], src_q[order])
        else:
            q_in = np.zeros(self.ny + 1)
        
        script = f'''#!/usr/bin/env python3
"""Kratos solver for subdomain B - Neumann side of coupling"""
import json
from pathlib import Path
import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication

# Problem parameters
L_A = {L_A}
L_B = {L_B}
H = {H}
K_B = {K_B}
nx = {self.nx}
ny = {self.ny}

# Incoming flux from partner (interpolated to our interface nodes)
q_in_values = {list(q_in)}

model = KM.Model()
mp = model.CreateModelPart("thermal")
mp.ProcessInfo[KM.DOMAIN_SIZE] = 2

# Setup ConvectionDiffusion settings
settings = KM.ConvectionDiffusionSettings()
settings.SetUnknownVariable(KM.TEMPERATURE)
settings.SetDiffusionVariable(KM.CONDUCTIVITY)
settings.SetVolumeSourceVariable(KM.HEAT_FLUX)
settings.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, settings)

for v in (KM.TEMPERATURE, KM.CONDUCTIVITY, KM.HEAT_FLUX, KM.FACE_HEAT_FLUX):
    mp.AddNodalSolutionStepVariable(v)

mp.SetBufferSize(1)

props = mp.CreateNewProperties(1)

# Create nodes
nid = {{}}
cnt = 1
for j in range(ny + 1):
    for i in range(nx + 1):
        x = L_A + L_B * i / nx
        y = H * j / ny
        mp.CreateNewNode(cnt, x, y, 0.0)
        nid[{{i}}, {{j}}] = cnt
        cnt += 1

# Create elements (triangles)
eid = 1
for j in range(ny):
    for i in range(nx):
        a, b, c, d = nid[{{i}}, {{j}}], nid[{{i + 1}}, {{j}}], nid[{{i + 1}}, {{j + 1}}], nid[{{i}}, {{j + 1}}]
        mp.CreateNewElement("LaplacianElement2D3N", eid, [a, b, d], props); eid += 1
        mp.CreateNewElement("LaplacianElement2D3N", eid, [b, c, d], props); eid += 1

# Create line conditions for Neumann BC on left edge
cid = 1
for j in range(ny):
    mp.CreateNewCondition("FluxCondition2D2N", cid,
                          [nid[{{0}}, {{j}}], nid[{{0}}, {{j + 1}}]], props)
    cid += 1

# Set material properties and sources
for node in mp.Nodes:
    node.SetSolutionStepValue(KM.CONDUCTIVITY, K_B)
    node.SetSolutionStepValue(KM.HEAT_FLUX, 0.0)
    node.SetSolutionStepValue(KM.FACE_HEAT_FLUX, 0.0)

# Apply Neumann BC on left edge (interface) - incoming flux from A
for j in range(ny + 1):
    mp.Nodes[nid[{{0}}, {{j}}]].SetSolutionStepValue(KM.FACE_HEAT_FLUX, float(q_in_values[j]))

# Apply Dirichlet BC on outer boundaries (u=0)
# Right edge (x = L_A + L_B)
for j in range(ny + 1):
    n = mp.Nodes[nid[{{nx}}, {{j}}]]
    n.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
    n.Fix(KM.TEMPERATURE)

# Bottom edge (y = 0) - skip corners
for i in range(1, nx):
    n = mp.Nodes[nid[{{i}}, {{0}}]]
    n.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
    n.Fix(KM.TEMPERATURE)

# Top edge (y = H) - skip corners
for i in range(1, nx):
    n = mp.Nodes[nid[{{i}}, {{ny}}]]
    n.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
    n.Fix(KM.TEMPERATURE)

# Add DOFs
KM.VariableUtils().AddDof(KM.TEMPERATURE, mp)

# Solve
scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
linear_solver = KM.SkylineLUFactorizationSolver()
builder = KM.ResidualBasedBlockBuilderAndSolver(linear_solver)
strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder, False, False, False, False)
strategy.Initialize()
strategy.Solve()

# Extract full solution for probe point evaluation
all_nodes = []
all_temps = []
for node in mp.Nodes:
    all_nodes.append([node.X, node.Y])
    all_temps.append(node.GetSolutionStepValue(KM.TEMPERATURE))

all_nodes = np.array(all_nodes)
all_temps = np.array(all_temps)

# Save full solution to file
np.savez("solution_full.npz", nodes=all_nodes, temps=all_temps)

# Extract interface data
dx = L_B / nx
T_if = np.array([mp.Nodes[nid[{{0}}, {{j}}]].GetSolutionStepValue(KM.TEMPERATURE)
                 for j in range(ny + 1)])
T_near = np.array([mp.Nodes[nid[{{1}}, {{j}}]].GetSolutionStepValue(KM.TEMPERATURE)
                   for j in range(ny + 1)])

# Outward normal flux (pointing out of B, i.e., -x direction)
# q_out = -k * grad(u) . n = -k * du/dx * (-1) = k * du/dx
q_out = K_B * (T_near - T_if) / dx

# Export results
export = {{
    "field_name": "temperature",
    "n_points": len(T_if),
    "coordinates": [[L_A, float(H * j / ny)] for j in range(ny + 1)],
    "values": [float(v) for v in T_if],
    "normal_fluxes": [float(v) for v in q_out],
}}

Path("exports.json").write_text(json.dumps(export, indent=2))

print(f"Kratos slab B: T_if(mean)={{T_if.mean():.6f}}, q_in(mean)={{np.mean(q_in_values):.6f}}")
'''
        
        script_path = self.work_dir / "MainKratos.py"
        script_path.write_text(script)
        
        return script_path
    
    def run(self, interface_fluxes_dict, log_file):
        """Run Kratos solver"""
        script_path = self.write_script(interface_fluxes_dict)
        
        cmd = [KRATOS_PYTHON, str(script_path)]
        
        # Run and capture output to log file
        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                   cwd=self.work_dir, timeout=300)
        
        if result.returncode != 0:
            raise RuntimeError(f"Kratos exited with rc={result.returncode}")
        
        # Read exports
        exports_path = self.work_dir / "exports.json"
        exports = json.loads(exports_path.read_text())
        
        # Also read full solution
        sol_path = self.work_dir / "solution_full.npz"
        if sol_path.exists():
            sol_data = np.load(sol_path)
            nodes = sol_data['nodes']
            temps = sol_data['temps']
        else:
            nodes = None
            temps = None
        
        return exports, nodes, temps


def interpolate_to_probes(nodes, u, probe_points):
    """Interpolate FEM solution to probe points using nearest neighbor"""
    results = []
    for px, py in probe_points:
        dists = np.sqrt((nodes[:, 0] - px)**2 + (nodes[:, 1] - py)**2)
        min_dist_idx = np.argmin(dists)
        results.append(float(u[min_dist_idx]))
    return results


def compute_interface_residual(temp_A, temp_B):
    """Compute relative interface residual"""
    diff = np.abs(np.array(temp_A) - np.array(temp_B))
    ref = np.max(np.abs(np.array(temp_A))) + 1e-15
    return np.max(diff) / ref


def run_coupling_level(level, h, base_dir):
    """Run coupling for one mesh level"""
    
    # Mesh parameters - use uniform h for both domains
    nx_A = max(1, int(L_A / h))
    ny_A = max(1, int(H / h))
    nx_B = max(1, int(L_B / h))
    ny_B = max(1, int(H / h))
    
    print(f"Level {level}: h={h}, A: {nx_A}x{ny_A}, B: {nx_B}x{ny_B}")
    
    work_dir_A = base_dir / f"level{level}_A"
    work_dir_B = base_dir / f"level{level}_B"
    
    solver_A = FourcSolver(nx_A, ny_A, level, work_dir_A)
    solver_B = KratosSolver(nx_B, ny_B, level, work_dir_B)
    
    # Initial guess for interface temperature (zeros since outer BC is zero)
    interface_temps = np.zeros(ny_A + 1)
    
    max_iter = 100
    tol = 1e-6
    
    residuals = []
    
    for iteration in range(max_iter):
        # Run solver A (Dirichlet side) - receives T, returns flux
        log_A = work_dir_A / f"run_level{level}_A.log"
        try:
            nodes_A, u_A = solver_A.run(interface_temps, log_A)
        except Exception as e:
            print(f"Error running 4C: {e}")
            raise
        
        # Extract interface data from A
        dx_A = L_A / nx_A
        on_if_A = np.abs(nodes_A[:, 0] - L_A) < 1e-9
        near_A = np.abs(nodes_A[:, 0] - (L_A - dx_A)) < 1e-9
        
        if not on_if_A.any():
            raise RuntimeError("Interface nodes not found in A")
        
        order_A = np.argsort(nodes_A[on_if_A, 1])
        if_coords_A = nodes_A[on_if_A][order_A]
        if_u_A = u_A[on_if_A][order_A]
        
        near_order_A = np.argsort(nodes_A[near_A, 1])
        near_u_A = u_A[near_A][near_order_A]
        
        # Flux leaving A (outward normal = +x)
        q_A = -K_A * (if_u_A - near_u_A) / dx_A
        
        # Prepare imports for B
        interface_fluxes_dict = {"normal_fluxes": list(q_A), 
                                 "coordinates": [[float(X_INTERFACE), float(y)] for y in if_coords_A[:, 1]]}
        
        # Run solver B (Neumann side) - receives flux, returns T
        log_B = work_dir_B / f"run_level{level}_B.log"
        try:
            exports_B, nodes_B, u_B = solver_B.run(interface_fluxes_dict, log_B)
        except Exception as e:
            print(f"Error running Kratos: {e}")
            raise
        
        interface_temps_new = np.array(exports_B["values"])
        
        # Compute residual
        residual = compute_interface_residual(if_u_A, interface_temps_new)
        residuals.append(residual)
        
        print(f"  Iteration {iteration+1}: residual = {residual:.6e}")
        
        if residual < tol:
            break
        
        # Update interface temperature
        interface_temps = interface_temps_new
    
    # Final solutions
    final_nodes_A, final_u_A = nodes_A, u_A
    final_exports_B = exports_B
    final_nodes_B, final_u_B = nodes_B, u_B
    
    # Add NDOF to log files
    ndof_A = len(final_u_A)
    with open(log_A, 'a') as f:
        f.write(f"\nNDOF = {ndof_A}\n")
    
    ndof_B = len(final_u_B) if final_u_B is not None else (nx_B + 1) * (ny_B + 1)
    with open(log_B, 'a') as f:
        f.write(f"\nNDOF = {ndof_B}\n")
    
    # Generate probe points
    probe_A = generate_probe_points_A()
    probe_B = generate_probe_points_B()
    interface_probes = generate_interface_probe_points()
    
    # Interpolate to probe points
    u_at_probe_A = interpolate_to_probes(final_nodes_A, final_u_A, probe_A)
    u_at_probe_B = interpolate_to_probes(final_nodes_B, final_u_B, probe_B)
    
    # Write solution CSV files
    sol_file_A = base_dir / f"solution_level{level}_A.csv"
    with open(sol_file_A, 'w') as f:
        f.write("x, y, u\n")
        for (x, y), u in zip(probe_A, u_at_probe_A):
            f.write(f"{x}, {y}, {u}\n")
    
    sol_file_B = base_dir / f"solution_level{level}_B.csv"
    with open(sol_file_B, 'w') as f:
        f.write("x, y, u\n")
        for (x, y), u in zip(probe_B, u_at_probe_B):
            f.write(f"{x}, {y}, {u}\n")
    
    # Write interface CSV files
    # For A: interpolate to interface probes
    u_at_interface_A = []
    qn_at_interface_A = []
    for px, py in interface_probes:
        idx = np.argmin(np.abs(if_coords_A[:, 1] - py))
        u_at_interface_A.append(float(if_u_A[idx]))
        qn_at_interface_A.append(float(q_A[idx]))
    
    # For B: interpolate to interface probes
    u_at_interface_B = []
    qn_at_interface_B = []
    for px, py in interface_probes:
        idx = np.argmin(np.abs(np.array([p[1] for p in final_exports_B["coordinates"]]) - py))
        u_at_interface_B.append(float(final_exports_B["values"][idx]))
        qn_at_interface_B.append(float(final_exports_B["normal_fluxes"][idx]))
    
    interface_file_A = base_dir / f"interface_level{level}_A.csv"
    with open(interface_file_A, 'w') as f:
        f.write("x, y, u, qn\n")
        for (x, y), u, qn in zip(interface_probes, u_at_interface_A, qn_at_interface_A):
            f.write(f"{x}, {y}, {u}, {qn}\n")
    
    interface_file_B = base_dir / f"interface_level{level}_B.csv"
    with open(interface_file_B, 'w') as f:
        f.write("x, y, u, qn\n")
        for (x, y), u, qn in zip(interface_probes, u_at_interface_B, qn_at_interface_B):
            f.write(f"{x}, {y}, {u}, {qn}\n")
    
    # Write residual history
    residual_file = base_dir / f"residual_level{level}.csv"
    with open(residual_file, 'w') as f:
        f.write("iteration, interface_residual\n")
        for i, r in enumerate(residuals):
            f.write(f"{i+1}, {r}\n")
    
    return {
        "level": level,
        "final_residual": residuals[-1],
        "iterations": len(residuals),
        "nodes_A": final_nodes_A,
        "u_A": final_u_A,
        "nodes_B": final_nodes_B,
        "u_B": final_u_B,
        "probe_A": probe_A,
        "probe_B": probe_B,
        "interface_probes": interface_probes,
    }


def main():
    base_dir = Path("coupling_results")
    base_dir.mkdir(exist_ok=True)
    
    # Mesh levels: h = 1/8, 1/16, 1/32
    levels = [1, 2, 3]
    h_values = [1/8, 1/16, 1/32]
    
    all_results = []
    csv_files = []
    
    for level, h in zip(levels, h_values):
        print(f"\n{'='*60}")
        print(f"Running level {level} with h = {h}")
        print('='*60)
        
        result = run_coupling_level(level, h, base_dir)
        all_results.append(result)
        
        # Track CSV files
        csv_files.extend([
            f"solution_level{level}_A.csv",
            f"solution_level{level}_B.csv",
            f"interface_level{level}_A.csv",
            f"interface_level{level}_B.csv",
            f"residual_level{level}.csv",
        ])
    
    # Compute mesh independence
    if len(all_results) >= 2:
        # Compare finest two levels
        r1 = all_results[-2]
        r2 = all_results[-1]
        
        # Compare at probe points - need to interpolate to common points
        # Use the finer level's probe points
        u1_at_probe = interpolate_to_probes(r1["nodes_A"], r1["u_A"], r2["probe_A"])
        u2_at_probe = interpolate_to_probes(r2["nodes_A"], r2["u_A"], r2["probe_A"])
        
        rel_changes = np.abs(np.array(u2_at_probe) - np.array(u1_at_probe)) / (np.abs(np.array(u1_at_probe)) + 1e-15)
        max_rel_change = float(np.max(rel_changes))
        
        # Check convergence: max change should be small
        converged = max_rel_change < 0.01  # 1% threshold
    else:
        max_rel_change = 0.0
        converged = False
    
    # Write RESULT.txt
    final_result = all_results[-1]
    
    with open(base_dir / "RESULT.txt", 'w') as f:
        f.write(f"LEVELS = {len(levels)}\n")
        f.write(f"FILES = {', '.join(csv_files)}\n")
        f.write(f"INTERFACE_RESIDUAL = {final_result['final_residual']}\n")
        f.write(f"COUPLING_ITERATIONS = {final_result['iterations']}\n")
        f.write(f"MESH_INDEPENDENCE = {'CONVERGED' if converged else 'NOT_CONVERGED'}\n")
        f.write(f"MAX_REL_CHANGE = {max_rel_change}\n")
    
    print("\nDone! Results written to coupling_results/")


if __name__ == "__main__":
    main()
