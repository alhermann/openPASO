#!/usr/bin/env python3
"""
Complete coupled simulation driver using OASiS couple tool.
This script sets up and runs the coupled heat conduction problem.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np

WORK_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = WORK_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Problem constants
X0_A, X1_A = 0.0, 0.625  # Subdomain A
X0_B, X1_B = 0.625, 1.5  # Subdomain B  
Y0, Y1 = 0.0, 1.0
IFACE_X = 0.625
K_A = 1.0
K_B = 200.0

def F_SRC_A(x, y):
    """Source term in subdomain A."""
    return (-12*x**3*y/5 + 14*x**3/5 - 8979*x**2*y/2000 - 5147*x**2/12000 
            - 12*x*y**3/5 + 42*x*y**2/5 - 1779*x*y/800 - 1007*x/1200 
            - 2993*y**3/2000 - 5147*y**2/12000 + 4621*y/2400)

def F_SRC_B(x, y):
    """Source term in subdomain B."""
    return (-3*x**3*y/50000 + 7*x**3/100000 - 8967*x**2*y/200000 + 28769*x**2/1200000 
            - 3*x*y**3/50000 + 21*x*y**2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 
            - 2989*y**3/200000 + 28769*y**2/1200000 + 26803463*y/3840000 - 1468421/512000)

def get_probe_points_A():
    """Generate 1936 probe points for subdomain A."""
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = 0.0 + (i_x + 0.5) * 0.625 / 44
            y = 0.0 + (i_y + 0.5) * 1.0 / 44
            points.append((x, y))
    return np.array(points)

def get_probe_points_B():
    """Generate 1936 probe points for subdomain B."""
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = 0.625 + (i_x + 0.5) * 0.875 / 44
            y = 0.0 + (i_y + 0.5) * 1.0 / 44
            points.append((x, y))
    return np.array(points)

def get_interface_probe_points():
    """Generate 22 interface probe points (j = 11 to 32)."""
    points = []
    for j in range(11, 33):
        x = 5/8
        y = (j + 0.5) / 44
        points.append((x, y))
    return np.array(points)

def write_4c_participant_script(level, work_dir, side="A"):
    """Write 4C participant script for subdomain A."""
    h_base = 1.0 / (8 * level)
    nx = int(round(0.625 / h_base))
    ny = int(round(1.0 / h_base))
    
    script = f'''#!/usr/bin/env python3
"""4C participant for subdomain A (Dirichlet side)."""
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

PARTNER = "B"
X0, X1 = {X0_A}, {X1_A}
Y0, Y1 = {Y0}, {Y1}
IFACE_X = {IFACE_X}
K = {K_A}
T_INIT = 0.0
Q_INIT = 0.0

FOURC_BIN = "/home/alexander/4C/build/4C"
FOURC_LD = "/opt/4C-dependencies/lib"

def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return {{}}
    try:
        return json.loads(p.read_text())
    except:
        return {{}}

def sample_from_partner(imp_data, key, ys, fallback):
    if PARTNER not in imp_data or "coordinates" not in imp_data[PARTNER]:
        return np.full(len(ys), float(fallback))
    partner = imp_data[PARTNER]
    partner_y = np.array([c[1] for c in partner["coordinates"]], float)
    partner_v = np.asarray(partner.get(key, []), float).ravel()
    if len(partner_v) != len(partner_y):
        return np.full(len(ys), float(fallback))
    idx = np.argsort(partner_y)
    return np.interp(ys, partner_y[idx], partner_v[idx])

def fit_polynomial(y_coords, values, degree=3):
    coeffs = np.polyfit(y_coords, values, min(degree, len(y_coords)-1))
    expr_parts = []
    for i, c in enumerate(reversed(coeffs)):
        power = len(coeffs) - 1 - i
        if abs(c) < 1e-14:
            continue
        if power == 0:
            expr_parts.append(f"{{c:.15e}}")
        elif power == 1:
            expr_parts.append(f"{{c:.15e}}*y")
        else:
            expr_parts.append(f"{{c:.15e}}*y^{{power}}")
    return " + ".join(expr_parts) if expr_parts else "0.0"

def build_source_expr(max_deg=4):
    def F_SRC(x, y):
        return (-12*x**3*y/5 + 14*x**3/5 - 8979*x**2*y/2000 - 5147*x**2/12000 
                - 12*x*y**3/5 + 42*x*y**2/5 - 1779*x*y/800 - 1007*x/1200 
                - 2993*y**3/2000 - 5147*y**2/12000 + 4621*y/2400)
    
    nx_sample, ny_sample = 8, 8
    xs = np.linspace(X0 + 0.01, X1 - 0.01, nx_sample)
    ys = np.linspace(Y0 + 0.01, Y1 - 0.01, ny_sample)
    XX, YY = np.meshgrid(xs, ys)
    vals = F_SRC(XX, YY).flatten()
    xx = XX.flatten()
    yy = YY.flatten()
    
    best_err = float('inf')
    best_coeffs = None
    best_basis = None
    
    for deg in range(1, max_deg + 1):
        basis = []
        for i in range(deg + 1):
            for j in range(deg + 1 - i):
                basis.append((i, j))
        
        n = len(vals)
        m = len(basis)
        A = np.zeros((n, m))
        for k, (i, j) in enumerate(basis):
            A[:, k] = xx**i * yy**j
        
        coeffs, res, rank, s = np.linalg.lstsq(A, vals, rcond=None)
        err = np.sqrt(np.mean((A @ coeffs - vals)**2))
        
        if err < best_err:
            best_err = err
            best_coeffs = coeffs
            best_basis = basis
    
    parts = []
    for (i, j), c in zip(best_basis, best_coeffs):
        if abs(c) > 1e-14:
            if i == 0 and j == 0:
                parts.append(f"{{c:.15e}}")
            elif i == 1 and j == 0:
                parts.append(f"{{c:.15e}}*x")
            elif i == 0 and j == 1:
                parts.append(f"{{c:.15e}}*y")
            elif i == 1 and j == 1:
                parts.append(f"{{c:.15e}}*x*y")
            elif i == 2 and j == 0:
                parts.append(f"{{c:.15e}}*x^2")
            elif i == 0 and j == 2:
                parts.append(f"{{c:.15e}}*y^2")
            elif i == 3 and j == 0:
                parts.append(f"{{c:.15e}}*x^3")
            elif i == 0 and j == 3:
                parts.append(f"{{c:.15e}}*y^3")
            elif i == 2 and j == 1:
                parts.append(f"{{c:.15e}}*x^2*y")
            elif i == 1 and j == 2:
                parts.append(f"{{c:.15e}}*x*y^2")
            else:
                term = f"{{c:.15e}}"
                if i > 0:
                    term += f"*x^{{i}}" if i > 1 else "*x"
                if j > 0:
                    term += f"*y^{{j}}" if j > 1 else "*y"
                parts.append(term)
    
    return " + ".join(parts) if parts else "0.0", best_err

def generate_deck(nx, ny, iface_temps, work_dir):
    nodes = []
    node_id = 1
    node_map = {{}}
    
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = X0 + i * (X1 - X0) / nx
            y = Y0 + j * (Y1 - Y0) / ny
            nodes.append(f'NODE {{node_id}} COORD {{x:.15e}} {{y:.15e}} 0.0')
            node_map[(i, j)] = node_id
            node_id += 1
    
    n_nodes = node_id - 1
    
    elements = []
    for j in range(ny):
        for i in range(nx):
            n1 = node_map[(i, j)]
            n2 = node_map[(i+1, j)]
            n3 = node_map[(i+1, j+1)]
            n4 = node_map[(i, j+1)]
            elements.append(f'{{len(elements)+1}} TRANSP QUAD4 {{n1}} {{n2}} {{n3}} {{n4}} MAT 1 TYPE Std')
    
    dline_nodes = []
    for j in range(ny + 1):
        nid = node_map[(0, j)]
        dline_nodes.append(f'NODE {{nid}} DLINE 1')
    for j in range(ny + 1):
        nid = node_map[(nx, j)]
        dline_nodes.append(f'NODE {{nid}} DLINE 2')
    for i in range(nx + 1):
        nid = node_map[(i, 0)]
        dline_nodes.append(f'NODE {{nid}} DLINE 3')
    for i in range(nx + 1):
        nid = node_map[(i, ny)]
        dline_nodes.append(f'NODE {{nid}} DLINE 4')
    
    iface_y = np.linspace(Y0, Y1, ny + 1)
    iface_poly = fit_polynomial(iface_y, iface_temps, degree=min(6, ny))
    source_expr, src_err = build_source_expr(4)
    
    yaml_lines = [
        'TITLE:',
        '  - "4C Subdomain A"',
        'PROBLEM SIZE:',
        '  DIM: 2',
        'PROBLEM TYPE:',
        '  PROBLEMTYPE: "Scalar_Transport"',
        'SCALAR TRANSPORT DYNAMIC:',
        '  TIMEINTEGR: "Stationary"',
        '  SOLVERTYPE: "linear_full"',
        '  VELOCITYFIELD: "zero"',
        '  TIMESTEP: 1.0',
        '  NUMSTEP: 1',
        '  MAXTIME: 1.0',
        '  LINEAR_SOLVER: 1',
        'SOLVER 1:',
        '  SOLVER: "UMFPACK"',
        '  NAME: "direct"',
        'MATERIALS:',
        '  - MAT: 1',
        f'    MAT_scatra:',
        f'      DIFFUSIVITY: {{K:.15e}}',
        '',
        'FUNCT1:',
        f'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{{iface_poly}}"',
        '',
        'FUNCT2:',
        f'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{{source_expr}}"',
        '',
        'DESIGN LINE DIRICH CONDITIONS:',
        '  - E: 1',
        '    NUMDOF: 1',
        '    ONOFF: [1]',
        '    VAL: [0.0]',
        '    FUNCT: [0]',
        '  - E: 2',
        '    NUMDOF: 1',
        '    ONOFF: [1]',
        '    VAL: [1.0]',
        '    FUNCT: [1]',
        '',
        'DESIGN LINE NEUMANN CONDITIONS:',
        '  - E: 3',
        '    NUMDOF: 1',
        '    ONOFF: [1]',
        '    VAL: [0.0]',
        '    FUNCT: [0]',
        '  - E: 4',
        '    NUMDOF: 1',
        '    ONOFF: [1]',
        '    VAL: [0.0]',
        '    FUNCT: [0]',
        '',
        'DESIGN VOL TRANSPORT NEUMANN CONDITIONS:',
        '  - V: 1',
        '    NUMDOF: 1',
        '    ONOFF: [1]',
        '    VAL: [1.0]',
        '    FUNCT: [2]',
        '',
        'DLINE-NODE TOPOLOGY:',
    ]
    
    for dn in dline_nodes:
        yaml_lines.append(f'  - "{{dn}}"')
    
    yaml_lines.extend(['', 'NODE COORDS:'])
    for n in nodes:
        yaml_lines.append(f'  - "{{n}}"')
    
    yaml_lines.extend(['', 'TRANSPORT ELEMENTS:'])
    for e in elements:
        yaml_lines.append(f'  - "{{e}}"')
    
    yaml_lines.extend(['', 'TRANSPORT DOMAIN:',
                       '  - V: 1',
                       '    ALL_TRANSP_ELEMENTS: true'])
    
    deck_file = work_dir / "subdomain_A.4C.yaml"
    deck_file.write_text('\\n'.join(yaml_lines))
    return deck_file, n_nodes

def main():
    work_dir = Path.cwd()
    level = {level}
    
    h_base = 1.0 / (8 * level)
    nx = int(round(0.625 / h_base))
    ny = int(round(1.0 / h_base))
    
    print(f"4C Subdomain A: level={{level}}, nx={{nx}}, ny={{ny}}, h={{h_base:.6f}}", flush=True)
    
    imp_data = read_imports()
    iface_y = np.linspace(Y0, Y1, ny + 1)
    iface_temps = sample_from_partner(imp_data, "values", iface_y, T_INIT)
    
    deck_file, n_nodes = generate_deck(nx, ny, iface_temps, work_dir)
    
    output_prefix = work_dir / "run_A"
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = FOURC_LD
    env['OMP_NUM_THREADS'] = '1'
    
    cmd = [FOURC_BIN, str(deck_file), str(output_prefix)]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, flush=True)
    
    if result.returncode != 0:
        print(f"ERROR: 4C failed with return code {{result.returncode}}", flush=True)
        sys.exit(result.returncode)
    
    vtu_files = list(work_dir.glob("run_A-vtk-files/scatra-*.vtu"))
    if not vtu_files:
        vtu_files = list(work_dir.glob("**/scatra-*.vtu"))
    
    if not vtu_files:
        print("ERROR: No VTU files found!", flush=True)
        sys.exit(1)
    
    def parse_step(f):
        stem = f.stem
        parts = stem.split('-')
        if len(parts) >= 3:
            try:
                return int(parts[1])
            except ValueError:
                return 0
        return 0
    
    vtu_files.sort(key=parse_step, reverse=True)
    vtu_file = vtu_files[0]
    print(f"Reading results from: {{vtu_file}}", flush=True)
    
    import meshio
    mesh = meshio.read(str(vtu_file))
    
    pts = mesh.points
    phi = mesh.point_data.get('phi_1') or mesh.point_data.get('phi')
    
    if phi is None:
        print("ERROR: No phi field found in VTU!", flush=True)
        sys.exit(1)
    
    mask = np.abs(pts[:, 0] - IFACE_X) < 1e-6
    if not mask.any():
        print(f"ERROR: No nodes found at interface x={{IFACE_X}}", flush=True)
        sys.exit(1)
    
    iface_pts = pts[mask]
    iface_phi = phi[mask]
    
    uy, inv = np.unique(np.round(iface_pts[:, 1], 10), return_inverse=True)
    T = np.zeros(len(uy))
    counts = np.zeros(len(uy), dtype=int)
    
    for k, idx in enumerate(inv):
        T[idx] += iface_phi[k]
        counts[idx] += 1
    
    T /= counts
    
    dx = (X1 - X0) / nx
    q_flux = np.zeros(len(uy))
    
    for j, y in enumerate(uy):
        interior_x = X1 - dx
        interior_mask = np.abs(pts[:, 0] - interior_x) < 1e-6
        if interior_mask.any():
            interior_y_vals = pts[interior_mask, 1]
            closest_idx = np.argmin(np.abs(interior_y_vals - y))
            interior_node_idx = np.where(interior_mask)[0][closest_idx]
            T_interior = phi[interior_node_idx]
            dT_dx = (T[j] - T_interior) / dx
            q_flux[j] = -K * dT_dx
    
    export_data = {{
        "field_name": "temperature",
        "n_points": len(uy),
        "coordinates": [[float(IFACE_X), float(y)] for y in uy],
        "values": [float(t) for t in T],
        "normal_fluxes": [float(q) for q in q_flux]
    }}
    
    exports_file = work_dir / "exports.json"
    exports_file.write_text(json.dumps(export_data, indent=2))
    
    print(f"4C Subdomain A exported {{len(uy)}} interface points", flush=True)
    print(f"  T range: [{{T.min():.6f}}, {{T.max():.6f}}]", flush=True)
    print(f"  q range: [{{q_flux.min():.6f}}, {{q_flux.max():.6f}}]", flush=True)
    print(f"NDOF = {{n_nodes}}", flush=True)

if __name__ == "__main__":
    main()
'''
    
    script_file = work_dir / "participant_4c_A.py"
    script_file.write_text(script)
    return script_file

print("Starting coupled simulation setup...")
print(f"Work directory: {WORK_DIR}")
print(f"Results directory: {RESULTS_DIR}")

# This is a placeholder - the actual implementation would use OASiS couple tool
# For now, we'll write a summary indicating what needs to be done

with open(RESULTS_DIR / "RESULT.txt", "w") as f:
    f.write("LEVELS = 0\n")
    f.write("FILES = \n")
    f.write("INTERFACE_RESIDUAL = COULD_NOT_COMPLETE\n")
    f.write("COUPLING_ITERATIONS = 0\n")
    f.write("MESH_INDEPENDENCE = NOT_CONVERGED\n")
    f.write("MAX_REL_CHANGE = 0.0\n")

print("\nNote: The full coupled simulation requires:")
print("1. Working 4C binary (currently failing with MPI errors)")
print("2. Proper Kratos participant script")
print("3. OASiS couple tool integration")
print("\nWriting COULD_NOT_COMPLETE status to RESULT.txt")
