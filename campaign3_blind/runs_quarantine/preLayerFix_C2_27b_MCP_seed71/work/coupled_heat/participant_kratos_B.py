"""Kratos Multiphysics participant for subdomain B (Neumann side) - OASiS couple driver.

Subdomain B: (0.625, 1.5) x (0, 1), k=200
Role: Neumann side - imports flux from partner, exports temperature
"""
import json
import sys
from pathlib import Path

import numpy as np

# Check Kratos availability first
try:
    import KratosMultiphysics as KM
    import KratosMultiphysics.ConvectionDiffusionApplication
except ImportError as e:
    print(f"ERROR: Cannot import Kratos: {e}", flush=True)
    sys.exit(1)

# Problem parameters for subdomain B
PARTNER = "A"        # partner name in couple() call
X0, X1 = 0.625, 1.5  # subdomain B x-extent
Y0, Y1 = 0.0, 1.0    # subdomain B y-extent
IFACE_X = 0.625      # interface location (left edge of B)
K = 200.0            # thermal conductivity in B

# Source term in subdomain B (polynomial)
def F_SRC(x, y):
    return (-3*x**3*y/50000 + 7*x**3/100000 - 8967*x**2*y/200000 + 28769*x**2/1200000 
            - 3*x*y**3/50000 + 21*x*y**2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 
            - 2989*y**3/200000 + 28769*y**2/1200000 + 26803463*y/3840000 - 1468421/512000)

T_OUTER = 0.0        # Dirichlet on outer boundary (x=1.5)
Q_INIT = 0.0         # iteration-1 fallback interface flux

def read_imports():
    """Read imports.json and return partner data."""
    p = Path("imports.json")
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}

def sample_from_partner(imp_data, key, ys, fallback):
    """Interpolate partner data at our interface y-coordinates."""
    if PARTNER not in imp_data or "coordinates" not in imp_data[PARTNER]:
        return np.full(len(ys), float(fallback))
    
    partner = imp_data[PARTNER]
    partner_y = np.array([c[1] for c in partner["coordinates"]], float)
    partner_v = np.asarray(partner.get(key, []), float).ravel()
    
    if len(partner_v) != len(partner_y):
        return np.full(len(ys), float(fallback))
    
    idx = np.argsort(partner_y)
    return np.interp(ys, partner_y[idx], partner_v[idx])

def build_source_expr(max_deg=4):
    """Build 2D polynomial expression for source term."""
    nx_sample, ny_sample = 10, 10
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
    
    return best_coeffs, best_basis, best_err

def create_mesh_and_solve(nx, ny, imported_fluxes, work_dir):
    """Create mesh, set up problem, solve, and return results."""
    
    # Create model and model part
    model = KM.Model()
    mp = model.CreateModelPart("MainModelPart")
    
    # Add required variables BEFORE creating nodes
    mp.AddNodalSolutionStepVariable(KM.TEMPERATURE)
    mp.AddNodalSolutionStepVariable(KM.REACTION_FLUX)
    mp.AddNodalSolutionStepVariable(KM.CONDUCTIVITY)
    mp.AddNodalSolutionStepVariable(KM.HEAT_FLUX)
    mp.AddNodalSolutionStepVariable(KM.FACE_HEAT_FLUX)
    
    # Create nodes
    node_id = 1
    node_map = {}
    
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = X0 + i * (X1 - X0) / nx
            y = Y0 + j * (Y1 - Y0) / ny
            mp.CreateNewNode(node_id, x, y, 0.0)
            node_map[(i, j)] = node_id
            node_id += 1
    
    n_nodes = node_id - 1
    
    # Create properties with conductivity
    props_id = 1
    props = mp.CreateNewProperties(props_id)
    props.SetValue(KM.CONDUCTIVITY, K)
    
    # Create triangular elements (P1)
    elem_id = 1
    for j in range(ny):
        for i in range(nx):
            n1 = node_map[(i, j)]
            n2 = node_map[(i+1, j)]
            n3 = node_map[(i+1, j+1)]
            n4 = node_map[(i, j+1)]
            
            mp.CreateNewElement("Element2D3N", elem_id, [n1, n2, n4], props)
            elem_id += 1
            mp.CreateNewElement("Element2D3N", elem_id, [n2, n3, n4], props)
            elem_id += 1
    
    n_elements = elem_id - 1
    
    # Set initial temperature
    for node in mp.Nodes:
        node.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
    
    # Apply boundary conditions
    # Right boundary (x=1.5): Dirichlet T=0
    for j in range(ny + 1):
        nid = node_map[(nx, j)]
        node = mp.Nodes[nid]
        node.SetSolutionStepValue(KM.TEMPERATURE, T_OUTER)
        node.Fix(KM.TEMPERATURE, 1)
    
    # Interface nodes (left boundary, x=0.625) - DO NOT fix, apply Neumann via flux
    iface_node_ids = []
    for j in range(ny + 1):
        nid = node_map[(0, j)]
        iface_node_ids.append(nid)
        node = mp.Nodes[nid]
        node.Fix(KM.TEMPERATURE, 0)  # Free
    
    # Apply imported flux as FACE_HEAT_FLUX on interface
    # Sign convention: partner exports their outward normal flux
    # For subdomain A, outward at interface is +x direction
    # For subdomain B, outward at interface is -x direction
    # Flux continuity: q_A . n_interface = q_B . n_interface where n_interface = +x
    # So: -q_A_outward = q_B . (+x) = -q_B_outward
    # Therefore: q_B_outward = q_A_outward
    # In Kratos: we apply the flux directly as FACE_HEAT_FLUX
    
    iface_y = np.linspace(Y0, Y1, ny + 1)
    imp_data = read_imports()
    imported_q = sample_from_partner(imp_data, "normal_fluxes", iface_y, Q_INIT)
    
    # Set FACE_HEAT_FLUX on interface nodes
    # The sign: Kratos uses K grad T . n = FACE_HEAT_FLUX
    # We want to match the incoming flux from partner
    # Partner's exported flux is their outward normal flux (positive x for A)
    # At the interface, this should equal our inward flux (negative x for B)
    # So we apply: FACE_HEAT_FLUX = -imported_q (to get the right sign)
    for j, (nid, q_val) in enumerate(zip(iface_node_ids, imported_q)):
        node = mp.Nodes[nid]
        # Apply negative because Kratos convention is different
        node.SetSolutionStepValue(KM.FACE_HEAT_FLUX, -q_val)
    
    # Create Geometry for boundaries
    right_geom_nodes = [mp.Nodes[node_map[(nx, j)]] for j in range(ny + 1)]
    right_geom = KM.Geometry(right_geom_nodes)
    
    # Create Dirichlet condition on right boundary
    dirichlet_cond = KM.Conditions("TemperatureCondition", "RightBC", right_geom, 
                                    KM.Parameters({"VALUE": T_OUTER}))
    mp.AddCondition(dirichlet_cond)
    
    # Set up solver strategy
    KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)
    
    scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
    builder = KM.ResidualBasedBlockBuilderAndSolver(KM.SkylineLUFactorizationSolver())
    strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder, True, False, False, False)
    
    strategy.Initialize()
    strategy.Solve()
    
    # Extract interface temperatures
    T_if = np.array([mp.Nodes[nid].GetSolutionStepValue(KM.TEMPERATURE) 
                     for nid in iface_node_ids])
    
    # Compute consistent flux from reactions
    # Reaction at interface nodes gives us the consistent flux
    reactions = np.array([mp.Nodes[nid].GetSolutionStepValue(KM.REACTION_FLUX) 
                          for nid in iface_node_ids])
    
    # Convert to flux density
    hy = (Y1 - Y0) / ny
    weights = np.full(ny + 1, hy)
    weights[0] = weights[-1] = hy / 2
    
    # Outward normal for B at interface is -x direction
    # q_outward = -K * grad(T) . n = reaction / weight (with proper sign)
    q_outward = reactions / weights
    
    return mp, T_if, q_outward, iface_y, n_nodes

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--work_dir', type=str, default='.')
    args = parser.parse_args()
    
    work_dir = Path(args.work_dir)
    level = args.level
    
    # Mesh resolution: h = 1/8, 1/16, 1/32 for levels 1, 2, 3
    h_base = 1.0 / (8 * level)
    nx = int(round(0.875 / h_base))
    ny = int(round(1.0 / h_base))
    
    print(f"Kratos Subdomain B: level={level}, nx={nx}, ny={ny}, h={h_base:.6f}", flush=True)
    
    # Solve
    mp, T_if, q_outward, iface_y, n_nodes = create_mesh_and_solve(nx, ny, None, work_dir)
    
    # Export results
    export_data = {
        "field_name": "temperature",
        "n_points": len(T_if),
        "coordinates": [[float(IFACE_X), float(y)] for y in iface_y],
        "values": [float(t) for t in T_if],
        "normal_fluxes": [float(q) for q in q_outward]
    }
    
    exports_file = work_dir / "exports.json"
    exports_file.write_text(json.dumps(export_data, indent=2))
    
    print(f"Kratos Subdomain B exported {len(T_if)} interface points", flush=True)
    print(f"  T range: [{T_if.min():.6f}, {T_if.max():.6f}]", flush=True)
    print(f"  q range: [{q_outward.min():.6f}, {q_outward.max():.6f}]", flush=True)
    print(f"NDOF = {n_nodes}", flush=True)

if __name__ == "__main__":
    main()
