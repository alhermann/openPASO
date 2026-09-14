"""Kratos participant for subdomain A (Dirichlet side)."""
import json
from pathlib import Path
import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication

PARTNER = "B"
X0, X1 = 0.0, 0.625
Y0, Y1 = 0.0, 1.0
Z0, Z1 = 0.0, 1.0
K = 1.0
T_INIT = 0.0


def source_term(x, y, z):
    return (-15*x**3*y**3/2 + 35*x**3*y**2/2 - 45*x**3*y*z**2/2 + 45*x**3*y*z/2 
            - 10*x**3*y + 35*x**3*z**2/2 - 35*x**3*z/2 + 3775*x**2*y**3/192 
            - 305*x**2*y**2/64 + 3775*x**2*y*z**2/64 - 3775*x**2*y*z/64 
            - 715*x**2*y/48 - 305*x**2*z**2/64 + 305*x**2*z/64 
            - 45*x*y**3*z**2/2 + 45*x*y**3*z/2 - 45*x*y**3/4 
            + 105*x*y**2*z**2/2 - 105*x*y**2*z/2 - 135*x*y**2/16 
            - 255*x*y*z**2/4 + 255*x*y*z/4 + 315*x*y/16 
            - 135*x*z**2/16 + 135*x*z/16 + 3775*y**3*z**2/192 
            - 3775*y**3*z/192 - 305*y**2*z**2/64 + 305*y**2*z/64 
            - 715*y*z**2/48 + 715*y*z/48)


def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text() or "{}").get(PARTNER)
    except:
        return None


def resample(imp, key, fallback, pts):
    n = len(pts)
    if imp is None or "coordinates" not in imp:
        return np.full(n, float(fallback))
    coords = np.asarray(imp["coordinates"], float)
    values = np.asarray(imp.get(key, []), float).ravel()
    if len(values) != len(coords):
        return np.full(n, float(fallback))
    src_yz = coords[:, 1:3]
    try:
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        interp = LinearNDInterpolator(src_yz, values)
        result = interp(pts)
        nan_mask = ~np.isfinite(result)
        if np.any(nan_mask):
            nn = NearestNDInterpolator(src_yz, values)
            result[nan_mask] = nn(pts[nan_mask])
        return result
    except:
        dist = ((pts[:, None, :] - src_yz[None, :, :]) ** 2).sum(-1)
        return values[np.argmin(dist, axis=1)]


# Read config
config = json.loads(Path("level_config.json").read_text()) if Path("level_config.json").exists() else {}
NX_GLOBAL = config.get("nx_global", 24)
level = config.get("level", 1)

# Calculate mesh sizes - ensure at least 2 elements per direction for interior points
nxA = int(round(NX_GLOBAL * 5 / 12))
nyA = int(round(NX_GLOBAL * 1 / 12))
nzA = int(round(NX_GLOBAL * 1 / 12))
# Ensure minimum of 2 elements to have interior interface nodes
nxA = max(nxA, 2)
nyA = max(nyA, 2)
nzA = max(nzA, 2)

print(f"[A] Mesh: {nxA}x{nyA}x{nzA}, level={level}")

model = KM.Model()
mp = model.CreateModelPart("Main")

mp.AddNodalSolutionStepVariable(KM.TEMPERATURE)
mp.AddNodalSolutionStepVariable(KM.REACTION_FLUX)
mp.AddNodalSolutionStepVariable(KM.HEAT_FLUX)
mp.AddNodalSolutionStepVariable(KM.FACE_HEAT_FLUX)
mp.AddNodalSolutionStepVariable(KM.CONDUCTIVITY)

nid = {}
node_id = 1
dx = (X1 - X0) / nxA
dy = (Y1 - Y0) / nyA
dz = (Z1 - Z0) / nzA

for k in range(nzA + 1):
    for j in range(nyA + 1):
        for i in range(nxA + 1):
            x = X0 + i * dx
            y = Y0 + j * dy
            z = Z0 + k * dz
            mp.CreateNewNode(node_id, x, y, z)
            nid[(i, j, k)] = node_id
            node_id += 1

props = mp.CreateNewProperties(1)
elem_id = 1
for k in range(nzA):
    for j in range(nyA):
        for i in range(nxA):
            corners = [nid[(i+di, j+dj, k+dk)] for di, dj, dk in 
                      [(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]]
            n000, n100, n010, n001, n110, n101, n011, n111 = corners
            for trios in [[n000,n100,n010,n001],[n100,n110,n010,n101],[n010,n110,n011,n001],
                         [n100,n110,n111,n101],[n001,n101,n011,n111],[n001,n101,n111,n011]]:
                mp.CreateNewElement("LaplacianElement3D4N", elem_id, trios, props)
                elem_id += 1

settings = KM.ConvectionDiffusionSettings()
settings.SetUnknownVariable(KM.TEMPERATURE)
settings.SetDiffusionVariable(KM.CONDUCTIVITY)
settings.SetVolumeSourceVariable(KM.HEAT_FLUX)
settings.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
mp.ProcessInfo[KM.CONVECTION_DIFFUSION_SETTINGS] = settings

for node in mp.Nodes:
    node.SetSolutionStepValue(KM.CONDUCTIVITY, K)
    node.SetSolutionStepValue(KM.HEAT_FLUX, source_term(node.X, node.Y, node.Z))

tol = 1e-9
outer_nodes = set()
iface_nodes = []
iface_coords = []

for k in range(nzA + 1):
    for j in range(nyA + 1):
        for i in range(nxA + 1):
            node = mp.Nodes[nid[(i, j, k)]]
            x, y, z = node.X, node.Y, node.Z
            
            on_outer = (abs(x - X0) < tol or abs(y - Y0) < tol or abs(y - Y1) < tol or 
                       abs(z - Z0) < tol or abs(z - Z1) < tol)
            
            if abs(x - X1) < tol:  # Interface face
                # Skip corners where interface meets outer boundary
                if not (abs(y - Y0) < tol or abs(y - Y1) < tol or abs(z - Z0) < tol or abs(z - Z1) < tol):
                    iface_nodes.append(nid[(i, j, k)])
                    iface_coords.append([X1, y, z])
            elif on_outer:
                outer_nodes.add(nid[(i, j, k)])

print(f"[A] Found {len(iface_nodes)} interior interface nodes, {len(outer_nodes)} outer nodes")

sorted_pairs = sorted(zip(iface_coords, iface_nodes), key=lambda p: (p[0][1], p[0][2]))
iface_coords = [p[0] for p in sorted_pairs]
iface_nodes = [p[1] for p in sorted_pairs]

if len(iface_nodes) == 0:
    print("[A] ERROR: No interior interface nodes found!")
    # Create dummy export to avoid crashing
    exports = {"field_name": "temperature", "n_points": 0, "coordinates": [], "values": [], "normal_fluxes": []}
    Path("exports.json").write_text(json.dumps(exports))
    Path(f"run_level{level}_A.log").write_text("NDOF = 0\nERROR: No interface nodes\n")
    exit(1)

imp = read_imports()
pts_yz = np.array([[c[1], c[2]] for c in iface_coords])
T_in = resample(imp, "values", T_INIT, pts_yz)

for node_id in outer_nodes:
    mp.Nodes[node_id].SetSolutionStepValue(KM.TEMPERATURE, 0.0)
    mp.Nodes[node_id].Fix(KM.TEMPERATURE)

for node_id, T_val in zip(iface_nodes, T_in):
    mp.Nodes[node_id].SetSolutionStepValue(KM.TEMPERATURE, float(T_val))
    mp.Nodes[node_id].Fix(KM.TEMPERATURE)

KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)

scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
builder = KM.ResidualBasedBlockBuilderAndSolver(KM.SkylineLUFactorizationSolver())
strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder, True, False, False, False)
strategy.Initialize()
strategy.Solve()

T_iface = np.array([mp.Nodes[nid].GetSolutionStepValue(KM.TEMPERATURE) for nid in iface_nodes])
reactions = np.array([mp.Nodes[nid].GetSolutionStepValue(KM.REACTION_FLUX) for nid in iface_nodes])

w_sum = dy * dz
Q = -reactions / w_sum

exports = {
    "field_name": "temperature",
    "n_points": len(iface_nodes),
    "coordinates": iface_coords,
    "values": [float(t) for t in T_iface],
    "normal_fluxes": [float(q) for q in Q]
}
Path("exports.json").write_text(json.dumps(exports, indent=2))

ndof = len(mp.Nodes)
Path(f"run_level{level}_A.log").write_text(f"NDOF = {ndof}\n")

print(f"[Kratos A] {ndof} DOFs, {len(iface_nodes)} iface nodes, T=[{T_iface.min():.4f},{T_iface.max():.4f}]")
