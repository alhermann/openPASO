#!/usr/bin/env python3
"""FEniCSx participant - Subdomain A (DIRICHLET side)."""
import json, os, sys
from pathlib import Path
import numpy as np

import dolfinx
dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

from mpi4py import MPI
from ufl import dx, grad, inner, TrialFunction, TestFunction
from dolfinx.fem import FunctionSpace, Constant, dirichletbc, form, assemble_matrix, assemble_vector
from dolfinx.mesh import create_box_mesh, locate_entities_boundary
from dolfinx.linear import LinearProblem

X0, X1 = 0.0, 0.625
Y0, Y1 = 0.0, 1.0
IFACE_X = 0.625
K = 1.0
PARTNER = "B"
NX = int(os.environ.get('NX_A', '5'))
NY = int(os.environ.get('NY_A', '8'))
T_INIT = 0.0

def f_source(x):
    return (-12*x[0]**3*x[1]/5 + 14*x[0]**3/5 - 8979*x[0]**2*x[1]/2000 - 5147*x[0]**2/12000 
            - 12*x[0]*x[1]**3/5 + 42*x[0]*x[1]**2/5 - 1779*x[0]*x[1]/800 - 1007*x[0]/1200 
            - 2993*x[1]**3/2000 - 5147*x[1]**2/12000 + 4621*x[1]/2400)

def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get(PARTNER)
    except:
        return None

def interpolate_data(imp, ys, key, fallback):
    if imp is None or "coordinates" not in imp:
        return np.full(len(ys), float(fallback))
    coords = np.array(imp["coordinates"])
    vals = np.array(imp.get(key, [])).flatten()
    if len(vals) != len(coords):
        return np.full(len(ys), float(fallback))
    src_y = coords[:, 1]
    o = np.argsort(src_y)
    return np.interp(ys, src_y[o], vals[o])

def main():
    print("FEniCSx Subdomain A starting...", flush=True)
    
    domain = ((X0, X1), (Y0, Y1), (0, 0))
    mesh = create_box_mesh(MPI.COMM_WORLD, domain, (NX, NY, 1), dtype=np.float64)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    
    imp = read_imports()
    iface_y = np.linspace(Y0, Y1, NY + 1)
    iface_temps = interpolate_data(imp, iface_y, "values", T_INIT)
    
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, 0)
    
    left_facets = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], X0, atol=1e-5))
    top_facets = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], Y1, atol=1e-5))
    bot_facets = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], Y0, atol=1e-5))
    iface_facets = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], IFACE_X, atol=1e-5))
    
    u, v = TrialFunction(V), TestFunction(V)
    a = K * inner(grad(u), grad(v)) * dx
    L = f_source(mesh.x) * v * dx
    
    bcs = [
        dirichletbc(Constant(mesh, 0.0), V.subset_dof_topological_entity(fdim, left_facets)),
        dirichletbc(Constant(mesh, 0.0), V.subset_dof_topological_entity(fdim, top_facets)),
        dirichletbc(Constant(mesh, 0.0), V.subset_dof_topological_entity(fdim, bot_facets)),
    ]
    
    iface_nodes = mesh.topology.connectivity(fdim, 0).transpose().connectivity(iface_facets).flatten()
    unique_iface_nodes = np.unique(iface_nodes)
    iface_node_y = mesh.geometry.x[unique_iface_nodes, 1]
    iface_temps_at_nodes = np.interp(iface_node_y, iface_y, iface_temps)
    
    bc_values = np.zeros(V.dofmap.index_map.size_local + V.dofmap.index_map.offset, dtype=np.float64)
    for i, node_idx in enumerate(unique_iface_nodes):
        dof = V.dofmap.list[node_idx]
        bc_values[dof] = iface_temps_at_nodes[i]
    
    bcs.append(dirichletbc(bc_values, V.dofmap.list[unique_iface_nodes]))
    
    problem = LinearProblem(form(a), form(L), bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    
    # Extract interface data and compute flux from reactions
    T_if, qn, coords_list = [], [], []
    
    A = assemble_matrix(form(a), bcs=bcs)
    A.assemble()
    b = assemble_vector(form(L))
    b.apply(bcs)
    
    u_vec = uh.x.array
    Au = A.mat.mult(u_vec, mode='normal')
    r = Au - b
    
    dy = (Y1 - Y0) / NY
    for i, node_idx in enumerate(unique_iface_nodes):
        y = mesh.geometry.x[node_idx, 1]
        t_val = uh.x.array[V.dofmap.list[node_idx]]
        T_if.append(t_val)
        coords_list.append([IFACE_X, y])
        
        dof = V.dofmap.list[node_idx]
        reaction = r[dof]
        
        if i == 0 or i == len(unique_iface_nodes) - 1:
            weight = dy / 2
        else:
            weight = dy
        
        qn.append(-reaction / weight)
    
    exports = {
        "field_name": "temperature",
        "n_points": len(unique_iface_nodes),
        "coordinates": coords_list,
        "values": [float(t) for t in T_if],
        "normal_fluxes": [float(q) for q in qn]
    }
    
    Path("exports.json").write_text(json.dumps(exports, indent=2))
    print(f"[FEniCSx Subdomain A] n={len(unique_iface_nodes)}, T=[{min(T_if):.6g},{max(T_if):.6g}], qn=[{min(qn):.6g},{max(qn):.6g}]", flush=True)
    print(f"NDOF = {V.dofmap.index_map.size}", flush=True)

if __name__ == "__main__":
    main()
