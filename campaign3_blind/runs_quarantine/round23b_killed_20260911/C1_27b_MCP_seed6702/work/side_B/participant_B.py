#!/usr/bin/env python3
"""
Side B participant script for thermo-elastic coupling (Neumann side).
Uses FEniCSx/dolfinx to solve the coupled problem.

Role: NEUMANN side - receives fluxes [qn, qx, qy] from partner, applies as natural BC,
returns values [T, ux, uy] at interface points.
"""

import json
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.mesh import locate_entities_boundary
from dolfinx.fem import functionspace, Element
import ufl

def main():
    # Read config
    with open("config.json", "r") as f:
        config = json.load(f)
    
    level = config["level"]
    nx = config["nx"]
    ny = config["ny"]
    x0 = config["domain"]["x0"]
    x1 = config["domain"]["x1"]
    y0 = config["domain"]["y0"]
    y1 = config["domain"]["y1"]
    
    k = config["material"]["k"]  # thermal conductivity
    lam = config["material"]["lambda"]  # Lamé parameter λ
    mu = config["material"]["mu"]  # shear modulus μ
    beta = config["material"]["beta"]  # thermal expansion coefficient
    
    # Source terms as strings
    f_T_str = config["sources"]["f_T"]
    f_x_str = config["sources"]["f_x"]
    f_y_str = config["sources"]["f_y"]
    
    # Create mesh for subdomain B
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD, 
        np.array([x0, y0]), 
        np.array([x1, y1]), 
        (nx, ny), 
        mesh.CellType.quadrilateral
    )
    
    # Define function spaces (P1 for both T and u)
    V_T = functionspace(domain, ("Lagrange", 1))
    V_u = functionspace(domain, ("Lagrange", 1, (2,)))  # Vector P1
    
    # Trial and test functions
    T = ufl.TrialFunction(V_T)
    u = ufl.TrialFunction(V_u)
    v_T = ufl.TestFunction(V_T)
    v_u = ufl.TestFunction(V_u)
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Interface location (left edge of subdomain B)
    interface_x = x0  # x = 0.625
    
    tol = 1e-14
    
    def left_boundary(x):
        return np.isclose(x[0], interface_x, atol=tol)
    
    def outer_boundary(x):
        return np.logical_or.reduce([
            np.isclose(x[0], x1, atol=tol),
            np.isclose(x[1], y0, atol=tol),
            np.isclose(x[1], y1, atol=tol)
        ])
    
    # Find boundary facets
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    
    left_facets = locate_entities_boundary(domain, fdim, left_boundary)
    outer_facets = locate_entities_boundary(domain, fdim, outer_boundary)
    
    # Create measure
    ds = ufl.Measure("ds", domain=domain, subdomain_data=None)
    dx = ufl.Measure("dx", domain=domain)
    
    # Read imports.json for fluxes
    try:
        with open("imports.json", "r") as f:
            imports = json.load(f)
        
        import_coords = np.array(imports["coordinates"])
        import_fluxes = np.array(imports["normal_fluxes"])  # [qn, qx, qy] for each point
        
        qn_import = import_fluxes[:, 0]  # Heat flux normal component
        qx_import = import_fluxes[:, 1]  # Traction x component
        qy_import = import_fluxes[:, 2]  # Traction y component
        
        print(f"Read {len(import_coords)} import points from imports.json")
    except FileNotFoundError:
        print("No imports.json found, using zero fluxes")
        qn_import = np.zeros(ny + 1)
        qx_import = np.zeros(ny + 1)
        qy_import = np.zeros(ny + 1)
        import_coords = np.column_stack([np.full(ny + 1, interface_x), np.linspace(y0, y1, ny + 1)])
    
    # Use average of imported fluxes for Neumann BC
    avg_qn = float(np.mean(qn_import))
    avg_qx = float(np.mean(qx_import))
    avg_qy = float(np.mean(qy_import))
    
    print(f"Average imported fluxes: qn={avg_qn}, qx={avg_qx}, qy={avg_qy}")
    
    # Create constants for source terms and fluxes
    f_T_const = fem.Constant(domain, default_scalar_type(0.0))
    f_vec_const = fem.Constant(domain, np.array([0.0, 0.0]))
    
    qn_const = fem.Constant(domain, default_scalar_type(avg_qn))
    qx_const = fem.Constant(domain, default_scalar_type(avg_qx))
    qy_const = fem.Constant(domain, default_scalar_type(avg_qy))
    
    # Identity tensor
    I = ufl.Identity(2)
    
    # Strain tensor
    eps = ufl.symmetrized(ufl.nabla_grad(u))
    
    # Stress tensor (with thermal expansion)
    sigma = lam * ufl.tr(eps) * I + 2 * mu * eps - beta * T * I
    
    # Thermal bilinear form: -div(k grad T) = f_T
    a_T = k * ufl.dot(ufl.grad(T), ufl.grad(v_T)) * dx
    
    # Thermal linear form with Neumann BC on interface
    L_T = f_T_const * v_T * dx + qn_const * v_T * ds
    
    # Elastic bilinear form
    a_u = ufl.inner(sigma, ufl.symmetrized(ufl.nabla_grad(v_u))) * dx
    
    # Elastic linear form with traction BC on interface
    L_u = ufl.dot(f_vec_const, v_u) * dx + (qx_const * v_u[0] + qy_const * v_u[1]) * ds
    
    # Apply Dirichlet BCs on outer boundary
    # T = 0 on outer boundary
    dofs_T_outer = fem.locate_dofs_topological(V_T, fdim, outer_facets)
    bc_T = fem.dirichletbc(default_scalar_type(0.0), dofs_T_outer, V_T)
    
    # u = (0, 0) on outer boundary
    dofs_u_outer_0 = fem.locate_dofs_topological(V_u.sub(0).collapse(), fdim, outer_facets)
    dofs_u_outer_1 = fem.locate_dofs_topological(V_u.sub(1).collapse(), fdim, outer_facets)
    bc_u_0 = fem.dirichletbc(default_scalar_type(0.0), dofs_u_outer_0, V_u.sub(0))
    bc_u_1 = fem.dirichletbc(default_scalar_type(0.0), dofs_u_outer_1, V_u.sub(1))
    
    # Solve thermal problem first
    A_T = fem.petsc.assemble_matrix(a_T, bcs=[bc_T])
    A_T.assemble()
    b_T = fem.petsc.assemble_vector(L_T)
    bc_T.apply(b_T)
    
    T_sol = fem.Function(V_T)
    fem.petsc.solve(A_T, T_sol.vector, b_T)
    
    # Now solve elastic problem with known T
    # Update stress with solved temperature
    eps_u = ufl.symmetrized(ufl.nabla_grad(u))
    sigma_u = lam * ufl.tr(eps_u) * I + 2 * mu * eps_u - beta * T_sol * I
    
    a_u_with_T = ufl.inner(sigma_u, ufl.symmetrized(ufl.nabla_grad(v_u))) * dx
    L_u_full = ufl.dot(f_vec_const, v_u) * dx + (qx_const * v_u[0] + qy_const * v_u[1]) * ds
    
    A_u = fem.petsc.assemble_matrix(a_u_with_T, bcs=[bc_u_0, bc_u_1])
    A_u.assemble()
    b_u = fem.petsc.assemble_vector(L_u_full)
    bc_u_0.apply(b_u)
    bc_u_1.apply(b_u)
    
    u_sol = fem.Function(V_u)
    fem.petsc.solve(A_u, u_sol.vector, b_u)
    
    # Count NDOF
    ndof_T = V_T.dofmap.index_map.size_local * V_T.dofmap.index_map.block_size
    ndof_u = V_u.dofmap.index_map.size_local * V_u.dofmap.index_map.block_size
    total_ndof = ndof_T + ndof_u
    
    print(f"NDOF = {total_ndof}")
    
    # Export results at interface
    # Get unique interface node coordinates
    seen_coords = set()
    interface_node_coords = []
    
    connectivity = domain.topology.get_connectivity(fdim, 0)
    for facet in left_facets:
        vertices = connectivity.get_entity(facet)
        for v in vertices:
            coord = tuple(domain.geometry.x[v])
            if coord not in seen_coords:
                seen_coords.add(coord)
                interface_node_coords.append(coord)
    
    interface_node_coords = sorted(interface_node_coords, key=lambda c: (c[0], c[1]))
    
    # Evaluate solution at interface points
    interface_values = []
    for coord in interface_node_coords:
        xi, yi = coord
        T_val = T_sol.eval(np.array([xi, yi, 0.0]))
        u_val = u_sol.eval(np.array([xi, yi, 0.0]))
        interface_values.append([float(T_val), float(u_val[0]), float(u_val[1])])
    
    # Write exports.json
    exports = {
        "field_name": "values",
        "n_points": len(interface_node_coords),
        "coordinates": [list(c) for c in interface_node_coords],
        "values": interface_values,
        "normal_fluxes": []
    }
    
    with open("exports.json", "w") as f:
        json.dump(exports, f, indent=2)
    
    print(f"Wrote exports.json with {len(interface_node_coords)} points")
    
    # Check for finite values
    has_finite = True
    for vals in interface_values:
        for v in vals:
            if not np.isfinite(v):
                has_finite = False
                break
        if not has_finite:
            break
    
    if has_finite:
        print("All exported values are finite")
    else:
        print("WARNING: Some exported values are not finite!")
    
    # Write field CSV
    field_filename = f"field_level{level}.csv"
    with open(field_filename, "w") as f:
        f.write("x,y,T,ux,uy\n")
        for i in range(domain.geometry.x.shape[0]):
            x_coord = domain.geometry.x[i, 0]
            y_coord = domain.geometry.x[i, 1]
            T_val = T_sol.eval(np.array([x_coord, y_coord, 0.0]))
            u_val = u_sol.eval(np.array([x_coord, y_coord, 0.0]))
            f.write(f"{x_coord:.11e},{y_coord:.11e},{T_val:.11e},{u_val[0]:.11e},{u_val[1]:.11e}\n")
    
    print(f"Wrote {field_filename}")
    
    # Write interface CSV
    interface_filename = f"interface_level{level}.csv"
    with open(interface_filename, "w") as f:
        f.write("x,y,T,ux,uy,qn,tx,ty\n")
        for i, coord in enumerate(interface_node_coords):
            f.write(f"{coord[0]:.11e},{coord[1]:.11e},{interface_values[i][0]:.11e},{interface_values[i][1]:.11e},{interface_values[i][2]:.11e},0.0,0.0,0.0\n")
    
    print(f"Wrote {interface_filename}")
    
    print("Side B participant completed successfully")
    return 0

if __name__ == "__main__":
    exit(main())
