#!/usr/bin/env python3
"""
Solve plane strain elasticity problem using FEniCSx/dolfinx.
Problem: -div(sigma(u)) = f with sigma(u) = 2*mu*eps(u) + lambda*tr(eps(u))*I
Domain: unit square (0,1) x (0,1)
BCs: u = 0 on all boundaries
Material: lambda = 4999000, mu = 1000 (nearly incompressible)
"""

import numpy as np
import dolfinx
from mpi4py import MPI
from dolfinx import fem, io, mesh, la
from dolfinx.fem import functionspace, Form
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.linear import LinearProblem
import ufl
from ufl import grad, inner, tr, sym, Identity, Variable

def solve_elasticity(N, output_prefix):
    """Solve the elasticity problem on a mesh with N elements per side."""
    
    # Create mesh
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N, 
                                              cell_type=dolfinx.mesh.CellType.quadrilateral)
    
    # Material parameters
    lam = 4999000.0
    mu = 1000.0
    
    # Define function space (vector valued, quadratic elements for better accuracy)
    V = functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))
    
    # Apply Dirichlet boundary conditions (u = 0 on all boundaries)
    def boundary(x):
        return np.logical_or(np.isclose(x[0], 0.0), 
                            np.logical_or(np.isclose(x[0], 1.0),
                                         np.logical_or(np.isclose(x[1], 0.0),
                                                      np.isclose(x[1], 1.0))))
    
    dofmap = V.dofmap
    ndofs_per_cell = dofmap.list.shape[1] // V.value_shape[0]
    cells = dolfinx.mesh.locate_entities_boundary(domain, 1, boundary)
    facets = dolfinx.mesh.entities_to_facets(domain.index_map, 1, cells)
    bdry_dofs = dolfinx.fem.dirichletzeroes(V, facets)
    
    # Define source term f
    # f_ux(x, y) = -24000*x**5*y + 12000*x**5 + ...
    # f_uy(x, y) = -40008*x**5*y/4999 + 20004*x**5/4999 + ...
    
    x = ufl.SpatialCoordinate(domain)
    
    # f_ux expression
    f_ux = (-24000*x[0]**5*x[1] + 12000*x[0]**5 
            + 74884980*x[0]**4*x[1]**2/4999 + 60088020*x[0]**4*x[1]/4999 
            - 52506170*x[0]**4/4999 - 401120240*x[0]**3*x[1]**3/4999 
            + 451630344*x[0]**3*x[1]**2/4999 + 39351872*x[0]**3*x[1]/4999 
            - 74938324*x[0]**3/4999 + 74884980*x[0]**2*x[1]**4/4999 
            + 121976400*x[0]**2*x[1]**3/4999 - 242452500*x[0]**2*x[1]**2/4999 
            - 59207844*x[0]**2*x[1]/4999 + 67446492*x[0]**2/4999 
            - 360072*x[0]*x[1]**5/4999 - 74124828*x[0]*x[1]**4/4999 
            + 238711752*x[0]*x[1]**3/4999 - 224034816*x[0]*x[1]**2/4999 
            + 59747952*x[0]*x[1]/4999 + 10002*x[0]/4999 
            + 180036*x[1]**5/4999 + 12097420*x[1]**4/4999 
            - 59727948*x[1]**3/4999 + 67446492*x[1]**2/4999 - 4000*x[1])
    
    # f_uy expression
    f_uy = (-40008*x[0]**5*x[1]/4999 + 20004*x[0]**5/4999 
            + 299039820*x[0]**4*x[1]**2/4999 - 299079828*x[0]**4*x[1]/4999 
            + 49829968*x[0]**4/4999 - 100113360*x[0]**3*x[1]**3/4999 
            - 117975600*x[0]**3*x[1]**2/4999 + 208291000*x[0]**3*x[1]/4999 
            - 39731948*x[0]**3/4999 + 299039820*x[0]**2*x[1]**4/4999 
            - 448189656*x[0]**2*x[1]**3/4999 - 61848372*x[0]**2*x[1]**2/4999 
            + 225875184*x[0]**2*x[1]/4999 - 30114024*x[0]**2/4999 
            - 6000*x[0]*x[1]**5 - 59087820*x[0]*x[1]**4/4999 
            + 158367680*x[0]*x[1]**3/4999 + 60768156*x[0]*x[1]**2/4999 
            - 135053016*x[0]*x[1]/4999 + 4000*x[0] 
            + 3000*x[1]**5 - 60138030*x[1]**4/4999 
            + 75245052*x[1]**3/4999 - 30114024*x[1]**2/4999 
            + 10002*x[1]/4999)
    
    f = ufl.as_vector([f_ux, f_uy])
    
    # Define trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Strain tensor: eps(u) = (grad(u) + grad(u)^T)/2
    eps = sym(grad(u))
    
    # Stress tensor: sigma(u) = 2*mu*eps(u) + lambda*tr(eps(u))*I
    sigma = 2*mu*eps + lam*tr(eps)*Identity(2)
    
    # Weak form: a(u, v) = L(v)
    # a(u, v) = int(sigma(u) : eps(v)) dx
    # L(v) = int(f . v) dx
    a = inner(sigma, eps(v)) * ufl.dx
    L = inner(f, v) * ufl.dx
    
    # Compile forms
    a_form = Form(a)
    L_form = Form(L)
    
    # Assemble system matrix and RHS
    A = fem.petsc.assemble_matrix(a_form, bcs=[bdry_dofs])
    A.assemble()
    
    b = fem.petsc.assemble_vector(L_form)
    fem.petsc.apply_lifting(b, [a_form], [[bdry_dofs]])
    b.ghostUpdate(addv=dolfinx.la.InsertMode.add)
    
    # Apply Dirichlet BCs to RHS
    dolfinx.fem.petsc.set_bc(b, bdry_dofs)
    
    # Solve linear system
    x = la.vector(A.size, A.comm)
    ksp = dolfinx.la.default_krylov_solver_options(MPI.COMM_WORLD)
    ksp["ksp_type"] = "preonly"
    ksp["pc_type"] = "lu"
    ksp["pc_factor_mat_solver_type"] = "mumps"
    
    solver = la.KrylovSolver("preonly", "lu")
    solver.set_operator(A)
    solver.solve(x, b)
    
    # Create solution function
    u_sol = fem.Function(V)
    u_sol.x.array[:] = x.array
    
    # Count degrees of freedom
    num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    print(f"NDOF = {num_dofs}")
    
    # Write solution to file
    with io.XDMFFile(domain.comm, output_prefix + ".xdmf", "w") as xdmf_file:
        xdmf_file.write_mesh(domain)
        xdmf_file.write_function(u_sol)
    
    return u_sol, domain, num_dofs

def interpolate_at_points(u_sol, domain, probe_points):
    """Interpolate solution at given probe points."""
    from dolfinx.fem import locate_dofs_topological
    from dolfinx.geometry import PointGeometry
    import pygeomalgebra as pga
    
    # Create a geometry module for point location
    geom_module = PointGeometry(domain.geometry.index_map, domain.geometry.cmap)
    
    results = []
    for px, py in probe_points:
        # Find the element containing the point
        # This is a simplified approach - in practice, you'd need proper point location
        pass
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: solve_elasticity.py <N> <output_prefix>")
        sys.exit(1)
    
    N = int(sys.argv[1])
    output_prefix = sys.argv[2]
    
    u_sol, domain, ndof = solve_elasticity(N, output_prefix)
    print(f"NDOF = {ndof}")
