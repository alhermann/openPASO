#!/usr/bin/env python3
import numpy as np
from dolfinx import fem, mesh
from dolfinx.fem import functionspace, Function, Constant
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl

INTERFACE_X = 0.625
DOMAIN_A_XMAX = 0.625
DOMAIN_B_XMIN = 0.625
DOMAIN_B_XMAX = 1.5
DOMAIN_YMAX = 1.0

K_A = 1.0
LAMBDA_A = 600.0
MU_A = 400.0
BETA_A = 1.0

K_B = 3.0
LAMBDA_B = 600.0
MU_B = 1600.0
BETA_B = 1.0

side = "B"
dirichlet_interface = False
nx, ny = 14, 16

if side == "A":
    x_min, x_max = 0.0, DOMAIN_A_XMAX
    k_val = K_A
    lambda_val = LAMBDA_A
    mu_val = MU_A
    beta_val = BETA_A
else:
    x_min, x_max = DOMAIN_B_XMIN, DOMAIN_B_XMAX
    k_val = K_B
    lambda_val = LAMBDA_B
    mu_val = MU_B
    beta_val = BETA_B

msh = mesh.create_rectangle(MPI.COMM_WORLD, [[x_min, 0.0], [x_max, DOMAIN_YMAX]], 
                            [nx, ny], cell_type=mesh.CellType.quadrilateral)

V_T = functionspace(msh, ("Lagrange", 1))
V_u = functionspace(msh, ("Lagrange", 1, (2,)))

T = ufl.TrialFunction(V_T)
u = ufl.TrialFunction(V_u)
v_T = ufl.TestFunction(V_T)
v_u = ufl.TestFunction(V_u)

x_coord = ufl.SpatialCoordinate(msh)
if side == "A":
    f_T_expr = (-6*x_coord[0]**3*x_coord[1] + 8*x_coord[0]**3/5 - x_coord[0]**2*x_coord[1]/2 - 2*x_coord[0]**2/3 
                - 6*x_coord[0]*x_coord[1]**3 + 24*x_coord[0]*x_coord[1]**2/5 + 67*x_coord[0]*x_coord[1]/10 - 11*x_coord[0]/15 
                - x_coord[1]**3/6 - 2*x_coord[1]**2/3 + 5*x_coord[1]/6)
    f_x_expr = (-33*x_coord[0]**2*x_coord[1]**3/5 - 312*x_coord[0]**2*x_coord[1]**2/25 + 453*x_coord[0]**2*x_coord[1]/25 - 18*x_coord[0]**2/5 
                + 599*x_coord[0]*x_coord[1]**3/10 + 4754*x_coord[0]*x_coord[1]**2/75 - 17597*x_coord[0]*x_coord[1]/150 + 112*x_coord[0]/5 
                - 84*x_coord[1]**5/25 - 147*x_coord[1]**4/25 + 18277*x_coord[1]**3/300 + 893*x_coord[1]**2/30 
                - 1549*x_coord[1]/20 + 15)
    f_y_expr = (3*x_coord[0]**3*x_coord[1]**2 - 8*x_coord[0]**3*x_coord[1]/5 - x_coord[0]**3/5 + 2693*x_coord[0]**2*x_coord[1]**2/20 
                + 7106*x_coord[0]**2*x_coord[1]/75 - 26333*x_coord[0]**2/300 - 12*x_coord[0]*x_coord[1]**4 - 84*x_coord[0]*x_coord[1]**3/5 
                + 4241*x_coord[0]*x_coord[1]**2/20 + 301*x_coord[0]*x_coord[1]/3 - 2173*x_coord[0]/20 + 56*x_coord[1]**4/15 
                + 392*x_coord[1]**3/75 - 364*x_coord[1]**2/25 + 28*x_coord[1]/5)
else:
    f_T_expr = (-2*x_coord[0]**3*x_coord[1]/3 + 8*x_coord[0]**3/45 - 8*x_coord[0]**2*x_coord[1]/3 + 4*x_coord[0]**2/9 
                - 2*x_coord[0]*x_coord[1]**3/3 + 8*x_coord[0]*x_coord[1]**2/15 + 251*x_coord[0]*x_coord[1]/120 - 41*x_coord[0]/90 
                - 8*x_coord[1]**3/9 + 4*x_coord[1]**2/9 + 829*x_coord[1]/144 - 11/12)
    f_x_expr = (365273*x_coord[0]**2*x_coord[1]**3/5985 + 273076*x_coord[0]**2*x_coord[1]**2/4275 - 3555593*x_coord[0]**2*x_coord[1]/29925 
                + 15192*x_coord[0]**2/665 - 119754589*x_coord[0]*x_coord[1]**3/251370 - 359593607*x_coord[0]*x_coord[1]**2/718200 
                + 4672588891*x_coord[0]*x_coord[1]/5027400 - 13314341*x_coord[0]/74480 + 2532*x_coord[1]**5/175 
                + 633*x_coord[1]**4/25 + 4520447879*x_coord[1]**3/20109600 + 894533323*x_coord[1]**2/2298240 
                - 2001017755*x_coord[1]/3217536 + 28506033/238336)
    f_y_expr = (x_coord[0]**3*x_coord[1]**2/9 - 8*x_coord[0]**3*x_coord[1]/135 - x_coord[0]**3/135 - 1720331*x_coord[0]**2*x_coord[1]**2/1764 
                - 1032893*x_coord[0]**2*x_coord[1]/1512 + 13423129*x_coord[0]**2/21168 + 27852*x_coord[0]*x_coord[1]**4/665 
                + 27852*x_coord[0]*x_coord[1]**3/475 + 9525905951*x_coord[0]*x_coord[1]**2/6703200 + 269425655*x_coord[0]*x_coord[1]/229824 
                - 872177333*x_coord[0]/846720 - 1436809*x_coord[1]**4/13965 - 1436809*x_coord[1]**3/9975 
                + 976524757*x_coord[1]**2/4468800 - 1508594569*x_coord[1]/5362560 + 13360741/112896)

a_T = k_val * ufl.dot(ufl.grad(T), ufl.grad(v_T)) * ufl.dx
L_T = f_T_expr * v_T * ufl.dx

eps = lambda w: ufl.sym(ufl.nabla_grad(w))
sigma = 2*mu_val*eps(u) + lambda_val*ufl.div(u)*ufl.Identity(2) - beta_val*T*ufl.Identity(2)

a_u = ufl.inner(sigma, ufl.grad(v_u)) * ufl.dx
L_u = ufl.dot(ufl.as_vector([f_x_expr, f_y_expr]), v_u) * ufl.dx

def outer_boundary(x):
    tol = 1e-14
    on_left = abs(x[0] - x_min) < tol
    on_right = abs(x[0] - x_max) < tol
    on_bottom = abs(x[1]) < tol
    on_top = abs(x[1] - DOMAIN_YMAX) < tol
    return on_left or on_right or on_bottom or on_top

def interface_boundary(x):
    tol = 1e-14
    if side == "A":
        return abs(x[0] - x_max) < tol
    else:
        return abs(x[0] - x_min) < tol

bc_T_outer = fem.dirichletbc(0.0, fem.locate_dofs_geometrical(V_T, outer_boundary), V_T)
bc_u_outer = fem.dirichletbc([0.0, 0.0], fem.locate_dofs_geometrical(V_u, outer_boundary), V_u)

if dirichlet_interface:
    bc_T_interface = fem.dirichletbc(0.0, fem.locate_dofs_geometrical(V_T, interface_boundary), V_T)
    bc_u_interface = fem.dirichletbc([0.0, 0.0], fem.locate_dofs_geometrical(V_u, interface_boundary), V_u)
    bcs_T = [bc_T_outer, bc_T_interface]
    bcs_u = [bc_u_outer, bc_u_interface]
else:
    bcs_T = [bc_T_outer]
    bcs_u = [bc_u_outer]

T_sol = Function(V_T)
problem_T = LinearProblem(a_T, L_T, bcs=bcs_T, 
                          petsc_options_prefix="heat_" + side + "_" + str(2),
                          petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
T_sol = problem_T.solve()

sigma_with_T = 2*mu_val*eps(u) + lambda_val*ufl.div(u)*ufl.Identity(2) - beta_val*T_sol*ufl.Identity(2)
a_u_form = ufl.inner(sigma_with_T, ufl.grad(v_u)) * ufl.dx
L_u_form = ufl.dot(ufl.as_vector([f_x_expr, f_y_expr]), v_u) * ufl.dx

u_sol = Function(V_u)
problem_u = LinearProblem(a_u_form, L_u_form, bcs=bcs_u,
                          petsc_options_prefix="elastic_" + side + "_" + str(2),
                          petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_sol = problem_u.solve()

ndof_T = V_T.dofmap.index_map.size_local
ndof_u = V_u.dofmap.index_map.size_local * 2
total_ndof = ndof_T + ndof_u
print("NDOF = " + str(total_ndof), flush=True)

T_vals = T_sol.x.array
u_vals = u_sol.x.array
coords = msh.geometry.x

np.save("/home/alexander/work/coupled_sim/T_B_level2.npy", T_vals)
np.save("/home/alexander/work/coupled_sim/u_B_level2.npy", u_vals)
np.save("/home/alexander/work/coupled_sim/coords_B_level2.npy", coords)

print("Solution saved for side B, level 2", flush=True)