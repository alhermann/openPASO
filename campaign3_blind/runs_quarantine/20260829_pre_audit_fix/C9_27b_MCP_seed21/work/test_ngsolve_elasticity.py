"""Test NGSolve elasticity setup."""
from ngsolve import *
from netgen.geom2d import SplineGeometry

# Simple rectangle
geo = SplineGeometry()
p0 = geo.AddPoint(0, 0)
p1 = geo.AddPoint(1, 0)
p2 = geo.AddPoint(1, 0.625)
p3 = geo.AddPoint(0, 0.625)

geo.Append(["line", p0, p1], leftdomain=1, rightdomain=0, bc="bottom")
geo.Append(["line", p1, p2], leftdomain=1, rightdomain=0, bc="right")
geo.Append(["line", p2, p3], leftdomain=1, rightdomain=0, bc="top")
geo.Append(["line", p3, p0], leftdomain=1, rightdomain=0, bc="left")

mesh = Mesh(geo.GenerateMesh(maxh=0.1))

fes = VectorH1(mesh, order=1, dirichlet=".*")
u, v = fes.TnT()

LAM, MU = 480, 1200

def Strain(u):
    return Sym(Grad(u))

def Stress(u):
    return 2 * MU * Strain(u) + LAM * Trace(Strain(u)) * Id(2)

a = BilinearForm(fes)
a += InnerProduct(Stress(u), Strain(v)) * dx
a.Assemble()

# Simple body force
f = LinearForm(fes)
f += CoefficientFunction((1.0, 0.0)) * v * dx
f.Assemble()

gfu = GridFunction(fes)
with TaskManager():
    inv_a = a.mat.Inverse(fes.FreeDofs(), inverse='sparsecholesky')
    gfu.vec.data = inv_a * f.vec

print(f"NDOF = {fes.ndof}")
print(f"Max displacement: {max(abs(gfu.vec[i]) for i in range(len(gfu.vec))):.6e}")
print("NGSolve elasticity test PASSED")
