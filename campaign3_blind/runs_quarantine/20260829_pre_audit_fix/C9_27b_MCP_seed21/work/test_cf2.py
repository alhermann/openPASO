from ngsolve import *
from netgen.geom2d import unit_square

mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))
fes_vec = VectorH1(mesh, order=1)
u, v = fes_vec.TnT()

# Try using ComponentWise or other approaches
print("Testing body force approaches...")

# Method 1: Using CoefficientFunction with tuple of expressions as strings
try:
    cf = CoefficientFunction(("x*x", "y*y"))
    f = LinearForm(fes_vec)
    f += cf * v * dx
    f.Assemble()
    print("Method 1 (tuple of string expressions): OK")
except Exception as e:
    print(f"Method 1 failed: {e}")

# Method 2: Using separate components
try:
    f = LinearForm(fes_vec)
    f += x*x * v[0] * dx + y*y * v[1] * dx
    f.Assemble()
    print("Method 2 (direct x,y in form): OK")
except Exception as e:
    print(f"Method 2 failed: {e}")

# Method 3: Using GridFunction interpolation
try:
    from numpy import pi
    def make_cf():
        return CoefficientFunction((lambda x, y: x*x, lambda x, y: y*y))
    
    # Actually try the working approach from template
    f = LinearForm(fes_vec)
    f += CoefficientFunction((x*x, y*y)) * v * dx
    f.Assemble()
    print("Method 3 (CoefficientFunction with x,y symbols): OK")
except Exception as e:
    print(f"Method 3 failed: {e}")
