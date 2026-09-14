from ngsolve import *
from netgen.geom2d import unit_square

mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))
fes = H1(mesh, order=1)
u, v = fes.TnT()

# Try different ways to create CoefficientFunction
print("Testing CoefficientFunction...")

# Method 1: Using lambda with x, y
try:
    cf1 = CoefficientFunction((lambda x, y: x**2 + y**2))
    print("Method 1 (single lambda): OK")
except Exception as e:
    print(f"Method 1 failed: {e}")

# Method 2: Using tuple of lambdas  
try:
    cf2 = CoefficientFunction((lambda x, y: x, lambda x, y: y))
    print("Method 2 (tuple of lambdas): OK")
except Exception as e:
    print(f"Method 2 failed: {e}")

# Method 3: Using string expression
try:
    cf3 = CoefficientFunction(("x", "y"))
    print("Method 3 (string tuple): OK")
except Exception as e:
    print(f"Method 3 failed: {e}")

# Method 4: Using C++ style expression
try:
    cf4 = CoefficientFunction("x*x + y*y")
    print("Method 4 (C++ string): OK")
except Exception as e:
    print(f"Method 4 failed: {e}")

# Test with vector space
fes_vec = VectorH1(mesh, order=1)
u, v = fes_vec.TnT()

# Method 5: For vector space
try:
    f = LinearForm(fes_vec)
    f += CoefficientFunction(("x", "y")) * v * dx
    f.Assemble()
    print("Method 5 (vector CF with strings): OK")
except Exception as e:
    print(f"Method 5 failed: {e}")
