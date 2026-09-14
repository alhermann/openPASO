from ngsolve import *
from netgen.geom2d import SplineGeometry

geo = SplineGeometry()
p0 = geo.AddPoint(0, 0)
p1 = geo.AddPoint(1, 0)
p2 = geo.AddPoint(1, 1)
p3 = geo.AddPoint(0, 1)
geo.Append(["line", p0, p1], leftdomain=1, rightdomain=0)
geo.Append(["line", p1, p2], leftdomain=1, rightdomain=0)
geo.Append(["line", p2, p3], leftdomain=1, rightdomain=0)
geo.Append(["line", p3, p0], leftdomain=1, rightdomain=0)

mesh = Mesh(geo.GenerateMesh(maxh=0.5))

print(f"Mesh has {mesh.nv} vertices")
print("First vertex:")
v = list(mesh.vertices)[0]
print(f"  type={type(v)}")
print(f"  dir=[{', '.join([x for x in dir(v) if not x.startswith('_')])}]")

# Try to get coordinates
verts = list(mesh.vertices)
for i, v in enumerate(verts[:3]):
    print(f"Vertex {i}: {v}")
