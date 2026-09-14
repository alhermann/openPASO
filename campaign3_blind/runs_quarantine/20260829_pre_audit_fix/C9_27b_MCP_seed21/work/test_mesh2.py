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

verts = list(mesh.vertices)
for i, v in enumerate(verts):
    pt = v.point
    print(f"Vertex {i}: point={pt}, type={type(pt)}")
    if hasattr(pt, '__getitem__'):
        print(f"  pt[0]={pt[0]}, pt[1]={pt[1]}")
    if hasattr(pt, 'x'):
        print(f"  pt.x={pt.x}, pt.y={pt.y}")
