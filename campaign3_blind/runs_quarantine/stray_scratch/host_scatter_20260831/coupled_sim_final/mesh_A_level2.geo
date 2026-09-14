
// Mesh for subdomain A: (0, 0.625) x (0, 1.0)
Point(1) = {0, 0, 0, 0.1};
Point(2) = {0.625, 0, 0, 0.1};
Point(3) = {0.625, 1.0, 0, 0.0625};
Point(4) = {0, 1.0, 0, 0.0625};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Physical Line("bottom") = {1};
Physical Line("interface") = {2};
Physical Line("top") = {3};
Physical Line("left") = {4};
Physical Surface("domain") = {1};

Mesh 2;
