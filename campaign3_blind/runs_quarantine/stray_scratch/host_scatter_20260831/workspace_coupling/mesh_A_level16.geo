// Mesh for subdomain A at level 16
SetFactory("OpenCASCADE");

// Define corners
Point(1) = {0, 0, 0, 1.0};
Point(2) = {0.625, 0, 0, 1.0};
Point(3) = {0.625, 1, 0, 1.0};
Point(4) = {0, 1, 0, 1.0};

// Define edges
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Create curve loop and plane surface
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Physical groups for boundary conditions
// Bottom edge (y=0)
Physical Line("bottom") = {1};
// Right edge
Physical Line("right") = {2};
// Top edge (y=1)
Physical Line("top") = {3};
// Left edge
Physical Line("left") = {4};

// Physical volume for material
Physical Surface("material") = {1};

// Mesh size control
Mesh.CharacteristicLengthMin = 0.0390625;
Mesh.CharacteristicLengthMax = 0.04;

// Generate mesh
Mesh 2;
