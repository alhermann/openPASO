
SetFactory("OpenCASCADE");
// Unit square extruded to thickness 0.125
lc = 1.0/16;
Point(1) = {0, 0, 0, lc};
Point(2) = {1, 0, 0, lc};
Point(3) = {1, 1, 0, lc};
Point(4) = {0, 1, 0, lc};
Point(5) = {0, 0, 0.125, lc};
Point(6) = {1, 0, 0.125, lc};
Point(7) = {1, 1, 0.125, lc};
Point(8) = {0, 1, 0.125, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {1, 5};
Line(6) = {2, 6};
Line(7) = {3, 7};
Line(8) = {4, 8};
Line(9) = {5, 6};
Line(10) = {6, 7};
Line(11) = {7, 8};
Line(12) = {8, 5};

Line Loop(1) = {1, 2, 3, 4};
Line Loop(2) = {5, 9, -1, 12};
Line Loop(3) = {6, 10, -2, -9};
Line Loop(4) = {7, 11, -3, -10};
Line Loop(5) = {8, 12, -4, -11};

Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};
Plane Surface(5) = {5};

Volume(1) = {1, 2, 3, 4, 5};

// Physical groups for boundary conditions
Physical Line("bottom", 101) = {1};   // y=0
Physical Line("top", 102) = {3};       // y=1
Physical Line("left", 103) = {4};      // x=0
Physical Line("right", 104) = {2};     // x=1
Physical Line("front_bottom", 105) = {9};   // z=thickness, y=0
Physical Line("front_top", 106) = {11};     // z=thickness, y=1
Physical Line("front_left", 107) = {12};    // z=thickness, x=0
Physical Line("front_right", 108) = {10};   // z=thickness, x=1
Physical Line("back_bottom", 109) = {5};    // z=0, y=0
Physical Line("back_top", 110) = {8};       // z=0, y=1
Physical Line("back_left", 111) = {4};      // z=0, x=0
Physical Line("back_right", 112) = {6};     // z=0, x=1

Physical Volume("solid", 201) = {1};

Mesh 3;
WriteMesh "mesh_level2.exo";
