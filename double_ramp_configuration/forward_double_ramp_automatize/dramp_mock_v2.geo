// Gmsh project created on Mon Apr 28 22:06:58 2025
//+
Point(1) = {0, 1, 0, 1.0};
//+
Point(2) = {0.2, 1, 0, 1.0};
//+
Point(3) = {0.4, 0.9, 0, 1.0};
//+
Point(4) = {0.55, 0.9, 0, 1.0};
//+
Point(5) = {0.7, 0.75, 0, 1.0};
//+
Point(6) = {1, 0.75, 0, 1.0};
//+
Point(7) = {1, 0, 0, 1.0};
//+
Point(8) = {0, 0, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
//+
Line(5) = {5, 6};
//+
Line(6) = {6, 7};
//+
Line(7) = {7, 8};
//+
Line(8) = {8, 1};
//+
Curve Loop(1) = {8, 1, 2, 3, 4, 5, 6, 7};
//+
Plane Surface(1) = {1};
//+
Physical Curve("Inlet", 9) = {8};
//+
Physical Curve("Outlet", 10) = {6};
//+
Physical Curve("Wall", 11) = {1, 2, 3, 4, 5, 7};
//+
Transfinite Curve {8} = 201 Using Progression 1;
//+
Transfinite Curve {6} = 201 Using Progression 1;
//+
Transfinite Curve {1} = 31 Using Progression 1;
//+
Transfinite Curve {2} = 31 Using Progression 1;
//+
Transfinite Curve {3} = 21 Using Progression 1;
//+
Transfinite Curve {4} = 31 Using Progression 1;
//+
Transfinite Curve {5} = 41 Using Progression 1;
//+
Transfinite Curve {7} = 151 Using Progression 1;
//+
Recombine Surface {1};
