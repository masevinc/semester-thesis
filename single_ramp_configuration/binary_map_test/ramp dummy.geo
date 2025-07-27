/*
Point(1) = {0, 0, 0, 1.0};
Point(2) = {0.5, 0, 0, 1.0};
Point(3) = {1.5, 0.2309, 0, 1.0};
Point(4) = {1.5, 0.9342, 0, 1.0};
Point(5) = {0, 0.9342, 0, 1.0};
*/

/*
Point(1) = {0.000000, 0.000000, 0, 1.0};
Point(2) = {0.002906, 0.934200, 0, 1.0};
Point(3) = {0.499790, 0.934200, 0, 1.0};
Point(4) = {1.500822, 0.755496, 0, 1.0};
Point(5) = {1.497916, -0.002906, 0, 1.0};
*/


Point(1) = {0.000000, 0.000000, 0, 1.0};
Point(2) = {0.002906, -0.934200, 0, 1.0};
Point(3) = {0.499790, -0.934200, 0, 1.0};
Point(4) = {1.500822, -0.755496, 0, 1.0};
Point(5) = {1.497916, 0.002906, 0, 1.0};




/*
Line(1) = {1, 5};
Line(2) = {5, 4};
Line(3) = {4, 3};
Line(4) = {3, 2};
Line(5) = {2, 1};

Curve Loop(1) = {2, 3, 4, 5, 1};

Plane Surface(1) = {1};

Physical Curve("Inlet", 6) = {1};
Physical Curve("Outlet", 7) = {3};
Physical Curve("Wall", 8) = {2, 5, 4};

Transfinite Surface {1} = {5, 4, 3, 1};

Transfinite Curve {1, 3} = 201 Using Progression 1;
Transfinite Curve {5} = 51 Using Progression 1;
Transfinite Curve {4} = 101 Using Progression 1;
Transfinite Curve {2} = 151 Using Progression 1;

Recombine Surface {1};
*///+
Line(1) = {2, 3};
//+
Line(2) = {3, 4};
//+
Line(3) = {4, 5};
//+
Line(4) = {5, 1};
//+
Line(5) = {1, 2};
//+
Curve Loop(1) = {5, 1, 2, 3, 4};
//+
Plane Surface(1) = {1};
