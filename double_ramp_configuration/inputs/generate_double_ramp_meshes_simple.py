#!/usr/bin/env python3
"""
Simplified DOUBLE RAMP mesh generator (edit variables below).

Edit ONLY the CONFIGURATION SECTION. No command line arguments needed.

Outputs:
    <OUTPUT_DIR>/geo_files/*.geo
    <OUTPUT_DIR>/cgns_meshes/*.cgns

Requires Gmsh in PATH or set GMSH_CMD to full path.
"""
from __future__ import annotations
import random
import subprocess
from pathlib import Path
from string import Template

# ==========================
# CONFIGURATION SECTION
# ==========================
N_SAMPLES      = 10        # How many random (y1,y2) pairs
Y1_MIN         = 0.01       # Lower bound for y_ramp_1
Y1_MAX         = 0.05       # Upper bound for y_ramp_1
DELTA          = 0.04       # y2 sampled in [max(0, y1-DELTA), y1+DELTA]
PRECISION      = 4          # Decimal places for y1,y2 naming
SEED           = 42         # Set to None for non-deterministic
OVERWRITE      = False      # If False skip meshes that already exist
OUTPUT_DIR     = "double_ramps_200"  # Root output directory
GMSH_CMD       = "gmsh"     # Or e.g. "/opt/local/bin/gmsh"
DIMENSION      = 2          # 2 (surface) or 3 (if you adapt to volume)
# ==========================

GEO_TEMPLATE = Template("""SetFactory("OpenCASCADE");
Point(1) = {0.08, 0, 0, 1.0};
Point(2) = {0.1, 0, 0, 1.0};
Point(3) = {0.2, $y1, 0, 1.0};
Point(4) = {0.3, $y1, 0, 1.0};
Point(5) = {0.4, $y2, 0, 1.0};
Point(6) = {0.7, $y2, 0, 1.0};
Point(7) = {0.7, 0.4, 0, 1.0};
Point(8) = {0.08, 0.4, 0, 1.0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 1};
Curve Loop(1) = {7, 8, 1, 2, 3, 4, 5, 6};
Plane Surface(1) = {1};
Physical Curve("Inlet", 9) = {8};
Physical Curve("Outlet", 10) = {6};
Physical Curve("Wall", 11) = {7, 1, 2, 3, 4, 5};
Physical Surface("Fluid", 12) = {1};
Transfinite Surface {1} = {8, 7, 6, 1};
Transfinite Curve {8, 6} = 201 Using Progression 1;
Transfinite Curve {2, 3, 4} = 51 Using Progression 1;
Transfinite Curve {1} = 11 Using Progression 1;
Transfinite Curve {5} = 151 Using Progression 1;
Transfinite Curve {7} = 311 Using Progression 1;
Recombine Surface {1};
""")

def sample_pair(rng: random.Random):
    y1 = round(rng.uniform(Y1_MIN, Y1_MAX), PRECISION)
    y2_low = max(0.0, y1 - DELTA)
    y2_high = y1 + DELTA
    y2 = round(rng.uniform(y2_low, y2_high), PRECISION)
    return y1, y2

def write_geo(path: Path, y1: float, y2: float):
    # Substitute only $y1 and $y2; other braces are literal for gmsh syntax.
    path.write_text(GEO_TEMPLATE.substitute(y1=y1, y2=y2))

def run_gmsh(geo: Path, cgns: Path):
    # Use Gmsh native CGNS writer
    cmd = [GMSH_CMD, str(geo), f"-{DIMENSION}", "-format", "cgns", "-o", str(cgns)]
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def main():
    if Y1_MIN >= Y1_MAX:
        raise SystemExit("Y1_MIN must be < Y1_MAX")
    if DELTA <= 0:
        raise SystemExit("DELTA must be positive")

    rng = random.Random(SEED)

    root = Path(OUTPUT_DIR)
    geo_dir = root / "geo_files"
    cgns_dir = root / "cgns_meshes"
    geo_dir.mkdir(parents=True, exist_ok=True)
    cgns_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0
    errors = 0
    error_examples = []

    for i in range(N_SAMPLES):
        y1, y2 = sample_pair(rng)
        geo_name = f"double_ramp_{y1}_{y2}.geo"
        cgns_name = f"double_ramp_{y1}_{y2}.cgns"
        geo_path = geo_dir / geo_name
        cgns_path = cgns_dir / cgns_name

        if cgns_path.exists() and not OVERWRITE:
            skipped += 1
            continue

        write_geo(geo_path, y1, y2)
        proc = run_gmsh(geo_path, cgns_path)
        if proc.returncode != 0 or not cgns_path.exists():
            errors += 1
            if len(error_examples) < 5:
                error_examples.append((y1, y2, proc.stderr.strip()))
        else:
            generated += 1

    print("Double ramp mesh generation summary:")
    print(f"  Requested samples: {N_SAMPLES}")
    print(f"  Generated: {generated}")
    print(f"  Skipped existing: {skipped}")
    print(f"  Errors: {errors}")
    if error_examples:
        print("  Example errors:")
        for y1, y2, msg in error_examples:
            print(f"    y1={y1} y2={y2} -> {msg[:120]}")
    print(f"Output root: {root.resolve()}")
    print("Done.")

if __name__ == "__main__":
    main()
