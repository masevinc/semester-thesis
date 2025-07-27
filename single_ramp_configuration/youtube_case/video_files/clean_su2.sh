#!/bin/bash

# Script to clean up SU2 output/analysis files

echo "Cleaning SU2 output files..."

# List of common SU2 file types to delete
rm -f *.dat *.csv *.vtk *vtu *.history *.restart *.surface_flow

# Optional: delete files in specific output directory (uncomment if needed)
# rm -rf output/*

echo "Cleanup complete!"
