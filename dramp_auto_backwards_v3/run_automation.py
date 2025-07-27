# run_automation.py

import os
import csv
import numpy as np
from mesh_generator import generate_mesh_from_points
from sweep_calculations_bw import generate_sweeps_for_mesh_folder

POINTS_DIR = './dramp_auto_backwards_v3/dummy_results' 
MESH_DIR = './dramp_auto_backwards_v3/outputs/results'
ERROR_LOG = './dramp_auto_backwards_v3/outputs/mesh_errors.csv'

os.makedirs(MESH_DIR, exist_ok=True)

# Open error log file and write header
with open(ERROR_LOG, mode='w', newline='') as errfile:
    writer = csv.writer(errfile)
    writer.writerow(['filename', 'error'])

    for fname in os.listdir(POINTS_DIR):
        if not fname.endswith('.npy'):
            continue

        full_path = os.path.join(POINTS_DIR, fname)
        points = np.load(full_path)

        print(f"Processing file: {fname}")
        if len(points) != 12:
            msg = f"expected 12 points, got {len(points)}"
            # print(f"  Skipped: {msg}")
            writer.writerow([fname, msg])
            # continue

        base_name = os.path.splitext(fname)[0]
        mesh_out = os.path.join(MESH_DIR, f"{base_name}.msh")

        try:
            generate_mesh_from_points(points, mesh_out, show_gui=False)
            print(f"  Mesh saved to {mesh_out}")
        except Exception as e:
            msg = str(e)
            print(f"  Failed to generate mesh: {msg}")
            writer.writerow([fname, msg])
            
#generate_sweeps_for_mesh_folder(MESH_DIR)