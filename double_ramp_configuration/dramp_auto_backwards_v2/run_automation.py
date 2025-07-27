# scripts/run_automation.py

import os
import numpy as np

POINTS_DIR = './dramp_auto_backwards_v2/dummy_results'

for fname in os.listdir(POINTS_DIR):
    if not fname.endswith('.npy'):
        continue

    full_path = os.path.join(POINTS_DIR, fname)
    points = np.load(full_path)

    print(f"\nProcessing file: {fname}")
    print(f"Loaded {len(points)} points")

    # Example use:
    # - generate .geo file
    # - mesh generation
    # - CFD setup
    # Placeholder:
    for i, (x, y) in enumerate(points):
        print(f"  Point {i + 1}: ({x:.5f}, {y:.5f})")
        
    '''
    For the extracted points:
    "ramp1": 0.01,
    "ramp2": 0.0304,
    "min_ma": None,
    "max_ma": None
    
    points: 
    Loaded 8 points
    Point 1: (0.00000, 1.00000)
    Point 2: (0.34932, 0.87696)
    Point 3: (0.45360, 0.83316)
    Point 4: (0.57873, 0.83316)
    Point 5: (0.70594, 0.87278)
    Point 6: (1.06569, 0.87278)
    Point 7: (1.33368, 0.00000)
    Point 8: (0.00000, 0.00000)
    '''
    # TODO: Point 6 also exceeds the domain - It should limited to 1 for x - Check the scaling, might have a problem!
    # TODO: Point 7 is not we expected - It should be 1,0
    
    '''
    For the extracted points:
    "ramp1": 0.046,
    "ramp2": 0.0112,
    "min_ma": None,
    "max_ma": None
    
    points: 
    Loaded 8 points
    Point 1: (0.00000, 1.00000)
    Point 2: (0.33994, 0.87696)
    Point 3: (0.45464, 0.79353)
    Point 4: (0.56934, 0.79353)
    Point 5: (0.70177, 0.86236)
    Point 6: (1.06569, 0.86236)
    Point 7: (1.33368, 0.00000)
    Point 8: (0.00000, 0.00000)
    
    Comparison with points created from V1 Code:
    [(0.0, 1.0), 
    (0.037037, 1.0), 
    (0.200542, 0.886179), 
    (0.493225, 0.95664), 
    (0.526649, 0.974706), 
    (0.588076, 0.95122), 
    (1.0, 1.0), 
    (1.0, 0.971093), 
    (1.0, 0.0), 
    (0.0, 0.0)]
    
    V2 creates outer of the boundaries
    V1 creates 10 points in that case - binary mapping is shit, and algorithm as well
    
    '''
    
    
    # TODO: integrate with meshing/simulation tools here
