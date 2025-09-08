#!/usr/bin/env python3
"""
NPZ post-processor for structured fields.

Features:
- Load a .npz, list available keys
- Extract a chosen field (2D) and optional x/y coordinate arrays if present
- Save CSV (x,y,value) for grid or index-based if no coords
- Quick plots: image and histogram
- Batch mode: process a folder recursively or a single file

Examples:
  # With CLI args
  python postprocess/npz_postprocess.py \
    --npz double_ramp_configuration/inputs/double_ramp_npz_files_clamped/your_file.npz \
    --field temperature \
    --out postprocess_outputs/npz_temperature

  # Rely on defaults (edit constants below) and run without args
  python postprocess/npz_postprocess.py
"""

from __future__ import annotations

import argparse
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# -------------------------
# Defaults (edit for convenience)
# -------------------------
DEFAULT_NPZ = "double_ramp_configuration/inputs/double_ramp_npz_files_clamped/double_ramp_0.011_0.0488_ma_2.892_pres_199070_interpolated_arrays.npz"  # file or folder
DEFAULT_FIELD = "density"  # e.g., temperature, pressure, density
DEFAULT_OUT = "postprocess_outputs/npz_auto"
DEFAULT_LIST_KEYS_ONLY = False
DEFAULT_RECURSIVE = True
DEFAULT_DIRECTION = None  # e.g., 'x', 'y', 'z', or 'mag' for magnitude (when vector)

# -------------------------
# Helpers
# -------------------------

def list_npz_keys(path: str) -> List[str]:
    with np.load(path, allow_pickle=True) as data:
        return list(data.keys())


def pick_coords(data: Dict[str, np.ndarray], nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    # Prefer x/y, else x_coords/y_coords; fallback to uniform grid
    x = data.get('x', data.get('x_coords', None))
    y = data.get('y', data.get('y_coords', None))
    if x is None or y is None:
        x = np.linspace(0.5 / nx, 1 - 0.5 / nx, nx)
        y = np.linspace(0.5 / ny, 1 - 0.5 / ny, ny)
    else:
        x = np.asarray(x).squeeze()
        y = np.asarray(y).squeeze()
        if x.ndim > 1:
            x = x.ravel()
        if y.ndim > 1:
            y = y.ravel()
        if x.size != nx:
            x = np.linspace(0.5 / nx, 1 - 0.5 / nx, nx)
        if y.size != ny:
            y = np.linspace(0.5 / ny, 1 - 0.5 / ny, ny)
    return x, y


def save_csv_grid(outdir: str, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, name: str):
    os.makedirs(outdir, exist_ok=True)
    arr = np.column_stack([X.ravel(), Y.ravel(), Z.astype(float).ravel()])
    np.savetxt(os.path.join(outdir, f"{name}.csv"), arr, delimiter=",", header="x,y,value", comments="")


def quick_plots(outdir: str, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, title: str, name: str):
    os.makedirs(outdir, exist_ok=True)
    # image plot
    plt.figure(figsize=(6, 5))
    extent = [float(X.min()), float(X.max()), float(Y.min()), float(Y.max())]
    im = plt.imshow(Z, origin='lower', aspect='equal', extent=extent)
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name}_field.png"), dpi=220)
    plt.close()
    # histogram
    plt.figure(figsize=(5, 4))
    plt.hist(Z[np.isfinite(Z)].ravel(), bins=60, color="#337ab7")
    plt.title(f"Histogram: {title}")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name}_hist.png"), dpi=180)
    plt.close()


def _sanitize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def candidate_keys_npz(keys: List[str], base: str, direction: Optional[str]) -> List[str]:
    """Order NPZ keys by preference based on base field and optional direction.

    Examples mapping when direction is provided:
    - x: tokens ['x','u']
    - y: tokens ['y','v']
    - z: tokens ['z','w']
    - mag: tokens ['mag','magnitude','norm','abs']
    """
    keys = list(keys)
    pref_s = _sanitize_key(base)
    exact = []
    sanitized = []
    token_like = []
    others = []
    dir_tokens = []
    if direction:
        d = direction.lower()
        if d == 'x':
            dir_tokens = ['x', 'u']
        elif d == 'y':
            dir_tokens = ['y', 'v']
        elif d == 'z':
            dir_tokens = ['z', 'w']
        elif d in ('mag', 'magnitude', 'norm'):
            dir_tokens = ['mag', 'magnitude', 'norm', 'abs']
        else:
            dir_tokens = [d]
    for k in keys:
        if k.lower() == base.lower():
            exact.append(k)
        elif _sanitize_key(k) == pref_s:
            sanitized.append(k)
        elif dir_tokens and any(t in k.lower() for t in dir_tokens):
            token_like.append(k)
        else:
            others.append(k)
    out: List[str] = []
    seen = set()
    for group in (exact, sanitized, token_like, others):
        for k in group:
            if k not in seen:
                seen.add(k)
                out.append(k)
    return out


def process_single_npz(npz_path: str, field: str, outdir: str, list_only: bool = False, direction: Optional[str] = None) -> None:
    os.makedirs(outdir, exist_ok=True)
    with np.load(npz_path, allow_pickle=True) as data:
        keys = list(data.keys())
        with open(os.path.join(outdir, "available_keys.txt"), "w") as f:
            for k in keys:
                f.write(f"- {k}\n")
        if list_only:
            return
        chosen_key = None
        # Prefer exact match; else try candidates using direction hints
        if field in data:
            chosen_key = field
        else:
            for k in candidate_keys_npz(keys, field, direction):
                if k in data:
                    chosen_key = k
                    break
        if chosen_key is None:
            raise KeyError(f"Field '{field}' not found in NPZ. Keys: {keys}")

        A = np.asarray(data[chosen_key]).astype(float)
        # Handle vector arrays stored as (ny, nx, ncomp)
        if A.ndim == 3:
            ncomp = A.shape[2]
            if direction is None:
                raise ValueError(f"Field '{chosen_key}' is vector with {ncomp} components; specify direction (x,y,z,mag)")
            d = direction.lower()
            if d in ('mag', 'magnitude', 'norm'):
                Z = np.linalg.norm(A, axis=2)
            else:
                comp_map = {'x': 0, 'y': 1, 'z': 2, 'u': 0, 'v': 1, 'w': 2}
                idx = comp_map.get(d)
                if idx is None or idx >= ncomp:
                    raise ValueError(f"Invalid direction '{direction}' for vector with {ncomp} components")
                Z = A[:, :, idx]
        elif A.ndim == 2:
            Z = A
        else:
            raise ValueError(f"Field '{chosen_key}' must be 2D or 3D (ny,nx[,ncomp]); got shape {A.shape}")
        ny, nx = Z.shape
        x, y = pick_coords(data, nx, ny)
        # ensure monotonic ascending
        idx_x = np.argsort(x)
        idx_y = np.argsort(y)
        if not np.all(idx_x == np.arange(x.size)):
            Z = Z[:, idx_x]
            x = x[idx_x]
        if not np.all(idx_y == np.arange(y.size)):
            Z = Z[idx_y, :]
            y = y[idx_y]
        X, Y = np.meshgrid(x, y)
        # Name output by chosen key and direction (if used)
        name = chosen_key if direction is None else f"{chosen_key}_{direction}"
        save_csv_grid(outdir, X, Y, Z, name)
        quick_plots(outdir, X, Y, Z, title=field, name=name)
        with open(os.path.join(outdir, "stats.txt"), "a") as f:
            f.write(f"File: {npz_path}\n")
            f.write(f"Field: {chosen_key}{(':'+direction) if direction else ''}\n")
            f.write(f"min={float(np.nanmin(Z)):.6g} max={float(np.nanmax(Z)):.6g} std={float(np.nanstd(Z)):.6g}\n\n")
        print(f"Done. Wrote CSV and plots to: {outdir}")


def main():
    parser = argparse.ArgumentParser(description="Post-process NPZ grid fields and plot")
    parser.add_argument("--npz", required=False, default=None, help="Path to .npz file or folder to scan")
    parser.add_argument("--field", required=False, default=None, help="Field name in NPZ (2D array)")
    parser.add_argument("--out", required=False, default=None, help="Output directory")
    parser.add_argument("--direction", required=False, default=None, help="Direction for vector fields: x,y,z, or mag")
    parser.add_argument("--list-keys", action="store_true", help="Only list keys and exit")
    parser.add_argument("--no-recursive", action="store_true", help="When folder is given, do not recurse")
    args = parser.parse_args()

    npz_input = args.npz or DEFAULT_NPZ
    field = args.field or DEFAULT_FIELD
    out = args.out or DEFAULT_OUT
    direction = args.direction or DEFAULT_DIRECTION
    list_only = args.list_keys or DEFAULT_LIST_KEYS_ONLY
    recursive = DEFAULT_RECURSIVE and not args.no_recursive

    if os.path.isdir(npz_input):
        root = os.path.abspath(npz_input)
        count = 0
        for dirpath, dirnames, filenames in os.walk(root):
            if not recursive and dirpath != root:
                continue
            npzs = [f for f in filenames if f.lower().endswith('.npz')]
            for fname in npzs:
                fpath = os.path.join(dirpath, fname)
                rel_dir = os.path.relpath(dirpath, root)
                outdir = os.path.join(out, rel_dir)
                try:
                    process_single_npz(fpath, field, outdir, list_only=list_only, direction=direction)
                    count += 1
                except Exception as e:
                    print(f"[WARN] Skipped {fpath}: {e}")
        print(f"Processed {count} NPZ files from folder: {npz_input}")
    else:
        if not os.path.isfile(npz_input):
            raise FileNotFoundError(f"NPZ not found: {npz_input}")
        process_single_npz(npz_input, field, out, direction=direction)


if __name__ == "__main__":
    main()
