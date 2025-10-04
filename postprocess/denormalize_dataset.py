#!/usr/bin/env python3
"""Denormalize previously min-max scaled NPZ datasets.

This inverts the custom linear transform used during normalization:

    y = a * x + b
    where
        a = (Rmax - Rmin) / (Xmax - Xmin)
        b = Rmax - a * Xmax   (equivalently b = Rmin - a * Xmin)

Provided we know (Xmin, Xmax) for each physical variable and the range
used originally (Rmin, Rmax), the inverse is any of the equivalent forms:

    x = (y - b)/a
    x = Xmax + (Xmax - Xmin) * (y - Rmax) / (Rmax - Rmin)
    x = Xmin + (Xmax - Xmin) * (y - Rmin) / (Rmax - Rmin)   (recommended)

This script applies the recommended last formula.

Usage pattern (no CLI needed now):
    1. Edit the CONFIG dict near the top of this file.
    2. Run: python postprocess/denormalize_dataset.py
    3. Outputs appear under CONFIG['OUT_DIR'].

Notes / Assumptions:
  * You MUST set --range-min / --range-max to the same values used during normalization.
    If unsure, inspect the training / preprocessing code (attributes range_min, range_max).
    Common choices are [0,1] or [-1,1]. Defaults here assume [0,1].
  * The script only denormalizes keys present in MIN_DICT / MAX_DICT below.
    Additional keys can be added easily.
  * Any array (2D, 3D, n-D) is processed elementwise; shape is preserved.
  * Non-matching keys are copied across unchanged unless --drop-others is set.
  * Output filename mirrors input name. Directory structure is mirrored when input is a folder.

Author: Automated helper
"""

from __future__ import annotations

import os
from typing import Dict, Tuple, Iterable
import numpy as np

# ---------------------------------------------------------------------------
# Min/Max dictionaries taken from the original normalization code snippet.
# Extend these if more physical variables were normalized.
# ---------------------------------------------------------------------------
MIN_DICT: Dict[str, float] = {
    'data': 0.0,
    'pressure': 0.0,
    'mach': 0.0,
    'temperature': 0.0,
    'velocity_x': -1249.970947265625,
    'velocity_y': -3758.77294921875,
}

MAX_DICT: Dict[str, float] = {
    'data': 8.634648323059082,
    'pressure': 2649507.0,
    'mach': 8.678845405578613,
    'temperature': 1624.8419189453125,
    'velocity_x': 2777.510009765625,
    'velocity_y': 824.901611328125,
}

ALL_KEYS = sorted(MIN_DICT.keys())

# ---------------------------------------------------------------------------
# USER CONFIG (edit these values instead of passing CLI arguments)
# ---------------------------------------------------------------------------
CONFIG = dict(
    INPUT_PATH="double_ramp_configuration/inputs/DDPM_fully",   # file OR directory (e.g. DDPM_fully or DDPM_semi)
    OUT_DIR="double_ramp_configuration/inputs/denorm/DDPM_fully", # destination folder for final files
    RANGE_MIN=0.0,                                     # original normalization range min (e.g. -1.0)
    RANGE_MAX=1.0,                                     # original normalization range max (e.g. 1.0)
    RECURSIVE=True,                                    # recurse into subfolders if INPUT_PATH is a directory
    DROP_OTHERS=False,                                 # drop keys not in MIN/MAX dicts
    OVERWRITE=False,                                   # overwrite existing output files
    VERBOSE=True,                                      # print per-file status
    RENAME_KEYS={'data': 'density'},                   # optional mapping: {'data':'density', ...}
    # ---------------- Filename pattern controls ----------------
    ENABLE_FILENAME_PATTERN=True,                      # build new filename pattern below
    LENGTH_FROM_INDEX=True,                            # derive 0.x length from trailing integer token (e.g. fully_4 -> 0.4)
    LENGTH_DIVISOR=10.0,                               # index / divisor => length (4 / 10 -> 0.4)
    FIXED_R1=None,                                     # override derived r1 (float or None)
    FIXED_R2=None,                                     # override derived r2 (float or None)
    DECIMALS=4,                                        # decimals in filename (use 1 for 0.4 style)
    FORCE_DECIMALS_RL=1,                               # show at least this many decimals for ramp lengths
    # Pattern specifics for Mach / Pressure tokens
    REQUIRE_MA=True,                                   # if True and mach not found, warn
    REQUIRE_PRES=True,                                 # if True and pressure not found, warn
    MACH_DECIMALS=4,                                   # limit Mach to 4 decimals
    STRIP_TRAILING_ZEROES_MACH=True,                   # remove trailing zeros from Mach
)


def inverse_minmax(y: np.ndarray, xmin: float, xmax: float, rmin: float, rmax: float) -> np.ndarray:
    """Inverse of custom min-max scaling.

    Works for any array shape, operates elementwise.
    Formula: x = Xmin + (Xmax - Xmin) * (y - Rmin) / (Rmax - Rmin)
    Handles NaNs transparently.
    """
    scale = (xmax - xmin) / (rmax - rmin)
    return xmin + scale * (y - rmin)


def denormalize_arrays(data: Dict[str, np.ndarray], rmin: float, rmax: float, keys: Iterable[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        if k not in data:
            continue
        xmin = MIN_DICT[k]
        xmax = MAX_DICT[k]
        arr = np.asarray(data[k])
        out[k] = inverse_minmax(arr, xmin, xmax, rmin, rmax)
    return out


def collect_npz_files(path: str, recursive: bool) -> Iterable[str]:
    if os.path.isfile(path):
        yield path
        return
    for dirpath, dirnames, filenames in os.walk(path):
        for fname in filenames:
            if fname.lower().endswith('.npz'):
                yield os.path.join(dirpath, fname)
        if not recursive:
            break


def relative_output_path(in_root: str, file_path: str, out_root: str) -> str:
    if os.path.isfile(in_root):
        # Single file mode
        base = os.path.basename(file_path)
        return os.path.join(out_root, base)
    rel = os.path.relpath(file_path, start=in_root)
    return os.path.join(out_root, rel)


def process_file(npz_path: str, out_path: str, rmin: float, rmax: float, drop_others: bool, rename_map: Dict[str,str]) -> Tuple[bool, str]:
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            keys = list(data.keys())
            # Determine which keys to denormalize (intersection with known keys)
            target_keys = [k for k in keys if k in MIN_DICT]
            if not target_keys:
                return False, f"No matching keys to denormalize in {npz_path} (available: {keys})"
            denorm = denormalize_arrays({k: data[k] for k in target_keys}, rmin, rmax, target_keys)
            # Compose output dict
            out_dict: Dict[str, np.ndarray] = {}
            if not drop_others:
                for k in keys:
                    if k not in target_keys:
                        out_dict[k] = data[k]
            # Add denormalized arrays (rename if mapping provided)
            for k, arr in denorm.items():
                new_key = rename_map.get(k, k)
                out_dict[new_key] = arr
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.savez_compressed(out_path, **out_dict)
        return True, f"Denormalized {len(target_keys)} keys -> {out_path}"
    except Exception as e:
        return False, f"ERROR processing {npz_path}: {e}"


def main():
    in_path = CONFIG['INPUT_PATH']
    out_root = CONFIG['OUT_DIR']
    rmin = CONFIG['RANGE_MIN']
    rmax = CONFIG['RANGE_MAX']
    recursive = CONFIG['RECURSIVE']
    drop_others = CONFIG['DROP_OTHERS']
    overwrite = CONFIG['OVERWRITE']
    verbose = CONFIG.get('VERBOSE', True)
    rename_map: Dict[str,str] = CONFIG.get('RENAME_KEYS', {}) or {}
    use_pattern = CONFIG.get('ENABLE_FILENAME_PATTERN', True)
    length_from_index = CONFIG.get('LENGTH_FROM_INDEX', True)
    length_div = float(CONFIG.get('LENGTH_DIVISOR', 10.0) or 10.0)
    fixed_r1 = CONFIG.get('FIXED_R1', None)
    fixed_r2 = CONFIG.get('FIXED_R2', None)

    if rmax == rmin:
        raise ValueError('range-max and range-min must differ')
    if not os.path.exists(in_path):
        raise FileNotFoundError(f'Input path not found: {in_path}')

    processed = 0
    skipped = 0
    errors = 0
    import re
    for f in collect_npz_files(in_path, recursive):
        # Build output directory mirror (but we always force final OUT_DIR as base root to match desired structure)
        rel_out_dir = out_root
        os.makedirs(rel_out_dir, exist_ok=True)
        # Default new filename constructed later
        out_path = None
        if use_pattern:
            in_base = os.path.basename(f)
            base_no_ext = in_base[:-4] if in_base.lower().endswith('.npz') else in_base
            # Extract trailing integer for ramp length mapping
            idx_val = None
            parts = base_no_ext.split('_')
            if parts and parts[-1].isdigit():
                try:
                    idx_val = int(parts[-1])
                except ValueError:
                    idx_val = None
            # Compute ramp lengths
            if fixed_r1 is not None:
                r1 = float(fixed_r1)
            else:
                r1 = (idx_val / length_div) if (length_from_index and idx_val is not None) else 0.4
            if fixed_r2 is not None:
                r2 = float(fixed_r2)
            else:
                r2 = (idx_val / length_div) if (length_from_index and idx_val is not None) else r1
            # Format ramp lengths
            rl_decimals_min = int(CONFIG.get('FORCE_DECIMALS_RL',1))
            rl_fmt = f"{{:.{max(rl_decimals_min,1)}f}}"
            def _fmt_rl(v: float) -> str:
                s = rl_fmt.format(v)
                # compress 0.4000 -> 0.4
                s = s.rstrip('0').rstrip('.') if '.' in s else s
                if rl_decimals_min == 1 and '.' not in s:
                    s += '.0'
                return s
            r1s = _fmt_rl(r1)
            r2s = _fmt_rl(r2)
            # Parse Mach & Pressure tokens
            ma_match = re.search(r'ma_([0-9]+\.?[0-9]*)', base_no_ext, re.IGNORECASE)
            pr_match = re.search(r'pres_([0-9]+\.?[0-9]*)', base_no_ext, re.IGNORECASE)
            mach_val = None
            pres_val = None
            if ma_match:
                try:
                    mach_val = float(ma_match.group(1))
                except ValueError:
                    mach_val = None
            if pr_match:
                try:
                    pres_val = float(pr_match.group(1))
                except ValueError:
                    pres_val = None
            if CONFIG.get('REQUIRE_MA', True) and mach_val is None and verbose:
                print(f"[WARN] Mach token not found in {in_base}")
            if CONFIG.get('REQUIRE_PRES', True) and pres_val is None and verbose:
                print(f"[WARN] Pressure token not found in {in_base}")
            # Format Mach
            mach_str = ''
            if mach_val is not None:
                m_dec = int(CONFIG.get('MACH_DECIMALS',4))
                m_fmt = f"{{:.{m_dec}f}}"
                mach_str = m_fmt.format(mach_val)
                if CONFIG.get('STRIP_TRAILING_ZEROES_MACH', True):
                    mach_str = mach_str.rstrip('0').rstrip('.')
            # Format Pressure (integer, strip .0)
            pres_str = ''
            if pres_val is not None:
                pres_int = int(round(pres_val))
                pres_str = str(pres_int)
            # Assemble final filename always ending _interpolated_arrays.npz
            segments = [f"double_ramp_{r1s}_{r2s}"]
            if mach_str:
                segments.append(f"ma_{mach_str}")
            if pres_str:
                segments.append(f"pres_{pres_str}")
            new_base = '_'.join(segments) + '_interpolated_arrays.npz'
            out_path = os.path.join(rel_out_dir, new_base)
        if not use_pattern:
            # fallback to mirrored relative path
            out_path = relative_output_path(in_path, f, out_root)
            if not out_path.lower().endswith('.npz'):
                out_path += '.npz'
        # Overwrite guard
        if (not overwrite) and os.path.exists(out_path):
            skipped += 1
            if verbose:
                print(f"[SKIP] Exists (set OVERWRITE=True): {out_path}")
            continue
        ok, msg = process_file(f, out_path, rmin, rmax, drop_others, rename_map)
        if ok:
            processed += 1
            if verbose:
                print(f"[OK] {msg}")
        else:
            errors += 1
            if verbose:
                print(f"[ERR] {msg}")

    print('\nSummary:')
    print(f'  Processed: {processed}')
    print(f'  Skipped  : {skipped}')
    print(f'  Errors   : {errors}')
    if errors > 0:
        print('Some files failed; review error messages above.')


if __name__ == '__main__':
    main()
