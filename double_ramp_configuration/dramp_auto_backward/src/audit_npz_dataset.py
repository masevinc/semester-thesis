"""Quick audit script for .npz dataset integrity.

Usage (from project root):
    python -m src.audit_npz_dataset \
        --root ./double_ramp_configuration/inputs/double_ramp_npz_files_clamped \
        --limit 200 \
        --keys temperature density pressure

Reports:
  * Non-numeric or object-dtype arrays
  * Non-2D shapes
  * NaN / inf statistics
  * Value ranges

Exit code 0 even if issues found (designed for manual review)."""
from __future__ import annotations
import argparse
import os
import numpy as np
import textwrap

def analyze_file(path: str, keys_filter: set[str] | None):
    issues = []
    stats = []
    try:
        with np.load(path, allow_pickle=True) as data:
            for k in data.keys():
                if keys_filter and k not in keys_filter:
                    continue
                arr = data[k]
                dtype = getattr(arr, 'dtype', None)
                shape = getattr(arr, 'shape', None)
                if not isinstance(arr, np.ndarray):
                    issues.append(f"{k}: not a numpy ndarray (type={type(arr)})")
                    continue
                if dtype is object:
                    issues.append(f"{k}: object dtype - potential legacy pickled content")
                if arr.ndim != 2:
                    issues.append(f"{k}: expected 2D got shape {shape}")
                finite_mask = np.isfinite(arr)
                finite_count = int(finite_mask.sum())
                total = arr.size
                if finite_count == 0:
                    issues.append(f"{k}: no finite values (all NaN/inf)")
                else:
                    vmin = float(arr[finite_mask].min())
                    vmax = float(arr[finite_mask].max())
                    stats.append(f"{k}: shape={shape} dtype={dtype} finite={finite_count}/{total} min={vmin:.4g} max={vmax:.4g}")
    except Exception as e:
        issues.append(f"<load>: {type(e).__name__}: {e}")
    return issues, stats

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__))
    ap.add_argument('--root', required=True, help='Directory containing .npz files')
    ap.add_argument('--limit', type=int, default=None, help='Limit number of files (random sample if --random)')
    ap.add_argument('--random', action='store_true', help='Randomly choose --limit files instead of first N')
    ap.add_argument('--keys', nargs='*', default=None, help='Subset of keys to inspect (default: all)')
    args = ap.parse_args()

    root = args.root
    keys_filter = set(args.keys) if args.keys else None
    files = [f for f in os.listdir(root) if f.endswith('.npz')]
    if not files:
        print('No .npz files found in', root)
        return
    import random
    files.sort()
    if args.limit is not None:
        if args.random:
            random.seed(0)
            files = random.sample(files, min(args.limit, len(files)))
        else:
            files = files[:args.limit]

    total_files = len(files)
    any_issues = 0
    print(f"Auditing {total_files} files (root={root})")
    for idx, fname in enumerate(files, 1):
        path = os.path.join(root, fname)
        issues, stats = analyze_file(path, keys_filter)
        if issues:
            any_issues += 1
            print(f"[ISSUES] {fname}")
            for line in issues:
                print("  -", line)
        else:
            # print concise stats line (first key stats only to keep output short)
            if stats:
                print(f"[OK] {fname} :: {stats[0]}")
            else:
                print(f"[OK] {fname} :: (no stats computed)")
    print(f"Finished. Files with issues: {any_issues}/{total_files}")

if __name__ == '__main__':
    main()
