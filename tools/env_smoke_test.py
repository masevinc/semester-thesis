"""env_smoke_test.py
Quick sanity checks for the double ramp pipeline environment.
Run:  python tools/env_smoke_test.py
"""
from __future__ import annotations
import os
import importlib
import numpy as np

REPORT = []

def _log(msg):
    REPORT.append(msg)
    print(msg)

def check_imports():
    modules = ["numpy", "matplotlib", "cv2"]
    for m in modules:
        try:
            mod = importlib.import_module(m)
            ver = getattr(mod, "__version__", "?")
            _log(f"[imports] {m} OK (version {ver})")
        except Exception as e:
            _log(f"[imports][FAIL] {m}: {e}")

def check_random_plot(out_dir="_smoke_outputs"):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    arr = np.random.rand(32, 48)
    fig, ax = plt.subplots(figsize=(4,3), dpi=120)
    im = ax.imshow(arr, cmap="viridis", origin="lower")
    try:
        fig.colorbar(im, ax=ax, shrink=0.8)
    except Exception as e:
        _log(f"[plot][warn] colorbar failed: {e}")
    try:
        fig.tight_layout()
    except Exception as e:
        _log(f"[plot][warn] tight_layout failed: {e}")
    png = os.path.join(out_dir, "smoke_plot.png")
    try:
        fig.savefig(png, dpi=120)
        _log(f"[plot] saved {png}")
    except Exception as e:
        _log(f"[plot][FAIL] savefig: {e}")
    finally:
        plt.close(fig)


def main():
    _log("== Smoke Test Start ==")
    check_imports()
    check_random_plot()
    _log("== Smoke Test End ==")

if __name__ == "__main__":
    main()
