"""Small helpers for reading VTU via meshio and extracting fields consistently."""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

import numpy as np
import meshio
from matplotlib.tri import Triangulation


def _sanitize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def candidate_keys(keys: Sequence[str], preferred: str, extra_tokens: Optional[Sequence[str]] = None) -> List[str]:
    if not keys:
        return []
    keys = list(keys)
    exact: List[str] = []
    sanitized: List[str] = []
    token_like: List[str] = []
    others: List[str] = []
    pref_s = _sanitize_key(preferred)
    tokens = [t.lower() for t in (extra_tokens or [])]
    for k in keys:
        if k.lower() == preferred.lower():
            exact.append(k)
        elif _sanitize_key(k) == pref_s:
            sanitized.append(k)
        elif any(t in k.lower() for t in tokens):
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


def is_nonconstant(a: np.ndarray, eps: float = 1e-12) -> bool:
    if a.size == 0:
        return False
    return float(np.nanstd(a)) > eps


def triangulation_from_mesh(m: meshio.Mesh) -> Optional[Triangulation]:
    tris: List[Tuple[int, int, int]] = []
    for block in m.cells:
        ctype = block.type
        conn = block.data
        if ctype == "triangle":
            tris.extend([tuple(t) for t in conn])
        elif ctype == "quad":
            for q in conn:
                i0, i1, i2, i3 = q
                tris.append((i0, i1, i2))
                tris.append((i0, i2, i3))
    if not tris:
        return None
    pts = m.points[:, :2]
    tri = Triangulation(pts[:, 0], pts[:, 1], np.asarray(tris))
    return tri


def extract_field_meshio_safe(m: meshio.Mesh, field_preference: str):
    """Extract a scalar field from meshio mesh, preferring point_data then cell_data.

    Only considers keys that match the preferred name or synonyms (rho, density), avoiding
    unrelated fields like Velocity that can be vector-valued or mismatched.

    Returns: (pts_xy, values, location, chosen_key)
    """
    synonyms = ["rho", "density"]

    def _matches(k: str) -> bool:
        kl = k.lower()
        if kl == field_preference.lower():
            return True
        if _sanitize_key(k) == _sanitize_key(field_preference):
            return True
        return any(t in kl for t in synonyms)

    # Point data first (prefer this as we can triangulate easily)
    pd_keys = [k for k in m.point_data.keys() if _matches(k)]
    for key in pd_keys:
        arr = np.asarray(m.point_data[key])
        # Skip vector/tensor fields unless they collapsed to 1D
        if arr.ndim > 1 and arr.shape[-1] != 1:
            continue
        vals = arr.astype(float).ravel()
        if vals.size != len(m.points):
            # Some odd files may have mismatched lengths; reject
            continue
        if is_nonconstant(vals):
            pts = m.points[:, :2]
            return pts, vals, "point", key

    # Then cell data
    cd_keys = [k for k in m.cell_data.keys() if _matches(k)]
    for key in cd_keys:
        per_block_arrays = m.cell_data[key]
        if not per_block_arrays:
            continue
        values_list: List[np.ndarray] = []
        centers_list: List[np.ndarray] = []
        for cells_block, cell_vals in zip(m.cells, per_block_arrays):
            cell_vals = np.asarray(cell_vals)
            if cell_vals.ndim > 1 and cell_vals.shape[-1] != 1:
                # vector/tensor cell field -> skip
                continue
            cell_vals = cell_vals.astype(float).ravel()
            if cell_vals.size == 0:
                continue
            for conn in cells_block.data:
                xy = m.points[conn, :2]
                centers_list.append(xy.mean(axis=0))
            values_list.append(cell_vals)
        if values_list:
            centers = np.asarray(centers_list)
            values = np.concatenate(values_list)
            if centers.shape[0] == values.shape[0] and is_nonconstant(values):
                return centers, values, "cell", key

    # Not found with strict matching; provide diagnostics
    raise KeyError(
        f"Field '{field_preference}' (synonyms: {synonyms}) not found as scalar in point_data or cell_data.\n"
        f"Available point keys: {list(m.point_data.keys())}\n"
        f"Available cell keys: {list(m.cell_data.keys())}"
    )
