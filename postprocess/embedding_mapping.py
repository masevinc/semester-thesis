#!/usr/bin/env python3
"""
Embedding / mapping post-process for ramp configurations and flow fields.

Features (no CLI; edit CONFIG below):
 1. Collect ramp configuration descriptors from filenames (e.g., double_ramp_0.0111_0.0102...).
 2. Collect NPZ field arrays (density, temperature, mach, pressure) and build embeddings.
 3. UMAP (preferred) or PCA fallback if umap-learn not installed.
 4. Generates scatter plots grouping original vs generated ramps and different model sources.

CONFIG keys:
  ENABLED_SECTIONS: list of section names to run: ['ramps','fields']
  RAMP_SOURCES: list of dict(name, path_glob, label) to collect ramp configs (filenames only)
  FIELD_SOURCES: list of dict(field, path_glob, label) for NPZ arrays
  EMBEDDING:
     method: 'umap' | 'pca'
     n_components: 2
     random_state: 42
     n_neighbors, min_dist (UMAP only)
     sample_limit_per_group: optional cap on number per label
  OUTPUT_DIR: destination folder
  NORMALIZE_FIELD: how to scale arrays before flattening: 'none' | 'minmax' | 'std' | 'max'
  DOWNSAMPLE: integer factor to subsample 2D fields (stride)

Assumptions:
 - NPZ files contain keys: density / temperature / mach / pressure (case-insensitive variants allowed).
 - Ramp configuration encoded in filename: double_ramp_<r1>_<r2>_ma_<Mach>_pres_<Pres> ...; we extract the numeric tokens.

Adding more sources: Just append to RAMP_SOURCES or FIELD_SOURCES.
"""
from __future__ import annotations
import os
import re
import glob
import json
from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ----------------- CONFIG (edit) -----------------
CONFIG = dict(
    ENABLED_SECTIONS=['ramps','fields'],
    OUTPUT_DIR='postprocess_outputs/mapping',
    # Ramp sources:
    #   type='npz': parse ramp parameters from NPZ filenames
    #   type='pipeline': parse from CFD sweep directory or extracted_points directory names
    RAMP_SOURCES=[
        dict(type='npz', path_glob='double_ramp_configuration/inputs/double_ramp_npz_files_clamped/double_ramp_*.npz', label='Ground Truth Ramps'),
        dict(type='pipeline', path_glob='double_ramp_configuration/outputs/backward/sweep/*/flow.vtu', label='Pipeline Ramps'),
        dict(type='npz', path_glob='double_ramp_configuration/outputs/backward/extracted_points/double_ramp_*.npy', label='Extracted Points Ramps'),
    ],
    # Field sources:
    #   kind='npz' provides ground truth field arrays
    #   kind='vtu' reads CFD results from flow.vtu (interpolated to uniform grid by simple scatter gridding)
    FIELD_SOURCES=[
        dict(kind='npz', field='density', path_glob='double_ramp_configuration/inputs/double_ramp_npz_files_clamped/*.npz', label='GT density'),
        dict(kind='vtu', field='density', path_glob='double_ramp_configuration/outputs/backward/sweep/*/flow.vtu', label='CFD density'),
        dict(kind='npz', field='temperature', path_glob='double_ramp_configuration/inputs/double_ramp_npz_files_clamped/*.npz', label='GT temperature'),
        dict(kind='vtu', field='Temperature', path_glob='double_ramp_configuration/outputs/backward/sweep/*/flow.vtu', label='CFD temperature'),
        dict(kind='npz', field='mach', path_glob='double_ramp_configuration/inputs/double_ramp_npz_files_clamped/*.npz', label='GT mach'),
        dict(kind='vtu', field='Mach', path_glob='double_ramp_configuration/outputs/backward/sweep/*/flow.vtu', label='CFD mach'),
        dict(kind='npz', field='pressure', path_glob='double_ramp_configuration/inputs/double_ramp_npz_files_clamped/*.npz', label='GT pressure'),
        dict(kind='vtu', field='Pressure', path_glob='double_ramp_configuration/outputs/backward/sweep/*/flow.vtu', label='CFD pressure'),
        # Additional CFD-only fields
        dict(kind='vtu', field='Energy', path_glob='double_ramp_configuration/outputs/backward/sweep/*/flow.vtu', label='CFD energy'),
        dict(kind='vtu', field='Momentum', path_glob='double_ramp_configuration/outputs/backward/sweep/*/flow.vtu', label='CFD momentum'),
        dict(kind='vtu', field='Pressure_Coefficient', path_glob='double_ramp_configuration/outputs/backward/sweep/*/flow.vtu', label='CFD Cp'),
        dict(kind='vtu', field='Velocity', path_glob='double_ramp_configuration/outputs/backward/sweep/*/flow.vtu', label='CFD velocity'),
    ],
    EMBEDDING=dict(
        method='umap',  # 'umap' or 'pca'
        n_components=2,
        random_state=42,
        n_neighbors=30,
        min_dist=0.2,
        sample_limit_per_group=400,
    ),
    NORMALIZE_FIELD='max',  # 'none'|'minmax'|'std'|'max'
    DOWNSAMPLE=2,           # stride (>=1). 2 reduces resolution by taking every 2nd pixel.
    VERBOSE=True,
)
# -------------------------------------------------

@dataclass
class RampRecord:
    label: str
    r1: float
    r2: float
    mach: Optional[float]
    pres: Optional[float]

# Regex list: first with mach & pres, second minimal (only r1,r2)
RAMP_REGEXES = [
    # With mach & pressure and optional trailing tokens
    re.compile(r"double_ramp_(?P<r1>[0-9\.]+)_(?P<r2>[0-9\.]+)_ma_(?P<mach>[0-9\.]+)_pres_(?P<pres>[0-9\.]+).*", re.IGNORECASE),
    # Only r1 & r2
    re.compile(r"double_ramp_(?P<r1>[0-9\.]+)_(?P<r2>[0-9\.]+).*", re.IGNORECASE),
]

FIELD_KEY_CANON = {
    'density':['density','rho'],
    'temperature':['temperature','temp','t'],
    'mach':['mach','machnumber','m'],
    'pressure':['pressure','p','press'],
}

# ----------------- Helpers -----------------

def _log(msg: str):
    if CONFIG.get('VERBOSE',True):
        print(msg)

def _collect_files(pattern: str) -> List[str]:
    files = glob.glob(pattern)
    files.sort()
    return files

# Ramp parsing
def parse_ramp_filename(path: str) -> Optional[Tuple[float,float,Optional[float],Optional[float]]]:
    base = os.path.basename(path)
    # Replace numeric compression: patterns like 0p0491 => 0.0491 (digit 'p' digit)
    token = re.sub(r"(?<=\d)p(?=\d)", ".", base)
    for rx in RAMP_REGEXES:
        m = rx.search(token)
        if not m:
            continue
        try:
            r1 = float(m.group('r1'))
            r2 = float(m.group('r2'))
            mach = float(m.group('mach')) if 'mach' in m.groupdict() and m.group('mach') else None
            pres = float(m.group('pres')) if 'pres' in m.groupdict() and m.group('pres') else None
            return r1, r2, mach, pres
        except Exception:
            return None
    return None

# Field extraction
def load_field_from_npz(path: str, field: str) -> Optional[np.ndarray]:
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None
    field_lower = field.lower()
    # find matching key
    for canon, aliases in FIELD_KEY_CANON.items():
        if canon == field_lower:
            for k in data.keys():
                kl = k.lower()
                if kl == field_lower or any(a == kl for a in aliases):
                    A = np.asarray(data[k]).astype(float)
                    if A.ndim == 2:
                        return A
                    elif A.ndim == 3:  # (ny,nx,nc) -> take magnitude
                        return np.linalg.norm(A, axis=2)
    return None

# Normalization
def normalize_array(A: np.ndarray, mode: str) -> np.ndarray:
    if mode == 'none':
        return A
    mask = np.isfinite(A)
    if not mask.any():
        return A
    if mode == 'max':
        m = np.nanmax(A)
        return A / m if m not in (0, np.nan) else A
    if mode == 'minmax':
        mn = np.nanmin(A)
        mx = np.nanmax(A)
        if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
            return A
        return (A - mn)/(mx - mn)
    if mode == 'std':
        mean = np.nanmean(A)
        std = np.nanstd(A)
        return (A-mean)/std if std>0 else A
    return A

# Downsample

def downsample(A: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return A
    return A[::stride, ::stride]

# Embedding method
class Embedder:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.method = cfg.get('method','umap').lower()
        self.n_components = int(cfg.get('n_components',2))
        self.random_state = cfg.get('random_state',42)
        self.n_neighbors = cfg.get('n_neighbors',30)
        self.min_dist = cfg.get('min_dist',0.1)
        self._impl = None
        self._init_impl()
    def _init_impl(self):
        if self.method == 'umap':
            try:
                import umap  # type: ignore
                self._impl = umap.UMAP(n_components=self.n_components, random_state=self.random_state,
                                       n_neighbors=self.n_neighbors, min_dist=self.min_dist)
            except Exception:
                _log('[WARN] umap-learn not available; falling back to PCA')
                self.method = 'pca'
        if self.method == 'pca':
            try:
                from sklearn.decomposition import PCA  # type: ignore
                self._impl = PCA(n_components=self.n_components, random_state=self.random_state)
            except Exception:
                _log('[WARN] scikit-learn not available; using NumPy SVD PCA fallback')
                self._impl = None
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        if self._impl is not None:
            return self._impl.fit_transform(X)
        # NumPy PCA fallback
        Xc = X - np.mean(X, axis=0)
        # economy SVD
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        comps = Vt[:self.n_components].T
        return Xc @ comps

# Plotting

def scatter_groups(out_path: str, points: np.ndarray, labels: List[str], title: str, legend_title: str='Group'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    unique = list(dict.fromkeys(labels))
    cmap = plt.get_cmap('tab10')
    plt.figure(figsize=(5.2,5.4))
    for i,u in enumerate(unique):
        idx = [j for j,label_j in enumerate(labels) if label_j == u]
        P = points[idx]
        plt.scatter(P[:,0], P[:,1], s=18, alpha=0.75, label=u, color=cmap(i%10))
    plt.xlabel('Embedding 1')
    plt.ylabel('Embedding 2')
    plt.title(title)
    plt.legend(frameon=True, title=legend_title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=260)
    plt.close()

# ----------------- Sections -----------------

def section_ramps():
    if 'ramps' not in CONFIG['ENABLED_SECTIONS']:
        return
    records: List[RampRecord] = []
    for src in CONFIG['RAMP_SOURCES']:
        paths = _collect_files(src['path_glob'])
        if not paths:
            _log(f"[INFO] No paths matched for ramp pattern: {src['path_glob']}")
        stype = src.get('type','npz')
        matched_this_source = 0
        for p in paths:
            if stype == 'pipeline':
                # Directory structure: .../sweep/<dir>/flow.vtu ; parse parent directory name (convert p->.)
                parent = os.path.basename(os.path.dirname(p))
                parsed = parse_ramp_filename(parent)
            else:
                # NPZ / others: parse file name
                parsed = parse_ramp_filename(p)
            if parsed:
                r1,r2,ma,pr = parsed
                records.append(RampRecord(label=src['label'], r1=r1, r2=r2, mach=ma, pres=pr))
                matched_this_source += 1
            else:
                _log(f"[DEBUG] Could not parse ramp tokens from: {p}")
        _log(f"[INFO] Parsed {matched_this_source} ramps from source label='{src['label']}' pattern='{src['path_glob']}'")
    if not records:
        _log('[WARN] No ramp configurations parsed.')
        return
    # Build feature matrix: [r1, r2, mach (opt), pres (opt)]
    feats = []
    labels = []
    for rec in records:
        row = [rec.r1, rec.r2]
        if rec.mach is not None:
            row.append(rec.mach)
        if rec.pres is not None:
            row.append(rec.pres)
        feats.append(row)
        labels.append(rec.label)
    X = np.asarray(feats, float)
    # Standardize
    mu = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std[std==0] = 1.0
    Xs = (X-mu)/std
    emb = Embedder(CONFIG['EMBEDDING']).fit_transform(Xs)
    out_path = os.path.join(CONFIG['OUTPUT_DIR'], 'ramps_embedding.png')
    scatter_groups(out_path, emb, labels, 'Ramp Configuration Embeddings', 'Source')
    _log(f'[OK] Saved ramp embeddings -> {out_path}')


def _read_vtu_field(vtu_path: str, field_pref: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    try:
        import meshio  # reuse existing dependency
        m = meshio.read(vtu_path)
    except Exception:
        return None
    # candidate keys search
    cands = []
    pref_low = field_pref.lower()
    for k in list(m.point_data.keys()):
        cands.append(k)
    if not cands:
        _log(f"[DEBUG] No point_data keys in {vtu_path}")
    chosen = None
    for k in cands:
        if k.lower() == pref_low:
            chosen = k
            break
    if chosen is None:
        for k in cands:
            if pref_low in k.lower():
                chosen = k
                break
    if chosen is None:
        _log(f"[DEBUG] Field '{field_pref}' not found in {vtu_path}; available keys={cands}")
        return None
    pts = m.points[:, :2]
    raw = np.asarray(m.point_data[chosen]).astype(float)
    if raw.ndim == 2 and raw.shape[1] in (2,3):
        vals = np.linalg.norm(raw[:,:2], axis=1)  # use planar magnitude
    else:
        vals = raw.ravel()
    # Build quick regular grid via bounding box scatter-binning
    nx = 128
    ny = 128
    xmin,xmax = pts[:,0].min(), pts[:,0].max()
    ymin,ymax = pts[:,1].min(), pts[:,1].max()
    gx = np.linspace(xmin,xmax,nx)
    gy = np.linspace(ymin,ymax,ny)
    # simple nearest bin assignment
    ix = np.clip(np.searchsorted(gx, pts[:,0]) - 1, 0, nx-1)
    iy = np.clip(np.searchsorted(gy, pts[:,1]) - 1, 0, ny-1)
    grid = np.full((ny,nx), np.nan, float)
    # average duplicates
    counts = np.zeros_like(grid)
    for xv,yv,v in zip(ix,iy,vals):
        if np.isnan(grid[yv,xv]):
            grid[yv,xv] = v
            counts[yv,xv] = 1
        else:
            grid[yv,xv] += v
            counts[yv,xv] += 1
    mask = counts>0
    grid[mask] = grid[mask]/counts[mask]
    return gx, gy, grid

def section_fields():
    if 'fields' not in CONFIG['ENABLED_SECTIONS']:
        return
    # Collect arrays grouped by field name
    per_field_outputs: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for src in CONFIG['FIELD_SOURCES']:
        field = src['field']
        kind = src.get('kind','npz')
        paths = _collect_files(src['path_glob'])
        if not paths:
            _log(f"[WARN] No files for pattern {src['path_glob']}")
            continue
        for p in paths:
            A = None
            if kind == 'npz':
                A = load_field_from_npz(p, field)
            elif kind == 'vtu':
                res = _read_vtu_field(p, field)
                if res is not None:
                    _,_,A = res
            if A is None:
                continue
            A = downsample(A, CONFIG.get('DOWNSAMPLE',1))
            A = normalize_array(A, CONFIG.get('NORMALIZE_FIELD','none'))
            per_field_outputs.setdefault(field.lower(), []).append((src['label'], A))
    if not per_field_outputs:
        _log('[WARN] No field data collected.')
        return
    # For each field create embedding plot comparing labels
    for fld, items in per_field_outputs.items():
        data_rows = []
        labels = []
        limit = CONFIG['EMBEDDING'].get('sample_limit_per_group')
        # group by label to apply limit
        grouped: Dict[str, List[np.ndarray]] = {}
        for label, arr in items:
            grouped.setdefault(label, []).append(arr)
        for label, arrs in grouped.items():
            if limit and len(arrs) > limit:
                arrs = arrs[:limit]
            for A in arrs:
                data_rows.append(A.ravel())
                labels.append(label)
        X = np.vstack(data_rows)
        col_std = np.nanstd(X, axis=0)
        keep = col_std > 1e-12
        Xr = X[:, keep]
        emb = Embedder(CONFIG['EMBEDDING']).fit_transform(Xr)
        out_path = os.path.join(CONFIG['OUTPUT_DIR'], f'{fld}_embedding.png')
        scatter_groups(out_path, emb, labels, f'{fld.capitalize()} Field Embeddings', 'Source')
        _log(f'[OK] Saved field embeddings -> {out_path}')

# ----------------- Main -----------------

def main():
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    if 'ramps' in CONFIG['ENABLED_SECTIONS']:
        section_ramps()
    if 'fields' in CONFIG['ENABLED_SECTIONS']:
        section_fields()
    # Save a small manifest
    with open(os.path.join(CONFIG['OUTPUT_DIR'], 'embedding_config.json'),'w') as f:
        json.dump(CONFIG, f, indent=2)

if __name__ == '__main__':
    main()
