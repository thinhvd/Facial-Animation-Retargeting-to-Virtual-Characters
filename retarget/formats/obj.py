"""Minimal Wavefront OBJ reader/writer that preserves vertex order.

Blendshape files (ICT-FaceKit, FLAME exports, ...) share one vertex
ordering across the neutral and every expression; a library that merges
or re-indexes vertices would silently break that, so this reader keeps
`v` lines exactly as given and only triangulates faces.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def read_obj(path: str | Path, want_uv: bool = False):
    """Return (positions (V,3) float32, triangles (F,3) uint32[, uvs (V,2) or None]).

    Faces referencing `v/vt/vn` keep only the `v` index.  UVs are returned per
    vertex (first vt seen for each vertex) when `want_uv` is set.
    """
    positions: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, float]] = []
    faces: list[tuple[int, int, int]] = []
    uv_of_vertex: dict[int, int] = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                positions.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith('vt ') and want_uv:
                p = line.split()
                texcoords.append((float(p[1]), float(p[2])))
            elif line.startswith('f '):
                corners = []
                for tok in line.split()[1:]:
                    parts = tok.split('/')
                    vi = int(parts[0])
                    vi = vi - 1 if vi > 0 else len(positions) + vi
                    corners.append(vi)
                    if want_uv and len(parts) > 1 and parts[1]:
                        ti = int(parts[1])
                        uv_of_vertex.setdefault(vi, ti - 1 if ti > 0 else len(texcoords) + ti)
                for k in range(1, len(corners) - 1):   # fan triangulation
                    faces.append((corners[0], corners[k], corners[k + 1]))
    pos = np.asarray(positions, np.float32)
    tri = np.asarray(faces, np.uint32).reshape(-1, 3)
    if not want_uv:
        return pos, tri
    uvs = None
    if texcoords:
        uvs = np.zeros((len(pos), 2), np.float32)
        tc = np.asarray(texcoords, np.float32)
        idx = np.fromiter(uv_of_vertex.keys(), dtype=np.int64)
        uvs[idx] = tc[np.fromiter(uv_of_vertex.values(), dtype=np.int64)]
    return pos, tri, uvs


def read_obj_positions(path: str | Path) -> np.ndarray:
    """Fast path for blendshape files: only the `v` lines, in order."""
    out = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                out.append((float(p[1]), float(p[2]), float(p[3])))
    return np.asarray(out, np.float32)


def write_obj(path: str | Path, positions: np.ndarray, triangles: np.ndarray, uvs: np.ndarray | None = None) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for p in positions:
            f.write(f'v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')
        if uvs is not None:
            for t in uvs:
                f.write(f'vt {t[0]:.6f} {t[1]:.6f}\n')
        for t in triangles:
            a, b, c = (int(v) + 1 for v in t)
            if uvs is not None:
                f.write(f'f {a}/{a} {b}/{b} {c}/{c}\n')
            else:
                f.write(f'f {a} {b} {c}\n')
