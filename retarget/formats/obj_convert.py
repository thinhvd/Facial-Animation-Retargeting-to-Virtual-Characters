"""OBJ -> CanonicalMesh (no rig, one lit material).

Used for template meshes (ICT-FaceKit) and raw scans.  Optional sibling
OBJ files with the same vertex order can be attached as morphs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..mesh import CanonicalMaterial, CanonicalMesh, CanonicalMorph
from .obj import read_obj


def vertex_normals(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    n = np.zeros_like(positions, dtype=np.float64)
    a, b, c = (positions[triangles[:, i]].astype(np.float64) for i in range(3))
    fn = np.cross(b - a, c - a)
    for i in range(3):
        np.add.at(n, triangles[:, i], fn)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return (n / np.where(length > 0, length, 1)).astype(np.float32)


def to_canonical(path: str | Path, scale: float = 1.0, up: str = '+y', front: str = '+z',
                 morphs: dict[str, np.ndarray] | None = None, name: str | None = None,
                 color=(0.8, 0.72, 0.65, 1.0)) -> CanonicalMesh:
    path = Path(path)
    pos, tri, uvs = read_obj(path, want_uv=True)
    R = _axis_matrix_rh(up, front)
    pos = (pos @ R.T) * scale
    if uvs is None:
        uvs = np.zeros((len(pos), 2), np.float32)
    mesh = CanonicalMesh(
        name=name or path.stem,
        positions=pos.astype(np.float32),
        normals=vertex_normals(pos, tri),
        uvs=uvs.astype(np.float32),
        triangles=tri.astype(np.uint32),
        primitives=[(0, 0, len(tri))],
        materials=[CanonicalMaterial(name='default', base_color=np.array(color, np.float32), unlit=False)],
        extras={'source_format': 'obj', 'source_path': str(path), 'scale': scale,
                'source_up': up, 'source_front': front},
    )
    for mname, disp in (morphs or {}).items():
        mesh.morphs.append(CanonicalMorph(mname, ((disp @ R.T) * scale).astype(np.float32)))
    return mesh


_AXES = {'+x': (1, 0, 0), '-x': (-1, 0, 0), '+y': (0, 1, 0), '-y': (0, -1, 0), '+z': (0, 0, 1), '-z': (0, 0, -1)}


def _axis_matrix_rh(up: str, front: str) -> np.ndarray:
    """Right-handed source (OBJ) -> canonical; the third axis follows from the cross product."""
    u = np.array(_AXES[up.lower()], np.float64)
    f = np.array(_AXES[front.lower()], np.float64)
    if abs(u @ f) > 1e-6:
        raise ValueError('up and front axes must be perpendicular')
    left = np.cross(u, f)       # right-handed: x = y cross z
    m = np.stack([left, u, f])
    assert abs(np.linalg.det(m) - 1) < 1e-6
    return m.astype(np.float32)
