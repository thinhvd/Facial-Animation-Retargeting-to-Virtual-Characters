"""Guess a converted model's up/front axes and unit scale from its skeleton.

Runs on the canonical mesh, which should be Y up, facing +Z, character's
left at +X, in metres.  Files saved with the wrong axes or units (a Z-up rig
exported as Y-up FBX, a model in centimetres) are rotated and rescaled back.

  up    head bone minus hips / pelvis / neck / skeleton root
  left  mean of (left - right) over bones paired by name (Left*, 左*, *_L, *.L, ...)
  front left x up (falls back to toes or eyes ahead of the head)
  scale a power of ten that brings the height into a plausible range
        (0.5-4 m with leg bones, 0.1-1 m for a head)

The user's --up / --front / --scale always win over this.
"""
from __future__ import annotations

import re

import numpy as np

from ..mesh import CanonicalMesh
from ..naming import BONE_ALIASES

_HEAD_EXCLUDE = re.compile(r'top|end|fore|nub|tip', re.I)
_BASE = re.compile(r'hips?$|pelvis|センター|下半身|^腰$', re.I)
_NECK = re.compile(r'neck|^首$', re.I)
_LOWER_BODY = re.compile(r'foot|leg|toe|knee|ankle|足|ひざ', re.I)
_TOE = re.compile(r'toe|つま先', re.I)
# side token -> (side, neutral key); first match wins
_SIDE = [(re.compile(r'left', re.I), 'L'), (re.compile(r'right', re.I), 'R'),
         (re.compile(r'左'), 'L'), (re.compile(r'右'), 'R'),
         (re.compile(r'(?<=[._\s-])l(?=$|[._\s-])|^l(?=[._\s-])', re.I), 'L'),
         (re.compile(r'(?<=[._\s-])r(?=$|[._\s-])|^r(?=[._\s-])', re.I), 'R')]


def _snap(v: np.ndarray, min_share: float = 0.6) -> np.ndarray | None:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return None
    i = int(np.argmax(np.abs(v)))
    if abs(v[i]) < min_share * n:
        return None
    out = np.zeros(3)
    out[i] = np.sign(v[i])
    return out


def _bone(mesh: CanonicalMesh, pattern: re.Pattern, exclude: re.Pattern | None = None) -> int:
    for i, b in enumerate(mesh.bones):
        if pattern.search(b.name) and not (exclude and exclude.search(b.name)):
            return i
    return -1


def _head(mesh: CanonicalMesh) -> int:
    for name in BONE_ALIASES['head']:
        i = mesh.bone(name)
        if i >= 0:
            return i
    return _bone(mesh, re.compile(r'head', re.I), _HEAD_EXCLUDE)


def _left_vector(mesh: CanonicalMesh) -> np.ndarray | None:
    sides: dict[str, dict[str, np.ndarray]] = {}
    for b in mesh.bones:
        for rx, side in _SIDE:
            if rx.search(b.name):
                key = rx.sub('#', b.name, count=1)
                sides.setdefault(key, {})[side] = np.asarray(b.position, np.float64)
                break
    diffs = [s['L'] - s['R'] for s in sides.values() if 'L' in s and 'R' in s]
    return np.mean(diffs, axis=0) if diffs else None


def guess(mesh: CanonicalMesh) -> tuple[np.ndarray, float, str]:
    """(R, scale, note): canonical = scale * R @ v.  R is a proper rotation."""
    R = np.eye(3)
    notes = []
    head = _head(mesh)
    if head >= 0:
        pos = lambda i: np.asarray(mesh.bones[i].position, np.float64)
        base = _bone(mesh, _BASE)
        if base < 0 or base == head:
            base = _bone(mesh, _NECK)
        if base < 0 or base == head:
            base = next((i for i, b in enumerate(mesh.bones) if b.parent < 0 and i != head), -1)
        up = _snap(pos(head) - pos(base)) if base >= 0 else None
        left = _left_vector(mesh)
        left = _snap(left) if left is not None else None
        if up is not None:
            if left is None or abs(left @ up) > 0.5:
                toe = _bone(mesh, _TOE)
                ahead = None
                if toe >= 0 and mesh.bones[toe].parent >= 0:
                    ahead = pos(toe) - pos(mesh.bones[toe].parent)
                if ahead is not None:
                    ahead = ahead - (ahead @ up) * up
                f = _snap(ahead) if ahead is not None else None
                if f is None or abs(f @ up) > 0.5:
                    f = np.array([0, 0, 1.0]) if abs(up[2]) < 0.5 else np.array([0, -up[2], 0])
                left = np.cross(up, f)
            front = np.cross(left, up)
            R = np.stack([left, up, front])
            if not np.allclose(R, np.eye(3)):
                names = {(1, 0, 0): '+x', (-1, 0, 0): '-x', (0, 1, 0): '+y', (0, -1, 0): '-y', (0, 0, 1): '+z', (0, 0, -1): '-z'}
                notes.append(f'axes: up={names[tuple(int(v) for v in up)]} front={names[tuple(int(v) for v in front)]} '
                             f'(from bones {mesh.bones[head].name} / {mesh.bones[base].name})')
    p = mesh.positions.astype(np.float64) @ R.T
    height = float(p[:, 1].max() - p[:, 1].min())
    scale = 1.0
    full_body = _bone(mesh, _LOWER_BODY) >= 0
    lo, hi, target = (0.5, 4.0, 1.6) if full_body else (0.1, 1.0, 0.3)
    if not lo < height < hi and height > 0:
        k = int(np.round(np.log10(target / height)))
        scale = 10.0 ** k
        notes.append(f'scale x{scale:g} (height {height:.3g} m -> {height * scale:.3g} m)')
    return R, scale, '; '.join(notes)


def apply(mesh: CanonicalMesh, R: np.ndarray, scale: float = 1.0) -> CanonicalMesh:
    """Rotate (proper rotation, so triangle winding is kept) and rescale in place."""
    R = np.asarray(R, np.float32)
    mesh.positions = (mesh.positions @ R.T) * scale
    mesh.normals = mesh.normals @ R.T
    for m in mesh.morphs:
        m.displacement = (m.displacement @ R.T) * scale
    for b in mesh.bones:
        b.position = (np.asarray(b.position, np.float32) @ R.T) * scale
    return mesh
