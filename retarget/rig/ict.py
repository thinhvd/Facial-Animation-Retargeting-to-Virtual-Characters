"""ICT-FaceKit template (MIT): generic neutral mesh + 53 expression shapes.

Expected layout (shallow clone of https://github.com/USC-ICT/ICT-FaceKit):
    data/ict-facekit/FaceXModel/generic_neutral_mesh.obj
    data/ict-facekit/FaceXModel/<expression>.obj      53 files, ARKit names with _L/_R
    data/ict-facekit/FaceXModel/identityNNN.obj       100 PCA identity modes
    data/ict-facekit/FaceXModel/vertex_indices.json   regions, landmarks, fitting vertices

Vertex regions (from the ICT README): Face [0:9408], Head and Neck [9409:11247],
Mouth socket [11248:13293], Eye sockets, Gums/tongue, Teeth [17039:21450],
Eyeballs [21451:24590], ... up to 26718.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..formats.obj import read_obj, read_obj_positions

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / 'data' / 'ict-facekit' / 'FaceXModel'

REGIONS = {
    'face': (0, 9408), 'head_neck': (9409, 11247), 'mouth_socket': (11248, 13293),
    'eye_socket_L': (13294, 13677), 'eye_socket_R': (13678, 14061), 'gums_tongue': (14062, 17038),
    'teeth': (17039, 21450), 'eyeball_L': (21451, 23020), 'eyeball_R': (23021, 24590),
    'lacrimal_L': (24591, 24794), 'lacrimal_R': (24795, 24998), 'eye_blend_L': (24999, 25022),
    'eye_blend_R': (25023, 25046), 'eye_occlusion_L': (25047, 25198), 'eye_occlusion_R': (25199, 25350),
    'eyelashes_L': (25351, 26034), 'eyelashes_R': (26035, 26718),
}


def arkit_name(ict_name: str) -> str:
    """ICT 'browInnerUp_L' -> ARKit 'browInnerUpLeft'; names without a side pass through."""
    if ict_name.endswith('_L'):
        return ict_name[:-2] + 'Left'
    if ict_name.endswith('_R'):
        return ict_name[:-2] + 'Right'
    return ict_name


@dataclass
class ICTTemplate:
    positions: np.ndarray                   # (V, 3) neutral, metres, canonical axes
    triangles: np.ndarray                   # (F, 3)
    uvs: np.ndarray | None
    expressions: dict[str, np.ndarray]      # ARKit name -> (V, 3) delta
    landmarks68: np.ndarray                 # (68,) vertex indices (Multi-PIE 68 convention)
    fitting_verts: np.ndarray               # vertices ICT uses for fitting
    rigid_verts: np.ndarray
    scale: float
    extras: dict = field(default_factory=dict)

    @property
    def vertex_count(self) -> int:
        return len(self.positions)

    def region(self, name: str) -> np.ndarray:
        a, b = REGIONS[name]
        return np.arange(a, b + 1)

    def submesh(self, vertex_ids: np.ndarray):
        """Triangles entirely inside `vertex_ids`, re-indexed; returns (local_tris, vertex_ids)."""
        keep = np.zeros(self.vertex_count, bool)
        keep[vertex_ids] = True
        tri = self.triangles[keep[self.triangles].all(1)]
        remap = -np.ones(self.vertex_count, np.int64)
        remap[vertex_ids] = np.arange(len(vertex_ids))
        return remap[tri], vertex_ids


def load(model_dir: str | Path = DEFAULT_DIR, scale: float | None = None,
         expressions: bool = True) -> ICTTemplate:
    """Load the template.  ICT units are centimetres; default scale converts to metres."""
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / 'vertex_indices.json').read_text())
    pos, tri, uvs = read_obj(model_dir / 'generic_neutral_mesh.obj', want_uv=True)
    if scale is None:
        height = float(pos[:, 1].max() - pos[:, 1].min())
        scale = 0.01 if height > 5 else 1.0       # cm -> m when the head is "tens of units" tall
    exprs: dict[str, np.ndarray] = {}
    if expressions:
        for name in cfg['expressions']:
            p = read_obj_positions(model_dir / f'{name}.obj')
            exprs[arkit_name(name)] = ((p - pos) * scale).astype(np.float32)
    return ICTTemplate(
        positions=(pos * scale).astype(np.float32), triangles=tri, uvs=uvs, expressions=exprs,
        landmarks68=np.asarray(cfg['idx_to_landmark_verts'], np.int64),
        fitting_verts=np.asarray(cfg['idx_to_fitting_verts'], np.int64),
        rigid_verts=np.asarray(cfg['idx_to_rigid_verts'], np.int64),
        scale=scale, extras={'dir': str(model_dir), 'expression_names_ict': cfg['expressions']},
    )


def load_identity(model_dir: str | Path, index: int, scale: float) -> np.ndarray:
    """Return the (V, 3) delta of identity mode `index` (metres)."""
    model_dir = Path(model_dir)
    base = read_obj_positions(model_dir / 'generic_neutral_mesh.obj')
    p = read_obj_positions(model_dir / f'identity{index:03d}.obj')
    return ((p - base) * scale).astype(np.float32)


if __name__ == '__main__':
    import sys
    t = load(expressions=('--no-expr' not in sys.argv))
    lo, hi = t.positions.min(0), t.positions.max(0)
    print(f'ICT template: V={t.vertex_count} F={len(t.triangles)} scale={t.scale} '
          f'bounds(m) min={lo.round(3).tolist()} max={hi.round(3).tolist()}')
    print(f'  expressions: {len(t.expressions)} e.g. {list(t.expressions)[:6]}')
    print(f'  landmarks68: {t.landmarks68[:8].tolist()}..., fitting verts: {len(t.fitting_verts)}')
    for name, d in list(t.expressions.items())[:5]:
        moved = (np.abs(d).sum(1) > 1e-6).sum()
        print(f'  {name:22} moves {moved:5d} verts, max {np.abs(d).max() * 1000:.1f} mm')
