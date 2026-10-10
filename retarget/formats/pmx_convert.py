"""PMX -> CanonicalMesh.

Mirrors what three.js MMDLoader does (mmd-parser `leftToRightModel`):
negate Z of positions, normals, bone positions and morph offsets, swap the
triangle winding, keep UVs.  On top of that we scale MMD units to metres
(1 MMD unit ~ 8 cm) so the model sits in the same space as glTF/VRM
avatars.
"""
from __future__ import annotations

import io
from pathlib import Path
from urllib.parse import quote, unquote

import numpy as np

from ..mesh import CanonicalBone, CanonicalImage, CanonicalMaterial, CanonicalMesh, CanonicalMorph
from . import pmx

MMD_UNIT_METRES = 0.08
PANELS = {1: 'eyebrow', 2: 'eye', 3: 'mouth', 4: 'other'}

_AXES = {'+x': (1, 0, 0), '-x': (-1, 0, 0), '+y': (0, 1, 0), '-y': (0, -1, 0),
         '+z': (0, 0, 1), '-z': (0, 0, -1), 'x': (1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, 1)}


def axis_matrix(up: str = '+y', front: str = '-z') -> np.ndarray:
    """3x3 matrix taking left-handed PMX coordinates to canonical (right-handed,
    Y up, character faces +Z, character's left at +X).

    Standard MMD models are up=+y, front=-z, which yields diag(1, 1, -1): the
    same mirror three.js MMDLoader applies.  Models exported from other tools
    (e.g. a Z-up BVH rig through mmd_tools) pass their own axes.
    """
    u = np.array(_AXES[up.lower()], np.float64)
    f = np.array(_AXES[front.lower()], np.float64)
    if abs(u @ f) > 1e-6:
        raise ValueError('up and front axes must be perpendicular')
    left = -np.cross(u, f)          # left-handed source: character's left = -(up x front)
    m = np.stack([left, u, f])      # rows: out_x, out_y, out_z as combinations of source axes
    assert abs(np.linalg.det(m) + 1) < 1e-6, 'expected a single reflection'
    return m.astype(np.float32)


def resolve_texture(base: Path, rel: str) -> Path | None:
    """Resolve a PMX texture path case-insensitively (files come from Windows),
    also matching percent-encoded file names."""
    rel = rel.replace('\\', '/').strip()
    cand = base / rel
    if cand.is_file():
        return cand
    cur = base
    for part in rel.split('/'):
        if not part or part == '.':
            continue
        match = None
        if cur.is_dir():
            # zip tools / git hosts sometimes store non-ASCII names percent-encoded (顔.png -> %E9%A1%94.png)
            wanted = {part.lower(), quote(part).lower()}
            for child in cur.iterdir():
                if child.name.lower() in wanted or unquote(child.name).lower() == part.lower():
                    match = child
                    break
        if match is None:
            return None
        cur = match
    return cur if cur.is_file() else None


def load_image(path: Path) -> CanonicalImage | None:
    from PIL import Image
    try:
        im = Image.open(path)
        im.load()
    except Exception:
        return None
    has_alpha = im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info)
    if im.format in ('PNG', 'JPEG'):
        return CanonicalImage(path.name, path.read_bytes(),
                              'image/png' if im.format == 'PNG' else 'image/jpeg', has_alpha)
    buf = io.BytesIO()
    im.convert('RGBA' if has_alpha else 'RGB').save(buf, format='PNG')
    return CanonicalImage(path.stem + '.png', buf.getvalue(), 'image/png', has_alpha)


def to_canonical(model: pmx.PMXModel, scale: float = MMD_UNIT_METRES,
                 include_textures: bool = True, up: str = '+y', front: str = '-z') -> CanonicalMesh:
    R = axis_matrix(up, front)
    xf = lambda a: a @ R.T          # (N, 3) source -> canonical, before scaling
    positions = (xf(model.positions) * scale).astype(np.float32)
    normals = xf(model.normals)
    n = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = (normals / np.where(n > 0, n, 1)).astype(np.float32)
    triangles = model.triangles[:, [0, 2, 1]].astype(np.uint32)

    # ---- textures / materials
    images: list[CanonicalImage] = []
    image_of_texture: dict[int, int] = {}
    base = model.path.parent
    if include_textures:
        for ti, rel in enumerate(model.textures):
            p = resolve_texture(base, rel)
            img = load_image(p) if p else None
            if img is not None:
                image_of_texture[ti] = len(images)
                images.append(img)

    materials: list[CanonicalMaterial] = []
    primitives: list[tuple[int, int, int]] = []
    for mi, (mat, t0, t1) in enumerate(model.material_triangle_ranges()):
        img = image_of_texture.get(mat.texture, -1)
        alpha = float(mat.diffuse[3])
        if alpha < 0.999:
            mode = 'BLEND'
        elif img >= 0 and images[img].has_alpha:
            mode = 'MASK'
        else:
            mode = 'OPAQUE'
        materials.append(CanonicalMaterial(
            name=mat.name or f'material_{mi}',
            base_color=np.clip(mat.diffuse, 0, 1).astype(np.float32),
            image=img,
            double_sided=bool(mat.flags & 0x01),
            alpha_mode=mode,
            extras={'name_en': mat.name_en, 'pmx_texture': mat.texture},
        ))
        primitives.append((mi, t0, t1))

    # ---- bones
    bones = [CanonicalBone(b.name, b.name_en, b.parent, (xf(b.position) * scale).astype(np.float32),
                           extras={'pmx_flags': int(b.flags)})
             for b in model.bones]

    # ---- morphs: vertex morphs and group morphs that resolve to vertex offsets
    morphs: list[CanonicalMorph] = []
    for i, mo in enumerate(model.morphs):
        if mo.type not in (0, 1):
            continue
        disp = model.morph_displacement(i)
        if not np.any(disp):
            continue
        morphs.append(CanonicalMorph(
            name=mo.name,
            displacement=(xf(disp) * scale).astype(np.float32),
            extras={'name_en': mo.name_en, 'panel': PANELS.get(mo.panel, 'other'),
                    'pmx_index': i, 'pmx_type': pmx.MORPH_TYPES[mo.type]},
        ))

    return CanonicalMesh(
        name=model.name or model.path.stem,
        positions=positions, normals=normals, uvs=model.uvs.astype(np.float32),
        triangles=triangles, primitives=primitives, materials=materials, images=images,
        bones=bones, joints=model.joints.copy(), weights=model.weights.copy(), morphs=morphs,
        extras={'source_format': 'pmx', 'source_path': str(model.path), 'scale': scale,
                'source_up': up, 'source_front': front,
                'source_name_en': model.name_en, 'source_comment': model.comment[:500]},
    )
