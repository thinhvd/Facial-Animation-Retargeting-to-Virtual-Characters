"""PMX 2.0/2.1 reader (MikuMikuDance model format).

Only what the retargeting pipeline needs is kept in structured form:
vertices (position, normal, uv, skin weights), triangles, materials with
textures, bones with hierarchy, and morphs (vertex / group / bone / uv /
material).  Rigid bodies, joints and soft bodies are not parsed; the
reader stops after display frames.

Coordinate system: PMX is left-handed, Y-up, 1 unit ~ 8 cm (MMD
convention).  No conversion is done here; see gltf_writer.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class _Reader:
    def __init__(self, data: bytes):
        self.b = data
        self.p = 0

    def take(self, n: int) -> bytes:
        v = self.b[self.p:self.p + n]
        self.p += n
        return v

    def unpack(self, fmt: str):
        fmt = '<' + fmt
        size = struct.calcsize(fmt)
        v = struct.unpack(fmt, self.b[self.p:self.p + size])
        self.p += size
        return v

    def u8(self) -> int:
        return self.unpack('B')[0]

    def i32(self) -> int:
        return self.unpack('i')[0]

    def f32(self) -> float:
        return self.unpack('f')[0]

    def vec(self, n: int):
        return np.array(self.unpack('f' * n), dtype=np.float32)


_SIGNED = {1: 'b', 2: 'h', 4: 'i'}
_UNSIGNED = {1: 'B', 2: 'H', 4: 'i'}  # vertex indices are unsigned (int32 for size 4)


@dataclass
class Material:
    name: str
    name_en: str
    diffuse: np.ndarray          # rgba
    specular: np.ndarray         # rgb
    specular_power: float
    ambient: np.ndarray          # rgb
    flags: int                   # bit0 double-sided, bit4 edge
    edge_color: np.ndarray
    edge_size: float
    texture: int                 # index into textures, -1 none
    sphere_texture: int
    sphere_mode: int
    toon_shared: bool
    toon_texture: int
    memo: str
    face_count: int              # number of vertex indices (3 * triangles)


@dataclass
class Bone:
    name: str
    name_en: str
    position: np.ndarray
    parent: int
    layer: int
    flags: int
    tail: int | np.ndarray       # bone index or offset vector
    inherit_parent: int = -1
    inherit_ratio: float = 1.0
    fixed_axis: np.ndarray | None = None
    local_x: np.ndarray | None = None
    local_z: np.ndarray | None = None
    external_key: int = -1
    ik_target: int = -1
    ik_links: list = field(default_factory=list)


@dataclass
class Morph:
    name: str
    name_en: str
    panel: int                   # 1 eyebrow, 2 eye, 3 mouth, 4 other
    type: int                    # 0 group, 1 vertex, 2 bone, 3-7 uv, 8 material, 9 flip, 10 impulse
    indices: np.ndarray          # per-offset target index
    data: np.ndarray             # per-offset payload (shape depends on type)


MORPH_TYPES = {0: 'group', 1: 'vertex', 2: 'bone', 3: 'uv', 4: 'uv1', 5: 'uv2',
               6: 'uv3', 7: 'uv4', 8: 'material', 9: 'flip', 10: 'impulse'}


@dataclass
class PMXModel:
    path: Path
    version: float
    name: str
    name_en: str
    comment: str
    comment_en: str
    positions: np.ndarray        # (V, 3)
    normals: np.ndarray          # (V, 3)
    uvs: np.ndarray              # (V, 2)
    add_uvs: np.ndarray          # (V, n, 4)
    weight_type: np.ndarray      # (V,) 0 BDEF1, 1 BDEF2, 2 BDEF4, 3 SDEF, 4 QDEF
    joints: np.ndarray           # (V, 4) bone indices, -1 padded
    weights: np.ndarray          # (V, 4)
    edge_scale: np.ndarray       # (V,)
    triangles: np.ndarray        # (F, 3) uint32
    textures: list[str]
    materials: list[Material]
    bones: list[Bone]
    morphs: list[Morph]

    @property
    def vertex_count(self) -> int:
        return len(self.positions)

    def bone_index(self, name: str) -> int:
        for i, b in enumerate(self.bones):
            if b.name == name or b.name_en == name:
                return i
        return -1

    def morph_index(self, name: str) -> int:
        for i, m in enumerate(self.morphs):
            if m.name == name or m.name_en == name:
                return i
        return -1

    def morph_displacement(self, index: int, ratio: float = 1.0) -> np.ndarray:
        """Dense (V, 3) displacement of a vertex or group morph.

        Group morphs are expanded recursively; non-vertex children are ignored.
        """
        out = np.zeros_like(self.positions)
        self._accumulate(index, ratio, out, depth=0)
        return out

    def _accumulate(self, index: int, ratio: float, out: np.ndarray, depth: int) -> None:
        if depth > 8 or index < 0 or index >= len(self.morphs):
            return
        m = self.morphs[index]
        if m.type == 1:
            np.add.at(out, m.indices, m.data * ratio)
        elif m.type == 0:
            for child, r in zip(m.indices, m.data):
                self._accumulate(int(child), ratio * float(r), out, depth + 1)

    def material_triangle_ranges(self):
        """Yield (material, start_tri, end_tri) in file order."""
        start = 0
        for mat in self.materials:
            n = mat.face_count // 3
            yield mat, start, start + n
            start += n


def load(path: str | Path) -> PMXModel:
    path = Path(path)
    r = _Reader(path.read_bytes())
    if r.take(4) != b'PMX ':
        raise ValueError(f'{path}: not a PMX file')
    version = r.f32()
    n_globals = r.u8()
    g = list(r.take(n_globals))
    enc, add_uv_count, vsz, tsz, msz, bsz, mosz, rsz = g[:8]
    codec = 'utf-16-le' if enc == 0 else 'utf-8'

    def text() -> str:
        n = r.i32()
        return r.take(n).decode(codec, errors='replace')

    def idx(size: int) -> int:
        return r.unpack(_SIGNED[size])[0]

    name, name_en, comment, comment_en = text(), text(), text(), text()

    # ---- vertices
    nv = r.i32()
    positions = np.empty((nv, 3), np.float32)
    normals = np.empty((nv, 3), np.float32)
    uvs = np.empty((nv, 2), np.float32)
    add_uvs = np.zeros((nv, add_uv_count, 4), np.float32)
    weight_type = np.empty(nv, np.uint8)
    joints = np.full((nv, 4), -1, np.int32)
    weights = np.zeros((nv, 4), np.float32)
    edge_scale = np.empty(nv, np.float32)
    bfmt = _SIGNED[bsz]
    for i in range(nv):
        positions[i] = r.unpack('fff')
        normals[i] = r.unpack('fff')
        uvs[i] = r.unpack('ff')
        for k in range(add_uv_count):
            add_uvs[i, k] = r.unpack('ffff')
        wt = r.u8()
        weight_type[i] = wt
        if wt == 0:
            joints[i, 0] = idx(bsz)
            weights[i, 0] = 1.0
        elif wt == 1:
            joints[i, :2] = r.unpack(bfmt + bfmt)
            w = r.f32()
            weights[i, :2] = (w, 1.0 - w)
        elif wt in (2, 4):
            joints[i] = r.unpack(bfmt * 4)
            weights[i] = r.unpack('ffff')
        elif wt == 3:  # SDEF: treat as BDEF2, drop C/R0/R1
            joints[i, :2] = r.unpack(bfmt + bfmt)
            w = r.f32()
            weights[i, :2] = (w, 1.0 - w)
            r.take(36)
        else:
            raise ValueError(f'unknown weight type {wt} at vertex {i}')
        edge_scale[i] = r.f32()

    # ---- faces
    nf = r.i32()
    vfmt = _UNSIGNED[vsz]
    tri = np.frombuffer(r.take(nf * vsz), dtype=np.dtype('<' + vfmt)).astype(np.uint32)
    triangles = tri.reshape(-1, 3)

    # ---- textures
    textures = [text() for _ in range(r.i32())]

    # ---- materials
    materials = []
    for _ in range(r.i32()):
        mname, mname_en = text(), text()
        diffuse = r.vec(4)
        specular = r.vec(3)
        spower = r.f32()
        ambient = r.vec(3)
        flags = r.u8()
        edge_color = r.vec(4)
        edge_size = r.f32()
        tex = idx(tsz)
        sph = idx(tsz)
        sph_mode = r.u8()
        toon_shared = r.u8() == 1
        toon = r.u8() if toon_shared else idx(tsz)
        memo = text()
        fcount = r.i32()
        materials.append(Material(mname, mname_en, diffuse, specular, spower, ambient, flags,
                                  edge_color, edge_size, tex, sph, sph_mode, toon_shared, toon,
                                  memo, fcount))

    # ---- bones
    bones = []
    for _ in range(r.i32()):
        bname, bname_en = text(), text()
        pos = r.vec(3)
        parent = idx(bsz)
        layer = r.i32()
        flags = r.unpack('H')[0]
        tail = idx(bsz) if flags & 0x0001 else r.vec(3)
        bone = Bone(bname, bname_en, pos, parent, layer, flags, tail)
        if flags & 0x0300:
            bone.inherit_parent = idx(bsz)
            bone.inherit_ratio = r.f32()
        if flags & 0x0400:
            bone.fixed_axis = r.vec(3)
        if flags & 0x0800:
            bone.local_x = r.vec(3)
            bone.local_z = r.vec(3)
        if flags & 0x2000:
            bone.external_key = r.i32()
        if flags & 0x0020:
            bone.ik_target = idx(bsz)
            r.i32()      # loop count
            r.f32()      # limit angle
            for _ in range(r.i32()):
                link = idx(bsz)
                if r.u8():
                    r.take(24)  # angle limits
                bone.ik_links.append(link)
        bones.append(bone)

    # ---- morphs
    morphs = []
    for _ in range(r.i32()):
        moname, moname_en = text(), text()
        panel = r.u8()
        mtype = r.u8()
        count = r.i32()
        if mtype == 0 or mtype == 9:          # group / flip: (morph index, ratio)
            ind = np.empty(count, np.int32)
            dat = np.empty(count, np.float32)
            for k in range(count):
                ind[k] = idx(mosz)
                dat[k] = r.f32()
        elif mtype == 1:                      # vertex: (vertex index, offset xyz)
            rec = np.dtype([('i', '<' + vfmt), ('d', '<f4', (3,))])
            arr = np.frombuffer(r.take(count * rec.itemsize), dtype=rec)
            ind = arr['i'].astype(np.int32)
            dat = arr['d'].astype(np.float32)
        elif mtype == 2:                      # bone: (bone index, translation xyz, rotation xyzw)
            rec = np.dtype([('i', '<' + bfmt), ('d', '<f4', (7,))])
            arr = np.frombuffer(r.take(count * rec.itemsize), dtype=rec)
            ind = arr['i'].astype(np.int32)
            dat = arr['d'].astype(np.float32)
        elif 3 <= mtype <= 7:                 # uv: (vertex index, xyzw)
            rec = np.dtype([('i', '<' + vfmt), ('d', '<f4', (4,))])
            arr = np.frombuffer(r.take(count * rec.itemsize), dtype=rec)
            ind = arr['i'].astype(np.int32)
            dat = arr['d'].astype(np.float32)
        elif mtype == 8:                      # material: (index, op, 28 floats)
            rec = np.dtype([('i', '<' + _SIGNED[msz]), ('op', 'u1'), ('d', '<f4', (28,))])
            arr = np.frombuffer(r.take(count * rec.itemsize), dtype=rec)
            ind = arr['i'].astype(np.int32)
            dat = np.concatenate([arr['op'][:, None].astype(np.float32), arr['d']], axis=1)
        elif mtype == 10:                     # impulse: (rigid index, flag, 6 floats)
            rec = np.dtype([('i', '<' + _SIGNED[rsz]), ('f', 'u1'), ('d', '<f4', (6,))])
            arr = np.frombuffer(r.take(count * rec.itemsize), dtype=rec)
            ind = arr['i'].astype(np.int32)
            dat = arr['d'].astype(np.float32)
        else:
            raise ValueError(f'unknown morph type {mtype} in {moname!r}')
        morphs.append(Morph(moname, moname_en, panel, mtype, ind, dat))

    return PMXModel(path, version, name, name_en, comment, comment_en,
                    positions, normals, uvs, add_uvs, weight_type, joints, weights,
                    edge_scale, triangles, textures, materials, bones, morphs)


if __name__ == '__main__':
    import sys
    for p in sys.argv[1:]:
        m = load(p)
        print(f'{p}: {m.name!r} v{m.version:.1f} V={m.vertex_count} F={len(m.triangles)} '
              f'mat={len(m.materials)} bones={len(m.bones)} morphs={len(m.morphs)} tex={len(m.textures)}')
        kinds = {}
        for mo in m.morphs:
            kinds[MORPH_TYPES[mo.type]] = kinds.get(MORPH_TYPES[mo.type], 0) + 1
        print('  morph kinds:', kinds)
        wt = np.bincount(m.weight_type, minlength=5)
        print('  weight types BDEF1/2/4/SDEF/QDEF:', wt.tolist())
