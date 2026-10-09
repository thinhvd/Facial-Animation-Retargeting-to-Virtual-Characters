"""Write a CanonicalMesh as a single binary glTF (.glb).

Hand-rolled on purpose: the file is small, we need full control over
morph targets, skins and extras, and it keeps the base pipeline free of
extra dependencies.  Output is readable by three.js GLTFLoader, which
exposes `mesh.extras.targetNames` as `morphTargetDictionary`.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from ..mesh import CanonicalMesh

FLOAT, UBYTE, USHORT, UINT = 5126, 5121, 5123, 5125
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
_TYPES = {1: 'SCALAR', 2: 'VEC2', 3: 'VEC3', 4: 'VEC4', 16: 'MAT4'}


class _Builder:
    def __init__(self):
        self.blob = bytearray()
        self.buffer_views = []
        self.accessors = []

    def view(self, data: bytes, target: int | None = None, stride: int | None = None) -> int:
        while len(self.blob) % 4:
            self.blob.append(0)
        bv = {'buffer': 0, 'byteOffset': len(self.blob), 'byteLength': len(data)}
        if target is not None:
            bv['target'] = target
        if stride is not None:
            bv['byteStride'] = stride
        self.blob.extend(data)
        self.buffer_views.append(bv)
        return len(self.buffer_views) - 1

    def accessor(self, arr: np.ndarray, component: int, target: int | None = None,
                 minmax: bool = False, normalized: bool = False) -> int:
        arr = np.ascontiguousarray(arr)
        ncomp = 1 if arr.ndim == 1 else int(np.prod(arr.shape[1:]))
        acc = {
            'bufferView': self.view(arr.tobytes(), target),
            'componentType': component,
            'count': int(arr.shape[0]),
            'type': _TYPES[ncomp],
        }
        if normalized:
            acc['normalized'] = True
        if minmax:
            flat = arr.reshape(arr.shape[0], -1)
            acc['min'] = [float(v) for v in flat.min(0)]
            acc['max'] = [float(v) for v in flat.max(0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1


def clean_skin_weights(joints_in: np.ndarray, weights_in: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Make (V, 4) joints/weights valid for glTF: no duplicate joint per vertex,
    joint 0 wherever the weight is 0, no negative weights, rows summing to 1."""
    joints = joints_in.astype(np.int64).copy()
    weights = weights_in.astype(np.float64).copy()
    weights[joints < 0] = 0.0
    joints[joints < 0] = 0
    weights = np.clip(weights, 0.0, None)
    # merge duplicate joints (e.g. BDEF2 with the same bone twice): add into the first slot
    for a in range(4):
        for c in range(a + 1, 4):
            dup = (joints[:, a] == joints[:, c]) & (weights[:, c] > 0)
            weights[dup, a] += weights[dup, c]
            weights[dup, c] = 0.0
    s = weights.sum(1, keepdims=True)
    bad = s[:, 0] <= 1e-8
    weights[bad] = np.array([1, 0, 0, 0])
    joints[bad] = 0
    s[bad] = 1.0
    weights /= s
    joints[weights <= 0] = 0
    return joints.astype(np.uint16), weights.astype(np.float32)


def write_glb(mesh: CanonicalMesh, path: str | Path) -> Path:
    path = Path(path)
    b = _Builder()
    V = mesh.vertex_count

    # ---- vertex attributes (shared by all primitives)
    attributes = {
        'POSITION': b.accessor(mesh.positions.astype(np.float32), FLOAT, ARRAY_BUFFER, minmax=True),
        'NORMAL': b.accessor(mesh.normals.astype(np.float32), FLOAT, ARRAY_BUFFER),
        'TEXCOORD_0': b.accessor(mesh.uvs.astype(np.float32), FLOAT, ARRAY_BUFFER),
    }
    has_skin = bool(mesh.bones) and mesh.joints is not None
    if has_skin:
        joints, weights = clean_skin_weights(mesh.joints, mesh.weights)
        attributes['JOINTS_0'] = b.accessor(joints, USHORT, ARRAY_BUFFER)
        attributes['WEIGHTS_0'] = b.accessor(weights, FLOAT, ARRAY_BUFFER)

    # ---- morph targets (shared accessors, referenced by every primitive)
    targets = []
    for m in mesh.morphs:
        targets.append({'POSITION': b.accessor(m.displacement.astype(np.float32), FLOAT,
                                               ARRAY_BUFFER, minmax=True)})

    # ---- primitives
    index_type = UINT if V > 65535 else USHORT
    index_dtype = np.uint32 if V > 65535 else np.uint16
    primitives = []
    for mat_index, t0, t1 in mesh.primitives:
        if t1 <= t0:
            continue
        idx = mesh.triangles[t0:t1].reshape(-1).astype(index_dtype)
        prim = {
            'attributes': attributes,
            'indices': b.accessor(idx, index_type, ELEMENT_ARRAY_BUFFER),
            'material': mat_index,
            'mode': 4,
        }
        if targets:
            prim['targets'] = targets
        primitives.append(prim)

    gltf_mesh = {'name': mesh.name, 'primitives': primitives}
    if targets:
        gltf_mesh['weights'] = [0.0] * len(targets)
        gltf_mesh['extras'] = {
            'targetNames': [m.name for m in mesh.morphs],
            'targetMeta': [m.extras for m in mesh.morphs],
        }

    # ---- images / textures / materials
    images, textures = [], []
    for img in mesh.images:
        images.append({'name': img.name, 'mimeType': img.mime, 'bufferView': b.view(img.data)})
        textures.append({'sampler': 0, 'source': len(images) - 1})
    samplers = [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 10497, 'wrapT': 10497}]

    used_unlit = False
    materials = []
    for mat in mesh.materials:
        pbr = {
            'baseColorFactor': [float(c) for c in mat.base_color],
            'metallicFactor': 0.0,
            'roughnessFactor': 1.0,
        }
        if 0 <= mat.image < len(textures):
            pbr['baseColorTexture'] = {'index': mat.image}
        gm = {'name': mat.name, 'pbrMetallicRoughness': pbr, 'doubleSided': bool(mat.double_sided),
              'alphaMode': mat.alpha_mode}
        if mat.alpha_mode == 'MASK':
            gm['alphaCutoff'] = 0.5
        if mat.unlit:
            gm['extensions'] = {'KHR_materials_unlit': {}}
            used_unlit = True
        if mat.extras:
            gm['extras'] = mat.extras
        materials.append(gm)

    # ---- nodes: mesh node + bone hierarchy
    nodes = [{'name': mesh.name, 'mesh': 0}]
    scene_nodes = [0]
    skins = []
    if has_skin:
        base = len(nodes)
        ibm = np.zeros((len(mesh.bones), 4, 4), np.float32)
        for i, bone in enumerate(mesh.bones):
            p = bone.position.astype(np.float32)
            local = p - mesh.bones[bone.parent].position if bone.parent >= 0 else p
            node = {'name': bone.name, 'translation': [float(v) for v in local]}
            if bone.name_en or bone.extras:
                node['extras'] = {'name_en': bone.name_en, **bone.extras}
            nodes.append(node)
            m = np.eye(4, dtype=np.float32)
            m[:3, 3] = -p
            ibm[i] = m.T  # column-major
        for i, bone in enumerate(mesh.bones):
            if bone.parent >= 0:
                nodes[base + bone.parent].setdefault('children', []).append(base + i)
            else:
                scene_nodes.append(base + i)
        roots = [base + i for i, bn in enumerate(mesh.bones) if bn.parent < 0]
        skins.append({
            'name': mesh.name + '_skin',
            'joints': [base + i for i in range(len(mesh.bones))],
            'inverseBindMatrices': b.accessor(ibm.reshape(len(mesh.bones), 16), FLOAT),
            'skeleton': roots[0],
        })
        nodes[0]['skin'] = 0

    doc = {
        'asset': {'version': '2.0', 'generator': 'retarget.formats.gltf_writer'},
        'scene': 0,
        'scenes': [{'nodes': scene_nodes}],
        'nodes': nodes,
        'meshes': [gltf_mesh],
        'materials': materials,
        'accessors': b.accessors,
        'bufferViews': b.buffer_views,
        'buffers': [{'byteLength': len(b.blob)}],
    }
    if images:
        doc['images'] = images
        doc['textures'] = textures
        doc['samplers'] = samplers
    if skins:
        doc['skins'] = skins
    if used_unlit:
        doc['extensionsUsed'] = ['KHR_materials_unlit']
    if mesh.extras:
        doc['asset']['extras'] = _jsonable(mesh.extras)

    json_bytes = json.dumps(doc, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    while len(json_bytes) % 4:
        json_bytes += b' '
    blob = bytes(b.blob)
    while len(blob) % 4:
        blob += b'\0'
    total = 12 + 8 + len(json_bytes) + 8 + len(blob)
    with open(path, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, total))
        f.write(struct.pack('<II', len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        f.write(struct.pack('<II', len(blob), 0x004E4942))
        f.write(blob)
    return path


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def read_glb_json(path: str | Path) -> dict:
    """Return the JSON chunk of a .glb (for quick inspection / tests)."""
    with open(path, 'rb') as f:
        magic, version, length = struct.unpack('<III', f.read(12))
        if magic != 0x46546C67:
            raise ValueError('not a glb')
        chunk_len, chunk_type = struct.unpack('<II', f.read(8))
        return json.loads(f.read(chunk_len).decode('utf-8'))
