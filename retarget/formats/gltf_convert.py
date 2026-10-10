"""Any glTF 2.0 (.glb / .gltf) -> CanonicalMesh.

The canonical layout is one mesh whose primitives share one vertex buffer,
bones with rest positions in model space, and dense morphs.  Files from
Blender, Sketchfab, ReadyPlayerMe, ... have many meshes, per-primitive
buffers, node transforms, and skins whose bind pose is not the identity.
This flattens all of it to the rest pose as three.js draws it:

  * skinned primitives: v' = sum_i w_i * world(joint_i) @ inverseBind_i @ v
  * other primitives:   v' = world(node) @ v, and if the node hangs below a
    joint (glasses parented to the head) the vertices are bound to that joint
  * morph targets are merged by name across meshes (a 'blink' on the face and
    one on the lashes become one canonical morph)
  * all skins become one skeleton (joint nodes in first-seen order)
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from ..mesh import CanonicalBone, CanonicalImage, CanonicalMaterial, CanonicalMesh, CanonicalMorph
from .gltf_reader import GLB

# compressed geometry this reader cannot decode; convert.py sends these through Blender
COMPRESSED = ('KHR_draco_mesh_compression', 'EXT_meshopt_compression')


def _node_matrix(node: dict) -> np.ndarray:
    if 'matrix' in node:
        return np.array(node['matrix'], np.float64).reshape(4, 4).T     # glTF is column-major
    t = np.array(node.get('translation', [0, 0, 0]), np.float64)
    x, y, z, w = node.get('rotation', [0, 0, 0, 1])
    s = np.array(node.get('scale', [1, 1, 1]), np.float64)
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    m = np.eye(4)
    m[:3, :3] = r * s
    m[:3, 3] = t
    return m


def _scene_graph(doc: dict) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """World matrix and parent of every node reachable from the default scene."""
    nodes = doc.get('nodes', [])
    scenes = doc.get('scenes') or [{'nodes': [i for i in range(len(nodes))
                                              if not any(i in n.get('children', []) for n in nodes)]}]
    roots = scenes[doc.get('scene', 0)].get('nodes', [])
    world, parent = {}, {}
    stack = [(r, np.eye(4), -1) for r in roots]
    while stack:
        nid, pm, par = stack.pop()
        world[nid] = pm @ _node_matrix(nodes[nid])
        parent[nid] = par
        stack.extend((c, world[nid], nid) for c in nodes[nid].get('children', []))
    return world, parent


def _triangles(glb: GLB, prim: dict, nverts: int) -> np.ndarray | None:
    mode = prim.get('mode', 4)
    idx = glb.accessor(prim['indices']).astype(np.int64) if 'indices' in prim else np.arange(nverts)
    if mode == 4:
        return idx.reshape(-1, 3)
    if mode == 5:   # strip: alternate winding
        tris = np.stack([idx[:-2], idx[1:-1], idx[2:]], 1)
        tris[1::2] = tris[1::2][:, [1, 0, 2]]
        return tris
    if mode == 6:   # fan
        return np.stack([np.full(len(idx) - 2, idx[0]), idx[1:-1], idx[2:]], 1)
    return None     # points / lines


def _image(glb: GLB, index: int) -> CanonicalImage | None:
    from PIL import Image
    data, mime = glb.image_bytes(index)
    name = glb.json['images'][index].get('name') or f'image_{index}'
    try:
        im = Image.open(io.BytesIO(data))
        has_alpha = im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info)
        if mime not in ('image/png', 'image/jpeg'):       # webp etc.: re-encode
            buf = io.BytesIO()
            im.convert('RGBA' if has_alpha else 'RGB').save(buf, format='PNG')
            data, mime = buf.getvalue(), 'image/png'
    except Exception:
        return None   # KTX2 / basisu and anything PIL cannot read: material keeps its colour
    return CanonicalImage(name, data, mime, has_alpha)


def _texture_source(doc: dict, tex_index: int) -> int:
    tex = doc['textures'][tex_index]
    if 'source' in tex:
        return tex['source']
    for ext in tex.get('extensions', {}).values():
        if 'source' in ext:
            return ext['source']
    return -1


def is_compressed(path: str | Path) -> bool:
    doc = GLB(path).json
    return bool(set(doc.get('extensionsRequired', [])) & set(COMPRESSED))


def to_canonical(path: str | Path) -> CanonicalMesh:
    path = Path(path)
    glb = GLB(path)
    doc = glb.json
    required = set(doc.get('extensionsRequired', [])) & set(COMPRESSED)
    if required:
        raise ValueError(f'{path.name}: compressed geometry ({", ".join(sorted(required))}); convert via Blender')
    world, parent = _scene_graph(doc)
    nodes = doc.get('nodes', [])

    # ---- images and materials (same order as the file; one extra default at the end)
    images, image_map = [], {}
    for i in range(len(doc.get('images', []))):
        im = _image(glb, i)
        if im is not None:
            image_map[i] = len(images)
            images.append(im)
    materials = []
    for m in doc.get('materials', []):
        pbr = m.get('pbrMetallicRoughness', {})
        img = -1
        if 'baseColorTexture' in pbr:
            img = image_map.get(_texture_source(doc, pbr['baseColorTexture']['index']), -1)
        materials.append(CanonicalMaterial(m.get('name', ''), np.array(pbr.get('baseColorFactor', [1, 1, 1, 1]), np.float32),
                                           img, m.get('doubleSided', False), m.get('alphaMode', 'OPAQUE'),
                                           'KHR_materials_unlit' in m.get('extensions', {}), m.get('extras', {})))
    default_material = len(materials)
    materials.append(CanonicalMaterial('default', np.array([0.8, 0.8, 0.8, 1], np.float32), unlit=False))

    # ---- skeleton: every joint of every skin, in first-seen order
    joint_nodes: list[int] = []
    for skin in doc.get('skins', []):
        for j in skin['joints']:
            if j not in joint_nodes:
                joint_nodes.append(j)
    bone_of = {nid: k for k, nid in enumerate(joint_nodes)}

    def bone_above(nid: int) -> int:
        p = parent.get(nid, -1)
        while p != -1 and p not in bone_of:
            p = parent.get(p, -1)
        return bone_of.get(p, -1)

    bones = [CanonicalBone(nodes[nid].get('name', f'joint_{k}'), '', bone_above(nid),
                           world[nid][:3, 3].astype(np.float32) if nid in world else np.zeros(3, np.float32))
             for k, nid in enumerate(joint_nodes)]
    root_bone = next((k for k, b in enumerate(bones) if b.parent < 0), 0)

    # ---- geometry
    P, N, UV, J, W, T = [], [], [], [], [], []
    prims: list[tuple[int, int, int]] = []
    morph_parts: dict[str, list[tuple[int, np.ndarray]]] = {}
    nv = nt = 0
    shared: dict[tuple, tuple[int, bool]] = {}   # same accessors -> one vertex block (offset, mirrored)
    for nid, node in enumerate(nodes):
        if 'mesh' not in node or nid not in world:
            continue
        mesh = doc['meshes'][node['mesh']]
        names = mesh.get('extras', {}).get('targetNames') or []
        skin = doc['skins'][node['skin']] if 'skin' in node else None
        if skin is not None:
            ibm = (glb.accessor(skin['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)
                   if 'inverseBindMatrices' in skin else np.tile(np.eye(4), (len(skin['joints']), 1, 1)))
            jm = np.stack([world.get(j, np.eye(4)) for j in skin['joints']]) @ ibm
        for prim in mesh['primitives']:
            a = prim['attributes']
            if 'POSITION' not in a:
                continue
            key = (nid, tuple(sorted(a.items())), tuple(t.get('POSITION') for t in prim.get('targets', [])))
            count = doc['accessors'][a['POSITION']]['count']
            tris = _triangles(glb, prim, count)
            if tris is None or not len(tris):
                continue
            mat = prim.get('material', default_material)
            if key in shared:
                start, flipped = shared[key]
                T.append((tris[:, [0, 2, 1]] if flipped else tris) + start)
                prims.append((mat, nt, nt + len(tris)))
                nt += len(tris)
                continue
            pos = glb.accessor(a['POSITION']).astype(np.float64)
            nrm = glb.accessor(a['NORMAL']).astype(np.float64) if 'NORMAL' in a else np.zeros_like(pos)
            uv = glb.accessor(a['TEXCOORD_0']).astype(np.float32) if 'TEXCOORD_0' in a else np.zeros((len(pos), 2), np.float32)
            if skin is not None and 'JOINTS_0' in a and 'WEIGHTS_0' in a:
                jl = glb.accessor(a['JOINTS_0']).astype(np.int64)
                wl = glb.accessor(a['WEIGHTS_0']).astype(np.float64)
                wl[wl.sum(1) < 1e-8, 0] = 1.0          # unweighted vertex: would collapse to the origin
                wl = wl / wl.sum(1, keepdims=True)
                M = np.einsum('vi,vijk->vjk', wl, jm[jl])                 # (V, 4, 4) rest skinning matrix
                joints = np.array([bone_of[skin['joints'][j]] for j in range(len(skin['joints']))])[jl]
                weights = wl
                flip = False
            else:
                M = np.broadcast_to(world[nid], (len(pos), 4, 4))
                b = bone_above(nid)
                joints = np.full((len(pos), 4), -1, np.int64)
                joints[:, 0] = b if b >= 0 else root_bone
                weights = np.zeros((len(pos), 4))
                weights[:, 0] = 1.0
                flip = np.linalg.det(world[nid][:3, :3]) < 0
            L = M[:, :3, :3]
            p = np.einsum('vij,vj->vi', L, pos) + M[:, :3, 3]
            n = np.einsum('vij,vj->vi', L, nrm)
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            for k, t in enumerate(prim.get('targets', [])):
                if 'POSITION' in t:
                    name = names[k] if k < len(names) else f'target_{k}'
                    d = np.einsum('vij,vj->vi', L, glb.accessor(t['POSITION']).astype(np.float64))
                    morph_parts.setdefault(name, []).append((nv, d.astype(np.float32)))
            shared[key] = (nv, flip)
            if flip:
                tris = tris[:, [0, 2, 1]]
            P.append(p); N.append(n); UV.append(uv); J.append(joints); W.append(weights)
            T.append(tris + nv)
            prims.append((mat, nt, nt + len(tris)))
            nv += len(pos)
            nt += len(tris)
    if not P:
        raise ValueError(f'{path.name}: no triangle meshes in the default scene')

    positions = np.concatenate(P).astype(np.float32)
    triangles = np.concatenate(T).astype(np.uint32)
    normals = np.concatenate(N).astype(np.float32)
    missing = np.linalg.norm(normals, axis=1) < 0.5
    if missing.any():
        from .obj_convert import vertex_normals
        normals[missing] = vertex_normals(positions, triangles)[missing]
    morphs = []
    for name, parts in morph_parts.items():
        d = np.zeros_like(positions)
        for start, disp in parts:
            d[start: start + len(disp)] = disp
        morphs.append(CanonicalMorph(name, d))
    has_skin = bool(bones)
    out = CanonicalMesh(
        name=path.stem, positions=positions, normals=normals,
        uvs=np.concatenate(UV).astype(np.float32), triangles=triangles,
        primitives=prims, materials=materials, images=images, bones=bones,
        joints=np.concatenate(J).astype(np.int32) if has_skin else None,
        weights=np.concatenate(W).astype(np.float32) if has_skin else None,
        morphs=morphs,
        extras={'source_format': path.suffix.lower().lstrip('.'), 'source_path': str(path),
                'generator': doc.get('asset', {}).get('generator', '')})
    return out

