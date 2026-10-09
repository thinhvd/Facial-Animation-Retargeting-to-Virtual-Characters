"""Minimal .glb reader: JSON chunk + typed accessor arrays.

Enough for the adapter generator and the rigging core to get positions,
triangles and morph-target displacements back out of a canonical .glb
(including .vrm files, which are .glb with extensions).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NCOMP = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT2': 4, 'MAT3': 9, 'MAT4': 16}


class GLB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        data = self.path.read_bytes()
        magic, version, length = struct.unpack_from('<III', data, 0)
        if magic != 0x46546C67:
            raise ValueError(f'{path}: not a glb')
        off = 12
        self.json: dict = {}
        self.bin: bytes = b''
        while off < length:
            clen, ctype = struct.unpack_from('<II', data, off)
            chunk = data[off + 8: off + 8 + clen]
            if ctype == 0x4E4F534A:
                self.json = json.loads(chunk.decode('utf-8'))
            elif ctype == 0x004E4942:
                self.bin = chunk
            off += 8 + clen

    def accessor(self, index: int) -> np.ndarray:
        acc = self.json['accessors'][index]
        dtype = _COMPONENT[acc['componentType']]
        n = _NCOMP[acc['type']]
        count = acc['count']
        out = np.zeros((count, n), dtype=dtype) if n > 1 else np.zeros(count, dtype=dtype)
        if 'bufferView' in acc:
            bv = self.json['bufferViews'][acc['bufferView']]
            start = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
            stride = bv.get('byteStride')
            item = np.dtype(dtype).itemsize * n
            if stride and stride != item:
                raw = np.frombuffer(self.bin, dtype=np.uint8, count=stride * (count - 1) + item, offset=start)
                rows = np.lib.stride_tricks.as_strided(raw, shape=(count, item), strides=(stride, 1))
                out = np.ascontiguousarray(rows).view(dtype).reshape(count, n)
            else:
                out = np.frombuffer(self.bin, dtype=dtype, count=count * n, offset=start).reshape(count, n)
                if n == 1:
                    out = out[:, 0]
            out = out.copy()
        if 'sparse' in acc:
            sp = acc['sparse']
            ibv = self.json['bufferViews'][sp['indices']['bufferView']]
            vbv = self.json['bufferViews'][sp['values']['bufferView']]
            idx = np.frombuffer(self.bin, dtype=_COMPONENT[sp['indices']['componentType']], count=sp['count'],
                                offset=ibv.get('byteOffset', 0) + sp['indices'].get('byteOffset', 0))
            val = np.frombuffer(self.bin, dtype=dtype, count=sp['count'] * n,
                                offset=vbv.get('byteOffset', 0) + sp['values'].get('byteOffset', 0)).reshape(sp['count'], n)
            out[idx] = val if n > 1 else val[:, 0]
        return out

    # ---- convenience for the single-mesh canonical layout
    def meshes(self):
        return self.json.get('meshes', [])

    def mesh_morphs(self, mesh_index: int = 0) -> dict[str, np.ndarray]:
        """Morph name -> (V, 3) displacement for one mesh (first primitive's targets)."""
        mesh = self.meshes()[mesh_index]
        names = mesh.get('extras', {}).get('targetNames') or [f'target_{i}' for i in range(len(mesh['primitives'][0].get('targets', [])))]
        targets = mesh['primitives'][0].get('targets', [])
        return {name: self.accessor(t['POSITION']) for name, t in zip(names, targets) if 'POSITION' in t}

    def mesh_positions(self, mesh_index: int = 0) -> np.ndarray:
        prim = self.meshes()[mesh_index]['primitives'][0]
        return self.accessor(prim['attributes']['POSITION'])

    def mesh_triangles(self, mesh_index: int = 0) -> np.ndarray:
        tris = []
        for prim in self.meshes()[mesh_index]['primitives']:
            if 'indices' in prim:
                tris.append(self.accessor(prim['indices']).reshape(-1, 3).astype(np.uint32))
        return np.concatenate(tris) if tris else np.zeros((0, 3), np.uint32)

    def node_names(self) -> list[str]:
        return [n.get('name', '') for n in self.json.get('nodes', [])]
