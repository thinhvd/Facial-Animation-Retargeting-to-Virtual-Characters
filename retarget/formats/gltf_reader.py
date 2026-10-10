"""Minimal glTF reader: JSON + typed accessor arrays.

Reads .glb (and .vrm, which is .glb with extensions) and .gltf with external
or data-URI buffers.  The convenience methods at the bottom assume the
canonical layout written by gltf_writer; formats/gltf_convert.py flattens
any other glTF into that layout.
"""
from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from urllib.parse import unquote

import numpy as np

_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NCOMP = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT2': 4, 'MAT3': 9, 'MAT4': 16}


def _uri_bytes(uri: str, base: Path) -> bytes:
    if uri.startswith('data:'):
        return base64.b64decode(uri.split(',', 1)[1])
    return (base / unquote(uri)).read_bytes()


class GLB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.json: dict = {}
        self.buffers: list[bytes] = []
        if self.path.suffix.lower() == '.gltf':
            self.json = json.loads(self.path.read_text(encoding='utf-8'))
            self.buffers = [_uri_bytes(b['uri'], self.path.parent) if 'uri' in b else b''
                            for b in self.json.get('buffers', [])]
        else:
            data = self.path.read_bytes()
            magic, version, length = struct.unpack_from('<III', data, 0)
            if magic != 0x46546C67:
                raise ValueError(f'{path}: not a glb')
            off = 12
            while off < length:
                clen, ctype = struct.unpack_from('<II', data, off)
                chunk = data[off + 8: off + 8 + clen]
                if ctype == 0x4E4F534A:
                    self.json = json.loads(chunk.decode('utf-8'))
                elif ctype == 0x004E4942:
                    self.buffers.append(chunk)
                off += 8 + clen
            # buffer 0 is the BIN chunk; later buffers (rare in .glb) are URIs
            for b in self.json.get('buffers', [])[1:]:
                self.buffers.append(_uri_bytes(b['uri'], self.path.parent) if 'uri' in b else b'')
        self.bin: bytes = self.buffers[0] if self.buffers else b''

    def _view(self, index: int) -> tuple[bytes, int]:
        bv = self.json['bufferViews'][index]
        return self.buffers[bv.get('buffer', 0)], bv.get('byteOffset', 0)

    def accessor(self, index: int) -> np.ndarray:
        """Accessor as a numpy array; `normalized` integer data comes back as float32."""
        acc = self.json['accessors'][index]
        dtype = _COMPONENT[acc['componentType']]
        n = _NCOMP[acc['type']]
        count = acc['count']
        out = np.zeros((count, n), dtype=dtype) if n > 1 else np.zeros(count, dtype=dtype)
        if 'bufferView' in acc:
            buf, start = self._view(acc['bufferView'])
            start += acc.get('byteOffset', 0)
            stride = self.json['bufferViews'][acc['bufferView']].get('byteStride')
            item = np.dtype(dtype).itemsize * n
            if stride and stride != item:
                raw = np.frombuffer(buf, dtype=np.uint8, count=stride * (count - 1) + item, offset=start)
                rows = np.lib.stride_tricks.as_strided(raw, shape=(count, item), strides=(stride, 1))
                out = np.ascontiguousarray(rows).view(dtype).reshape(count, n)
            else:
                out = np.frombuffer(buf, dtype=dtype, count=count * n, offset=start).reshape(count, n)
            if n == 1:
                out = out[:, 0]
            out = out.copy()
        if 'sparse' in acc:
            sp = acc['sparse']
            ibuf, ioff = self._view(sp['indices']['bufferView'])
            vbuf, voff = self._view(sp['values']['bufferView'])
            idx = np.frombuffer(ibuf, dtype=_COMPONENT[sp['indices']['componentType']], count=sp['count'],
                                offset=ioff + sp['indices'].get('byteOffset', 0))
            val = np.frombuffer(vbuf, dtype=dtype, count=sp['count'] * n,
                                offset=voff + sp['values'].get('byteOffset', 0)).reshape(sp['count'], n)
            out[idx] = val if n > 1 else val[:, 0]
        if acc.get('normalized') and np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            out = np.maximum(out.astype(np.float32) / info.max, -1.0)
        return out

    def image_bytes(self, index: int) -> tuple[bytes, str]:
        """Encoded bytes and mime type of images[index]."""
        im = self.json['images'][index]
        if 'bufferView' in im:
            buf, off = self._view(im['bufferView'])
            data = buf[off: off + self.json['bufferViews'][im['bufferView']]['byteLength']]
        else:
            data = _uri_bytes(im['uri'], self.path.parent)
        mime = im.get('mimeType') or ('image/png' if data[:4] == b'\x89PNG' else 'image/jpeg')
        return bytes(data), mime

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
