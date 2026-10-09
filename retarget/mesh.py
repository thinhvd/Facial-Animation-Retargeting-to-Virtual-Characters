"""Canonical in-memory representation of a face model.

Every importer (PMX, OBJ, Blender-exported glTF, ...) produces a
CanonicalMesh; the rigging core and the glTF writer only ever see this.

Conventions (match glTF 2.0):
  * right-handed, Y up, character faces +Z, units in metres
  * UV origin at the top-left of the image (no flip needed for glTF)
  * triangles counter-clockwise when seen from outside
  * bone rest positions are in model space; parent index -1 = root
  * morphs are dense (V, 3) displacements added to `positions`
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CanonicalImage:
    name: str
    data: bytes               # encoded file bytes
    mime: str                 # 'image/png' or 'image/jpeg'
    has_alpha: bool


@dataclass
class CanonicalMaterial:
    name: str
    base_color: np.ndarray    # rgba in [0, 1]
    image: int = -1           # index into CanonicalMesh.images
    double_sided: bool = False
    alpha_mode: str = 'OPAQUE'  # OPAQUE | MASK | BLEND
    unlit: bool = True
    extras: dict = field(default_factory=dict)


@dataclass
class CanonicalBone:
    name: str
    name_en: str
    parent: int
    position: np.ndarray      # (3,) rest position in model space
    extras: dict = field(default_factory=dict)


@dataclass
class CanonicalMorph:
    name: str
    displacement: np.ndarray  # (V, 3)
    extras: dict = field(default_factory=dict)   # e.g. panel, source index, name_en


@dataclass
class CanonicalMesh:
    name: str
    positions: np.ndarray                 # (V, 3) float32
    normals: np.ndarray                   # (V, 3) float32
    uvs: np.ndarray                       # (V, 2) float32
    triangles: np.ndarray                 # (F, 3) uint32
    primitives: list[tuple[int, int, int]]  # (material index, first triangle, last triangle exclusive)
    materials: list[CanonicalMaterial]
    images: list[CanonicalImage] = field(default_factory=list)
    bones: list[CanonicalBone] = field(default_factory=list)
    joints: np.ndarray | None = None      # (V, 4) int32, -1 = unused slot
    weights: np.ndarray | None = None     # (V, 4) float32
    morphs: list[CanonicalMorph] = field(default_factory=list)
    extras: dict = field(default_factory=dict)  # provenance: format, path, scale, ...

    @property
    def vertex_count(self) -> int:
        return len(self.positions)

    def morph_names(self) -> list[str]:
        return [m.name for m in self.morphs]

    def morph(self, name: str) -> CanonicalMorph | None:
        for m in self.morphs:
            if m.name == name:
                return m
        return None

    def bone(self, name: str) -> int:
        for i, b in enumerate(self.bones):
            if b.name == name or b.name_en == name:
                return i
        return -1

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.positions.min(0), self.positions.max(0)
