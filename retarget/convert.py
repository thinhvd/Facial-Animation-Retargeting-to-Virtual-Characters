"""Convert a face model to the canonical .glb.

    python -m retarget.convert NodeServer/models/headmodel/HeadA.pmx -o web/models/HeadA.glb

  .pmx, .obj          pure Python
  .glb, .gltf         pure Python, flattened to one mesh (formats/gltf_convert.py)
  .fbx, .blend, .dae  through Blender (formats/blender.py), then as .glb
  .vrm                copied through (the writer would drop the VRM extensions)

--up/--front default to each format's convention: MMD faces -Z, glTF and OBJ face +Z.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from .formats import blender, gltf_writer, orient, pmx, pmx_convert


def _report(src: Path, dst: Path, mesh, note: str = '') -> None:
    lo, hi = mesh.bounds()
    print(f'{src.name}: V={mesh.vertex_count} F={len(mesh.triangles)} prims={len(mesh.primitives)} '
          f'images={len(mesh.images)} bones={len(mesh.bones)} morphs={len(mesh.morphs)}{note}')
    print(f'  bounds (m): min={lo.round(3).tolist()} max={hi.round(3).tolist()}')
    height = float(hi[1] - lo[1])
    if not 0.1 < height < 4.0:
        print(f'  WARNING: model is {height:.3f} m tall; pass --scale so a head is ~0.2-0.3 m '
              '(registration thresholds are metric)')
    print(f'  -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)')


def _auto_orient(mesh, axes_given: bool, scale_given: bool):
    """Fix axes / units guessed from the skeleton unless the user set them."""
    if axes_given and scale_given:
        return mesh
    R, k, note = orient.guess(mesh)
    if axes_given:
        R = np.eye(3)
    if scale_given:
        k = 1.0
    if note and (not np.allclose(R, np.eye(3)) or k != 1.0):
        print(f'  auto-orient: {note}  (override with --up/--front/--scale)')
        orient.apply(mesh, R, k)
    return mesh


def convert(src: Path, dst: Path, scale: float | None = None, textures: bool = True,
            up: str | None = None, front: str | None = None) -> Path:
    ext = src.suffix.lower()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if ext == '.pmx':
        model = pmx.load(src)
        mesh = pmx_convert.to_canonical(model, scale or pmx_convert.MMD_UNIT_METRES, textures,
                                        up or '+y', front or '-z')
        _auto_orient(mesh, bool(up or front), scale is not None)
        gltf_writer.write_glb(mesh, dst)
        _report(src, dst, mesh)
        return dst
    if ext == '.vrm':
        shutil.copyfile(src, dst)
        print(f'{src.name}: copied through -> {dst}')
        return dst
    if ext == '.obj':
        from .formats import obj_convert
        mesh = obj_convert.to_canonical(src, scale or 1.0, up or '+y', front or '+z')
        gltf_writer.write_glb(mesh, dst)
        _report(src, dst, mesh, ' (obj, lit, no rig)')
        return dst
    if ext in ('.glb', '.gltf') or ext in blender.FORMATS:
        from .formats import gltf_convert, obj_convert
        with tempfile.TemporaryDirectory() as tmp:
            gltf = src
            if ext in blender.FORMATS or gltf_convert.is_compressed(src):
                print(f'{src.name}: converting with Blender')
                gltf = blender.export_glb(src, Path(tmp) / (src.stem + '.glb'))
            mesh = gltf_convert.to_canonical(gltf)
        mesh.name = src.stem
        if up or front or scale:
            orient.apply(mesh, obj_convert._axis_matrix_rh(up or '+y', front or '+z'), scale or 1.0)
        _auto_orient(mesh, bool(up or front), scale is not None)
        gltf_writer.write_glb(mesh, dst)
        _report(src, dst, mesh)
        return dst
    raise SystemExit(f'{src}: format {ext!r} is not supported (pmx, obj, glb, gltf, vrm; fbx, blend, dae via Blender)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', type=Path)
    ap.add_argument('-o', '--out', type=Path, help='output .glb (default: web/models/<name>.glb)')
    ap.add_argument('--scale', type=float, default=None, help='source unit -> metres (pmx default 0.08)')
    ap.add_argument('--no-textures', action='store_true')
    ap.add_argument('--up', default=None, help='source up axis, e.g. -z (default: the format\'s, +y)')
    ap.add_argument('--front', default=None, help='source facing axis, e.g. -y (default: -z for MMD, +z otherwise)')
    args = ap.parse_args(argv)
    out = args.out or Path('web/models') / (args.src.stem + '.glb')
    convert(args.src, out, args.scale, not args.no_textures, args.up, args.front)


if __name__ == '__main__':
    sys.exit(main())
