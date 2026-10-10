"""Convert a face model to the canonical .glb.

    python -m retarget.convert NodeServer/models/headmodel/HeadA.pmx -o web/models/HeadA.glb

Supported today: .pmx (pure Python).  .glb/.gltf/.vrm are copied through.
Other formats (.fbx, .obj, .blend) need Blender; see --blender.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .formats import gltf_writer, pmx, pmx_convert


def convert(src: Path, dst: Path, scale: float | None = None, textures: bool = True,
            up: str = '+y', front: str = '-z') -> Path:
    ext = src.suffix.lower()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if ext == '.pmx':
        model = pmx.load(src)
        mesh = pmx_convert.to_canonical(model, scale or pmx_convert.MMD_UNIT_METRES, textures, up, front)
        gltf_writer.write_glb(mesh, dst)
        lo, hi = mesh.bounds()
        print(f'{src.name}: V={mesh.vertex_count} F={len(mesh.triangles)} prims={len(mesh.primitives)} '
              f'images={len(mesh.images)} bones={len(mesh.bones)} morphs={len(mesh.morphs)}')
        print(f'  bounds (m): min={lo.round(3).tolist()} max={hi.round(3).tolist()}')
        print(f'  -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)')
        return dst
    if ext in ('.glb', '.vrm'):
        shutil.copyfile(src, dst)
        print(f'{src.name}: copied through -> {dst}')
        return dst
    if ext == '.obj':
        from .formats import obj_convert
        mesh = obj_convert.to_canonical(src, scale or 1.0, up if up != '+y' else '+y',
                                        front if front != '-z' else '+z')
        gltf_writer.write_glb(mesh, dst)
        lo, hi = mesh.bounds()
        print(f'{src.name}: V={mesh.vertex_count} F={len(mesh.triangles)} (obj, lit, no rig)')
        print(f'  bounds (m): min={lo.round(3).tolist()} max={hi.round(3).tolist()}')
        print(f'  -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)')
        return dst
    raise SystemExit(f'{src}: format {ext!r} not supported by the pure-Python path; '
                     f'use Blender (tools/blender_export.py) to produce a .glb first')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', type=Path)
    ap.add_argument('-o', '--out', type=Path, help='output .glb (default: web/models/<name>.glb)')
    ap.add_argument('--scale', type=float, default=None, help='source unit -> metres (pmx default 0.08)')
    ap.add_argument('--no-textures', action='store_true')
    ap.add_argument('--up', default='+y', help="source up axis, e.g. +y (MMD default) or -z")
    ap.add_argument('--front', default='-z', help="source facing axis, e.g. -z (MMD default) or -y")
    args = ap.parse_args(argv)
    out = args.out or Path('web/models') / (args.src.stem + '.glb')
    convert(args.src, out, args.scale, not args.no_textures, args.up, args.front)


if __name__ == '__main__':
    sys.exit(main())
