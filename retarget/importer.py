"""One command: any supported model file -> ready-to-drive asset in web/models/.

    python -m retarget.importer path/to/model.pmx [--scale 10 --up=-z --front=-y]
    python -m retarget.importer path/to/avatar.vrm
    python -m retarget.importer scan.obj --rig            # force blendshape generation

Steps
  1. convert   -> web/models/<name>.glb  (PMX / OBJ natively, glb / vrm copied)
  2. adapter   -> <name>.adapter.json, coverage report
  3. if coverage of the mouth/brow/eye regions is low and --no-rig is not given:
     landmarks -> generate (ICT template) -> <name>.rigged.glb + adapter
Prints which file to pick in the viewer.  The server must be running for step 3
(landmarks are taken from a headless render of the viewer).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapter import build_adapter
from .convert import convert

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'web' / 'models'


def write_adapter(glb: Path) -> dict:
    adapter = build_adapter(glb)
    glb.with_suffix('.adapter.json').write_text(json.dumps(adapter, ensure_ascii=False, indent=1), encoding='utf-8')
    return adapter


def needs_rig(adapter: dict, threshold: float = 0.6) -> bool:
    s = adapter['coverageSummary']
    face_regions = ('brow', 'mouth')
    covered = sum(s['byRegion'][r]['covered'] for r in face_regions)
    total = sum(s['byRegion'][r]['total'] for r in face_regions)
    return covered < threshold * total


def import_model(src: Path, scale: float | None, up: str, front: str, rig: bool | None, log=print) -> Path:
    name = src.stem
    glb = MODELS / f'{name}.glb' if src.suffix.lower() != '.vrm' else MODELS / f'{name}.vrm'
    log(f'[1/3] convert {src.name}')
    convert(src, glb, scale, True, up, front)
    log(f'[2/3] adapter')
    adapter = write_adapter(glb)
    s = adapter['coverageSummary']
    log(f'      coverage {s["covered"]}/{s["total"]}  bones={ {k: v for k, v in adapter["bones"].items() if v} }')
    do_rig = rig if rig is not None else (needs_rig(adapter) and adapter['vrm'] is None)
    if not do_rig:
        log(f'[3/3] rig: skipped ({"VRM" if adapter["vrm"] else "coverage sufficient"})')
        log(f'done -> {glb.name} (open in the viewer)')
        return glb
    log(f'[3/3] rig: landmarks + ICT template transfer')
    from .rig.generate import generate
    from .rig.landmarks import landmarks_3d
    if not (MODELS / 'ict_neutral.landmarks.json').exists():
        raise SystemExit('template landmarks missing: convert generic_neutral_mesh.obj to web/models/ict_neutral.glb '
                         'and run: python -m retarget.rig.landmarks ict_neutral')
    if (MODELS / f'{name}.landmarks.json').exists():
        log('      landmarks: reusing existing file')
    else:
        landmarks_3d(name)
    generate(name, log=lambda m: log('      ' + m))
    rigged = MODELS / f'{name}.rigged.glb'
    adapter2 = write_adapter(rigged)
    s2 = adapter2['coverageSummary']
    log(f'      coverage {s["covered"]}/{s["total"]} -> {s2["covered"]}/{s2["total"]} '
        f'(sources: {s2["bySource"]})')
    log(f'done -> {rigged.name} (open in the viewer)')
    return rigged


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', type=Path)
    ap.add_argument('--scale', type=float, default=None)
    ap.add_argument('--up', default='+y')
    ap.add_argument('--front', default='-z')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--rig', action='store_true', help='always generate blendshapes')
    g.add_argument('--no-rig', action='store_true', help='never generate blendshapes')
    args = ap.parse_args(argv)
    rig = True if args.rig else (False if args.no_rig else None)
    import_model(args.src, args.scale, args.up, args.front, rig)


if __name__ == '__main__':
    sys.exit(main())
