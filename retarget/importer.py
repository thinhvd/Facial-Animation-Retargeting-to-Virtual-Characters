"""One command: any supported model file -> ready-to-drive asset in web/models/.

    python -m retarget.importer path/to/model.pmx [--scale 10 --up=-z --front=-y]
    python -m retarget.importer path/to/avatar.vrm
    python -m retarget.importer scan.obj --rig            # force blendshape generation
    python -m retarget.importer --existing model4         # rig a model already in web/models
                                                          # (what the web UI's Generate button runs)

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
import os
import sys
import time
from contextlib import contextmanager
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


def import_model(src: Path, scale: float | None, up: str | None, front: str | None, rig: bool | None,
                 log=print) -> Path:
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
    rigged = rig_existing(name, log)
    if rigged is None:
        log(f'done -> {glb.name}: drives its own morphs and bones ({s["covered"]}/{s["total"]}).')
        log(f'      To generate the rest, pick ~20 points by hand at '
            f'http://127.0.0.1:6790/landmarks.html?model={name}')
        log(f'      then press Generate there (or run this command again; it reuses {name}.landmarks.json).')
        return glb
    return rigged


@contextmanager
def rig_lock(name: str):
    """One rig job per model across processes (web UI job and a shell importer would both write
    <name>.rigged.glb).  A lock left by a dead process is taken over."""
    lock = MODELS / f'{name}.rig.lock'
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                pid = int(lock.read_text() or 0)
                os.kill(pid, 0)
            except (ValueError, ProcessLookupError, FileNotFoundError):
                lock.unlink(missing_ok=True)          # stale
                continue
            raise SystemExit(f'{name}: another rig job is running (pid {pid})')
    else:
        raise SystemExit(f'{name}: cannot take {lock}')
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def rig_existing(name: str, log=print) -> Path | None:
    """landmarks (reuse, else MediaPipe) -> generate -> adapter for web/models/<name>.glb.

    Returns the rigged .glb, or None when MediaPipe found no face (pick landmarks by hand);
    that outcome is remembered in <name>.landmarks_failed.json for the web UI."""
    from .rig.generate import generate
    from .rig.landmarks import landmarks_3d
    if not (MODELS / 'ict_neutral.landmarks.json').exists():
        raise SystemExit('template landmarks missing: convert generic_neutral_mesh.obj to web/models/ict_neutral.glb '
                         'and run: python -m retarget.rig.landmarks ict_neutral')
    if not (MODELS / f'{name}.glb').exists():
        raise SystemExit(f'{name}.glb not in web/models (import the source file first)')
    failed = MODELS / f'{name}.landmarks_failed.json'
    with rig_lock(name):
        before = json.loads((MODELS / f'{name}.adapter.json').read_text())['coverageSummary'] \
            if (MODELS / f'{name}.adapter.json').exists() else None
        if (MODELS / f'{name}.landmarks.json').exists():
            source = json.loads((MODELS / f'{name}.landmarks.json').read_text()).get('source', 'mediapipe')
            log(f'      landmarks: reusing existing file ({source})')
        else:
            try:
                landmarks_3d(name)
            except RuntimeError as e:      # MediaPipe sees no face (stylised / anime renders)
                log(f'      automatic landmarks failed: {e}')
                failed.write_text(json.dumps({'time': time.time(), 'error': str(e)}))
                return None
        failed.unlink(missing_ok=True)
        generate(name, log=lambda m: log('      ' + m))
        rigged = MODELS / f'{name}.rigged.glb'
        s2 = write_adapter(rigged)['coverageSummary']
    was = f'{before["covered"]}/{before["total"]} -> ' if before else ''
    log(f'      coverage {was}{s2["covered"]}/{s2["total"]} (sources: {s2["bySource"]})')
    log(f'done -> {rigged.name} (open in the viewer)')
    return rigged


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', type=Path, nargs='?')
    ap.add_argument('--existing', metavar='NAME', help='rig web/models/NAME.glb (no conversion)')
    ap.add_argument('--scale', type=float, default=None)
    ap.add_argument('--up', default=None, help='source up axis (default: the format\'s convention)')
    ap.add_argument('--front', default=None, help='source facing axis (default: -z for MMD, +z otherwise)')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--rig', action='store_true', help='always generate blendshapes')
    g.add_argument('--no-rig', action='store_true', help='never generate blendshapes')
    args = ap.parse_args(argv)
    if args.existing:
        # exit code 3 = no face found: the caller (server / web UI) sends the user to the landmark picker
        return 0 if rig_existing(args.existing) else 3
    if args.src is None:
        ap.error('give a model file, or --existing NAME')
    rig = True if args.rig else (False if args.no_rig else None)
    import_model(args.src, args.scale, args.up, args.front, rig)


if __name__ == '__main__':
    sys.exit(main())
