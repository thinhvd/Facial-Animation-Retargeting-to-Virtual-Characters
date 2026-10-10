"""Blender as an external converter for formats the pure-Python path cannot read.

Blender runs as a subprocess (`blender -b`), so its Python and libraries stay
out of this environment.  Looked up in order: $BLENDER, `blender` on PATH,
data/blender/*/blender (the portable build the README installs).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'tools' / 'blender_export.py'
FORMATS = ('.fbx', '.blend', '.dae')


def find_blender() -> Path | None:
    if os.environ.get('BLENDER'):
        return Path(os.environ['BLENDER'])
    if shutil.which('blender'):
        return Path(shutil.which('blender'))
    found = sorted((ROOT / 'data' / 'blender').glob('blender-*/blender'))
    return found[-1] if found else None


def export_glb(src: Path, dst: Path, timeout: float = 600.0, log=print) -> Path:
    blender = find_blender()
    if blender is None:
        raise SystemExit(f'{src.name}: {src.suffix} needs Blender.  Install the portable build into data/blender/ '
                         '(see retarget/README.md) or set $BLENDER.')
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(blender), '-b', '--factory-startup', '--python-exit-code', '1',
           '--python', str(SCRIPT), '--', str(src.resolve()), str(dst.resolve())]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    for line in proc.stdout.splitlines():
        if line.startswith('blender_export:'):
            log('  ' + line)
    if proc.returncode != 0 or not dst.exists():
        tail = '\n'.join((proc.stdout + proc.stderr).splitlines()[-15:])
        raise SystemExit(f'Blender failed on {src.name} (exit {proc.returncode}):\n{tail}')
    return dst
