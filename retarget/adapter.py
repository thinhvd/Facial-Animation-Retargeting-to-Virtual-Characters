"""Generate the per-model adapter: how a converted model answers each name
of the canonical expression space.

    python -m retarget.adapter web/models/HeadA.glb        # -> web/models/HeadA.adapter.json

Priority (see proposal 4.3): existing morph (artist) > bone > generated
blendshape (P1, not yet) > rigid fallback.  The adapter records which
canonical inputs are covered and by what, so coverage can be reported
per model and per region.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .canonical import ARKIT_52, GAZE_SHAPES, REGIONS, SEMANTIC_BONES, mirror_name
from .formats.gltf_reader import GLB
from .formats.gltf_writer import read_glb_json
from .naming import VRM_MORPH_DRIVERS, lookup_morph_driver, resolve_bone

VRM0_PRESETS = {
    'a': 'aa', 'i': 'ih', 'u': 'ou', 'e': 'ee', 'o': 'oh',
    'blink': 'blink', 'blink_l': 'blinkLeft', 'blink_r': 'blinkRight',
    'lookup': 'lookUp', 'lookdown': 'lookDown', 'lookleft': 'lookLeft', 'lookright': 'lookRight',
    'joy': 'happy', 'angry': 'angry', 'sorrow': 'sad', 'fun': 'relaxed', 'neutral': 'neutral',
}


def inspect_glb(path: Path) -> dict:
    doc = read_glb_json(path)
    morphs: list[str] = []
    generated: list[str] = []
    for mesh in doc.get('meshes', []):
        names = mesh.get('extras', {}).get('targetNames', [])
        metas = mesh.get('extras', {}).get('targetMeta', [{}] * len(names))
        for name, meta in zip(names, metas):
            if name not in morphs:
                morphs.append(name)
            if isinstance(meta, dict) and meta.get('generated'):
                generated.append(name)
    nodes = [n.get('name', '') for n in doc.get('nodes', [])]
    ext = doc.get('extensions', {})
    vrm_version = None
    vrm_expressions: list[str] = []
    vrm_bones: dict[str, str] = {}
    if 'VRMC_vrm' in ext:
        vrm_version = ext['VRMC_vrm'].get('specVersion', '1.0')
        vrm_expressions = list(ext['VRMC_vrm'].get('expressions', {}).get('preset', {}).keys())
        for sem, b in ext['VRMC_vrm'].get('humanoid', {}).get('humanBones', {}).items():
            if 'node' in b and b['node'] < len(nodes):
                vrm_bones[sem] = nodes[b['node']]
    elif 'VRM' in ext:
        vrm_version = ext['VRM'].get('specVersion', '0.x')
        for g in ext['VRM'].get('blendShapeMaster', {}).get('blendShapeGroups', []):
            preset = VRM0_PRESETS.get(str(g.get('presetName', '')).lower())
            if preset and preset not in vrm_expressions:
                vrm_expressions.append(preset)
        for b in ext['VRM'].get('humanoid', {}).get('humanBones', []):
            if 'bone' in b and 'node' in b and b['node'] < len(nodes):
                vrm_bones[b['bone']] = nodes[b['node']]
    return {'morphs': morphs, 'generated': generated, 'nodes': nodes, 'vrm_version': vrm_version,
            'vrm_expressions': vrm_expressions, 'vrm_bones': vrm_bones,
            'generator': doc.get('asset', {}).get('generator'),
            'source': doc.get('asset', {}).get('extras', {})}


def _side(inputs: list[str]) -> str | None:
    """'Left' / 'Right' if the inputs that matter agree on a side, else None."""
    sides = {('Left' if i.endswith('Left') else 'Right' if i.endswith('Right') else None) for i in inputs}
    sides.discard(None)
    return sides.pop() if len(sides) == 1 else None


def verify_laterality(path: Path, entries: list[dict], min_ratio: float = 0.25) -> list[tuple[dict, float]]:
    """Return (entry, offset_ratio) for morph entries whose displaced vertices
    sit on the opposite side from what their inputs claim.

    offset_ratio = (centroid x of the touched vertices - face centre) / x extent
    of the touched vertices: ~0 for a symmetric morph, > 1 for a one-eye wink.
    """
    try:
        glb = GLB(path)
        positions = glb.mesh_positions()
        morphs = glb.mesh_morphs()
    except Exception:
        return []
    if not len(positions):
        return []
    centre = float(np.median(positions[:, 0]))
    wrong = []
    for e in entries:
        if e['kind'] != 'morph' or e['target'] not in morphs:
            continue
        # for 'diff' the first input names the side that closes; otherwise all inputs must agree
        side = _side(e['inputs'][:1] if e['combine'] == 'diff' else e['inputs'])
        if side is None:
            continue
        disp = morphs[e['target']]
        mask = np.abs(disp).sum(1) > 0
        if mask.sum() < 3:
            continue
        xs = positions[mask, 0]
        extent = max(float(xs.max() - xs.min()), 1e-6)
        ratio = (float(xs.mean()) - centre) / extent
        if abs(ratio) < min_ratio:
            continue  # symmetric morph, nothing to verify
        if (side == 'Left') != (ratio > 0):
            wrong.append((e, ratio))
    return wrong


VRM_EMOTIONS = ('happy', 'angry', 'sad', 'relaxed', 'surprised')


def build_adapter(path: Path, emotions: bool = False) -> dict:
    info = inspect_glb(path)
    entries: list[dict] = []
    consumers: dict[str, list[str]] = defaultdict(list)

    # ---- bones
    bones: dict[str, str | None] = {}
    for sem in SEMANTIC_BONES:
        name = info['vrm_bones'].get(sem) or resolve_bone(sem, info['nodes'])
        bones[sem] = name

    # ---- VRM expressions take precedence over raw morph targets (they bundle several).
    # Emotion presets (happy, angry, ...) change eyes + mouth + brows at once, so they are
    # not driven from single-region inputs unless explicitly enabled.
    if info['vrm_expressions']:
        for preset in info['vrm_expressions']:
            drv = VRM_MORPH_DRIVERS.get(preset)
            if not drv:
                continue
            if preset in VRM_EMOTIONS and not emotions:
                continue
            if preset.startswith('look') and bones['leftEye'] and bones['rightEye']:
                continue  # bones give continuous gaze; skip the discrete look* expressions
            entries.append({'kind': 'vrmExpression', 'target': preset, **drv})
            for i in drv['inputs']:
                consumers[i].append(f'vrm:{preset}')

    # ---- raw morph targets: artist-made first, generated ones only for inputs nobody covers yet
    generated = set(info['generated'])
    artist = [n for n in info['morphs'] if n not in generated]
    for name in artist:
        drv = lookup_morph_driver(name)
        if not drv:
            continue
        if any(e['target'] == name for e in entries):
            continue
        # on a VRM the presets already bundle the raw morphs: only fill inputs the presets miss
        if info['vrm_expressions'] and all(consumers.get(i) for i in drv['inputs']):
            continue
        entries.append({'kind': 'morph', 'target': name, **drv})
        for i in drv['inputs']:
            consumers[i].append(f'morph:{name}')
    for name in info['morphs']:
        if name not in generated:
            continue
        drv = lookup_morph_driver(name)
        if not drv:
            continue
        covered_by_bone = all(i in GAZE_SHAPES for i in drv['inputs']) and bones['leftEye'] and bones['rightEye']
        if covered_by_bone or all(consumers.get(i) for i in drv['inputs']):
            continue
        entries.append({'kind': 'morph', 'target': name, 'generated': True, **drv})
        for i in drv['inputs']:
            consumers[i].append(f'generated:{name}')

    # ---- verify left/right of lateral morphs against the geometry.
    # Name tables assume the character's own left/right, but some models name winks
    # from the viewer's side (kizunaai does).  The displaced vertices tell the truth:
    # the character's left is +X in canonical space.
    fixes = verify_laterality(path, entries)
    for e, ratio in fixes:
        e['inputs'] = [mirror_name(i) for i in e['inputs']]
        e['lateralityFixed'] = round(ratio, 2)
    if fixes:
        consumers = defaultdict(list)
        for e in entries:
            kind = 'vrm' if e['kind'] == 'vrmExpression' else ('generated' if e.get('generated') else 'morph')
            for i in e['inputs']:
                consumers[i].append(f"{kind}:{e['target']}")

    # ---- coverage
    coverage: dict[str, str | None] = {}
    for n in ARKIT_52:
        if n in GAZE_SHAPES and bones['leftEye'] and bones['rightEye']:
            coverage[n] = 'bone'
        elif consumers.get(n):
            coverage[n] = consumers[n][0]
        else:
            coverage[n] = None
    by_region = {r: {'covered': sum(1 for n in names if coverage[n]), 'total': len(names)}
                 for r, names in REGIONS.items()}
    covered = sum(1 for n in ARKIT_52 if coverage[n])
    by_source = defaultdict(int)
    for c in coverage.values():
        by_source[c.split(':')[0] if c else 'none'] += 1

    return {
        'version': 1,
        'model': path.name,
        'source': info['source'],
        'vrm': info['vrm_version'],
        'headMode': 'bone' if bones['head'] else 'root',
        'gazeMode': 'bone' if (bones['leftEye'] and bones['rightEye']) else 'morph',
        'bones': bones,
        'entries': entries,
        'unmappedMorphs': [m for m in info['morphs']
                           if m not in generated and not any(e['target'] == m for e in entries)],
        'unusedGenerated': [m for m in info['morphs']
                            if m in generated and not any(e['target'] == m for e in entries)],
        'coverage': coverage,
        'coverageSummary': {'covered': covered, 'total': len(ARKIT_52), 'byRegion': by_region,
                            'bySource': dict(by_source)},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('glb', type=Path, nargs='+')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--emotions', action='store_true', help='also drive VRM emotion presets (happy, angry, ...)')
    args = ap.parse_args(argv)
    for glb in args.glb:
        adapter = build_adapter(glb, emotions=args.emotions)
        out = glb.with_suffix('.adapter.json')
        out.write_text(json.dumps(adapter, ensure_ascii=False, indent=1), encoding='utf-8')
        s = adapter['coverageSummary']
        bones = {k: v for k, v in adapter['bones'].items() if v}
        regions = ', '.join(f'{r} {v["covered"]}/{v["total"]}' for r, v in s['byRegion'].items())
        sources = ', '.join(f'{k} {v}' for k, v in s['bySource'].items())
        print(f'{glb.name}: bones={bones} head={adapter["headMode"]} gaze={adapter["gazeMode"]} '
              f'entries={len(adapter["entries"])} coverage={s["covered"]}/{s["total"]} ({regions}) [{sources}]')
        fixed = [e['target'] for e in adapter['entries'] if e.get('lateralityFixed')]
        if fixed:
            print(f'  laterality corrected from geometry: {", ".join(fixed)}')
        if not args.quiet:
            missing = [n for n, c in adapter['coverage'].items() if not c]
            print(f'  uncovered ({len(missing)}): {", ".join(missing)}')
            if adapter['unmappedMorphs']:
                print(f'  unmapped morphs ({len(adapter["unmappedMorphs"])}): '
                      f'{", ".join(adapter["unmappedMorphs"])}')
        print(f'  -> {out}')


if __name__ == '__main__':
    main()
