"""P1 gate: how well does template registration + deformation transfer
recover known blendshapes on a held-out ICT identity?

    python -m retarget.rig.evaluate --identity 7                 # same topology as the template
    python -m retarget.rig.evaluate --identity 7 --remesh 0.5    # decimated to 50% faces (different topology)
    python -m retarget.rig.evaluate --identity 7 --remesh 0.5 --clicks --click-noise 4   # as if hand-picked

The target is `neutral + coeff * identity_mode`; its ground-truth
expression deltas are ICT's own (the light model is linear, so they are
identity-independent).  We hide them, run the full pipeline (render ->
landmarks -> register -> transfer) and compare on the face region.

Error = mean |generated - truth| over face vertices the truth moves,
reported in mm and as % of the inter-ocular distance (IOD).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

from ..formats import gltf_writer, obj_convert
from ..formats.gltf_reader import GLB
from . import ict
from .generate import MODELS, generate
from .landmarks import CLICK_POINTS, SHARED_CLICKS, expand_clicks, landmarks_3d, landmarks_from_pixels

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / 'data' / 'eval'


def make_target(tmpl: ict.ICTTemplate, identity: int, coeff: float, remesh: float | None):
    """Return (name, positions, triangles, gt_deltas dict, face_mask, iod)."""
    delta_id = ict.load_identity(Path(tmpl.extras['dir']), identity, tmpl.scale)
    pos = tmpl.positions + coeff * delta_id
    tris = tmpl.triangles.astype(np.int64)
    face_ids = tmpl.region('face')
    name = f'eval_id{identity:03d}'
    lm = tmpl.landmarks68
    iod = float(np.linalg.norm(pos[lm[36]] - pos[lm[45]]))     # outer eye corners (68-pt convention)

    if remesh is None:
        face_mask = np.zeros(len(pos), bool)
        face_mask[face_ids] = True
        return name, pos.astype(np.float64), tris, dict(tmpl.expressions), face_mask, iod

    import fast_simplification
    # keep only the head surface (face + head/neck) for the remeshed target: eyeballs/teeth/lashes
    # are separate shells that would confuse decimation
    keep = np.zeros(len(pos), bool)
    keep[tmpl.region('face')] = True
    keep[tmpl.region('head_neck')] = True
    sub_tris = tris[keep[tris].all(1)]
    used = np.unique(sub_tris)
    remap = -np.ones(len(pos), np.int64)
    remap[used] = np.arange(len(used))
    src_pos, src_tris = pos[used].astype(np.float64), remap[sub_tris]
    new_pos, new_tris = fast_simplification.simplify(src_pos, src_tris, target_reduction=1 - remesh)
    new_pos = np.asarray(new_pos, np.float64)
    new_tris = np.asarray(new_tris, np.int64)
    # ground truth on the new vertices: barycentric interpolation of the template deltas
    from .geometry import SurfaceQuery
    src = trimesh.Trimesh(src_pos, src_tris, process=False)
    closest, dist, tri_id = SurfaceQuery(src_pos, src_tris).closest(new_pos)
    bary = trimesh.triangles.points_to_barycentric(src.triangles[tri_id], closest)
    corners = used[src_tris[tri_id]]                                # (n, 3) template vertex ids
    gt = {}
    for ename, d in tmpl.expressions.items():
        gt[ename] = (d[corners] * bary[:, :, None]).sum(1).astype(np.float32)
    in_face = np.zeros(len(pos), bool)
    in_face[face_ids] = True
    face_mask = in_face[corners].all(1)
    return f'{name}_r{int(remesh * 100):02d}', new_pos, new_tris, gt, face_mask, iod


def simulated_clicks(name: str, noise_px: float, seed: int = 0) -> Path:
    """Keep only the CLICK_POINTS of the MediaPipe landmarks, jittered like a hand click."""
    full = json.loads((MODELS / f'{name}.landmarks.json').read_text())
    cam = json.loads((MODELS / f'{name}.camera.json').read_text())
    mp = np.asarray(full['pixels'], np.float64)
    ids = np.array([i for _, i, _ in CLICK_POINTS])
    px = np.full((full['total'], 2), np.nan)
    px[ids] = mp[ids]
    for i, others in SHARED_CLICKS.items():          # a person clicks the middle of the lip line
        px[i] = mp[[i, *others]].mean(0)
    px[ids] += np.random.default_rng(seed).normal(0, noise_px, (len(ids), 2))
    expand_clicks(px)
    out = MODELS / f'{name}.clicks.landmarks.json'
    r = landmarks_from_pixels(name, px, cam, source=f'simulated clicks, noise {noise_px} px', out=out)
    print(f'[eval] simulated clicks: {len(ids)} clicks, {r["count"]} landmarks hit the mesh, noise {noise_px} px')
    return out


def evaluate(identity: int, coeff: float = 1.0, remesh: float | None = None, log=print,
             subset: str = 'face', face_dist: float = 0.012, anchor_dist: float = 0.04,
             tag: str = '', clicks: bool = False, click_noise: float = 0.0, seed: int = 0, **gen_kw) -> dict:
    t0 = time.time()
    tmpl = ict.load()
    name, pos, tris, gt, face_mask, iod = make_target(tmpl, identity, coeff, remesh)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # target as a canonical .glb (no morphs: the truth stays hidden)
    from ..mesh import CanonicalMaterial, CanonicalMesh
    mesh = CanonicalMesh(name, pos.astype(np.float32), obj_convert.vertex_normals(pos, tris),
                         np.zeros((len(pos), 2), np.float32), tris.astype(np.uint32), [(0, 0, len(tris))],
                         [CanonicalMaterial('default', np.array([0.8, 0.72, 0.65, 1], np.float32), unlit=False)],
                         extras={'source_format': 'ict-eval', 'identity': identity, 'coeff': coeff, 'remesh': remesh})
    gltf_writer.write_glb(mesh, MODELS / f'{name}.glb')
    log(f'[eval] target {name}: V={len(pos)} F={len(tris)} face verts={face_mask.sum()} IOD={iod * 1000:.1f} mm')

    if not (MODELS / f'{name}.landmarks.json').exists():
        landmarks_3d(name)
    lm_path = simulated_clicks(name, click_noise, seed) if clicks else None
    report = generate(name, log=log, subset=subset, face_dist=face_dist, anchor_dist=anchor_dist,
                      landmarks_path=lm_path, **gen_kw)

    gen = GLB(MODELS / f'{name}.rigged.glb').mesh_morphs()
    rows = []
    for ename, truth in gt.items():
        g = gen.get(ename)
        if g is None:
            continue
        moved = face_mask & (np.linalg.norm(truth, axis=1) > 1e-4)      # truth moves it by > 0.1 mm
        if moved.sum() < 10:
            continue
        err = np.linalg.norm(g[moved] - truth[moved], axis=1)
        mag = np.linalg.norm(truth[moved], axis=1)
        rows.append({'expression': ename, 'n': int(moved.sum()),
                     'mean_mm': float(err.mean() * 1000), 'p95_mm': float(np.percentile(err, 95) * 1000),
                     'truth_mean_mm': float(mag.mean() * 1000),
                     'rel': float(err.mean() / max(mag.mean(), 1e-9)),
                     'iod_pct': float(err.mean() / iod * 100)})
    rows.sort(key=lambda r: -r['mean_mm'])
    overall = {
        'mean_mm': float(np.mean([r['mean_mm'] for r in rows])),
        'p95_mm': float(np.mean([r['p95_mm'] for r in rows])),
        'rel': float(np.mean([r['rel'] for r in rows])),
        'iod_pct': float(np.mean([r['iod_pct'] for r in rows])),
        'expressions': len(rows), 'iod_mm': iod * 1000,
    }
    result = {'target': name, 'identity': identity, 'coeff': coeff, 'remesh': remesh, 'subset': subset,
              'landmarks': f'clicks ({len(CLICK_POINTS)}), noise {click_noise} px' if clicks else 'mediapipe',
              'face_dist': face_dist, 'anchor_dist': anchor_dist, 'params': gen_kw,
              'registration': report['registration'], 'overall': overall, 'per_expression': rows,
              'seconds': round(time.time() - t0, 1)}
    out = EVAL_DIR / f'{name}_{subset}{("_" + tag) if tag else ""}.json'
    out.write_text(json.dumps(result, indent=1))
    log(f'[eval] {name}: mean err {overall["mean_mm"]:.2f} mm ({overall["iod_pct"]:.1f}% IOD), '
        f'rel {overall["rel"]:.2f}, p95 {overall["p95_mm"]:.2f} mm over {len(rows)} expressions -> {out}')
    log('  worst: ' + ', '.join(f'{r["expression"]} {r["mean_mm"]:.1f}mm' for r in rows[:5]))
    log('  best:  ' + ', '.join(f'{r["expression"]} {r["mean_mm"]:.1f}mm' for r in rows[-5:]))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--identity', type=int, default=7)
    ap.add_argument('--coeff', type=float, default=1.0)
    ap.add_argument('--remesh', type=float, default=None, help='keep this fraction of faces (e.g. 0.5)')
    ap.add_argument('--subset', default='face', choices=['fitting', 'face', 'face+neck'])
    ap.add_argument('--face-dist', type=float, default=0.012)
    ap.add_argument('--anchor-dist', type=float, default=0.04)
    ap.add_argument('--smooth-free', type=float, default=0.5)
    ap.add_argument('--smooth-corr', type=float, default=0.05)
    ap.add_argument('--identity-w', type=float, default=0.05)
    ap.add_argument('--identity-skin', type=float, default=0.3)
    ap.add_argument('--stiff-min', type=float, default=1.0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--clicks', action='store_true', help='register on the hand-pickable subset only')
    ap.add_argument('--click-noise', type=float, default=0.0, help='pixel std of the simulated clicks')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--landmark-weight', type=float, default=5.0)
    args = ap.parse_args(argv)
    evaluate(args.identity, args.coeff, args.remesh, subset=args.subset,
             face_dist=args.face_dist, anchor_dist=args.anchor_dist, tag=args.tag,
             clicks=args.clicks, click_noise=args.click_noise, seed=args.seed,
             landmark_weight=args.landmark_weight,
             smooth_free=args.smooth_free, smooth_corr=args.smooth_corr, identity_w=args.identity_w,
             identity_skin=args.identity_skin, stiff_min=args.stiff_min)


if __name__ == '__main__':
    sys.exit(main())
