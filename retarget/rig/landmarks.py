"""3D landmarks on a mesh: render front view -> MediaPipe 478 landmarks -> ray cast.

    python -m retarget.rig.landmarks HeadA            # -> web/models/HeadA.landmarks.json (+ debug png)
    python -m retarget.rig.landmarks ict_neutral
    python -m retarget.rig.landmarks model4 --render-only   # front.png + camera.json for web/landmarks.html

The same detector on the same kind of render gives the template and the
target corresponding landmark sets by construction (index i <-> index i),
which is what registration needs.  Points that miss the mesh (ray hits
nothing) are marked invalid.

When MediaPipe sees no face (anime, stylised), a person clicks CLICK_POINTS
on the same render in web/landmarks.html; the clicks are stored under their
MediaPipe indices and every other index is invalid, so registration runs
unchanged on the smaller set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ..formats.gltf_reader import GLB
from .render import pixel_rays, render_front, save_camera

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / 'web' / 'models'
TASK = ROOT / 'weights' / 'face_landmarker.task'

# A few MediaPipe indices worth knowing (subject's left = image right when not mirrored):
MP_KEY = {'nose_tip': 1, 'chin': 152, 'left_eye_outer': 263, 'left_eye_inner': 362,
          'right_eye_outer': 33, 'right_eye_inner': 133, 'mouth_left': 291, 'mouth_right': 61,
          'left_iris': 473, 'right_iris': 468, 'forehead': 10, 'upper_lip': 13, 'lower_lip': 14}

# Points a person can place on any face, stylised ones included: (key, MediaPipe index, hint).
# Left/right are the character's; in a front render the character's right is on the image left.
CLICK_POINTS = [
    ('right_eye_outer', 33, "outer corner of the character's right eye (image left)"),
    ('right_eye_inner', 133, "inner corner of the character's right eye"),
    ('right_eye_top', 159, 'right eye: middle of the upper lid'),
    ('right_eye_bottom', 145, 'right eye: middle of the lower lid'),
    ('left_eye_inner', 362, "inner corner of the character's left eye (image right)"),
    ('left_eye_outer', 263, "outer corner of the character's left eye"),
    ('left_eye_top', 386, 'left eye: middle of the upper lid'),
    ('left_eye_bottom', 374, 'left eye: middle of the lower lid'),
    ('right_brow_outer', 70, 'right eyebrow: outer end'),
    ('right_brow_inner', 107, 'right eyebrow: inner end'),
    ('left_brow_inner', 336, 'left eyebrow: inner end'),
    ('left_brow_outer', 300, 'left eyebrow: outer end'),
    ('nose_tip', 1, 'tip of the nose'),
    ('mouth_right', 61, "mouth corner on the character's right (image left)"),
    ('mouth_left', 291, "mouth corner on the character's left"),
    ('upper_lip', 0, 'upper lip: top edge at the centre (skip if the mouth is a drawn line)'),
    ('mouth_centre', 13, 'centre of the mouth line, where the lips meet'),
    ('lower_lip', 17, 'lower lip: bottom edge at the centre (skip if the mouth is a drawn line)'),
    ('chin', 175, 'front of the chin, a little above its lowest point'),
]
# one click stands for several MediaPipe points (13/14: inner upper / lower lip, one line when closed).
# Points on the face outline (234/454, chin bottom 152) are left out on purpose: the ray grazes the
# surface there, so a few pixels of click error move the 3D point by centimetres.
SHARED_CLICKS = {13: (14,)}
CLICK_IDS = np.array([i for _, i, _ in CLICK_POINTS] + [j for js in SHARED_CLICKS.values() for j in js])


def expand_clicks(px: np.ndarray) -> np.ndarray:
    """Copy a click to the other MediaPipe indices it stands for (in place)."""
    for i, others in SHARED_CLICKS.items():
        if not np.isnan(px[i]).any():
            for j in others:
                px[j] = px[i]
    return px


def detect_2d(png: Path, task: Path = TASK) -> np.ndarray | None:
    """(478, 3) normalized landmarks [x, y in 0..1, z relative] or None."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(task)),
        running_mode=vision.RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3)
    lm = vision.FaceLandmarker.create_from_options(opts)
    img = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2RGB)
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
    lm.close()
    if not res.face_landmarks:
        return None
    return np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]], np.float64)


# Points that sit on the face itself on any model: eye corners, nose tip, mouth corners, chin.
FACE_CORE = (33, 133, 362, 263, 1, 61, 291, 152, 175)


def _visible_triangles(glb: GLB) -> np.ndarray:
    """Triangle mask without fully transparent materials (MMD expression overlays such as
    青ざめ / 紅潮 have alpha 0 until a material morph shows them; the render does not show them)."""
    mats = glb.json.get('materials', [])
    mask = []
    for prim in glb.meshes()[0]['primitives']:
        if 'indices' not in prim:
            continue
        n = glb.json['accessors'][prim['indices']]['count'] // 3
        mat = mats[prim['material']] if 'material' in prim and prim['material'] < len(mats) else {}
        alpha = mat.get('pbrMetallicRoughness', {}).get('baseColorFactor', [1, 1, 1, 1])[3]
        mask.append(np.full(n, alpha > 0.01))
    return np.concatenate(mask)


def raycast(glb_path: Path, origins: np.ndarray, dirs: np.ndarray, core: np.ndarray | None = None):
    """First visible hit per ray -> (points, triangle ids, mesh).

    With `core` (mask of rays known to land on the face), each ray takes its first hit on a
    mesh piece the core rays landed on, so a brow under the fringe is placed on the forehead
    behind the hair rather than on the hair.
    """
    import trimesh
    from .generate import weld
    glb = GLB(glb_path)
    pos = glb.mesh_positions().astype(np.float64)
    tri = glb.mesh_triangles()
    mesh = trimesh.Trimesh(pos, tri, process=False)
    vis = np.where(_visible_triangles(glb))[0]
    vmesh = trimesh.Trimesh(pos, tri[vis], process=False)
    locs, ray_ids, tri_ids = vmesh.ray.intersects_location(origins, dirs, multiple_hits=True)
    tri_ids = vis[tri_ids]
    pts = np.full((len(origins), 3), np.nan)
    hit_tri = -np.ones(len(origins), np.int64)
    if not len(ray_ids):
        return pts, hit_tri, mesh
    order = np.lexsort((((locs - origins[ray_ids]) * dirs[ray_ids]).sum(1), ray_ids))
    locs, ray_ids, tri_ids = locs[order], ray_ids[order], tri_ids[order]
    first = np.r_[True, ray_ids[1:] != ray_ids[:-1]]
    pick = first.copy()
    if core is not None and core.any():
        welded = weld(pos, tri.astype(np.int64))
        groups = trimesh.graph.connected_components(welded.face_adjacency, nodes=np.arange(len(tri)))
        comp = np.empty(len(tri), np.int64)
        for gi, g in enumerate(groups):
            comp[g] = gi
        face = set(comp[tri_ids[first & core[ray_ids]]].tolist())
        on_face = np.isin(comp[tri_ids], list(face))
        for r in np.unique(ray_ids):
            hits = np.where(ray_ids == r)[0]
            good = hits[on_face[hits]]
            if len(good) and not on_face[hits[0]]:
                pick[hits[0]] = False
                pick[good[0]] = True
    pts[ray_ids[pick]] = locs[pick]
    hit_tri[ray_ids[pick]] = tri_ids[pick]
    return pts, hit_tri, mesh


def render(model: str, out_dir: Path = MODELS, size: int = 1024, lit: bool = True) -> dict:
    """Front render + camera, the input both for MediaPipe and for hand-picked points."""
    out_dir = Path(out_dir)
    png = out_dir / f'{model}.front.png'
    cam = render_front(model, png, size=size, lit=lit)
    save_camera(cam, out_dir / f'{model}.camera.json')
    return cam


def landmarks_3d(model: str, out_dir: Path = MODELS, size: int = 1024, lit: bool = True) -> dict:
    out_dir = Path(out_dir)
    cam = render(model, out_dir, size, lit)
    lm2d = detect_2d(Path(cam['png']))
    if lm2d is None:
        raise RuntimeError(f'MediaPipe found no face in {cam["png"]}')
    px = lm2d[:, :2] * np.array([cam['width'], cam['height']])
    return landmarks_from_pixels(model, px, cam, out_dir, source='mediapipe')


def landmarks_from_pixels(model: str, px: np.ndarray, cam: dict, out_dir: Path = MODELS,
                          source: str = 'manual', out: Path | None = None) -> dict:
    """(478, 2) pixel positions (NaN = not given) -> 3D points on the mesh -> landmarks.json."""
    out_dir = Path(out_dir)
    png = Path(cam['png'])
    px = np.asarray(px, np.float64)
    given = ~np.isnan(px).any(1)
    pts = np.full((len(px), 3), np.nan)
    hit_tri = -np.ones(len(px), np.int64)
    if not given.any():
        raise ValueError('no landmark pixels given')
    origins, dirs = pixel_rays(cam, px[given])
    core = np.isin(np.where(given)[0], FACE_CORE)
    p_hit, t_hit, mesh = raycast(out_dir / f'{model}.glb', origins, dirs, core)
    pts[given], hit_tri[given] = p_hit, t_hit
    valid = ~np.isnan(pts[:, 0])
    if source != 'mediapipe' and valid.sum() >= 4:
        # A click on a feature lands on the front of the face.  A hit far behind the others went
        # through a hole (an eye socket without eyeball) or grazed the outline: treat as a miss.
        depth = np.full(len(px), np.nan)
        depth[given] = ((p_hit - origins) * dirs).sum(1)
        width = np.ptp(pts[valid][:, 0])
        deep = valid & (depth > np.nanmedian(depth[valid]) + 0.4 * width)
        pts[deep] = np.nan
        valid &= ~deep
    # nearest vertex of the hit triangle (useful for vertex-based constraints)
    nearest = -np.ones(len(pts), np.int64)
    if valid.any():
        tv = mesh.faces[hit_tri[valid]]                      # (n, 3) vertex ids
        d = np.linalg.norm(mesh.vertices[tv] - pts[valid][:, None, :], axis=2)
        nearest[valid] = tv[np.arange(len(tv)), d.argmin(1)]
    result = {
        'model': model, 'png': str(png), 'source': source, 'count': int(valid.sum()), 'total': len(pts),
        'points': np.where(valid[:, None], pts, 0.0).round(6).tolist(),
        'valid': valid.tolist(), 'nearest_vertex': nearest.tolist(),
        'pixels': np.where(given[:, None], px, -1.0).round(2).tolist(),
        'missed': [int(i) for i in np.where(given & ~valid)[0]],     # given, but no hit on the face front
    }
    out = out or out_dir / f'{model}.landmarks.json'
    out.write_text(json.dumps(result))
    _debug_image(png, np.where(given[:, None], px, -100), valid, out.with_suffix('.png'))
    return result


def _debug_image(png: Path, px: np.ndarray, valid: np.ndarray, out: Path) -> None:
    import cv2
    img = cv2.imread(str(png))
    for (x, y), ok in zip(px, valid):
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0) if ok else (0, 0, 255), -1, cv2.LINE_AA)
    for name, i in MP_KEY.items():
        cv2.putText(img, name, (int(px[i, 0]) + 3, int(px[i, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), img)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('models', nargs='+')
    ap.add_argument('--size', type=int, default=1024)
    ap.add_argument('--unlit', action='store_true')
    ap.add_argument('--render-only', action='store_true', help='only write <model>.front.png + camera.json')
    args = ap.parse_args(argv)
    for m in args.models:
        if args.render_only:
            render(m, size=args.size, lit=not args.unlit)
            print(f'{m}: render + camera written; pick points at /landmarks.html?model={m}')
            continue
        r = landmarks_3d(m, size=args.size, lit=not args.unlit)
        pts = np.array(r['points'])[np.array(r['valid'])]
        print(f'{m}: {r["count"]}/{r["total"]} landmarks hit the mesh; '
              f'x[{pts[:, 0].min():.3f},{pts[:, 0].max():.3f}] y[{pts[:, 1].min():.3f},{pts[:, 1].max():.3f}] '
              f'z[{pts[:, 2].min():.3f},{pts[:, 2].max():.3f}] -> {MODELS / (m + ".landmarks.json")}')


if __name__ == '__main__':
    sys.exit(main())
