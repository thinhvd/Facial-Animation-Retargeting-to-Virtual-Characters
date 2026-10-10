"""3D landmarks on a mesh: render front view -> MediaPipe 478 landmarks -> ray cast.

    python -m retarget.rig.landmarks HeadA            # -> web/models/HeadA.landmarks.json (+ debug png)
    python -m retarget.rig.landmarks ict_neutral

The same detector on the same kind of render gives the template and the
target corresponding landmark sets by construction (index i <-> index i),
which is what registration needs.  Points that miss the mesh (ray hits
nothing) are marked invalid.
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


def raycast(glb_path: Path, origins: np.ndarray, dirs: np.ndarray):
    import trimesh
    glb = GLB(glb_path)
    pos = glb.mesh_positions().astype(np.float64)
    tri = glb.mesh_triangles()
    mesh = trimesh.Trimesh(pos, tri, process=False)
    locs, ray_ids, tri_ids = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
    pts = np.full((len(origins), 3), np.nan)
    hit_tri = -np.ones(len(origins), np.int64)
    if len(ray_ids):
        pts[ray_ids] = locs
        hit_tri[ray_ids] = tri_ids
    return pts, hit_tri, mesh


def landmarks_3d(model: str, out_dir: Path = MODELS, size: int = 1024, lit: bool = True) -> dict:
    out_dir = Path(out_dir)
    png = out_dir / f'{model}.front.png'
    cam = render_front(model, png, size=size, lit=lit)
    save_camera(cam, out_dir / f'{model}.camera.json')
    lm2d = detect_2d(png)
    if lm2d is None:
        raise RuntimeError(f'MediaPipe found no face in {png}')
    px = lm2d[:, :2] * np.array([cam['width'], cam['height']])
    origins, dirs = pixel_rays(cam, px)
    pts, hit_tri, mesh = raycast(out_dir / f'{model}.glb', origins, dirs)
    valid = ~np.isnan(pts[:, 0])
    # nearest vertex of the hit triangle (useful for vertex-based constraints)
    nearest = -np.ones(len(pts), np.int64)
    if valid.any():
        tv = mesh.faces[hit_tri[valid]]                      # (n, 3) vertex ids
        d = np.linalg.norm(mesh.vertices[tv] - pts[valid][:, None, :], axis=2)
        nearest[valid] = tv[np.arange(len(tv)), d.argmin(1)]
    result = {
        'model': model, 'png': str(png), 'count': int(valid.sum()), 'total': len(pts),
        'points': np.where(valid[:, None], pts, 0.0).round(6).tolist(),
        'valid': valid.tolist(), 'nearest_vertex': nearest.tolist(),
        'pixels': px.round(2).tolist(),
    }
    (out_dir / f'{model}.landmarks.json').write_text(json.dumps(result))
    _debug_image(png, px, valid, out_dir / f'{model}.landmarks.png')
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
    args = ap.parse_args(argv)
    for m in args.models:
        r = landmarks_3d(m, size=args.size, lit=not args.unlit)
        pts = np.array(r['points'])[np.array(r['valid'])]
        print(f'{m}: {r["count"]}/{r["total"]} landmarks hit the mesh; '
              f'x[{pts[:, 0].min():.3f},{pts[:, 0].max():.3f}] y[{pts[:, 1].min():.3f},{pts[:, 1].max():.3f}] '
              f'z[{pts[:, 2].min():.3f},{pts[:, 2].max():.3f}] -> {MODELS / (m + ".landmarks.json")}')


if __name__ == '__main__':
    sys.exit(main())
