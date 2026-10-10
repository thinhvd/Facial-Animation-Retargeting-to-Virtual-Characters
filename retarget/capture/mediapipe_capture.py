"""Webcam / video -> canonical frames via MediaPipe Face Landmarker.

    python -m retarget.capture.mediapipe_capture 0                 # webcam
    python -m retarget.capture.mediapipe_capture clip.mp4 --no-mirror

Outputs per frame (see retarget.canonical):
  bs    52 ARKit blendshape scores (MediaPipe gives 51 of them + _neutral; tongueOut = 0)
  head  Euler XYZ degrees for the avatar's head bone
  gaze  [pitch, yaw] degrees from the eyeLook* scores (MediaPipe has no explicit gaze)

bs and head go through a One Euro filter per channel (--filter, see filters.py);
the viewer's own smoothing can then stay at 0.

"Mirror" mode (default, like a mirror / VTuber view): the subject's left
drives the avatar's right and yaw/roll are negated, so the avatar copies
the user as seen on screen.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import socketio

from ..canonical import ARKIT_52, mirror_name
from .filters import FrameFilter

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / 'weights' / 'face_landmarker.task'
NS = '/retarget'


def matrix_to_euler_xyz(m: np.ndarray) -> tuple[float, float, float]:
    """Rotation matrix (3x3) -> Euler XYZ in degrees (three.js 'XYZ' order)."""
    sy = -m[2, 0]
    sy = max(-1.0, min(1.0, sy))
    y = math.asin(sy)
    if abs(sy) < 0.9999:
        x = math.atan2(m[2, 1], m[2, 2])
        z = math.atan2(m[1, 0], m[0, 0])
    else:
        x = math.atan2(-m[1, 2], m[1, 1])
        z = 0.0
    return math.degrees(x), math.degrees(y), math.degrees(z)


V4L2_CID_EXPOSURE_AUTO_PRIORITY = 0x009a0903
VIDIOC_S_CTRL = (3 << 30) | (8 << 16) | (ord('V') << 8) | 28


def _v4l2_set_ctrl(dev: str, cid: int, value: int) -> bool:
    import fcntl
    import os
    import struct
    try:
        fd = os.open(dev, os.O_RDWR | os.O_NONBLOCK)
        try:
            fcntl.ioctl(fd, VIDIOC_S_CTRL, struct.pack('Ii', cid, value))
        finally:
            os.close(fd)
        return True
    except OSError:
        return False


def open_webcam(index: int, width: int, height: int, fps: float = 30.0) -> cv2.VideoCapture:
    """Open a webcam at a fixed frame rate.

    UVC webcams with exposure_auto_priority=1 stretch the exposure in dim light
    and fall to 7.5 fps; blinks then last one frame and no filter can both
    smooth and keep up.  Priority 0 (the driver default) holds the frame rate
    and raises gain instead.
    """
    linux = sys.platform.startswith('linux')
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2) if linux else cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if linux and _v4l2_set_ctrl(f'/dev/video{index}', V4L2_CID_EXPOSURE_AUTO_PRIORITY, 0):
        print('[capture] exposure_auto_priority=0 (fixed frame rate)')
    return cap


class FaceCapture:
    def __init__(self, model_path: Path = DEFAULT_MODEL, gpu: bool = False, filter: str = 'euro'):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self.mp = mp
        self.vision = vision
        if not model_path.exists():
            raise SystemExit(f'{model_path} missing: download face_landmarker.task (see README)')
        delegate = mp_python.BaseOptions.Delegate.GPU if gpu else mp_python.BaseOptions.Delegate.CPU
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path), delegate=delegate),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.last_ms = -1
        self.calibration = np.eye(3)   # head pose offset captured with 'c' key
        self.last_rotation = None      # MediaPipe head rotation of the last processed frame
        self.filter = FrameFilter(ARKIT_52, filter)
        self.raw = None                # {'bs', 'head'} of the last frame before filtering

    def process(self, frame_bgr: np.ndarray, t: float, mirror: bool = True) -> dict | None:
        """t: frame time in seconds (media time for a file, clock time for a webcam)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        self.last_ms = max(int(t * 1000), self.last_ms + 1)   # VIDEO mode wants increasing timestamps
        res = self.landmarker.detect_for_video(image, self.last_ms)
        if not res.face_blendshapes:
            return None

        raw = {c.category_name: float(c.score) for c in res.face_blendshapes[0]}
        bs = {}
        for name in ARKIT_52:
            src = mirror_name(name) if mirror else name
            bs[name] = raw.get(src, 0.0)

        # head pose: MediaPipe matrix maps the canonical face model (facing +Z toward
        # the camera, Y up, X to the subject's left... in camera space) to the detected
        # face.  Its rotation part, expressed in the avatar frame, gives the head Euler.
        head = [0.0, 0.0, 0.0]
        if res.facial_transformation_matrixes:
            m = np.array(res.facial_transformation_matrixes[0]).reshape(4, 4)[:3, :3]
            self.last_rotation = m
            m = self.calibration.T @ m
            rx, ry, rz = matrix_to_euler_xyz(m)
            # camera frame -> avatar frame: the subject faces the camera, so yaw/roll flip
            # when the avatar faces the viewer; mirror view flips them back.
            if mirror:
                head = [rx, ry, rz]
            else:
                head = [rx, -ry, -rz]

        self.raw = {'bs': bs, 'head': head}
        bs, head = self.filter(bs, head, t)

        # gaze from eyeLook* scores (already mirrored above through `bs`)
        up = 0.5 * (bs['eyeLookUpLeft'] + bs['eyeLookUpRight'])
        down = 0.5 * (bs['eyeLookDownLeft'] + bs['eyeLookDownRight'])
        # subject's left eye looking "out" = looking to the subject's left
        left = 0.5 * (bs['eyeLookOutLeft'] + bs['eyeLookInRight'])
        right = 0.5 * (bs['eyeLookInLeft'] + bs['eyeLookOutRight'])
        pitch = (down - up) * 25.0       # degrees; positive rotation.x looks down for a +Z facing avatar
        yaw = (left - right) * 30.0      # positive rotation.y turns toward +X (avatar's left)
        return {'t': t, 'bs': bs, 'head': head,
                'gaze': [pitch, yaw], 'src': 'mediapipe',
                'landmarks': None if res.face_landmarks is None else len(res.face_landmarks[0])}

    def close(self) -> None:
        try:
            self.landmarker.close()
        except Exception:
            pass
        self.landmarker = None

    def zero_head(self) -> None:
        if self.last_rotation is not None:
            self.calibration = self.last_rotation
            print('[capture] head pose zeroed')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source', help='webcam index (0) or video path')
    ap.add_argument('--server', default='http://127.0.0.1:6790')
    ap.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    ap.add_argument('--no-mirror', action='store_true')
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--no-preview', action='store_true')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--realtime', action='store_true', help='pace a video file at its own FPS')
    ap.add_argument('--filter', choices=('euro', 'ema', 'none'), default='euro')
    ap.add_argument('--dump', type=Path, help='write raw + filtered frames as JSON lines')
    args = ap.parse_args(argv)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = open_webcam(source, args.width, args.height) if isinstance(source, int) else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f'cannot open {source}')
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    sio = socketio.Client()
    try:
        sio.connect(args.server, namespaces=[NS])
    except Exception as e:  # keep running without a server (useful for FPS tests)
        print(f'[capture] no server at {args.server} ({e}); running without emit')
        sio = None

    fc = FaceCapture(args.model, gpu=args.gpu, filter=args.filter)
    mirror = not args.no_mirror
    dump = open(args.dump, 'w') if args.dump else None
    n, t_start, t_infer = 0, time.perf_counter(), 0.0
    while True:
        t_frame = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        # filter and tracker run on media time for a file, so offline runs match real time
        t = time.perf_counter() - t_start if isinstance(source, int) else n / src_fps
        t1 = time.perf_counter()
        out = fc.process(frame, t, mirror=mirror)
        t_infer += time.perf_counter() - t1
        n += 1
        if out is not None and sio is not None:
            sio.emit('frame', {k: v for k, v in out.items() if k != 'landmarks'}, namespace=NS)
        if out is not None and dump is not None:
            dump.write(json.dumps({'t': t, 'raw': fc.raw, 'bs': out['bs'], 'head': out['head'],
                                   'gaze': out['gaze']}) + '\n')
        if n == 90 and isinstance(source, int) and n / (time.perf_counter() - t_start) < 20:
            print(f'[capture] WARNING: camera delivers {n / (time.perf_counter() - t_start):.1f} fps; '
                  'more light or a different --width/--height usually brings it back to 30')
        if n % 60 == 0:
            el = time.perf_counter() - t_start
            print(f'[capture] {n / el:5.1f} fps loop | {1000 * t_infer / n:5.1f} ms/frame inference'
                  + ('' if out is None else f' | jawOpen={out["bs"]["jawOpen"]:.2f} blinkL={out["bs"]["eyeBlinkLeft"]:.2f}'
                     f' head={np.round(out["head"], 1).tolist()}'))
        if not args.no_preview:
            view = cv2.flip(frame, 1) if mirror else frame
            if out is not None:
                cv2.putText(view, f'{fc.filter.kind}  head {np.round(out["head"], 0).astype(int).tolist()}  jaw {out["bs"]["jawOpen"]:.2f}',
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('capture (q quit, c zero head pose, f filter)', view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                fc.zero_head()
            if key == ord('f'):          # cycle filters to compare them live
                kinds = ('euro', 'none', 'ema')
                fc.filter = FrameFilter(ARKIT_52, kinds[(kinds.index(fc.filter.kind) + 1) % len(kinds)])
                print(f'[capture] filter: {fc.filter.kind}')
        if args.realtime and not isinstance(source, int):
            dt = 1.0 / src_fps - (time.perf_counter() - t_frame)
            if dt > 0:
                time.sleep(dt)
    cap.release()
    if dump is not None:
        dump.close()
    cv2.destroyAllWindows()
    fc.close()
    if sio is not None:
        sio.disconnect()
    el = time.perf_counter() - t_start
    print(f'[capture] done: {n} frames, {n / el:.1f} fps, {1000 * t_infer / max(n, 1):.1f} ms/frame inference')


if __name__ == '__main__':
    sys.exit(main())
