"""The canonical expression space shared by capture and every model.

A frame is a dict:

    {
      "t":    <float>  seconds (monotonic, capture side)
      "bs":   {<ARKit name>: <float in [0, 1]>, ...}   # ARKIT_52, missing = 0
      "head": [rx, ry, rz]   degrees, Euler XYZ in avatar space (avatar faces +Z, Y up)
      "gaze": [pitch, yaw]   degrees, eyeball rotation in avatar space
      "src":  "mediapipe" | "legacy" | ...
    }

Everything downstream (adapters, viewer, metrics) speaks this and nothing
else.  Left/Right are the *subject's* left/right, as in ARKit; mirroring
for a "mirror" style display is the capture side's job (see capture/).
"""
from __future__ import annotations

ARKIT_52: tuple[str, ...] = (
    'eyeBlinkLeft', 'eyeLookDownLeft', 'eyeLookInLeft', 'eyeLookOutLeft', 'eyeLookUpLeft',
    'eyeSquintLeft', 'eyeWideLeft',
    'eyeBlinkRight', 'eyeLookDownRight', 'eyeLookInRight', 'eyeLookOutRight', 'eyeLookUpRight',
    'eyeSquintRight', 'eyeWideRight',
    'jawForward', 'jawLeft', 'jawRight', 'jawOpen',
    'mouthClose', 'mouthFunnel', 'mouthPucker', 'mouthLeft', 'mouthRight',
    'mouthSmileLeft', 'mouthSmileRight', 'mouthFrownLeft', 'mouthFrownRight',
    'mouthDimpleLeft', 'mouthDimpleRight', 'mouthStretchLeft', 'mouthStretchRight',
    'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper',
    'mouthPressLeft', 'mouthPressRight', 'mouthLowerDownLeft', 'mouthLowerDownRight',
    'mouthUpperUpLeft', 'mouthUpperUpRight',
    'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
    'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
    'noseSneerLeft', 'noseSneerRight',
    'tongueOut',
)
assert len(ARKIT_52) == 52

# Groups used by adapters and by the coverage report.
REGIONS: dict[str, tuple[str, ...]] = {
    'eye': tuple(n for n in ARKIT_52 if n.startswith('eye')),
    'brow': tuple(n for n in ARKIT_52 if n.startswith('brow')),
    'mouth': tuple(n for n in ARKIT_52 if n.startswith(('mouth', 'jaw', 'tongue'))),
    'cheek_nose': tuple(n for n in ARKIT_52 if n.startswith(('cheek', 'nose'))),
}

# Blendshapes that are better served by a bone than by a morph when the model has one.
GAZE_SHAPES = ('eyeLookUpLeft', 'eyeLookDownLeft', 'eyeLookInLeft', 'eyeLookOutLeft',
               'eyeLookUpRight', 'eyeLookDownRight', 'eyeLookInRight', 'eyeLookOutRight')

SEMANTIC_BONES = ('head', 'neck', 'leftEye', 'rightEye', 'jaw')


def mirror_name(name: str) -> str:
    if name.endswith('Left'):
        return name[:-4] + 'Right'
    if name.endswith('Right'):
        return name[:-5] + 'Left'
    return name


def empty_frame(src: str = 'none') -> dict:
    return {'t': 0.0, 'bs': {}, 'head': [0.0, 0.0, 0.0], 'gaze': [0.0, 0.0], 'src': src}
