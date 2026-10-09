# Canonical retargeting pipeline (P0)

Video → MediaPipe → **canonical frame** → Socket.IO → viewer → per-model **adapter** → morph / bone.

```
retarget/
  canonical.py        ARKit-52 names, frame schema, regions
  naming.py           MMD / VRM / ARKit name tables, bone aliases
  mesh.py             CanonicalMesh: positions, triangles, materials, bones, skin, morphs (metres, Y up, +Z front)
  formats/pmx.py      PMX 2.x reader (vertices, faces, materials, bones, morphs)
  formats/pmx_convert.py   PMX -> CanonicalMesh (axis remap, MMD units -> metres)
  formats/gltf_writer.py   CanonicalMesh -> .glb (skin, morph targets with targetNames, KHR_materials_unlit)
  convert.py          CLI: model -> web/models/<name>.glb
  adapter.py          CLI: .glb -> <name>.adapter.json (bones, morph drivers, coverage report)
  server.py           aiohttp static server + Socket.IO relay (namespace /retarget)
  capture/mediapipe_capture.py   webcam/video -> canonical frames
web/viewer.html       three.js r186 + three-vrm: loads .glb/.vrm + adapter, applies frames, slider panel
tools/validate_glb.js gltf-validator wrapper
```

## Setup

```bash
conda env create -f environment.yml && conda activate fr
curl -L -o weights/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
npm i -g gltf-validator          # optional, for tools/validate_glb.js
```

## Convert a model

```bash
python -m retarget.convert NodeServer/models/headmodel/HeadA.pmx            # -> web/models/HeadA.glb
python -m retarget.convert NodeServer/models/model1/kizunaai.pmx
# a PMX exported from a Z-up rig (mmd_tools), 0.18 units tall:
python -m retarget.convert NodeServer/models/HumanModel/MaleModel/MaleModel.pmx --up=-z --front=-y --scale 10
python -m retarget.adapter web/models/*.glb                                 # -> *.adapter.json + coverage
node tools/validate_glb.js web/models/*.glb
```

`.glb` and `.vrm` files are copied through; put them in `web/models/` and run the adapter step.
FBX/OBJ/blend need Blender (not wired yet; see proposal P0).

## Run

```bash
python -m retarget.server                      # http://127.0.0.1:6790/
python -m retarget.capture.mediapipe_capture 0 # webcam; or a video path (+ --realtime)
```

Open the viewer, pick a model, Load.  Keys in the capture window: `q` quit, `c` zero the head pose.
With "Drive from sliders" ticked, the viewer ignores the socket and uses the panel sliders —
useful to check a model's adapter without a camera.

## Canonical frame

```json
{"t": 12.34, "bs": {"jawOpen": 0.31, "...": 0}, "head": [rx, ry, rz], "gaze": [pitch, yaw], "src": "mediapipe"}
```

* `bs`: the 52 ARKit blendshape names, subject's left/right, values in [0, 1]
  (MediaPipe provides 51 of them; `tongueOut` is always 0).
* `head`: Euler XYZ degrees applied to the head bone of a +Z-facing avatar.
* `gaze`: eyeball pitch/yaw degrees, derived from the `eyeLook*` scores.
* Mirror mode (default) swaps left/right and negates yaw/roll so the avatar copies the user like a mirror.

## Adapter

```json
{"bones": {"head": "頭", "leftEye": "左目", "rightEye": "右目"},
 "entries": [{"kind": "morph", "target": "あ", "inputs": ["jawOpen"], "combine": "max", "gain": 1.0},
             {"kind": "morph", "target": "ウィンク", "inputs": ["eyeBlinkLeft", "eyeBlinkRight"], "combine": "diff"}],
 "coverage": {"jawOpen": "morph:あ", "browInnerUp": "morph:上", "cheekPuff": null},
 "coverageSummary": {"covered": 34, "total": 52}}
```

Influence of a target = clamp(gain × combine(inputs)); `combine` ∈ max | min | mean | diff.
Gaze uses eye bones when both exist, otherwise the `eyeLook*` morphs.

Left/right of one-sided morphs is **verified against the geometry** (centroid of the displaced
vertices vs. the face centre; the character's left is +X). Name tables alone are not enough:
`kizunaai.pmx` names its winks from the viewer's side, HeadA and model2–4 from the character's.
Entries that were flipped carry `"lateralityFixed": <offset ratio>`.

## Checking without a browser

```bash
python tools/screenshot_viewer.py HeadA --set jawOpen=1 eyeBlinkLeft=1 browInnerUp=1 --head 0,30,0 --out /tmp/shots
python tools/screenshot_viewer.py HeadA --live      # shoot whatever the socket delivers
```

Renders `web/viewer.html` in headless Chromium (SwiftShader) and writes a neutral and a posed PNG.

## Current coverage (P0)

| Model | Morphs | Head | Gaze | Coverage |
| --- | --- | --- | --- | --- |
| HeadA.pmx | 54 | bone 頭 | bones | 34/52 (eye 14/14, brow 5/5, mouth 13/28) |
| kizunaai.pmx | 33 | bone 頭 | bones | 28/52 |
| MaleModel.pmx | 0 | bone Head | none | 0/52 — P1 (template + deformation transfer) fills this |
