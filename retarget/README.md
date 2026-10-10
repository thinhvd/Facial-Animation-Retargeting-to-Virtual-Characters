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

## P1: generating blendshapes for a model without morphs

```
retarget/rig/ict.py          ICT-FaceKit template (data/ict-facekit, MIT): neutral + 53 expressions + regions
retarget/rig/render.py       front render through the viewer (headless Chromium) + camera description
retarget/rig/landmarks.py    render -> MediaPipe 478 landmarks -> ray cast -> 3D points on the mesh
retarget/rig/registration.py similarity (Umeyama) + non-rigid ICP (Amberg 2007), sparse least squares
retarget/rig/transfer.py     deformation transfer (Sumner & Popović 2004): gradients, smoothness, anchors, attach
retarget/rig/generate.py     the pipeline: register -> classify vertices -> correspond -> solve 53 shapes -> .rigged.glb
retarget/rig/evaluate.py     gate test on held-out ICT identities (optionally remeshed)
```

```bash
git clone --depth 1 https://github.com/USC-ICT/ICT-FaceKit data/ict-facekit
python -m retarget.convert data/ict-facekit/FaceXModel/generic_neutral_mesh.obj -o web/models/ict_neutral.glb --scale 0.01
python -m retarget.rig.landmarks ict_neutral MaleModel        # server must be running
python -m retarget.rig.generate MaleModel                      # -> web/models/MaleModel.rigged.glb (~1 min)
python -m retarget.adapter web/models/MaleModel.rigged.glb     # coverage 51/52 (no tongueOut in ICT)
python -m retarget.rig.evaluate --identity 7 --remesh 0.5      # held-out identity, different topology
```

How `generate` treats the target (all on a position-welded copy of the mesh):

| Class | Rule | Constraint |
| --- | --- | --- |
| exterior | largest connected component, < 1.2 cm from the registered template, normals agree | gradient = template gradient |
| attachable | other pieces that satisfy the same test (eyebrows, lashes) | keep rest offset to nearest exterior vertex |
| ring | skin in the outer 1 cm of the 4 cm region | anchored at rest |
| free | everything else inside (mouth socket, gums, tongue, eyelid insides) | smoothness only |
| rigid shells | small watertight components (eyeballs, teeth) | excluded |

Things that went wrong on the way and are now handled: duplicate vertices along UV seams tore the
mesh (weld first); interior shells matched to face triangles and anchored deep in the mouth pulled the
lips together (classify by component + normal); floating eyebrow shells had no anchor and flew away
(attach + Tikhonov regularisation); sliver triangles blew up the gradients (area threshold).

Gate results (identity held out, blendshapes hidden, error over face vertices the truth moves):

| Target | Topology | Mean error | p95 | Worst |
| --- | --- | --- | --- | --- |
| identity 7 | same as template (26,719 v, with interior shells) | 0.70 mm (0.8 % IOD) | 0.93 mm | mouthRollUpper 3.5 mm |
| identity 7, remesh 50 % | different (5,648 v) | 0.34 mm (0.4 % IOD) | 0.85 mm | mouthClose 1.8 mm |
| identity 23, remesh 30 % | different (3,399 v) | 0.35 mm (0.4 % IOD) | 0.86 mm | mouthClose 1.5 mm |
| identity 42, coeff 1.5 | same as template | 0.75 mm (0.8 % IOD) | 1.00 mm | mouthRollUpper 3.8 mm |

The same-topology targets score worse only because they carry ICT's interior shells (gums, tongue
move up to 30 mm in jawOpen and are left free); on the face skin alone the numbers match the remeshed
runs. Oracle check (identity correspondence on the template itself): 0.04 mm, so the solver is exact
and the remaining error is registration + correspondence. Results are in `data/eval/*.json`.

## P2: one-command import and the hybrid adapter

```bash
python -m retarget.importer NodeServer/models/model2/model2.pmx                 # MMD: morphs + generated gaps
python -m retarget.importer path/to/avatar.vrm                                   # VRM: presets + custom morphs
python -m retarget.importer scan.obj --rig                                       # no rig at all: everything generated
```

`importer` = convert → adapter → (if brow+mouth coverage < 60 % and not a VRM) landmarks → generate →
adapter again. The adapter ranks sources per canonical input:

1. VRM expression presets (`aa`, `blink`, `lookUp`, …; emotion presets `happy/angry/sad/relaxed/surprised`
   are off by default because they move eyes + mouth + brows at once — `--emotions` enables them)
2. artist morphs by name: MMD Japanese table, VRM presets, English/VRoid table (`lip_a`, `blink_L`,
   `eye_brow_up_L`, `Fcl_MTH_A`, …), ARKit-named targets
3. bones for head and gaze
4. generated blendshapes (`targetMeta.generated`) — only for inputs nothing above covers
5. rigid fallback (whole model rotates when there is no head bone)

| Model | Before | After (hybrid) | Sources |
| --- | --- | --- | --- |
| HeadA.pmx | 34/52 | 52/52 | 26 artist morphs, 8 bone, 18 generated |
| Seed-san.vrm (VRM 1.0 sample) | 24/52 (presets only) | 31/52 | 17 presets, 14 custom morphs |
| MaleModel.pmx | 0/52 | 51/52 | 51 generated |
| model2.pmx (MMD, face is a separate mesh piece) | 24/52 | 51/52 | 16 artist morphs, 8 bone, 27 generated |
| testFemale1.pmx (`--up=-z --front=-y --scale 10`) | 0/52 | 51/52 | 51 generated, one command |

VRM files are not rigged: the writer would drop the VRM extensions (MToon, spring bones). Their
custom morphs cover most of what the presets miss.

Resource note: generate (~1 min, a few GB for large meshes) and each headless-Chromium render
(~1–2 GB) should run one at a time on a 16 GB machine; running several in parallel has crashed
VS Code. Start long jobs detached (`setsid nohup python -u -m ... > log 2>&1 &`).

## Current coverage (P0)

| Model | Morphs | Head | Gaze | Coverage |
| --- | --- | --- | --- | --- |
| HeadA.pmx | 54 | bone 頭 | bones | 34/52 (eye 14/14, brow 5/5, mouth 13/28) |
| kizunaai.pmx | 33 | bone 頭 | bones | 28/52 |
| MaleModel.pmx | 0 | bone Head | none | 0/52 — P1 (template + deformation transfer) fills this |
