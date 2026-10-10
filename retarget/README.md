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
  formats/gltf_convert.py  any .glb/.gltf -> CanonicalMesh (flattens meshes, node transforms, skins, morphs by name)
  formats/blender.py       Blender subprocess for .fbx/.blend/.dae (and Draco/meshopt .glb)
  formats/orient.py        guess up/front axes and unit scale from the skeleton
  convert.py          CLI: model -> web/models/<name>.glb
  adapter.py          CLI: .glb -> <name>.adapter.json (bones, morph drivers, coverage report)
  server.py           aiohttp static server + Socket.IO relay (namespace /retarget)
  capture/mediapipe_capture.py   webcam/video -> canonical frames
  capture/filters.py  One Euro / EMA per-channel filters
web/viewer.html       three.js r186 + three-vrm: loads .glb/.vrm + adapter, applies frames, slider panel
web/landmarks.html    hand-picked landmarks when MediaPipe sees no face (P3)
web/rigjob.js         Generate-blendshapes job client shared by both pages (server: /api/rig/<model>)
tools/validate_glb.js gltf-validator wrapper
tools/blender_export.py  runs inside Blender: import a scene file, export .glb with skins + shape keys
```

## Setup

```bash
conda env create -f environment.yml && conda activate fr
curl -L -o weights/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
npm i -g gltf-validator          # optional, for tools/validate_glb.js
# optional, for .fbx / .blend / .dae: portable Blender 4.5 LTS into data/blender/ (no root; ~1 GB unpacked)
mkdir -p data/blender && cd data/blender && \
  curl -LO https://mirror.freedif.org/blender/release/Blender4.5/blender-4.5.14-linux-x64.tar.xz && \
  tar -xJf blender-4.5.14-linux-x64.tar.xz && rm blender-4.5.14-linux-x64.tar.xz && cd ../..
```

`retarget/formats/blender.py` looks for `$BLENDER`, then `blender` on PATH, then `data/blender/*/blender`.

## Convert a model

```bash
python -m retarget.convert NodeServer/models/headmodel/HeadA.pmx            # -> web/models/HeadA.glb
python -m retarget.convert NodeServer/models/model1/kizunaai.pmx
python -m retarget.convert NodeServer/models/HumanModel/FemaleModel/FemaleModel.fbx   # via Blender
python -m retarget.adapter web/models/*.glb                                 # -> *.adapter.json + coverage
node tools/validate_glb.js web/models/*.glb
```

| Input | Path |
| --- | --- |
| .pmx, .obj | pure Python |
| .glb, .gltf | pure Python, flattened to one mesh (`gltf_convert`) |
| .fbx, .blend, .dae; .glb with Draco / meshopt | Blender → .glb → `gltf_convert` |
| .vrm | copied through (the writer would drop MToon, spring bones, expressions) |
| .pmd | not supported (needs the MMD Tools add-on in Blender) |

**Axes and units** are guessed from the skeleton when `--up/--front/--scale` are not given:
up = head bone minus hips/neck, left = mean of (left − right) over bones paired by name
(`Left*`, `左*`, `*_L`, `*.L`), front = left × up, and a power-of-ten scale when the height is
implausible (0.5–4 m with leg bones, 0.1–1 m for a head).  On the repo models this leaves the six
correct MMD models alone and fixes the three Z-up exports (MaleModel.pmx, testFemale1.pmx,
FemaleModel.fbx, all `up=+z front=-y`, ×10 / ×100); the orientation matches the hand-set flags to 1e-7.
Explicit flags still win (`--up=-z --front=-y --scale 10` for MaleModel.pmx as before).

`gltf_convert` bakes the rest pose as three.js draws it: skinned vertices through
world(joint) · inverseBind, rigid meshes through their node, and binds rigid meshes that hang below a
joint (glasses under the head) to that joint.  Round trip on the canonical HeadA/model2 .glb is exact.
Checked on external samples from the three.js repo: RobotExpressive (14 meshes, 2 skins, rigid parts
parented to bones) and Xbot (Mixamo).  Mixamo bone names lose the `:` in three.js
(`mixamorig:Head` → `mixamorigHead`); the viewer looks up both spellings.

## Run

```bash
python -m retarget.server                      # http://127.0.0.1:6790/
python -m retarget.capture.mediapipe_capture 0 # webcam; or a video path (+ --realtime)
```

Open the viewer, pick a model, Load.  Keys in the capture window: `q` quit, `c` zero the head pose,
`f` cycle the filter (euro → none → ema, shown top-left) to compare live.
The viewer's **Input** box chooses what drives the model: **Live capture** (frames from the
capture process over the socket) or **Sliders** (head, gaze and the 52 inputs by hand, to check a
model's adapter without a camera; live frames are ignored meanwhile).  The sliders are greyed out and
locked unless Input is Sliders, and switching back to Live with no capture running returns the
model to the neutral face.

Capture filters `bs` and `head` per channel with a One Euro filter (`retarget/capture/filters.py`;
`--filter ema|none` for comparison), on media time for video files, so the viewer's "Extra
smoothing" stays at 0.  `--dump frames.jsonl` writes raw and filtered values per frame.
Defaults: blendshapes `min_cutoff 1.5 Hz, beta 5`, head `0.7 Hz, 0.2`.  On synthetic noise
(0.4° head) this gives the jitter of a heavy EMA (0.18° vs 0.14° for smoothing 0.8) with the
step latency of the old light one (90 % in one frame vs 333 ms).

The webcam opens as MJPG at 30 fps with `exposure_auto_priority=0` (Linux, V4L2).  With priority 1
the test webcam stretched its exposure in room light and fell to 7.7 fps (30.3 fps with 0); at that
rate a blink is one frame and no filter setting reduced shimmer without delaying blinks.

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
| identity 7 | same as template (26,719 v, with interior shells) | 0.57 mm (0.6 % IOD) | 0.82 mm | mouthRollUpper 2.2 mm |
| identity 7, remesh 50 % | different (5,648 v) | 0.23 mm (0.3 % IOD) | 0.55 mm | mouthClose 1.6 mm |
| identity 23, remesh 30 % | different (3,399 v) | 0.27 mm (0.3 % IOD) | 0.66 mm | mouthClose 1.7 mm |
| identity 42, coeff 1.5 | same as template | 0.65 mm (0.7 % IOD) | 0.90 mm | mouthRollUpper 2.5 mm |

(P3 registration fixes, below, took these from 0.70 / 0.34 / 0.35 / 0.75 mm.)

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
adapter again.  When MediaPipe finds no face in the render (stylised faces such as model4.pmx, robots,
faceless mannequins) the importer stops after the adapter: the model still drives its own morphs and
bones, and the missing blendshapes need hand-picked landmarks. The adapter ranks sources per canonical input:

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
| HeadB.pmx (first import, no flags) | 34/52 | 52/52 | 26 artist morphs, 8 bone, 18 generated; 56 s |
| model3.pmx (full-body anime) | 26/52 | 51/52 | 18 artist morphs, 8 bone, 25 generated; 64 s |
| FemaleModel.fbx (Z-up, 0.17 m; auto-oriented, ×10) | 0/52 | 51/52 | 51 generated; Blender 1.5 s + generate 47 s; no eye bones, so no eyeball gaze |
| model4.pmx (anime) | 22/52 | 22/52 → 51/52 with clicks | MediaPipe finds no face (also on tight crops); 13 hand-picked points, see P3 |

VRM files are not rigged: the writer would drop the VRM extensions (MToon, spring bones). Their
custom morphs cover most of what the presets miss.

Resource note: generate (~1 min, a few GB for large meshes) and each headless-Chromium render
(~1–2 GB) should run one at a time on a 16 GB machine; running several in parallel has crashed
VS Code. Start long jobs detached (`setsid nohup python -u -m ... > log 2>&1 &`).

## P3: hand-picked landmarks (web/landmarks.html)

When MediaPipe finds no face in the front render (model4.pmx and most anime faces; cropping does not
help), a person places 19 points on the same render:

In the browser, the viewer's **Face rig** box shows where the selected model stands and offers only
the next step (the server computes the state, `rig_state` in `server.py`, and refuses any POST the
state does not allow):

| State | Box offers |
| --- | --- |
| VRM, or own morphs cover every input but tongueOut | nothing to do |
| never tried | **Generate blendshapes** (MediaPipe first; about 1 min, then loads `<model>.rigged`) |
| MediaPipe found no face (`<model>.landmarks_failed.json`) | **Pick landmarks →** only |
| landmarks saved, result missing or older than them | **Generate blendshapes** |
| result newer than the model and its landmarks | **Load `<model>.rigged`**, and *Edit landmarks…* to correct points |
| job running | progress; everything locked |

On the landmark page **Save** is enabled only with unsaved changes (and the eye corners, nose, mouth
corners and chin placed), **Generate** only when the saved points are newer than the result, **Open in
viewer** only when the result matches the points; editing is locked while a job runs, and leaving with
unsaved changes asks first.  One job runs at a time; a lock file (`<model>.rig.lock`) also keeps a shell
`importer` from writing the same files.  Generating twice from the same points gives identical morphs
(checked: max difference 0.0 over 106 targets), so a repeat only costs the minute; the UI does not
offer it (`POST /api/rig/<model>?force=1` does).

The same from the shell:

```bash
python -m retarget.importer NodeServer/models/model4/model4.pmx     # prints the picker link on failure
# open http://127.0.0.1:6790/landmarks.html?model=model4 : click, skip what is hidden, Save
python -m retarget.importer --existing model4                        # rig with the saved points
python -m retarget.rig.landmarks <model> --render-only               # render for a model never imported
```

The page shows the render (wheel zoom, right-drag pan) and the ICT template with the next point in
red.  Points: 4 eye corners, 4 lid midpoints, 4 brow ends, nose tip, 2 mouth corners, top of the upper
lip, the mouth line centre (stands for MediaPipe 13 and 14), bottom of the lower lip, front of the chin
(`CLICK_POINTS` in `rig/landmarks.py`).  Outline points (MediaPipe 234/454, chin bottom 152) are left out:
the ray grazes the surface there and a few pixels move the 3D point by centimetres.  Saved clicks go
through the same ray cast as MediaPipe's, under their MediaPipe indices, so registration is unchanged.
A click whose ray hits nothing, or lands far behind the other points (through an eye socket), comes
back red.  With MediaPipe landmarks present the page pre-fills them (orange) for correction.

Simulated clicks on the gate (MediaPipe pixels of the 19 points + Gaussian noise, `evaluate --clicks
--click-noise 3`): identity 7 remesh 50 % 0.24 mm, identity 23 remesh 30 % 0.25 mm, i.e. as good as
478 detected points, and flat from 0 to 6 px of noise.

Getting there needed three registration fixes, which also improved the automatic path:

* **Template landmarks bound to the fitted skin.**  Each template landmark used the vertex of the triangle
  its ray hit; for eye corners and lids that is ICT's lash / eye shell, outside the fitted subset, so the
  eyes were never constrained.  Now the nearest subset vertex within 5 mm (468 of 478 landmarks instead
  of 421).
* **Only similarity inliers constrain ICP** for MediaPipe sets (rays through an eye hole used to pull).
* **Hand-picked sets are trusted as a whole.**  On model4 a similarity explains only 4 of 14 clicked
  points (the eyes are far larger than a human's); treating the rest as outliers left them out of ICP.
  Sparse sets (< 100 points) now constrain ICP with every point; bad clicks are caught at the ray cast.

model4.pmx, all 19 points clicked by hand: registered template landmarks within 0.6 px of the clicks,
coverage 22 → 51/52 (14 artist morphs, 8 bone, 29 generated).  Generated mouth shapes (pucker, funnel)
read well; generated eye shapes (squint, wide) barely show, since ICT moves a human lid by millimetres
and the anime eye is a large textured piece.  Blinks come from the model's own morphs.

That run exposed two problems with hair and hidden geometry, fixed for every model:

* **Ray cast = what the render shows.**  Rays skip fully transparent materials (MMD expression overlays
  such as 青ざめ / 紅潮, alpha 0 until a material morph shows them), and each ray takes its first hit on
  the mesh pieces the core points (eye corners, nose, mouth corners, chin) landed on.  Brows clicked on
  the fringe now land on the forehead behind it (z 0.111 → 0.103 m on model4).
* **Hair stays rigid.**  A floating piece rides on the skin only if ≥ 80 % of it lies in the solved region
  (brows, lashes); the fringe used to bend with the brows: model4 前髪 moved 4.9 mm on 75 vertices in
  browInnerUp, model3 前髪 5.6 mm on 171 vertices in browDown, more than the brows themselves.  Now 0 on
  model4 and 8 vertices on model3, while model3's brows move 4 mm instead of 2.  Extra fitting pieces
  must also be ≥ 20 % of the main piece's size, so lashes and eye highlights hit by eye-corner points do
  not count as skin.

MediaPipe on anime renders is luck: it found model3's face while the iris texture was missing (white
eyes) and finds nothing now that the eyes render; model3 keeps its earlier landmarks, re-projected with
the new ray cast.  The viewer no longer overwrites its `loaded …` status when the socket connects after
a fast load (that race made `render_front` time out at random).

## Current coverage (P0)

| Model | Morphs | Head | Gaze | Coverage |
| --- | --- | --- | --- | --- |
| HeadA.pmx | 54 | bone 頭 | bones | 34/52 (eye 14/14, brow 5/5, mouth 13/28) |
| kizunaai.pmx | 33 | bone 頭 | bones | 28/52 |
| MaleModel.pmx | 0 | bone Head | none | 0/52 — P1 (template + deformation transfer) fills this |
