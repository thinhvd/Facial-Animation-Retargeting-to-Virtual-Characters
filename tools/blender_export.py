"""Runs inside Blender: a file Blender can open -> .glb for the pipeline.

    blender -b --factory-startup --python tools/blender_export.py -- in.fbx out.glb

.fbx / .dae / .obj / .glb / .gltf are imported into an empty scene, .blend
is opened as is.  The export keeps skins and shape keys (morph targets named
after the keys, in mesh.extras.targetNames) and drops animation, cameras and
lights; retarget.formats.gltf_convert then flattens it to the canonical
layout.  Normally called through retarget.formats.blender.export_glb.
"""
import sys

import bpy

argv = sys.argv[sys.argv.index('--') + 1:]
src, dst = argv[0], argv[1]
ext = src.lower().rsplit('.', 1)[-1]

if ext == 'blend':
    bpy.ops.wm.open_mainfile(filepath=src)
else:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if ext == 'fbx':
        bpy.ops.import_scene.fbx(filepath=src, use_anim=False)
    elif ext == 'dae':
        bpy.ops.wm.collada_import(filepath=src)
    elif ext == 'obj':
        bpy.ops.wm.obj_import(filepath=src)
    elif ext in ('glb', 'gltf'):
        bpy.ops.import_scene.gltf(filepath=src)
    else:
        sys.exit(f'blender_export: unsupported input .{ext}')

meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
keys = sum(len(o.data.shape_keys.key_blocks) - 1 for o in meshes if o.data.shape_keys)
arms = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']
print(f'blender_export: {len(meshes)} meshes, {len(arms)} armatures '
      f'({sum(len(a.data.bones) for a in arms)} bones), {keys} shape keys')

bpy.ops.export_scene.gltf(
    filepath=dst, export_format='GLB',
    use_visible=(ext == 'blend'),          # .blend preview scenes hide helper objects
    export_apply=False,                    # applying modifiers would drop the shape keys
    export_animations=False, export_skins=True,
    export_morph=True, export_morph_normal=False,
    export_yup=True, export_cameras=False, export_lights=False, export_extras=True)
print(f'blender_export: wrote {dst}')
