"""Generate ARKit-named blendshapes for a converted model from the ICT template.

    python -m retarget.rig.generate MaleModel          # needs web/models/MaleModel.landmarks.json
                                                       # and web/models/ict_neutral.landmarks.json
Writes web/models/<model>.rigged.glb (+ .adapter.json) with the 53 generated
morph targets appended after any existing ones, and a .rig.json report.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

from ..formats.gltf_reader import GLB
from ..formats.gltf_writer import write_glb
from ..mesh import CanonicalBone, CanonicalImage, CanonicalMaterial, CanonicalMesh, CanonicalMorph
from . import ict
from .geometry import SurfaceQuery
from .registration import apply_h, register, vertex_normals
from .transfer import build_solver, correspond_triangles, deformation_gradients, frame_inverse

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / 'web' / 'models'


def load_landmarks(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = json.loads((MODELS / f'{name}.landmarks.json').read_text())
    return np.array(d['points'], np.float64), np.array(d['valid'], bool), np.array(d['nearest_vertex'], np.int64)


def glb_to_canonical(glb: GLB) -> CanonicalMesh:
    """Rebuild a CanonicalMesh from a canonical .glb (one mesh, shared attributes)."""
    j = glb.json
    mesh = j['meshes'][0]
    prim0 = mesh['primitives'][0]
    pos = glb.accessor(prim0['attributes']['POSITION']).astype(np.float32)
    nrm = glb.accessor(prim0['attributes']['NORMAL']).astype(np.float32) if 'NORMAL' in prim0['attributes'] else np.zeros_like(pos)
    uv = glb.accessor(prim0['attributes']['TEXCOORD_0']).astype(np.float32) if 'TEXCOORD_0' in prim0['attributes'] else np.zeros((len(pos), 2), np.float32)
    tris, prims = [], []
    start = 0
    for p in mesh['primitives']:
        t = glb.accessor(p['indices']).reshape(-1, 3).astype(np.uint32)
        tris.append(t)
        prims.append((p.get('material', 0), start, start + len(t)))
        start += len(t)
    materials = []
    for m in j.get('materials', []):
        pbr = m.get('pbrMetallicRoughness', {})
        img = pbr.get('baseColorTexture', {}).get('index', -1)
        if img >= 0:
            img = j['textures'][img]['source']
        materials.append(CanonicalMaterial(m.get('name', ''), np.array(pbr.get('baseColorFactor', [1, 1, 1, 1]), np.float32),
                                           img, m.get('doubleSided', False), m.get('alphaMode', 'OPAQUE'),
                                           'KHR_materials_unlit' in m.get('extensions', {}), m.get('extras', {})))
    images = []
    for im in j.get('images', []):
        bv = j['bufferViews'][im['bufferView']]
        data = glb.bin[bv.get('byteOffset', 0): bv.get('byteOffset', 0) + bv['byteLength']]
        images.append(CanonicalImage(im.get('name', ''), bytes(data), im.get('mimeType', 'image/png'), True))
    bones, joints, weights = [], None, None
    if 'skins' in j:
        skin = j['skins'][0]
        node_ids = skin['joints']
        index_of = {nid: k for k, nid in enumerate(node_ids)}
        parent_of = {}
        for nid, node in enumerate(j['nodes']):
            for c in node.get('children', []):
                parent_of[c] = nid
        world = {}
        def world_pos(nid):
            if nid in world:
                return world[nid]
            t = np.array(j['nodes'][nid].get('translation', [0, 0, 0]), np.float32)
            p = parent_of.get(nid)
            world[nid] = t + (world_pos(p) if p is not None and p in index_of else 0)
            return world[nid]
        for nid in node_ids:
            node = j['nodes'][nid]
            bones.append(CanonicalBone(node.get('name', ''), node.get('extras', {}).get('name_en', ''),
                                       index_of.get(parent_of.get(nid), -1), world_pos(nid)))
        joints = glb.accessor(prim0['attributes']['JOINTS_0']).astype(np.int32)
        weights = glb.accessor(prim0['attributes']['WEIGHTS_0']).astype(np.float32)
    morphs = [CanonicalMorph(n, d, meta) for (n, d), meta in
              zip(glb.mesh_morphs().items(), mesh.get('extras', {}).get('targetMeta', [{}] * 999))]
    return CanonicalMesh(mesh.get('name', glb.path.stem), pos, nrm, uv, np.concatenate(tris), prims, materials,
                         images, bones, joints, weights, morphs, j.get('asset', {}).get('extras', {}))


def weld(pos: np.ndarray, tris: np.ndarray, decimals: int = 6) -> trimesh.Trimesh:
    """Mesh with coincident vertices merged (faces keep their order); carries the
    original vertex ids per welded vertex in `metadata['orig']`."""
    key = np.round(pos, decimals)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    m = trimesh.Trimesh(pos[first], inverse[tris], process=False)
    m.metadata['inverse'] = inverse
    return m


def face_components(groups, tris: np.ndarray, inverse: np.ndarray, lm_vertices: np.ndarray,
                    min_share: float = 0.1) -> np.ndarray:
    """Face indices of the connected components that hold the landmarks.

    groups: list of face-index arrays (connected components of the welded mesh);
    lm_vertices: ORIGINAL vertex ids nearest to each valid landmark.  Returns the
    faces of the component with most landmarks plus any component holding at
    least `min_share` of them (a face split into upper/lower pieces).
    """
    if len(groups) <= 1:
        return np.arange(len(tris))
    wtris = inverse[tris]
    comp_of_wvertex = np.full(int(inverse.max()) + 1, -1, np.int64)
    for gi, g in enumerate(groups):
        verts = np.unique(wtris[g])
        unset = comp_of_wvertex[verts] < 0
        comp_of_wvertex[verts[unset]] = gi
    hit = comp_of_wvertex[inverse[lm_vertices]]
    hit = hit[hit >= 0]
    if len(hit) == 0:
        return max(groups, key=len)
    counts = np.bincount(hit, minlength=len(groups))
    chosen = [gi for gi in range(len(groups)) if counts[gi] >= max(3, min_share * counts.sum())]
    return np.concatenate([groups[gi] for gi in chosen])


def rigid_shells(welded: trimesh.Trimesh, max_fraction: float = 0.2) -> np.ndarray:
    """Boolean mask over the ORIGINAL vertices lying on small watertight components
    (eyeballs, teeth) of the welded mesh."""
    inverse = welded.metadata['inverse']
    mask_w = np.zeros(len(welded.vertices), bool)
    groups = trimesh.graph.connected_components(welded.face_adjacency, nodes=np.arange(len(welded.faces)))
    if len(groups) > 1:
        biggest = max(len(g) for g in groups)
        for g in groups:
            if len(g) >= max_fraction * biggest:
                continue
            sub = trimesh.Trimesh(welded.vertices, welded.faces[g], process=False)
            if sub.is_watertight:
                mask_w[np.unique(welded.faces[g])] = True
    return mask_w[inverse]


def template_subset(tmpl: ict.ICTTemplate, subset: str) -> np.ndarray:
    """Which template vertices take part in registration / correspondence."""
    if subset == 'fitting':
        return tmpl.fitting_verts
    if subset == 'face':
        return tmpl.region('face')
    if subset == 'face+neck':
        return np.concatenate([tmpl.region('face'), tmpl.region('head_neck')])
    raise ValueError(subset)


def generate(model: str, template_name: str = 'ict_neutral', face_dist: float = 0.012,
             anchor_dist: float = 0.04, log=print, skip_existing: bool = True,
             subset: str = 'face', smooth_free: float = 0.5, smooth_corr: float = 0.05,
             identity_w: float = 0.05, identity_skin: float = 0.3, stiff_min: float = 1.0,
             ring_width: float = 0.01) -> dict:
    t0 = time.time()
    tmpl = ict.load()
    tmpl_lm, tmpl_valid, tmpl_lm_vertex = load_landmarks(template_name)
    tgt_lm, tgt_valid, tgt_nearest = load_landmarks(model)

    glb = GLB(MODELS / f'{model}.glb')
    mesh = glb_to_canonical(glb)
    tpos = mesh.positions.astype(np.float64)
    ttris = mesh.triangles.astype(np.int64)
    whole = trimesh.Trimesh(tpos, ttris, process=False)
    # Connectivity on a position-welded copy: PMX/OBJ duplicate vertices along UV seams,
    # which would otherwise cut the skin into dozens of "parts".
    welded = weld(tpos, ttris)
    groups = trimesh.graph.connected_components(welded.face_adjacency, nodes=np.arange(len(ttris)))
    # The face is often its own mesh piece (MMD models split face / body / hair), so "largest
    # component" is wrong; take the pieces the landmarks landed on instead.
    biggest = face_components(groups, ttris, welded.metadata['inverse'], tgt_nearest[tgt_valid])
    skin = trimesh.Trimesh(tpos, ttris[biggest], process=False)
    log(f'[{model}] target V={len(tpos)} F={len(ttris)}; fitting surface F={len(biggest)} '
        f'(landmark-selected) of {len(groups)} parts')

    # ---- registration (template subset = ICT fitting vertices)
    log('[register]')
    stiffness = tuple(a for a in (50.0, 20.0, 10.0, 5.0, 2.0, 1.0) if a >= stiff_min) or (stiff_min,)
    reg = register(tmpl.positions.astype(np.float64), tmpl.triangles.astype(np.int64),
                   template_subset(tmpl, subset), tmpl_lm, tmpl_valid, skin, tgt_lm, tgt_valid,
                   tmpl_lm_vertex, log=log, stiffness=stiffness)

    # deformed template subset as a mesh
    keep = np.zeros(tmpl.vertex_count, bool)
    keep[reg.template_ids] = True
    remap = -np.ones(tmpl.vertex_count, np.int64)
    remap[reg.template_ids] = np.arange(len(reg.template_ids))
    sub_tris = remap[tmpl.triangles[keep[tmpl.triangles].all(1)]]
    sub_rest_tmpl = apply_h(reg.similarity, tmpl.positions[reg.template_ids].astype(np.float64))

    # ---- transfer works on the WELDED mesh (duplicates along UV seams would tear apart otherwise);
    # displacements are copied back to every duplicate at the end.
    wpos = np.asarray(welded.vertices, np.float64)
    wtris = np.asarray(welded.faces, np.int64)
    inverse = welded.metadata['inverse']
    dquery = SurfaceQuery(reg.deformed, sub_tris)
    _, dist_all, tri_near = dquery.closest(wpos)
    wnormals = vertex_normals(wpos, wtris)
    agree = (wnormals * dquery.face_normals[tri_near]).sum(1)      # +1: same facing as the template skin
    rigid_v = rigid_shells(welded, max_fraction=0.2)
    rigid_w = np.zeros(len(wpos), bool)
    rigid_w[inverse[rigid_v]] = True
    if rigid_w.any():
        log(f'[transfer] excluding {rigid_w.sum()} welded vertices on small closed shells (eyeballs/teeth)')
    # Vertex classes (all on the welded mesh, `skin_w` = largest connected component):
    #   exterior:   skin near the template, facing the same way   -> gradient (data) constraints
    #   attachable: floating pieces on the skin (brows, lashes)   -> ride on the nearest exterior vertex
    #   ring:       skin at the far rim of the solved region      -> anchored at rest
    #   everything else inside the region (mouth socket, gums, tongue, eyelid insides) -> free
    skin_w = np.zeros(len(wpos), bool)
    skin_w[np.unique(wtris[biggest])] = True
    exterior = skin_w & (dist_all < face_dist) & (agree > 0.3) & ~rigid_w
    attachable = ~skin_w & (dist_all < face_dist) & (agree > 0.3) & ~rigid_w
    # only the face pieces and the attachable shells enter the system; hair strands, clothes and
    # other parts near the face stay out (they would add tens of thousands of unknowns)
    near_v = ((dist_all < anchor_dist) & skin_w & ~rigid_w) | attachable
    ring = skin_w & near_v & (dist_all >= anchor_dist - ring_width)
    face_v = near_v & ~ring
    unknown_ids = np.where(near_v)[0]
    local = -np.ones(len(wpos), np.int64)
    local[unknown_ids] = np.arange(len(unknown_ids))
    tri_mask = near_v[wtris].all(1)
    ltris = local[wtris[tri_mask]]
    anchor = ring[unknown_ids]
    if anchor.sum() < 8:                      # degenerate: hold the vertices farthest from the template
        anchor[np.argsort(-dist_all[unknown_ids])[:max(8, len(unknown_ids) // 50)]] = True
    log(f'[transfer] exterior={exterior.sum()} attachable={attachable.sum()} ring(anchored)={anchor.sum()} '
        f'free={(near_v.sum() - exterior.sum() - attachable.sum() - anchor.sum())} triangles={len(ltris)}')

    corr = correspond_triangles(wpos[unknown_ids], ltris, reg.deformed, sub_tris, normal_cos=0.8)
    ext_tri = exterior[unknown_ids][ltris].all(1)
    corr[~ext_tri] = -1                       # only exterior skin triangles carry data
    skin_tri = skin_w[unknown_ids][ltris].all(1)
    identity_tri = np.where(skin_tri, identity_skin, identity_w)   # uncorresponded skin resists, shells are free
    log(f'[transfer] triangles with template correspondence: {(corr >= 0).sum()}/{len(corr)} '
        f'(exterior triangles {ext_tri.sum()})')
    np.savez_compressed(MODELS / f'{model}.rig_debug.npz', unknown_ids=unknown_ids, ltris=ltris, corr=corr,
                        anchor=anchor, face_v=face_v, inverse=inverse, sub_tris=sub_tris,
                        template_ids=reg.template_ids, deformed=reg.deformed, dist_all=dist_all)
    # floating exterior pieces (brows, lashes) ride on the nearest exterior skin vertex
    from scipy.spatial import cKDTree
    ext_unknown = np.where(exterior[unknown_ids])[0]
    float_unknown = np.where(attachable[unknown_ids])[0]
    attach = None
    if len(float_unknown) and len(ext_unknown):
        _, nn = cKDTree(wpos[unknown_ids][ext_unknown]).query(wpos[unknown_ids][float_unknown])
        attach = np.c_[float_unknown, ext_unknown[nn]]
        log(f'[transfer] attaching {len(attach)} floating vertices to the skin')
    solver = build_solver(wpos[unknown_ids], ltris, corr, anchor, anchor_w=1.0, attach=attach,
                          smooth_w_free=smooth_free, smooth_w_corr=smooth_corr, identity_w=identity_tri)

    # ---- template gradients per expression (in the *similarity-aligned* template; gradients are
    # invariant to the uniform scale and rotation cancels inside Ṽ V^-1 when both frames rotate)
    Vinv_t = frame_inverse(sub_rest_tmpl, sub_tris)
    existing = {m.name for m in mesh.morphs}
    new_morphs: list[CanonicalMorph] = []
    stats = {}
    for name, delta in tmpl.expressions.items():
        if skip_existing and name in existing:
            continue
        deformed_tmpl = apply_h(reg.similarity, (tmpl.positions + delta)[reg.template_ids].astype(np.float64))
        S = deformation_gradients(sub_rest_tmpl, deformed_tmpl, sub_tris, Vinv_t)
        new_pos = solver.solve(S, wpos[unknown_ids])
        disp_w = np.zeros((len(wpos), 3))
        d = new_pos - wpos[unknown_ids]
        d[anchor] = 0.0
        disp_w[unknown_ids] = d
        disp = disp_w[inverse]                     # back to the original (duplicated) vertices
        new_morphs.append(CanonicalMorph(name, disp.astype(np.float32),
                                         {'generated': 'ict-deformation-transfer', 'template': template_name}))
        stats[name] = float(np.abs(disp).max() * 1000)
    mesh.morphs.extend(new_morphs)

    out = MODELS / f'{model}.rigged.glb'
    write_glb(mesh, out)
    import resource
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    report = {
        'peak_rss_gb': round(peak_gb, 2),
        'model': model, 'template': template_name, 'seconds': round(time.time() - t0, 1),
        'registration': {'scale': float(np.cbrt(np.linalg.det(reg.similarity[:3, :3]))),
                         'residual_mm': reg.residual * 1000, 'landmark_error_mm': reg.landmark_error * 1000},
        'face_vertices': int(face_v.sum()), 'triangles': int(len(ltris)),
        'corresponded': int((corr >= 0).sum()), 'generated': len(new_morphs),
        'max_disp_mm': {k: round(v, 1) for k, v in stats.items()},
        'output': str(out),
    }
    (MODELS / f'{model}.rig.json').write_text(json.dumps(report, indent=1))
    log(f'[done] {len(new_morphs)} blendshapes -> {out} in {report["seconds"]}s (peak RSS {peak_gb:.1f} GB)')
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model')
    ap.add_argument('--template', default='ict_neutral')
    ap.add_argument('--face-dist', type=float, default=0.012, help='m: target vertices this close to the template move')
    ap.add_argument('--anchor-dist', type=float, default=0.04, help='m: extent of the solved region; its outer 1 cm rim is held at rest')
    ap.add_argument('--identity-skin', type=float, default=0.3, help='identity pull on uncorresponded SKIN triangles (neck, scalp)')
    ap.add_argument('--keep-existing', action='store_true', help='do not skip morphs the model already has')
    ap.add_argument('--subset', default='face', choices=['fitting', 'face', 'face+neck'],
                    help='template vertices used for registration and correspondence')
    ap.add_argument('--smooth-free', type=float, default=0.5, help='smoothness weight where no correspondence')
    ap.add_argument('--smooth-corr', type=float, default=0.05, help='smoothness weight between corresponded triangles')
    ap.add_argument('--identity-w', type=float, default=0.05, help='identity pull on uncorresponded triangles')
    ap.add_argument('--stiff-min', type=float, default=1.0, help='stop the ICP stiffness schedule at this alpha')
    args = ap.parse_args(argv)
    generate(args.model, args.template, args.face_dist, args.anchor_dist, skip_existing=not args.keep_existing,
             subset=args.subset, smooth_free=args.smooth_free, smooth_corr=args.smooth_corr,
             identity_w=args.identity_w, identity_skin=args.identity_skin, stiff_min=args.stiff_min)


if __name__ == '__main__':
    sys.exit(main())
