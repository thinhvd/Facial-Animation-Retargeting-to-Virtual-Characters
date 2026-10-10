"""Deformation transfer (Sumner & Popović 2004) from the registered template
to the target mesh.

For every target triangle with a template correspondence we ask its
deformation gradient to equal the template triangle's gradient for the
expression; triangles without a correspondence keep identity; vertices
far from the face are anchored.  One sparse factorisation serves all
expressions (same matrix, different right-hand sides).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh


def fourth_vertex(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> np.ndarray:
    n = np.cross(v2 - v1, v3 - v1)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return v1 + n / np.sqrt(np.where(norm > 0, norm, 1))


def frame_inverse(pos: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """(F, 3, 3) inverse of [v2-v1, v3-v1, v4-v1] (columns) per triangle.
    Degenerate (zero-area) triangles get a zero inverse, i.e. no constraint."""
    v1, v2, v3 = (pos[tris[:, i]] for i in range(3))
    v4 = fourth_vertex(v1, v2, v3)
    V = np.stack([v2 - v1, v3 - v1, v4 - v1], axis=2)      # (F, 3, 3) columns
    area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)
    ok = area > 1e-3 * np.median(area)                      # slivers: no constraint
    out = np.zeros_like(V)
    out[ok] = np.linalg.inv(V[ok])
    return out


def deformation_gradients(rest: np.ndarray, deformed: np.ndarray, tris: np.ndarray, Vinv: np.ndarray) -> np.ndarray:
    """(F, 3, 3) S = Ṽ V^-1 per triangle."""
    v1, v2, v3 = (deformed[tris[:, i]] for i in range(3))
    v4 = fourth_vertex(v1, v2, v3)
    Vt = np.stack([v2 - v1, v3 - v1, v4 - v1], axis=2)
    return Vt @ Vinv


@dataclass
class TransferSolver:
    tris: np.ndarray             # (F, 3) target triangles used (local vertex ids)
    verts: np.ndarray            # (n_v,) target vertex ids (global) that are unknowns
    n_v: int
    n_f: int
    corr: np.ndarray             # (F,) template triangle id or -1
    lu: object                   # factorisation of AtA
    A: sp.csr_matrix
    Vinv: np.ndarray             # (F, 3, 3) target rest frame inverses
    anchor_w: float

    def solve(self, S_template: np.ndarray, rest_local: np.ndarray) -> np.ndarray:
        """S_template: (F_template, 3, 3) gradients for one expression -> (n_v, 3) new positions."""
        F = self.n_f
        target_S = np.tile(np.eye(3), (F, 1, 1))
        has = self.corr >= 0
        target_S[has] = S_template[self.corr[has]]
        # rows of uncorresponded triangles carry identity_w: scale their rhs the same way
        target_S[~has] *= np.where(self.corr >= 0, 1.0, self.identity_w)[~has][:, None, None]
        # rows: for each triangle j and column m of S: 3 rows; rhs per coordinate k = S[j, k, m]
        n_v = self.n_v
        off = 3 * F + self.n_smooth
        base = off + n_v + n_v + F
        rhs = np.zeros((base + len(self.attach), 3))
        rhs[:3 * F] = target_S.transpose(0, 2, 1).reshape(3 * F, 3)     # row (3j+m) -> S[j, :, m]
        # smoothness rows: rhs 0
        rhs[off:off + n_v] = self.anchor_w * rest_local[:, :3]           # only anchored rows have A != 0
        rhs[off + n_v:base] = self.reg_w * self.rest_all
        rhs[base:] = self.attach_rhs
        AtB = self.A.T @ rhs
        x = self.lu.solve(AtB)
        return np.asarray(x)[:self.n_v]


def build_solver(target_pos: np.ndarray, target_tris: np.ndarray, corr: np.ndarray,
                 anchor_mask: np.ndarray, anchor_w: float = 1.0, identity_w: float = 0.05,
                 reg: float = 1e-3, attach: np.ndarray | None = None, attach_w: float = 0.5,
                 smooth_w_free: float = 0.5, smooth_w_corr: float = 0.05) -> TransferSolver:
    """target_pos (n_v, 3) local vertices; target_tris (F, 3) local; corr (F,) template tri id or -1;
    anchor_mask (n_v,) vertices softly held at rest; reg: Tikhonov pull of every unknown
    (vertices and 4th vertices) toward its rest position, scaled to the gradient rows;
    attach (k, 2) pairs (i, j): vertex i (e.g. on a floating eyebrow shell) keeps its rest
    offset to vertex j (on the skin), so disconnected pieces ride along with the face."""
    n_v, F = len(target_pos), len(target_tris)
    target_pos = target_pos.astype(np.float64)
    Vinv = frame_inverse(target_pos, target_tris)
    # vertices no triangle refers to (or only sliver ones) would be free: hold them at rest
    referenced = np.zeros(n_v, bool)
    referenced[np.unique(target_tris[(Vinv != 0).any((1, 2))])] = True
    anchor_mask = anchor_mask | ~referenced
    v1, v2, v3 = (target_pos[target_tris[:, i]] for i in range(3))
    rest4 = fourth_vertex(v1, v2, v3)
    rest_all = np.vstack([target_pos, rest4])                     # (n_v + F, 3)
    scale = np.median(np.abs(Vinv[Vinv != 0])) if (Vinv != 0).any() else 1.0
    rows, cols, vals = [], [], []
    w_tri = np.where(corr >= 0, 1.0, identity_w)

    def gradient_rows(r, tri_ids, m, weight):
        """Append rows r: weight * T_{tri}[., m] expressed in the unknowns."""
        c1 = -Vinv[tri_ids, :, m].sum(1) * weight
        for l, col_ids in enumerate([target_tris[tri_ids, 1], target_tris[tri_ids, 2], n_v + tri_ids]):
            rows.append(r); cols.append(col_ids); vals.append(Vinv[tri_ids, l, m] * weight)
        rows.append(r); cols.append(target_tris[tri_ids, 0]); vals.append(c1)

    all_tri = np.arange(F)
    for m in range(3):
        gradient_rows(3 * all_tri + m, all_tri, m, w_tri)

    # smoothness: adjacent triangles should deform alike (strong where no correspondence)
    adj = trimesh.Trimesh(target_pos, target_tris, process=False).face_adjacency
    n_smooth = 0
    if len(adj):
        ws = np.where((corr[adj[:, 0]] >= 0) & (corr[adj[:, 1]] >= 0), smooth_w_corr, smooth_w_free)
        base_s = 3 * F
        for m in range(3):
            r = base_s + 3 * np.arange(len(adj)) + m
            gradient_rows(r, adj[:, 0], m, ws)
            gradient_rows(r, adj[:, 1], m, -ws)
        n_smooth = 3 * len(adj)
    # anchors: identity rows on anchored vertices (weight anchor_w * scale, so they compete
    # with gradient rows), then a weak Tikhonov row for every unknown toward rest
    off = 3 * F + n_smooth
    aw = np.where(anchor_mask, anchor_w * scale, 0.0)
    rows.append(off + np.arange(n_v)); cols.append(np.arange(n_v)); vals.append(aw)
    rw = np.full(n_v + F, reg * scale)
    rows.append(off + n_v + np.arange(n_v + F)); cols.append(np.arange(n_v + F)); vals.append(rw)
    n_rows = off + n_v + n_v + F
    attach = np.zeros((0, 2), np.int64) if attach is None else np.asarray(attach, np.int64)
    if len(attach):
        ar = n_rows + np.arange(len(attach))
        rows += [ar, ar]; cols += [attach[:, 0], attach[:, 1]]
        vals += [np.full(len(attach), attach_w * scale), np.full(len(attach), -attach_w * scale)]
        n_rows += len(attach)
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n_rows, n_v + F))
    AtA = (A.T @ A).tocsc()
    lu = spla.splu(AtA)
    solver = TransferSolver(target_tris, np.arange(n_v), n_v, F, corr, lu, A, Vinv, anchor_w * scale)
    solver.rest_all = rest_all
    solver.reg_w = reg * scale
    solver.n_smooth = n_smooth
    solver.identity_w = identity_w
    solver.attach = attach
    solver.attach_rhs = attach_w * scale * (target_pos[attach[:, 0]] - target_pos[attach[:, 1]]) if len(attach) else np.zeros((0, 3))
    return solver


def correspond_triangles(target_pos: np.ndarray, target_tris: np.ndarray,
                         tmpl_pos: np.ndarray, tmpl_tris: np.ndarray,
                         max_dist: float = 0.012, normal_cos: float = 0.6) -> np.ndarray:
    """For each target triangle, the template triangle whose surface is closest to
    its centroid (distance + normal gated), else -1."""
    tmpl = trimesh.Trimesh(tmpl_pos, tmpl_tris, process=False)
    cent = target_pos[target_tris].mean(1)
    tn = trimesh.Trimesh(target_pos, target_tris, process=False).face_normals
    closest, dist, tri_id = trimesh.proximity.ProximityQuery(tmpl).on_surface(cent)
    ok = (dist < max_dist) & ((tn * tmpl.face_normals[tri_id]).sum(1) > normal_cos)
    return np.where(ok, tri_id, -1)
