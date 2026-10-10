"""Register the ICT template onto a target mesh.

1. similarity (Umeyama) alignment on the MediaPipe landmark pairs
2. non-rigid ICP with per-vertex affine transforms (Amberg et al. 2007,
   "Optimal step nonrigid ICP"): stiffness term on template edges,
   landmark term, closest-point term with normal / distance gating.

Everything is sparse least squares; the template subset is ICT's
`fitting_verts` (7.8k vertices) so one solve is well under a second.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh


@dataclass
class Registration:
    template_ids: np.ndarray        # vertex ids of the template subset that was deformed
    deformed: np.ndarray            # (n, 3) their positions after registration (target space)
    affine: np.ndarray              # (n, 4, 3) per-vertex affine X_i (row-vector convention)
    similarity: np.ndarray          # 4x4 initial similarity transform template -> target
    residual: float                 # mean closest-point distance at the end (metres)
    landmark_error: float           # mean landmark distance at the end (metres)


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> np.ndarray:
    """Similarity transform (4x4) minimising ||s R src + t - dst||."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = d0.T @ s0 / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    scale = (D * np.diag(S)).sum() / (s0 ** 2).sum() * len(src) if with_scale else 1.0
    t = mu_d - scale * R @ mu_s
    M = np.eye(4)
    M[:3, :3] = scale * R
    M[:3, 3] = t
    return M


def apply_h(M: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ M[:3, :3].T + M[:3, 3]


def robust_similarity(src: np.ndarray, dst: np.ndarray, iters: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Umeyama with outlier rejection (2.5 sigma); returns (M, inlier mask)."""
    keep = np.ones(len(src), bool)
    M = umeyama(src, dst)
    for _ in range(iters):
        M = umeyama(src[keep], dst[keep])
        err = np.linalg.norm(apply_h(M, src) - dst, axis=1)
        thr = err[keep].mean() + 2.5 * err[keep].std()
        keep = err < max(thr, 1e-4)
    return M, keep


def vertex_normals(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    n = np.zeros_like(positions, dtype=np.float64)
    a, b, c = (positions[triangles[:, i]] for i in range(3))
    fn = np.cross(b - a, c - a)
    for i in range(3):
        np.add.at(n, triangles[:, i], fn)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(length > 0, length, 1)


def nonrigid_icp(template_pos: np.ndarray, template_tris: np.ndarray, target: trimesh.Trimesh,
                 lm_template_ids: np.ndarray | None, lm_target_pts: np.ndarray | None,
                 stiffness: tuple[float, ...] = (50.0, 20.0, 10.0, 5.0, 2.0, 1.0),
                 landmark_weight: float = 5.0, gamma: float = 1.0, max_inner: int = 8, tol: float = 1e-4,
                 max_dist: float = 0.03, normal_cos: float = 0.5, log=print) -> tuple[np.ndarray, np.ndarray, float]:
    """Amberg et al. optimal-step non-rigid ICP.

    template_pos: (n, 3) already roughly aligned to the target
    template_tris: (m, 3) local triangle ids over the template subset
    Returns (deformed positions (n, 3), affine X (n, 4, 3), final mean residual).
    """
    n = len(template_pos)
    # ---- node-arc incidence matrix over unique edges
    e = np.concatenate([template_tris[:, [0, 1]], template_tris[:, [1, 2]], template_tris[:, [2, 0]]])
    e = np.sort(e, axis=1)
    e = np.unique(e, axis=0)
    ne = len(e)
    M = sp.coo_matrix((np.r_[-np.ones(ne), np.ones(ne)], (np.r_[np.arange(ne), np.arange(ne)], np.r_[e[:, 0], e[:, 1]])),
                      shape=(ne, n)).tocsr()
    G = sp.diags([1.0, 1.0, 1.0, gamma])
    MG = sp.kron(M, G).tocsr()                       # (4ne, 4n)

    hom = np.c_[template_pos, np.ones(n)]            # (n, 4)
    def D_of(hpos):
        rows = np.repeat(np.arange(n), 4)
        cols = (np.arange(n)[:, None] * 4 + np.arange(4)[None, :]).ravel()
        return sp.csr_matrix((hpos.ravel(), (rows, cols)), shape=(n, 4 * n))
    D = D_of(hom)

    X = np.tile(np.r_[np.eye(3), np.zeros((1, 3))], (n, 1))   # (4n, 3): identity affines
    prox = trimesh.proximity.ProximityQuery(target)
    tnormals = target.face_normals
    tmpl_normals = vertex_normals(template_pos, template_tris)

    lm_rows = None
    if lm_template_ids is not None and len(lm_template_ids):
        lm_rows = D[lm_template_ids]                 # (L, 4n)

    residual = np.inf
    for alpha in stiffness:
        for it in range(max_inner):
            cur = D @ X                                  # (n, 3) current deformed positions
            closest, dist, tri_id = prox.on_surface(cur)
            cur_normals = vertex_normals(cur, template_tris)
            w = ((dist < max_dist) & ((cur_normals * tnormals[tri_id]).sum(1) > normal_cos)).astype(np.float64)
            W = sp.diags(w)
            blocks = [alpha * MG, W @ D]
            rhs = [np.zeros((4 * ne, 3)), W @ closest]
            if lm_rows is not None:
                blocks.append(landmark_weight * lm_rows)
                rhs.append(landmark_weight * lm_target_pts)
            A = sp.vstack(blocks).tocsr()
            B = np.vstack(rhs)
            AtA = (A.T @ A).tocsc()
            AtB = A.T @ B
            X_new = spla.spsolve(AtA, AtB)
            X_new = np.asarray(X_new).reshape(4 * n, 3)
            step = np.abs(X_new - X).max()
            X = X_new
            residual = float((dist * w).sum() / max(w.sum(), 1))
            if step < tol:
                break
        log(f'    alpha={alpha:5.1f} iters={it + 1} residual={residual * 1000:.2f} mm used={int(w.sum())}/{n}')
    deformed = D @ X
    return deformed, X.reshape(n, 4, 3), residual


def register(template_pos_all: np.ndarray, template_tris_all: np.ndarray, template_subset: np.ndarray,
             tmpl_landmarks: np.ndarray, tmpl_lm_valid: np.ndarray,
             target: trimesh.Trimesh, tgt_landmarks: np.ndarray, tgt_lm_valid: np.ndarray,
             tmpl_lm_vertex: np.ndarray, log=print, **icp_kw) -> Registration:
    """Full pipeline: similarity on landmarks, then non-rigid ICP on the subset."""
    valid = tmpl_lm_valid & tgt_lm_valid
    M, inl = robust_similarity(tmpl_landmarks[valid], tgt_landmarks[valid])
    scale = np.cbrt(np.linalg.det(M[:3, :3]))
    err0 = np.linalg.norm(apply_h(M, tmpl_landmarks[valid]) - tgt_landmarks[valid], axis=1)
    log(f'  similarity: scale={scale:.3f} landmark rms={np.sqrt((err0 ** 2).mean()) * 1000:.1f} mm '
        f'inliers={inl.sum()}/{valid.sum()}')

    # subset + local triangles
    keep = np.zeros(len(template_pos_all), bool)
    keep[template_subset] = True
    remap = -np.ones(len(template_pos_all), np.int64)
    remap[template_subset] = np.arange(len(template_subset))
    tris = remap[template_tris_all[keep[template_tris_all].all(1)]]
    pos = apply_h(M, template_pos_all[template_subset].astype(np.float64))

    # landmark constraints: template landmark = nearest template vertex (within subset)
    lm_ids_local = remap[tmpl_lm_vertex]
    ok = valid & (lm_ids_local >= 0)
    deformed, X, residual = nonrigid_icp(pos, tris, target, lm_ids_local[ok], tgt_landmarks[ok], log=log, **icp_kw)
    lm_err = float(np.linalg.norm(deformed[lm_ids_local[ok]] - tgt_landmarks[ok], axis=1).mean())
    log(f'  non-rigid ICP: residual={residual * 1000:.2f} mm, landmark error={lm_err * 1000:.2f} mm')
    return Registration(template_subset, deformed, X, M, residual, lm_err)
