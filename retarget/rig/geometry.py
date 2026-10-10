"""Closest point on a triangle mesh with bounded memory.

trimesh's `closest_point` gathers every face within the nearest-vertex
distance as a candidate; for query points far from a small surface that is
all faces, i.e. points x faces arrays of tens of millions of rows (this
OOM-killed the machine on an MMD face piece).  Here a k-d tree over face
centroids gives k candidates per point and the exact closest point is
evaluated only on those (Ericson, Real-Time Collision Detection, 5.1.5).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def closest_point_on_triangles(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Closest point to p on triangle (a, b, c), all (N, 3)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2

    out = np.empty_like(p)
    done = np.zeros(len(p), bool)

    def take(mask, value):
        m = mask & ~done
        out[m] = value[m]
        done[m] = True

    with np.errstate(divide='ignore', invalid='ignore'):
        take((d1 <= 0) & (d2 <= 0), a)                                      # vertex a
        take((d3 >= 0) & (d4 <= d3), b)                                     # vertex b
        v = d1 / (d1 - d3)
        take((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + v[:, None] * ab)         # edge ab
        take((d6 >= 0) & (d5 <= d6), c)                                     # vertex c
        w = d2 / (d2 - d6)
        take((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + w[:, None] * ac)         # edge ac
        w2 = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        take((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), b + w2[:, None] * (c - b))  # edge bc
        denom = 1.0 / (va + vb + vc)
        v2, w3 = vb * denom, vc * denom
        take(~done, a + ab * v2[:, None] + ac * w3[:, None])                 # inside
    return out


class SurfaceQuery:
    """Closest point / distance / face id on a triangle mesh (k candidates per query)."""

    def __init__(self, positions: np.ndarray, triangles: np.ndarray, k: int = 24):
        self.pos = np.asarray(positions, np.float64)
        self.tris = np.asarray(triangles, np.int64)
        self.tri_pts = self.pos[self.tris]                      # (F, 3, 3)
        self.centroids = self.tri_pts.mean(1)
        self.tree = cKDTree(self.centroids)
        self.k = max(1, min(k, len(self.tris)))
        n = np.cross(self.tri_pts[:, 1] - self.tri_pts[:, 0], self.tri_pts[:, 2] - self.tri_pts[:, 0])
        self.face_normals = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)

    def closest(self, points: np.ndarray, chunk: int = 20000):
        """Return (closest (N,3), distance (N,), face id (N,))."""
        points = np.asarray(points, np.float64)
        N = len(points)
        out_p = np.empty((N, 3))
        out_d = np.empty(N)
        out_t = np.empty(N, np.int64)
        for s in range(0, N, chunk):
            q = points[s:s + chunk]
            _, cand = self.tree.query(q, k=self.k)
            cand = np.atleast_2d(cand)
            if cand.shape[0] != len(q):
                cand = cand.T
            n, K = cand.shape
            flat = cand.reshape(-1)
            T = self.tri_pts[flat]
            cp = closest_point_on_triangles(np.repeat(q, K, axis=0), T[:, 0], T[:, 1], T[:, 2]).reshape(n, K, 3)
            d = np.linalg.norm(cp - q[:, None, :], axis=2)
            j = d.argmin(1)
            idx = np.arange(n)
            out_p[s:s + n] = cp[idx, j]
            out_d[s:s + n] = d[idx, j]
            out_t[s:s + n] = cand[idx, j]
        return out_p, out_d, out_t
