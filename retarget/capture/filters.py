"""Per-channel temporal filters for capture frames.

OneEuroFilter (Casiez, Roussel & Vogel, CHI 2012) is a first-order low-pass
whose cutoff rises with the signal's speed:

    cutoff = min_cutoff + beta * |dx/dt|        (dx/dt itself low-passed at d_cutoff)

so a still face is smoothed hard (min_cutoff) while blinks and head turns pass
with little lag.  Both filters take the sample time, so they behave the same
at 15, 30 or 60 fps and when frames are dropped.  EMAFilter is the per-sample
exponential average the viewer used before; kept for comparison.

Parameters are per channel (numpy broadcasting), because the channels have
different units: blendshape scores are in [0, 1], head angles in degrees.
"""
from __future__ import annotations

import math

import numpy as np

# (min_cutoff Hz, beta) per channel group; starting points tuned on synthetic noise
# (0.02 score, 0.4 deg) and a synthetic clip -- check with --dump on a real webcam
EURO_BS = (1.5, 5.0)        # scores in [0, 1]; velocities of a blink reach ~10 /s
EURO_HEAD = (0.7, 0.2)      # degrees; a quick head turn is ~100 deg/s
D_CUTOFF = 1.0              # Hz, for the speed estimate


def _alpha(cutoff, dt: float):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff, beta, d_cutoff: float = D_CUTOFF):
        self.min_cutoff = np.asarray(min_cutoff, dtype=float)
        self.beta = np.asarray(beta, dtype=float)
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self.x = self.dx = self.t = None

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return self.x.copy()
        dt = t - self.t
        if dt <= 0:                      # duplicate timestamp: nothing new to integrate
            return self.x.copy()
        self.dx += _alpha(self.d_cutoff, dt) * ((x - self.x) / dt - self.dx)
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        self.x += _alpha(cutoff, dt) * (x - self.x)
        self.t = t
        return self.x.copy()


class EMAFilter:
    """y += alpha * (x - y) per sample, alpha = 1 - smoothing."""

    def __init__(self, smoothing: float = 0.5):
        self.alpha = 1.0 - smoothing
        self.reset()

    def reset(self) -> None:
        self.x = None

    def __call__(self, x, t: float = 0.0) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        self.x = x.copy() if self.x is None else self.x + self.alpha * (x - self.x)
        return self.x.copy()


class FrameFilter:
    """Filters the `bs` scores and `head` angles of a canonical frame.

    Gaze is derived from the (filtered) eyeLook* scores by the capture, so it
    needs no filter of its own.
    """

    def __init__(self, names, kind: str = 'euro', bs=EURO_BS, head=EURO_HEAD, smoothing: float = 0.5):
        self.names = tuple(names)
        n = len(self.names)
        if kind == 'euro':
            self.f = OneEuroFilter(np.r_[np.full(n, bs[0]), np.full(3, head[0])],
                                   np.r_[np.full(n, bs[1]), np.full(3, head[1])])
        elif kind == 'ema':
            self.f = EMAFilter(smoothing)
        elif kind == 'none':
            self.f = None
        else:
            raise ValueError(f'unknown filter {kind!r}')
        self.kind = kind

    def reset(self) -> None:
        if self.f is not None:
            self.f.reset()

    def __call__(self, bs: dict, head, t: float) -> tuple[dict, list]:
        if self.f is None:
            return bs, list(head)
        n = len(self.names)
        y = self.f(np.r_[[bs[k] for k in self.names], head], t)
        return {k: float(v) for k, v in zip(self.names, y[:n])}, y[n:].tolist()
