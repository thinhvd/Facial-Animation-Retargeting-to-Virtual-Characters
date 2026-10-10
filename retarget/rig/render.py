"""Front-view render of a model through web/viewer.html in headless Chromium.

Returns the PNG and the camera the viewer used, so pixels can be
unprojected back onto the mesh (see landmarks.py).  Requires the server
(python -m retarget.server) and playwright + chromium.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def render_front(model: str, out_png: Path, size: int = 1024, server: str = 'http://127.0.0.1:6790',
                 lit: bool = True, background: str = 'gray', timeout: float = 90.0) -> dict:
    from playwright.sync_api import sync_playwright
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader',
                                          '--ignore-gpu-blocklist', '--enable-webgl'])
        page = browser.new_page(viewport={'width': size, 'height': size}, device_scale_factor=1)
        page.goto(f'{server}/viewer.html?model={model}&panel=0&stats=0&lit={1 if lit else 0}')
        t0 = time.time()
        while time.time() - t0 < timeout:
            status = page.inner_text('#status')
            if status.startswith('loaded') or 'failed' in status:
                break
            time.sleep(0.3)
        if not status.startswith('loaded'):
            browser.close()
            raise RuntimeError(f'viewer did not load {model}: {status}')
        page.evaluate('''(bg) => {
            const v = window.viewer;
            v.controls.enabled = false;
            if (bg === 'gray') v.renderer.setClearColor(0x808080, 1), (v.model.root.parent.background = new v.THREE.Color(0x808080));
        }''', background)
        page.wait_for_timeout(600)
        cam = page.evaluate('() => window.viewer.cameraInfo()')
        page.screenshot(path=str(out_png))
        browser.close()
    cam['png'] = str(out_png)
    cam['model'] = model
    return cam


def pixel_rays(cam: dict, px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixels (N, 2) in image coordinates -> (origins (N,3), directions (N,3)) in world space.

    Matches three.js PerspectiveCamera (vertical fov in degrees, lookAt target, up vector).
    """
    P = np.asarray(cam['position'], np.float64)
    T = np.asarray(cam['target'], np.float64)
    up = np.asarray(cam['up'], np.float64)
    f = T - P
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    w, h = float(cam['width']), float(cam['height'])
    aspect = w / h
    tan_half = np.tan(np.radians(cam['fov']) / 2)
    nx = (px[:, 0] / w) * 2 - 1
    ny = 1 - (px[:, 1] / h) * 2
    d = f[None, :] + (nx * tan_half * aspect)[:, None] * r[None, :] + (ny * tan_half)[:, None] * u[None, :]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.repeat(P[None, :], len(px), 0), d


def save_camera(cam: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(cam, indent=1))
