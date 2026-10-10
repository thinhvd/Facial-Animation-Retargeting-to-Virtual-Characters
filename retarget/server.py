"""Static file server + Socket.IO relay for the new pipeline.

    python -m retarget.server            # http://127.0.0.1:6790/

Replaces NodeServer for the canonical pipeline (NodeServer stays as the
B0 baseline).  Capture processes emit `frame` on namespace `/retarget`;
every other client on that namespace receives it as `frame`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import socketio
from aiohttp import web

WEB_ROOT = Path(__file__).resolve().parent.parent / 'web'
NS = '/retarget'

sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
app = web.Application()
sio.attach(app)


@sio.on('connect', namespace=NS)
async def on_connect(sid, environ):
    print(f'[server] client connected {sid}')


@sio.on('disconnect', namespace=NS)
async def on_disconnect(sid):
    print(f'[server] client disconnected {sid}')


@sio.on('frame', namespace=NS)
async def on_frame(sid, data):
    await sio.emit('frame', data, namespace=NS, skip_sid=sid)


async def index(request):
    raise web.HTTPFound('/viewer.html')


async def api_models(request):
    """List converted models and whether an adapter exists for each."""
    models = []
    for glb in sorted((WEB_ROOT / 'models').glob('*.glb')) + sorted((WEB_ROOT / 'models').glob('*.vrm')):
        adapter = glb.with_suffix('.adapter.json')
        models.append({'name': glb.stem, 'url': f'models/{glb.name}',
                       'adapter': f'models/{adapter.name}' if adapter.exists() else None,
                       'size_mb': round(glb.stat().st_size / 1e6, 1)})
    return web.json_response(models)


_NAME = re.compile(r'^[\w.\-]+$')


def _model_or_404(request) -> str:
    name = request.match_info['model']
    if not _NAME.match(name) or not (WEB_ROOT / 'models' / f'{name}.glb').is_file():
        raise web.HTTPNotFound(text=f'no model {name!r} in web/models')
    return name


def _click_pixels(path: Path) -> dict:
    """{mediapipe index: [x, y]} of the clickable points stored in a landmarks.json."""
    from .rig.landmarks import CLICK_POINTS
    if not path.is_file():
        return {}
    d = json.loads(path.read_text())
    px = d.get('pixels', [])
    return {i: px[i] for _, i, _ in CLICK_POINTS if i < len(px) and px[i][0] >= 0}


async def api_landmarks_get(request):
    """What web/landmarks.html needs: the render, the point list, current and guide positions."""
    from .rig.landmarks import CLICK_POINTS
    name = _model_or_404(request)
    models = WEB_ROOT / 'models'
    lm = models / f'{name}.landmarks.json'
    source = json.loads(lm.read_text()).get('source', 'mediapipe') if lm.is_file() else None
    have_render = (models / f'{name}.front.png').is_file() and (models / f'{name}.camera.json').is_file()
    return web.json_response({
        'model': name,
        'png': f'models/{name}.front.png' if have_render else None,
        'points': [{'key': k, 'index': i, 'hint': h} for k, i, h in CLICK_POINTS],
        'current': _click_pixels(lm), 'source': source,
        'guide': {'png': 'models/ict_neutral.front.png', 'pixels': _click_pixels(models / 'ict_neutral.landmarks.json')},
    })


async def api_landmarks_post(request):
    """Clicked pixels -> rays -> points on the mesh -> <model>.landmarks.json (source: manual)."""
    import numpy as np
    from .rig.landmarks import expand_clicks, landmarks_from_pixels
    name = _model_or_404(request)
    body = await request.json()
    cam = json.loads((WEB_ROOT / 'models' / f'{name}.camera.json').read_text())
    px = np.full((478, 2), np.nan)
    for i, xy in body.get('pixels', {}).items():
        px[int(i)] = xy
    if np.isnan(px).all():
        raise web.HTTPBadRequest(text='no points')
    expand_clicks(px)
    loop = asyncio.get_running_loop()
    r = await loop.run_in_executor(None, lambda: landmarks_from_pixels(name, px, cam, WEB_ROOT / 'models'))
    print(f'[server] {name}: {r["count"]} hand-picked landmarks saved, missed {r["missed"]}')
    return web.json_response({'count': r['count'], 'missed': r['missed']})


async def api_landmarks_render(request):
    """Front render + camera for a model that has none yet (headless Chromium, ~10 s)."""
    name = _model_or_404(request)
    proc = await asyncio.create_subprocess_exec(
        sys.executable, '-m', 'retarget.rig.landmarks', name, '--render-only',
        cwd=str(WEB_ROOT.parent), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    out, _ = await proc.communicate()
    if proc.returncode != 0:
        raise web.HTTPInternalServerError(text=out.decode(errors='replace')[-2000:])
    return web.json_response({'png': f'models/{name}.front.png'})


app.router.add_get('/', index)
app.router.add_get('/api/models', api_models)
app.router.add_get('/api/landmarks/{model}', api_landmarks_get)
app.router.add_post('/api/landmarks/{model}', api_landmarks_post)
app.router.add_post('/api/landmarks/{model}/render', api_landmarks_render)
app.router.add_static('/', WEB_ROOT, show_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=6790)
    args = ap.parse_args(argv)
    print(f'[server] serving {WEB_ROOT} on http://{args.host}:{args.port}/')
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == '__main__':
    main()
