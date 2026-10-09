"""Static file server + Socket.IO relay for the new pipeline.

    python -m retarget.server            # http://127.0.0.1:6790/

Replaces NodeServer for the canonical pipeline (NodeServer stays as the
B0 baseline).  Capture processes emit `frame` on namespace `/retarget`;
every other client on that namespace receives it as `frame`.
"""
from __future__ import annotations

import argparse
import json
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


app.router.add_get('/', index)
app.router.add_get('/api/models', api_models)
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
