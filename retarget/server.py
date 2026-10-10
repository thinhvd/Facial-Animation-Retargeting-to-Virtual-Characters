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
import time
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
    if _running(name):
        raise web.HTTPConflict(text=f'{name} is being generated; save the points when it has finished')
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
    (WEB_ROOT / 'models' / f'{name}.landmarks_failed.json').unlink(missing_ok=True)
    print(f'[server] {name}: {r["count"]} hand-picked landmarks saved, missed {r["missed"]}')
    return web.json_response({'count': r['count'], 'missed': r['missed']})


async def api_landmarks_render(request):
    """Front render + camera for a model that has none yet (headless Chromium, ~10 s)."""
    name = _model_or_404(request)
    if JOB['status'] == 'running':
        raise web.HTTPConflict(text=f'busy generating {JOB["model"]}; render when it has finished')
    proc = await asyncio.create_subprocess_exec(
        sys.executable, '-m', 'retarget.rig.landmarks', name, '--render-only',
        cwd=str(WEB_ROOT.parent), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    out, _ = await proc.communicate()
    if proc.returncode != 0:
        raise web.HTTPInternalServerError(text=out.decode(errors='replace')[-2000:])
    return web.json_response({'png': f'models/{name}.front.png'})


# ---- blendshape generation as a background job (one at a time: generate and the headless
# browser used for automatic landmarks each take 1-2 GB)
JOB: dict = {'model': None, 'status': 'idle', 'log': [], 'started': 0.0, 'seconds': 0.0}
_NOISE = re.compile(r'^(W0000|I0000|WARNING: All log messages|INFO: Created TensorFlow|E0000)')


async def _run_rig(name: str) -> None:
    JOB.update(model=name, status='running', log=[], started=time.time(), seconds=0.0)
    proc = await asyncio.create_subprocess_exec(
        sys.executable, '-u', '-m', 'retarget.importer', '--existing', name,
        cwd=str(WEB_ROOT.parent), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    async for raw in proc.stdout:
        line = raw.decode(errors='replace').rstrip()
        if line and not _NOISE.match(line):
            JOB['log'] = (JOB['log'] + [line])[-200:]
    rc = await proc.wait()
    JOB['seconds'] = round(time.time() - JOB['started'], 1)
    JOB['status'] = {0: 'done', 3: 'needs_landmarks'}.get(rc, 'failed')
    print(f'[server] rig {name}: {JOB["status"]} in {JOB["seconds"]} s')


def _running(name: str | None = None) -> bool:
    return JOB['status'] == 'running' and (name is None or JOB['model'] == name)


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.is_file() else 0.0


def rig_state(name: str) -> dict:
    """Where a model stands in import -> (landmarks) -> generate, and the one next step.

    state:  running          a job for this model is running
            complete         its own morphs and bones cover every input but tongueOut: nothing to generate
            up_to_date       <name>.rigged.glb is newer than the model and its landmarks
            ready            landmarks exist and the result is missing or older: Generate
            needs_landmarks  MediaPipe found no face and nobody has placed points: Pick landmarks
            ready_auto       never tried: Generate (MediaPipe first)
    Both pages derive their buttons from this; POST /api/rig refuses what the state does not allow.
    """
    m = WEB_ROOT / 'models'
    glb, rigged, lm = m / f'{name}.glb', m / f'{name}.rigged.glb', m / f'{name}.landmarks.json'
    read = lambda p: json.loads(p.read_text()) if p.is_file() else None
    cov = (read(m / f'{name}.adapter.json') or {}).get('coverageSummary')
    rcov = (read(m / f'{name}.rigged.adapter.json') or {}).get('coverageSummary')
    lmd = read(lm)
    landmarks = None
    if lmd:
        from .rig.landmarks import CLICK_POINTS
        manual = lmd.get('source') == 'manual'
        landmarks = {'source': lmd.get('source', 'mediapipe'),
                     'count': sum(lmd['valid'][i] for _, i, _ in CLICK_POINTS) if manual else lmd.get('count')}
    up_to_date = rigged.is_file() and _mtime(rigged) >= max(_mtime(glb), _mtime(lm))
    if _running(name):
        state = 'running'
    elif up_to_date:
        state = 'up_to_date'
    elif cov and cov['covered'] >= cov['total'] - 1 and not rigged.is_file():
        state = 'complete'
    elif landmarks:
        state = 'ready'
    elif (m / f'{name}.landmarks_failed.json').is_file():
        state = 'needs_landmarks'
    else:
        state = 'ready_auto'
    mine = JOB['model'] == name
    return {
        'model': name, 'state': state,
        'busy_with': JOB['model'] if _running() and not mine else None,
        'coverage': cov and [cov['covered'], cov['total']],
        'rigged': f'{name}.rigged' if rigged.is_file() else None,
        'rigged_coverage': rcov and [rcov['covered'], rcov['total']],
        'rig_report': read(m / f'{name}.rig.json'),
        'landmarks': landmarks, 'render': (m / f'{name}.front.png').is_file(),
        # the last job for this model (kept after it ends, until the server restarts)
        'job': {'status': JOB['status'], 'log': JOB['log'][-12:], 'seconds': JOB['seconds'],
                'elapsed': round(time.time() - JOB['started'], 1) if _running(name) else 0.0} if mine else None,
    }


async def api_rig_get(request):
    return web.json_response(rig_state(_model_or_404(request)))


async def api_rig_post(request):
    """Start generation, but only when the state allows it (?force=1 regenerates anyway)."""
    name = _model_or_404(request)
    if name.endswith('.rigged'):
        raise web.HTTPBadRequest(text='pick the source model, not its .rigged result')
    if _running():
        raise web.HTTPConflict(text=f'busy: generating {JOB["model"]}')
    st = rig_state(name)
    if 'force' not in request.query:
        refuse = {'up_to_date': 'already generated from these points; change the landmarks first',
                  'complete': 'the model covers every input with its own morphs; nothing to generate',
                  'needs_landmarks': 'MediaPipe found no face last time; pick the landmarks first'}
        if st['state'] in refuse:
            raise web.HTTPConflict(text=refuse[st['state']])
    asyncio.get_running_loop().create_task(_run_rig(name))
    await asyncio.sleep(0)
    return web.json_response(rig_state(name))


app.router.add_get('/', index)
app.router.add_get('/api/models', api_models)
app.router.add_get('/api/landmarks/{model}', api_landmarks_get)
app.router.add_post('/api/landmarks/{model}', api_landmarks_post)
app.router.add_post('/api/landmarks/{model}/render', api_landmarks_render)
app.router.add_get('/api/rig/{model}', api_rig_get)
app.router.add_post('/api/rig/{model}', api_rig_post)
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
