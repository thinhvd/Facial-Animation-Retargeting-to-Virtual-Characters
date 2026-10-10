"""Headless screenshots of web/viewer.html for a model, neutral and with a
given set of canonical inputs.  Used to check conversions and adapters
without a browser session.

    python tools/screenshot_viewer.py HeadA --out /tmp/shots \
        --set jawOpen=1 eyeBlinkLeft=1 browInnerUp=1 --head 0,30,0

Requires the server (python -m retarget.server) and playwright + chromium.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model')
    ap.add_argument('--url', default='http://127.0.0.1:6790/viewer.html')
    ap.add_argument('--out', type=Path, default=Path('/tmp/viewer_shots'))
    ap.add_argument('--set', nargs='*', default=[], help='name=value canonical inputs (ARKit names)')
    ap.add_argument('--head', default='0,0,0', help='rx,ry,rz degrees')
    ap.add_argument('--gaze', default='0,0', help='pitch,yaw degrees')
    ap.add_argument('--size', default='900x900')
    ap.add_argument('--timeout', type=float, default=60.0)
    ap.add_argument('--tag', default=None, help='suffix for the posed screenshot file name')
    ap.add_argument('--live', action='store_true',
                    help='do not drive from sliders; wait and shoot whatever the socket delivers')
    ap.add_argument('--lit', action='store_true', help='replace unlit materials with shaded ones')
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    w, h = (int(v) for v in args.size.split('x'))

    with sync_playwright() as p:
        browser = p.chromium.launch(args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader',
                                          '--ignore-gpu-blocklist', '--enable-webgl'])
        page = browser.new_page(viewport={'width': w, 'height': h})
        console: list[str] = []
        page.on('console', lambda m: console.append(f'[{m.type}] {m.text}'))
        page.on('pageerror', lambda e: console.append(f'[pageerror] {e}'))
        page.goto(f'{args.url}?model={args.model}&panel=0&stats=0&lit={1 if args.lit else 0}')

        t0 = time.time()
        status = ''
        while time.time() - t0 < args.timeout:
            status = page.inner_text('#status')
            if status.startswith('loaded') or 'failed' in status:
                break
            time.sleep(0.5)
        print('status:', status.replace('\n', ' | '))
        if not status.startswith('loaded'):
            print('\n'.join(console[-20:]))
            browser.close()
            return 1

        page.wait_for_timeout(800)
        page.screenshot(path=str(args.out / f'{args.model}_neutral.png'))

        if args.live:
            page.evaluate("() => { document.getElementById('smooth').value = 0.2; }")
            page.wait_for_timeout(2500)
            page.screenshot(path=str(args.out / f'{args.model}_{args.tag or "live"}.png'))
            print('status:', page.inner_text('#status').replace('\n', ' | '))
            browser.close()
            return 0

        # drive from sliders
        values = {}
        for kv in args.set:
            k, v = kv.split('=')
            values[k] = float(v)
        head = [float(v) for v in args.head.split(',')]
        gaze = [float(v) for v in args.gaze.split(',')]
        page.evaluate('''([values, head, gaze]) => {
            const set = (id, v) => { const el = document.getElementById(id); if (el) { el.value = v; el.dispatchEvent(new Event('input')); } };
            for (const [k, v] of Object.entries(values)) set('bs_' + k, v);
            set('p_rx', head[0]); set('p_ry', head[1]); set('p_rz', head[2]);
            set('p_gp', gaze[0]); set('p_gy', gaze[1]);
            document.getElementById('smooth').value = 0;
            document.getElementById('manual').checked = true;
            document.getElementById('manual').dispatchEvent(new Event('change'));
        }''', [values, head, gaze])
        page.wait_for_timeout(1200)
        tag = args.tag or ('_'.join(f'{k}{v:g}' for k, v in values.items()) or 'pose')
        page.screenshot(path=str(args.out / f'{args.model}_{tag}.png'))

        # a little introspection for the log
        info = page.evaluate('''() => {
            const s = document.getElementById('status').innerText;
            return s;
        }''')
        errors = [c for c in console if c.startswith('[error]') or c.startswith('[pageerror]')]
        print(f'wrote {args.out}/{args.model}_neutral.png and {args.model}_{tag}.png')
        if errors:
            print('console errors:')
            print('\n'.join(errors[:10]))
        browser.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
