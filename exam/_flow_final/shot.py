#!/usr/bin/env python3
"""Headless-Chrome screenshot of a local html file, trimmed at the bottom, plus console errors.

  python3 -I shot.py page.html out.png [--width 1280] [--height 9000]
"""
import argparse, pathlib, subprocess, tempfile
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('html'); ap.add_argument('out')
ap.add_argument('--width', type=int, default=1280)
ap.add_argument('--height', type=int, default=9000)
a = ap.parse_args()
html = pathlib.Path(a.html).resolve(); out = pathlib.Path(a.out).resolve()
out.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as ud:
    r = subprocess.run(['google-chrome', '--headless=new', '--no-sandbox', '--disable-gpu', '--hide-scrollbars',
                        f'--user-data-dir={ud}', '--enable-logging=stderr', '--v=0', '--virtual-time-budget=4000',
                        '--allow-file-access-from-files', f'--window-size={a.width},{a.height}',
                        f'--screenshot={out}', html.as_uri()], capture_output=True, text=True, timeout=180)
errs = [l for l in r.stderr.splitlines() if ('CONSOLE' in l or 'Uncaught' in l) and 'font' not in l.lower()]
im = Image.open(out).convert('RGB'); w, h = im.size; px = im.load()
bg = px[w // 2, h - 1]; y = h - 1
while y > 0 and all(px[x, y] == bg for x in range(0, w, 5)):
    y -= 1
im.crop((0, 0, w, min(h, y + 40))).save(out)
print(f'{out}  {w}x{min(h, y + 40)}')
print('console: clean' if not errs else 'console:\n  ' + '\n  '.join(errs))
