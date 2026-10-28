#!/usr/bin/env python3
"""flow.html → exam/flow.pdf (A4, 접힌 그림·정답 전부 펼침, 라이트 테마, 목차 bookmark).

  python3 -I pdf.py [--out PATH] [--zoom 0.88]

build.py --out 으로 이미지 절대경로판을 만든 뒤 <details> 를 모두 open 하고
인쇄용 CSS 를 끼워서 headless Chrome 으로 print-to-pdf 한다. flow.html 자체는 안 건드림.
인터랙티브 그래프는 기본값 상태로 멈춘 그림으로 들어감.
"""
import argparse, pathlib, re, subprocess, sys, tempfile

FLOW = pathlib.Path(__file__).resolve().parent
OUT = FLOW.parent / 'flow.pdf'

PRINT_CSS = """
<style>
@page{size:A4;margin:13mm 12mm 14mm}
:root{--bg:#FFFFFF}
html{zoom:%(zoom)s;-webkit-print-color-adjust:exact;print-color-adjust:exact}
body::before,.rail,.theme-btn,.lb,.done,.local-note{display:none!important}
.wrap{padding-left:0;padding-right:0}
.shell{display:block;padding-block:0}
main>section{break-before:page}
h2,h3,.ch-head,.quiz-h,.pc-h,summary{break-after:avoid}
.fig,.figrow,.viz,pre,details.q,.note,tr,.split>div,.pc,.matrix{break-inside:avoid}
pre,.tbl,.viz-scroll{overflow:visible}
pre{white-space:pre-wrap}
/* 카드 본문 grid 는 인쇄 페이지 경계에서 그림이 위 글자를 덮는 Chrome 버그가 있어서 block 으로 */
.m-body{display:block}
.m-body>*{margin-top:0;margin-bottom:.3rem}
.m-body>*:last-child{margin-bottom:0}
.m-body .fig{margin:.55rem 0 .8rem}.m-body .viz{margin:.6rem 0 .85rem}.m-body .note{margin:.45rem 0}
details.q summary::after{display:none}
details.slide>summary,details.more>summary{cursor:default;color:var(--muted)}
a.zoom{cursor:default}
</style>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--zoom', default='0.88')
    a = ap.parse_args()
    out = pathlib.Path(a.out).resolve()
    with tempfile.TemporaryDirectory() as td:
        page = pathlib.Path(td) / 'flow_print.html'
        r = subprocess.run([sys.executable, '-I', str(FLOW / 'build.py'), '--out', str(page)],
                           capture_output=True, text=True)
        if r.returncode or r.stderr.strip():
            sys.exit('build.py 실패/경고:\n' + r.stdout + r.stderr)
        t = page.read_text(encoding='utf-8')
        t = t.replace('<html lang="ko">', '<html lang="ko" data-theme="light">', 1)
        t = re.sub(r'<details(?![^>]*\bopen\b)', '<details open', t)
        t = t.replace(' loading="lazy"', '')
        t = t.replace('</head>', PRINT_CSS % {'zoom': a.zoom} + '</head>', 1)
        page.write_text(t, encoding='utf-8')
        r = subprocess.run(['google-chrome', '--headless=new', '--no-sandbox', '--disable-gpu',
                            '--no-pdf-header-footer', '--generate-pdf-document-outline',
                            '--run-all-compositor-stages-before-draw', '--virtual-time-budget=15000',
                            f'--user-data-dir={td}/prof', f'--print-to-pdf={out}', page.as_uri()],
                           capture_output=True, text=True, timeout=300)
        if not out.exists():
            sys.exit('chrome print 실패:\n' + r.stderr[-2000:])
    print(out, f'{out.stat().st_size/1024/1024:.1f} MB')


if __name__ == '__main__':
    main()
