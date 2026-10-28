#!/usr/bin/env python3
"""Assemble the local study page.

  python3 -I build.py                      -> <repo>/exam/flow.html (final, relative ../assets paths)
  python3 -I build.py --preview viz/X.html -> preview/X.html + preview/X_dark.html (one fragment alone)
  python3 -I build.py --chapter l04        -> preview/chapter_l04.html + _dark (one section with its figs + viz)
  --base FILE (base.html 대신), --out FILE (전체 페이지를 다른 곳에, 절대 이미지 경로)

Inputs (all in this folder):
  base.html          page text with <!--slot:ID--> markers (see slots.md)
  figs/*.json        figure manifests  [{slot, src, alt, caption, page, width?, row?}]
  viz/*.json         viz manifests     [{slot, file}]   (file = fragment in viz/)
"""
import argparse, html, json, pathlib, re, sys

FLOW = pathlib.Path(__file__).resolve().parent
REPO = FLOW.parent.parent  # exam/_flow -> repo root
OUT = REPO / 'exam' / 'flow.html'

CODE_SLIDES = {
    'code-x-clr': ('assets/lecture03/p067.png', 'L03 p.67'),
    'code-simclr': ('assets/lecture04/p025.png', 'L04 p.25'),
    'code-dino': ('assets/lecture04/p040.png', 'L04 p.40'),
    'code-moco': ('assets/lecture05/p012.png', 'L05 p.12'),
    'code-clip': ('assets/lecture05/p035.png', 'L05 p.35'),
    'code-model': ('assets/lecture06/p018.png', 'L06 p.18'),
    'code-temporal-ensembling': ('assets/lecture06/p020.png', 'L06 p.20'),
    'code-label-propagation': ('assets/lecture07/p011_lp_algorithm.png', 'L07 p.11'),
    'code-dividemix': ('assets/lecture07/p054_algorithm.png', 'L07 p.54'),
}

EXTRA_CSS = r"""
<style>
/* ---- local-build additions: reset, slide figures, viz frame, lightbox, theme toggle ---- */
*,*::before,*::after{box-sizing:border-box}
html,body{margin:0}
:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
img{max-width:100%;height:auto}
[hidden]{display:none!important}
.local-note{font-size:.9rem;color:var(--muted);margin:0 0 1.2rem}
/* slide figures */
.fig{margin:1.1rem 0 1.3rem;display:grid;gap:.45rem;justify-items:center}
.fig a.zoom{display:block;width:100%;max-width:32.5rem;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#fff;cursor:zoom-in}
.fig img{display:block;width:100%;height:auto}
.fig figcaption{font-size:.86rem;color:var(--muted);line-height:1.6;justify-self:stretch}
.fig .pg{font:500 .72rem/1 var(--f-mono);color:var(--muted);margin-left:.35rem;white-space:nowrap}
.figrow{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.9rem;margin:1.1rem 0 1.3rem}
.figrow .fig{margin:0}
/* inline: 설명하는 문단 바로 옆(카드 본문·계보 단계·용어 dd 안)에 들어간 그림·그래프 */
.m-body .fig,.steps .fig,dd .fig{margin:.55rem 0 .8rem}
.m-body .viz,.steps .viz,dd .viz{margin:.6rem 0 .85rem}
.m-body .viz-scroll>svg,.steps .viz-scroll>svg,dd .viz-scroll>svg{min-width:28rem}
.m-body details.slide,.steps details.slide,dd details.slide{margin:.1rem 0 .4rem}
details.slide{margin:.2rem 0 0}
details.slide summary{cursor:pointer;font:500 .8rem/1.4 var(--f-mono);color:var(--accent)}
details.slide .fig{margin:.6rem 0 0}
:root[data-theme="dark"] .fig img{filter:brightness(.93)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .fig img{filter:brightness(.93)}}
/* viz frame (shared by every fragment) */
.viz{margin:1.2rem 0 1.4rem;padding:1rem 1rem .9rem;border:1px solid var(--line);border-radius:10px;background:var(--surface);display:grid;gap:.7rem;min-width:0}
.viz-head{display:flex;flex-wrap:wrap;gap:.3rem .6rem;align-items:baseline}
.viz-tag{font:600 .66rem/1 var(--f-mono);letter-spacing:.06em;text-transform:uppercase;color:var(--accent)}
.viz-title{font:600 1rem/1.35 var(--f-display);margin:0}
.illus{font:500 .64rem/1 var(--f-mono);color:var(--muted);border:1px solid var(--line);border-radius:999px;padding:.25rem .45rem}
.viz-scroll{overflow-x:auto;min-width:0}
.viz svg{display:block;width:100%;height:auto;max-width:100%}
.viz-scroll>svg{min-width:34rem}
.viz canvas{display:block;width:100%;max-width:100%}
.viz-ctrl{display:flex;flex-wrap:wrap;gap:.6rem 1.2rem;align-items:center;font-size:.85rem;color:var(--muted)}
.viz-ctrl label{display:inline-flex;gap:.5rem;align-items:center}
.viz-ctrl input[type=range]{accent-color:var(--accent);width:min(13rem,60vw)}
.viz-ctrl output{font:600 .8rem/1 var(--f-mono);color:var(--ink);min-width:3.5ch}
.viz-ctrl button{font:500 .8rem/1 var(--f-body);color:var(--ink);background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:.4rem .65rem;cursor:pointer}
.viz-ctrl button[aria-pressed="true"]{border-color:var(--accent);color:var(--accent);background:var(--accent-soft)}
.viz figcaption{font-size:.86rem;color:var(--muted);line-height:1.6}
/* SVG helper classes: use these (or style="fill:var(--x)"), never literal colors */
.viz .t{fill:var(--ink);font-family:var(--f-body);font-size:13px}
.viz .t-m{fill:var(--muted)}
.viz .t-acc{fill:var(--accent)}
.viz .t-rl{fill:var(--rl)}
.viz .t-trap{fill:var(--trap)}
.viz .t-ok{fill:var(--ok)}
.viz .mono{font-family:var(--f-mono)}
.viz .b{font-weight:600}
.viz .ax{stroke:var(--muted);stroke-width:1;fill:none}
.viz .grid{stroke:var(--line);stroke-width:1;fill:none}
.viz .ln-ink{stroke:var(--ink);fill:none}
.viz .ln-acc{stroke:var(--accent);fill:none}
.viz .ln-rl{stroke:var(--rl);fill:none}
.viz .ln-trap{stroke:var(--trap);fill:none}
.viz .ln-ok{stroke:var(--ok);fill:none}
.viz .ln-m{stroke:var(--muted);fill:none}
.viz .f-acc{fill:var(--accent)}
.viz .f-accs{fill:var(--accent-soft)}
.viz .f-rl{fill:var(--rl)}
.viz .f-rls{fill:var(--rl-soft)}
.viz .f-trap{fill:var(--trap)}
.viz .f-traps{fill:var(--trap-soft)}
.viz .f-ok{fill:var(--ok)}
.viz .f-surf{fill:var(--surface)}
.viz .f-bg{fill:var(--bg)}
.viz .f-line{fill:var(--line)}
.viz .f-m{fill:var(--muted)}
.viz .box{fill:var(--surface);stroke:var(--ink);stroke-width:1.2}
.viz .box-train{fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5}
.viz .box-ema{fill:var(--surface);stroke:var(--muted);stroke-width:1.5;stroke-dasharray:5 4}
.viz .box-loss{fill:var(--trap-soft);stroke:var(--trap);stroke-width:1.4}
.viz .box-data{fill:var(--bg);stroke:var(--muted);stroke-width:1}
.viz .dash{stroke-dasharray:5 4}
/* lightbox */
.lb{position:fixed;inset:0;z-index:50;background:rgba(8,10,14,.9);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:.6rem;overflow:auto;padding:max(16px,env(safe-area-inset-top,0px)) max(16px,env(safe-area-inset-right,0px)) max(16px,env(safe-area-inset-bottom,0px)) max(16px,env(safe-area-inset-left,0px))}
.lb img{max-width:100%;max-height:calc(100% - 5rem);border-radius:6px;background:#fff;cursor:zoom-in}
.lb.full{justify-content:flex-start;align-items:flex-start}
.lb.full img{max-width:none;max-height:none;width:auto;cursor:zoom-out}
.lb-x{position:fixed;top:calc(12px + env(safe-area-inset-top,0px));right:calc(12px + env(safe-area-inset-right,0px));font:600 .8rem/1 var(--f-body);color:#fff;background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.35);border-radius:999px;padding:.55rem .8rem;cursor:pointer;z-index:2}
.lb-hint{margin:0;font:500 .75rem/1.4 var(--f-mono);color:rgba(255,255,255,.7)}
.lb.full .lb-hint{display:none}
details.more{margin:.4rem 0 1.2rem}
/* theme toggle */
.theme-btn{position:fixed;right:calc(14px + env(safe-area-inset-right,0px));bottom:calc(14px + env(safe-area-inset-bottom,0px));z-index:20;font:600 .74rem/1 var(--f-mono);color:var(--ink);background:var(--surface);border:1px solid var(--line);border-radius:999px;padding:.55rem .8rem;cursor:pointer;box-shadow:0 2px 10px rgba(0,0,0,.12)}
@media (max-width:640px){.figrow{grid-template-columns:minmax(0,1fr)}.theme-btn{position:absolute;top:calc(10px + env(safe-area-inset-top,0px));bottom:auto;right:calc(12px + env(safe-area-inset-right,0px));box-shadow:none;padding:.4rem .6rem}}
@media (max-width:600px){.viz:has(.viz-scroll) .viz-head::after{content:"옆으로 밀어서 보기 →";font:500 .66rem/1 var(--f-mono);color:var(--accent);margin-left:auto}}
</style>
"""

LOCAL_JS = r"""
<button class="theme-btn" id="theme-btn" type="button" aria-label="테마 바꾸기">테마: 시스템</button>
<script>
(function(){
  var KEY='arl-flow-theme', order=['system','light','dark'], names={system:'시스템',light:'라이트',dark:'다크'};
  var btn=document.getElementById('theme-btn'), cur='system';
  try{ cur=localStorage.getItem(KEY)||'system'; }catch(e){}
  function apply(){
    if(cur==='system') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme',cur);
    btn.textContent='테마: '+names[cur];
  }
  if(document.documentElement.hasAttribute('data-theme')&&cur==='system'){ cur=document.documentElement.getAttribute('data-theme'); }
  apply();
  btn.addEventListener('click',function(){ cur=order[(order.indexOf(cur)+1)%3]; try{localStorage.setItem(KEY,cur);}catch(e){} apply(); });
  document.addEventListener('click',function(ev){
    var a=ev.target.closest&&ev.target.closest('a.zoom'); if(!a) return;
    ev.preventDefault();
    var lb=document.createElement('div'); lb.className='lb'; lb.setAttribute('role','dialog'); lb.setAttribute('aria-modal','true'); lb.tabIndex=-1;
    var im=document.createElement('img'); im.src=a.getAttribute('href'); im.alt=(a.querySelector('img')||{}).alt||'';
    var x=document.createElement('button'); x.type='button'; x.className='lb-x'; x.textContent='닫기 ✕';
    var hint=document.createElement('p'); hint.className='lb-hint'; hint.textContent='그림을 누르면 원래 크기 ↔ 화면 맞춤';
    lb.appendChild(x); lb.appendChild(im); lb.appendChild(hint); document.body.appendChild(lb);
    var prev=document.documentElement.style.overflow; document.documentElement.style.overflow='hidden'; lb.focus();
    function close(){ lb.remove(); document.documentElement.style.overflow=prev; document.removeEventListener('keydown',esc); a.focus(); }
    function esc(e){ if(e.key==='Escape') close(); }
    im.addEventListener('click',function(e){ e.stopPropagation(); lb.classList.toggle('full'); });
    x.addEventListener('click',close);
    lb.addEventListener('click',function(e){ if(e.target===lb) close(); });
    document.addEventListener('keydown',esc);
  });
})();
</script>
"""


def load_manifests(folder):
    items = {}
    for f in sorted((FLOW / folder).glob('*.json')):
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
        except Exception as e:  # noqa: BLE001
            print(f'!! bad json {f.name}: {e}', file=sys.stderr)
            continue
        for d in data:
            d['_from'] = f.name
            items.setdefault(d.get('slot', '?'), []).append(d)
    return items


def fig_html(d, prefix):
    src = d['src']
    if not (REPO / src).exists():
        print(f"!! missing image {src} ({d['_from']})", file=sys.stderr)
    w = d.get('width')
    style = f' style="max-width:{int(w)}px"' if w else ''
    url = prefix + src
    cap = d.get('caption', '')
    pg = d.get('page', '')
    pg_html = f'<span class="pg">{html.escape(str(pg))}</span>' if pg else ''
    return (f'<figure class="fig"><a class="zoom" href="{url}"{style}>'
            f'<img src="{url}" alt="{html.escape(d.get("alt", ""), quote=True)}"></a>'
            f'<figcaption>{cap}{pg_html}</figcaption></figure>')


def slot_content(sid, figs, vizs, prefix):
    parts, row = [], []
    allf = figs.get(sid, [])
    folded = [d for d in allf if d.get('fold')]
    before = [d for d in allf if not d.get('after') and not d.get('fold')]
    after = [d for d in allf if d.get('after') and not d.get('fold')]
    for d in before:
        h = fig_html(d, prefix)
        if d.get('row'):
            row.append(h)
            if len(row) == 2:
                parts.append('<div class="figrow">' + ''.join(row) + '</div>'); row = []
        else:
            if row:
                parts.append('<div class="figrow">' + ''.join(row) + '</div>'); row = []
            parts.append(h)
    if row:
        parts.append(''.join(row))
    if sid in CODE_SLIDES:
        src, lab = CODE_SLIDES[sid]
        url = prefix + src
        parts.append(f'<details class="slide"><summary>슬라이드 원본 보기 ({lab})</summary>'
                     f'<figure class="fig"><a class="zoom" href="{url}"><img src="{url}" alt="{lab} pseudo code slide"></a></figure></details>')
    for d in vizs.get(sid, []):
        f = FLOW / 'viz' / d['file']
        if not f.exists():
            print(f"!! missing viz fragment {d['file']} ({d['_from']})", file=sys.stderr); continue
        parts.append(f.read_text(encoding='utf-8'))
    for d in after:
        parts.append(fig_html(d, prefix))
    if folded:
        parts.append(f'<details class="slide more"><summary>슬라이드 그림 더 보기 ({len(folded)}장)</summary>'
                     + ''.join(fig_html(d, prefix) for d in folded) + '</details>')
    return '\n'.join(parts)


def split_base(path=None):
    base = pathlib.Path(path or FLOW / 'base.html').read_text(encoding='utf-8')
    i = base.index('</style>') + len('</style>')
    head, body = base[:i], base[i:]
    head = head.replace('<meta name="claude:viewport-fit-sides" content="cover">', '')
    return head, body


def wrap(head, body, dark=False, preview=False):
    theme = ' data-theme="dark"' if dark else ''
    if preview:
        body = body + '<style>.theme-btn{display:none}</style>'
    return ('<!doctype html>\n<html lang="ko"' + theme + '>\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + head + EXTRA_CSS + '\n</head>\n<body>\n' + body + LOCAL_JS + '\n</body>\n</html>\n')


def fill(body, prefix):
    figs, vizs = load_manifests('figs'), load_manifests('viz')
    known = set(re.findall(r'<!--slot:([a-z0-9-]+)-->', body))
    for sid in set(figs) | set(vizs):
        if sid not in known:
            print(f'!! unknown slot "{sid}" in manifest', file=sys.stderr)
    body = re.sub(r'<!--slot:([a-z0-9-]+)-->', lambda m: slot_content(m.group(1), figs, vizs, prefix), body)
    body = body.replace('<p class="thesis">',
                        '<p class="local-note">로컬 버전: 슬라이드 그림과 개념 그래프 포함. 그림을 누르면 크게 보여.</p>\n      <p class="thesis">', 1)
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview')
    ap.add_argument('--chapter')
    ap.add_argument('--base', help='base.html 대신 쓸 파일 (미리보기용)')
    ap.add_argument('--out', help='전체 페이지를 exam/flow.html 대신 여기에 (이미지 경로는 절대경로)')
    a = ap.parse_args()
    head, body = split_base(a.base)
    (FLOW / 'preview').mkdir(exist_ok=True)
    if a.preview:
        frag = pathlib.Path(a.preview)
        if not frag.is_absolute():
            frag = FLOW / frag
        inner = frag.read_text(encoding='utf-8')
        page = ('<div class="wrap" style="max-width:48rem;padding-block:1.5rem"><main>'
                '<section id="preview" style="border:0">' + inner + '</section></main></div>')
        for dark in (False, True):
            out = FLOW / 'preview' / (frag.stem + ('_dark' if dark else '') + '.html')
            out.write_text(wrap(head, page, dark, preview=True), encoding='utf-8')
            print(out)
        return
    prefix = REPO.as_uri() + '/' if (a.chapter or a.out) else '../'
    filled = fill(body, prefix)
    if a.chapter:
        m = re.search(r'(<section id="' + re.escape(a.chapter) + r'">.*?</section>)', filled, re.S)
        if not m:
            sys.exit(f'no section {a.chapter}')
        page = ('<div class="wrap" style="max-width:48rem;padding-block:1rem"><main>' + m.group(1) + '</main></div>')
        for dark in (False, True):
            out = FLOW / 'preview' / (f'chapter_{a.chapter}' + ('_dark' if dark else '') + '.html')
            out.write_text(wrap(head, page, dark, preview=True), encoding='utf-8')
            print(out)
        return
    out = pathlib.Path(a.out).resolve() if a.out else OUT
    out.write_text(wrap(head, filled), encoding='utf-8')
    print(out, f'{out.stat().st_size/1024:.0f} KB')


if __name__ == '__main__':
    main()
