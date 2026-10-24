#!/usr/bin/env python3
"""base_src.html -> base.html with <!--slot:ID--> markers + slots.md listing."""
import re, pathlib
here = pathlib.Path(__file__).resolve().parent
lines = (here/'base_src.html').read_text(encoding='utf-8').split('\n')
out, listing = [], []
sec = 'head'; cnt = {}; in_m = None; in_tbl = False; in_steps = False; in_pc = None
def slug(s):
    s = re.sub(r'<[^>]+>', '', s)
    a = re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')
    return a[:24] or 'x'
def nid(kind, name=''):
    k = (sec, kind); cnt[k] = cnt.get(k, 0) + 1
    return f"{sec}-{kind}{cnt[k]:02d}" + (f"-{slug(name)}" if name else '')
ctx_q = None
for ln in lines:
    m = re.search(r'<section id="([a-z0-9]+)"', ln)
    if m: sec = m.group(1)
    if sec.startswith('l0') and '<div class="note rl">' in ln:
        sid = f"{sec}-end"; out.append(f"  <!--slot:{sid}-->"); listing.append((sec, sid, 'chapter 끝, RL 박스 바로 앞'))
    out.append(ln)
    if 'class="ch-q"' in ln:
        sid = f"{sec}-intro"; out.append(f"  <!--slot:{sid}-->"); listing.append((sec, sid, 'chapter 질문 바로 뒤: ' + re.sub(r'<[^>]+>', '', ln).strip()))
    m = re.search(r'<div class="m"><div><span class="m-name">(.+?)</span>(?:<span class="m-src">(.+?)</span>)?', ln)
    if m: in_m = (m.group(1), m.group(2) or '')
    elif in_m and ln.strip() == '</div></div>':
        sid = nid('m', in_m[0]); out.append(f"    <!--slot:{sid}-->"); listing.append((sec, sid, f"method 블록 뒤: {re.sub(r'<[^>]+>','',in_m[0])} ({in_m[1]})")); in_m = None
    if '<ol class="steps">' in ln: in_steps = True; steps_names = []
    if in_steps:
        steps_names += re.findall(r'<span class="s-name">(.+?)</span>', ln)
        if ln.strip() == '</ol>':
            sid = nid('steps'); out.append(f"  <!--slot:{sid}-->"); listing.append((sec, sid, 'steps 목록 뒤: ' + ' → '.join(re.sub(r'<[^>]+>','',s).replace('&amp;','&') for s in steps_names))); in_steps = False
    if '<div class="tbl">' in ln: in_tbl = True; tbl_head = None
    if in_tbl and tbl_head is None and '<th>' in ln: tbl_head = re.sub(r'<[^>]+>', ' ', ln).split()
    if in_tbl and ln.strip() == '</div>' and prev.strip() == '</table>':
        sid = nid('tbl'); out.append(f"  <!--slot:{sid}-->"); listing.append((sec, sid, 'table 뒤 (헤더: ' + ' '.join(tbl_head or [])[:70] + ')')); in_tbl = False
    if sec == 'terms' and ln.strip() == '</dl>':
        sid = nid('dl'); out.append(f"  <!--slot:{sid}-->"); listing.append((sec, sid, 'glossary 그룹 뒤 (dl 순서: 모델 부품, 학습·평가, Contrastive, Teacher–student, Semi-supervised)'))
    m = re.search(r'<div class="pc-h"><span class="m-name">(.+?)</span>', ln)
    if m: in_pc = m.group(1)
    elif in_pc and ln.strip() == '</div>':
        sid = f"code-{slug(in_pc)}"; out.append(f"  <!--slot:{sid}-->"); listing.append((sec, sid, f"pseudo code 블록 뒤: {in_pc}")); in_pc = None
    prev = ln
(here/'base.html').write_text('\n'.join(out), encoding='utf-8')
md = ['# Slot list (base.html)', '', '| section | slot id | 위치 |', '|---|---|---|']
md += [f"| {a} | `{b}` | {c} |" for a, b, c in listing]
(here/'slots.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
print(len(listing), 'slots')
