"""exam/_parts/NN_*.md의 A·B 섹션을 모아 exam/core_ideas_final.md를 만든다.

  python3 -I exam/_tools/make_core_final.py

새 강의가 나오면 parts 목록에 한 줄만 추가해.
"""
import pathlib, re
repo = pathlib.Path(__file__).resolve().parents[2]
parts = [('08', 'Weakly-Supervised Learning I', repo / 'exam/_parts/08_Weakly_Supervised_Learning_I.md'),
         ('09', 'Weakly-Supervised Learning II', repo / 'exam/_parts/09_Weakly_Supervised_Learning_II.md')]
head = """# 고급표현학습 기말 대비 — Core Ideas (L08–)

> 기말 치트시트야. **L08부터** 강의가 끝날 때마다 블록을 이어 붙여.
> 기말 일정은 2026-12-10 (W15)이고, 범위는 W10–W14로 적어 뒀어. **누적 여부는 아직 몰라.** 누적이면 중간 치트시트 `exam/core_ideas.md`도 같이 봐.
> 이 파일은 `exam/_parts/NN_*.md`의 A·B 섹션을 그대로 모은 거야. 고칠 땐 pool 파일을 고치고 `python3 -I exam/_tools/make_core_final.py`로 다시 생성해.
> 각 방법은 **Problem → Core idea → vs. previous** 순서고, 뒤집기 함정은 `Definition ↔ trap` 줄에 있어. 문제는 pool의 C·D, 부품 역할은 E, pipeline은 F, 코드 카드는 `exam/pseudocode_final.md`에 있어.

## 목차

"""
toc, body = [], []
for num, title, path in parts:
    t = path.read_text()
    a = t.index('## A. Core ideas')
    c = t.index('## C. Question pool')
    block = t[a:c].rstrip()
    block = re.sub(r'^## A\. Core ideas.*$', '### Core ideas', block, flags=re.M)
    block = re.sub(r'^## B\. Must-know definitions & distinctions$', '### Must-know definitions & distinctions', block, flags=re.M)
    # method blocks drop one level under the new "### Core ideas" heading
    block = re.sub(r'^### (?!Core ideas$|Must-know)', '#### ', block, flags=re.M)
    toc.append(f'- [{num}. {title}](#{num})')
    body.append(f'<a id="{num}"></a>\n\n## {num}. {title}\n\n{block}\n')
out = head + '\n'.join(toc) + '\n\n---\n\n' + '\n---\n\n'.join(body)
(repo / 'exam/core_ideas_final.md').write_text(out)
print('wrote', len(out.splitlines()), 'lines')
