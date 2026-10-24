# EASY_SPEC: flow.html 본문·해설을 "처음 배우는 사람" 눈높이로 다시 쓰기

REPO = `/home/theo_lab/Advanced-Representation-Learning-class`
FLOW = `REPO/exam/_flow` (페이지 소스) · EASY = `FLOW/work/easy` (이번 작업 폴더)

## 0. 왜 하나

`REPO/exam/flow.html`은 고급표현학습 L01–L07 전체를 "학습 target은 어디서 오나?"라는 한 줄기로 훑는 복습 페이지야. 지금 글은 **이미 내용을 아는 사람이 복습하는** 톤이라 너무 압축돼 있어. 사용자 요청은 이래.

> "말이 길어져도 상관 없으니까 전반적으로 용어나, 해설은 좀 쉽게 풀어서 해줘. 마치 처음 공부하는 사람이 하는 것처럼. 원래 내용을 알던 사람이 복습하는 게 아니라."

**독자**: 한양대 AI 석박통합 학생이야.
- 주 분야는 **RL · flow matching · simulation · VLA(action head 쪽)**. 딥러닝 기본은 알아: backprop, loss, CNN/ViT가 뭔지, PyTorch, policy softmax.
- **representation learning 쪽은 처음이야**. embedding, projection head, positive/negative, augmentation, linear probing, collapse, stop-gradient, teacher/student, pseudo label 같은 말이 "다들 알지?" 하고 지나가면 못 따라가.
- 수업은 영어 논문 세미나 연속이라 첫 수업부터 놓쳤어.
- 시험은 2026-10-29 퀴즈: 영어 T/F · MCQ · 빈칸 · pseudo code, 쉬움, "basic concept & core idea of the papers".

## 1. 쓰는 방식 (핵심)

1. **처음 나오는 용어는 그 자리에서 풀어.** chapter마다 독자가 처음 보는 단어는 바로 옆에 괄호나 짧은 문장으로 설명해.
   - 예: "augmentation(같은 사진을 자르거나 색을 바꿔서 만든 변형본)".
   - 용어집 링크 `<a href="#t-...">`를 걸어도 되지만, 링크만 달고 설명을 생략하면 안 돼.
   - 다른 chapter에서 이미 설명한 용어도 한 번은 짧게 다시 상기시켜. 독자가 중간부터 읽을 수 있어.
2. **방법 하나를 이렇게 풀어** (필요한 것만 골라):
   - **한 줄**: 이 방법이 하는 일을 일상어로.
   - **문제**: 이전 방법의 어떤 불편함 때문에 나왔나.
   - **핵심 아이디어**: 쉬운 말로.
   - **어떻게**: 단계가 있으면 `<ol>`로 1, 2, 3.
   - **왜 돼?**: 직관.
   - **이전과 차이**.
   - **시험 포인트**.
3. **수식은 기호 하나하나를 말로 풀어.** 무엇을 키우고 무엇을 줄이는지, 값이 커지거나 작아지면 어떻게 되는지까지. 수식은 지우지 말 것 (시험에 나와).
4. **구체적 예시**를 써. 숫자가 작은 toy 예시면 좋아.
   - 예: "batch에 사진 4장 → view 8개 → 한 view당 positive 1개, negative 6개".
   - 예시 숫자는 설명용이니까 슬라이드 실험 수치처럼 보이게 쓰지 말 것.
5. **비유**는 RL 비유와 일상 비유 둘 다 좋아. 기존 `RL로 번역` 박스는 유지하고 더 쉽게 다듬어도 돼. 비유가 오해를 부르지 않는지 꼭 따져봐.
6. 문장은 짧게, 한 문장에 한 생각. 문단이 4–5줄을 넘으면 쪼개거나 `<ul>`로 나눠.
7. 말투는 기존과 같아: **한국어 반말**. 기술 용어·방법 이름은 영어 그대로 (시험이 영어라서). 처음 나올 때만 한국어 풀이를 붙여. 이모지 금지.
8. 길이는 지금의 **2–3배까지 OK**. 그래도 같은 말 반복이나 군더더기는 빼. 목표는 "처음 보는 사람이 이 페이지만 읽고 따라올 수 있는가"야.

**새 박스** (CSS 이미 있음, 라이트/다크 대응): 직관·비유·예시를 본문과 분리하고 싶을 때만 써. method 하나에 많아야 1개, 남발 금지.

```html
<div class="note easy"><span class="tag">쉽게 말하면</span><p>…</p></div>
```

- tag 텍스트는 `쉽게 말하면` | `비유` | `예시로 보면` | `왜?` 중 하나.
- method 블록(`.m-body`) 안에서는 `<p><span class="k">쉽게</span>…</p>`, `<span class="k">어떻게</span>`, `<span class="k">왜 돼?</span>` 같은 키 라벨도 쓸 수 있어. 기존 키(`문제`, `핵심`, `차이`, `정의` …)는 그대로 둬.
- `.m-body` 안에 `<ul>`/`<ol>`도 쓸 수 있어 (스타일 있음).

**Before → After 예시** (L04 Debiased Contrastive)

```html
<!-- before -->
<p><span class="k">문제</span>random으로 뽑은 negative에 같은 class(false negative)가 섞여 성능이 크게 떨어져.</p>

<!-- after -->
<p><span class="k">한 줄</span>"negative라고 뽑았는데 사실 같은 종류"인 샘플 때문에 생기는 손해를 label 없이 보정하는 방법.</p>
<p><span class="k">문제</span>contrastive learning은 label이 없으니까 negative(멀어져야 할 샘플)를 batch에서 그냥 아무거나 뽑아. 그러면 anchor가 강아지 사진일 때 negative 중에도 강아지 사진이 섞일 수 있어. 이게 <b>false negative</b>야. 같은 class를 억지로 밀어내니까 성능이 크게 떨어져 (p.7–8).</p>
<p><span class="k">핵심</span>어느 샘플이 false negative인지는 label 없이 몰라. 대신 "아무거나 뽑으면 η<sup>+</sup> 비율로 같은 class가 섞인다"는 사실을 이용해서, 섞인 분포에서 positive 몫을 빼 진짜 negative 분포를 추정해: <span class="math">p<sup>−</sup> = (p − η<sup>+</sup>p<sup>+</sup>) / η<sup>−</sup></span>.</p>
<ul>
  <li><span class="math">p</span>: 그냥 random으로 뽑을 때의 분포 (진짜 negative + false negative 섞임)</li>
  <li><span class="math">p<sup>+</sup></span>: 같은 class(positive) 분포, <span class="math">η<sup>+</sup></span>: 같은 class일 확률 (class 비율)</li>
  <li><span class="math">η<sup>−</sup> = 1 − η<sup>+</sup></span>로 나눠서 다시 확률 분포로 맞춰</li>
</ul>
```

## 2. 지켜야 할 것 (깨지면 빌드가 망가지거나 시험 정보가 사라짐)

- **`<!--slot:ID-->` 줄은 글자 하나 바꾸지 말고, 순서와 위치를 그대로 둬.** 빌드할 때 그 자리에 슬라이드 그림·그래프가 들어가.
  - method 블록 바로 뒤의 slot은 그 블록 바로 뒤에 있어야 해.
  - 설명을 추가할 땐 블록 **안**(`.m-body`)에 넣어. 블록과 slot 사이에 끼워 넣지 말 것.
  - 표에 대한 설명은 표 **앞**이나 셀 안에 넣어 (표 뒤 slot 다음에 두면 그림 뒤로 밀려).
- `<section id>`, `.ch-head`, `.ch-q`, method 블록 뼈대(`<div class="m"><div><span class="m-name">…</span><span class="m-src">…</span></div><div class="m-body">…</div></div>`), `.steps`, `.tbl`, `.quiz`, `<label class="done">`의 구조와 id는 유지해.
  - `m-name`(방법 이름)과 `m-src`(페이지)는 바꾸지 마.
  - 새 method 블록이나 표는 꼭 필요할 때만 추가해.
- **정보는 빼지 마.** 페이지 번호 `p.N`, 정의 원문 영어 인용, 수식, `시험 함정` 박스의 모든 항목, `다음으로` 박스, 셀프 체크 3문제는 유지해.
  - 셀프 체크의 영어 문제와 정답 자체는 바꾸지 마. 답 해설은 더 쉽게 풀어도 돼.
- **정확성 최우선.** 쉽게 쓰다가 틀린 말을 만들면 최악이야.
  - 근거는 `REPO/notes/NN_*.md` (가장 자세함), `REPO/exam/core_ideas.md`, `REPO/exam/pseudocode.md`, 슬라이드 PDF `REPO/lectures/NN_*.pdf`.
  - 슬라이드에 없는 **새 사실**(수치, 논문 디테일)은 추가하지 마. 일반 ML 기초 정의(softmax가 뭔지 등)와 비유는 OK야.
  - 확신이 없으면 원래 문장을 살려.
  - 흔한 함정 짝은 특히 조심해: student↔teacher, prediction EMA↔weight EMA, MSE↔KL↔CE, temperature τ↔EMA τ↔threshold τ, centering은 teacher 쪽만, 어떤 쪽에 stop-gradient가 있나, hard↔super-hard, Π-model↔Temporal Ensembling↔Mean Teacher 차이.
- **본문(parts)에서는 "위 그림", "아래 그래프" 같은 그림 위치 언급 금지.** 같은 본문이 그림 없는 claude.ai artifact로도 쓰일 수 있어서야. 그림을 가리키는 말은 figure/viz 캡션 안에서만 써.
- HTML: `<script>`/`<style>` 추가 금지. 색 리터럴(`#123`, `red`) 금지, 필요하면 기존 class만 써. 수식은 기존처럼 `<span class="math">` + `<sub>`/`<sup>`로 써 (MathJax 없음).
  - 태그 짝을 맞춰. `&`는 `&amp;`, `<`는 `&lt;`로.

## 3. 파일 담당 (자기 파일만 수정)

| 담당 | 본문 part (`EASY/parts/`) | 캡션 |
|---|---|---|
| W01 front | `01_hero.html`, `02_how.html`, `03_map.html` | `FLOW/viz/viz_map_timeline.html` |
| W02 terms | `04_terms.html` | `FLOW/viz/viz_terms_temperature.html` |
| W03 l01 | `05_l01.html` | `FLOW/figs/l01.json`, `FLOW/viz/viz_l01_*.html` |
| W04 l02 | `06_l02.html` | `FLOW/figs/l02.json`, `FLOW/viz/viz_l02_*.html` |
| W05 l03 | `07_l03.html` | `FLOW/figs/l03.json`, `FLOW/viz/viz_l03_*.html` |
| W06 l04 | `08_l04.html` | `FLOW/figs/l04.json`, `FLOW/viz/viz_l04_*.html` |
| W07 l05 | `09_l05.html` | `FLOW/figs/l05.json`, `FLOW/viz/viz_l05_*.html` |
| W08 l06 | `10_l06.html` | `FLOW/figs/l06.json`, `FLOW/viz/viz_l06_*.html` |
| W09 l07 | `11_l07.html` | `FLOW/figs/l07.json`, `FLOW/viz/viz_l07_*.html` |
| W10 back | `12_patterns.html`, `14_bridge.html`, `15_last.html` | `FLOW/viz/viz_patterns_*.html` |
| W11 code | `13_code.html` | (없음) |

- `00_head.html`, `16_tail.html`, `FLOW/base.html`, `FLOW/build.py`, `exam/flow.html`, `notes/`, `exam/*.md`, `lectures/`는 **수정 금지**. git 명령도 금지.
- 수정 전 원본은 `EASY/orig/`에 있어 (parts, figs, viz 복사본). 비교용이고 수정 금지.

**캡션 수정 범위**
- `figs/lNN.json`: 각 항목의 `caption`만 수정해 (`alt`는 필요하면). `slot`, `src`, `page`, `width`, `row`, `after`, `fold`는 그대로 둬.
  - JSON은 Python으로 읽어서 고치고 `json.dump(..., ensure_ascii=False, indent=2)`로 저장해.
  - 저장 후 `python3 -I -c "import json;json.load(open('FILE'))"`로 확인해.
  - 캡션은 "이 그림에서 뭘 봐야 하나"를 처음 보는 사람 기준으로 써. 그림 속 기호나 화살표가 무슨 뜻인지 1–4문장.
- `viz/*.html`: `<figcaption>`, `.viz-title`, 그리고 화면에 뜨는 설명 문자열(JS의 메시지 텍스트, 범례 아래 설명 문장)만 수정해. SVG 도형이나 좌표, 축 라벨은 건드리지 마 (겹침 검수를 이미 통과했어). 라벨이 어렵다면 캡션에서 풀어 줘.
  - viz를 고쳤으면 아래 미리보기와 스크린샷으로 확인해.

**용어집 앵커** (W02가 이 id를 `<dt id="…">`에 반드시 달고, 다른 사람은 이 id로만 링크해)

`t-supervised` (supervised / self-supervised / semi-supervised 구분) · `t-label` · `t-encoder` · `t-embedding` · `t-mlp` · `t-projection-head` · `t-predictor` · `t-prototype` · `t-cls-token` · `t-softmax` · `t-logit` · `t-cross-entropy` · `t-mse` · `t-kl` · `t-entropy` · `t-one-hot` · `t-cosine` · `t-pretext` · `t-downstream` · `t-linear-probing` · `t-fine-tuning` · `t-zero-shot` · `t-augmentation` · `t-collapse` · `t-invariance` · `t-anchor` · `t-hard-negative` · `t-temperature` · `t-infonce` · `t-memory-bank` · `t-student-teacher` · `t-ema` · `t-stop-gradient` · `t-distillation` · `t-centering` · `t-semi-objective` · `t-consistency` · `t-pseudo-label` · `t-threshold` · `t-weak-strong` · `t-confirmation-bias` · `t-sharpening`

## 4. 미리보기 · 점검 (cwd = EASY)

```bash
cd /home/theo_lab/Advanced-Representation-Learning-class/exam/_flow/work/easy
python3 -I easy.py preview l04            # → FLOW/preview/chapter_l04.html (+ _dark). id: how map terms l01..l07 patterns code bridge last
python3 -I easy.py full W01               # 전체 페이지 → EASY/full_W01.html (hero 확인용, 이름은 자기 담당)
python3 -I easy.py check                  # slot 순서, #t- 링크, 태그 짝, 금지 패턴 (남의 part 문제도 보일 수 있음, 자기 것만 고치면 됨)
python3 -I ../../shot.py ../../preview/chapter_l04.html ../../shots/easy_l04.png --width 900 --height 12000      # 라이트 스크린샷 + console
python3 -I ../../shot.py ../../preview/chapter_l04_dark.html ../../shots/easy_l04_dark.png --width 420 --height 20000
python3 -I ../../build.py --preview viz/viz_l04_frameworks.html   # (FLOW에서) viz 하나만 미리보기
```

- 스크린샷은 Read로 직접 열어 봐. 긴 페이지라 일부만 봐도 되지만, 새 박스·목록·수식이 깨지지 않았는지는 확인해.
- `easy.py check`에서 **자기 part의 문제는 0건**이어야 해. 다른 담당 part의 `#t-` 링크 경고는 W02 작업이 끝나면 사라져.

## 5. 끝나면 보고 (최종 메시지, 짧게)

- 바꾼 파일 목록, 길이 변화(대략 몇 배)
- 새로 풀어 쓴 용어나 개념 목록 (한 줄씩)
- 확신 없는 부분: 원래 문장을 살렸거나, 슬라이드와 노트가 달라 보이는 곳

---

## 6. 검수자 (reviewer) 규칙

검수자는 **파일을 수정하지 않아.** `EASY/review/<담당ID>.md` 하나만 써.

**보는 것**
1. 원본과 새 버전 비교: `diff EASY/orig/NN_x.html EASY/parts/NN_x.html`, `EASY/orig/figs/lNN.json` ↔ `FLOW/figs/lNN.json`, `EASY/orig/viz/…` ↔ `FLOW/viz/…`.
2. 렌더 결과: `easy.py preview <id>` → shot.py 스크린샷을 Read로 확인 (라이트 900px, 다크 420px).
3. 근거: `REPO/notes/NN_*.md`, `REPO/exam/core_ideas.md`, `REPO/exam/pseudocode.md`.
   - 슬라이드 확인은 `pdftoppm -r 80 -f P -l P -png REPO/lectures/NN_*.pdf EASY/render/<내ID>/p`로 렌더한 뒤 Read로 봐.
   - 텍스트는 `pdftotext -f P -l P -layout …`로 볼 수 있어.

**체크리스트**
- **A. 정확성 (MUST급)**
  - 쉽게 풀다가 틀린 말이 됐나? 과잉 일반화도 포함. 예: "teacher는 학습 안 한다"는 틀렸어. EMA로 갱신돼.
  - 슬라이드/노트와 다른 주장이 있나? 슬라이드에 없는 새 사실이 추가됐나?
  - 원래 있던 시험 정보가 빠졌나? 페이지 번호, `시험 함정` 항목, 수식, 정의 인용, 셀프 체크.
  - 함정 짝이 흐려졌나? student↔teacher, MSE↔KL↔CE, τ의 세 의미, centering 위치, stop-gradient 위치 등.
- **B. 처음 배우는 사람 기준**
  - 설명 없이 처음 나오는 용어가 있나? (가장 중요)
  - 단계를 건너뛴 곳, "왜?"가 빠진 곳이 있나?
  - 너무 길거나 빽빽한 문단, 같은 말 반복이 있나?
  - 비유가 오해를 부르나?
  - 수식 기호를 말로 안 풀었나?
- **C. 형식**
  - 2장 규칙(slot 위치, 구조, 그림 위치 언급 금지, 색 리터럴 등)을 지켰나? `easy.py check`로 확인.
  - 말투가 반말이고 기술 용어는 영어인가?
  - 렌더가 깨졌나? 겹침, 잘림, console 에러.

**출력 형식** (`EASY/review/<ID>.md`)

```
# Review: <담당> (<파일들>)
## MUST-FIX
1. <파일>:<줄 또는 "인용구"> 무엇이 문제 → 근거(p.N / notes) → 이렇게 고쳐 (구체적인 대체 문장)
## SHOULD-FIX
## NICE
## Verified OK (확인한 것 짧게)
```
수정 문장까지 구체적으로 써. 작성자가 그대로 반영할 수 있게.
