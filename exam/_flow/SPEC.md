> 이 폴더(`exam/_flow/`)가 `exam/flow.html`의 소스. 아래 FLOW 경로는 작업 당시 scratchpad 기준이고, 지금은 FLOW = `exam/_flow/`.
> 다시 만들기: `python3 -I exam/_flow/build.py` → `exam/flow.html`. 미리보기는 `--chapter l04` / `--preview viz/<file>.html` (preview/, shots/는 gitignore).
> **2026-10-24 다시 쓰기**: 본문·캡션을 "처음 배우는 사람" 눈높이로 풀어 씀 (Opus 작성 11 + 검수 9). 규칙은 `EASY_SPEC.md`.
> 이제 **`base.html`이 본문 원본**이야 (slot marker 포함, 직접 수정). `base_src.html`은 base.html에서 slot 줄만 뺀 사본(그림 없는 artifact용).
> `make_slots.py`는 최초 slot 생성용이라 다시 돌리면 slot id가 바뀔 수 있음 → 돌리지 말 것.

# SPEC — 고급표현학습 한 바퀴 (local HTML 버전) 팀 작업 규칙

FLOW = `/tmp/claude-1002/-home-theo-lab-Advanced-Representation-Learning-class/4a20c7d1-d36b-49d7-90bf-1077a8aec3db/scratchpad/flow`
REPO = `/home/theo_lab/Advanced-Representation-Learning-class`

## 1. 무엇을 만드나

`REPO/exam/flow.html` = 7강(L01–L07) 전체를 "학습 target은 어디서 오나?"라는 한 줄기로 가볍게 훑는 복습 페이지.
본문 텍스트는 이미 `FLOW/base.html`에 있음 (읽어볼 것). 여기에 **슬라이드 그림**과 **개념 그래프/다이어그램**을 끼워 넣는 게 이번 작업.

- 독자: 한양대 석박통합 학생. 주 분야는 **RL · flow matching · simulation · VLA (action head 쪽)** — representation learning은 비전공이라 기본 용어부터 필요함. 세미나식 수업이라 첫 수업부터 못 따라감.
- 목적: **10/29 퀴즈** (영어 T/F · MCQ · 빈칸, 쉬움, "basic concept & core idea of the papers" + pseudo code). 수치 암기보다 **개념·방법 간 차이·설계 이유**.
- 톤: 캡션·설명은 **한국어 반말**, 기술 용어·방법 이름은 영어 그대로 (예: "teacher는 EMA로만 갱신돼"). 근거 페이지는 `p.N` (PDF 페이지 번호).
- 슬라이드에 없는 내용은 만들지 말 것. 개념 그래프는 개념을 보여주는 그림일 뿐이고, **실험 수치를 지어내지 말 것** (가짜 데이터로 그린 그림은 반드시 `illustrative` 표시 + 캡션에 "개념 그림").

## 2. 끼워 넣는 위치 = slot

`FLOW/base.html` 안에 `<!--slot:ID-->` 주석이 98개 있음. 목록과 위치 설명은 `FLOW/slots.md`.
너는 manifest에 `slot` ID만 적으면 되고, `build.py`가 그 자리에 그림/그래프를 넣어줌. 한 slot에 여러 개 가능 (manifest 순서대로, 그림 먼저 → viz 나중).
**base.html, build.py, shot.py, slots.md, SPEC.md는 수정 금지.**

## 3-A. 슬라이드 그림 담당 (figure curator)

**산출물**
1. 필요하면 새 crop 이미지: `REPO/assets/lectureNN/pNNN_<slug>.png` (NN = 강의 2자리, NNN = PDF 페이지 3자리). **기존 파일 덮어쓰기 금지** — 이름이 겹치면 slug를 바꿀 것.
2. manifest: `FLOW/figs/lNN.json` (JSON 배열)

```json
[
  {"slot": "l04-m03-simclr", "src": "assets/lecture04/p024_simclr_framework.png",
   "alt": "SimCLR framework: two augmentations, encoder f, projection head g, contrastive loss",
   "caption": "SimCLR 구조. 같은 이미지에 augmentation 두 번 → encoder f → projection head g → 두 z를 contrastive loss로 당김. 학습 후엔 g 버리고 h 사용.",
   "page": "L04 p.24", "width": 620}
]
```
- `src`는 REPO 기준 상대경로. `width` = 화면 표시 최대 px (본문 칼럼 폭 ≈ 730px; 작은 도식은 420–560, 큰 그림은 생략하면 칼럼 꽉 채움). `"row": true`를 연속 2개에 주면 나란히 배치.
- `caption`: 1–2문장 반말. **그림에서 무엇을 봐야 하는지** 말해줄 것 (그림 내용 나열 X). HTML `<b>`, `<code>` 사용 가능.

**어떻게**
- 먼저 `REPO/notes/NN_*.md` 를 읽어라. 이미 `<img src="../assets/lectureNN/...">`로 많은 그림이 들어가 있고 앞뒤 문맥이 있음. `REPO/assets/lectureNN/`의 기존 파일을 **우선 재사용**.
- 부족한 그림은 PDF에서 렌더 후 crop:
  `pdftoppm -r 150 -f P -l P -png REPO/lectures/NN_*.pdf /tmp/.../pageP` (출력 2000×1125) → PIL로 그림 영역만 crop (`python3 -I`), 여백 약간 남기기. 슬라이드 제목줄·페이지 번호·로고는 빼도 됨.
  - 렌더 결과는 FLOW/work/ 같은 작업 폴더에 둘 것 (REPO에 중간 파일 남기지 말 것).
  - crop 후 **반드시 Read로 이미지를 직접 보고** 잘림·흐림·엉뚱한 영역 없는지 확인.
- 기존 asset이 슬라이드 전체(2000×1125)라 글자가 너무 작거나 그림이 일부일 땐 새 crop을 만드는 게 나음.
- **고르는 기준** (중요도 순): (1) 방법의 구조/파이프라인 figure (model figure) (2) 핵심 개념 도식 (3) 직관을 주는 예시 그림. **결과 표·수치표는 거의 넣지 말 것** (시험 비중 낮음). 방법당 0–2장, 강의 전체 목표 개수는 assignment에 있음. 가볍게 훑는 페이지라 과하면 안 됨.
- `lectures/` PDF는 **수정 금지**.

## 3-B. 개념 그래프 담당 (viz builder)

**산출물**
1. fragment 파일: `FLOW/viz/viz_<chapter>_<slug>.html` (하나당 viz 하나)
2. manifest: `FLOW/viz/<chapter>.json` → `[{"slot": "l04-tbl02", "file": "viz_l04_frameworks.html"}]`

**fragment 형식** — `FLOW/viz/_example.html` 을 그대로 따라 할 것:
```html
<figure class="viz" id="viz-<chapter>-<slug>">
  <div class="viz-head"><span class="viz-tag">개념 그래프|구조 비교|인터랙티브</span><p class="viz-title">…</p><span class="illus">illustrative</span></div>
  <div class="viz-scroll"><svg viewBox="…" role="img" aria-label="…">…</svg></div>   <!-- 또는 <svg> 바로, 또는 <canvas> -->
  <div class="viz-ctrl">…slider/button…</div>   <!-- 인터랙티브일 때만 -->
  <figcaption>무엇을 봐야 하는지 1–3문장 반말 + 근거 (Lxx p.N)</figcaption>
  <script>(function(){ var root=document.getElementById('viz-…'); … })();</script>
</figure>
```
**규칙**
- 색: **CSS 변수만** — 공용 클래스(`.t .t-m .t-acc .t-rl .t-trap .t-ok .mono .b .ax .grid .ln-ink .ln-acc .ln-rl .ln-trap .ln-ok .ln-m .f-acc .f-accs .f-rl .f-rls .f-trap .f-traps .f-ok .f-surf .f-bg .f-line .f-m .box .box-train .box-ema .box-loss .box-data .dash`, 정의는 build.py의 EXTRA_CSS) 또는 `style="fill:var(--accent)"`. SVG presentation attribute에 색 리터럴(`fill="#123"`, `fill="red"`) 금지. 라이트/다크 둘 다에서 읽혀야 함.
  - 토큰: `--bg --surface --ink --muted --line --accent --accent-soft --rl --rl-soft --trap --trap-soft --ok --code-bg`, 폰트 `--f-display --f-body --f-mono`.
  - 의미: accent = 학습되는 쪽/핵심, trap = loss·주의, rl = RL 비유, ok = 정답·positive, muted = 보조.
- 추가 CSS가 필요하면 fragment 안 `<style>`에서 **`#viz-…` 아래로만** scope. 전역 셀렉터 금지.
- JS: fragment 안 IIFE 하나, `root` 안에서만 querySelector, 전역 변수 금지, 외부 라이브러리 금지. **JS 없이도(초기 상태로) 그림이 다 보여야** 함 → 초기 값으로 바로 `draw()` 호출. 애니메이션은 필요 없음 (쓰면 `prefers-reduced-motion` 존중).
- id/for는 페이지 전체에서 유일하게: `viz-<chapter>-<slug>-…` 접두어.
- 크기: viewBox 폭 560–720. 글자는 viewBox 단위로 13–15px 이상. 라벨이 많은 도식은 `<div class="viz-scroll">`로 감싸면 모바일에서 가로 스크롤(최소 34rem)됨. 단순한 차트는 감싸지 말고 그냥 축소되게.
- 축·눈금·라벨은 실제 값과 맞을 것. 라벨끼리, 라벨과 선이 겹치지 않을 것. viewBox 밖으로 나가지 않을 것.
- fragment 하나 ≤ 15KB 정도.
- **구조 다이어그램 공통 문법** (모든 chapter 통일):
  - 입력 이미지/데이터 = `.box-data` 작은 사각형 (x, x₁, x₂, u …)
  - **gradient로 학습되는 네트워크** = `.box-train` (accent 실선)
  - **EMA/momentum으로만 갱신되는 네트워크** = `.box-ema` (회색 점선) + 학습 네트워크 → EMA 네트워크로 가는 **점선 화살표에 "EMA" 라벨**
  - **stop-gradient** = 화살표 위 짧은 이중 사선 `//` + 작은 라벨 `sg` (trap 색)
  - loss = `.box-loss` 둥근 사각형, 안에 loss 이름 (InfoNCE, MSE, CE …)
  - 화살표 marker는 `<defs><marker>` 안 path에 `style="fill:var(--ink)"` 처럼 변수로.
  - 범례가 필요하면 그림 아래쪽에 작게.

**미리보기·자기 점검 (필수)**
```bash
cd FLOW
python3 -I build.py --preview viz/viz_l04_frameworks.html          # preview/*.html (light, _dark)
python3 -I shot.py preview/viz_l04_frameworks.html shots/viz_l04_frameworks.png --width 900 --height 2000
python3 -I shot.py preview/viz_l04_frameworks_dark.html shots/viz_l04_frameworks_dark.png --width 420 --height 2000
```
`shot.py`가 console 에러도 출력함 → **clean**이어야 함. 만든 PNG를 Read로 직접 보고 겹침·잘림·안 보이는 글자 고칠 것 (라이트 900px + 다크 420px 둘 다).

## 4. 공통 금지
- git 명령(add/commit/push) 금지. `lectures/`, `notes/`, `exam/*.md` 수정 금지.
- 자기 담당 외 파일 수정 금지 (다른 chapter의 figs/viz/assets 건드리지 말 것).
- 웹 검색 불필요. 근거는 슬라이드(PDF), `REPO/notes/NN_*.md`, `REPO/exam/core_ideas.md`, `REPO/exam/pseudocode.md`.

## 5. 끝나면 보고 (최종 메시지, 짧게)
- 만든/재사용한 파일 목록 + 각 slot
- 새로 만든 crop 이미지 목록
- 확신 없는 부분(페이지·개념) 있으면 명시

## 6. 추가 (리뷰 후)
- figure manifest 항목에 `"after": true`를 주면 같은 slot의 viz **뒤에** 배치됨 (기본은 viz 앞).
- 600px 이하에서 `viz-scroll`이 있는 viz는 머리줄에 "옆으로 밀어서 보기 →" 힌트가 자동으로 붙음. 그래도 핵심이 첫 화면(≈380px)에서 안 보이면 레이아웃을 세로로 쌓는 쪽을 고려.
- figure manifest 항목에 `"fold": true`를 주면 slot 끝의 "슬라이드 그림 더 보기 (N장)" 접힘 블록으로 들어감 (integrator 전용).
