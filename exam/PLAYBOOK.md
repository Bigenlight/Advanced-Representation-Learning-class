# 시험 공부 자료 만드는 요령 — 중간고사 경험 정리 (기말에 그대로 쓰기)

> **언제 읽나**: 기말 범위 강의가 끝나 갈 때, 또는 새 세션에서 "기말 준비하자" 할 때 이 파일부터 읽어.
> **한 줄 요약**: 처음 배우는 사람 눈높이로 쓰고, **질문 → 방법 → 차이** 순서로 묶어. 그림은 **설명 문단 바로 옆에 작게** 넣고, 문제는 **실제 시험 형식대로** 만들어.
> 규칙이 `exam/_flow/SPEC.md`·`EASY_SPEC.md`와 겹치면 **이 문서가 우선**이야. 그 둘에선 명령어와 파일 형식만 참고해.

중간고사(2026-10-29) 때 만든 결과물이 기준이야. 직접 열어 보면 감이 빨리 와. 단, 중간 결과물이 이 문서 규칙을 다 지킨 건 아니야 (2.3 참고).

| 결과물 | 무엇 |
|---|---|
| `notes/NN_topic.md` | 강의별 정식 노트. 가장 자세하고, 모든 자료의 근거야 |
| `exam/_parts/NN_*.md`, `pc_*.md` | 강의별 core idea + 문제 pool (T/F·MCQ·빈칸), 코드 계열별 pseudo code pool |
| `exam/core_ideas.md` | `_parts`의 A·B 섹션을 강의 순서로 모은 치트시트 |
| `exam/pseudocode.md` | 슬라이드 Algorithm·코드 박스 전수 정리 + 핵심 줄 |
| `exam/mock_quiz_N.md` | 실제 형식대로 짠 20문제 모의고사 + 답란 |
| `exam/flow.html` / `flow.pdf` | 범위 전체를 한 바퀴 도는 학습 페이지 (그림·그래프 포함) / 같은 내용 A4 PDF |

---

## 0. 기말 시작할 때

### 0.1 파일 정책 (먼저 정하고 시작)

- **범위부터 확인해.** 기말이 누적(L01–)인지, L08부터인지 공지로 확정해서 `CLAUDE.md`에 날짜와 같이 적어.
- **중간 파일은 덮어쓰지 마.**
  - 노트·pool: `notes/NN_*.md`, `exam/_parts/NN_*.md`를 새 강의 번호로 이어서 추가해.
  - 치트시트·pseudo code: 누적이면 기존 파일에 강의 블록을 추가하고, 아니면 `core_ideas_final.md`·`pseudocode_final.md`를 새로 만들어.
  - 모의고사: `mock_quiz_final_N.md`
  - 학습 페이지:
    - 누적이면 `base.html`에 `<section id="l08">`부터 추가해.
    - 아니면 `exam/_flow/`를 `exam/_flow_final/`로 복사하고, `build.py`·`pdf.py`의 출력 경로(`OUT`)를 `flow_final.html`·`flow_final.pdf`로 바꿔.
    - `--out` 옵션은 이미지가 절대경로로 들어가서 커밋용으로는 못 써.
- **손필기할 PDF는 레이아웃을 확정한 뒤 버전을 붙여 고정해** (예: `flow_final_v1.pdf`).
  - 중간고사 땐 패드로 PDF에 필기하는 중에 그림 배치를 바꾼 새 PDF가 나와서, 필기를 옮길 수 없었어.
  - 필기를 시작한 뒤에 내용이 바뀌면 PDF를 새로 뽑지 말고 **바뀐 페이지 목록**만 알려 줘. 배치가 크게 바뀌는 작업은 시험 뒤로 미뤄.

### 0.2 타임라인

> 일정(L01 p.4): W10 11/5 Weakly-Supervised I · W11 11/12 II · W12 11/19 Active Learning I · W13 11/26 II · W14 12/3 MAE · **W15 12/10 Final Quiz**.
> 마지막 강의가 **D-7**이야 (중간 때도 마지막 강의 10/22가 D-7이었어). 그래서 아래 D-14·D-10 작업은 실제론 **강의마다 그 주에** 하고, 마지막 1주엔 MAE 몫과 통합만 해.
> 중간고사가 끝나면 실제로 나온 유형(빈칸·pseudo code·표 수치 출제 여부)을 바로 `CLAUDE.md`에 적어. 기말 대비의 제일 좋은 근거야.

| 시점 | 할 일 |
|---|---|
| **범위 공지 직후** | 범위·누적 여부·형식(문항 수, 유형) 기록 → 0.1 파일 정책 결정 |
| **매 수업 직후** (그 주 안) | PDF 저장(`lectures/NN_topic.pdf`) → 페이지 렌더 → 청크 초안(`notes/_drafts/`) → 정식 노트 → README 진도 표 ☑. 모르는 용어 목록 적기. 교수님 시험 얘기는 즉시 기록. → 학습 페이지 단원 초안(`<section id="lNN">`, 카드 + 그림 slot 자리)도 이때 |
| **D-14** (실제론 강의마다) | core ideas 치트시트, 용어집, pseudo code 전수 스캔 |
| **D-10** (실제론 강의마다) | 강의별 문제 pool + 독립 검수 → 모의고사 |
| **D-7** | 학습 페이지의 큰 그림·용어집·공통 패턴·pseudo code·마지막 체크를 조립 → 독립 검수 → PDF **확정(버전 고정)** |
| **D-3** | 모의고사 풀기 → 틀린 문제는 해당 카드·노트로 돌아가 보강 |
| **D-1** | 마지막 체크 섹션, 함정 Top N, pseudo code 핵심 줄만 다시 |

---

## 1. 먼저 정하는 것: 독자와 시험

- **독자 = 나.** 주 분야는 RL · flow matching · simulation · VLA(action head 쪽)고, representation learning은 처음이야.
  - 막히는 지점은 "다들 알지?" 하고 지나가는 기본 용어야 (embedding, projection head, collapse, stop-gradient, teacher/student, pseudo label …).
  - 수업은 영어 논문 세미나 연속이라 키워드만 잡고 넘어간 부분이 많아.
- **시험 = 교수님 공지 그대로.**
  - 기말 형식(문항 수·유형·범위·누적 여부)은 새로 확인해서 적어.
  - 확인 전까지는 중간 기준을 가정해: 20문제, 영어 T/F · MCQ · 빈칸 + pseudo code, "please study basic concept & core idea of the papers", 쉬움.
  - 수업 중 교수님이 시험 얘기를 하면 **들은 날 바로** `CLAUDE.md`·`README.md`에 날짜와 함께 적어. 중간고사 땐 빈칸·pseudo code 공지를 마지막 수업에서야 들었어.
- 이 두 가지가 아래 모든 결정의 기준이야.
  - 무엇을 풀어 쓸지는 **내가 모르는 용어**가 정해.
  - 무엇을 남길지는 **시험 형식**이 정해.

---

## 2. 서술·설명 요령

### 2.1 톤

- **한국어 반말, 짧은 문장, 한 문장에 한 생각.** 문단이 4–5줄을 넘으면 쪼개거나 목록으로 나눠. 이모지는 안 써.
- **기술 용어·방법 이름은 영어 그대로.** 시험이 영어라서 지문에서 그 단어를 바로 알아봐야 해. 한국어 풀이는 처음 나올 때만 괄호로 붙여.
- **슬라이드 정의는 영어 원문으로 인용해.** T/F 지문은 대개 그 문장을 살짝 비틀어서 나와.
  - 예: "Provide more informative supervision than easy negatives" (L04 p.6)

### 2.2 용어는 처음 나오는 자리에서 푼다

- 처음 보는 단어는 **바로 옆 괄호나 짧은 문장**으로 풀어.
  - 예: `augmentation(같은 사진을 자르거나 색을 바꿔서 만든 변형본)`
- 용어집 링크만 달고 설명을 생략하면 안 돼.
- 다른 강의에서 이미 설명한 용어도 강의마다 한 번은 짧게 다시 상기시켜. 중간부터 읽을 수 있으니까.
- **본문 괄호 약어 풀이는 페이지에서 처음 나올 때 한 번만** 해.
  - 형식: `KL(Kullback–Leibler)`, `TD(Temporal Difference)`
  - 약자 글자는 굵게: **B**ootstrap **Y**our **O**wn **L**atent
- **카드에는 카드마다** 약자 풀이와 출판 논문 제목을 이름 밑에 작게 달아.
- **확인 안 된 풀이는 지어내지 마.**
  - 예: X-CLR은 논문 문장("called X-Sample Contrastive (X-CLR)")대로 "X-Sample Contrastive의 줄임"까지만 적었어. CLR을 글자별로 풀지 않았어.
  - 예: InfoNCE의 "Info"는 논문이 뜻을 정의하지 않았다고 적었어.
- 카드의 논문은 **슬라이드가 cite한 논문** 기준이야 (예: SimCLRv2 카드 → "Big Self-Supervised Models are Strong Semi-Supervised Learners"). 제목은 학회판 표기로.
- 괄호 풀이는 `code`·수식·퀴즈 문제 문장·카드 이름 안에는 넣지 마. 같은 논문 카드가 두 번 나오면 메인 카드 하나에만 달아.

### 2.3 방법(논문) 하나는 같은 틀로 쓴다

라벨 순서를 정해 두고, 방법마다 필요한 것만 골라 써.

| 라벨 | 내용 |
|---|---|
| **한 줄** | 이 방법이 하는 일을 일상어로 |
| **문제** | 이전 방법의 어떤 불편함 때문에 나왔나 |
| **핵심** | 아이디어를 쉬운 말로 |
| **어떻게** | 단계가 있으면 1, 2, 3 번호 목록으로 (Algorithm 순서를 따라) |
| **왜 돼?** | 직관 |
| **차이** | 이전/짝꿍 방법과 다른 점. 시험의 "A vs B"가 여기서 나와 |
| **예시로 보면** | 작은 숫자 toy 예시 (선택) |

- 교수님이 말한 "basic concept & core idea"는 결국 **한 줄 · 문제 · 핵심 · 차이** 네 칸이야.
  - `core_ideas.md`(Problem / Core idea / vs. previous / 한 줄)에선 모든 방법에 빠짐없이 채워.
  - 학습 페이지 카드도 가능하면 네 라벨을 두고, 최소한 `한 줄`과 `차이`는 꼭 둬. 중간 페이지는 카드 63개 중 네 라벨이 다 있는 게 4개뿐이었어 (예: FixMatch 카드엔 `한 줄`·`차이`가 없음).
  - 공통 비교표가 있어도 카드의 `차이`는 한 줄이라도 써.

**Before → After** (L04 Debiased Contrastive):

```
before: random으로 뽑은 negative에 같은 class(false negative)가 섞여 성능이 크게 떨어져.

after:  한 줄  "negative라고 뽑았는데 사실 같은 종류"인 샘플 때문에 생기는 손해를 label 없이 보정하는 방법.
        문제  contrastive learning은 label이 없으니까 negative(멀어져야 할 샘플)를 batch에서 그냥 아무거나 뽑아.
              anchor가 강아지 사진이면 negative 중에도 강아지 사진이 섞일 수 있어. 이게 false negative야. (p.7–8)
        핵심  어느 게 false negative인지는 몰라. 대신 "아무거나 뽑으면 η⁺ 비율로 같은 class가 섞인다"는 사실로
              섞인 분포에서 positive 몫을 빼서 진짜 negative 분포를 추정해: p⁻ = (p − η⁺p⁺) / η⁻
              + 기호마다 한 줄씩 풀이
```

### 2.4 수식

- **수식은 지우지 마.** 시험에 그대로 나와.
- **기호 하나하나를 말로 풀어.** 무엇을 키우고 무엇을 줄이는지, 값이 커지거나 작아지면 어떻게 되는지까지.
  - 예: SimCLR $\ell(i,j)$
    - 분자 = anchor와 positive의 유사도라서 클수록 loss가 줄어.
    - 분모 = 자기 자신만 뺀 2N−1개 후보야.
    - τ가 작을수록 가까운(hard) negative에 더 반응해.
    - 결국 "2N−1개 중 진짜 짝 맞히기" 분류의 cross-entropy야.
- **유도는 한 줄씩.** 슬라이드에 없는 유도는 "notes 보충, 슬라이드엔 결과 식만"이라고 표시해.
- 표기 혼동이 시험 함정이 돼. 같은 글자가 다른 뜻이면 그 자리에서 짚어.
  - 예: τ = temperature / EMA 계수 / threshold

### 2.5 예시와 비유

- **작은 숫자 toy 예시**를 써.
  - 예: "사진 4장(N = 4) → view 8개 → anchor 하나에 positive 1, negative 6, softmax 후보 7"
  - 예시 숫자는 "설명용"이라고 밝혀. **슬라이드 실험 수치처럼 보이면 안 돼.**
- **RL 비유**는 내가 아는 개념이라 제일 빨리 꽂혀. 비유가 오해를 부르지 않는지는 꼭 따져.
  - 아래 표는 중간 범위 예시야. 기말 범위 짝은 새로 만들어.

| 이 수업 | RL 쪽 짝 |
|---|---|
| EMA teacher / momentum encoder | target network soft update (단, soft update 계수 τ는 작고 BYOL의 τ는 ≈1이라 식에서 방향이 반대야) |
| stop-gradient | semi-gradient TD target (target 쪽으로 gradient 안 흘림) |
| confirmation bias (자기 pseudo label로 오차 누적) | bootstrapping error 누적 |
| DivideMix 두 network가 서로의 분할을 씀 | Double DQN · TD3 twin Q — "다른 estimator로 내 target을 견제"하는 동기만 같아 (TD3는 두 critic의 min을 써서 메커니즘은 달라) |
| FixMatch weak→strong (target은 안정적인 입력에서) | target을 덜 흔들리게 두는 동기 (단, FixMatch엔 EMA teacher가 없어. 같은 network에 weak 입력을 넣어 target을 만들어) |
| FreeMatch 올라가는 threshold | curriculum |

- 슬라이드 밖 연결고리(VLA가 vision encoder로 DINOv2·SigLIP를 쓴다는 것, R3M·OpenVLA·π0 같은 모델)는 동기 부여용으로 `>` 보충에 넣고, **"시험 범위 아님"**을 꼭 붙여.
  - 단 슬라이드에 있는 방법 자체(CURL L05 p.20–21, DINOv2 L04 p.55–67)는 **시험 범위**야. 연결 블록에 섞이면 그 줄만 "슬라이드 내용, 시험 범위"라고 따로 표시해.

### 2.6 정확성 (쉽게 쓰다 틀리면 최악)

- **근거 순서**: 슬라이드 원문(`pdftotext -f N -l N -layout lectures/NN_*.pdf -`) → `notes/` → 웹(arXiv, CVF, OpenReview).
  - 웹 검색엔 **논문 이름·약어만** 넣어. 강의 PDF 내용은 붙여 넣지 마.
- 슬라이드에 없는 **새 사실(수치, 논문 디테일)은 추가하지 마.** 기초 ML 정의(softmax가 뭔지 등)와 비유는 괜찮아.
- **과잉 일반화 주의.** 대표 사례가 "teacher는 학습 안 한다"야. 틀렸어.
  - BYOL·DINO·Mean Teacher의 teacher(MoCo key encoder도)는 gradient는 안 받지만 EMA로 계속 갱신돼.
  - 그렇다고 "teacher = EMA"로 뒤집어도 틀려. Meta Pseudo Labels의 teacher는 student 성능으로 학습되고(L07 p.24–25), Noisy Student의 teacher는 앞 단계에서 학습한 모델이야(L07 p.17).
- **함정 짝은 쓸 때마다 다시 확인해.** 아래는 중간 범위 예시고, 기말 범위 짝은 새로 모아.
  - student ↔ teacher
  - prediction EMA ↔ weight EMA
  - MSE ↔ KL ↔ CE
  - τ의 세 의미
  - centering은 teacher 쪽만, stop-gradient가 붙는 쪽
  - hard ↔ super-hard
  - Π-model ↔ Temporal Ensembling ↔ Mean Teacher
- **슬라이드 오타·의심**은 원문을 유지하고 옆에 `← ` 플래그를 달아.
  - 중간고사 땐 슬라이드 cite의 학회·연도가 실제와 다른 경우가 여럿 있었어 (예: FixMatch는 실제 NeurIPS 2020, DivideMix는 실제 ICLR 2020).
  - 웹으로 확인해서 실제 학회를 적되, 슬라이드 표기는 `← slide엔 …`으로 남겨. 시험 지문은 슬라이드 표기를 따를 수 있어.

---

## 3. 정리 구조 (어떻게 묶나)

### 3.1 범위 전체를 꿰는 질문 하나

- 중간고사 범위(L01–L07)는 **"학습 target은 어디서 오나?"** 한 질문으로 꿰었어.
  - L02: data를 망가뜨려서 → L03–L05: 같은 사진의 다른 view에서 → L06: 흔든 입력의 예측에서 → L07: 모델 자신의 pseudo label에서
- 맨 앞 "큰 그림"에 이 질문의 답 목록 표와 강의별 한 줄 요약을 둬.
- 기말도 **범위를 꿸 질문 하나를 먼저 정하고** 시작해. 강의끼리 연결이 보이면 외울 양이 줄어.

### 3.2 위계: 단원 → 파트 → 카드

| 층 | 무엇 | 모양 |
|---|---|---|
| **단원** | 강의 하나 | 큰 번호(L04) + 제목 + **질문 한 줄**(이 강의가 던지는 질문) + 도입 2문단 |
| **파트** | 슬라이드 목차(보통 p.2–3)의 큰 덩어리 | 파란 네모 번호 배지 + 굵은 제목 + 밑줄 (번호는 CSS가 자동으로 붙여, 손으로 쓰지 마) |
| **카드** | 논문·방법·개념 하나 | `파트-번호` 배지(2-3) + 이름 + 저자·학회·`p.N` + 약어 풀이 + 논문 제목 |

- **파트는 슬라이드 목차를 따라.** 목차 밖에 붙은 세미나 논문들(보통 강의 끝)은 "넓히기: …" 파트로 따로 묶어.
- **파트 제목과 카드 이름이 한눈에 구분돼야 해.** 처음엔 글자 크기만 조금 달라서 헷갈렸어 → 번호 배지 + 밑줄로 고쳤어.
- **첫 파트 제목보다 카드가 먼저 나오면 안 돼.** 모든 카드는 어떤 파트에 속해야 해.
- 파트 하나에 카드가 6–7개를 넘으면 쪼개.
- **계보(진화 순서)**는 카드 대신 timeline 목록으로 써.
  - 예: Contrastive loss → Triplet → Lifted → N-pair → NCE → InfoNCE
- **여러 방법 비교는 표 하나로** 해. 행 = 비교 축, 열 = 방법.
  - 예: SimCLR/BYOL/Barlow Twins/DINO × negative 유무, 두 branch, loss, collapse 방지, 한 줄

### 3.3 단원 끝에 늘 같은 블록 (이 순서)

1. **RL로 번역**: 2.5의 비유를 그 강의 내용으로.
2. **시험 함정**: 그 강의에서 헷갈리는 짝·문장 목록.
3. **다음으로**: 다음 강의가 왜 필요한지 한 줄. 강의 사이 다리야. 마지막 강의에선 생략해.
4. **셀프 체크 3문제**: 영어 T/F · MCQ · 빈칸 각 1개, 접어 둔 정답 + 근거 `p.N` (중간 L02·L07은 MCQ가 빠졌어).
5. **다 봤음** 체크박스

### 3.4 페이지 앞뒤

- **앞**:
  - 읽는 법: 박스 종류, 번호 뜻, `p.N` 뜻, 학회 약어
  - 큰 그림
  - 용어 먼저: 기초 · 모델 부품 · 학습·평가 그룹에, 범위의 방법 계열별 그룹을 더해 (중간 예: Contrastive · Teacher–student · Semi-supervised).
    - 용어 항목 = 뜻 → 예 → 왜 필요한지 + RL 짝 한 줄. 이름 옆에 슬라이드 `Lxx p.N`을 달아.
    - 각 용어에 `t-<영어-slug>` id를 달아. 여러 명이 쓸 땐 용어집 담당이 id 목록을 먼저 확정하고, 나머지는 그 id로만 링크해.
- **뒤**:
  - 공통 패턴: EMA·stop-gradient·loss를 "누가 뭘 쓰나" 표로
  - pseudo code 모음
  - Physical AI 연결 (시험 범위 아님)
  - 마지막 체크 (D-1)

---

## 4. 양식 (파일·표기 규칙)

### 4.1 공통 표기

- 페이지는 **PDF 페이지 번호** `p.N`으로 써. 다른 강의를 가리키면 `L04 p.25`.
- 핵심은 `**굵게**`, 부연은 `← `, 슬라이드 밖 보충은 `>` 인용 블록.
- 수식: md에선 LaTeX(`$…$`, `$$…$$`), 학습 페이지에선 `<span class="math">` + `<sub>`/`<sup>` (MathJax 없음).
- 용어·정의는 슬라이드 원문 영어를 살려.

### 4.2 노트 `notes/NN_topic.md`

- 파일 이름 번호는 PDF 번호와 같게 해.
- **맨 위**:
  - `>` 한 줄 요약
  - 출처 (교수, PDF 페이지 수). 원 논문 학회 목록을 `> 보충: 원 논문 — …`으로 붙이면 좋아 (노트 05–07 방식).
  - **"이 강의의 큰 그림"** ASCII 흐름도 (슬라이드 목차 → 방법들 → 다음)
  - **핵심 질문** 목록
- **본문 섹션 번호 = 슬라이드 목차.**
  - 섹션 제목: `### 3.2 BYOL: Bootstrap Your Own Latent (p.30–34) — Grill et al., NeurIPS 2020`
  - 순서: 개념 → 정의/수식 → 직관 → 예시/실험 → 비교. 슬라이드를 그대로 옮기지 말고 재구성해.
- **끝**: 본문 마지막 섹션에 방법 비교 표 → `## 시험 대비 핵심 정리` (꼭 외울 수식 / 꼭 외울 정의 / 비교 포인트 / 예상 질문 + 모범답안 방향 / 필요하면 숫자 체크리스트)
- 그림은 `assets/lectureNN/`에 넣고 `<img src="../assets/lectureNN/…" width="600">`으로 써.

### 4.3 core idea · 문제 pool `exam/_parts/NN_*.md`

**섹션 네 개**: A. Core ideas / B. Must-know definitions & distinctions / C. Question pool (T/F·MCQ, `### Q1 [T/F]`) / D. Fill-in-the-blank pool (`### F1 [Blank]`)

- `core_ideas.md`는 각 파일의 A·B만 강의 순서로 모은 거야.
- A·B 본문은 영어 문장으로 쓰고, `한 줄`만 한국어로 써. B는 표가 아니라 `- **용어**: 정의` 목록이야.
- 분량은 강의당 C 20개 안팎 + D 10개 안팎 (중간 실적: 강의당 약 21 + 13).
- `pc_*.md`(pseudo code pool)는 강의별이 아니라 코드 계열별로 묶어 (InfoNCE / distill / consistency / pseudo label, 중간 59문항). 섹션은 A. Annotated pseudo code / B. Cross-method comparison / C. Question pool.

**A 블록** (강의 순서, 방법 하나에 블록 하나):

```markdown
### SimCLR (Chen et al., ICML 2020, p.23–29)
- **Problem**: …
- **Core idea**: …
- **vs. previous / key difference**: …
- **Keywords**: …
- 한 줄: (한국어)
```

**C·D 문항**:

```markdown
### Q1 [T/F]
Hard negatives are … but they provide less informative supervision than easy negatives.
**Answer**: False — hard negative는 … **더** informative ("Provide more informative supervision than easy negatives"). (p.6)
```

- 지문은 영어로, 정답 해설은 한국어로 쓰고 **슬라이드 원문 인용 + `p.N`**을 붙여.
- **함정 유형**:
  - 슬라이드 문장을 한 단어만 비틀기
  - 짝 뒤바꾸기 (student↔teacher, MSE↔KL)
  - 방법 이름 ↔ 아이디어 매칭
  - "Which is NOT …"
  - 표 수치는 비중 낮게
- **빈칸**은 철자까지 써야 하는 핵심 단어나 식을 비워. 정답엔 허용 답도 같이 적어.
  - 예: `irrelevant (accept: task-irrelevant, nuisance, redundant)`
- pool은 **작성자와 다른 에이전트가 독립 검수**해. 정답 오류가 제일 치명적이야.

### 4.4 모의고사 `exam/mock_quiz_N.md`

- **회차 구성은 최신 공지의 유형 비율 그대로** 해.
  - 중간 4–5회 예: 20문제 = T/F 6 + MCQ 7 + 빈칸 7, 그중 pseudo code 4
- 문항마다 `[L04]` 출처 태그(pseudo code면 `[L04·PC]`)와 `>` 답란을 달고, 정답은 맨 아래에 모아. (중간 땐 답란을 1회에만 나중에 추가했어. 처음부터 다 달아.)
- 정답 해설 끝에 출처 문항을 달아 (`← pool: 04_Contrastive_Learning_II Q1`).
- pool에서 뽑되 앞 회차와 겹치지 않게 해.
- 푼 답은 `>` 답란에 적어.
  - 헷갈리면 `False?`처럼 `?`를 붙여. 맞았어도 `?` 붙은 문제는 다시 봐.
  - 틀린 문제는 해당 노트·카드로 돌아가.

### 4.5 pseudo code `exam/pseudocode.md`

- 슬라이드를 **전수 스캔**해서 Algorithm·코드 박스 목록을 확정해 (중간고사는 508p에서 9개).
  - 코드는 가족별로 묶어 (중간: PC1 InfoNCE / PC2 self-distillation / PC3 consistency / PC4 pseudo label).
  - 준-알고리즘(BYOL 단계·DeepCluster process)은 같은 가족 본문에 넣고, 나머지 번호·Step 절차는 부록 표로.
- 맨 앞에 **"한눈에 보기" 표**를 둬: 방법(강의 p.) · 형태 · **꼭 기억할 핵심 줄** · core idea 한 줄.
- 코드마다 표시할 것:
  - core idea가 드러나는 줄: `detach`/`no_grad`, momentum·EMA update, loss 식, `labels`, centering, bias correction …
  - 그 줄의 역할
  - 예상 빈칸 (`k = k.______()`)
- 공통 패턴(EMA · stop-gradient · labels · loss)과 함정 Top N을 따로 모아.

### 4.6 학습 페이지 뼈대 (`exam/_flow/base.html`)

단원 하나는 이 모양이야. 클래스 이름은 그대로 써야 스타일과 자동 번호가 붙어.

```html
<section id="l08">
  <div class="ch-head"><span class="ch-num">L08</span><h2>제목</h2><p class="ch-meta">NNp</p></div>
  <p class="ch-q"><span class="k">질문</span>이 강의가 던지는 질문 한 줄</p>
  <p>도입 문단 …</p>
  <h3>파트 제목 (슬라이드 목차)</h3>
  <div class="methods">
    <div class="m"><div><span class="m-name">BYOL</span><span class="m-src">Grill 2020 · p.30–34</span><small class="m-full"><span class="ab"><b>B</b>ootstrap <b>Y</b>our <b>O</b>wn <b>L</b>atent</span><span class="pt">논문 제목 (NeurIPS 2020)</span></small></div><div class="m-body">
      <p><span class="k">한 줄</span>…</p>
      <p><span class="k">문제</span>…</p>
      <p><span class="k">핵심</span>… <span class="math">ξ ← τξ + (1−τ)θ</span></p>
      <ol><li>어떻게 1</li><li>…</li></ol>
      <!--slot:l08-in01-->          ← 이 자리에 그림이 들어가 (5.2)
      <p><span class="k">차이</span>…</p>
    </div></div>
  </div>
  <div class="note rl"><span class="tag">RL로 번역</span><ul>…</ul></div>
  <div class="note trap"><span class="tag">시험 함정</span><ul>…</ul></div>
  <div class="note next"><span class="tag">다음으로</span><p>…</p></div>
  <div class="quiz"><p class="quiz-h">셀프 체크</p>
    <details class="q"><summary><span class="typ">T/F</span><span class="qs en">English statement.</span></summary><p class="ans"><b>False.</b> 한국어 해설 (p.N).</p></details>
  </div>
  <label class="done"><input type="checkbox" id="done-l08" data-done="l08">다 봤음</label>
</section>
```

- 용어집 항목: `<dt id="t-ema">EMA<small>L04 p.30</small></dt><dd>뜻 … <p><span class="k">예</span>…</p></dd>`
- 본문 약어: `TD<span class="fn">(<b>T</b>emporal <b>D</b>ifference)</span>`
- 직관·예시를 본문과 나누고 싶으면 `<div class="note easy"><span class="tag">예시로 보면</span><p>…</p></div>`를 써. 카드 하나에 많아야 1개야.

---

## 5. 시각자료

### 5.1 무엇을 넣나

- **우선순위**: ① 방법의 구조/파이프라인 figure ② 핵심 개념 도식 ③ 직관을 주는 예시 그림
- 결과표·수치표는 거의 안 넣어 (시험 비중이 낮아). 방법 하나에 0–2장이면 충분해.
- **슬라이드 crop**:
  - 해당 페이지를 `pdftoppm -r 150`으로 렌더하고, 그림 영역만 잘라 (제목줄·페이지 번호·로고는 빼).
  - 자른 뒤엔 **반드시 이미지를 직접 열어** 잘림·흐림을 확인해.
  - 파일 이름은 `assets/lectureNN/pNNN_<slug>.png`. 기존 asset은 덮어쓰지 말고 재사용을 먼저 해. 렌더 중간 파일은 리포 밖(scratchpad)에 둬.

### 5.2 어디에 넣나 — "설명하는 문단이 끝나는 바로 그 자리"

중간고사 땐 처음에 그림을 카드·표·목록 **뒤에 몰아서** 넣었다가, 나중에 그림·그래프 85개(슬라이드 그림 57 + 개념 그래프 28)를 전부 옮겼어. 기말엔 처음부터 이렇게 해.

- 그 그림이 보여 주는 내용을 설명하는 문단이 **끝난 직후**, **그 카드 안**에 넣어.
  - 구조 그림이면 보통 "어떻게" 목록 바로 뒤야. 목록 중간엔 끼우지 마.
- 계보 단계를 그린 그림은 **그 단계 안**에, 용어 그래프는 **그 용어 설명 안**에 넣어.
- **여러 방법을 비교하는 그림·그래프**만 비교 표 바로 뒤에 둬.
- **한 자리에 3개 넘게 쌓지 마.**
  - 같은 내용의 슬라이드 그림과 개념 그래프가 연달아 나오면 다른 문단으로 나눠 (예: Invariance vs Equivariance).
- 본문에 "위 그림", "아래 그래프" 같은 위치 표현은 쓰지 마. 그림을 가리키는 말은 캡션 안에서만 해.

### 5.3 크기 — 보이는 높이 기준

"좀 크다"는 피드백으로 정한 기준이야. 화면에 보이는 높이(= 폭 ÷ 가로세로비)가 **170–300px** 정도가 되게 해.

| 그림 모양 | 가로세로비 | 폭 (px) |
|---|---|---|
| 아주 넓음 | ≥ 2.5 | 440–520 |
| 중간 | 1.6–2.5 | 380–480 |
| 정사각형 근처 | 0.9–1.6 | 260–360 |
| 세로로 긺 | < 0.9 | 200–260 |

- **표가 기준이고, 170–300px는 대략적인 목표야.**
  - 띠 모양(≥ 3.5)은 100–150px, 세로로 긴 그림은 400px 근처까지 나와도 돼.
  - 그럴 땐 핵심 글자가 읽히는지 꼭 확인해. 안 읽히면 crop을 좁히거나 두 장으로 잘라.
- 카드 안은 약 500px까지, 페이지 기본 상한은 520px. 누르면 원본 크기로 확대돼 (lightbox).
- **너무 줄이면 핵심 글자가 안 읽혀.** 중간고사 땐 이 이유로 3장을 다시 키웠어 (Modern setup, FixMatch, consistency vs pseudo label).
- **접기(fold)는 보조 그림에만 써.** 핵심 그림을 접어 두면 놓쳐 (DINO attention 사례).

### 5.4 캡션 — "이 그림에서 뭘 봐야 하나"

- 처음 보는 사람 기준으로 써. 그림 내용을 나열하지 말고 **읽는 법**을 이 순서로 알려 줘.
  1. 읽는 방향 ("아래에서 위로 읽어", "위 줄이 online, 아래 줄이 target")
  2. 색·기호·화살표의 뜻 (`//` = stop-gradient, 점선 = EMA, 빨간 X = 버린 샘플)
  3. 핵심 한 줄 ("loss는 h가 아니라 z에서 걸리고, 끝나면 g는 버려")
  4. 근거 `L04 p.30`
- 보통 1–4문장, 구조 그림은 6문장까지 괜찮아. 길이보다 위 순서가 중요해.
- 슬라이드 그림 속 표기가 헷갈리면 캡션에서 짚어 ("여기서 τ는 temperature가 아니라 EMA 계수야").

### 5.5 개념 그래프 (직접 그린 그림·인터랙티브)

- **언제 그리나**:
  - 슬라이드 그림으로는 안 보이는 **변화**를 보여 줄 때. 예: τ를 바꾸면 softmax가 얼마나 뾰족해지나, momentum 값에 따른 EMA, ramp-up w(t), threshold가 올라가는 모양
  - 여러 방법을 **같은 그림 문법으로 비교**할 때
- **가짜 수치 금지.** 설명용 데이터로 그렸으면 `illustrative` 표시를 하고 캡션에 "개념 그림"이라고 적어.
- **그림 문법을 전 강의에서 통일해.**
  - 데이터: 작은 회색 박스
  - **gradient로 학습되는 network**: 파랑 실선 박스 (`.box-train`)
  - **EMA로만 갱신되는 network**: 회색 점선 박스 + "EMA" 점선 화살표
  - **stop-gradient**: 화살표 위 `//` + `sg`
  - loss: 둥근 박스 안에 이름 (InfoNCE, MSE, CE)
- **색은 의미로 써**: 핵심·학습되는 쪽 = 파랑(`--accent`), loss·주의 = 주황(`--trap`), RL 비유 = 보라(`--rl`), 정답·positive = 초록(`--ok`), 보조 = 회색(`--muted`).
  - 색은 **CSS 변수로만** 써. `fill="#123"`·`red` 같은 리터럴은 금지야. 그래야 라이트/다크가 같이 바뀌어.
- 그래프 조각 형식: JS는 IIFE 하나, CSS는 `#viz-…` 범위 안에서만, id 접두어 통일, 외부 라이브러리 금지. viewBox 폭 560–720, 글자 13px 이상, 파일 15KB 안팎. 자세한 형식은 `exam/_flow/SPEC.md` 3-B.
- 슬라이더는 "값을 바꿔 보면 감이 오는 것"에만 달아. 슬라이더를 안 건드려도 초기 상태 그대로 그림이 다 보여야 해.
- 너무 긴 그래프(1000px 이상)는 폭 상한을 줘.

### 5.6 확인

- 데스크톱(1280px), 폰(390px), 다크 모드 세 가지로 봐.
  - 그림만 보지 말고 **바로 위 문단과 같이** 찍어서 "이 문단이 이 그림을 설명하나"를 확인해.
  - 명령 예:
    1. `python3 -I exam/_flow/build.py --chapter l08`로 미리보기를 만들어.
    2. `python3 -I exam/_flow/shot.py exam/_flow/preview/chapter_l08_dark.html <scratchpad>/l08_dark.png --width 390 --height 20000`으로 찍어.
    3. `console: clean`을 확인하고 PNG를 직접 열어 봐.
- 폰에서 페이지 전체가 옆으로 밀리면 안 돼. 긴 수식이나 넓은 그래프가 주범이야.
- PDF판은 접힌 그림·정답을 전부 펼쳐서 만들어 (`python3 -I exam/_flow/pdf.py`).
  - **그림이 글자를 덮는지 검사**해. Chrome 인쇄 버그로 실제로 한 번 겹쳤어. 카드 본문의 grid 배치를 인쇄 때 block으로 바꿔서 고쳤어 (`pdf.py`).
  - 검사 방법: `pdfplumber`로 페이지마다 이미지 bbox와 단어 bbox가 30% 넘게 겹치는 곳을 찾고, 걸린 페이지는 렌더해서 눈으로 봐.

---

## 6. 중간고사 때 시행착오 → 기말엔 처음부터

| 겪은 문제 | 고친 방법 | 기말엔 |
|---|---|---|
| "아는 사람이 복습하는" 압축 톤이라 못 따라감 | 처음 배우는 사람 눈높이로 전면 재작성 (작성 11 + 검수 9) | **처음부터 쉬운 톤** (2장) |
| claude.ai 페이지엔 그림이 없어서 아쉬움 | 로컬 HTML에 슬라이드 그림 + 개념 그래프 | 처음부터 로컬 HTML |
| 방법·논문 약어가 뭔지 몰라 막힘 | 약어 풀네임 + 카드 밑 논문 제목 | 첫 작성 때 같이 (2.2) |
| 그림이 표·목록 뒤에 몰리고 큼 | 그림·그래프 85개를 설명 문단 옆으로 옮기고 크기 규칙 적용 | 처음부터 5.2–5.3대로 |
| 파트와 카드가 구분 안 됨 | 파트 번호 배지 + 밑줄, 카드 `파트-번호` | 처음부터 3.2대로 |
| 빈칸·pseudo code 공지를 늦게 들음 | pool·모의고사 새 형식 추가 | 공지는 그날 기록, 형식이 바뀌면 바로 반영 |
| 패드 손필기 중에 PDF 배치가 바뀌어 필기를 못 옮김 | (못 고침) 본문은 거의 같아서 옛 PDF로 계속 공부 | 필기용 PDF는 확정 후 버전 고정 (0.1) |
| PDF에서 그림이 글자를 덮음 | 인쇄용 CSS 수정 + 겹침 검사 (5.6) | PDF 만들 때마다 검사 |
| 슬라이드 학회 표기 오류 | 웹으로 확인해 실제 학회를 적고, 슬라이드 표기는 `← slide엔 …`으로 남김 | 인용 정보는 항상 확인 (2.6) |

---

## 부록. 멀티에이전트로 만들 때

- **나누는 단위는 강의(단원)야.**
  - 순서: 작성(fan-out) → **작성자와 다른 에이전트가 독립 검수** → 내가 통합
  - 글쓰기·검수는 Opus, 기계적 배치(그림 자리·크기)와 넓은 조사는 Sonnet이 잘 맞았어.
- **작성자는 원본을 직접 고치지 않아.** 자기 몫의 조각 파일이나 계획(plan) 파일만 써.
  - 통합할 땐 스크립트로 검증해: 추가만 했나, 자리 문자열이 유일한가, 넣으면 안 되는 위치는 아닌가.
  - 동시에 여러 명이 같은 파일을 고치면 꼬여.
- **검수자에게 받을 것**: 심각도, 근거(`p.N`), **바로 반영할 수 있는 수정안**(대체 문장, 자리 문자열, 숫자). 그리고 "취향 차이로 부풀리지 말 것".
- **학습 페이지 조립 방식** (꼭 알고 시작):
  - 본문 원본은 `exam/_flow/base.html` 하나야. `exam/flow.html`은 `python3 -I exam/_flow/build.py`의 빌드 결과라 직접 고치면 다음 빌드 때 덮여. `base_src.html`은 옛 artifact용 사본이라 안 고쳐.
  - 그림·그래프는 본문에 직접 넣지 않아.
    - 넣을 자리에 `<!--slot:l08-in01-->`을 두고, `figs/l08.json`(`slot, src, alt, caption, page, width`, 선택 `fold`)과 `viz/l08.json`(`slot, file`)에 적으면 `build.py`가 채워.
    - 기존 slot id는 바꾸지 마 (`make_slots.py` 재실행 금지).
  - 파트·카드 번호는 CSS counter가 자동으로 붙여. 손으로 쓰지 마. `<section>` 바로 밑 `<h3>`가 파트, `.methods>.m`과 `.steps>li`가 카드야.
  - **새 강의를 넣을 때 손볼 곳 3개**:
    1. `base.html` CSS의 `:is(#l01,…,#l07)` 목록. 여기에 추가 안 하면 카드 번호가 안 붙어.
    2. 상단 nav(rail) 목차
    3. `build.py`의 `CODE_SLIDES` (pseudo code 슬라이드 목록)
- **도구 위치**:
  - 스킬: `git-pull-push`(커밋·푸시), `md-img-resize`(노트 이미지 폭), `recall-quiz`(노트로 복습 퀴즈. README 진도 표 "복습 퀴즈" 열이 아직 전부 ☐)
  - PDF 페이지 렌더: `lectures/_pages/` (100dpi png + txt, gitignore)
  - `exam/_flow/`: `build.py`(조립), `pdf.py`(PDF), `shot.py`(스크린샷)
  - `exam/_flow/work/{easy,abbr,inline}/`: 쉬운 톤 재작성 / 약어 풀네임 / 그림 배치 도구와 규칙
    - **gitignore라 이 컴퓨터에만 있어.** 기말 전에 `easy.py`·`abbr.py`·`inline.py`와 각 SPEC을 `exam/_flow/tools/`로 옮겨 커밋해 둬.
