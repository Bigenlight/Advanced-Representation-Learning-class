# 중간고사 복기 & 기말 전략 — 2026-10-29

> - 문제를 복원하는 문서가 아니야. 목표는 두 가지야.
>   - **시험 모양**(형식 · 난이도 · 출제 패턴)을 잡는 것
>   - **기말 전략**을 세우는 것
> - 기억은 파편뿐이라, 문항 내용은 "무슨 주제였고 핵심이 뭐였나"까지만 적었어.
> - 시험 직후 쓴 원본 메모는 [부록](#memo)에 그대로 있어.

## 0. 한 줄 결론

**개념은 교수님 말대로 쉬웠어 (basic concept & core idea). 어려웠던 건 "영어로 설명하기"였어.**
→ 기말도 같은 모양이라면, 공부할 **범위**보다 **답하는 방식**을 연습해야 해.

---

## 1. 시험 프로필

| 유형 | 문항 | 실제 모양 | 체감 |
|---|---|---|---|
| **T/F** | 10 | 문장 하나를 보고 T/F를 고른 뒤 **왜 그런지 설명**. F면 **뭐가 틀렸는지**까지 써야 함 | 개념은 쉬움. 영어로 쓰는 게 부담 |
| **객관식** | 5 | 개념 MCQ. "X의 목적·역할은?" 같은 유형 | 모의퀴즈와 비슷하고 살짝 어려움. 풀 만했음 |
| **빈칸 + 서술** | 5 | **단순화해서 잘라 낸 짧은 코드 조각**. 빈칸은 보통 **1개**이고, 한 줄 전체가 아니라 **한 요소**만 씀. 여기에 **이 과정이 뭔지 설명**을 붙임 | 알고리즘 박스가 아니라 코드라서 당황. 설명까지 써야 함 |

- 20문제 중 **15개에 영어 서술**이 붙어.
- 빈칸 자체는 한 요소라 가벼워. 무게는 **설명**에 있어.
- 표 수치 · 저자 · 학회를 묻는 문제는 기억나는 것 중엔 없었어.

**빈칸 문제 모양 예시** (실제 문제 아님, 모양만 이렇다는 뜻)

```python
with torch.no_grad():
    probs = torch.softmax(model(u_weak), dim=1)
    max_prob, pseudo_label = probs.max(dim=1)
    mask = (max_prob >= ____).float()
```
> Fill in the blank and explain what this code does.

- 빈칸: confidence threshold `tau`
- 설명: "FixMatch. Pseudo labels come from the weakly augmented view, only predictions with confidence ≥ τ are kept, and the model is trained to predict them on the strongly augmented view."

---

## 2. 기억난 문제 (주제와 핵심만)

✅ = 우리 자료(모의퀴즈 · 문제 pool)에 거의 그대로 있었음 · △ = 개념은 있었지만 그 형식(코드, "역할" 질문)은 처음

**T/F (10개 중 6개 기억)**

| 주제 | 답 | 핵심 한 줄 | 준비 |
|---|---|---|---|
| pretext task의 label은 사람이 단다 | **F** | label은 data에서 자동으로 생성됨, *"No manual annotation"* (L02 p.16–17) | ✅ |
| rotation label은 사람이 일일이 단다 | **F** | *"The applied rotation becomes the label"* (L02 p.41) | ✅ |
| FixMatch는 weak 먼저, strong 나중 | **T** (순서가 맞게 나왔다면) | weak로 pseudo label → threshold로 거르기 → strong 예측에 CE (L07 p.43) | ✅ |
| AllMatch 과정 설명 | 지문에 따라 다름 | 버리던 low-confidence 샘플도 candidate vs negative(BCC)로 활용 (L07 p.79–87) | △ 모의퀴즈엔 0문제 |
| BYOL과 negative pair | **F** ✔ | negative 없이 prediction + EMA target으로 학습 (L04 p.32) | ✅ 준비 덕에 바로 맞힘 |
| FreeMatch는 (threshold가) 고정 | **F** ("고정"이라 했다면) | 고정 τ는 FixMatch. FreeMatch는 self-adaptive (L07 p.67–71) | ✅ |

**객관식 (5개 중 2개 기억)**

| 주제 | 핵심 | 준비 |
|---|---|---|
| InfoNCE의 목적 | N개 후보 중 positive를 골라내는 categorical CE → x와 c의 **mutual information ↑** (L03 p.23–28) | ✅ |
| SimCLR g의 역할 | 아래 설명 참고 (L04 p.24–28) | △ "버린다"만 알고 "왜"는 몰랐음 |

> **g가 하는 일**
> 1. loss는 g를 거친 z = g(h)에 걸려. 그래서 z는 augmentation 차이(색, crop 등)를 무시하도록 정보를 지워.
> 2. 그 "지우는 일"을 g가 떠맡아. 그래서 g 앞의 h는 정보를 그대로 보존해.
> 3. 학습이 끝나면 정보가 많은 h를 쓰고 g는 버려. "버린다"는 결론은 이 역할에서 나온 결과야.

**빈칸 + 서술 (5개 중 4개 기억)**

| 주제 | 핵심 | 준비 |
|---|---|---|
| Rotation 증강 · 평가 | 4방향으로 돌리고 회전 index를 label로 써서 4-way CE. 평가는 encoder를 고정한 채 얕은 classifier를 붙여서 함 (L02 p.18, p.41) | △ 코드 연습 0 |
| InfoNCE positive score / denominator | 분자 = positive 점수, 분모 = positive + negatives, 둘 다 /τ (L03 p.20, L04 p.24) | △ |
| FixMatch | 위 예시와 같은 흐름 | △ |
| Mean Teacher teacher 갱신 | teacher = student weight의 EMA, backprop 없음 (L06 p.23–24) | △ |

- 코드 4개 중 3개(Rotation, FixMatch, Mean Teacher)는 **슬라이드에 코드가 없는 방법**이었어.
- 4개가 **계열마다 대표 방법 하나씩**이야: pretext · contrastive · consistency · pseudo label.

---

## 3. 이 파편으로 알 수 있는 것 / 없는 것

**알 수 있는 것 (꽤 확실)**
- **형식**: 위 1장 표. 직접 겪은 거라 확실해.
- **난이도의 정체**: 개념 깊이는 정의 · 핵심 아이디어 수준이야. 부담은 **쓰기**에서 와.
- **소재**: 방법을 정의하는 문장, pipeline 순서, 부품의 역할, 대표 방법의 코드 한 조각.
- **코드 고르는 기준**: 슬라이드 Algorithm 박스가 아니라 **계열 대표 방법**이야.

**알 수 없는 것**
- **강의별 비중**: 쉽게 푼 8문제는 기억에서 빠졌어. 그래서 기억난 쪽(L07 · L02)으로 치우쳐 있어.
- **채점 기준**: 서술을 얼마나 써야 만점인지, 키워드로 채점하는지 몰라.

---

## 4. 출제 패턴: 교수님이 보는 것

1. **label · target은 누가 만드나**: 사람인가 data인가, weak view의 예측이 pseudo label이 되나, teacher가 target이 되나
2. **방법을 정의하는 문장 뒤집기**: "BYOL은 negative를 안 쓴다", "FreeMatch는 adaptive", "pretext는 no manual annotation"
3. **pipeline 순서와 방향**: 무엇을 먼저 하나, 누가 target을 만들고 누가 gradient를 받나
4. **부품의 목적**: InfoNCE는 뭘 하려는 건가, g는 왜 있나
5. **대표 방법의 학습 코드 한 조각**: 한 요소를 채우고 과정을 설명

→ 공통 질문은 하나야: **"이 방법은 무엇을 위해 이렇게 생겼나"를 내 말로 설명할 수 있나?**

---

## 5. 기말 전략

> 기말은 **2026-12-10 (W15)**이야. 범위는 W10–W14: Weakly-Supervised I·II, Active Learning I·II, MAE. 누적 여부는 아직 몰라.
> 형식은 중간과 같다고 가정해.

### 5.1 공부 방식: "고르기"에서 "설명하기"로

방법마다 아래 네 가지를 **영어로** 만들어 둬. T/F 이유, MCQ, 코드 설명에 전부 재활용돼.

1. **3문장 요약**: 무슨 문제를 풀었나 → 핵심 아이디어 → 이전 방법과 뭐가 다른가
2. **정의 문장 1개 + 그걸 뒤집은 함정 1개**: 예) "BYOL learns without negative samples." ↔ "BYOL uses negative pairs."
3. **부품 역할**: 부품마다 "What is it for?"에 1문장으로 답하고, 빼면 어떻게 되는지도
4. **pipeline**: 3–5단계, 그리고 target을 만드는 쪽 / gradient를 받는 쪽

### 5.2 코드: 길게 외우지 말고 "조각 읽기"

- 계열 대표 방법마다 **4–8줄짜리 핵심 조각**만 봐. 슬라이드에 코드가 없는 방법도 대표 방법이면 포함해 (중간 땐 이걸 놓쳤어).
- 빈칸은 **한 요소**야. 단골 후보:
  - threshold `tau`, temperature `tau`
  - `no_grad` / `detach`
  - EMA 계수 `m` / `β`
  - label (`k`, `zeros`, `arange`)
  - `argmax`, `mean`, `sum`
  - weak / strong 중 어느 view
- 조각을 보면 **영어 2–3문장**으로 설명해. 순서는 이렇게:
  1. 어떤 입력(view)이 들어가나
  2. target은 어디서 오나
  3. 어떤 loss를 쓰나
  4. 누가 update되나
- 기말 후보 (추정, 강의를 듣고 확정):
  - Weakly-supervised: CAM 계산
  - Active Learning: acquisition score → top-K query 루프
  - MAE: random masking, **masked patch에만** loss

### 5.3 답안 템플릿

- **T/F True**: "True. [Method] [does X] because [mechanism]."
- **T/F False**: "False. [Method] does not [X]; instead, it [Y]." 고친 문장을 덧붙이면 더 안전해.
- **코드 설명**: "This code implements [step] of [method]. It [step 1], then [step 2]. The blank is [token], which [role]."

### 5.4 줄일 것

- 표 수치 · 저자 · 학회 · 연도
- 슬라이드 코드 박스를 글자 그대로 외우기, "이 코드는 어느 방법?" 식별 문제
- 학습 페이지 디자인 손보기 (그림 재배치 등). 그 시간을 쓰기 연습으로 돌려.
- 고르기 전용 T/F를 대량으로 만들기

### 5.5 자료 만들 때 (Claude 몫)

- **모의고사는 실제 비율과 모양으로**: T/F 10 (이유 서술) + MCQ 5 + 코드 조각 5 (빈칸 1개 + 설명).
- **정답마다 영어 모범 답안을 붙이기**: 규칙은 중간 때도 있었는데 안 지켰어.
- **채점**: 네가 영어로 쓴 답을 받아서 "핵심 키워드가 들어갔나"를 기준으로 봐.
- 모의고사를 대량으로 만들기보다 **적게 만들고 직접 써 보는 횟수**를 늘려.

### 5.6 확인할 것

- **마지막 수업에서 물어볼 것**: 기말 형식이 중간과 같은지, 누적인지, 서술은 어느 정도까지 써야 하는지.

---

<a id="memo"></a>
## 부록. 원본 메모 (시험 직후, 그대로)

<details>
<summary>펼치기</summary>

```text
문제 기억나는대로.

문제 총 20개 맞음.

T/F 10개

객관식 5개

빈칸 및 서술형 5개

# T/F

아니!!! 교수님 쉽게 내주신데메!! T/F인데 이게 왜 T인지 F인지 설명하고, F이면 뭐가 틀렸는지도 적으라는게 뭐야!! 이러면 T/F의 의미가 없잖아?! 그냥 서술형이라고 해!!

어떤 문제가 있었는지 생각나는대로 열거해줄게

- SSL에서 Pretext task의 라벨링은 인간이 하는거다.

- Rotation pretext task 할 때 사람이 일일히 label하는거다.

- Fixmatch 과정 설명이 맞는지 (weak하게 먼저하고 strong하게 하고)

- Allmatch 과정 설명, 이거 맞냐?

- BYOL negative pair으로 뭐시기 (너 덕분에 보자마자 F 갈김)

- Freematch는 학습이 고정되있다 뭐시기

# 객관식

괜찮았음. 너가 냈던 문제 보다 살짝 어려운 정도?

- 다음 중 InfoNCE의 목적은?

- 다음 중 SimCLR할 때 g의 역할은?

객관식으로 비교적 풀만했어서 기억이 잘 안나네. 너가 냈던 문제랑 비슷한 문제도 나왔던 것 같아. 위 2문제가 기억나는 것은 내 기억이 애매했어서 잘 기억이 낫어. (g는 버리는걸로만 알고 생각해보니까 그 역할이 뭐였는지 진지하게 생각안하고 있었음)

# 빈칸

알고리즘말고 코드 밖에 안나옴. 나는 알고리즘인 줄 알았는데..

그래서 실제로 학습의 핵심 부분이 코드 빈칸이 뚫려 있었음.

근데 중요한 것은 빈칸만 잇는게 아니라 이 과정이 뭔지도 설명해야함.

- Rotation 증강 및 평가 과정

- 뭐 positive score? demoniator? 뭐 이런거

- Fixmatch 과정

- Mean Teacher에서 teacher 학슴 과정
```

</details>
