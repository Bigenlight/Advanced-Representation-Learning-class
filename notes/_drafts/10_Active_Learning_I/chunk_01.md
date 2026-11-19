<!-- DRAFT | lecture: 10_Active_Learning_I | chunk: 01 | pages 1–19 -->
<!-- VERIFIED | fixes: 10 | verifier: sonnet ×2 (p.1–13, p.14–19) -->

## Title (p.1)

슬라이드 텍스트:
- Advanced Representation Learning (고급표현학습)
- Instructor: Donghyeon Cho
- The Department of Computer Science
- Hanyang University

**설명**
- 강의 10 표지야. 주제는 **Active Learning I**이야.
- 기말 범위를 꿰는 질문 "annotation을 덜 쓰면서 모자란 supervision을 어떻게 메우나?"에 대한 세 번째 답이야.
  - L06–L07 (semi-supervised): label 없는 data엔 **모델이 pseudo label**을 붙여.
  - L08–L09 (weakly-supervised): label을 **더 싸고 약한 모양**(image label, box)으로 줘.
  - **L10 (active learning)**: label은 그대로 제대로 주는데, **어느 sample에 줄지를 모델이 골라.**

> 보충: PDF 메타데이터 제목은 `[2026][W10] Active Learning I.pptx`인데, `exam/PLAYBOOK.md` 0.2 일정(L01 p.4 기준)엔 **W12 11/19**로 적혀 있어. L08(W8 ↔ W10)·L09(W9 ↔ W11)와 같은 두 칸 밀림이야. 슬라이드 본문엔 주차 표기가 없어서 시험과는 무관해.

---

## Contents of Table (p.2)

슬라이드 텍스트:
- 1. Introduction to Active Learning Framework
- 2. Sampling Strategy (Query Strategies)
- 3. Deep Active Learning (Ensemble-based Estimation)

**설명**
- 슬라이드 제목은 "Contents of Table"이야 ← "Table of Contents"의 순서가 바뀐 표기. L08·L09도 같은 표기였어. 원문 유지.
- 목차는 세 덩어리야.
  1. **Active learning이 뭐고 어떤 틀로 도나** (p.3–7): motivation, workflow, notation, acquisition function
  2. **어떤 sample을 고르나** (p.8–23): uncertainty sampling(LC · Margin · Entropy), Query-by-Committee, diversity, expected model change, hybrid, 한계
  3. **deep network에서 uncertainty를 어떻게 재나** (p.24–35): aleatoric vs epistemic, ensemble, MC Dropout, DBAL · BALD
- 이 chunk(p.1–19)는 1과 2의 앞부분(uncertainty sampling, QBC)까지야.

---

## Section: Introduction to Active Learning Framework (p.3)

슬라이드 텍스트:
- Introduction to Active Learning Framework

**설명**
- 섹션 구분 슬라이드야. p.4–7이 이 섹션이고, **문제 설정과 기본 틀**(무엇을 최적화하나, 어떤 순서로 도나, 기호)을 깔아.

---

## Active Learning: Motivation (p.4)

슬라이드 텍스트:
- Active Learning
- Motivation
  - We often have…
    - Huge amounts of **unlabeled data**
    - Only a **limited labeling budget**
    - Human annotation is **expensive and time-consuming**
  - Key question
    - If we can label only a small subset, which samples should we choose?
  - Goal of Active Learning
    - Select the **most informative** unlabeled samples
    - Maximize model improvement with **minimum annotation cost**

**설명**
- 상황은 세 줄이야.
  - **unlabeled data**(정답이 안 달린 data)는 엄청 많아. 사진은 긁어 오면 끝이야.
  - **labeling budget**(label을 달 수 있는 최대 개수나 비용)은 한정돼 있어.
  - **human annotation**(사람이 정답을 다는 일)은 비싸고 오래 걸려.
- 그래서 질문이 하나로 모여: **"작은 부분집합에만 label을 달 수 있다면, 어느 sample을 골라야 하나?"**
- **Goal** 두 줄이 시험 정의 문장 후보야.
  - "Select the **most informative** unlabeled samples"
  - "Maximize model improvement with **minimum annotation cost**"
- **informative**(정보가 많은) = 그 sample의 label을 알게 됐을 때 모델이 많이 나아진다는 뜻이야. 어떻게 재는지가 이 강의의 나머지 전부야.
- 이전 강의와 비교:
  - semi-supervised(L06–L07)는 label 없는 data를 **모델이 직접** 쓰는 법이었어 (consistency, pseudo label).
  - weakly-supervised(L08–L09)는 **더 약한 label**을 쓰는 법이었어.
  - active learning은 **label의 질은 그대로**, **label을 줄 sample을 고르는** 법이야.
- 시험 함정: "Active learning aims to label as many samples as possible" → False. budget 안에서 **적게, 잘 골라서**가 목표야.

---

## Active Learning: Objective & Workflow (p.5)

슬라이드 텍스트:
- Active Learning
- Which samples should we label?
- Objective
  - Achieve the highest accuracy
  - With the fewest labeled examples
- Workflow
  - Select informative samples
  - Human annotation
  - Retrain the model
  - Repeat
- (하단 굵은 글씨) How do we measure which samples are the most informative?
- 그림: 순환 도식. 왼쪽 위 폭발 아이콘 "Too much unlabeled data" → 화살표 "Which to label?" → 가운데 노란 박스 **"Active learning"** → 오른쪽 펜 아이콘 "Let's label!" → 아래 돈주머니 아이콘 "Valuable labeled dataset" → 왼쪽 주판 아이콘 "Better model" → 다시 "Active learning" 박스로.

**설명**
- **Objective**: "Achieve the highest accuracy / With the fewest labeled examples". 같은 정확도면 label이 적을수록, 같은 label 수면 정확도가 높을수록 좋아.
- **Workflow** 네 단계가 active learning의 한 바퀴(round)야.
  1. **Select informative samples**: 지금 모델로 unlabeled sample마다 점수를 매기고 가장 informative한 것을 골라 (LC·Entropy는 점수가 큰 쪽, Margin은 작은 쪽, p.12).
  2. **Human annotation**: 고른 sample만 사람에게 보내서 정답을 받아.
  3. **Retrain the model**: 늘어난 labeled set으로 모델을 다시 학습해.
  4. **Repeat**: budget을 다 쓸 때까지 반복해.
- 그림의 순환과 짝지으면 (단계 번호 대응은 해석): "Which to label?" = 1단계, "Let's label!" = 2단계, "Valuable labeled dataset → Better model" = 3단계, 다시 Active learning 박스로 = 4단계.
- **핵심 포인트**: 고르는 기준이 **지금 모델**에서 나와. 모델이 바뀌면 다음 round에 고르는 sample도 바뀌어. 그래서 한 번에 다 고르지 않고 round를 돌아.
- 하단 질문 "How do we measure which samples are the most informative?"가 다음 슬라이드(acquisition function)와 2장 전체로 이어져.

> 보충 (슬라이드 밖): label을 다는 사람(또는 정답을 주는 쪽)을 문헌에선 보통 **oracle**이라고 불러. 그림의 "unlabeled data(candidate pool)"에서 골라 오는 이 설정을 **pool-based** active learning이라고 해. 첫 모델을 학습하려면 처음엔 작은 labeled set이 있어야 해 (슬라이드엔 이 얘기가 없어).

> 보충 (RL 비유, 시험 범위 아님): 이 loop는 imitation learning의 **DAgger**와 모양이 같아. 지금 policy를 돌려 보고 → 전문가에게 label(정답 action)을 받고 → 다시 학습하고 → 반복해. 원래 DAgger는 방문한 state 전부에 label을 받지만, 사람이 개입할 때를 **uncertainty로 고르는** 변형(ThriftyDAgger 등)은 active learning의 "어느 sample에 label을 받나"와 같은 질문을 풀어.

---

## Notations (p.6)

슬라이드 텍스트 (표, Symbol / Meaning):

| Symbol | Meaning |
|---|---|
| $K$ | Number of unique class labels. |
| $(\mathbf{x}^l, y) \sim \mathcal{X},\ y \in \{0,1\}^K$ | Labeled dataset. $y$ is a one-hot representation of the true label. |
| $\mathbf{u} \sim \mathcal{U}$ | Unlabeled dataset. |
| $\mathcal{D} = \mathcal{X} \cup \mathcal{U}$ | The entire dataset, including both labeled and unlabeled examples. |
| $\mathbf{x}$ | Any sample which can be either labeled or unlabeled. |
| $\mathbf{x}_i$ | The $i$-th sample. |
| $U(\mathbf{x})$ | Scoring function for active learning selection. |
| $P_\theta(y \mid \mathbf{x})$ | A softmax classifier parameterized by $\theta$. |
| $\hat{y} = \arg\max_{y \in \mathcal{Y}} P_\theta(y \mid \mathbf{x})$ | The most confident prediction by the classifier. |
| $B$ | Labeling budget (the maximum number of samples to label). |
| $b$ | Batch size. |

**설명**
- 기호를 하나씩 풀면:
  - $K$: class 개수야. 예: cat / dog / fox면 $K=3$.
  - $(\mathbf{x}^l, y) \sim \mathcal{X}$: **labeled** set에서 뽑은 (sample, 정답) 쌍이야. 위첨자 $l$은 labeled라는 표시야. $y \in \{0,1\}^K$는 **one-hot**(정답 class 칸만 1이고 나머지는 0인 길이 $K$ 벡터)이야.
  - $\mathbf{u} \sim \mathcal{U}$: **unlabeled** set에서 뽑은 sample이야. active learning이 고르는 후보가 여기 있어.
  - $\mathcal{D} = \mathcal{X} \cup \mathcal{U}$: 둘을 합친 전체 dataset이야.
  - $U(\mathbf{x})$: **scoring function**. sample 하나에 "label 달 가치" 점수를 하나 매겨. p.7의 acquisition function이 이거야.
  - $P_\theta(y \mid \mathbf{x})$: parameter $\theta$인 **softmax classifier**의 class별 확률이야. 합이 1인 길이 $K$ 벡터야.
  - $\hat{y}$: 그 확률이 제일 큰 class, 즉 모델의 예측이야.
  - $B$: **labeling budget**. label을 달 수 있는 sample의 최대 개수야.
  - $b$: **batch size**. 한 round에 몇 개를 골라서 label을 받나야.
- 두 개를 같이 읽으면 (해석): round마다 가장 informative한 $b$개(= $U$가 가리키는 쪽)를 골라. LC·Entropy는 $U$가 **큰** 쪽, Margin은 $U$가 **작은** 쪽이야 (p.12), label을 받고, 모은 개수가 $B$가 되면 멈춰. 대략 $B/b$ round야.
- ⚠️ 헷갈리는 표기 세 개
  - $\mathcal{U}$(필기체) = unlabeled **dataset**, $U(\mathbf{x})$(이탤릭) = **scoring function**. 글자는 같은 U인데 뜻이 전혀 달라.
  - $\mathcal{X}$가 여기선 **labeled dataset**이야. ML에서 흔히 $\mathcal{X}$는 "입력 공간"을 뜻하니 헷갈리지 마.
  - $B$(budget, 전체 상한) ↔ $b$(batch, 한 round에 고르는 수). 대소문자 하나 차이야.
- 뒤 슬라이드(p.11–13)는 $\theta$를 빼고 $P(y \mid x)$로 써. 같은 classifier 확률이야.

---

## Acquisition Function (p.7)

슬라이드 텍스트:
- Acquisition Function
- Core Question
  - Which unlabeled sample should we label next?
- Acquisition Function
  - Assigns an informativeness score to each unlabeled sample.
- The process of identifying the most valuable examples to label next is referred to as "sampling strategy" or "query strategy".

**설명**
- **acquisition function** = unlabeled sample마다 **informativeness score**(label 달 가치 점수)를 매기는 함수야. p.6의 $U(\mathbf{x})$가 이거야.
- 슬라이드 정의: "Assigns an informativeness score to each unlabeled sample."
- **sampling strategy = query strategy**: "다음에 label 달 가장 가치 있는 sample을 찾는 과정" 전체를 부르는 이름이야. 두 이름은 같은 뜻이야. 2장 제목 "Sampling Strategy (Query Strategies)"가 이거야.
- 정리하면: acquisition function은 **점수를 매기는 함수**, sampling/query strategy는 **그 점수로 고르는 과정**이야. 슬라이드도 둘을 거의 같은 맥락으로 써.
- 2장의 방법들은 전부 "$U(\mathbf{x})$를 어떻게 정의하나"의 변형이야.
  - 한 모델의 확률이 얼마나 애매한가 → uncertainty sampling (p.9–13)
  - 여러 모델이 얼마나 엇갈리나 → QBC (p.14–19)
  - 얼마나 대표적이고 다양한가 → diversity (p.20)
  - label을 받으면 모델이 얼마나 바뀌나 → expected model change (p.21)

> 보충 (RL 비유, 시험 범위 아님): acquisition function은 exploration에서 "어느 state·action을 시도할 가치가 큰가"를 매기는 점수(exploration bonus, UCB 항)와 역할이 비슷해. 둘 다 "정보를 제일 많이 줄 곳에 한정된 예산을 쓰자"는 결정이야.

---

## Section: Sampling Strategy (Query Strategies) (p.8)

슬라이드 텍스트:
- Sampling Strategy (Query Strategies)

**설명**
- 섹션 구분 슬라이드야. p.9–23이 이 섹션이고, **acquisition function을 어떻게 정의하나**를 다뤄.
- 순서: uncertainty sampling (p.9–13) → Query-by-Committee (p.14–19) → diversity (p.20) → expected model change (p.21) → hybrid (p.22) → limitations (p.23).

---

## Uncertainty Sampling (p.9)

슬라이드 텍스트:
- Uncertainty Sampling
- Key Idea
  - Label the samples that the **current model is least confident** about.
  - **Not all** unlabeled samples are **equally informative**.
  - **Samples near the decision boundary** often provide the **greatest learning benefit.**
- 그림: 2차원 평면($x_1$, $x_2$ 축). 왼쪽 파란 배경에 파란 점(Labeled data, Class 1), 오른쪽 빨간 배경에 빨간 삼각형(Labeled data, Class 2), 곳곳에 회색 빈 원(Unlabeled data, candidate pool). 가운데 구불구불한 검은 점선(Decision boundary, current model)과 그 양옆 회색 점선(Uncertainty region, near the boundary). 경계 근처의 회색 원 몇 개가 보라색 원(Selected by uncertainty sampling)으로 표시됨.
- 그림 안 박스 "Key Idea": Samples near the decision boundary are the most uncertain and are expected to provide the greatest learning benefit when labeled.

**설명**
- **uncertainty sampling** = 지금 모델이 **제일 자신 없어 하는** sample에 label을 달아.
- **decision boundary**(결정 경계) = classifier가 class 1과 class 2를 나누는 선이야. 경계 위에선 두 class 확률이 거의 같아 (일반 상식, 슬라이드 밖).
- 왜 경계 근처가 좋아? (해석, 슬라이드엔 "often provide the greatest learning benefit"까지만 있어)
  - 경계에서 먼 sample(파란 덩어리 한가운데)은 이미 확실하게 맞혀. label을 받아도 "역시 class 1이네"라서 모델이 거의 안 바뀌어.
  - 경계 근처 sample은 label을 받으면 **경계를 어디로 그어야 하는지**를 직접 알려 줘. 같은 label 한 개로 얻는 게 커.
- 그림 읽는 법: 회색 원(후보) 중에서 **두 회색 점선 사이**(uncertainty region)에 있는 것만 보라색(선택)이 됐어. 파란·빨간 덩어리 한가운데의 회색 원은 안 골라.
- 세 문장 중 "Not all unlabeled samples are equally informative"가 active learning 전체의 전제야. 다 똑같이 가치 있으면 random으로 뽑아도 되니까.

> 보충 (L07 연결): L07 p.5–7의 entropy minimization은 unlabeled data에서 예측 entropy를 **줄여서** decision boundary를 data가 적은 곳(low-density region)으로 밀어냈어. FixMatch(L07 p.45–46)는 confidence가 threshold $\tau$를 **넘는** 확실한 sample만 pseudo label로 썼어. active learning은 반대로 **가장 불확실한** sample을 골라 **사람에게** 보내. 같은 confidence 값을 보는데, semi-supervised는 확실한 쪽을 모델이 스스로 쓰고, active learning은 애매한 쪽을 사람에게 맡겨.

---

## How to Measure Uncertainty? (p.10)

슬라이드 텍스트:
- How to Measure Uncertainty?
- How uncertain is this probability?
  - $P_\theta(y \mid \mathbf{x})$
- Three common measures
  - Least Confidence
  - Margin Sampling
  - Entropy

**설명**
- 모델이 내는 건 class별 확률 벡터 $P_\theta(y \mid \mathbf{x})$ 하나야. "이 확률이 얼마나 애매한가"를 숫자 하나로 바꾸는 방법이 세 가지야.
  1. **Least Confidence (LC)**: 1등 확률만 봐 (p.11).
  2. **Margin Sampling**: 1등과 2등 확률의 차이를 봐 (p.12).
  3. **Entropy**: 확률 분포 전체를 봐 (p.13).
- 셋 다 **모델 하나**의 확률만 써 (해석: p.10엔 $P_\theta(y \mid \mathbf{x})$ 하나만 나와). 여러 모델을 쓰는 건 p.14의 QBC부터야.

---

## Least Confidence (p.11)

슬라이드 텍스트:
- Least Confidence
- Predict
  - $\hat{y} = \arg\max_y P(y \mid x)$
- Acquisition score
  - $U(x) = 1 - P(\hat{y} \mid x)$
- Uses only the highest predicted probability
- Lower confidence → Higher priority

**설명**
- 두 줄로 계산해.
  1. $\hat{y} = \arg\max_y P(y \mid x)$: 확률이 제일 큰 class(모델의 예측)를 찾아.
  2. $U(x) = 1 - P(\hat{y} \mid x)$: 그 1등 확률을 1에서 빼.
- **confidence** = 1등 확률 $P(\hat{y} \mid x)$야. 1등 확률이 낮을수록 $U$가 커지고, **$U$가 큰 sample부터** 골라.
- 슬라이드 문장: "Uses only the highest predicted probability", "Lower confidence → Higher priority".
- 범위 (보충, 슬라이드 밖): 1등 확률은 최소 $1/K$(완전 균등)이고 최대 1이야. 그래서 $0 \le U(x) \le 1 - 1/K$.
- 약점: **1등 확률만** 봐서 나머지 class가 어떻게 나뉘었는지는 몰라. (0.5, 0.49, 0.01)과 (0.5, 0.25, 0.25)는 LC 점수가 0.5로 같아.

---

## Margin Sampling (p.12)

슬라이드 텍스트:
- Margin Sampling
- Use the top two probabilities
  - $U(x) = P(\hat{y}_1 \mid x) - P(\hat{y}_2 \mid x)$
- Small margin → High uncertainty

**설명**
- $\hat{y}_1$ = 확률 1등 class, $\hat{y}_2$ = 확률 2등 class야. **margin** = 1등 확률 − 2등 확률.
- margin이 작으면 모델이 **두 후보 사이에서 망설이는** 중이야. "Small margin → High uncertainty".
- ⚠️ **방향 함정**: LC와 Entropy는 $U$가 **큰** sample을 고르는데, Margin은 슬라이드 식 그대로면 $U$가 **작은** sample을 골라.
  - p.6에서 $U(\mathbf{x})$는 그냥 "Scoring function for active learning selection"이라서, 어느 쪽이 선택인지는 측도마다 따로 기억해야 해.
  - 구현에선 보통 $-U$나 $1-U$의 최대를 고르거나, $U$의 최소(argmin)를 골라 (슬라이드 밖).
  - 시험 지문 "Margin sampling selects the sample with the largest margin" → False.
- LC와의 차이: LC는 1등만, Margin은 **1등과 2등 둘**을 봐. (0.5, 0.49, 0.01)은 margin 0.01로 매우 불확실, (0.5, 0.25, 0.25)는 margin 0.25로 덜 불확실해. LC로는 구분 못 하던 둘을 구분해.

---

## Entropy Sampling (p.13)

슬라이드 텍스트:
- Entropy Sampling
- Uses the entire probability distribution
- Higher entropy → Higher uncertainty
  - $U(x) = -\sum_y P(y \mid x) \log P(y \mid x)$

**설명**
- **entropy**(Shannon entropy) = 확률 분포가 얼마나 퍼져 있나를 재는 양이야. 아래 두 범위 값은 식에서 계산한 보충이야 (슬라이드 밖).
  - 한 class에 확률 1이 몰리면 (one-hot) $-1 \cdot \log 1 = 0$이라 entropy가 **0**(최소)이야.
  - 모든 class가 $1/K$로 똑같으면 entropy가 $\log K$로 **최대**야.
- 식 풀이: class마다 $P(y \mid x)\log P(y \mid x)$를 더하고 부호를 바꿔. $\log P \le 0$이라 앞에 $-$를 붙여서 양수로 만든 거야.
- **$U$가 큰 sample부터** 골라. "Higher entropy → Higher uncertainty".
- 세 측도 중 유일하게 **분포 전체**를 써 ("Uses the entire probability distribution"). class가 많을 때 2등 밑의 class들이 얼마나 퍼졌는지도 반영해.

**세 측도 비교 (p.11–13)**

| 측도 | 식 | 쓰는 정보 | 고르는 쪽 |
|---|---|---|---|
| Least Confidence | $1 - P(\hat{y} \mid x)$ | 1등 확률만 | $U$ 큰 것 |
| Margin | $P(\hat{y}_1 \mid x) - P(\hat{y}_2 \mid x)$ | 1등·2등 확률 | $U$ **작은** 것 |
| Entropy | $-\sum_y P(y \mid x)\log P(y \mid x)$ | 분포 전체 | $U$ 큰 것 |

- 설명용 toy 예시 (슬라이드 수치 아님, 3-class, 자연로그):
  - S1 = (0.90, 0.05, 0.05): LC 0.10, Margin 0.85, Entropy 0.39 → 셋 다 "확실함".
  - S2 = (0.45, 0.45, 0.10): LC 0.55, Margin **0.00**, Entropy 0.95
  - S3 = (0.40, 0.30, 0.30): LC **0.60**, Margin 0.10, Entropy **1.09**
  - LC와 Entropy는 **S3**를 고르고, Margin은 **S2**를 골라. S2는 두 class 사이의 동전 던지기라 margin이 0이야. S3는 세 class에 고루 퍼져서 1등 확률과 entropy가 더 나빠.
- 슬라이드 밖 사실: class가 **2개**면 세 측도가 항상 **같은 순서**로 골라. 1등 확률을 $p$라 하면 LC $= 1-p$, Margin $= 2p-1$, Entropy는 $p \ge 0.5$에서 $p$가 커질수록 줄어들어서 셋 다 $p$ 하나로 정해져. 차이는 $K \ge 3$에서만 생겨.

---

## Query-by-Committee (QBC) (p.14)

슬라이드 텍스트:
- Query-by-Committee (QBC)
- Key Idea
  - Instead of relying on a single model, measure uncertainty by disagreement among multiple models.
    - Train multiple models (committee)
    - Each model predicts independently
    - Select samples with the largest disagreement
- 그림 (위): "Unlabeled sample x" (고양이 사진 + ?) → 점선 박스 "Committee of models" (Model 1, Model 2, Model 3, …, Model C) → "Predictions (votes)": Cat, Dog, Cat, …, Dog → "Votes on each class (e.g., Cat vs Dog)": Cat 3, Dog 2, "Vote distribution [3/5, 2/5]" → 빨간 박스 "Uncertainty (disagreement) High (e.g., entropy, KL divergence, etc.)" → "Select this sample for labeling"
- 그림 (아래 왼쪽) "Low disagreement (low uncertainty)": 강아지 사진 → M1, M2, M3, …, MC 전부 Dog → Dog 5, Cat 0 → "Most models agree → low uncertainty"
- 그림 (아래 오른쪽) "High disagreement (high uncertainty)": 고양이 사진 → M1 Cat, M2 Dog, M3 Cat, …, MC Dog → Cat 2, Dog 3 → "Models disagree → high uncertainty"

**설명**
- **QBC (Query-by-Committee)** = 모델 **하나**의 자신감 대신, **여러 모델**(committee, 위원회)이 서로 **얼마나 엇갈리나**(disagreement)로 uncertainty를 재.
- 순서 (슬라이드 세 줄):
  1. **Train multiple models (committee)**: 같은 labeled set으로 모델 여러 개를 학습해.
  2. **Each model predicts independently**: unlabeled sample 하나를 각 모델이 따로 예측해.
  3. **Select samples with the largest disagreement**: 예측이 제일 엇갈리는 sample을 골라.
- 그림 위 줄은 committee 5명(Cat 3 + Dog 2)이 갈라진 경우야. 투표 분포 [3/5, 2/5]가 고르게 퍼질수록 disagreement가 커.
- 아래 두 그림은 대비야. 강아지 사진은 5명 전원이 Dog라서 안 골라. 고양이 사진은 2 : 3으로 갈려서 골라.
- 왜 돼? 모델들이 다 같은 데이터로 학습했는데도 엇갈린다면, 그 sample 근처는 **아직 데이터가 결정해 주지 못한 영역**이야. label이 오면 committee의 의견을 하나로 모아 줘.
- uncertainty sampling과 차이: uncertainty sampling은 **한 모델의 확률이 애매한가**, QBC는 **여러 모델의 답이 서로 다른가**를 봐. disagreement를 숫자로 바꾸는 방법은 p.15–19.
- committee를 어떻게 다르게 만드는지(다른 초기값, 다른 bootstrap 표본 등)는 슬라이드에 없어. p.26의 ensemble 예시(AdaBoost, Bootstrapping, Random forests)와 p.27의 MC Dropout이 이 질문으로 이어져.

> 보충: 슬라이드엔 cite가 없어. QBC는 Seung, Opper & Sompolinsky, "Query by committee" (COLT 1992)에서 나온 이름이야.

> 보충 (RL 비유, 시험 범위 아님): ensemble dynamics model의 **disagreement를 intrinsic reward(exploration bonus)**로 쓰는 것(Pathak et al., "Self-Supervised Exploration via Disagreement")과 같은 발상이야. 모델들이 엇갈리는 곳 = 아직 모르는 곳 = 가 볼(label 받을) 가치가 있는 곳.

---

## Measuring Disagreement (p.15)

슬라이드 텍스트:
- Measuring Disagreement
- Voter Entropy
- Consensus Entropy
- KL Divergence

**설명**
- committee의 disagreement를 숫자 하나로 만드는 방법 세 가지야. 차이는 **각 모델에게서 무엇을 받느냐**야.
  1. **Voter Entropy**: 각 모델의 **최종 label(표)** 만 받아 (p.16).
  2. **Consensus Entropy**: 각 모델의 **확률 분포**를 받아서 평균 내 (p.17).
  3. **KL Divergence**: 각 모델의 분포가 **평균에서 얼마나 떨어졌나**를 재 (p.18).
- 정리 표는 p.19에 있어.

---

## Voter Entropy (p.16)

슬라이드 텍스트:
- Voter Entropy
- Count committee votes.
  - $U(x) = H\left(\dfrac{V(y)}{C}\right)$
- Only considers the predicted label of each model
- High vote diversity → High uncertainty
- 기호 표 (Symbol / Meaning):
  - $\mathbf{x}$: Unlabeled sample
  - $C$: Number of models in the committee
  - $V(y)$: Number of committee members voting for class $y$
  - $\frac{V(y)}{C}$: Fraction of votes for class $y$
  - $\mathcal{H}(\cdot)$: Shannon entropy

**설명**
- 계산 순서:
  1. 각 모델의 예측 label(보통 argmax, 슬라이드는 "predicted label"이라고만 함)을 **한 표**로 세.
  2. class마다 표 수 $V(y)$를 committee 크기 $C$로 나눠 **투표 비율** $V(y)/C$을 만들어. 합이 1인 분포야.
  3. 그 투표 분포의 **entropy**를 재: $U(x) = -\sum_y \frac{V(y)}{C}\log\frac{V(y)}{C}$.
- 표가 한쪽에 몰리면(5 : 0) entropy 0, 고루 갈리면 커. "High vote diversity → High uncertainty".
- 예시 (p.14 그림의 숫자): Cat 3, Dog 2 → $H(3/5, 2/5) \approx 0.67$ (자연로그). 전원 Dog(5 : 0) → $0$.
- 약점: "Only considers the predicted label of each model". 모델이 cat을 0.51로 겨우 찍었든 0.99로 확신했든 **똑같은 한 표**야.
- 표기: 식에선 $H$, 기호 표에선 $\mathcal{H}(\cdot)$. 같은 Shannon entropy야.

> 보충: 슬라이드 이름은 "Voter Entropy"야. 문헌에선 보통 **vote entropy**라고 불러. 시험엔 슬라이드 표기로 나올 거야.

---

## Consensus Entropy (p.17)

슬라이드 텍스트:
- Consensus Entropy
- Average prediction probabilities.
  - $U(x) = H(P_C)$
  - $P_C(y) = \dfrac{1}{C}\sum_{c=1}^{C} P_{\theta_c}(y \mid \mathbf{x})$
- Uses the full probability distribution
- Measures uncertainty of the committee consensus
- 기호 표:
  - $\theta_c$: Parameters of model $c$
  - $P_C(y)$: Average probability over all committee members
  - $\mathcal{H}(\cdot)$: Shannon entropy

**설명**
- 계산 순서:
  1. 모델 $c$(parameter $\theta_c$)마다 확률 분포 $P_{\theta_c}(y \mid \mathbf{x})$를 받아.
  2. $C$개를 class별로 **평균** 내서 committee의 합의 분포 **consensus** $P_C(y)$를 만들어.
  3. 그 평균 분포의 entropy $H(P_C)$를 재.
- "Uses the full probability distribution": Voter와 달리 확신의 정도가 들어가. 0.51짜리 cat과 0.99짜리 cat이 다르게 반영돼.
- "Measures uncertainty of the committee consensus": committee가 **합의한 결과가** 얼마나 애매한가를 재.
- 함정: $P_C$의 아래첨자 $C$는 **committee**(그리고 그 크기 $C$)를 가리켜. class 개수는 $K$야.
- ⚠️ 약점 (슬라이드 밖 분석, p.29–31 BALD로 이어짐): 평균의 entropy는 두 경우에 **둘 다** 커.
  - 모든 모델이 똑같이 애매한 경우 (예: 다 (0.5, 0.5)).
  - 모델마다 확신하는데 서로 다른 답을 내는 경우 (예: 반은 cat 0.99, 반은 dog 0.99).
  - consensus entropy 하나로는 이 둘을 구분 못 해. 구분하는 게 p.18의 KL이고, 3장의 BALD야.

---

## KL Divergence (p.18)

슬라이드 텍스트:
- KL Divergence
- Measures disagreement between each model and the committee average
- Larger KL → Larger disagreement
  - $U(x) = \dfrac{1}{C}\sum_{c=1}^{C} D_{KL}\left(P_{\theta_c} \,\|\, P_C\right)$
- (굵은 글씨) How far each model's prediction is from the committee average.

**설명**
- **KL divergence**(Kullback–Leibler divergence) $D_{KL}(P \| Q) = \sum_y P(y)\log\frac{P(y)}{Q(y)}$ = 분포 $P$가 분포 $Q$와 얼마나 다른가야. 같으면 0, 다를수록 커. 순서를 바꾸면 값이 달라져 (대칭이 아님).
- 계산 순서:
  1. consensus $P_C$를 p.17처럼 만들어.
  2. 모델마다 **자기 분포 $P_{\theta_c}$가 평균 $P_C$에서 얼마나 떨어졌나** $D_{KL}(P_{\theta_c} \| P_C)$를 재.
  3. $C$개를 평균 내.
- 방향: 앞이 **각 모델** $P_{\theta_c}$, 뒤가 **committee 평균** $P_C$야. 지문에서 순서를 뒤집으면 슬라이드 식과 달라져.
- "Larger KL → Larger disagreement". 모든 모델이 같은 분포를 내면 (아무리 애매해도) 각자 평균과 같아서 **KL = 0**이야.

> 보충 (슬라이드 밖 유도, 검산함): 이 KL 평균은 정확히 이렇게 풀려.
> $$\frac{1}{C}\sum_{c} D_{KL}(P_{\theta_c} \| P_C) = H(P_C) - \frac{1}{C}\sum_{c} H(P_{\theta_c})$$
> 1. $D_{KL}(P_{\theta_c}\|P_C) = \sum_y P_{\theta_c}(y)\log P_{\theta_c}(y) - \sum_y P_{\theta_c}(y)\log P_C(y) = -H(P_{\theta_c}) - \sum_y P_{\theta_c}(y)\log P_C(y)$
> 2. $c$에 대해 평균을 내면 둘째 항의 $\frac1C\sum_c P_{\theta_c}(y)$가 $P_C(y)$가 돼서 $-\sum_y P_C(y)\log P_C(y) = H(P_C)$.
> 3. 그래서 **평균의 entropy (consensus entropy) − 각 모델 entropy의 평균**이야.
> - 이건 p.29의 **BALD 근사식** $\mathcal{H}[\bar p] - \frac1T\sum_t \mathcal{H}[p_t]$과 같은 모양이야. committee 멤버를 MC Dropout sample로 바꾸면 BALD가 돼. 3장에서 다시 봐.

---

## Voter vs Consensus vs KL 정리 (p.19)

슬라이드 텍스트 (표):

| Method | Uses |
|---|---|
| Voter Entropy | Final class labels only |
| Consensus Entropy | Full probability distributions |
| KL Divergence | Distance between distributions |

- Voter Entropy: "How many people chose which class?"
- Consensus Entropy: "How much uncertainty is there on average?"
- KL Divergence: "How much does each model differ from the average opinion?"

**설명**
- 시험에 매칭·"What is X for?" 문제로 나오기 좋은 표야. **무엇을 쓰나**로 외워.
  - Voter = **최종 label만** (표 세기)
  - Consensus = **확률 분포 전체**의 평균
  - KL = **분포 사이 거리** (각 모델 ↔ 평균)
- 설명용 toy 예시 (슬라이드 수치 아님, 2-class, committee 5명, 자연로그):

| 경우 | 각 모델 예측 | Voter | Consensus | KL 평균 |
|---|---|---|---|---|
| A. 다 같이 애매함 | 5명 모두 (0.51, 0.49) | 0 (5 : 0) | 0.69 (최대 ln 2 ≈ 0.693에 거의 닿음) | **0** |
| B. 각자 확신, 서로 다름 | 3명 (0.99, 0.01), 2명 (0.01, 0.99) | 0.67 (3 : 2) | 0.67 | **0.62** |

- 읽는 법:
  - A는 표가 한쪽으로 몰려서 Voter는 0인데, 평균 분포가 거의 반반이라 Consensus는 최대에 가까워. 모델끼리 **같은 말**을 해서 KL은 0이야.
  - B는 셋 다 커. 모델끼리 진짜로 **엇갈려**.
  - Consensus는 A와 B를 거의 같은 값(0.69 vs 0.67)으로 봐. Voter는 이 예시에선 A/B를 가르지만 label만 봐서 확신도(0.51 vs 0.99)를 못 봐. **확률을 쓰면서** "모델끼리 다른가"를 직접 재는 건 **KL**이야. 이 구분이 3장 aleatoric vs epistemic, BALD Case 1 vs Case 2(p.30–31)와 같은 이야기야.

> 보충 (L09 연결, 슬라이드 밖): L09 CD-CLIP(p.69)에 나온 JS divergence(Jensen–Shannon)는 정의상 두 분포 $P, Q$의 평균 $M = (P+Q)/2$에 대한 KL의 평균 $\frac12 D_{KL}(P\|M) + \frac12 D_{KL}(Q\|M)$이야 (L09 슬라이드엔 정의가 없었어). p.18의 KL 평균은 그걸 $C$개 분포로 넓힌 **generalized JS divergence**야.
