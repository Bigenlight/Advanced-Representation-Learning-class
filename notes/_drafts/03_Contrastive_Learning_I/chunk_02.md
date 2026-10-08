<!-- DRAFT | lecture: 03_Contrastive_Learning_I | chunk: 02 | pages 9–16 -->
<!-- VERIFIED | fixes: 3 | verifier: sonnet -->

## Contrastive Representation Learning (p.9)

- Learn embedding space
  - Similar sample pairs close to each other
  - Dissimilar ones are far apart
- Can be applied to both supervised and unsupervised settings.
- What we want:

$$\text{score}(f(x), f(x^+)) \gg \text{score}(f(x), f(x^-))$$

- x: reference sample; x⁺ positive sample; x⁻ negative sample
- Given a chosen score function, we aim to learn an encoder function f that yields high score for positive pairs (x, x⁺) and low scores for negative pairs (x, x⁻).

**설명**

- **Contrastive learning**의 목표는 입력 공간이 아니라 **embedding space**의 기하 구조를 설계하는 것임. 비슷한 샘플은 가깝게, 다른 샘플은 멀게 배치하는 encoder $f$를 학습함.
- 기호: $x$ = **reference(anchor)**, $x^+$ = "비슷하다고 정의된" **positive**, $x^-$ = "다르다고 정의된" **negative**, $f$ = encoder, score = 유사도 함수(cosine similarity, dot product, −distance 등). **score function은 미리 고정**하고 $f$만 학습함.
- 핵심 요구는 절대값이 아니라 **상대적 대소관계**($\gg$)임. 이후 모든 loss(contrastive, triplet, lifted, N-pair, InfoNCE …)는 이 부등식을 미분 가능한 objective로 바꾼 서로 다른 방식임.
- **supervised / unsupervised 둘 다 가능**한 이유는 positive 정의만 다르기 때문임.
  - supervised: 같은 label = positive, 다른 label = negative.
  - unsupervised(self-supervised): 같은 이미지의 서로 다른 augmentation(view) = positive, 다른 이미지 = negative.
> 보충: classification의 cross-entropy와 달리 label 개수에 묶이지 않아, 학습 때 못 본 class(새 얼굴 등)에도 거리 기반으로 일반화 가능함 (face verification, retrieval에 유리).

---

## Contrastive Training Objectives (p.10)

(섹션 구분 슬라이드: "Contrastive Training Objectives")

---

## Contrastive Loss (p.11)

슬라이드는 좌우 두 박스 구성.

**왼쪽 박스: Intuitive Goal**
- Learn an encoder function $f_\theta(x)$ that maps input samples into an embedding space where:
  - **Positive Pairs — Minimize distance**: Samples from the **same class** are pulled close together.
  - **Negative Pairs — Maximize distance**: Samples from **different classes** are pushed apart beyond a margin $\varepsilon$.

**오른쪽 박스: Mathematical Formulation**
- Contrastive Loss Function

$$L_{\text{cont}} = \mathbb{1}_{[y_i = y_j]}\,\|f_\theta(x_i) - f_\theta(x_j)\|_2^2 + \mathbb{1}_{[y_i \neq y_j]}\,\max\big(0,\ \varepsilon - \|f_\theta(x_i) - f_\theta(x_j)\|_2\big)^2$$

1. Equivalent Class Case ($y_i = y_j$): $L = \|f_\theta(x_i) - f_\theta(x_j)\|_2^2$
2. Different Class Case ($y_i \neq y_j$): $L = \max\big(0,\ \varepsilon - \|f_\theta(x_i) - f_\theta(x_j)\|_2\big)^2$
- where $\varepsilon$ (margin) defines the minimum distance required between different classes.

**하단 텍스트**
- Contrastive Loss considers only pairs
  - Many negatives become inactive
  - Does not explicitly enforce relative ordering

**출처**: Chopra et al., 2005, *Learning a Similarity Metric Discriminatively, with Application to Face Verification*

**설명**

- 입력은 **pair $(x_i, x_j)$ + 같은 class 여부**. indicator $\mathbb{1}[\cdot]$로 두 경우 중 하나만 켜짐.
- **positive pair**: loss = 거리 제곱 → 두 embedding이 같은 점으로 모일수록 감소.
- **negative pair**: loss = $\max(0, \varepsilon - d)^2$ (**hinge**).
  - $d < \varepsilon$이면 margin 안쪽이라 loss > 0 → 밀어냄.
  - $d \ge \varepsilon$이면 loss와 gradient가 0 → **충분히 멀면 더 밀지 않음**. 덕분에 negative를 무한히 밀어 embedding이 발산하는 것을 막음.
- $\varepsilon$ = **margin**: negative가 최소한 지켜야 할 거리.
- 구조: **Siamese network**(p.8)처럼 같은 가중치 $f_\theta$를 두 입력에 공유함.
- 한계
  - **pair만 본다**: 한 번에 상대 하나만 비교해 정보량이 적음.
  - **많은 negative가 inactive**: 대부분의 negative pair는 이미 $d \ge \varepsilon$라 gradient = 0.
  - **relative ordering을 명시적으로 강제하지 않음**: positive는 0으로, negative는 $\varepsilon$ 이상으로 각각 독립적인 절대 목표를 줌. 이것이 **triplet loss**의 동기.
> 보충: Hadsell et al. 2006(DrLIM) 등에서는 $\frac12$ 계수와 $Y$ label 표기를 쓰는 변형이 흔함. 슬라이드는 indicator 표기로 쓴 형태.

---

## Triplet Loss — FaceNet (p.12)

- [FaceNet] learn face recognition of the same person at different poses and angles.
  - Triplet Loss uses Anchor–Positive–Negative triplets
    - Enforces relative similarity
    - Focuses on ranking relationships

**그림 (좌측)**: 같은 인물의 서로 다른 각도/조명 얼굴 사진 2장(위: 어두운 푸른 조명의 옆얼굴, 아래: 정면에 가까운 얼굴)을 세로 배치. 같은 사람의 pose 변화를 보여줌.

**그림 (우측 위, FaceNet 파이프라인)**: 왼쪽→오른쪽 화살표 흐름. **Batch**(색 줄무늬 = 이미지 묶음) → **DEEP ARCHITECTURE**(회색 박스) → **L2**(노란 박스, L2 normalization) → **EMBEDDING**(세로로 쓴 벡터 박스) → **Triplet Loss**(주황 박스).

**그림 (우측 아래, LEARNING 전후)**: 학습 전에는 Anchor(파랑)에서 Negative(빨강)가 Positive(초록)보다 가깝게 위치. 가운데 곡선 화살표 "LEARNING" 이후에는 Anchor–Positive가 가까워지고 Negative는 멀리(오른쪽 위) 이동함.

**출처**: Schroff et al., 2015, *FaceNet: A Unified Embedding for Face Recognition and Clustering*

**설명**

- **FaceNet** (Schroff, Kalenichenko, Philbin; Google, CVPR 2015): 얼굴 이미지를 직접 Euclidean embedding으로 매핑해 **거리 자체가 얼굴 유사도**가 되게 함. verification = threshold, recognition = k-NN, clustering = k-means로 해결 가능함.
- **Triplet** = (Anchor, Positive, Negative): anchor와 같은 사람(positive), 다른 사람(negative).
- contrastive와의 차이: 절대 거리 목표가 아니라 **"anchor–positive 거리 < anchor–negative 거리"라는 상대 순서(ranking)** 를 학습함.
- 파이프라인의 **L2 normalization**으로 embedding이 단위 hypersphere 위에 놓여 거리 스케일이 제한되고 학습이 안정됨.
- before/after 그림은 negative가 positive보다 가까운 잘못된 배치가 학습 후 positive가 더 가깝고 negative는 margin 이상 멀어지는 효과를 보여줌.
> 보충: FaceNet은 학습 시 batch 안에서 online으로 triplet을 고르며 semi-hard negative mining을 사용함.

---

## Triplet Loss — 수식 (p.13)

**왼쪽 박스: Intuitive Goal**
- Map samples into an embedding space using triplets: Anchor ($x$), Positive ($x^+$), and Negative ($x^-$).
  - **POSITIVE — Minimize Distance**: Anchor and positive samples (same class) are pulled close together.
  - **NEGATIVE — Maximize Distance**: Anchor and negative samples (different classes) are pushed apart.
- Key: Selecting challenging negative samples (**hard negatives**) is crucial to truly improve the model.

**오른쪽 박스: Mathematical Formulation**
- Triplet Loss Function

$$L_{\text{triplet}} = \sum \max\big(0,\ \|f(x) - f(x^+)\|_2^2 - \|f(x) - f(x^-)\|_2^2 + \varepsilon\big)$$

- Margin Parameter ($\varepsilon$): The margin $\varepsilon$ is configured as the minimum offset required between the distances of similar (anchor-positive) vs dissimilar (anchor-negative) pairs.

**하단 텍스트**
- Problem 1: Triplet Explosion
  - Possible triplet number is N^3 for N samples
- Problem 2: Most Triplets are Useless
  - Easy triplets dominate

**설명**

- $d_{ap} = \|f(x)-f(x^+)\|_2^2$, $d_{an} = \|f(x)-f(x^-)\|_2^2$라 하면 loss = $\max(0,\ d_{ap} - d_{an} + \varepsilon)$.
  - $d_{an} - d_{ap} \ge \varepsilon$이면 **loss = 0** → 만족된 triplet은 gradient 없음.
  - 위반 triplet만 loss > 0 → $d_{ap}$ 감소, $d_{an}$ 증가 방향으로 gradient가 흐름.
- contrastive와의 차이
  - contrastive: positive는 0으로, negative는 $\varepsilon$ 이상으로 **각각 절대 목표**.
  - triplet: **두 거리의 차(상대 격차)** 만 margin 이상이면 됨. positive를 한 점으로 모을 필요가 없어 intra-class variance를 허용하는 더 유연한 제약임.
- margin $\varepsilon$은 contrastive에서는 "절대 거리", triplet에서는 "거리 차이" 기준이라는 점이 다름.
- Key 문구: 아무 negative나 뽑으면 이미 조건을 만족해 loss = 0이므로 **hard negative**가 필요함 (다음 슬라이드).
- 문제점
  - **Triplet Explosion**: 가능한 triplet 조합이 $O(N^3)$ → 전수 사용 불가, 샘플링 필요.
  - **Most Triplets are Useless**: 학습이 진행되면 대부분 triplet이 easy라 loss = 0, gradient = 0 → 계산 낭비. 소수의 hard triplet이 학습을 좌우함.
> 보충: FaceNet 원 논문은 $\|f(x)\|_2 = 1$ 제약 하에서 squared L2 거리를 쓰고, hardest 대신 **semi-hard negative**($d_{ap} < d_{an} < d_{ap} + \varepsilon$)를 선택하는 것이 안정적이라고 보고함 (hardest만 쓰면 collapse 위험).

---

## Hard Negative Mining (p.14)

- Most negative samples are already far from the anchor.
  - Easy negatives contribute little to the loss and gradients.
- Hard negatives are close to the anchor but belong to different classes.
  - Hard negatives provide stronger learning signals.
  - Selecting informative negatives improves representation learning.
  - Hard Negative Mining focuses training on the most challenging samples.

**그림**: 제목 "Embedding space of input image $\mathcal{X}$"의 사각 영역. 중앙에 흰 점 **z (Query point)**. 주변에 7개의 파란 점(**Negative samples**, 그중 2개는 아래의 hard negative)이 방사형으로 흩어져 있고 각각 z와 양방향 파란 화살표로 연결됨. z 주위에 점점 옅어지는 **동심원 음영**이 있어 가까울수록 진함. z에 가장 가까운 두 점은 **빨간 원으로 강조된 Hard negatives**. 범례: 흰 점 = Query point, 파란 점 = Negative samples, 빨간 테두리 점 = Hard negatives.

**설명**

- 대부분의 negative는 이미 멀어서(easy) loss/gradient 기여가 거의 0 → 평균을 내면 학습 신호가 희석됨.
- **Hard negative** = anchor와 **embedding 상에서 가깝지만 class는 다른** 샘플. 모델이 헷갈리는 샘플이라 gradient가 크고 정보량이 큼.
- **Hard Negative Mining** = 학습을 가장 어려운 샘플에 집중시키는 전략 (batch 안에서 가장 가까운 negative를 고르는 online mining 등).
- 그림의 동심원은 "query에 가까운 영역일수록 hard"임을 시각화함. 빨간 원 2개만 decision boundary를 위협함.
> 보충: 너무 hard한 것만 쓰면 label noise/outlier에 과적합하거나 collapse가 생길 수 있어 semi-hard나 softmax 가중 방식이 쓰임. 다음 두 슬라이드가 이를 부드럽게 구현하는 방법.

---

## Lifted Structured Loss (p.15)

- Lifted Structured Loss utilizes all the pairwise edges within one training batch for better computational efficiency.

**그림 (좌측)**: 3개 행의 그래프. 각 행에 6개 점 $x_1 \dots x_6$ 가로 나열. **빨간 선 = similar pair, 파란 선 = dissimilar pair**.
- (a) **Contrastive embedding**: $(x_1,x_2)$ 빨강, $(x_3,x_4)$ 파랑, $(x_5,x_6)$ 빨강 — 독립된 3개 pair만 사용.
- (b) **Triplet embedding**: $x_1$–$x_2$ 빨강 & $x_2$–$x_3$ 파랑 / $x_4$–$x_5$ 빨강 & $x_5$–$x_6$ 파랑 — 2개 triplet 사용.
- (c) **Lifted structured embedding**: 6개 점을 위쪽 아치로 모두 연결. 빨간 선($x_1$–$x_2$, $x_3$–$x_4$, $x_5$–$x_6$)과, 나머지 모든 쌍을 잇는 많은 파란 곡선 → batch 안의 **모든 pairwise 관계** 사용.

**수식 (우측)**

$$\mathcal{L}_{\text{struct}} = \frac{1}{2|\mathcal{P}|} \sum_{(i,j)\in\mathcal{P}} \max\big(0,\ \mathcal{L}_{\text{struct}}^{(ij)}\big)^2$$

$$\text{where } \mathcal{L}_{\text{struct}}^{(ij)} = D_{ij} + \max\Big(\max_{(i,k)\in\mathcal{N}} \epsilon - D_{ik},\ \max_{(j,l)\in\mathcal{N}} \epsilon - D_{jl}\Big)$$

$$D_{ij} = |f(\mathbf{x}_i) - f(\mathbf{x}_j)|_2$$

← $D_{ij}$ 노름이 단일 막대로 표기됨 ($\|\cdot\|_2$의 오타로 보임, 원문 유지)

- $\mathcal{P}$ contains the set of positive pairs
- $\mathcal{N}$ is the set of negative pairs
- (슬라이드에서 바깥 `max`와 안쪽 두 `max`가 **빨간색**으로 강조됨)
- ⇒ "not smooth, so relaxed to be"

$$\mathcal{L}_{\text{struct}}^{(ij)} = D_{ij} + \log\Big(\sum_{(i,k)\in\mathcal{N}} \exp(\epsilon - D_{ik}) + \sum_{(j,l)\in\mathcal{N}} \exp(\epsilon - D_{jl})\Big)$$

**출처**: Song et al., 2015, *Deep Metric Learning via Lifted Structured Feature Embedding* (CVPR 2016 게재본으로 흔히 인용됨)

**설명**

- **동기**: contrastive는 pair 하나, triplet은 triplet 하나씩만 보지만 mini-batch에는 이미 $O(N^2)$개의 pairwise 거리가 계산되어 있음. 이를 **batch 내 모든 pair 관계로 확장해 한 번에 활용**함(computational efficiency). 그림 (a)→(b)→(c)가 사용하는 edge 증가를 보여줌.
- 기호: $\mathcal{P}$ = positive pair 집합, $\mathcal{N}$ = negative pair 집합, $D_{ij}$ = embedding 간 Euclidean 거리. positive pair $(i,j)$마다 $i$ 또는 $j$와 연결된 **모든 negative $k$, $l$** 과 비교함.
- 수식 직관: $\mathcal{L}^{(ij)} = D_{ij} + \max(\epsilon - D_{ik},\ \epsilon - D_{jl})$.
  - positive 거리 $D_{ij}$를 줄이면서, 가장 가까운(= $\epsilon - D$가 가장 큰) negative와의 거리가 $\epsilon$ 이상이 되도록 함.
  - 즉 **batch 내 hardest negative를 자동 선택하는 hard negative mining이 loss에 내장**됨 (빨간 `max`).
  - $\frac{1}{2|\mathcal{P}|}$: positive pair 수로 평균, $\frac12$은 제곱 hinge 미분 편의용.
- **문제**: `max`는 non-smooth → 매 step에서 positive pair당 (i쪽·j쪽 각각) **가장 hard한 negative 하나에만 gradient**가 흘러 수렴/안정성이 나쁨.
- **해결**: `max`를 **log-sum-exp**로 완화(relax). 모든 negative의 $\exp(\epsilon - D)$를 합하고 $\log$를 취함 → smooth하며 hard negative에 큰 가중을 주면서 다른 negative도 gradient를 받음. (p.16에서 시각적으로 설명.)
> 보충: 원 논문은 relaxed 버전을 $\max(0,\cdot)^2$ 안에 넣어 사용함. $\log\sum\exp$는 이후 InfoNCE/N-pair loss의 softmax 형태와 같은 계열임.

---

## Why Replace Max? (p.16)

**그림 (전체가 하나의 인포그래픽)**

- **맨 위 축**: "Negative samples (distances)" 라벨 박스 + 수평 화살표 축. 4개의 점이 $0.2$(초록, **easy**), $0.4$(초록), $0.6$(주황), $0.8$(빨강, **hard**). 축 오른쪽 라벨: "larger = closer (harder negative)" → 값 $s_i$가 클수록 anchor와 가까운 어려운 negative.
- **3개 행 (각 행: 수식 + 점별 화살표 + 막대그래프 + 요약 박스)**
  1. **Max** (빨강) — $\max_i s_i$. 0.8에만 가중치 1, 나머지 0,0,0. 막대그래프(Gradient / Weight)는 0.8에서만 높이 1인 빨간 막대. 요약: "Uses only the hardest negative. **Sparse gradient**".
  2. **Sum (Average)** (파랑) — $\frac1N \sum_i s_i$. 모든 점에 화살표 1,1,1,1. 막대는 4개가 비슷한 높이(약 0.55). 요약: "Uses all negatives equally. **Not focused on hard negatives**".
  3. **Log-Sum-Exp (Soft Max)** (초록) — $\log\sum_i e^{s_i}$. 점별 화살표에 **0.07, 0.13, 0.24, 0.57**. 막대(softmax weights)는 0.2→0.8로 갈수록 커짐. 요약: "Uses all negatives, but weights increase exponentially with hardness. **Dense and focused gradient**".
- **하단 Takeaway 박스(노란색)**: "log-sum-exp is a smooth approximation of max. It performs hard negative mining while keeping gradients dense and stable." 우측 수식: $\max(s_i) \approx \log\sum_i e^{s_i}$ (라벨: hard max ≈ soft max).

**설명**

- 세 aggregation 비교 = 여러 negative를 하나의 scalar로 요약하는 방식. 핵심은 $s_i$로 미분했을 때 각 negative가 받는 **gradient 가중치**임.
  1. **Max**: $\partial \max_i s_i / \partial s_j = \mathbb{1}[j = \arg\max]$ → hardest 하나만 gradient 1 (**sparse**). hard에 집중하는 장점은 있으나 정보 낭비, outlier/noisy negative에 민감, 불연속이라 불안정.
  2. **Sum / Average**: 모든 negative에 **균등**($1/N$) gradient. 정보는 다 쓰지만 easy negative가 대부분이라 hard 신호가 희석됨.
  3. **Log-Sum-Exp**: $\partial\,\text{LSE}/\partial s_j = \dfrac{e^{s_j}}{\sum_i e^{s_i}}$ = **softmax weight**. 모든 negative가 gradient를 받되(**dense**) 어려울수록 지수적으로 큰 가중 → "soft한 hard negative mining".
- $\text{LSE}(s)$는 $\max_i s_i$의 **smooth approximation**: $\max_i s_i \le \text{LSE}(s) \le \max_i s_i + \log N$.
- 이 관점이 p.15의 relaxation을 정당화하고, **softmax/InfoNCE 계열 loss(+temperature $\tau$)** 가 hard negative를 자동으로 강조하는 이유와 직결됨.
> 보충: $\tau\log\sum e^{s_i/\tau}$에서 $\tau \to 0$이면 $\max_i s_i$에 수렴하고(gradient 가중치 $\text{softmax}(s/\tau)$가 one-hot), $\tau \to \infty$이면 gradient 가중치가 균등($1/N$)해져 평균처럼 동작함. temperature가 "얼마나 hard-focused한가"를 조절함.

> 슬라이드 의심 점 ← (1) Sum 행의 막대 높이가 약 0.55로 그려져 있으나 평균이면 가중치는 $1/N = 0.25$여야 함. 크기 비교용 도식으로만 해석. (2) Log-Sum-Exp 행의 0.07/0.13/0.24/0.57은 $s=(0.2,0.4,0.6,0.8)$에 그대로 softmax를 취한 값(약 0.18/0.22/0.27/0.33)과 **일치하지 않음**. 더 큰 스케일(작은 temperature)을 가정한 예시값으로 보임 ← 원문 유지.

---

## FIGURES

- p.11: Contrastive loss의 Intuitive Goal / 수식 박스 (positive는 당기고 negative는 margin $\varepsilon$까지 밂) — 중요도 중
- p.12: FaceNet 파이프라인(Batch → Deep Architecture → L2 → Embedding → Triplet Loss) + LEARNING 전후 Anchor/Positive/Negative 배치 변화 — 중요도 상
- p.14: Hard negative 시각화 (query z 주변 동심원, 빨간 원 hard negatives) — 중요도 중
- p.15: (a) contrastive / (b) triplet / (c) lifted structured embedding의 edge 사용 범위 비교 그래프 + loss 수식 — 중요도 상
- p.16: Max vs Sum vs Log-Sum-Exp의 gradient/weight 비교 인포그래픽 — 중요도 상

## EXAM POINTS

1. **Contrastive loss 수식**: positive는 $\|f(x_i)-f(x_j)\|_2^2$, negative는 $\max(0,\varepsilon-\|\cdot\|_2)^2$. margin $\varepsilon$의 역할과 $d\ge\varepsilon$일 때 gradient가 0이 되는 이유(= inactive negative).
2. **Contrastive vs Triplet**: contrastive는 pair별 절대 거리 목표, triplet은 anchor–positive / anchor–negative **상대 거리 차**가 margin 이상이라는 relative ordering 제약. 왜 triplet이 intra-class 변동에 더 유연한지.
3. **Triplet loss 수식**: $\max(0, \|f(x)-f(x^+)\|^2 - \|f(x)-f(x^-)\|^2 + \varepsilon)$ 와 FaceNet 구성(L2 normalization 포함), Schroff et al. 2015.
4. **Triplet의 두 문제**: (1) Triplet explosion $O(N^3)$, (2) 대부분 triplet이 easy라 loss=0 → 해결책으로 hard/semi-hard negative mining.
5. **Hard negative 정의와 필요성**: anchor와 embedding 상 가깝지만 다른 class인 샘플, easy negative의 gradient가 거의 없는 이유.
6. **Lifted Structured Loss**: batch 내 모든 pairwise edge($O(N^2)$) 활용, $\mathcal{P}$/$\mathcal{N}$ 정의, loss에 내장된 hardest negative `max`, non-smooth `max`를 log-sum-exp로 relax하는 이유 (Song et al.).
7. **Why Replace Max? (why 질문)**: Max = sparse gradient, Sum = 균등 가중(hard에 비집중), Log-Sum-Exp = softmax 가중의 dense & focused gradient. $\partial\text{LSE}/\partial s_j$ = softmax weight 유도.
8. **목표 부등식**: $\text{score}(f(x),f(x^+)) \gg \text{score}(f(x),f(x^-))$ 와 supervised/unsupervised에서 positive를 정의하는 방식 차이.

## VERIFIER REPORT

- p.14 그림 설명: 파란 점 개수 6개 -> 7개(일반 5 + hard negative 2)로 수정.
- p.15 설명: max의 gradient가 "단 하나의 negative"라는 표현을 positive pair당 i쪽/j쪽 각각 hardest negative 하나로 정정.
- p.16 보충: temperature 식을 $\tau\log\sum e^{s_i/\tau}$로 바로잡고, $\tau\to\infty$는 값이 아니라 gradient 가중치가 균등해지는 것으로 정정.
