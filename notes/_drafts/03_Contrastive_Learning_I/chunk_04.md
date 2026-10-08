<!-- DRAFT | lecture: 03_Contrastive_Learning_I | chunk: 04 | pages 25–32 -->
<!-- VERIFIED | fixes: 3 | verifier: sonnet -->

## InfoNCE (p.25)

- The InfoNCE loss uses **categorical cross-entropy loss** to identify the positive sample amongst a set of unrelated noise samples.
- 인용 박스 (lilianweng 블로그 캡처): "The InfoNCE loss optimizes the negative log probability of classifying the positive sample correctly:"

$$\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\left[\log \frac{f(\mathbf{x},\mathbf{c})}{\sum_{\mathbf{x}'\in X} f(\mathbf{x}',\mathbf{c})}\right]$$

- 출처 표기: Oord et al., 2018, Representation Learning with Contrastive Predictive Coding / https://lilianweng.github.io/

**설명**
- 기호: $\mathbf{c}$ = context(anchor 역할), $\mathbf{x}$ = context와 짝인 positive sample, $X=\{\mathbf{x}_1,\dots,\mathbf{x}_N\}$ = positive 1개 + noise(negative) $N-1$개로 이뤄진 후보 집합, $f(\mathbf{x},\mathbf{c})$ = 둘이 얼마나 잘 맞는지 나타내는 **양의 score 함수**(exp 꼴이라 양수).
- 분수 $\frac{f(\mathbf{x},\mathbf{c})}{\sum_{\mathbf{x}'} f(\mathbf{x}',\mathbf{c})}$ 는 **N개 후보 중 정답(positive)을 고를 확률**인 softmax 출력과 동일한 형태임. 여기에 $-\log$ 를 씌우면 **N-way 분류의 categorical cross-entropy**가 됨 (정답 클래스 = positive의 인덱스).
- 즉 contrastive learning을 "binary(같다/다르다) 판별"이 아니라 **"N개 중에서 positive 찾기"라는 multi-class 분류 문제**로 바꾼 것이 핵심 아이디어임. negative가 많을수록 문제가 어려워지고, 일반적으로(경험적으로) 더 좋은 표현을 학습하는 경향이 있음 (항상 보장되는 것은 아님).
- 분모에 positive 자기 자신도 포함됨(합이 $X$ 전체) → 확률로 정규화되려면 필요한 설계.
> 보충: 원 논문(CPC, van den Oord et al., 2018)에서 $f_k(x_{t+k}, c_t)=\exp(z_{t+k}^\top W_k c_t)$ 로 두는 log-bilinear 모델을 씀. 이 loss를 최소화하면 $I(x;c)\ge \log N - \mathcal{L}_{\text{InfoNCE}}$ 라는 **mutual information 하한**이 성립함 (다음 슬라이드들의 동기).

## N-pair Loss (p.26)

$$L = -\mathbb{E}_X\left[\log \frac{\exp\big(s(f(x), f(x^+))\big)}{\exp\big(s(f(x), f(x^+))\big) + \sum_{j=1}^{N-1}\exp\big(s(f(x), f(x_j^-))\big)}\right]$$

- 수식 주석(그림): 분자와 분모 첫 항(초록 밑줄) = **score for the positive pair**, 분모의 $\sum_{j=1}^{N-1}$ 항(빨간 밑줄) = **score for the N-1 negative pairs**.
- 텍스트: "This seems familiar …" / "Cross entropy loss for a N-way softmax classifier!" / "I.e., learn to find the positive sample from the N samples"
- 하단 메모: "Anchor (x) can be regarded as context (c)"

**설명**
- 기호: $x$ = anchor, $x^+$ = positive, $x_j^-$ = negative ($N-1$개), $f(\cdot)$ = encoder, $s(\cdot,\cdot)$ = similarity (dot product, cosine 등).
- **Triplet loss가 negative 1개만 쓰던 것**과 달리, N-pair loss는 anchor 하나당 positive 1개 + negative $N-1$개를 한꺼번에 비교함 → 매 step마다 더 많은 negative 정보를 활용해 학습 효율이 좋음.
- 형태가 **softmax cross-entropy**와 같음: 정답 클래스 = positive, 나머지 $N-1$개 = 오답 클래스. 따라서 p.25의 InfoNCE와 **사실상 같은 식**이고, 슬라이드가 "Anchor (x) = context (c)" 라고 적은 이유는 InfoNCE의 $f(\mathbf{x}',\mathbf{c})\leftrightarrow \exp(s(f(x),f(x')))$ (후보 $x'\in\{x^+, x_1^-,\dots,x_{N-1}^-\}$, positive 항은 $x'=x^+$) 로 대응시키기 위함임.
> 보충: N-pair loss는 Sohn, 2016 (Improved Deep Metric Learning with Multi-class N-pair Loss Objective)에서 제안됨. 이후 SimCLR의 NT-Xent도 같은 구조에 temperature를 더한 것.

## InfoNCE – Mutual information optimization (p.27)

- Mutual information optimization

$$I(\mathbf{x};\mathbf{c}) = \sum_{\mathbf{x},\mathbf{c}} p(\mathbf{x},\mathbf{c})\log\frac{p(\mathbf{x},\mathbf{c})}{p(\mathbf{x})p(\mathbf{c})}$$

- 출처: Oord et al., 2018 / lilianweng.github.io

**설명**
- **Mutual information(MI)** 정의 식임. 결합분포 $p(\mathbf{x},\mathbf{c})$ 와 독립 가정 시의 분포 $p(\mathbf{x})p(\mathbf{c})$ 사이의 **KL divergence**: $I(\mathbf{x};\mathbf{c})=D_{KL}\big(p(\mathbf{x},\mathbf{c})\,\|\,p(\mathbf{x})p(\mathbf{c})\big)$.
- 직관: $\mathbf{x}$ 와 $\mathbf{c}$ 가 독립이면 로그 안의 비가 1이라 $I=0$, 서로 많은 정보를 공유할수록 값이 커짐. 즉 **"c를 알면 x에 대해 얼마나 더 알게 되는가"** 의 척도.
- 이 슬라이드는 다음 슬라이드(p.28)에서 식을 변형하기 위한 출발점임.

## InfoNCE – Mutual information optimization (p.28)

- Mutual information optimization

$$I(\mathbf{x};\mathbf{c}) = \sum_{\mathbf{x},\mathbf{c}} p(\mathbf{x},\mathbf{c})\log\frac{p(\mathbf{x},\mathbf{c})}{p(\mathbf{x})p(\mathbf{c})} = \sum_{\mathbf{x},\mathbf{c}} p(\mathbf{x},\mathbf{c})\log\frac{p(\mathbf{x}|\mathbf{c})}{p(\mathbf{x})}$$

- 그림: 우변에서 $p(\mathbf{x}|\mathbf{c})$ 가 파란색으로 강조되고, 분수 아래로 아래쪽 화살표(⇩)가 나가 $f(x,c)$ 를 가리킴 → 이 density ratio를 $f(x,c)$ 로 모델링한다는 의미.
- 텍스트:
  - Learning a good f(x,c)
  - → Estimating a good density ratio
  - → Increasing Mutual Information
  - InfoNCE is not simply learned to match positives,
  - but rather learned in a direction that increases the shared information between x and c.

**설명**
- 유도: $p(\mathbf{x},\mathbf{c})=p(\mathbf{x}|\mathbf{c})p(\mathbf{c})$ 이므로 $\frac{p(\mathbf{x},\mathbf{c})}{p(\mathbf{x})p(\mathbf{c})}=\frac{p(\mathbf{x}|\mathbf{c})}{p(\mathbf{x})}$ 로 정리됨.
- **핵심**: $f(\mathbf{x},\mathbf{c})\propto \frac{p(\mathbf{x}|\mathbf{c})}{p(\mathbf{x})}$ (**density ratio**)가 되도록 $f$ 를 학습하는 것이 InfoNCE의 목표. 이 비는 "c가 주어졌을 때 x가 나올 확률이 prior보다 얼마나 큰가"이므로 **c와 연관된(positive) x에서 크고, 무관한(noise) x에서 작음**.
- 따라서 "좋은 $f$ 학습 → 좋은 density ratio 추정 → MI 증가"의 흐름이 성립. InfoNCE는 단순히 positive끼리 가깝게 만드는 matching이 아니라 **x와 c의 공유 정보를 늘리는 방향**으로 학습된다는 것이 슬라이드의 메시지.
- 왜 density ratio를 쓰는가: $p(\mathbf{x}|\mathbf{c})$ 를 직접 생성모델로 추정하는 것은 고차원(이미지 등)에서 어렵고 불필요한 세부까지 모델링해야 함. ratio만 추정하면 분류 문제(NCE)로 풀 수 있음.
> 보충: CPC 논문에서 optimal $f$ 는 $p(x|c)/p(x)$ 에 비례하며, 이때 $I(x;c)\ge\log N-\mathcal{L}_{\text{InfoNCE}}$ 가 증명됨. 이 하한은 $N$ 이 클수록 tight해지므로 **negative 수를 늘리는 것이 중요**한 이유가 됨 (단, $\log N$ 에서 saturate).

## Soft-Nearest Neighbors Loss (p.29)

- Soft-Nearest Neighbors Loss extends it to include **multiple positive samples**.
- 인용 박스: "Given a batch of samples, $\{\mathbf{x}_i,y_i\}_{i=1}^{B}$ where $y_i$ is the class label of $\mathbf{x}_i$ and a function $f(\cdot,\cdot)$ for measuring similarity between two inputs, the soft nearest neighbor loss at temperature $\tau$ is defined as:"

$$\mathcal{L}_{\text{snn}} = -\frac{1}{B}\sum_{i=1}^{B}\log\frac{\sum_{i\neq j,\,y_i=y_j,\,j=1,\dots,B}\exp\big(-f(\mathbf{x}_i,\mathbf{x}_j)/\tau\big)}{\sum_{i\neq k,\,k=1,\dots,B}\exp\big(-f(\mathbf{x}_i,\mathbf{x}_k)/\tau\big)}$$

- 그림 주석: 분자의 합 조건 부분(빨간 박스) = "Neighbors of the same class", 분모의 합 조건 부분(빨간 박스) = "All neighbors".
- "The temperature is used for tuning how concentrated the features are in the representation space."
- 출처: Salakhutdinov et al., 2007, Learning a Nonlinear Embedding by Preserving Class Neighbourhood Structure; Frosst et al., 2019, Analyzing and Improving Representations with the Soft Nearest Neighbor Loss

**설명**
- 기호: $B$ = batch size, $y_i$ = class label, $\tau$ = temperature, 분자 = **anchor $i$ 와 같은 클래스인 모든 샘플** $j$ 에 대한 유사도 합, 분모 = **자기 자신을 제외한 batch 내 모든 샘플** $k$ 에 대한 유사도 합.
- N-pair/InfoNCE는 positive가 1개였지만, 이 loss는 **label을 이용해 같은 클래스 샘플 전부를 positive로 취급**(multiple positives). 분자가 positive들의 합이므로, "anchor의 이웃을 soft하게 뽑았을 때 **같은 클래스일 확률**"의 $-\log$ 가 됨 → 이웃 중 같은 클래스 비율을 높이는 방향으로 학습.
- **Temperature $\tau$**: 작으면 가장 가까운 소수의 이웃만 지배적(feature가 클래스별로 뭉침), 크면 멀리 있는 샘플도 기여해 분포가 퍼짐. 슬라이드 설명대로 representation space에서 feature가 **얼마나 concentrated 되는지를 조절**함.
- ← 슬라이드는 "$f$ = similarity" 라 쓰지만 식에는 $\exp(-f/\tau)$ 가 있어 부호상 **$f$ 는 거리(distance, 예: squared Euclidean)로 쓰이는 것**이 맞음. 원문 유지, 표현 불일치 가능성 플래그.
> 보충: 원 아이디어는 Salakhutdinov & Hinton (2007)의 NCA 계열이고, Frosst, Papernot & Hinton (2019)이 이를 deep network의 hidden layer에서 클래스 얽힘(entanglement)을 재는 loss로 재조명함. Supervised Contrastive Learning(SupCon)의 전신 격 loss임.

## Common Setup for Contrastive Learning (p.30)

- (섹션 구분 슬라이드) Common Setup for Contrastive Learning

## Common Setup for Modern Contrastive Learning (p.31)

- 그림 (3개 열 + 우측 패널):
  - **1. Anchor** — $\mathbf{x}$ (query): 잔디 위 개 사진 1장 → Encoder $f(\cdot)$ (파란 사다리꼴) → 파란 벡터(5칸) $\mathbf{z}=f(\mathbf{x})$.
  - **2. Positive** — $\mathbf{x}^+$ (another view of $\mathbf{x}$): 같은 개의 다른 view 이미지 2장(+ "…") → 각각 Encoder $f(\cdot)$ (초록) → 초록 벡터 $\mathbf{z}_1^+,\mathbf{z}_2^+,\dots$
  - **3. Negatives** — $\mathbf{x}_1^-,\mathbf{x}_2^-,\mathbf{x}_3^-,\dots,\mathbf{x}_N^-$: 허스키, 자동차, 비행기, … 자전거 → 각각 Encoder $f(\cdot)$ (빨강) → 빨간 벡터 $\mathbf{z}_1^-,\mathbf{z}_2^-,\mathbf{z}_3^-,\dots,\mathbf{z}_N^-$
  - 모든 $\mathbf{z}$ 에서 점선 화살표(파랑/초록/빨강)가 아래 박스로 모임 → **Similarity (e.g., cosine)**: $s(a,b)=\frac{\mathbf{z}_a^\top \mathbf{z}_b}{\|\mathbf{z}_a\|\|\mathbf{z}_b\|}$
  - 그 아래 보라색 박스 → **Contrastive Loss (InfoNCE / NT-Xent)**:

$$\mathcal{L}=-\log\frac{\exp\big(s(\mathbf{z},\mathbf{z}^+)/\tau\big)}{\exp\big(s(\mathbf{z},\mathbf{z}^+)/\tau\big)+\sum_{i=1}^{N}\exp\big(s(\mathbf{z},\mathbf{z}_i^-)/\tau\big)}$$

  - 옆에 "→ $\tau$ : temperature"
- 우측 **Key Components** 패널:
  1. Anchor (Query): A sample $\mathbf{x}$ from the dataset.
  2. Positive(s): Different augmented views of the same sample $\mathbf{x}$.
  3. Negatives: Other samples in the batch (treated as negatives).
  4. Encoder $f(\cdot)$: Maps inputs to representations (often L2-normalized).
  5. Similarity: Measures closeness between representations.
  6. Contrastive Loss: Pulls positives together and pushes negatives apart.
- **Legend**: 파란 실선 = Anchor path, 초록 실선 = Positive path, 빨간 실선 = Negative path, 점선 = Similarity computation.

**설명**
- 앞의 InfoNCE / N-pair 식을 **현대 contrastive learning(SimCLR, MoCo 등)의 공통 파이프라인**으로 정리한 슬라이드. 흐름은 (입력 샘플링) → **같은 encoder $f$ 로 인코딩** → similarity 계산 → loss.
- **Anchor/Positive/Negative**: positive = anchor의 **augmented view**(label 없이 만든 짝), negative = **batch 내 다른 샘플들**. 모든 branch가 같은 $f(\cdot)$ (weight 공유)를 통과함.
- **L2-normalize + cosine similarity**: 벡터 크기 영향을 없애고 방향만 비교해 학습 안정화. $s\in[-1,1]$ 이므로 그대로 exp하면 gradient가 약해, **temperature $\tau$** 로 나눠 분포의 sharpness 조절 (작은 $\tau$ → hard negative에 더 큰 가중).
- p.26 N-pair loss와 비교: 구조는 동일하고 **$s/\tau$ 로 temperature 추가**, 분모의 negative 합이 $N-1$ 대신 $N$ 으로 표기됨 ← p.26은 $N-1$, p.31은 $N$ 으로 negative 개수 표기가 다름 (총 후보 수 정의 차이일 뿐 구조는 동일).
- Loss는 positive의 logit을 올리고(= pull) negative들의 logit을 내리는(= push) 효과를 한 식에서 동시에 냄.
> 보충: "NT-Xent" = Normalized Temperature-scaled Cross Entropy (SimCLR, Chen et al., 2020).

## Positive Pair Construction (p.32)

- Contrastive learning incorporates multiple positive and negative samples.
- 그림 (좌우 두 패널, 가운데 연결 박스):
  - **좌: With Labels (or Semantics)** — "Same class ⇒ **Positive pair**". Dog 행: 골든리트리버 사진 3장 + "…", 위쪽에 파란 점선 화살표가 서로를 연결. Car 행: 흰/검정/회색 자동차 3장 + "…", 빨간 점선 화살표로 연결. 캡션: "Same-class samples are treated as positives."
  - **가운데**: 보라색 박스 "We need positive pairs without labels" (좌→우 화살표).
  - **우: Without Labels (Unsupervised Data)** — "How?" → "Create different views of the same sample using **data augmentation**!" Original image $\mathbf{x}$ (산+호수 풍경) → 초록 화살표 → 서로 다르게 augment된 $\mathbf{v}_1,\mathbf{v}_2,\dots,\mathbf{v}_K$ ("Different augmented views (positive pairs)"), 초록 점선 화살표로 서로 연결. 캡션: "Different images in the batch ⇒ **negative pairs**".
- 불릿 (positive pair distribution $p_{\text{pos}}$ 의 조건):
  - Symmetry: $\forall \mathbf{x},\mathbf{x}^+,\ p_{\text{pos}}(\mathbf{x},\mathbf{x}^+)=p_{\text{pos}}(\mathbf{x}^+,\mathbf{x})$
  - Matching marginal: $\forall \mathbf{x},\ \int p_{\text{pos}}(\mathbf{x},\mathbf{x}^+)\,d\mathbf{x}^+=p_{\text{data}}(\mathbf{x})$

**설명**
- **Positive pair를 어떻게 만드는가**가 contrastive learning의 핵심 설계 요소임.
  - **Supervised(label 있음)**: 같은 class면 positive (p.29 Soft-NN, SupCon 방식). 한 anchor에 positive가 다수.
  - **Unsupervised(label 없음)**: 같은 이미지에 **서로 다른 data augmentation**을 적용해 만든 view끼리 positive, batch 내 다른 이미지는 negative. 이 경우 "무엇이 invariant해야 하는가"를 augmentation 선택이 정의함.
- 두 수식은 $p_{\text{pos}}(\mathbf{x},\mathbf{x}^+)$ (positive pair를 뽑는 분포)가 만족해야 할 조건:
  - **Symmetry**: $(\mathbf{x},\mathbf{x}^+)$ 와 $(\mathbf{x}^+,\mathbf{x})$ 가 같은 확률 → anchor/positive 역할이 구분되지 않음 (augmentation 두 view는 대칭).
  - **Matching marginal**: positive쪽을 적분해 없앤 marginal이 원래 데이터 분포 $p_{\text{data}}$ 와 같아야 함 → positive 샘플링이 데이터 분포를 왜곡하지 않고, 모든 샘플이 anchor로서 데이터 분포대로 나타남.
> 보충: 이 두 조건은 Wang & Isola, 2020 ("Understanding Contrastive Representation Learning through Alignment and Uniformity") 계열에서 positive pair 분포 $p_{\text{pos}}$ 에 두는 가정과 같은 형태임 (슬라이드에는 출처 미표기). 같은 논문에서 loss를 **alignment**(positive 가깝게)와 **uniformity**(hypersphere 상 균일 분포)로 분해함.
> 보충: 우측 패널의 "different images ⇒ negative" 는 같은 클래스인 다른 이미지가 negative가 되는 **false negative** 문제를 내포함 (다음 슬라이드 p.33 질문 "Are positive and negative pairs always constructed correctly?" 로 이어짐).

## FIGURES

- p.31: Anchor/Positive/Negative → Encoder → Similarity → Contrastive Loss 전체 파이프라인 다이어그램 + Key Components 패널 (중요도: 상)
- p.32: With labels vs without labels(augmentation)로 positive pair 만드는 방법 비교 그림 (중요도: 상)
- p.26: N-pair loss 수식의 positive score / negative score 색 주석 (중요도: 중)
- p.28: MI → density ratio $p(x|c)/p(x)$ → $f(x,c)$ 유도 도식 (중요도: 중)
- p.29: Soft-NN loss 수식의 "same class neighbors / all neighbors" 주석 (중요도: 중)

## EXAM POINTS

1. **InfoNCE의 정의와 해석**: $-\mathbb{E}\log\frac{f(x,c)}{\sum_{x'\in X}f(x',c)}$ = N개 후보 중 positive를 고르는 **N-way softmax categorical cross-entropy**. (수식 쓰기 + 설명)
2. **N-pair loss vs InfoNCE vs NT-Xent**: 구조 동일(N-way softmax CE), 차이는 표기/세부: anchor=context 대응, negative 개수 표기($N-1$ in p.26, $N$ in p.31 — 본질적 차이 아님), NT-Xent는 temperature $\tau$ 및 (L2-normalized) cosine similarity 사용.
3. **InfoNCE와 mutual information의 관계**: $I(x;c)=\sum p(x,c)\log\frac{p(x|c)}{p(x)}$, optimal $f\propto p(x|c)/p(x)$ (density ratio), $I(x;c)\ge\log N-\mathcal{L}_{\text{InfoNCE}}$ (보충). "단순 positive matching이 아니라 shared information 증가" 서술형.
4. **Why density ratio?** $p(x|c)$ 를 직접 모델링하지 않고 비만 추정하면 되는 이유 (고차원 생성 모델링 회피, 분류 문제로 환원).
5. **Soft-Nearest Neighbors loss**: multiple positives(같은 클래스 전부) 처리 방식, 분자/분모의 합 범위, temperature가 feature 집중도에 주는 영향.
6. **Contrastive learning의 공통 구성요소 6가지**: anchor, positive, negative, encoder, similarity, loss + L2-normalization과 temperature의 역할.
7. **Positive pair 구성**: label 있을 때(같은 class) vs 없을 때(augmentation으로 만든 view), 배치 내 다른 샘플을 negative로 쓰는 이유와 한계(false negative).
8. **$p_{\text{pos}}$ 의 두 조건**: symmetry와 matching marginal의 수식과 의미.

## VERIFIER REPORT

- p.25 설명: "negative가 많을수록 더 좋은 표현" 단정을 경향(경험적)으로 완화.
- p.26 설명: InfoNCE f(x',c) 와 exp(s(f(x),f(x'))) 대응을 후보 x' 전체 기준으로 정정 (기존엔 positive 항만 대응).
- EXAM POINT 2: "negative 개수 N-1"을 차이점으로 적은 부분을 표기 차이일 뿐이라고 정정.
- 그 외 수식/그림 설명/인용(Oord 2018, Sohn 2016, Salakhutdinov 2007, Frosst 2019, Chen 2020, Wang & Isola 2020)은 슬라이드와 일치, 누락 없음.
