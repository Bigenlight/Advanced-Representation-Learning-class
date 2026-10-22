<!-- DRAFT | lecture: 07_Semi_Supervised_Learning_II | chunk: 10 | pages 76–85 -->
<!-- VERIFIED | fixes: 5 | verifier: sonnet -->

## Section title: Are low-confidence unlabeled samples useless? (p.76)

슬라이드 텍스트:
- "Are low-confidence unlabeled samples useless?" (섹션 구분 슬라이드)

**설명**
- 여기서부터 **AllMatch** 파트. 질문 자체가 핵심 동기: FixMatch류 threshold 방식은 confidence가 낮은 unlabeled sample을 **버리는데**, 정말 쓸모없는가?
- 06 강의의 consistency regularization (UDA 등) + 이번 강의 앞부분의 pseudo-labeling (FixMatch, FreeMatch 등)의 **다음 단계**로 위치.

---

## AllMatch: Reference (p.77)

슬라이드 텍스트:
- Wu, et. al., IJCAI 2024, *AllMatch: Exploiting All Unlabeled Data for Semi-Supervised Learning*

**설명**
- 이 파트의 근거 논문. 제목이 곧 목표: **"모든(All) unlabeled data를 활용"**.
- 시험에서 방법 이름 ↔ 핵심 아이디어 매칭으로 나올 수 있으니 "AllMatch = **all** unlabeled samples 사용" 기억.
> 보충: IJCAI 2024 = International Joint Conference on Artificial Intelligence. 슬라이드 표기를 그대로 따름.

---

## Are Low-Confidence Pseudo-Labels Completely Useless? (p.78)

슬라이드 텍스트:
- Threshold-based methods divide unlabeled samples into two groups.
  - High confidence
    - Use the top-1 prediction as a pseudo-label.
  - Low confidence
    - Discard the sample.
    - Provide no training signal.
- 그림: 두 개의 line plot (x축 iterations (K) 0~1000+, y축 Accuracy). 범례: FreeMatch(파랑), SoftMatch(주황), FlexMatch(초록), FixMatch(분홍/연보라).
  - (a) **Dropped pseudo-label acc.** — y축 0~70%. 모든 방법이 초반 급상승 후 약 55~60%에서 plateau, **SoftMatch만 약 62%→67%까지 상승**. (FreeMatch ≈ 60%, FlexMatch/FixMatch ≈ 55~58%.)
  - (b) **Top-5 acc of pseudo-labels.** — y축 70~100%. 모든 방법이 매우 빠르게 **약 99~100%에 수렴**.
- 그림 옆 설명: "The proportion of pseudo-labels discarded (falling below the threshold) that are actually correct in their top-1/5 prediction."
- 결론 문장: **Low confidence does not mean no semantic information.**

<img src="../../../assets/lecture07/p078_dropped_pl_acc.png" width="600">

**설명**
- **Threshold-based 방법 (FixMatch, FlexMatch, FreeMatch, SoftMatch …)** 의 공통 구조: confidence $\max_c p(c|x)$ 가 threshold $\tau$ 이상이면 top-1을 **hard pseudo-label**로 쓰고, 미만이면 **loss에서 완전히 제외** (mask = 0). 즉 **low-confidence sample은 training signal이 0**.
- 그림 (a) 읽는 법: threshold 때문에 **버려진 sample들만** 모아서 top-1 prediction의 정확도를 잼 → 약 **55~67%**. 버려진 것들인데도 **절반 넘게 top-1이 맞음** ← 버리기엔 아까운 신호.
- 그림 (b): 같은 버려진 sample들의 **top-5 안에 정답이 들어있는 비율**은 **거의 100%**. 즉 **"정답 클래스가 어디 있는지"는 거의 항상 상위 5개 후보 안에 있음**.
- 직관: top-1 confidence가 낮다 = "단일 클래스로 못 찍겠다" 일 뿐, **"정답이 될 만한 후보 집합"에 대한 정보는 풍부**함. 그래서 **"하나의 정답"이 아니라 "후보 집합 vs 나머지"** 로 supervision을 바꾸자는 것이 AllMatch의 발상 (p.79~).
- ← 슬라이드에 dataset/백본 명시 없음 (그림은 논문 figure 인용). 숫자는 plot에서 읽은 **근사치**.
> 보충: 이 분석은 논문의 motivation figure로, 보통 CIFAR-100 계열 설정으로 추정되나 슬라이드엔 명시 없음 → 시험에서 dataset까지 묻기는 어려움.

---

## From FreeMatch to AllMatch: Two Components (p.79)

슬라이드 텍스트:
- AllMatch contains two components.
  - **Class-Specific Adaptive Threshold, CAT**
    - Estimates global learning progress
    - Estimates class-specific learning progress
    - Selects reliable top-1 pseudo-labels
  - **Binary Classification Consistency, BCC**
    - Divides classes into candidate and negative groups
    - Uses low-confidence samples
    - Enforces the same division across augmentations
- AllMatch aims to provide learning signals for all unlabeled samples by combining CAT and BCC.
- CAT improves pseudo-label selection. BCC learns from the remaining samples.

**설명**
- AllMatch = **CAT + BCC**, 역할 분담이 명확:
  - **CAT** : *high-confidence* 쪽. "어떤 sample의 top-1을 믿을지"를 **더 잘 고르는** threshold (global + class-specific). → 기존 FixMatch 식 hard pseudo-label loss $\mathcal{L}_u$ 를 담당.
  - **BCC** : *CAT를 통과 못 한 나머지(low-confidence)* sample을 **버리지 않고** 학습. 클래스를 **candidate (positive) vs negative** 두 그룹으로 나눠 **binary 예측**으로 변환 → **weak view와 strong view에서 그 division이 같도록** consistency 학습.
- 정리: **"CAT improves pseudo-label selection. BCC learns from the remaining samples."** ← 한 문장으로 외워둘 것.
- BCC의 "across augmentations" = weak ↔ strong augmentation 간 consistency (FixMatch와 동일한 weak/strong 구조, p.83 그림에서 확인).
> 보충: 논문 전체 loss는 대략 supervised loss + CAT 기반 pseudo-label loss $\mathcal{L}_u$ + BCC loss $\mathcal{L}_b$ 의 합 구조 (p.81, p.83 그림에 $\mathcal{L}_u$, $\mathcal{L}_b$ 표기). 정확한 가중치는 이 chunk 범위 밖.

---

## FreeMatch vs AllMatch (p.80)

슬라이드 텍스트 (표):

| FreeMatch | AllMatch |
|---|---|
| Adaptive global threshold | Adaptive global threshold |
| Prediction-based local threshold | Classifier-weight-based local threshold |
| Discards low-confidence samples | Uses every unlabeled sample |
| Top-1 pseudo-label | Candidate versus negative classes |

**설명**
- 4행 비교, 각 행의 변화:
  1. **Global threshold**: 둘 다 **adaptive (EMA 기반)** — 동일. ← "같은 행"이라 함정으로 "AllMatch가 global threshold를 새로 바꿨다"는 틀린 지문 가능.
  2. **Local threshold**: FreeMatch는 **class별 평균 예측 확률(prediction-based)** 로 class-specific 조정, AllMatch는 **classifier weight norm(classifier-weight-based)** 으로 조정 ← **핵심 차이** (p.81 하단 "The main difference from FreeMatch lies in the local adjustment", p.82).
  3. **Low-confidence 처리**: FreeMatch는 버림 → AllMatch는 **모든 unlabeled sample 사용**.
  4. **Supervision 형태**: FreeMatch = **top-1 hard pseudo-label** → AllMatch = top-1 (CAT 통과 시) + **candidate vs negative classes (BCC)**.
> 보충: FreeMatch(Wang et al., ICLR 2023)는 self-adaptive threshold(global EMA of confidence + class-wise EMA of predicted probability로 local scaling)와 fairness regularization을 쓰는 방법. 위 "prediction-based local" 설명은 그 기준. 슬라이드엔 FreeMatch 세부 수식 없음(이전 chunk에서 다뤘을 수 있음).

---

## Class-Specific Adaptive Threshold: Global Estimation (p.81)

슬라이드 텍스트:
- Global estimation
  - Uses the average confidence of unlabeled samples
  - Tracks the overall learning progress
- Local adjustment (슬라이드에서 흐리게 처리됨 — 아직 설명 전)
  - Uses the norm of each classifier weight
  - Estimates how well each class has been learned
- 수식:
$$\tau_t = \begin{cases} \dfrac{1}{C} & \text{if } t=0 \\[4pt] m\,\tau_{t-1} + (1-m)\,\dfrac{1}{B_U}\displaystyle\sum_{i=1}^{B_U} \max(p_i) & \text{otherwise} \end{cases}$$
- 그림 (CAT pipeline): unlabeled $u$ → weak aug $\omega(u)$ / strong aug $\Omega(u)$ → 공유된 backbone $\mathcal{F}$ + classifier $\mathcal{G}$ → weak 쪽 예측 $p$, strong 쪽 예측 $q$. $p$ 위에서 **Global Estimation**(시간 $t$에 따라 증가하는 $\tau_t$ 곡선) → **Local Adjustment**(classifier $\mathcal{G}$의 weight를 입력으로 받음) → class별 threshold 막대 **CAT** ($\tau_t(1),\dots,\tau_t(5)$). 예: 예측 $p$ 위에 $\tau_t(3)$ 빨간 점선 threshold를 그린 $\tilde p$ → top-1(3번 class)이 이를 넘으면 one-hot pseudo-label $\hat p$ → strong 쪽 $q$와 비교하는 $\mathcal{L}_u$ (stopping gradients는 weak 쪽 $\hat p$에 적용). (그림 표기: 가운데 $\tilde p$ = thresholding 단계, 오른쪽 $\hat p$ = one-hot pseudo-label.)
- 하단: The global threshold is the EMA of the average confidence on unlabeled data. The main difference from FreeMatch lies in the local adjustment.

<img src="../../../assets/lecture07/p081_cat_pipeline.png" width="600">

**설명**
- **기호**: $C$ = 클래스 수, $B_U$ = unlabeled batch 크기, $p_i$ = sample $i$의 (weak view) 예측 확률 벡터, $\max(p_i)$ = confidence, $m$ = **EMA momentum**, $t$ = iteration.
- **수식 읽기**:
  - $t=0$: $\tau_0 = 1/C$ ← 클래스가 $C$개면 **uniform 예측의 confidence가 $1/C$** → 학습 시작 시점의 "아무것도 모르는 상태"에서 출발. 즉 초기 threshold를 **낮게** 시작해서 초반에 pseudo-label이 충분히 선택되게 함 (고정 높은 $\tau=0.95$ 방식의 초기 낭비 방지).
  - $t>0$: $\tau_t = m\tau_{t-1}+(1-m)\cdot\overline{\max p}$ ← **현재 batch의 평균 confidence의 EMA**. 모델이 학습될수록 평균 confidence가 올라가 → **threshold도 자동으로 올라감** (curriculum 효과).
- **의미**: "Tracks the overall learning progress" — 모델 전체의 학습 진행도를 **평균 confidence 하나**로 요약. **label 없이** 계산 가능한 지표라는 점이 장점.
- ← 이 global 부분은 **FreeMatch와 동일** (p.80 표 1행). 차이는 이어지는 **local adjustment** (p.82).
- 그림 읽기: global estimation이 만든 $\tau_t$에 **class별 scaling**(local adjustment)을 곱해서 $\tau_t(c)$ (CAT)를 만들고, weak view top-1 class의 $\tau_t(c)$를 넘는 경우에만 pseudo-label로 채택.
- ← 그림의 $\tilde p$(threshold 선 그린 weak 예측)와 p.83·p.85의 $\tilde p_i$는 같은 기호지만 맥락이 조금 다름 — 슬라이드엔 정의 없음.

---

## Classifier Weights as Learning-Status Indicators (p.82)

슬라이드 텍스트:
- A larger classifier-weight norm tends to produce larger logits.
- Classes with more confident samples tend to have larger weight norms.
- The weight norm reflects the learning status of each class.
- Difficult classes receive lower thresholds.
- 수식:
$$z_c = f W_c^\top, \qquad z_c = \|f\|\,\|W_c\|\cos\theta_c$$
  - $f$: feature of an input image
  - $W_c$: classifier weight for class $c$
  - $\theta_c$: angle between $f$ and $W_c$
  - $z_c$: logit for class $c$
- Class-specific threshold:
$$\tau_t(c) = \tau_t \cdot \frac{\|W_c\|}{\max\{\|W_c\| : c\in[1,\cdots,C]\}}$$
- The classifier itself provides evidence about class-specific learning progress.

**설명**
- **유도 (왜 weight norm이 logit 크기를 좌우하나)**: linear classifier에서 logit은 feature와 class weight의 **내적** $z_c=f W_c^\top$ → 내적의 기하학적 표현 $z_c=\|f\|\|W_c\|\cos\theta_c$. 같은 feature $f$에 대해 클래스 간 차이는 $\|W_c\|$와 $\cos\theta_c$에서 나옴 → **$\|W_c\|$가 클수록 logit이 커지고, softmax 후 confidence도 커지기 쉬움**.
- **왜 learning status의 지표인가**: 어떤 class의 sample이 **자주 confident하게 맞춰지면** 학습이 그 방향 $W_c$를 키우는 쪽으로 진행 → **잘 학습된 class = 큰 $\|W_c\|$**, **어려운/덜 학습된 class = 작은 $\|W_c\|$**. 그래서 "Classes with more confident samples tend to have larger weight norms."
- **Threshold 수식**: $\tau_t(c)=\tau_t\cdot \dfrac{\|W_c\|}{\max_{c'}\|W_{c'}\|}$.
  - 분수는 $(0,1]$ — **가장 큰 norm을 가진 class는 factor = 1** (threshold = global $\tau_t$ 그대로), 나머지는 **더 작은 threshold**.
  - → **Difficult classes receive lower thresholds** : 덜 학습된 class는 confidence가 원래 낮게 나오므로, threshold를 낮춰줘야 그 class의 sample이 **pseudo-label로 선택될 기회**를 얻음 (class imbalance/hard class 편향 완화 — FlexMatch/FreeMatch와 같은 동기).
  - 즉 최종 threshold는 **global ($\tau_t$) × local ($\|W_c\|/\max\|W\|$)** 의 곱 → 이름이 **Class-Specific Adaptive Threshold**.
- **FreeMatch와의 차이 (시험 포인트)**: 둘 다 local scaling을 쓰지만 **FreeMatch는 "prediction-based"** (class별 평균 예측 확률), **AllMatch는 "classifier weight norm"** 사용. ← 슬라이드 결론: "The classifier itself provides evidence about class-specific learning progress."
  > 보충: prediction-based보다 weight가 더 안정적이라는 해석은 내 추정 (슬라이드엔 명시 없음).
> 보충: weight norm이 class별 학습 정도/빈도와 관련 있다는 관찰은 long-tail 학습 쪽에서 알려진 현상으로 알고 있음 (슬라이드엔 근거 논문 없음). $W_c$는 원래 있는 classifier 파라미터라 추가 파라미터 불필요 (p.81 그림: $\mathcal{G}$의 weight가 Local Adjustment로 들어감).

---

## Binary Classification Consistency (p.83)

슬라이드 텍스트:
- When top-1 confidence is insufficient:
  - Select several high-ranked classes as candidates.
  - Treat the remaining classes as negative options.
  - Convert the multi-class prediction into a binary prediction.
  - Match the binary prediction across weak and strong views.
- 수식:
$$b_i = \Big[\sum_{c\in\mathcal{C}_i} p_i(c),\ \sum_{c\in\mathcal{N}_i} p_i(c)\Big]$$
  - 아래 라벨: $c\in\mathcal{C}_i$ → **positive (high-ranked)**, $c\in\mathcal{N}_i$ → **negative (remaining)**
- 그림 (BCC pipeline): unlabeled $u$ → weak $\omega(u)$ / strong $\Omega(u)$ → 공유된 $\mathcal{F}$, $\mathcal{G}$ → weak 예측 $p$, strong 예측 $q$. weak 예측 $p$에서 **Global Top-k Confidence** ($\mu_t^1,\mu_t^2,\mu_t^3,\mu_t^4,\dots,\mu_t^K$ 막대, $k$가 커질수록 증가) 계산 → 이 값을 이용해 weak 예측을 candidate 클래스(진한 남색)/negative 클래스(연한 청회색)로 나눈 $\tilde p$ 와 strong 예측 $q$ (같은 division 적용)를 만들고, 각각 **2차원 binary 벡터** $b^\omega$, $b^\Omega$ 로 압축 → 둘을 맞추는 $\mathcal{L}_b$ (weak 쪽 **stopping gradients** 표시). 범례: stopping gradients, candidate classes(진한색), negative classes(연한색).

<img src="../../../assets/lecture07/p083_bcc_pipeline.png" width="600">

**설명**
- **아이디어**: top-1을 **하나로 못 찍겠는** sample도 "상위 몇 개 후보 안에는 정답이 있다"(p.78 (b)의 top-5 acc ≈ 100%)는 정보는 믿을 수 있음 → **multi-class 문제를 "후보 vs 나머지"의 2-class 문제로 환원**해서 consistency를 건다.
- **수식 읽기**: $\mathcal{C}_i$ = sample $i$의 **candidate (top-$k_i$) 클래스 집합**, $\mathcal{N}_i$ = 나머지 클래스 집합. 
  - $b_i=[\sum_{c\in\mathcal{C}_i}p_i(c),\ \sum_{c\in\mathcal{N}_i}p_i(c)]$ → **candidate에 들어간 확률 질량 vs 나머지 확률 질량**. 두 성분의 합 = 1이므로 **valid Bernoulli(binary) 분포**.
- **Weak ↔ strong consistency**: weak view에서 **정해진 candidate 집합 $\mathcal{C}_i$를 strong view에도 그대로 적용**해서 $b^\omega$ (weak)와 $b^\Omega$ (strong)를 만들고, 둘이 같아지도록 $\mathcal{L}_b$ 최소화. ← "Enforces the same division across augmentations" (p.79).
  - 그림에서 $b^\omega$ 은 candidate 성분이 매우 큼 (candidate에 확률 질량 집중), $b^\Omega$ 는 strong aug 때문에 negative 성분이 더 큼 → $\mathcal{L}_b$가 strong 쪽을 weak에 맞추도록 유도 (stopping gradient는 weak 쪽).
- **왜 binary 변환이 좋은가**: (1) 후보 내부의 **어느 클래스가 정답인지는 아직 몰라도 됨** → noisy한 top-1 hard label 강요 회피. (2) 정답이 candidate 안에만 있으면 **candidate 합이 커지는 방향**으로 학습 → soft한 supervision. (3) 모든 sample에 적용 가능 → **"use all unlabeled samples"**.
- ← 슬라이드는 $\mathcal{L}_b$의 구체 수식(예: cross-entropy)을 주지 않음. 그림에서는 **weak 쪽 $b^\omega$에 stop-gradient**가 걸려 있고 strong 쪽 $b^\Omega$가 student 역할.
> 보충: $\mathcal{L}_b$의 구체 형태(예: binary cross-entropy 류)는 슬라이드에 없어 불확실 → 시험에서는 "binary 분포를 weak/strong view 간 맞춘다" 수준으로 충분.
> 보충: 그림의 $\tilde p$ 는 weak 예측을 "candidate 클래스 vs negative 클래스"로 색칠해 구분한 것으로, 확률값 자체가 바뀌는 건 아님(색 구분 = division).

---

## How Many Candidate Classes? — Global Top-k Confidence (p.84)

슬라이드 텍스트:
- The number of candidate classes $k$ is not set to the same value for all samples.
  - This is the cumulative probability assigned to the top $k$ classes for sample $i$.
$$p_i^k = \sum_{j=1}^{k} p_{i,c_j} \qquad (p_{i,c_1}\ge p_{i,c_2}\ge\cdots)$$
  - This indicates how much probability the overall model typically assigns to the top $k$ classes.
$$\mu_t^k = \begin{cases} \dfrac{k}{C} & \text{if } t=0 \\[4pt] m\,\mu_{t-1}^k + (1-m)\,\dfrac{1}{B_U}\displaystyle\sum_{i=1}^{B_U} p_i^k & \text{otherwise} \end{cases}$$

**설명**
- **$p_i^k$ (sample-level)**: sample $i$의 확률을 **큰 순서로 정렬**($c_1,c_2,\dots$)한 뒤 **상위 $k$개의 누적 확률**. $k=1$이면 그냥 confidence $\max p_i$.
- **$\mu_t^k$ (model-level, global top-$k$ confidence)**: $p_i^k$의 **batch 평균을 EMA**로 추적. "모델이 평균적으로 상위 $k$ 클래스에 얼마의 확률을 주는가". 
  - $t=0$: $\mu_0^k=k/C$ ← uniform 예측일 때 상위 $k$개 누적 확률($k\times\frac1C$). **p.81의 $\tau_0=1/C$ 와 같은 논리** ($k=1$일 때 정확히 일치).
  - $\mu_t^1$ = p.81의 global threshold $\tau_t$ 와 동일한 형태(둘 다 평균 confidence의 EMA). → **CAT global threshold의 top-$k$ 일반화**.
  - 학습이 진행될수록 $\mu_t^k$가 증가 (그림 p.83의 $\mu_t^1<\mu_t^2<\dots$ 막대가 $k$가 클수록 커지는 모양).
- **왜 sample마다 $k$가 달라야 하나**: 고정 $k$(예: 항상 top-3)를 쓰면 이미 **자신있는 sample**은 불필요하게 큰 candidate 집합이 되고, **매우 애매한 sample**은 candidate이 부족함. → **모델이 평소 쓰는 누적 확률 수준($\mu_t^k$)을 기준**으로 sample별로 결정 (p.85).
- ← 슬라이드의 "This indicates how much probability the overall model typically assigns to the top k classes." 는 $\mu_t^k$를 설명하는 문장. 위 $p_i^k$ 설명 문장과 위치가 슬라이드에서 수식 위/아래로 붙어 있어 헷갈리기 쉬움.

---

## How Many Candidate Classes? — Per-sample $k_i$ (p.85)

슬라이드 텍스트:
- Confident samples use one candidate class.
- Ambiguous samples use multiple candidate classes.
- The candidate-set size depends on each sample.
- $K$ prevents the candidate set from becoming trivial.
$$k_i = \begin{cases} 1 & \text{if } \lambda(\tilde p_i)=1 \\ \min\big(\min\{k : \tilde p_i^k \ge \mu_t^k\},\ K\big) & \text{otherwise} \end{cases}$$
- More ambiguous samples receive larger candidate sets.

**설명**
- **수식 읽기**:
  - $\lambda(\tilde p_i)=1$ : sample이 **confident하다는 조건**(CAT의 class-specific threshold를 통과한 경우로 해석) → **$k_i=1$**, 즉 candidate = top-1 하나뿐 ← 이 경우는 **일반 pseudo-label(CAT)** 와 같은 상황.
  - otherwise (ambiguous) : $\min\{k:\ \tilde p_i^k\ge\mu_t^k\}$ = **sample의 top-$k$ 누적확률이 모델의 평균 수준 $\mu_t^k$ 이상이 되는 가장 작은 $k$**. 
  - 그 값을 $K$로 **clip**: $\min(\cdot,\ K)$.
- **직관**: 애매한 sample은 확률이 여러 클래스에 퍼져 있어서 $\tilde p_i^k$가 천천히 늘어남 → 기준($\mu_t^k$)에 도달하려면 **더 큰 $k$**가 필요 → **candidate set이 커짐**. 확신 있는 sample은 작은 $k$에서 이미 넘음. 그래서 "More ambiguous samples receive larger candidate sets."
- **$K$의 역할**: "$K$ prevents the candidate set from becoming trivial." — 극단적으로 $k=C$면 candidate가 **전체 클래스**라서 $\sum_{c\in\mathcal{C}}p(c)=1$ → binary 예측이 항상 $[1,0]$ → **정보량 0 (trivial)**. 그러므로 상한 $K$ ($K<C$)를 둠.
- **Design rationale**: (a) 고정 $k$ 대신 **sample-wise adaptive $k$** → 모델 상태(EMA $\mu_t^k$)와 sample의 애매함에 모두 반응. (b) confident sample은 **hard top-1 pseudo-label**, ambiguous sample은 **multi-candidate binary consistency**로 분업 → 모든 sample이 supervision을 받음.
- ← p.83 그림: Global Top-k Confidence ($\mu_t^1\dots\mu_t^K$)가 weak 예측 $p$에서 계산되어 weak/strong **양쪽의 candidate 분할**에 쓰임.
- ← 슬라이드에 $\lambda(\cdot)$ 정의와 $K$의 구체 값은 없음. $\tilde p_i$가 무엇인지(그림 p.83의 $\tilde p$)도 이 chunk 범위에서는 명시되지 않음.
> 보충: $\lambda(\cdot)$는 CAT threshold 통과 여부 indicator로 추정 (슬라이드 정의 없음 → 시험에선 \"confident → $k_i=1$\"만 알면 됨). $\tilde p_i$는 weak view 예측으로 추정.

---

## FIGURES

- p.78: (a) Dropped pseudo-label acc / (b) Top-5 acc of pseudo-labels line plots — `assets/lecture07/p078_dropped_pl_acc.png`, 중요도 상 (low-confidence ≠ no information 근거; 숫자는 근사)
- p.80: FreeMatch vs AllMatch 비교 표 (텍스트로 전사함, 이미지 없음), 중요도 상
- p.81: CAT pipeline 그림 + $\tau_t$ EMA 수식 — `assets/lecture07/p081_cat_pipeline.png`, 중요도 상
- p.82: $z_c=\|f\|\|W_c\|\cos\theta_c$ 및 $\tau_t(c)$ 수식 (텍스트로 전사), 중요도 상
- p.83: BCC pipeline (weak/strong, Global Top-k Confidence, $b^\omega,b^\Omega$, $\mathcal{L}_b$) — `assets/lecture07/p083_bcc_pipeline.png`, 중요도 상
- p.84~85: $p_i^k$, $\mu_t^k$, $k_i$ 수식 (텍스트로 전사), 중요도 중상

## EXAM POINTS

1. **Threshold-based 방법의 한계 (p.78)**: high confidence → top-1을 pseudo-label로 사용, low confidence → **discard, training signal 없음**. 슬라이드 결론 "**Low confidence does not mean no semantic information.**" 근거: 버려진 pseudo-label도 **top-1 acc ≈ 55~67%**, **top-5 acc ≈ 100%** (그림은 근사). *Trap*: "top-1 accuracy가 거의 100%" (X, top-5가 ~100%), "버려진 sample은 모두 오답" (X).
2. **AllMatch = CAT + BCC (p.79)**: CAT = **Class-Specific Adaptive Threshold** (pseudo-label selection 개선), BCC = **Binary Classification Consistency** (나머지 low-confidence sample 학습). *Trap*: 두 약자 풀네임 바꾸기, 역할 뒤바꾸기("BCC improves pseudo-label selection" → X).
3. **FreeMatch vs AllMatch 표 (p.80)**: global threshold는 **둘 다 adaptive (같음)**, local threshold는 FreeMatch = **prediction-based** vs AllMatch = **classifier-weight-based**, FreeMatch는 low-confidence **discard** vs AllMatch는 **every unlabeled sample 사용**, FreeMatch = **top-1 pseudo-label** vs AllMatch = **candidate vs negative classes**. *Trap*: prediction-based ↔ weight-based 뒤바꾸기, "AllMatch는 global threshold가 없다" (X).
4. **Global threshold (p.81)**: $\tau_0=1/C$, 이후 $\tau_t = m\tau_{t-1}+(1-m)\frac{1}{B_U}\sum\max(p_i)$ = **unlabeled 평균 confidence의 EMA**. "Tracks overall learning progress". *Trap*: EMA of **weights**(X) / **labeled** data의 confidence(X) / 초기값 $1/C$ vs 0.95 등 고정값.
5. **Weight norm ≈ learning status (p.82)**: $z_c=\|f\|\|W_c\|\cos\theta_c$ → **큰 $\|W_c\|$ → 큰 logit**; confident sample이 많은 class일수록 큰 norm. $\tau_t(c)=\tau_t\cdot\|W_c\|/\max_c\|W_c\|$ → **difficult(norm 작은) class는 낮은 threshold**. *Trap*: "difficult classes get **higher** thresholds"(X), 분모를 평균/합이 아니라 **max**라는 점, class-specific scaling이 가장 큰 class에서는 1.
6. **BCC 메커니즘 (p.83)**: top-1 confidence가 불충분할 때 **상위 몇 개를 candidate**, 나머지를 **negative**로 → $b_i=[\sum_{c\in\mathcal{C}_i}p_i(c),\sum_{c\in\mathcal{N}_i}p_i(c)]$ (binary) → **weak vs strong view** 간 맞춤 ($\mathcal{L}_b$). *Trap*: "candidate 안에서 정답 1개를 고르는 multi-class loss" (X), weak↔strong 뒤바꾸기 (weak 쪽이 stop-gradient target), "binary"가 아닌 "ternary" 등.
7. **Global top-k confidence (p.84)**: $p_i^k=\sum_{j=1}^k p_{i,c_j}$ (내림차순 상위 $k$ 누적 확률), $\mu_t^k$ = 그 batch 평균의 EMA, 초기 $\mu_0^k=k/C$. *Trap*: 초기값 $1/C$(그건 $k=1$일 때, $\tau_0$) vs $k/C$.
8. **sample별 $k_i$ (p.85)**: confident → $k_i=1$, ambiguous → 여러 개; $k_i=\min(\min\{k:\tilde p_i^k\ge\mu_t^k\},K)$. **$K$의 역할 = candidate set이 trivial(전체 클래스)이 되는 것 방지.** "More ambiguous samples receive larger candidate sets." *Trap*: "모든 sample에 같은 $k$"(X), "$K$는 하한(lower bound)"(X, 상한), "confident sample이 더 큰 candidate set"(X).
9. **$\mu_t^1=\tau_t$ 구조 (p.81, p.84)**: global top-1 confidence와 global threshold는 같은 EMA 형태, $\mu_t^k$는 top-$k$ 일반화. $b_i$는 **2차원** (candidate 합, negative 합), 합 = 1. *Trap*: $\tau_0=k/C$처럼 초기값 섞기, \"$b_i$는 $C$차원\".\n10. **설계 의도 요약 (p.79)**: CAT → 신뢰할 수 있는 top-1 pseudo-label을 **class별로 공정하게** 선택 / BCC → threshold를 못 넘은 sample도 **"후보 vs 나머지"라는 약한 supervision**으로 사용 → **all unlabeled samples**에 learning signal.
11. **참고 출처**: Wu et al., **IJCAI 2024**, *AllMatch: Exploiting All Unlabeled Data for Semi-Supervised Learning*.
