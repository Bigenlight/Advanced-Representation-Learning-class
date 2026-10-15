<!-- DRAFT | lecture: 06_Semi_Supervised_Learning_I | chunk: 02 | pages 10–16 -->
<!-- VERIFIED | fixes: 2 | verifier: sonnet -->

## Notations (p.10)

슬라이드 텍스트 (표, Symbol → Meaning):
- $L$ : Number of unique labels.
- $(\mathbf{x}^l, y) \sim \mathcal{X},\ y \in \{0,1\}^L$ : Labeled dataset. $y$ is a one-hot representation of the true label.
- $\mathbf{u} \sim \mathcal{U}$ : Unlabeled dataset.
- $\mathcal{D} = \mathcal{X} \cup \mathcal{U}$ : The entire dataset, including both labeled and unlabeled examples.
- $\mathbf{x}$ : Any sample which can be either labeled or unlabeled.
- $\bar{\mathbf{x}}$ : $\mathbf{x}$ with augmentation applied.
- $\mathbf{x}_i$ : The $i$-th sample.
- $\mathcal{L}, \mathcal{L}_s, \mathcal{L}_u$ : Loss, supervised loss, and unsupervised loss.
- $\mu(t)$ : The unsupervised loss weight, increasing in time.
- $p(y|\mathbf{x}),\ p_\theta(y|\mathbf{x})$ : The conditional probability over the label set given the input.
- $f_\theta(\cdot)$ : The implemented neural network with weights $\theta$, the model that we want to train.
- $\mathbf{z} = f_\theta(\mathbf{x})$ : A vector of logits output by $f$.
- $\hat{y} = \mathrm{softmax}(\mathbf{z})$ : The predicted label distribution.
- $D[\cdot,\cdot]$ : A distance function between two distributions, such as MSE, cross entropy, KL divergence, etc.
- $\beta$ : EMA weighting hyperparameter for teacher model weights.
- $\alpha, \lambda$ : Parameters for MixUp, $\lambda \sim \mathrm{Beta}(\alpha, \alpha)$.
- $T$ : Temperature for sharpening the predicted distribution.
- $\tau$ : A confidence threshold for selecting the qualified prediction.

**설명**
- 이 강의 전체(SSL)에서 쓰는 **기호 사전**이라 시험 때 수식 읽을 때 계속 돌아오게 됨. 핵심은 데이터가 두 덩어리라는 점: **labeled set $\mathcal{X}$** (정답 $y$ 있음)와 **unlabeled set $\mathcal{U}$** (입력만 있음), 합쳐서 $\mathcal{D}=\mathcal{X}\cup\mathcal{U}$.
- $y\in\{0,1\}^L$ 은 **one-hot** 벡터, $L$은 클래스 수. $\hat y=\mathrm{softmax}(\mathbf z)$ 는 모델이 내놓는 **예측 분포**라서 $y$(one-hot)와 같은 차원의 확률벡터 → cross entropy 등으로 비교 가능.
- 전체 loss는 보통 $\mathcal{L}=\mathcal{L}_s+\mu(t)\,\mathcal{L}_u$ 형태. **$\mu(t)$가 시간에 따라 증가**하는 이유: 학습 초기엔 모델 예측이 부정확해서 unlabeled 신호(예: pseudo-label, consistency target)가 noisy → 초반엔 가중치를 낮추고 점차 올림 (ramp-up).
- $D[\cdot,\cdot]$ 는 **두 분포 사이 거리**의 일반 표기: MSE, cross entropy, KL 중 방법마다 다른 걸 씀 (consistency regularization 수식 p.16에서 바로 등장).
- $\beta$ (EMA weight)는 Mean Teacher류의 **teacher = student weight의 EMA** 갱신에 쓰임: $\theta_{t} \leftarrow \beta\,\theta_{t}+(1-\beta)\,\theta_{s}$ 식.
- $\alpha,\lambda$ 는 **MixUp**용: $\lambda\sim\mathrm{Beta}(\alpha,\alpha)$ 를 뽑아 두 샘플을 $\lambda x_i+(1-\lambda)x_j$ 로 섞음.
- $T$ 는 **sharpening**용 temperature (낮은 $T$ → 분포가 더 뾰족해짐), $\tau$ 는 **confidence threshold** (이 값 넘는 예측만 pseudo-label로 채택, FixMatch류).
> 보충: sharpening은 보통 $\mathrm{Sharpen}(p)_i = p_i^{1/T}/\sum_j p_j^{1/T}$ 형태 (MixMatch). 슬라이드엔 수식 없음. 해당 수식은 이후 슬라이드에서 나오면 거기서 확인.
> 보충: $\bar{\mathbf x}$ (augmented sample)와 $\mathbf x_i$ ($i$-th sample)는 표기가 비슷해서 헷갈리기 쉬움 — bar가 붙으면 **augmentation 적용**, 아래첨자 $i$는 **인덱스**.

---

## Hypotheses: H1. Smoothness Assumption (p.11)

슬라이드 텍스트:
- H1. Smoothness Assumption
  - Nearby samples should have similar predictions.
  - Small perturbations should not change the label.
- 그림: 왼쪽 **Feature Space (high-density regions)** — x축 Feature dim. 1, y축 Feature dim. 2. 파란 점 cluster와 초록 점 cluster가 있고 각각 "Two nearby samples (high-density region)"로 가까운 두 점을 표시, 가운데는 회색 점이 드문 "Low-density region (samples are far apart)".
- 그림: 오른쪽 **Consistency under perturbations (augmentations)** — Original sample $x$ (산+호수 이미지)에 Augmentations (e.g., crop, color jitter, flip)을 적용한 여러 view → 각각의 Model predictions (bar chart)가 서로 $\approx$. 아래는 Original sample $x'$ (초록 들판+산)도 마찬가지로 augmentation view들의 prediction이 서로 $\approx$.

**설명**
- **Smoothness assumption**: 입력 공간에서 **가까운 두 점은 비슷한 output(label)** 을 가져야 한다는 가정. 즉 예측 함수 $f$가 입력에 대해 급격히 변하지 않는 **smooth function**이어야 함.
- 두 문장은 같은 말의 다른 표현: "nearby samples → similar predictions"는 **다른 샘플 간** 얘기, "small perturbation → label 불변"은 **같은 샘플에 노이즈/augmentation을 준 경우** 얘기. perturbation된 샘플도 원본 근처에 있는 샘플로 볼 수 있으니 동일 가정으로 묶임.
- 그림 해석: 오른쪽은 이 가정이 **consistency regularization으로 구현**되는 모습을 미리 보여줌 — 원본과 augmented view들의 예측 분포가 $\approx$ 이어야 함 (p.16에서 loss로 정식화). 서로 다른 이미지($x$ vs $x'$)는 예측이 달라도 되고 (색이 다른 bar), **같은 이미지의 view끼리만** 같아야 함.
- **왜 SSL에 쓸모 있나**: 이 조건은 **label 없이도 검사 가능**함 (같은 입력의 두 view 예측이 같은지만 보면 됨). 그래서 unlabeled data를 loss에 활용하는 통로가 됨.
> 보충: 엄밀한 SSL 교과서(Chapelle et al.)의 "semi-supervised smoothness assumption"은 "두 입력이 **high-density region에서 가까우면** 대응 output도 가까워야 한다"로, **density 조건이 붙음**. 슬라이드는 이를 단순화하여 "nearby ⇒ similar"로 표현했고 그림에서 "high-density region"을 강조하는 이유가 이것임.

---

## Hypotheses: H2. Cluster Assumption (p.12)

슬라이드 텍스트:
- H2. Cluster Assumption
  - Samples within the same cluster are likely to share the same label.
  - Decision boundaries should not split a cluster.
- 그림: Feature dim. 1 / Feature dim. 2 평면에 세 cluster — Cluster 1 (label: Blue), Cluster 2 (label: Green), Cluster 3 (label: Orange). cluster 사이 영역은 "Low-density region"으로 표시(회색 점이 듬성듬성).

**설명**
- **Cluster assumption**: 데이터가 **discrete한 cluster들로 뭉쳐 있고, 같은 cluster 안의 점들은 같은 label**일 가능성이 높다는 가정. 따라서 **decision boundary는 cluster를 쪼개면 안 됨**.
- 이게 SSL이 성립하는 핵심 직관: labeled sample이 cluster 안에 **하나만** 있어도 그 cluster 전체 unlabeled 점들에 label을 **전파**할 수 있음 ← unlabeled data는 **cluster의 모양(구조)** 을 알려주는 역할.
- H1과의 관계: smoothness가 "가까우면 비슷"이라는 **local** 성질이라면, cluster는 **chain처럼 이어진 점들이 한 덩어리로 같은 label**이라는 좀 더 **global**한 성질 (가까움이 연결(chain)되어 퍼짐).
> 보충: cluster assumption은 다음 H3(low-density separation)와 **동치로 볼 수 있음** — cluster를 쪼개지 않는 boundary = cluster 사이 low-density 영역을 지나는 boundary. 다만 H2는 "label의 구조"(cluster ⇒ label), H3는 "boundary의 위치"(low-density ⇒ boundary)로 **서술 관점이 다름**.

---

## Hypotheses: H3. Low-Density Separation (p.13)

슬라이드 텍스트:
- H3. Low-Density Separation
  - Decision boundaries should lie in low-density regions.
  - Separate clusters rather than cutting through them.
- 그림 (좌): **Good (consistent with H3)** — 점선 decision boundary가 세 cluster 사이 빈 공간을 구불구불 지나감. 캡션: "Boundary lies in low-density regions (between clusters) ⇒ Does not cut through any cluster."
- 그림 (우): **Bad (violates H3)** — 점선 boundary가 파란 cluster와 주황 cluster를 가로질러 지나감. 캡션: "Boundary cuts through a high-density region (inside a cluster) ⇒ Splits a cluster into different labels."

**설명**
- **Low-density separation**: decision boundary는 **데이터 밀도 $p(\mathbf x)$가 낮은 곳**에 있어야 함. cluster를 **가르는(separate)** 경계지, 관통하는(cut through) 경계가 아님.
- 직관: high-density 영역 한가운데를 boundary가 지나가면, 거의 같은 분포에서 나온 이웃 점들이 서로 다른 label을 받게 됨 → 위 H1/H2 위반. 반대로 low-density 영역에는 **샘플이 거의 없으니** boundary가 지나가도 영향받는 점이 적음.
- **SSL에서의 실용적 의미**: boundary를 정할 때 labeled data만 보면 cluster 안을 지나는 boundary도 (labeled 점 사이 어디든) 똑같이 좋아 보이지만, **unlabeled 점들이 어디에 몰려있는지** 알면 그런 boundary를 배제할 수 있음 → unlabeled data가 **boundary의 위치를 low-density 쪽으로 밀어줌**. (예: SVM의 maximum margin을 unlabeled 포함해서 확장한 **TSVM**, entropy minimization도 이 가정에서 나옴.)
> 보충: Entropy minimization / Pseudo-labeling이 low-density separation을 암묵적으로 구현한다고 해석함 — 예측을 confident(낮은 entropy)하게 만들면 unlabeled 점이 boundary 근처(= 예측이 애매한 곳)에 못 있게 되어 boundary가 밀도 낮은 곳으로 이동. 슬라이드에는 없는 보충.

---

## Hypotheses: H4. Manifold Assumption (p.14)

슬라이드 텍스트:
- H4. Manifold Assumption
  - High-dimensional data lie on a low-dimensional manifold.
  - Similar samples are connected on the underlying manifold.
- 그림 (좌): **High-dimensional observation space** (e.g., image pixels: 100K+ dimensions) — 3D 상자(Dimension 1, Dimension 2, ... Dimension D ($D \gg 2$))에 회색 점들이 흩어져 있고, 그 안에 휘어진 곡면(파랑→초록→노랑→주황 그라데이션)이 있어 "Data lie on a low-dimensional manifold" 표시. 곡면 위 점들이 색별로 이어짐.
- 화살표: "(Unknown) manifold mapping".
- 그림 (우): **Underlying low-dimensional manifold** (intrinsic dimension $d \ll D$) — 곡면을 펼친 2D 평면 ($z_1, z_2$ 축)에 파랑·초록·주황 점들이 나열. 설명 박스: "Although the data are high-dimensional, they can be described by a low-dimensional manifold ($d=2$ here)."

**설명**
- **Manifold assumption**: 겉보기 차원 $D$ (예: 이미지 픽셀 수 10만+)는 매우 높지만, 실제 데이터는 **intrinsic dimension $d\ll D$ 인 저차원 manifold** 위에 놓여 있다는 가정. 이미지가 가질 수 있는 모든 픽셀 조합 중 **의미 있는 이미지는 극히 일부**라서 성립.
- **"Similar samples are connected on the manifold"**: 유사성은 ambient space의 **Euclidean 거리가 아니라 manifold 위의 거리(geodesic)** 로 재야 함. 그림에서 파랑→초록→주황이 곡면을 따라 연속적으로 변하는 것이 그 의미. 3D 상자 안에서는 파랑 점과 주황 점이 가까워 보일 수 있어도 manifold 위에서는 멀 수 있음 (반대도 가능).
- **왜 SSL에 도움?** (1) 고차원에서는 **curse of dimensionality** 때문에 labeled 샘플만으로 구조를 파악 불가능. (2) unlabeled data가 있으면 manifold의 **모양을 더 조밀하게 추정**할 수 있고, 그 위에서 label이 smooth하게 변하도록 학습 가능. (3) 이 가정 하에서 **저차원 표현을 학습**하는 것이 representation learning의 동기와 연결됨 (이 수업 주제와 직결).
- 그림 수식: 우측 $z_1,z_2$ 는 manifold의 **intrinsic coordinate** (latent 좌표)를 의미. $d=2$는 그림 예시용.
> 보충: 슬라이드 그림 제목은 "Underlying low-dimensional manifold (intrinsic dimension $d \ll D$)"로 되어 있음. Manifold assumption이 H1~H3를 보완하는 점은, cluster 구조가 **비선형 구조(곡선·나선 형태)** 여도 manifold를 따라 label이 퍼지는 걸 설명한다는 것. 예: two-moons 데이터는 Euclidean 거리로는 두 moon 사이가 가깝지만 manifold 거리로는 멀다.
> 보충: Graph-based SSL (label propagation)은 manifold assumption의 대표적 구현 (kNN graph = manifold의 근사). 슬라이드엔 없음.

---

## Consistency Regularization (section title, p.15)

슬라이드 텍스트:
- 섹션 구분 슬라이드: "Consistency Regularization" (진한 남색 배경, 중앙 흰 글씨).

**설명**
- 여기서부터 SSL 방법론 파트 시작. 앞의 H1–H4 가정이 이후 방법들의 **이론적 근거**임 (특히 H1 smoothness ⇒ consistency regularization).

---

## Consistency Regularization: Core Idea (p.16)

슬라이드 텍스트:
- Core Idea
  - Predictions should remain **consistent under small perturbations.**
  - Perturbations may come from:
    - Data augmentations
    - Model stochasticity (e.g., Dropout)
- 수식:
$$\mathcal{L}_u = D\big(f(x),\ f(\mathcal{A}(x))\big)$$

**설명**
- **핵심 아이디어**: 같은 입력 $x$에 작은 perturbation을 줘도 모델의 **예측이 일관(consistent)** 해야 함. H1(smoothness)을 loss로 바꾼 것.
- **Perturbation의 출처 두 가지**:
  1. **Data augmentation** $\mathcal{A}(x)$: 입력을 직접 변형 (crop, flip, color jitter 등; 강한 augmentation도 포함 가능).
  2. **Model stochasticity**: 입력은 그대로 두고 **모델 내부의 randomness** (Dropout 등)로 두 번의 forward가 다른 출력을 내게 함 ($\Pi$-model류가 이 방식도 사용).
- **수식 읽기**: $\mathcal{L}_u = D\big(f(x), f(\mathcal{A}(x))\big)$.
  - $\mathcal{L}_u$ : **unsupervised loss** (p.10 notation). **label $y$가 수식에 전혀 안 나옴** → unlabeled sample에도 계산 가능 → $x\in\mathcal{U}$ 에 적용.
  - $f(x)$ : 원본(또는 약하게 변형한) 입력에 대한 모델 출력 (확률 분포 $\hat y$).
  - $f(\mathcal{A}(x))$ : augmentation $\mathcal{A}$를 적용한 입력의 출력.
  - $D[\cdot,\cdot]$ : p.10의 distance function (MSE, cross entropy, KL 등). 이 선택에 따라 방법이 달라짐.
- 전체 목적식은 notation(p.10)을 이용해 $\mathcal{L}=\mathcal{L}_s+\mu(t)\,\mathcal{L}_u$ 꼴로 합쳐짐 — **labeled 데이터엔 supervised loss, unlabeled 데이터엔 consistency loss**.
> 보충: 슬라이드 수식은 두 출력을 대칭적으로 쓰고 있지만 방법에 따라 **한쪽에 stop-gradient**를 걸어 target처럼 취급함 (예: Mean Teacher는 teacher 쪽 gradient 없음). 슬라이드에는 없는 내용이며 이후 슬라이드(Π-model, Mean Teacher 등)에서 구체화될 것으로 보임.
> 보충: trivial solution 우려 — 모든 입력에 같은 출력을 내는 상수 함수는 consistency loss를 0으로 만들지만, supervised loss $\mathcal{L}_s$ 가 함께 있어서 이를 막아줌.

---

## FIGURES

- p.10: Notations 표 (기호 18개), 중요도 상 (시험에서 수식 해석에 필수)
- p.11: H1 Smoothness — 좌측 feature space(high/low-density 영역) + 우측 augmentation별 prediction이 $\approx$ 인 diagram, 중요도 중
- p.12: H2 Cluster — 3개 cluster(Blue/Green/Orange)와 low-density region 그림, 중요도 중
- p.13: H3 Low-density separation — Good(boundary가 cluster 사이)과 Bad(cluster를 관통) 비교 그림, 중요도 상
- p.14: H4 Manifold — 고차원 3D 공간의 곡면과 2D로 펼친 manifold ($d \ll D$), 중요도 중
- p.16: $\mathcal{L}_u = D(f(x), f(\mathcal{A}(x)))$ 수식, 중요도 상

## EXAM POINTS

1. **SSL의 4가지 가정 정의** (H1 Smoothness, H2 Cluster, H3 Low-Density Separation, H4 Manifold)를 영어로 한 문장씩 쓸 수 있어야 함: 각각 "nearby samples → similar predictions", "same cluster → same label / boundary should not split a cluster", "boundary in low-density regions", "high-dim data on low-dim manifold".
2. **H2 vs H3 구분**: H2는 label이 cluster 단위로 공유됨(label 구조), H3는 decision boundary의 위치가 low-density 영역(boundary 위치). 사실상 같은 직관의 두 표현이라는 점도 설명 가능해야 함.
3. **왜 unlabeled data가 도움이 되나**: 가정 H1–H4 하에서는 $p(\mathbf x)$의 구조(cluster, density, manifold)가 $p(y|\mathbf x)$에 대한 정보를 담고 있기 때문. 가정이 틀리면 unlabeled data가 오히려 해가 될 수 있음.
4. **Notation 숙지**: $\mathcal{X}$ (labeled), $\mathcal{U}$ (unlabeled), $\mathcal{D}=\mathcal{X}\cup\mathcal{U}$, $\mathbf z=f_\theta(\mathbf x)$ (logits), $\hat y=\mathrm{softmax}(\mathbf z)$, $\mathcal{L}_s/\mathcal{L}_u$, $\mu(t)$ (**시간에 따라 증가하는** unsupervised loss weight), $\beta$ (teacher EMA), $\lambda\sim\mathrm{Beta}(\alpha,\alpha)$ (MixUp), $T$ (sharpening temperature), $\tau$ (confidence threshold).
5. **Consistency Regularization 정의와 수식**: $\mathcal{L}_u = D(f(x), f(\mathcal{A}(x)))$. perturbation의 두 출처(**data augmentation, model stochasticity (Dropout)**)를 쓸 것. label 없이 계산 가능하다는 점이 핵심.
6. **$D[\cdot,\cdot]$ 후보**: MSE, cross entropy, KL divergence. 어떤 걸 쓰는지가 방법마다 다름 (뒤 슬라이드에서 확인).
7. **$\mu(t)$ 가 증가하는 이유** (design rationale): 초기 모델 예측이 부정확하므로 unsupervised 신호의 noise를 피하기 위해 ramp-up.
8. **Manifold assumption이 representation learning과 연결**되는 이유: 저차원 intrinsic 표현 $z$ 를 학습하면 유사도를 manifold 상에서 잴 수 있음.
