<!-- DRAFT | lecture: 04_Contrastive_Learning_II | chunk: 04 | pages 25–32 -->
<!-- VERIFIED | fixes: 2 | verifier: sonnet -->

## SimCLR (p.25)

슬라이드 내용: SimCLR 논문(Algorithm 1, "SimCLR's main learning algorithm")의 알고리즘 박스를 그대로 가져온 슬라이드. 박스별로 색깔 주석이 달려 있음.

**Algorithm 1 SimCLR's main learning algorithm**
- **input**: batch size $N$, constant $\tau$, structure of $f, g, \mathcal{T}$.
- **for** sampled minibatch $\{x_k\}_{k=1}^{N}$ **do**
  - **for all** $k \in \{1,\dots,N\}$ **do**
    - draw two augmentation functions $t \sim \mathcal{T},\ t' \sim \mathcal{T}$ (빨간 박스)
    - \# the first augmentation
    - $\tilde{x}_{2k-1} = t(x_k)$ (빨간 박스)
    - $h_{2k-1} = f(\tilde{x}_{2k-1})$  \# representation
    - $z_{2k-1} = g(h_{2k-1})$  \# projection
    - \# the second augmentation
    - $\tilde{x}_{2k} = t'(x_k)$ (빨간 박스)
    - $h_{2k} = f(\tilde{x}_{2k})$  \# representation
    - $z_{2k} = g(h_{2k})$  \# projection
  - **end for**
  - **for all** $i \in \{1,\dots,2N\}$ and $j \in \{1,\dots,2N\}$ **do**
    - $s_{i,j} = z_i^\top z_j / (\lVert z_i \rVert \lVert z_j \rVert)$  \# pairwise similarity
  - **end for**
  - **define** $\ell(i,j) = -\log \dfrac{\exp(s_{i,j}/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k\neq i]} \exp(s_{i,k}/\tau)}$ (파란 박스)
  - $\mathcal{L} = \dfrac{1}{2N}\sum_{k=1}^{N}\left[\ell(2k-1,2k) + \ell(2k,2k-1)\right]$ (분홍 박스)
  - update networks $f$ and $g$ to minimize $\mathcal{L}$
- **end for**
- **return** encoder network $f(\cdot)$, and throw away $g(\cdot)$

슬라이드 주석(색깔 박스 텍스트):
- 빨간 박스(왼쪽): "Generate a **positive pair** by sampling data augmentation functions"
- 파란 박스(오른쪽): "InfoNCE loss: Use all non-positive samples in the batch as x -"
- 분홍 박스(왼쪽 아래): "Iterate through and use each of the 2N sample as reference, compute average loss"

출처: Chen et al., ICML 2020, A Simple Framework for Contrastive Learning of Visual Representations

**설명**
- **전체 흐름**: 미니배치 $N$개 이미지 $x_k$ 각각에 서로 다른 augmentation $t, t'$ (같은 augmentation family $\mathcal{T}$에서 샘플)를 적용해 view 2개 생성 → 총 **2N개 augmented sample**. 각 view는 encoder $f$로 representation $h$, projection head $g$로 projection $z$를 얻음.
- **Positive pair 생성**: 같은 원본 $x_k$에서 나온 두 view $(\tilde{x}_{2k-1}, \tilde{x}_{2k})$가 positive pair. 라벨 없이 "같은 이미지에서 파생됨"이라는 사실만으로 positive를 정의하는 것이 self-supervised의 핵심. **augmentation이 곧 supervision 신호** 역할을 함.
- **유사도**: $s_{i,j}$는 $z_i$, $z_j$의 **cosine similarity** (L2 normalize 후 내적). 범위 $[-1, 1]$.
- **손실 $\ell(i,j)$ (NT-Xent / InfoNCE 형태)**: 분자는 positive pair $(i,j)$의 유사도, 분모는 anchor $i$ 자신($k=i$)을 제외한 나머지 $2N-1$개 sample 전부와의 유사도 합. 즉 **softmax cross-entropy를 하되 정답 클래스가 positive인 (2N-1)-way classification**. 분모에 들어가는 positive가 아닌 $2N-2$개는 모두 **negative**로 취급 (슬라이드 표현: "use all non-positive samples in the batch as x −"). 별도의 memory bank 없이 **in-batch negatives**를 씀.
- **$\tau$ (temperature)**: 유사도를 $\tau$로 나눈 뒤 softmax. 작을수록 분포가 sharp해져 hard negative를 더 강하게 밀어냄.
- **$\mathbb{1}_{[k\neq i]}$**: 지시함수, 자기 자신과의 유사도(항상 1)가 분모에 들어가는 것을 막음.
- **대칭 손실**: $\ell(2k-1,2k)$와 $\ell(2k,2k-1)$ 둘 다 계산 → 두 view가 번갈아 anchor가 됨. 따라서 2N개 sample 전부가 reference가 되고 $\frac{1}{2N}$으로 평균.
- **학습 후**: $g$는 버리고 **$f$ (encoder)만 downstream에 사용**. 이 설계 이유는 p.28 참고.
- 이전 방법(InstDisc, MoCo 등)과의 차이: memory bank/queue 없이 **큰 batch 자체가 negative 공급원**. 대신 큰 batch 필요 (p.29).

> 보충: 논문에서 이 손실은 NT-Xent (Normalized Temperature-scaled Cross Entropy)라 부름. 슬라이드는 이를 InfoNCE로 칭함 (van den Oord et al., 2018, CPC에서 유래). 구조가 동일해서 같은 계열로 취급.
> 보충: 논문의 기본 설정은 ResNet-50 encoder, 2-layer MLP projection head, augmentation은 random crop+resize, color distortion, Gaussian blur 조합이 핵심. (슬라이드에는 이 내용 없음)

---

## Training Linear Classifier on SimCLR Features (p.26)

- Train feature encoder on ImageNet (entire training set) using SimCLR.
- Freeze feature encoder, train a linear classifier on top with labeled data.

**그림**: 산점도. x축 = Number of Parameters (Millions), 눈금 25, 50, 100, 200, 400, 626 (로그 스케일로 보임). y축 = ImageNet Top-1 Accuracy (%), 눈금 55~75.
- 회색 X: **Supervised** (x≈25M, y≈76.5), 상한선 기준.
- 빨간 별 + 점선 연결: **SimCLR** (≈24M, ≈69.3), **SimCLR (2x)** (≈94M, ≈74.2), **SimCLR (4x)** (≈375M, ≈76.5). 파라미터를 늘릴수록 정확도 상승.
- 파란 점(기존 self-supervised 방법): CPCv2-L (~305M, ~71.5), MoCo (4x) (~375M, ~68.6), AMDIM (626M, ~68.1), CMC (~190M, ~68.4), PIRL-c2x (~94M, ~67.4), MoCo (2x) (~94M, ~65.4), PIRL-ens. (~50M, ~65.7), BigBiGAN (~86M, ~61.3), CPCv2 (~24M, ~63.8), PIRL (~24M, ~63.6), MoCo (~24M, ~60.6), LA (~24M, ~60.2), Rotation (~86M, ~55.4), InstDisc (~24M, ~54.0).

출처: Chen et al., ICML 2020

**설명**
- **Linear evaluation protocol**: self-supervised representation 품질 평가의 표준 방식. (1) 라벨 없이 ImageNet 전체로 encoder 사전학습, (2) encoder를 **freeze**, (3) 그 위에 **linear classifier 하나만** 라벨로 학습해 Top-1 accuracy 측정. encoder가 이미 선형 분리 가능한 feature를 배웠는지 보는 테스트.
- **결과 해석**: 같은 파라미터 수(~24M, ResNet-50)에서 SimCLR가 기존 방법들(CPCv2, MoCo, InstDisc 등)보다 **크게 우수** (~69.3 vs 최대 ~63.8). 폭(width)을 4배로 키운 SimCLR (4x)는 **supervised baseline(~76.5)과 거의 동급**. 즉 self-supervised가 supervised 성능에 근접했다는 주장.
- 슬라이드의 수치는 그림에서 읽은 근사값이라 정확한 값은 원 논문 참조. (논문 본문 기준 SimCLR ResNet-50 69.3, 2x 74.2, 4x 76.5로 알려져 있음)

> 보충: (2x), (4x)는 ResNet의 채널 폭 배수. 모델이 커질수록 self-supervised의 이득이 supervised보다 더 크게 늘어난다는 것이 논문의 관찰 중 하나.

---

## Training Linear Classifier on SimCLR Features (p.27)

- Finetune the encoder with 1% / 10% of labeled data on ImageNet.

**표**: ImageNet Top-5 accuracy, Label fraction 1% / 10%. (슬라이드 제목은 위와 같지만 실제 내용은 semi-supervised fine-tuning 결과)

| Method | Architecture | 1% (Top 5) | 10% (Top 5) |
|---|---|---|---|
| Supervised baseline | ResNet-50 | 48.4 | 80.4 |
| *Methods using other label-propagation:* | | | |
| Pseudo-label | ResNet-50 | 51.6 | 82.4 |
| VAT+Entropy Min. | ResNet-50 | 47.0 | 83.4 |
| UDA (w. RandAug) | ResNet-50 | - | 88.5 |
| FixMatch (w. RandAug) | ResNet-50 | - | 89.1 |
| S4L (Rot+VAT+En. M.) | ResNet-50 (4×) | - | 91.2 |
| *Methods using representation learning only:* | | | |
| InstDisc | ResNet-50 | 39.2 | 77.4 |
| BigBiGAN | RevNet-50 (4×) | 55.2 | 78.8 |
| PIRL | ResNet-50 | 57.2 | 83.8 |
| CPC v2 | ResNet-161(*) | 77.9 | 91.2 |
| SimCLR (ours) | ResNet-50 | 75.5 | 87.8 |
| SimCLR (ours) | ResNet-50 (2×) | 83.0 | 91.2 |
| SimCLR (ours) | ResNet-50 (4×) | **85.8** | **92.6** |

(SimCLR 세 행은 빨간 박스로 강조됨)

출처: Chen et al., ICML 2020

**설명**
- **Semi-supervised 평가**: 사전학습 후 **라벨 1% 또는 10%만** 사용해 encoder 전체를 fine-tune. 라벨이 적은 상황에서 pretrained representation이 얼마나 도움이 되는지 측정. (p.26의 linear eval은 encoder freeze, 여기서는 **fine-tune**이라는 점이 차이)
- **결과**: SimCLR (4×)가 1%에서 85.8, 10%에서 92.6으로 최고. ResNet-50 supervised baseline(48.4 / 80.4)과 비교하면 **라벨 1%에서 동일 ResNet-50(1×)도 +27.1(75.5), 4×는 +37.4**, 압도적. representation learning만 쓴 방법 중 CPC v2 (ResNet-161)를 넘어섬(1%: 2×/4×, 10%: 4×만 초과, 2×는 91.2로 동률). 별도의 label propagation 기법(Pseudo-label, UDA, FixMatch, S4L 등)보다도 SimCLR 2×/4×가 우수 (1× 10% 87.8은 UDA 88.5 등보다 낮음).
- 의미: **라벨이 부족한 환경에서 self-supervised pretraining이 강력**. 슬라이드 제목이 "Training Linear Classifier"인데 내용은 fine-tuning이므로 제목과 불일치 ← 제목 오해 소지 (원문 유지)
- ← 슬라이드 표의 CPC v2 행은 빨간 박스 밖이지만 1%에서 77.9로 SimCLR (ResNet-50 2×, 83.0)보다 낮고 ResNet-50 (1×, 75.5)보다 높음. 아키텍처(ResNet-161)가 달라 공정 비교는 아님.

---

## SimCLR Design Choices: Projection Head (p.28)

- Linear / non-linear projection heads improve representation learning.

**그림**: 막대그래프. x축 = Projection output dimensionality (32, 64, 128, 256, 512, 1024, 2048). y축 = Top 1 (30~70). 범례 Projection: Linear(파랑), Non-linear(주황), None(초록).
- 각 차원에서 Non-linear(~63.5~65)가 Linear(~60~61)보다 항상 높음.
- None(projection 없음)은 2048 차원에서만 존재, 약 50으로 가장 낮음.
- 출력 차원이 달라져도 Linear/Non-linear 성능은 거의 일정 (차원에 둔감).

오른쪽 텍스트 "A possible explanation":
- contrastive learning objective may discard useful information for downstream tasks
- representation space z is trained to be invariant to data transformation.
- by leveraging the projection head $g(\cdot)$, more information can be preserved in the $h$ representation space

출처: Chen et al., ICML 2020

**설명**
- **핵심**: projection head $g$를 쓰면 **$g$ 이전의 $h$를 평가에 쓸 때** 성능이 크게 오름 (None은 $h$ 자체에 loss를 거는 경우로, 가장 낮음). **non-linear > linear > none**.
- **왜?** (슬라이드의 가설) contrastive loss는 서로 다른 augmentation에서 나온 view를 가깝게 만들도록 $z$를 학습 → $z$는 **augmentation에 invariant**해짐. 그러면 color, 회전 같은 정보가 $z$에서 버려질 수 있음. 이런 정보는 downstream task에 유용할 수 있음. $g$가 이 "invariance 학습의 부담"을 흡수하므로 **$h$에는 더 풍부한 정보가 보존**됨.
- 그래서 학습 후 **$g$를 버리고 $h$ (= $f$의 출력)를 사용** (p.25의 "throw away $g$"와 연결).
- projection 출력 차원은 성능에 크게 영향 없음 → head의 존재/비선형성이 중요.

> 보충: 논문에서는 projection head 앞(h)과 뒤(z)의 representation으로 augmentation 종류(예: 회전, 색 변환)를 예측하는 실험을 해서, $h$가 $z$보다 훨씬 많은 변환 정보를 담고 있음을 보였음 (Chen et al., 2020, Table 3). 슬라이드에는 없음.

---

## SimCLR Design Choices: Large Batch Size (p.29)

- Large training batch size is crucial for SimCLR!
  - Large batch size causes large memory footprint during backpropagation: requires distributed training on TPUs (ImageNet experiments)

**그림**: 막대그래프 (논문 Figure 9). x축 = Training epochs (100, 200, ..., 1000), y축 = Top 1 (50.0~70.0). 각 epoch 그룹 안에 batch size별 막대 6개: 256(파랑), 512(주황), 1024(초록), 2048(진갈색/주황), 4096(보라), 8192(연갈색).
- 대체로 batch size가 클수록 Top-1이 높음 (큰 batch끼리는 역전되기도 함, 예: 400 epoch에서 4096이 8192보다 약간 높음). 특히 **epoch가 짧을 때(100 epoch)** 차이가 큼: batch 256은 ~57.5, batch 8192는 ~64.8 정도.
- epoch가 늘면 격차가 줄어듦 (1000 epoch에서 ~67 vs ~69).
- 큰 batch(2048~8192)끼리는 epoch가 길어지면 거의 포화.

캡션: *Figure 9.* Linear evaluation models (ResNet-50) trained with different batch size and epochs. Each bar is a single run from scratch.¹⁰

출처: Chen et al., ICML 2020

**설명**
- **왜 큰 batch가 중요한가**: SimCLR는 memory bank 없이 **in-batch negative**에 의존. batch가 클수록 negative가 많아져(2N−2개) 더 어려운/다양한 negative와 대비할 수 있고, softmax 분모의 근사가 좋아져 gradient 신호가 개선됨.
- **같은 성능 도달에 필요한 epoch 수가 줄어듦**: 큰 batch면 짧은 학습으로도 효율적. 반대로 작은 batch는 더 오래 학습해야 따라잡음.
- **단점**: backprop 시 **메모리 사용량이 큼** → 논문은 TPU 여러 장의 분산 학습 사용 (슬라이드 명시). 일반 GPU 환경에서는 재현이 어려움. 이것이 이후 MoCo(queue로 negative 확보)나 BYOL(negative 불필요) 같은 방법이 등장한 동기 중 하나.

> 보충: 논문은 큰 batch에서 LARS optimizer를 사용. (슬라이드에는 없음)

---

## BYOL (Bootstrap Your Own Latent) (p.30)

- A new state-of-the-art results without using negative samples.
- Two neural networks interact and learn from each other.
  - Online (parameterized by $\theta$)
  - Target (parameterized by $\xi$)

**그림**: BYOL 구조도. 왼쪽 input image $x$에서 두 갈래로 분기. 위쪽 파란 영역이 **online**, 아래쪽 분홍 영역이 **target**.
- 열(column) 구성: **view → representation → projection → prediction**.
- online 경로: $x \xrightarrow{t} v \xrightarrow{f_\theta} y \xrightarrow{g_\theta} z \xrightarrow{q_\theta} q_\theta(z)$.
- target 경로: $x \xrightarrow{t'} v' \xrightarrow{f_\xi} y' \xrightarrow{g_\xi} z' \xrightarrow{\text{sg}} \text{sg}(z')$. ($z'$ 뒤에 두 줄 사선 "//" 표시와 "sg" 라벨 = stop-gradient.)
- 오른쪽 끝: online의 $q_\theta(z)$와 target의 $\text{sg}(z')$ 사이를 **점선 화살표(loss)** 로 연결. loss는 target 쪽으로 향함.
- 오른쪽 bullet: An encoder $f_\theta$; A projector $g_\theta$; A predictor $q_\theta$.
- 수식: $\theta: \xi \leftarrow \tau\xi + (1-\tau)\theta.$ — "Exponential Moving Average (EMA)"

출처: Grill et al., NeurIPS 2020, Bootstrap Your Own Latent

**설명**
- **BYOL의 핵심**: **negative sample 없이** SOTA 달성. SimCLR의 대조(positive는 당기고 negative는 밀기) 대신, **한 view의 online 예측이 다른 view의 target representation을 맞추도록** 학습.
- **두 네트워크**: online($\theta$: encoder $f_\theta$, projector $g_\theta$, **predictor $q_\theta$** 포함)와 target($\xi$: encoder $f_\xi$, projector $g_\xi$, predictor 없음). **predictor는 online에만 존재**하는 비대칭 구조가 중요 (collapse 방지에 기여).
- **target 업데이트**: gradient가 아니라 **EMA**: $\xi \leftarrow \tau\xi + (1-\tau)\theta$. $\tau$는 1에 가까운 값(예: 0.99 이상)이라 target은 online을 **천천히 따라감**. 여기의 $\tau$는 SimCLR의 temperature $\tau$와 **다른 기호/의미**임에 주의 (momentum coefficient).
- **stop-gradient (sg)**: target 쪽으로는 gradient가 흐르지 않음 → target은 backprop 없이 EMA로만 변함.
- ← 슬라이드 수식 "$\theta: \xi \leftarrow \dots$"의 "$\theta:" 표기는 모호 (원문 유지). 의미는 "online 파라미터 $\theta$로 target $\xi$를 갱신".

> 보충: BYOL 논문(Grill et al., NeurIPS 2020)은 negative가 없으면 모든 입력을 같은 값으로 매핑하는 **collapse**가 trivial solution이 될 수 있는데, predictor + EMA target 덕분에 이를 피한다고 보고함. 이론적 완전한 설명은 논문에서도 empirical. (MoCo의 momentum encoder와 유사한 아이디어: He et al., CVPR 2020)

---

## BYOL (Bootstrap Your Own Latent) (p.31)

- Why Two Networks?
- Online Network
  - Learns through gradient descent
  - Predicts the target representation
- Target Network
  - Provides a stable learning target
  - Updated by EMA
  - No backpropagation

출처: Grill et al., NeurIPS 2020

**설명**
- **Online network**: 실제로 **gradient descent로 학습**되는 쪽. target representation을 **예측**하는 것이 목적 (그래서 predictor $q_\theta$가 붙음).
- **Target network**: online의 느린 이동평균. 타깃이 매 step 급격히 변하지 않아 **안정적인 학습 목표**를 제공. backprop 없음 → 계산/메모리 절약.
- **왜 두 개인가?** 하나의 네트워크가 자기 출력을 타깃으로 삼으면 타깃도 같이 요동쳐서 학습이 불안정하고 collapse하기 쉬움. 두 네트워크로 분리하고 target을 천천히 갱신하면 **bootstrapping**(자기 자신의 과거 버전을 target으로 사용) 구조가 안정화됨. 강화학습의 target network (DQN)와 같은 발상.
- 질문형 첫 줄 "Why Two Networks?"는 위 세 가지 설명의 제목 역할.

---

## BYOL (Bootstrap Your Own Latent) (p.32)

- Create two augmented views: $\mathbf{v} = t(\mathbf{x});\ \mathbf{v}' = t'(\mathbf{x})$ with augmentations sampled $t \sim \mathcal{T}, t' \sim \mathcal{T}'$;
- Then they are encoded into representations, $\mathbf{y}_\theta = f_\theta(\mathbf{v}),\ \mathbf{y}' = f_\xi(\mathbf{v}')$;
- Then they are projected into latent variables, $\mathbf{z}_\theta = g_\theta(\mathbf{y}_\theta),\ \mathbf{z}' = g_\xi(\mathbf{y}')$;
- The online network outputs a prediction $q_\theta(\mathbf{z}_\theta)$;
- Both $q_\theta(\mathbf{z}_\theta)$ and $\mathbf{z}'$ are L2-normalized, giving us $\bar{q}_\theta(\mathbf{z}_\theta) = q_\theta(\mathbf{z}_\theta)/\lVert q_\theta(\mathbf{z}_\theta)\rVert$ and $\bar{\mathbf{z}}' = \mathbf{z}'/\lVert\mathbf{z}'\rVert$;
- The loss $\mathcal{L}_\theta^{\text{BYOL}}$ is MSE between L2-normalized prediction $\bar{q}_\theta(\mathbf{z})$ and $\bar{\mathbf{z}}'$;
- The other symmetric loss $\tilde{\mathcal{L}}_\theta^{\text{BYOL}}$ can be generated by switching $\mathbf{v}'$ and $\mathbf{v}$; that is, feeding $\mathbf{v}'$ to online network and $\mathbf{v}$ to target network.
- The final loss is $\mathcal{L}_\theta^{\text{BYOL}} + \tilde{\mathcal{L}}_\theta^{\text{BYOL}}$ and only parameters $\theta$ are optimized.

**BYOL does not use negative pairs.**
**BYOL directly boostrapps the latent representation.** ← "boostrapps" 오타 (bootstraps), 원문 유지

출처: Grill et al., NeurIPS 2020

**설명**
- **단계별 흐름**: 이미지 $x$ → 두 augmentation $t, t'$로 view $v, v'$ → online은 $f_\theta, g_\theta$, target은 $f_\xi, g_\xi$로 $z_\theta$, $z'$ 생성 → online에만 predictor $q_\theta$를 적용.
- 슬라이드의 $\mathcal{T}'$는 두 번째 view용 augmentation 분포를 따로 둘 수 있음을 뜻함(BYOL 논문은 $\mathcal{T}$와 $\mathcal{T}'$의 augmentation 확률/강도를 약간 다르게 설정). SimCLR는 둘 다 같은 $\mathcal{T}$.
- **Loss**: L2 normalize 후 MSE:
  $$\mathcal{L}_\theta^{\text{BYOL}} = \lVert \bar{q}_\theta(z_\theta) - \bar{z}' \rVert_2^2 = 2 - 2\cdot\frac{\langle q_\theta(z_\theta),\, z'\rangle}{\lVert q_\theta(z_\theta)\rVert\,\lVert z'\rVert}$$
  즉 **정규화된 벡터 간 MSE = 2 − 2·cosine similarity** 이므로, SimCLR의 cosine similarity와 연결되지만 **negative 항(분모)이 전혀 없음**. (이 유도는 슬라이드 외 보충)
- **대칭화**: $v$와 $v'$의 역할을 바꿔 $\tilde{\mathcal{L}}$을 구하고 합산 → 한 쌍에서 두 방향의 학습 신호를 모두 사용(SimCLR가 $\ell(2k-1,2k)+\ell(2k,2k-1)$ 두 방향을 쓰는 것과 같은 취지).
- **최적화 대상**: 오직 $\theta$. $\xi$는 gradient 없이 EMA로만 갱신(p.30–31).
- **마무리 요점**: negative pair 없이 **latent representation 자체를 bootstrap**(target의 예측값을 학습 목표로 삼음). SimCLR 대비 장점은 **큰 batch 불필요**, augmentation 선택에 덜 민감 (논문 주장).
- ← 슬라이드는 "$\mathbf{z}' = g_\xi(\mathbf{y}')$"에 stop-gradient를 명시하지 않았으나 p.30 그림에서 sg로 표시됨.

> 보충: BYOL의 최종 downstream 평가는 online encoder $f_\theta$를 사용. SimCLR는 $g$를 버리고 $f$ 사용, BYOL은 $f_\theta$만 사용하고 $g_\theta$, $q_\theta$는 버림.

---

## FIGURES

- p.25: SimCLR Algorithm 1 박스 (색 주석 포함) — 상 (알고리즘 흐름과 loss 정의 정리용, 정식 노트에는 LaTeX로 재작성하되 이미지도 보조로 가치 있음)
- p.26: Linear evaluation 산점도 (파라미터 수 vs ImageNet Top-1, SimCLR 대 기존 방법) — 상
- p.27: Semi-supervised (1%/10% label) 비교 표 — 중
- p.28: Projection head 막대그래프 (None / Linear / Non-linear) — 상
- p.29: Batch size × epoch 막대그래프 (Figure 9) — 중
- p.30: BYOL 구조도 (online/target, sg, EMA) — 상

## EXAM POINTS

1. **SimCLR의 loss(NT-Xent/InfoNCE) 수식과 의미**: $\ell(i,j)=-\log\frac{\exp(s_{i,j}/\tau)}{\sum_{k=1}^{2N}\mathbb{1}_{[k\neq i]}\exp(s_{i,k}/\tau)}$. positive 1개, negative $2N-2$개, 분모에서 자기 자신 제외, cosine similarity, temperature $\tau$ 역할. 대칭 평균 $\frac{1}{2N}\sum[\ell(2k-1,2k)+\ell(2k,2k-1)]$.
2. **SimCLR 파이프라인**: augmentation $t,t'\sim\mathcal{T}$ → encoder $f$ → projection head $g$ → loss. 학습 후 **$g$ 버리고 $f$ 사용**.
3. **Why projection head? ($g$를 쓰는 이유)**: contrastive loss가 $z$를 augmentation에 invariant하게 만들어 유용한 정보를 버릴 수 있음 → $g$가 이를 흡수해 $h$에 더 많은 정보 보존. non-linear > linear > none.
4. **Why large batch size?**: in-batch negative 수 증가, 짧은 epoch에서 특히 효과. 단점은 메모리 및 TPU 분산 학습 필요. (MoCo/BYOL의 등장 동기와 연결)
5. **Linear evaluation vs semi-supervised fine-tuning**: linear eval은 encoder freeze + linear classifier, semi-supervised는 1%/10% 라벨로 encoder fine-tune. SimCLR (4×)는 linear eval에서 supervised에 근접(~76.5), 1%/10% 라벨에서 최고(85.8/92.6, Top-5).
6. **BYOL 구조**: online($f_\theta,g_\theta,q_\theta$) vs target($f_\xi,g_\xi$), predictor는 online에만 존재, target은 **EMA** $\xi\leftarrow\tau\xi+(1-\tau)\theta$ + **stop-gradient**.
7. **BYOL loss**: L2-normalized prediction과 target projection 사이의 MSE (= $2-2\cos$), 두 view를 바꿔 대칭화, $\theta$만 최적화. **negative pair 없음**.
8. **SimCLR vs BYOL 비교 (why 질문)**: negative 필요 여부, batch size 의존성, 네트워크 구조(단일 encoder vs online/target 비대칭), target의 안정화 방법(EMA, sg, predictor)과 collapse 방지.

## VERIFIER REPORT
- p.27 설명: "+27 이상" 표현이 4× 기준 비교로는 부정확(+37.4)해서 1×/4× 각각 수치로 정정, CPC v2 초과 조건(10%에서는 4×만) 및 label-propagation 방법 대비 우위가 SimCLR 2×/4×에 한정됨을 명시(1× 10% 87.8 < UDA 88.5, FixMatch 89.1).
- p.29 그림 설명: "모든 epoch에서 batch가 클수록 높음"을 "대체로"로 완화 (400 epoch 등에서 4096 > 8192).
- 그 외 p.25–32 수식, 표 수치, 그림 설명, 인용(Chen ICML 2020, Grill NeurIPS 2020, van den Oord 2018, He CVPR 2020) 이상 없음.
