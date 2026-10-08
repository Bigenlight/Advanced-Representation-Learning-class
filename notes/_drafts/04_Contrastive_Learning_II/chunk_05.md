<!-- DRAFT | lecture: 04_Contrastive_Learning_II | chunk: 05 | pages 33–40 -->
<!-- VERIFIED | fixes: 1 | verifier: sonnet -->

## BYOL (Bootstrap Your Own Latent) — 성능 결과 (p.33)

출처: Grill et al., NeurIPS 2020, *Bootstrap Your Own Latent*

**표 1 (왼쪽): Linear evaluation on ImageNet** (Top-1 / Top-5, %)

| Method | Top-1 | Top-5 |
|---|---|---|
| Local Agg. | 60.2 | - |
| PIRL [35] | 63.6 | - |
| CPC v2 [32] | 63.8 | 85.3 |
| CMC [11] | 66.2 | 87.0 |
| SimCLR [8] | 69.3 | 89.0 |
| MoCo v2 [37] | 71.1 | - |
| InfoMin Aug. [12] | 73.0 | 91.1 |
| **BYOL (ours)** | **74.3** | **91.6** |

**표 2 (오른쪽): Semi-supervised training on ImageNet** (1% / 10% 라벨)

| Method | Top-1 1% | Top-1 10% | Top-5 1% | Top-5 10% |
|---|---|---|---|---|
| Supervised [77] | 25.4 | 56.4 | 48.4 | 80.4 |
| InstDisc | - | - | 39.2 | 77.4 |
| PIRL [35] | - | - | 57.2 | 83.8 |
| SimCLR [8] | 48.3 | 65.6 | 75.5 | 87.8 |
| **BYOL (ours)** | **53.2** | **68.8** | **78.4** | **89.0** |

(BYOL 행은 파란색 하이라이트 처리됨.)

**설명**
- 두 표 모두 **BYOL이 negative sample 없이도** 기존 contrastive 계열(SimCLR, MoCo v2, CMC, PIRL 등)을 능가함을 보여주는 결과 슬라이드임.
- **Linear evaluation**: encoder를 freeze하고 그 위에 linear classifier만 학습해 ImageNet 정확도를 측정하는 표준 프로토콜. representation 자체의 품질을 재는 지표임. BYOL은 Top-1 **74.3%** (SimCLR 69.3 대비 +5.0p, MoCo v2 71.1 대비 +3.2p).
- **Semi-supervised**: 라벨의 1% / 10%만으로 fine-tuning. 라벨이 극단적으로 적을 때(1%)도 supervised-only(25.4)보다 훨씬 높고(53.2), SimCLR(48.3)보다도 높음 → pretrain된 representation이 label-efficient함.
- 표에서 InstDisc, PIRL의 일부 칸이 "-"인 것은 원 논문이 해당 수치를 보고하지 않았기 때문임.
> 보충: 논문 기준 backbone은 ResNet-50 (표준 설정). 논문 본문에서 더 큰 ResNet(200 등)으로 키우면 Top-1 79.6%까지 올라간다고 보고함. 또 BYOL은 SimCLR 대비 batch size, augmentation 종류 변화에 덜 민감하다는 점이 주요 주장임.

## BYOL (Bootstrap Your Own Latent) — 요약 (p.34)

- Learn by predicting another augmented view.
  - Learn by prediction
  - No negative samples
  - Online → Target Network

**설명**
- BYOL의 핵심 요약 슬라이드임. 한 이미지의 두 augmented view 중 **한쪽(online network)이 다른 쪽(target network)의 representation을 예측**하도록 학습함.
- **Learn by prediction**: contrastive의 "같은 건 가깝게, 다른 건 멀게"가 아니라, online branch에 붙은 **predictor**가 target branch의 projection을 맞히는 regression 문제로 바꿈 (MSE 계열 loss).
- **No negative samples**: negative가 없으므로 큰 batch / memory bank / queue가 필요 없음. 대신 모든 입력을 같은 상수로 보내는 **collapse** 위험이 생기는데, BYOL은 (1) predictor의 비대칭 구조, (2) target network를 gradient가 아닌 **EMA**로만 갱신(stop-gradient)하는 것으로 이를 피함 (앞 슬라이드 p.31–32의 two-network 설명과 연결).
- **Online → Target**: 화살표 방향이 중요함. gradient는 online으로만 흐르고, target은 online의 EMA(느리게 움직이는 평균)로 따라감 → target이 안정적인 학습 목표를 제공함.
> 보충: online 파라미터 $\theta$, target 파라미터 $\xi$일 때 $\xi \leftarrow \tau\xi + (1-\tau)\theta$ (논문 기본 $\tau$는 0.996에서 1로 cosine 스케줄로 증가). Loss는 $\ell_2$-normalize한 두 벡터의 MSE, 즉 $2 - 2\cdot\frac{\langle q_\theta(z_\theta), z'_\xi\rangle}{\|q_\theta(z_\theta)\|\|z'_\xi\|}$이며 두 view를 서로 바꿔 대칭화함.

## Barlow Twins — 아이디어와 loss (p.35)

출처: Zbontar et al., ICML 2021, *Barlow Twins: Self-Supervised Learning via Redundancy Reduction*

- Generate **two augmented views** of the same image.
- **Encode** both views using a shared encoder.
- Compute the **cross-correlation** matrix between the two representations.
- Learn an **identity correlation** matrix.
- **Key Idea**: Learn invariant yet non-redundant representations without negative samples.

**그림 (왼쪽 아래, 파이프라인)**: 왼쪽에서 원본 Images $X$가 $T \sim \mathcal{T}$ (augmentation 분포에서 샘플한 변환)를 거쳐 위/아래 두 갈래로 분기 → 위쪽은 Distorted images $Y^B$, 아래쪽은 $Y^A$. 각각 동일한 Net $f_\theta$(사다리꼴, 같은 가중치)를 통과해 Representations가 됨 (슬라이드 표기상 위쪽 $Y^B \to Z^A$, 아래쪽 $Y^A \to Z^B$로 첨자가 엇갈려 그려져 있음). 두 $Z$ 가 만나 **Empirical cross-corr. $\mathcal{C}$** (회색조 행렬, 축은 feature dimension)를 만들고, 오른쪽의 **Target cross-corr. $\mathcal{I}$** (대각선만 검은 identity 행렬)과 $\mathcal{L}_{\mathcal{BT}}$로 비교함. Net 아래에 "backprop." 화살표가 양쪽 branch 모두에 표시됨(두 branch 모두 gradient 받음, 비대칭 구조 없음).

**수식**

$$\mathcal{L}_{\mathrm{BT}} = \underbrace{\sum_i (1-\mathcal{C}_{ii})^2}_{\text{invariance term}} + \lambda \underbrace{\sum_i \sum_{i\neq j} \mathcal{C}_{ij}^2}_{\text{redundancy reduction term}}$$

$$\text{where } \mathcal{C}_{ij} = \frac{\sum_b \mathbf{z}^A_{b,i}\,\mathbf{z}^B_{b,j}}{\sqrt{\sum_b (\mathbf{z}^A_{b,i})^2}\,\sqrt{\sum_b (\mathbf{z}^B_{b,j})^2}}$$

슬라이드 주석: "C be be a cross-correlation matrix computed between outputs from two identical networks along the batch dimension." ← 원문 오타("be be")

**설명**
- 기호: $b$ = batch 내 샘플 index, $i,j$ = feature(embedding 차원) index, $\mathbf{z}^A_{b,i}$ = view A의 $b$번째 샘플의 $i$번째 feature. $\mathcal{C}$는 $D\times D$ 행렬 ($D$ = embedding 차원)이고 $\mathcal{C}_{ij}\in[-1,1]$.
- $\mathcal{C}_{ij}$는 **batch 방향으로** "view A의 $i$번째 feature 벡터(길이 $B$)"와 "view B의 $j$번째 feature 벡터"의 정규화된 내적(=cosine/상관계수)임. 즉 샘플 간 비교가 아니라 **feature 축 간 비교**라는 점이 SimCLR 등과 근본적으로 다름.
- **Invariance term** $\sum_i(1-\mathcal{C}_{ii})^2$: 대각성분을 1로 → 같은 feature가 두 augmentation에서 **같이 움직이도록**(augmentation에 불변) 강제함.
- **Redundancy reduction term** $\lambda\sum_{i\ne j}\mathcal{C}_{ij}^2$: 비대각성분을 0으로 → 서로 다른 feature들이 **서로 상관 없도록(decorrelated)** 만들어 각 차원이 다른 정보를 담게 함. $\lambda$는 두 항의 균형 계수.
- 설계 이유: 모든 feature가 같은 값/같은 방향으로 붕괴(collapse)하면 비대각성분이 커져 penalty를 받음 → **negative pair, predictor, stop-gradient, EMA 없이도 collapse가 자연스럽게 방지**됨. 이전 방법(SimCLR/MoCo)이 "다른 샘플은 멀리"로 collapse를 막은 것과 대비됨.
> 보충: 논문에서는 $\mathcal{C}$ 계산 전에 각 feature를 batch 방향으로 mean-centering + 표준편차 정규화(batch-norm과 유사)함. 이 연산 때문에 $\mathcal{C}_{ij}$가 엄밀한 상관계수가 됨. 이 목적함수는 신경과학자 H. Barlow의 redundancy reduction 원리(1961)에서 이름을 따옴. 논문 기본 $\lambda \approx 5\times10^{-3}$, projector는 8192차원 MLP를 사용.

## Barlow Twins — Cross-Correlation Matrix 계산 도식 (p.36)

출처: Zbontar et al., ICML 2021

**그림 (3단계 도식)**
1. **Embeddings from Two Views**: View A(녹색 표)와 View B(파란 표) 두 개의 표. 행 = img1..imgB (**Batch dimension, $B$ samples**), 열 = $f_1, f_2, f_3, \dots, f_D$ (**Feature dimension $D$**). A의 값은 1,2,3 / 4,5,6 / 7,8,9 / 10,11,12 … $a_{BD}$, B의 값은 $b_{11}\dots b_{BD}$로 표기됨.
2. **Compute Cross-Correlation Along the Batch Dimension** (보라색 화살표): "각 feature 쌍 $(i,j)$에 대해 batch 전체에 걸쳐 cosine similarity를 계산". 말풍선 "How $C_{ij}$ is computed": $\mathbf{z}^A_{:,i}$ = view A에서 $i$번째 feature의 batch 전체 벡터(길이 $B$), $\mathbf{z}^B_{:,j}$ = view B에서 $j$번째 feature 벡터일 때
$$C_{ij} = \frac{\mathbf{z}^A_{:,i}\cdot \mathbf{z}^B_{:,j}}{\|\mathbf{z}^A_{:,i}\|_2\,\|\mathbf{z}^B_{:,j}\|_2}$$
3. **Cross-Correlation Matrix $C$**: 행 = View A의 features, 열 = View B의 features인 $D\times D$ 행렬($C_{11}\dots C_{DD}$). **대각선(녹색, $i=j$)** = same feature correlation, **비대각선(연노랑, $i\neq j$)** = different feature correlation. "$C$ is a $D\times D$ matrix where $D$ = feature dimension."
- 하단 박스: **Diagonal ($C_{ii}$) → Invariance**: 두 view 사이에서 같은 feature의 상관을 측정, $C_{ii}\to1$ (서로 다른 augmentation 후에도 feature가 비슷해야 함). **Off-diagonal ($C_{ij}, i\neq j$) → Redundancy Reduction**: 서로 다른 feature 간 상관, $C_{ij}\to0$ (feature들이 uncorrelated, 즉 고유한 정보를 담음).
- 오른쪽 아래 **Goal**: "Make $C$ close to the identity matrix $I_D$" — 대각 1, 나머지 0인 행렬 그림.

**설명**
- p.35의 수식을 "표 → 행렬"로 시각화한 슬라이드임. 핵심은 **어느 축을 따라 내적하는가**: 행(샘플) 방향이 아니라 **열(feature) 방향 벡터끼리** 내적하므로 결과가 $B\times B$가 아닌 $D\times D$임. (SimCLR는 샘플끼리 $B\times B$ similarity를 만드는 것과 대조적.)
- 해석: 대각성분 = "이 feature는 augmentation에 대해 일관적인가", 비대각성분 = "이 feature가 다른 feature와 중복 정보인가". 목표가 identity이므로 결과적으로 **불변이면서 서로 직교(decorrelated)하는 표현**을 얻음.
- 표의 숫자(1,2,3…)는 예시용 더미 값이며 실제 값이 아님.
> 보충: 이 도식의 $\mathbf{z}$는 슬라이드 설명상 cosine similarity로 표기했으나, 원 논문 식(1)은 batch 방향 mean-centering 후의 상관계수임. 정규화 방식의 미세한 차이이고 개념은 동일함. 비대각 항을 0으로 만드는 것은 feature 간 **decorrelation**이며 완전한 statistical independence까지 보장하지는 않음.

## Barlow Twins — 성능 비교 (p.37)

- Barlow Twins is competitive with SOTA methods for self-supervised learning. It naturally avoids trivial constants (i.e. collapsed representations), and is robust to different training batch sizes.

**Table 1** (caption: *Top-1 and top-5 accuracies (in %) under linear evaluation on ImageNet. All models use a ResNet-50 encoder. Top-3 best self-supervised methods are underlined.*)

| Method | Top-1 | Top-5 |
|---|---|---|
| Supervised | 76.5 | |
| MoCo | 60.6 | |
| PIRL | 63.6 | - |
| SimCLR | 69.3 | 89.0 |
| MoCo v2 | 71.1 | 90.1 |
| SimSiam | 71.3 | - |
| SwAV (w/o multi-crop) | 71.8 | - |
| BYOL | 74.3 (밑줄) | 91.6 |
| SwAV | 75.3 (밑줄) | - |
| BARLOW TWINS (ours) | 73.2 (밑줄) | 91.0 |

**Table 2** (caption: *Semi-supervised learning on ImageNet using 1% and 10% training examples. Results for the supervised method are from (Zhai et al., 2019). Best results are in bold.*)

| Method | Top-1 1% | Top-1 10% | Top-5 1% | Top-5 10% |
|---|---|---|---|---|
| Supervised | 25.4 | 56.4 | 48.4 | 80.4 |
| PIRL | - | - | 57.2 | 83.8 |
| SimCLR | 48.3 | 65.6 | 75.5 | 87.8 |
| BYOL | 53.2 | 68.8 | 78.4 | 89.0 |
| SwAV | 53.9 | **70.2** | 78.5 | **89.9** |
| BARLOW TWINS (ours) | **55.0** | 69.7 | **79.2** | 89.3 |

**설명**
- 슬라이드의 주장 3가지: (1) SOTA와 **competitive**, (2) **trivial constant(= collapsed representation)를 자연스럽게 회피**, (3) **training batch size에 robust**.
- Linear eval: Barlow Twins 73.2%로 BYOL(74.3), SwAV(75.3)보다 약간 낮지만 SimCLR(69.3), MoCo v2(71.1), SimSiam(71.3)보다 높고 상위 3위권(밑줄). 간단한 loss(negative, predictor, EMA 없음)로 이 수준이라는 점이 포인트.
- Semi-supervised: 1% 라벨에서 Top-1 55.0 / Top-5 79.2로 **최고**, 10%에서는 SwAV가 근소하게 우위.
- 이유(batch size robustness): SimCLR는 negative 수 = batch size에 의존해 큰 batch가 필수지만, Barlow Twins의 loss는 $D\times D$ 상관행렬 기반이라 negative 개수와 무관함. 대신 상관 추정을 위해 batch가 너무 작으면 불안정해질 수 있으나 논문은 비교적 작은 batch(예: 256)에서도 성능 하락이 작다고 보고함.
> 보충: SwAV는 multi-crop augmentation을 쓰면 75.3, 쓰지 않으면 71.8로, 공정 비교를 위해 두 값을 모두 표기한 것임. SimSiam은 Chen & He, CVPR 2021 (stop-gradient + predictor, no negative, no EMA). 두 표의 BYOL, SimCLR 수치는 p.33과 동일함.

## Barlow Twins — 요약 (p.38)

- Uses Cross-Correlation loss
- No negative samples
- Aligns positive pairs only
- Removes redundancy between feature dimensions
- Learns decorrelated representations

**설명**
- Barlow Twins 정리 슬라이드임. **positive pair(같은 이미지의 두 view)만 정렬**하고 negative는 쓰지 않음. 서로 다른 샘플을 밀어내는 역할은 **feature 차원 간 redundancy 제거(decorrelation)**가 대신 수행해 collapse를 막음.
- 방법 비교 관점: SimCLR/MoCo = sample-level contrast (negative 필요) / BYOL = predictor + EMA target (비대칭) / **Barlow Twins = feature-level(차원 간) 상관 제어, 대칭 구조, EMA·predictor 불필요**.
> 보충: 이후 VICReg (Bardes et al., ICLR 2022)가 variance / invariance / covariance 항으로 이 아이디어를 확장함.

## DINO: Self-Distillation with No Labels — 개요 (p.39)

출처: Caron et al., ICCV 2021, *Emerging Properties in Self-Supervised Vision Transformers*

- DINO (2021)
  - Learn visual representations through **self-distillation.**
  - Use a **student** and a **teacher** network with identical architectures.
  - Train the student to match the teacher's output distribution.
  - Update the teacher using an **EMA** of the student parameters.
- **Key Idea**: Self-distillation without labels.

**설명**
- **Knowledge distillation**은 보통 크고 잘 학습된 teacher의 출력을 작은 student가 따라 하게 하는 기법인데, DINO는 **teacher를 별도 사전학습 없이 student 자신의 EMA로 만들어** 라벨 없이 self-distillation을 수행함 (이름의 "DI-NO" = self-**DI**stillation with **NO** labels).
- 학습 목표는 BYOL의 feature 회귀와 달리 **출력 확률분포(distribution) 일치**임. student의 출력 분포가 teacher의 출력 분포와 같아지도록 cross-entropy를 최소화함.
- teacher는 gradient로 학습되지 않고 student의 EMA로만 갱신됨 → BYOL의 target network와 같은 역할(안정적 목표). 구조는 동일 아키텍처(ViT 또는 ResNet).
> 보충: 논문의 핵심 발견은 self-supervised ViT의 마지막 self-attention map에 별도 supervision 없이도 **object segmentation 같은 의미적 영역**이 드러나고, k-NN 분류에서도 매우 높은 성능을 보인다는 것(제목의 "Emerging Properties"). 원 논문은 multi-crop(global 2 + local 여러 개)을 사용: student는 모든 crop, teacher는 global crop만 받아 "local-to-global" 대응을 학습함. teacher 출력에는 **centering + sharpening**을 적용해 collapse를 방지함 (다음 p.40).

## DINO: Self-Distillation with No Labels — 구조와 Pseudocode (p.40)

출처: Caron et al., ICCV 2021

- DINO (2021)

**그림 (왼쪽, 구조도)**: 맨 아래 입력 $x$에서 두 화살표가 갈라져 $x_1$, $x_2$ (두 augmented view)가 됨. $x_1$ → **student $g_{\theta_s}$** (회색 박스) → **softmax** → $p_1$. $x_2$ → **teacher $g_{\theta_t}$** (회색 박스) → **centering** → **softmax** → $p_2$. student에서 teacher로 가는 화살표에 **"ema"** 라벨. teacher 쪽 softmax 위에 **"sg"**(stop-gradient, 두 줄 슬래시 표시)가 있어 gradient가 teacher로 흐르지 않음. 상단에 **loss: $-p_2 \log p_1$**.

**Algorithm 1: DINO PyTorch pseudocode w/o multi-crop** (오른쪽)

```
# gs, gt: student and teacher networks
# C: center (K)
# tps, tpt: student and teacher temperatures
# l, m: network and center momentum rates
gt.params = gs.params
for x in loader: # load a minibatch x with n samples
    x1, x2 = augment(x), augment(x) # random views

    s1, s2 = gs(x1), gs(x2) # student output n-by-K
    t1, t2 = gt(x1), gt(x2) # teacher output n-by-K

    loss = H(t1, s2)/2 + H(t2, s1)/2
    loss.backward() # back-propagate

    # student, teacher and center updates
    update(gs) # SGD
    gt.params = l*gt.params + (1-l)*gs.params
    C = m*C + (1-m)*cat([t1, t2]).mean(dim=0)

def H(t, s):
    t = t.detach() # stop gradient
    s = softmax(s / tps, dim=1)
    t = softmax((t - C) / tpt, dim=1) # center + sharpen
    return - (t * log(s)).sum(dim=1).mean()
```

center 갱신식 (하단 수식):
$$c \leftarrow m c + (1-m)\frac{1}{B}\sum_{i=1}^{B} g_{\theta_t}(x_i),$$

**설명**
- **흐름**: 한 이미지 $x$에서 두 view $x_1,x_2$ 생성 → 각각 student/teacher에 통과 → 출력은 $K$차원 logit → softmax로 확률분포 $p$ → cross-entropy $H(p_t, p_s) = -\sum p_t \log p_s$ 최소화. pseudocode에서는 대칭화를 위해 $\mathrm{H}(t_1,s_2)/2 + \mathrm{H}(t_2,s_1)/2$로 view를 교차해 계산함 (같은 view끼리는 비교하지 않음). 그림은 한 방향만 그린 단순화임.
- **Stop-gradient (sg)**: teacher 출력은 `detach()`되어 loss의 gradient가 teacher로 가지 않음. teacher는 **오직 EMA**로 갱신: $\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$ (코드의 `l`). `gt.params = gs.params`로 초기화는 동일하게 시작.
- **Temperature**: student는 $\tau_s$(`tps`), teacher는 $\tau_t$(`tpt`)로 나눠 softmax. teacher temperature를 더 낮게 두어 분포를 **sharpening**(뾰족하게) 함.
- **Centering**: teacher logit에서 center $C$(teacher 출력의 이동평균, $K$차원)를 빼줌. $C \leftarrow mC + (1-m)\cdot$(batch 평균 teacher 출력). 이 때 $m$은 center momentum.
- **왜 centering + sharpening?** 한 차원이 항상 지배하는 collapse는 **centering**이 막고(출력을 평균으로 눌러 특정 차원 우세를 상쇄), 반대로 모든 차원이 균등한 uniform 분포로 collapse하는 것은 **sharpening**이 막음. 두 효과가 서로 상쇄 관계라 균형이 잡힘. 이것이 negative sample 없이 collapse를 피하는 DINO의 메커니즘임.
- **BYOL과의 차이**: BYOL은 feature vector를 MSE로 맞추고 online에 **predictor**가 있음. DINO는 **predictor 없이** softmax 분포를 cross-entropy로 맞추고 centering/sharpening으로 collapse 방지 (다음 슬라이드 p.41에서 비교).
- ← 슬라이드 그림의 loss 표기 "$-p_2\log p_1$"은 한 방향만 보여주는 단순화이며, 실제 코드는 두 view 교차 대칭 loss임 (원문 유지).
> 보충: $K$는 출력 차원(prototype 수, 논문 기본 65536). 논문 기본 $\tau_s=0.1$, $\tau_t$는 0.04→0.07 warmup. student는 ViT-S/B 같은 backbone + 3-layer MLP head를 사용하고 마지막 층은 weight-normalized임. 코드의 `l, m`은 슬라이드 주석처럼 각각 network, center momentum rate임.

## FIGURES

- p.33: BYOL vs SimCLR/MoCo v2 등 linear evaluation / semi-supervised 성능 표 (중요도 중)
- p.35: Barlow Twins 파이프라인 도식(두 view → 공유 $f_\theta$ → $Z^A,Z^B$ → cross-corr $\mathcal{C}$ vs identity $\mathcal{I}$) + $\mathcal{L}_{BT}$ 수식 (중요도 상)
- p.36: Barlow Twins cross-correlation matrix 계산 3단계 도식 (batch × feature 표 → $D\times D$ 행렬, diagonal=invariance / off-diagonal=redundancy reduction, Goal=identity) (중요도 상)
- p.37: Barlow Twins vs SOTA 비교 Table 1, 2 (중요도 중)
- p.40: DINO 구조도(student/teacher, ema, centering, softmax, sg, loss) + PyTorch pseudocode + center 갱신 수식 (중요도 상)

## EXAM POINTS

1. **Barlow Twins loss 쓰기**: $\mathcal{L}_{BT}=\sum_i(1-\mathcal{C}_{ii})^2+\lambda\sum_i\sum_{j\ne i}\mathcal{C}_{ij}^2$ — 첫 항은 invariance, 둘째 항은 redundancy reduction. $\mathcal{C}$는 **batch 방향**으로 계산한 $D\times D$ cross-correlation matrix, 목표는 identity.
2. **Barlow Twins에서 collapse가 왜 방지되는가**: 모든 feature가 같아지면 비대각 상관이 커져 penalty → negative / predictor / EMA 없이 redundancy reduction이 collapse 방지 역할. batch size에도 robust한 이유(negative 개수 의존 없음).
3. **SimCLR(sample-level) vs Barlow Twins(feature-level)** 비교: $B\times B$ sample similarity vs $D\times D$ feature 상관행렬, negative 필요 여부.
4. **BYOL의 구성과 collapse 회피**: online(gradient, predictor 있음) vs target(EMA, stop-gradient), negative 없음, MSE 회귀로 다른 view의 representation 예측. 성능(linear eval 74.3%)이 SimCLR(69.3)보다 높다는 점.
5. **DINO 학습 메커니즘**: student–teacher 동일 구조, teacher는 student의 **EMA**(gradient 없음, stop-gradient), 두 view 교차 cross-entropy $H(t,s)=-\sum t\log s$, 라벨 없음.
6. **DINO의 centering과 sharpening의 역할** (why 질문): centering = 한 차원 지배 collapse 방지, sharpening(낮은 teacher temperature) = uniform collapse 방지. center 갱신식 $c\leftarrow mc+(1-m)\frac1B\sum g_{\theta_t}(x_i)$.
7. **BYOL vs DINO 비교**: feature matching(MSE) + predictor vs distribution matching(cross-entropy, softmax) + centering/sharpening, 둘 다 EMA teacher·라벨 없음.
8. **세 방법(BYOL / Barlow Twins / DINO)의 공통점**: negative sample 없이 collapse를 피하는 서로 다른 전략 — 비대칭+EMA(BYOL), 상관행렬 identity화(Barlow Twins), centering+sharpening(DINO).

## VERIFIER REPORT

- p.35 그림 설명: Net 출력 표기 순서 수정 (슬라이드는 위쪽 Y^B -> Z^A, 아래쪽 Y^A -> Z^B). 그 외 표 수치, 수식, pseudocode, 인용(연도/학회)은 원본과 일치 확인.
