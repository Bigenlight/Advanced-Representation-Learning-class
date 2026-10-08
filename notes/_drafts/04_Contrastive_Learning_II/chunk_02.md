<!-- DRAFT | lecture: 04_Contrastive_Learning_II | chunk: 02 | pages 9–16 -->
<!-- VERIFIED | fixes: 4 | verifier: sonnet -->

## Debiased Contrastive Loss – true negative 분포 유도 (p.9)

- Let us assume the probability of anchor class c is uniform
  $$\rho(c) = \eta^+$$
- Probability of observing a different class
  $$\eta^- = 1 - \eta^+$$
- The probability of observing a positive example for $\mathbf{x}$ is $p_x^+(\mathbf{x}') = p(\mathbf{x}' \mid \mathbf{h}_{x'} = \mathbf{h}_x)$;
- The probability of getting a negative sample for $\mathbf{x}$ is $p_x^-(\mathbf{x}') = p(\mathbf{x}' \mid \mathbf{h}_{x'} \neq \mathbf{h}_x)$.
- The actual sampling data distribution:
  $$p(x') = p(x', h_{x'} = h_x) + p(x', h_{x'} \neq h_x)$$
  $$\Rightarrow\; p(x') = p(h_{x'} = h_x)\,p(x' \mid h_{x'} = h_x) + p(h_{x'} \neq h_x)\,p(x' \mid h_{x'} \neq h_x).$$
  $$\Rightarrow\; p(\mathbf{x}') = \eta^+ p_x^+(\mathbf{x}') + \eta^- p_x^-(\mathbf{x}')$$
  $$p_x^-(\mathbf{x}') = \big(p(\mathbf{x}') - \eta^+ p_x^+(\mathbf{x}')\big)/\eta^-$$
- 박스: True Negative = (Random Sampling − Positive Contribution) / $\eta^-$

(앞 슬라이드 p.8 요약: Problem = random negative sampling이 false negative를 포함할 수 있음 / Idea = true negative 분포를 추정함.)

**설명**
- 문제 설정: self-supervised에서는 label이 없으므로 anchor $x$의 negative를 데이터셋에서 **random sampling**함. 이때 뽑힌 $x'$가 우연히 anchor와 같은 class(latent class $h$)일 수 있음 → **false negative**(sampling bias). 이걸 loss에서 그대로 밀어내면 같은 class 표현이 서로 멀어짐.
- 기호
  - $h_x$: $x$의 latent class (관측 불가, 개념상으로만 존재).
  - $\eta^+ = \rho(c)$: 임의로 뽑은 샘플이 anchor와 같은 class일 확률. class가 uniform이라 가정하면 $\eta^+ = 1/K$ (K = class 수). $\eta^- = 1-\eta^+$는 다른 class일 확률.
  - $p_x^+(x')$: anchor와 같은 class인 샘플의 분포 (true positive 분포). $p_x^-(x')$: 다른 class인 샘플의 분포 (true negative 분포, 우리가 원하는 것).
  - $p(x')$: 실제로 random sampling할 때 따르는 data 분포.
- 유도 직관: 전체 확률의 법칙(law of total probability)으로 $p(x')$를 "같은 class일 때" + "다른 class일 때"의 혼합(mixture)으로 분해함 → $p = \eta^+ p^+ + \eta^- p^-$. 이를 $p^-$에 대해 정리하면 **$p^- = (p - \eta^+ p^+)/\eta^-$**. 즉 **random sample 분포에서 positive 성분을 빼고 $\eta^-$로 정규화**하면 true negative 분포가 됨.
- 핵심: $p^-$를 직접 샘플링할 수는 없지만, 기댓값(expectation) 안에서는 $p$(random)와 $p^+$(augmentation으로 만든 positive)로 대체 계산 가능 → 다음 슬라이드에서 loss 형태로 구현.
- 이전 방법(표준 InfoNCE / SimCLR)과의 차이: 기존은 $p$에서 뽑은 샘플을 전부 negative로 간주(= $\eta^+ \approx 0$ 가정과 동일). Debiased는 $\eta^+$를 알려진 prior로 두고 bias를 보정함.

> 보충: 이 슬라이드는 Chuang, Robinson, Lin, Torralba, Jegelka, "Debiased Contrastive Learning", NeurIPS 2020의 유도를 따름. 논문에서는 class prior를 $\tau^+$로 표기하나 슬라이드는 $\eta^+$ 사용 (온도 $\tau$와의 혼동 방지로 보임). $\eta^+$는 학습되는 값이 아니라 **hyperparameter**(class 수에서 주어지거나 추정)임.

## Debiased Contrastive Loss – 최종 loss (p.10)

- (p.9와 동일한 가정 bullet: $\rho(c)=\eta^+$, $\eta^- = 1-\eta^+$, $p_x^+$, $p_x^-$ 정의 반복)
- 화살표(⇒)로 박스 연결: True Negative = (Random Sampling − Positive Contribution) / $\eta^-$
- 빨간 박스 수식 (negative 항 $g$):
  $$g(\mathbf{x}, \{\mathbf{u}_i\}_{i=1}^N, \{\mathbf{v}_i\}_{i=1}^M) = \max\left\{ \frac{1}{\eta^-}\left( \frac{1}{N}\sum_{i=1}^{N}\exp(f(\mathbf{x})^\top f(\mathbf{u}_i)) - \frac{\eta^+}{M}\sum_{i=1}^{M}\exp(f(\mathbf{x})^\top f(\mathbf{v}_i)) \right),\; \exp(-1/\tau) \right\}$$
- 최종 loss:
  $$\mathcal{L}_{\text{debias}}^{N,M}(f) = \mathbb{E}_{\mathbf{x},\{\mathbf{u}_i\}_{i=1}^N \sim p;\ \mathbf{x}^+,\{\mathbf{v}_i\}_{i=1}^M \sim p^+}\left[ -\log \frac{\exp(f(\mathbf{x})^\top f(\mathbf{x}^+))}{\exp(f(\mathbf{x})^\top f(\mathbf{x}^+)) + N g(x, \{\mathbf{u}_i\}_{i=1}^N, \{\mathbf{v}_i\}_{i=1}^M)} \right]$$
- 그림: 빨간 박스로 $g$ 수식을 강조하고, 빨간 화살표가 loss의 분모 $N g(\cdot)$ 항으로 연결됨 (분모의 negative 항을 $g$로 대체했음을 표시).

**설명**
- 표준 InfoNCE의 분모는 $\exp(f(x)^\top f(x^+)) + \sum_{i=1}^N \exp(f(x)^\top f(u_i))$ (negative 합). Debiased loss는 negative 합 부분 $N\cdot\frac1N\sum\exp(\cdot)$를 **true negative 기댓값의 추정치 $N g$** 로 교체함.
- $g$의 구성
  - $u_i \sim p$ (N개): random sampling된 샘플 → p.9의 "Random Sampling" 항. $\frac1N\sum \exp(f(x)^\top f(u_i))$는 $\mathbb{E}_{x'\sim p}[\exp(\cdot)]$의 Monte Carlo 추정.
  - $v_i \sim p^+$ (M개): anchor의 positive(augmented view 등) 샘플 → "Positive Contribution" 항. $\eta^+ \cdot \frac1M\sum \exp(f(x)^\top f(v_i))$가 $\eta^+\mathbb{E}_{x'\sim p^+}[\exp(\cdot)]$ 추정.
  - 둘을 빼고 $1/\eta^-$로 나눔 = p.9 공식 $p^- = (p-\eta^+p^+)/\eta^-$를 $\exp(f(x)^\top f(\cdot))$의 기댓값에 적용한 것 → $\mathbb{E}_{x'\sim p^-}[\exp(f(x)^\top f(x'))]$의 추정.
- $\max\{\cdot, \exp(-1/\tau)\}$의 이유: 유한 샘플에서는 뺄셈 결과가 **음수**가 될 수 있어 $\log$가 정의되지 않거나 불안정. 이론적 하한인 $\exp(-1/\tau)$로 clamp함 (feature를 $\|f(x)\|=1/\sqrt{\tau}$로 정규화하면 $f(x)^\top f(x') \ge -1/\tau$ 이므로 $\exp$ 값은 $\exp(-1/\tau)$ 이상).
- M의 역할: positive 샘플 수. $M=1$이면 $x^+$ 하나만으로 positive 기댓값을 추정(분산 큼), M을 키우면 추정이 정확해짐 (p.11의 $M=1$ vs $M=8$ 비교).
- 구현 이점: SimCLR 등 기존 코드에서 **loss 몇 줄만 수정**하면 되고 추가 label/모델 불필요.

> 보충: 논문(Chuang et al., NeurIPS 2020)의 class prior $\tau^+$가 슬라이드에서는 $\eta^+$로 표기됨. 논문은 $N\to\infty$에서 debiased loss가 true-negative 기반 ideal unbiased loss로 수렴한다고 보이고, 유한 N에 대한 estimation error bound도 제시함. 슬라이드의 $\mathcal{L}^{N,M}_{\text{debias}}$ 표기도 논문과 동일. 슬라이드 수식에서 $\mathbf{u}_i, \mathbf{v}_i$ 첨자 $i$ 위첨자/아래첨자 표기는 이미지 상 판독 가능한 범위에서 옮김.

## Debiased Contrastive Loss – t-SNE 시각화 (p.11)

- t-SNE visualization of learned representation with debiased contrastive learning.
- 그림: 4개의 t-SNE 산점도 패널이 가로로 나열됨. 점 색 = class(약 10색: 보라, 자홍, 빨강, 주황/황토, 연두, 청록, 파랑, 남색 등). 패널 아래 라벨:
  1. **Biased** — 기존 contrastive loss. class 간 군집이 서로 섞여 있고 가운데에 여러 색(연두/하늘/초록/보라)이 뒤엉킨 큰 덩어리가 있음.
  2. **Debiased M=1** — Biased보다 일부 군집(보라, 자홍, 빨강)이 분리되나 중앙 쪽은 여전히 혼재.
  3. **Debiased M=8** — 군집이 더 뚜렷하게 분리됨(보라, 빨강, 자홍/주황, 남색 등 독립적), 일부 청록/연두는 아직 인접.
  4. **Unbiased** — class label로 false negative를 제거한 이상적(supervised negative) 경우. 대부분의 class가 깔끔히 분리된 compact cluster를 이룸(단, 연두와 청록 클러스터는 우측에서 서로 맞닿아 있음).
- 축은 t-SNE 2차원 임베딩(축 눈금 없음).

**설명**
- 메시지: false negative를 보정할수록(Biased → Debiased M=1 → M=8 → Unbiased) **class별 cluster가 선명**해짐. 즉 debiasing이 label 없이도 이상적인 Unbiased 결과에 근접하게 만듦.
- $M$이 클수록(1 → 8) positive 기댓값 추정이 정확해져 성능/군집 품질 향상.
- "Unbiased"는 **upper bound 성격의 reference** (진짜 label로 같은 class 샘플을 negative에서 제외). 실제 self-supervised 설정에서는 쓸 수 없음.

> 보충: 논문의 해당 figure는 CIFAR-10(또는 STL10)의 학습된 표현에 대한 t-SNE로 알려짐(데이터셋은 슬라이드에 명시 없음, 확인 필요). t-SNE 시각화는 정성적 증거일 뿐이며, 정량 결과(linear probing accuracy)는 논문 표에 있음.

## Augmentation (p.12)

- 섹션 구분 슬라이드: "Augmentation" (파란 배경, 흰 글씨).

**설명**
- 이후 슬라이드는 contrastive learning에서 positive pair를 만드는 핵심 요소인 data augmentation을 다룸.

## Image Augmentations (p.13)

- Image augmentations create diverse views of the same image while preserving semantic meaning.
- Key Points
  - Create different **views** of the same image
  - Significantly change **visual appearance**
  - Preserve the **semantic meaning**
  - Form **positive pairs** for contrastive learning
- 그림: "Positive Pair" 제목의 점선 박스. 좌측 View 1(골든 리트리버 얼굴, 초록 배경), 우측 View 2(같은 개이나 약간 흐릿하고 색감/배경이 달라진 모습)가 **양방향 화살표(↔)** 로 연결. 하단 캡션 "(Same semantic meaning)".

**설명**
- contrastive learning(SimCLR 계열)에는 label이 없으므로 **같은 이미지에서 파생된 두 augmented view = positive pair**로 간주함. 서로 다른 이미지 = negative.
- 좋은 augmentation의 조건은 두 가지가 동시에 필요: (1) **appearance는 크게 변화** — 안 그러면 pixel/low-level 정보만으로 쉽게 맞춰서 trivial solution이 됨, (2) **semantic은 보존** — 안 그러면 positive가 더 이상 같은 의미가 아니라 잘못된 invariance를 학습함.
- 즉 augmentation이 모델이 어떤 **invariance**를 배울지를 정의함. 어떤 augmentation을 쓰는지가 표현의 품질을 크게 좌우함.

> 보충: SimCLR(Chen et al., ICML 2020)에서 augmentation 조합(특히 random crop + color distortion)이 가장 중요하다고 ablation으로 보임.

## Basic Image Augmentations (p.14)

- Random Resized Crop
- Color Jitter
- Gaussian Blur
- Horizontal Flip
- Grayscale
- Resize
- 그림: 7개의 작은 패널(모두 골든 리트리버 사진 예시)을 2행으로 배치.
  - 윗줄: **Random Crop**(원본에 점선 박스로 영역 표시 → 아래 화살표 → 확대된 얼굴 crop), **Color Jitter**(원본 → 색감이 더 진한 이미지), **Gaussian Blur**(선명한 이미지 → 흐릿한 이미지), **Horizontal Flip**(원본 → 좌우 반전된 이미지).
  - 아랫줄: **Grayscale**(흑백 → 흑백 변환 결과), **Resize**(큰 이미지 → 작은 이미지), **Multi-Crop (SwAV)**(큰 이미지 하나 → 오른쪽에 작은 crop 4개).

**설명**
- 각 연산 의미
  - **Random Resized Crop**: 무작위 영역·비율로 crop 후 고정 크기로 resize. global-to-local, 부분 간 일관성을 학습하게 함.
  - **Color Jitter**: brightness/contrast/saturation/hue를 랜덤 변경. 색 histogram만으로 같은 이미지를 구별하는 shortcut을 막음.
  - **Gaussian Blur**: 고주파 디테일 제거.
  - **Horizontal Flip**: 좌우 반전 (좌우 대칭 불변성).
  - **Grayscale**: 색 정보 제거(보통 확률적으로 적용).
  - **Resize**: 해상도 변경.
  - **Multi-Crop (SwAV)**: 한 이미지에서 여러 개의 crop(큰 global view 몇 개 + 작은 local view 여러 개)을 만들어 적은 계산으로 더 많은 positive pair 확보.
- 이 조합들이 SimCLR/MoCo v2/SwAV 등 vision contrastive learning의 기본 pipeline.

> 보충: Multi-Crop은 Caron et al., "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments (SwAV)", NeurIPS 2020 제안. SimCLR에서는 crop + color distortion의 조합이 특히 중요하고, 하나만 쓰면 성능이 크게 낮다고 보고됨.

## Augmentation Strategies – AutoAugment (p.15)

- AutoAugment
  - Automatically searches for the **best augmentation policy**
  - Formulates augmentation search as a **reinforcement learning** problem
  - Maximizes validation accuracy
  - **Key Idea**: Learn which augmentations to apply.
- **Operations**: ShearX/Y, TranslateX/Y, Rotate, AutoContrast, Invert, Equalize, Solarize, Posterize, Contrast, Color, Brightness, Sharpness, Cutout, Sample Pairing.
- 출처: Cubuk et al., CVPR 2019, AutoAugment: Learning Augmentation Policies from Data
- 그림: 4개의 둥근 박스가 시계 방향 루프를 이룸.
  1. 상단: "Sample a strategy S (Operation type, probability and magnitude)"
  2. → 우측: "Train a child network with strategy S to get validation accuracy R"
  3. → 하단: "Use R to update the controller"
  4. → 좌측: "The controller (RNN)"
  5. → 다시 상단(controller가 strategy를 샘플링)으로 화살표가 올라감.

**설명**
- 동작 루프: **controller(RNN)** 가 augmentation policy S(어떤 operation을, 어떤 확률로, 어떤 강도(magnitude)로 적용할지)를 샘플 → S로 **child network**를 학습시키고 validation accuracy R 측정 → R을 **reward**로 삼아 controller를 RL로 업데이트 → 반복.
- 장점: 사람이 hand-crafting하던 augmentation 설계를 **데이터셋에 맞게 자동화**하여 정확도 향상.
- 단점: child network를 매번 학습해야 해서 **search 비용이 매우 큼** (→ 다음 슬라이드 RandAugment 동기).
- 주의: AutoAugment 원래 목적은 **supervised 분류의 validation accuracy 최대화**이며, contrastive learning용 positive-pair 생성에 그대로 최적인지는 별개 문제임.

> 보충: 원 논문은 controller RNN을 Proximal Policy Optimization(PPO)으로 학습하고, policy는 sub-policy(연산 2개, 각 (확률, magnitude)) 여러 개로 구성됨. ImageNet/CIFAR-10 등에서 SOTA를 달성했으나 search에 수천 GPU 시간이 들었음.

## Augmentation Strategies – RandAugment (p.16)

- RandAugment
  - Eliminates the expensive search process
  - Uses **randomly** selected augmentations
  - Controls all transformations with a single magnitude parameter
  - **Key Idea**: Simpler and more efficient than AutoAugment.
- 출처: Cubuk et al., 2019, RandAugment: Practical automated data augmentation with a reduced search space ← 슬라이드는 2019로 표기 (arXiv 2019, 정식 학회는 CVPR Workshops 2020)
- 그림: 두 개의 박스 + 중간 화살표(⇒) 비교.
  - 좌 **AutoAugment** — Search: ● Which operation? ● What probability? ● What magnitude? → Huge search space
  - 우 **RandAugment** — Search: ● Number of operations for an image (**N**) ● One global magnitude (**M**) → Very small search space

**설명**
- AutoAugment는 operation 종류/확률/강도를 모두 탐색해서 search space가 큼. RandAugment는 이를 **두 개의 hyperparameter**로 축소:
  - **N**: 이미지 한 장에 적용할 operation의 개수 (연산 풀에서 균등 무작위 선택)
  - **M**: 모든 operation에 공통으로 쓰는 global magnitude
- 별도의 proxy task/child network 학습 없이 **grid search 몇 번**(예: N, M 작은 격자)으로 충분 → 계산 효율 대폭 개선, 성능은 AutoAugment와 비슷하거나 더 좋음.
- 설계 이유(해석): 탐색 공간을 줄여도 성능이 유지되므로, 세부 policy 값보다 **다양한 연산 + 적절한 전역 강도**가 중요하다는 관점으로 볼 수 있음(슬라이드에는 명시 없음).
- 이 구분: AutoAugment = "**learn** which augmentations", RandAugment = "**randomly sample**, tune only N and M".

> 보충: 슬라이드 인용 연도 표기가 AutoAugment(CVPR 2019)와 달리 단순 "2019"인 것은 RandAugment가 arXiv 2019 preprint이기 때문으로 보임. 이후 p.17은 PBA(Ho et al., ICML 2019)로 이어지나 이는 다음 chunk 범위.

## FIGURES

- p.9: Debiased loss 유도 수식 전개(혼합 분해 → $p^-$ 식, True Negative 박스) — 중요도 상
- p.10: $g(\cdot)$ 정의 및 $\mathcal{L}_{\text{debias}}^{N,M}$ 최종 수식(빨간 박스 강조) — 중요도 상
- p.11: 4패널 t-SNE (Biased / Debiased M=1 / Debiased M=8 / Unbiased) — 중요도 상
- p.13: Positive Pair 그림(View 1 ↔ View 2, same semantic meaning) — 중요도 중
- p.14: 기본 augmentation 7종 예시 그리드(Random Crop, Color Jitter, Gaussian Blur, Horizontal Flip, Grayscale, Resize, Multi-Crop SwAV) — 상
- p.15: AutoAugment 루프 다이어그램(controller RNN → strategy S → child network → reward R → update) — 상
- p.16: AutoAugment vs RandAugment search space 비교 박스 — 중

## EXAM POINTS

1. **False negative / sampling bias 문제**: random negative sampling이 같은 latent class 샘플을 포함하는 이유와 이것이 contrastive loss에 미치는 영향(같은 class를 밀어냄) 설명.
2. **True negative 분포 유도**: $p(x') = \eta^+ p^+ + \eta^- p^-$ 에서 $p^-(x') = (p(x') - \eta^+ p^+(x'))/\eta^-$ 를 유도하고 각 기호($\eta^+$, $\eta^-$, $p^+$, $p^-$) 의미 서술.
3. **Debiased loss 수식**: $g$의 구성(N개 random negative, M개 positive, $\eta^+$, $1/\eta^-$), 그리고 **$\max\{\cdot, \exp(-1/\tau)\}$ clamp를 쓰는 이유**(음수 방지, 이론적 하한).
4. **M의 역할**: positive 샘플 수 M이 커질수록 추정이 정확해져 t-SNE 군집이 Unbiased에 근접(M=1 < M=8 < Unbiased).
5. **좋은 augmentation의 조건**: appearance는 크게 바꾸되 semantic은 보존 → positive pair로 사용. 왜 둘 다 필요한지(trivial solution vs 잘못된 invariance).
6. **기본 augmentation 종류와 효과**: Random Resized Crop, Color Jitter, Gaussian Blur, Horizontal Flip, Grayscale, Multi-Crop(SwAV) — 각각이 막는 shortcut/학습하는 invariance.
7. **AutoAugment vs RandAugment 비교**: AutoAugment = RNN controller + RL(reward = validation accuracy)로 operation/확률/magnitude 탐색 → 비용 큼. RandAugment = 무작위 선택, 하이퍼파라미터 N(연산 수), M(global magnitude) 두 개만 → 탐색 공간 매우 작음.
8. **Why 질문**: RandAugment가 왜 AutoAugment와 비슷한 성능을 내면서 훨씬 싸게 되는가 (search space 축소, 다양성이 중요).

## VERIFIER REPORT

- p.11 Unbiased 패널 설명: 연두/청록 클러스터가 맞닿아 있음을 명시 (과장 수정)
- p.10 보충: 깨진 문장을 정상 문장으로 수정
- p.16 설명: "다양성이 중요하다는 관찰"을 슬라이드 근거 없는 단정에서 해석으로 완화
- p.13 View 2 그림 설명 보정 (확대 아님, 약간 흐림)
