<!-- DRAFT | lecture: 04_Contrastive_Learning_II | chunk: 01 | pages 1–8 -->
<!-- VERIFIED | fixes: 2 | verifier: sonnet -->

## Advanced Representation Learning (고급표현학습) — 표지 (p.1)

- Advanced Representation Learning (고급표현학습)
- Instructor: Donghyeon Cho
- The Department of Computer Science
- Hanyang University
- 그림: 한양대학교 로고(좌하단), 한양대 교표(1939), 연구실 로고(우하단), 파란 가로선 장식

**설명**
- 강의 표지 슬라이드. 이 강의(04)는 Contrastive Learning II로, 목차(p.2)상 **contrastive learning의 핵심 요소 / augmentation / parallel augmentation 기반 방법(SimCLR, BYOL, Barlow Twins)**을 다룸.

## Contents of Table (p.2)

- 1. Key Ingredients of Contrastive Learning
- 2. Augmentation
- 3. Learning from Parallel Augmentation
  - SimCLR
  - BYOL
  - Barlow Twins

**설명**
- 강의 구성: (1) contrastive learning이 잘 작동하기 위한 핵심 요소(augmentation, batch size, hard negative mining), (2) augmentation의 구체적 종류, (3) 같은 이미지의 두 augmented view를 병렬로 넣어 학습하는 방법 3가지.
- 흐름상 SimCLR는 **negative를 쓰는** contrastive 방식, BYOL은 **negative 없이** (predictor + momentum target), Barlow Twins는 **cross-correlation matrix를 identity에 맞추는** redundancy reduction 방식으로 negative 의존성을 점점 줄이는 방향임.
> 보충: SimCLR (Chen et al., 2020), BYOL (Grill et al., 2020), Barlow Twins (Zbontar et al., 2021).

## Key Ingredients of Contrastive Learning — 섹션 구분 (p.3)

- 섹션 구분 슬라이드: "Key Ingredients of Contrastive Learning" (남색 배경, 흰 글씨)

## Heavy Data Augmentation (p.4)

- Needed for creating **noise versions** of given training data
  - As **positive** samples
  - **Without modifying semantic** meanings
- Proper data augmentation setup is critical for learning good and **generalizable** embedding features

**설명**
- Contrastive learning(self-supervised)에는 label이 없으므로 **positive pair를 만드는 유일한 수단이 augmentation**임. 한 이미지 $x$에서 서로 다른 augmentation $t, t'$를 적용해 anchor $t(x)$, positive $t'(x)$를 만들고 둘을 가깝게 당김.
- "noise version"은 원본에 변형(crop, color jitter, blur 등)을 가한 버전. 핵심 제약은 **semantic(의미)을 바꾸지 않는** 것: 의미가 바뀌면 positive pair가 틀린 supervision이 됨.
- 반대로 augmentation이 너무 약하면 모델이 low-level shortcut(색 히스토그램, 위치 등)만으로 두 view를 맞춰버려 **trivial solution**에 빠짐. 따라서 "heavy"하게 주되 semantic은 보존해야 하는 균형이 핵심이고, 이 setup이 **generalizable embedding**을 결정함.
> 보충: SimCLR 논문(Chen et al., 2020)은 augmentation 조합 중 random crop + color distortion 조합이 특히 중요하다고 보고함. 다음 섹션(Augmentation)에서 자세히 다뤄질 것으로 보임.

## Large Batch Size (p.5)

- Provides a large number of in-batch negative samples
- Increases the **diversity** of negative examples
- **Reduces false similarity** between different instances
- Encourages learning more **discriminative** representations
- Essential for methods relying on InfoNCE (e.g., SimCLR, CLIP)

**설명**
- InfoNCE류 loss는 batch 안의 다른 샘플들을 모두 negative로 사용(in-batch negatives). batch size $B$면 anchor당 negative가 약 $2(B-1)$개(SimCLR 기준)이므로 **batch가 클수록 negative 수와 다양성이 증가**함.
- negative가 다양하면 anchor와 우연히 비슷한 "가짜 유사성(false similarity)"에 의존하지 않고 다양한 대조 대상에 대해 구분하도록 학습되어 **discriminative representation**이 만들어짐.
- InfoNCE는 $N$-way classification 기반의 mutual information lower bound라 $N$이 클수록 bound가 tight해짐.
> 보충: InfoNCE $\mathcal{L} = -\log \frac{\exp(s(z_i,z_j)/\tau)}{\sum_{k\ne i}\exp(s(z_i,z_k)/\tau)}$ 에서 분모의 negative 항 수가 batch size에 비례. $I(x;c) \ge \log N - \mathcal{L}_{\text{InfoNCE}}$ (van den Oord et al., 2018, CPC). SimCLR는 batch 4096~8192, CLIP은 32768 정도를 사용했고 큰 GPU/TPU 메모리 비용이 듦.

## Hard Negative Mining (p.6)

- Hard **negatives** are **close to the anchor** in the embedding space
- Provide more **informative** supervision than easy negatives
- Improve **discriminative** representation learning
- **Easy in supervised** learning using class labels
- **Challenging in self-supervised** learning
  - No ground-truth labels
  - Hard negatives are difficult to identify
  - Larger batches (or memory banks) increase the chance of finding hard negatives
  - Trade-off: Better negatives vs. higher memory cost

**설명**
- **Hard negative**: anchor와 의미는 다르지만 embedding 공간에서 거리가 가까운 샘플. 이미 멀리 떨어진 **easy negative**는 loss/gradient가 거의 0이라 학습 기여가 작고, hard negative가 decision boundary를 정교하게 만들어줌.
- supervised에서는 class label이 있어 "다른 class인데 가까운 것"을 바로 찾을 수 있음. self-supervised에서는 label이 없어 어떤 샘플이 진짜 negative인지조차 모르므로 hard negative를 가려내기 어려움.
- 현실적 대안: **batch를 키우거나 memory bank / queue**를 써서 negative pool을 키우면 그 안에 hard negative가 포함될 확률이 올라감. 대가는 메모리/연산 비용(trade-off).
> 보충: MoCo(He et al., 2020)는 momentum encoder + queue로 큰 negative set을 저렴하게 유지함. hard negative를 reweighting하는 방법으로 Robinson et al., 2021 (Contrastive Learning with Hard Negative Samples)가 있음.

## Hard Negative Mining — Sampling bias (p.7)

- Sampling bias can refers to false negative samples ← "can refers to" 문법 오류 (can refer to), 원문 유지
  - → lead to a big performance drop
- 출처: Chuang et al., 2020, Debiased Contrastive Learning
- 그림 (좌): anchor $x$ (golden retriever 사진)에서 화살표 두 갈래. 위쪽은 positive $x^+ \sim p_x^+$ (다른 포즈/배경의 golden retriever). 아래쪽은 negative $x_i^- \sim p$로 고양이, 비행기, 그리고 **붉은 테두리로 표시된 golden retriever 사진 = "false negative sample"**이 섞여 있음.
- 그림 (우): 그래프. x축 = Negative Sample Size (N) (30, 62, 126, 254, 510), y축 = Top-1 Accuracy (80~95). 두 곡선: **Unbiased**(파랑)는 약 88 → 92 → 93 → 94 → 94.4로 빠르게 포화, **Biased**(초록)는 약 80 → 85 → 87.5 → 90 → 91로 느리게 상승. 모든 N에서 Unbiased > Biased이고 N이 작을수록 gap이 큼(N=30에서 약 8 point).

**설명**
- 무작위로 negative를 뽑으면 **anchor와 같은 class인 샘플이 negative로 뽑히는 false negative**가 생김(그림의 두 번째 golden retriever). 이를 anchor에서 밀어내면 같은 class 표현을 오히려 분리시켜 representation이 망가짐 → **sampling bias**.
- 그래프: 같은 negative 수에서도 bias를 제거(Unbiased, debiased loss)하면 accuracy가 크게 높음. 즉 negative 수만이 아니라 **negative의 "질"(false negative 제거)**이 중요함을 보여줌.
- Biased 쪽은 N이 커질수록 좋아지지만(큰 batch 효과) Unbiased는 작은 N에서도 이미 높음 → debiasing이 **작은 batch에서 특히 이득**.
> 보충: 그래프는 Chuang et al. (NeurIPS 2020)의 실험으로 보이며 데이터셋은 슬라이드에 명시 없음. 수치는 그림에서 읽은 근사값.

## Debiased Contrastive Loss (p.8)

- Problem: Random negative sampling may include **false negatives**.
- Idea: Estimate the **true negative distribution**.

**설명**
- 문제: negative를 $x^-_i \sim p(x)$ (전체 데이터 분포)에서 뽑기 때문에 anchor와 같은 class인 샘플이 섞임. 실제로 원하는 것은 anchor와 **다른 class에서만** 뽑은 true negative $p_x^-(x')$임.
- 아이디어: label을 모르니 직접 뽑을 수는 없지만, **전체 분포 $p(x)$가 "같은 class 분포 $p_x^+$와 다른 class 분포 $p_x^-$의 혼합"**임을 이용해 $p_x^-$를 수학적으로 추정하고 loss를 보정함. (이 슬라이드(p.8)에는 수식 없이 문제/아이디어 두 줄만 있음. 아래 수식은 Chuang et al. 논문 기반 보충.)
> 보충(Chuang et al., 2020): $p(x') = \tau^+ p_x^+(x') + \tau^- p_x^-(x')$, 여기서 $\tau^+$는 class prior(같은 class일 확률), $\tau^- = 1-\tau^+$. 따라서 $p_x^-(x') = \frac{p(x') - \tau^+ p_x^+(x')}{\tau^-}$. 이를 loss의 negative 항 기대값에 대입하면 $p$에서 뽑은 negative와 positive만으로 계산 가능한 debiased estimator를 얻음.

---

## FIGURES

- p.7: false negative 개념도 (anchor, positive, negative 중 붉은 테두리 golden retriever) — 중요도 상
- p.7: Biased vs Unbiased, negative sample size N에 따른 Top-1 Accuracy 그래프 — 중요도 상
- p.2: 강의 목차 (SimCLR/BYOL/Barlow Twins 흐름 파악용) — 중요도 중

## EXAM POINTS

1. **Contrastive learning의 핵심 요소 3가지**: heavy data augmentation, large batch size, hard negative mining을 나열하고 각각의 역할 설명.
2. Augmentation 요건: positive 생성 수단이며 **semantic을 보존**해야 함. 너무 약하면 trivial solution, 너무 강하면 semantic 손상 (why).
3. **Large batch가 왜 필요한가?** in-batch negative 수/다양성 증가, InfoNCE(SimCLR, CLIP)에 필수. 대가는 메모리 비용.
4. **Hard negative 정의**(anchor와 embedding 공간에서 가까운 negative)와 easy negative보다 informative한 이유. supervised에서는 쉽지만 self-supervised에서 어려운 이유(label 부재).
5. **False negative / sampling bias**: 정의, 성능 저하 원인, 그림(같은 class 강아지가 negative로 뽑힘).
6. Debiased contrastive loss (Chuang et al., 2020)의 아이디어: 전체 분포 $p$를 same-class와 true-negative 분포의 혼합으로 보고 **true negative distribution을 추정**.
7. Biased vs Unbiased 그래프 해석: 모든 N에서 unbiased가 우수, N이 작을수록 격차 큼.
8. Large batch/memory bank의 trade-off: better negatives vs higher memory cost.

## VERIFIER REPORT

- p.1 설명: "이전 강의의 InfoNCE 기반 방법을 이어받아"는 슬라이드 근거 없는 추정이라 제거하고 목차 기준 서술로 수정, "negative sampling"은 목차 항목이 아니므로 "핵심 요소"로 교체.
- p.8 설명: "다음 페이지 p.9에서 class prior uniform 가정으로 이어짐"은 범위 밖 추측이라 삭제하고 p.8 슬라이드에 수식이 없음을 명시.
- 그 외 p.1-8 대조 결과 그래프 수치, 그림 설명, 인용(Chuang 2020, CPC 2018, MoCo 2020, Robinson 2021, SimCLR/BYOL/Barlow Twins), 수식(InfoNCE, 혼합분포 τ+)은 이상 없음.
