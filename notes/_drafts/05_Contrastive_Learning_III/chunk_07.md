<!-- DRAFT | lecture: 05_Contrastive_Learning_III | chunk: 07 | pages 49–56 -->
<!-- VERIFIED | fixes: 4 | verifier: sonnet -->

## Feature Contrast (refer Barlow Twins) (p.49)

**슬라이드 내용**
- 제목: Feature Contrast (refer Barlow Twins)
- 우측 텍스트: "Compute cross-correlation matrix using Z" / "Use projected representations of the two augmented views (Z)"
- 수식:

$$
L_{Red.} = \frac{1}{2}\sqrt{\frac{1}{2D}\sum_{i=1}^{D}\left((1-cc^{1}_{ii})^2 + (1-cc^{2}_{ii})^2\right)} + \frac{1}{2}\sqrt{\frac{1}{2D(D-1)}\sum_{i=1}^{D}\sum_{j\neq i}^{D}\left((cc^{1}_{ij})^2 + (cc^{2}_{ij})^2\right)}
$$

- bullet
  - Diagonal terms: The same feature dimension remains consistent across views.
  - Off-diagonal terms: Different feature dimensions become less redundant.
- 하단 굵은 문장: **Neighbor losses compare samples. Feature contrast compares feature dimensions.**
- 그림(All4One 전체 구조도): 아래에서 위로 흐름. 좌하단 `encoder`(초록)에 입력 $X^1$, 우하단 `momentum encoder`(파랑)에 입력 $X^2$. encoder → momentum encoder로 점선 화살표 "EMA". 각 encoder 위에 $Z^1$, $Z^2$가 있고, 이 둘 사이를 양방향 빨간 화살표로 묶은 빨간 박스가 "Feature Contrast". $Z^1$ 위에 `predictor`(주황) → $P^1_{nn}$ → "Neighbour Contrast" (초록 화살표, 우측에서 $nn_1$이 들어옴; $Z^2$ → `KNN` → $nn_1$). 또한 predictor에서 위로 올라가는 $P^1_{cc}$ → `KNN`(분홍) → `transformer encoder`(보라) ← $nn_1\,nn_2 \ldots nn_k$ (우측 KNN에서 k개 이웃). transformer encoder 출력 $C^1$, $C^2$ 사이가 "Centroid Contrast". 빨간 박스로 Feature Contrast 부분이 강조됨.

**설명**
- All4One의 세 번째 objective인 **Feature Contrast**는 **Barlow Twins**식 redundancy reduction loss를 그대로 가져온 것임. 앞의 Neighbor Contrast(NNCLR식)와 Centroid Contrast가 **sample 단위(행)** 비교인 반면, 이건 **feature dimension 단위(열)** 비교임.
- 기호: $Z$ = projector를 통과한 두 augmented view의 표현 ($D$차원). $cc_{ij}$ = batch 방향으로 정규화한 두 view의 feature $i$, $j$ 사이 cross-correlation 원소. 슬라이드 수식은 $cc^1$, $cc^2$ 두 개를 쓰는데, 서로 다른 pair의 cross-correlation matrix(두 view 방향 대칭)로 보임 ← 슬라이드에서 $cc^1$, $cc^2$의 정의가 명시되지 않음.
- **Diagonal term** $(1-cc_{ii})^2$: 같은 feature dimension $i$가 두 view에서 **상관 1**이 되도록 → augmentation에 **invariance**.
- **Off-diagonal term** $(cc_{ij})^2,\ i\neq j$: 서로 다른 dimension끼리 상관 0이 되도록 → feature 간 **redundancy 제거**, dimensional collapse 방지.
- 원 Barlow Twins loss는 $\sum_i (1-C_{ii})^2 + \lambda\sum_{i\neq j} C_{ij}^2$ 형태인데, All4One 버전은 두 항을 각각 평균내고 **제곱근(sqrt)** 을 씌운 뒤 1/2씩 합함 → 항 개수 정규화 + sqrt로 두 항의 스케일을 맞추어 $\lambda$ 가중치를 따로 두지 않는 구성으로 보임(정규화 상수 $\frac{1}{2D}$, $\frac{1}{2D(D-1)}$로 항 개수 보정).
- 왜 필요한가: neighbor/centroid contrast는 **instance-level contrastive**라 feature 차원 간 중복을 직접 제약하지 않음. Feature contrast가 이를 보완해 dimension 사용 효율을 올림.

> 보충: Barlow Twins (Zbontar et al., 2021, "Barlow Twins: Self-Supervised Learning via Redundancy Reduction"). 이름은 neuroscientist H. Barlow의 redundancy-reduction principle에서 따옴. negative sample, predictor, stop-gradient 없이도 collapse를 피함. All4One 논문은 Estepa et al., ICCV 2023 ("All4One: Symbiotic Neighbour Contrastive Learning via Self-Attention and Redundancy Reduction").

---

## Contribution of Each Objective (p.50)

**슬라이드 내용**
- 표 (Table 3: All4One objective function ablation study (using CIFAR-100). All4One$_{v0}$ is equal to vanilla NNCLR. NN retrieval accuracy marked by * represents the NN retrieval obtained by increasing the queue size from 65503 to 98304 [14])

| Method | NNCLR obj. | Cen. obj. | Feat. obj. | EMA | Top-1 | k-NN Top-1 | NN Top-1 |
|---|---|---|---|---|---|---|---|
| All4One$_{v0}$ | ✓ | | | | 69.62 | 62.16 | 68.8 (77.8*) |
| All4One$_{v1}$ | | ✓ | | | 67.4 | 59.61 | 82.8 |
| All4One$_{v2}$ | ✓ | ✓ | | | 71.02 | 63.21 | 85.28 |
| All4One$_{v3}$ | ✓ | ✓ | | ✓ | 71.08 | 63.83 | **86.16** |
| All4One$_{v4}$ | | ✓ | ✓ | ✓ | 71.31 | 63.72 | 80.6 |
| All4One$_{v5}$ | ✓ | | ✓ | ✓ | 71.64 | 64.58 | 78.8 |
| All4One$_{v6}$ | ✓ | ✓ | ✓ | ✓ | **72.17** | **64.84** | **82.16** |

- bullet
  - Centroid alone: 67.4 → Centroid contrast alone is weaker than NNCLR.
  - Neighbor + Centroid: 69.62 --> 71.02 → Multiple-neighbor context improves NNCLR.
  - Full objective: 72.17 → The three objectives work best together.
- 하단 굵은 문장: **Centroid contrast is useful as a complementary objective, not as a replacement for neighbor contrast.**

**설명**
- CIFAR-100에서 objective 조합별 **ablation**. v0 = vanilla NNCLR(기준선). 지표는 linear Top-1, k-NN Top-1, 그리고 NN retrieval Top-1(검색된 nearest neighbor의 class가 query와 같은 비율).
- **Centroid만 쓰면(v1) 67.4로 NNCLR(69.62)보다 낮음** → centroid는 단독 대체재가 아님. 단 NN retrieval은 82.8로 NNCLR(68.8)보다 훨씬 높음 → centroid가 embedding 공간의 이웃 구조를 개선하긴 함.
- **NNCLR + Centroid(v2): 71.02 (+1.4)** → 여러 neighbor를 self-attention으로 합친 context가 단일 NN 하나보다 유익함.
- v3: EMA(momentum encoder) 추가 시 71.08로 소폭 상승, NN retrieval 최고 86.16.
- Feature objective를 더한 v4, v5는 각각 71.31, 71.64; **셋 다 쓰는 v6가 72.17 / 64.84로 linear, k-NN 모두 최고**.
- 주의: NN Top-1은 v3(86.16)가 최고이고 full(v6)은 82.16 → 다음 슬라이드(p.51)의 "retrieval 정확도 ≠ 최종 성능" 논지로 연결됨.
- v0의 `68.8 (77.8*)`: queue 크기를 65503 → 98304로 키웠을 때 NN retrieval이 77.8. ← 슬라이드 caption의 65503은 65536(=2^16) 오타 가능성 (원문 유지).

---

## Neighbor Retrieval Analysi (p.51)

**슬라이드 내용** (원문 제목 "Neighbor Retrieval Analysi"로 잘림 ← 오타, 의도는 Analysis)
- 좌측 그래프: x축 epoch (0~900+), y축 NN retrieval accuracy (0~1). 4개 곡선: NNCLR + Cent. + Feat. (파랑), NNCLR + Feat. (주황), NNCLR + Cent. (초록), NNCLR (보라). 모든 곡선이 초반 급상승 후 완만히 증가. 초록(NNCLR + Cent.)이 가장 위(~0.86), 파랑(full)이 그다음(~0.82), 주황과 보라가 가장 아래(~0.78~0.79).
- 우측 그림: "Query | Top-5 NNs extracted from the Support Set". 3개 행: 노란 트럭 query → 이웃들(트럭, 일부는 빨간 트럭), 상어/고래 같은 수중 query → 이웃들(상어, 돌고래, 고래 등), 단풍 든 나무 query → 이웃들(나무, 노랑/빨강 단풍 등).
- 좌측 bullet (NN Retrieval Accuracy)
  - NNCLR provides the baseline.
  - Adding the centroid objective improves retrieval.
  - The full objective does not have the highest retrieval accuracy.
  - The full objective still gives the best linear evaluation result.
- 우측 bullet (Retrieved Neighbors)
  - Most retrieved samples share visual or semantic characteristics.
  - Some retrieved samples belong to different classes.
  - Different-class neighbors can still contain useful visual similarity.
- 하단 굵은 문장: **Class-correct neighbor retrieval is helpful, but it is not the final objective.**

**설명**
- 학습 중 **NN retrieval accuracy**(support set에서 뽑은 이웃이 query와 같은 class일 비율)를 추적한 분석. Centroid objective를 넣으면 retrieval이 가장 좋아지지만(초록), feature objective까지 더한 full(파랑)은 retrieval은 오히려 낮아지는데 linear eval은 최고 → **retrieval 정확도가 downstream 표현 품질과 1:1 대응하지 않음**.
- 정성적 결과: 이웃은 대체로 시각적·의미적으로 유사(색, 형태, 배경). 일부는 class가 다름(예: 상어 query ↔ 고래/돌고래 이웃처럼 시각적으로 비슷한 다른 종류가 섞일 수 있음 ← 슬라이드에 어떤 이웃이 오답인지/데이터셋이 명시되진 않음, 일반적 해석). 이런 **false positive 이웃도 유용한 시각 정보**를 담고 있어 positive로 써도 큰 해가 없음.
- 핵심 메시지: 목표는 class가 맞는 이웃을 찾는 게 아니라 **좋은 표현**을 학습하는 것. Self-supervised이므로 이웃 선택은 label이 아니라 embedding 유사도에 의존함.

---

## Linear Evaluation (p.52)

**슬라이드 내용**
- 표 (Table 1: Linear evaluation results on CIFAR-10, CIFAR-100 and ImageNet100. The results are extracted from Solo-learn Self-supervised learning library [11].)

| Method | CIFAR-10 Top-1 | Top-5 | CIFAR100 Top-1 | Top-5 | ImageNet100 Top-1 | Top-5 |
|---|---|---|---|---|---|---|
| BYOL | 92.58 | 99.79 | 70.46 | 91.96 | 80.16 | 95.02 |
| DC V2 | 88.85 | 99.58 | 63.61 | 88.09 | 75.36 | 93.22 |
| DINO | 89.52 | 99.71 | 66.76 | 90.34 | 74.84 | 92.92 |
| MoCoV2+ | 92.94 | 99.79 | 69.89 | 91.65 | 78.20 | 95.50 |
| MoCoV3 | 93.10 | 99.80 | 68.83 | 90.57 | 80.36 | 95.18 |
| ReSSL | 90.63 | 99.62 | 65.92 | 89.73 | 76.92 | 94.20 |
| SimCLR | 90.74 | 99.75 | 65.78 | 89.04 | 77.64 | 94.06 |
| Simsiam | 90.51 | 99.72 | 66.04 | 89.62 | 74.54 | 93.16 |
| SwAV | 89.17 | 99.68 | 64.88 | 88.78 | 74.04 | 92.70 |
| VIbCReg | 91.18 | 99.74 | 67.37 | 90.07 | 79.86 | 94.98 |
| VICReg | 92.07 | 99.74 | 68.54 | 90.83 | 79.22 | 95.06 |
| W-MSE | 88.67 | 99.68 | 61.33 | 87.26 | 67.60 | 90.94 |
| BT | 92.10 | 99.73 | 70.90 | 91.91 | 80.38 | 95.28 |
| NNCLR | 91.88 | 99.78 | 69.62 | 91.52 | 79.80 | 95.28 |
| All4One | **93.24** | **99.88** | **72.17** | **93.35** | **81.93** | **96.23** |

(BT / NNCLR / All4One 위에 가로선으로 구분됨.)
- 우측 텍스트: All4One improves both neighbor contrast and redundancy reduction baselines.
  - It shows improvements across all three datasets compared to NNCLR.
  - It also shows improvements compared to Barlow Twins.
  - The margin of improvement is greatest on CIFAR-100.
  - This suggests that multiple-neighbor context is likely more useful for datasets containing many similar classes.

**설명**
- **Linear evaluation**: backbone을 freeze하고 linear classifier만 학습해 표현 품질을 측정. 비교 baseline 수치는 Solo-learn 라이브러리 결과를 가져옴.
- All4One는 **neighbor contrast 계열 baseline(NNCLR)** 과 **redundancy reduction 계열 baseline(BT)** 둘 다를 능가: NNCLR 대비 CIFAR-10 +1.36, CIFAR-100 +2.55, ImageNet100 +2.13 (Top-1). BT 대비 +1.14, +1.27, +1.55.
- 개선폭은 **CIFAR-100에서 최대**(NNCLR 대비). 슬라이드 해석: 비슷한 class가 많은(fine-grained 성격) 데이터셋에서는 단일 NN보다 **여러 neighbor의 context**가 더 도움됨. 이건 해석(가설)이며 직접 검증된 건 아님 ("likely").
- All4One = NNCLR(neighbor) + Centroid(self-attention) + Barlow Twins식 Feature(redundancy reduction)의 결합이라 두 계열을 모두 개선하는 게 자연스러움.

---

## How Many Neighbors? (p.53)

**슬라이드 내용**
- bullet
  - A small Transformer is sufficient.
  - Five neighbors give the best result.
  - More neighbors introduce additional computation.
  - Less relevant neighbors can dilute the contextual representation.
- 표 (c) Number of transformer layers.

| Layer number | Top-1 | k-NN Top-1 |
|---|---|---|
| 3 | 72.17 | 64.84 |
| 6 | 71.86 | 64.50 |
| 9 | 71.75 | 64.52 |

- 표 (d) Number of NNs extracted.

| Number of NN | Top-1 | k-NN Top-1 |
|---|---|---|
| 5 | 72.17 | 64.84 |
| 10 | 72.00 | 64.54 |
| 15 | 71.92 | 64.63 |
| 20 | 71.79 | 64.6 |

- 하단 bullet
  - If the Transformer is too powerful, it might take over the role of generating representations from the backbone.
    - Since the goal of the paper is to train a backbone for downstream tasks, a smaller Transformer is more appropriate.
  - Having a large number of neighbors is not always beneficial;
    - as the number of neighbors increases, samples with low relevance may be included, potentially diluting the contextual information.

**설명**
- **Transformer layer 수**: 3층(가장 작음)이 최고(72.17), 6, 9층으로 깊어질수록 소폭 하락. 이유(슬라이드의 가설적 설명, "might"): Transformer가 너무 강하면 **표현을 만드는 역할을 backbone 대신 가져갈** 수 있어 backbone 학습이 약해질 수 있음. 논문 목표는 downstream에 쓸 **backbone** 학습이므로 작은 Transformer가 적절함 (centroid 계산용 보조 모듈이라는 위상).
- **이웃 수 k**: 5개가 최고, 20까지 늘릴수록 서서히 하락(72.17 → 71.79). 이유: k가 커지면 **관련성이 낮은 sample**이 섞여 context가 희석(dilute)되고, 연산량도 증가함. 차이는 크지 않지만(0.4 이내) 경향이 일관적이라 "많을수록 좋다"가 아님을 보임.
- 수치 차이가 작아 seed 분산이 있을 수 있음 ← 슬라이드에는 분산/반복 실험 언급 없음.

---

## (섹션 구분) Can cluster-based SSL preserve fine-grained visual details? (p.54)

섹션 구분 슬라이드: 진한 파랑 배경에 "Can cluster-based SSL preserve fine-grained visual details?" — 다음 주제(prototype/cluster 기반 SSL의 fine-grained 문제)의 질문 제시.

---

## Prototype-based Contrastive Learning with Stage-wise Progressive Augmentation for Self-Supervised Fine-Grained Learning (p.55)

**슬라이드 내용**
- 논문 첫 페이지 캡처: CVF 로고, "This ICCV paper is the Open Access version, provided by the Computer Vision Foundation. Except for this watermark, it is identical to the accepted version; the final published version of the proceedings is available on IEEE Xplore."
- 제목: "Prototype-based Contrastive Learning with Stage-wise Progressive Augmentation for Self-Supervised Fine-Grained Learning"
- 저자: Baofeng Tan$^1$, Xiu-Shen Wei$^{2*}$, Lin Zhao$^1$
- 소속: 1 School of Computer Science and Engineering, Nanjing University of Science and Technology / 2 School of Computer Science and Engineering, and Key Laboratory of New Generation Artificial Intelligence Technology and Its Interdisciplinary Applications, Southeast University
- 하단 출처: Tan et al., ICCV 2025, Prototype-based Contrastive Learning with Stage-wise Progressive Augmentation for Self-Supervised Fine-Grained Learning

**설명**
- 새 논문 소개 슬라이드: **Tan et al., ICCV 2025**. 제목 키워드 세 가지가 방법의 구성: (1) **Prototype-based contrastive learning** (cluster/prototype 기반 SSL), (2) **Stage-wise progressive augmentation**, (3) **self-supervised fine-grained learning**이 대상 문제.
- 앞 슬라이드(p.54)의 질문 "cluster-based SSL이 fine-grained 디테일을 보존할 수 있는가?"에 대한 답을 제시하는 논문으로 소개됨. 방법의 구체적 내용은 이후 슬라이드(p.57~)에서 다룸.

---

## Fine-Grained Recognition Needs Fine Details (p.56)

**슬라이드 내용**
- Generic recognition
  - "Is this a bird?"
  - Global shape may be sufficient.
- Fine-grained recognition
  - "Which bird species is this?"
  - Small parts become important.
  - Beak, head, wing, and tail patterns may distinguish classes.
- 하단 굵은 문장: **Fine-grained recognition requires both holistic and part-level representations.**

**설명**
- **Generic recognition**(새인가 아닌가)은 전체 윤곽(global shape)만으로 충분. **Fine-grained recognition**(어떤 종인가)은 class 간 차이가 **부리, 머리, 날개, 꼬리 무늬** 같은 작은 part에 있음 → class 간 variance는 작고 class 내 variance(자세, 배경)는 큼.
- 따라서 표현이 **holistic(전역) + part-level(국소)** 둘 다 담아야 함. 일반적 cluster/prototype 기반 SSL은 전역 의미 중심으로 sample을 묶어 **세부 디테일이 뭉개질 위험**이 있다는 문제의식이 깔려 있음 (다음 슬라이드 "Why Does Multi-Stage Alignment Fail?"로 이어짐).

> 보충: CUB-200-2011(새 200종), Stanford Cars, FGVC-Aircraft 등이 대표적 fine-grained benchmark.

---

## FIGURES

- p.49: All4One 전체 구조도 (encoder/momentum encoder, Feature·Neighbour·Centroid Contrast 세 경로) — 상
- p.50: objective ablation 표 (Table 3, CIFAR-100) — 상
- p.51: NN retrieval accuracy vs epoch 곡선 + Top-5 NN 정성 예시 — 중
- p.52: Linear evaluation 비교 표 (Table 1) — 중
- p.53: transformer layer 수 / neighbor 수 ablation 표 (c), (d) — 중

## EXAM POINTS

1. **Feature Contrast(Barlow Twins식)**: cross-correlation matrix에서 **diagonal → 1(view 간 invariance)**, **off-diagonal → 0(feature redundancy 제거)**. 수식 $L_{Red.}$의 두 항(sqrt 및 정규화)의 의미 설명.
2. **Neighbor loss vs Feature contrast 차이**: 전자는 sample 간(행) 비교, 후자는 feature dimension 간(열) 비교.
3. **Ablation 결론**: Centroid 단독(67.4)은 NNCLR(69.62)보다 약함 → centroid는 **보완 objective**이지 대체재 아님. NNCLR+Cen=71.02, 세 objective 모두=**72.17**(최고).
4. **Why 질문**: full objective가 NN retrieval 최고가 아닌데(v3 86.16 vs v6 82.16) linear eval은 최고인 이유 → class-correct retrieval은 목표가 아님, 다른 class 이웃도 유용한 시각 유사성 포함.
5. **Linear eval 결과 해석**: All4One가 NNCLR, Barlow Twins 모두 능가, CIFAR-100에서 개선폭 최대 → 유사 class 많은 데이터에서 multi-neighbor context가 유리.
6. **Why 질문**: 왜 작은 Transformer(3 layer)와 적은 이웃(k=5)이 최적? → 큰 Transformer는 backbone의 표현 학습 역할을 가져감, 이웃이 많으면 관련성 낮은 sample이 context를 희석 + 연산 증가.
7. **Fine-grained vs generic recognition**: 필요한 정보 수준 차이(global shape vs part-level 디테일), 둘 다 holistic + part-level 표현이 필요.
8. **논문 식별**: All4One (Estepa et al., ICCV 2023), Prototype-based CL w/ Stage-wise Progressive Augmentation (Tan et al., ICCV 2025).

## VERIFIER REPORT
- p.51 설명: 슬라이드에 근거 없는 "트럭 query와 다른 class 차량" 단정 제거, 상어/고래 예시만 남기고 슬라이드 미명시임을 표시.
- p.49 설명: "λ 튜닝 없이 스케일 맞추는 효과" 단정을 "λ를 따로 두지 않는 구성으로 보임"으로 완화.
- p.49 그림 설명: predictor에서 P_cc^1로 올라가는 경로 누락 보완.
- p.53 설명: Transformer가 backbone 역할을 가져간다는 단정을 슬라이드의 가설("might") 수준으로 완화.
- 수식, 표 수치(p.50/52/53), 논문명/연도, 페이지 누락은 원본과 일치 확인.
