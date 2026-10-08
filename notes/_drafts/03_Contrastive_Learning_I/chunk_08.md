<!-- DRAFT | lecture: 03_Contrastive_Learning_I | chunk: 08 | pages 57–64 -->
<!-- VERIFIED | fixes: 3 | verifier: sonnet -->

## What Does Each Stage Learn? (p.57)

- The nearest samples reveal what each stage has learned.
- 그림: 3행(STAGE0, STAGE1, STAGE2) × 4열(Anchor | Top1, Top2, Top3) 격자. Anchor와 Top열 사이에 세로 점선. 회색 배경 위에 녹색 도트 텍스처 도형(오각형, 원, 사각형 등)이 놓인 합성 이미지.
  - STAGE0: Anchor = 녹색 도트 오각형. Top1 = 녹색 도트 오각형(회전), Top2 = 녹색 도트 원, Top3 = 녹색 도트 원 → 색이 같은 샘플이 근처에 옴.
  - STAGE1: Anchor = 녹색 도트 오각형. Top1 = 녹색 오각형, Top2 = 녹색 사각형/사다리꼴, Top3 = 녹색 도트 원 → 텍스처가 같은 샘플이 근처에 옴(모양은 제각각).
  - STAGE2: Anchor = 녹색 도트 오각형. Top1 = 녹색 오각형, Top2 = 녹색 오각형, Top3 = 흰색 줄무늬 오각형 → 모양이 같은 샘플이 근처에 옴(색·텍스처는 달라도 됨).
- 우측 텍스트:
  - Stage 0: Samples of the same color as the anchor are nearby
  - Stage 1: Samples with the same texture are nearby
  - Stage 2: Samples with the same shape are nearby
- 하단 굵은 문장: **MCL changes the dominant feature across stages.**

**설명**
- 임베딩 공간에서 anchor와 가까운 nearest neighbor를 보면 해당 stage의 encoder가 **무엇을 기준으로 유사하다고 보는지** 알 수 있음. 정성적 검증(qualitative probe)임.
- Trifeature(shape/texture/color가 독립적으로 변하는 합성 데이터)에서 stage가 올라갈수록 **지배적 feature가 color → texture → shape**로 바뀜. 이전 슬라이드(p.53–55)의 Multistage Contrastive Learning(MCL) 설계 의도와 일치함: Stage 1은 같은 색 cluster 안에서만 negative를 뽑아 color를 못 쓰게 하고, Stage 2는 같은 색 + 같은 텍스처 cluster 안에서만 뽑아 shape를 쓰게 만듦.
- 핵심: standard contrastive learning(SimCLR 등)은 가장 쉬운 discriminative feature(여기서는 color)에 **feature suppression**이 일어나 나머지를 학습 안 함. MCL은 stage마다 이미 학습한 feature를 negative 선택에서 무력화해 억눌린 feature를 순차적으로 끌어냄.
- Stage 2의 Top3가 흰색 줄무늬 오각형인 것처럼, 상위 stage encoder 단독으로는 이전 feature(color, texture)에 둔감해질 수 있음. 그래서 이전 feature를 잃지 않도록 최종 표현은 stage별 표현을 **concatenation**해서 씀(p.55).

## Results on Suppressed Features (p.58)

- MCL improves suppressed features without losing previously learned features.
- **Table 2**: Linear evaluation accuracy of MCL and the baselines. Bold indicates the best performance.

| | Trifeature: Shape | Texture | Color | CelebA: Hair | Smiling | Gender | CIFAR-MNIST: CIFAR | MNIST | Average |
|---|---|---|---|---|---|---|---|---|---|
| IFM | 0.99 | 0.99 | **1.00** | 0.82 | 0.70 | 0.93 | 0.11 | **0.99** | 0.82 |
| TS | 0.93 | **1.00** | **1.00** | 0.62 | 0.65 | 0.87 | 0.09 | **0.99** | 0.77 |
| FD | 0.75 | 0.73 | 0.78 | 0.81 | **0.89** | 0.94 | 0.78 | 0.86 | 0.82 |
| SimCLR | 0.81 | 0.99 | **1.00** | 0.84 | 0.75 | 0.94 | 0.29 | 0.98 | 0.83 |
| MCL | **1.00** | **1.00** | **1.00** | **0.85** | 0.79 | **0.95** | **0.87** | **0.99** | **0.93** |

**설명**
- 평가 방식은 **linear evaluation**: encoder를 freeze하고 linear classifier만 학습해 각 속성을 얼마나 잘 읽어낼 수 있는지 측정. 세 벤치마크:
  - **Trifeature**: shape/texture/color 각각 예측.
  - **CelebA**: Hair / Smiling / Gender 속성 예측.
  - **CIFAR-MNIST**: CIFAR 물체와 MNIST 숫자를 한 이미지에 합성한 데이터. CIFAR 클래스와 MNIST 클래스를 각각 예측.
- Baseline: IFM, TS, FD, SimCLR (feature suppression을 다루는 기존 방법들; 약어 풀네임은 슬라이드에 없음).
  - SimCLR는 Trifeature에서 Shape 0.81로 낮음(color/texture가 지배) → **suppression** 증거. CIFAR-MNIST의 CIFAR 0.29도 매우 낮음(MNIST가 쉬운 shortcut이라 CIFAR가 억눌림).
  - IFM·TS는 CIFAR 0.11/0.09로 거의 chance 수준 → MNIST만 학습하고 CIFAR는 완전히 억눌림.
  - FD는 CIFAR 0.78로 좋지만 Trifeature(0.75/0.73/0.78)와 MNIST(0.86)에서 떨어짐 → 억눌린 feature를 올리는 대신 기존 feature를 잃음.
- MCL은 Trifeature 3개 모두 1.00, CIFAR 0.87·MNIST 0.99, **Average 0.93**(2위 SimCLR 0.83 대비 +0.10). 따라서 "**억눌린 feature는 개선하면서 이미 학습한 feature는 잃지 않음**"이 뒷받침됨. Smiling은 FD(0.89)가 MCL(0.79)보다 높음 → MCL이 모든 항목에서 1등은 아님.
- 주의: MCL 결과는 stage별 표현을 concatenate한 것이라 차원이 baseline보다 큼(p.61 Limitation과 연결).

> 보충: Average는 8개 속성 점수의 평균. 예: MCL = (1.00+1.00+1.00+0.85+0.79+0.95+0.87+0.99)/8 ≈ 0.93.

## Multimodal Feature Suppression (p.59)

- **Table 3**: Accuracy of the MCL tuned CLIP models on the MMVP benchmark. Bold indicates the best performance across four stages. The STAGE0 model is the original OpenAI pre-trained model. The final MCL model result is obtained by using the concatenation of the representations across all four stages as the final representation.

| | O&D | PSF | S&C | Q&C | P&R | C&A | S&P | Texts | V&P | Average |
|---|---|---|---|---|---|---|---|---|---|---|
| ResNet STAGE0 | 6.7 | 6.7 | **46.7** | 0.0 | **13.3** | 46.7 | **33.3** | 6.7 | 13.3 | 19.3 |
| ResNet STAGE1 | **13.3** | **13.3** | 40.0 | **20.0** | 0.0 | 33.3 | 33.3 | 6.7 | 26.7 | 20.7 |
| ResNet STAGE2 | 6.7 | 6.7 | 33.3 | 6.7 | 6.7 | **73.3** | 13.3 | 13.3 | **33.3** | 21.5 |
| ResNet STAGE3 | 0.0 | 13.3 | 33.3 | 20.0 | 0.0 | 60.0 | 33.3 | **26.7** | 20.0 | 23.0 |
| ResNet MCL | 0.0 | 13.3 | 40.0 | 6.67 | 6.67 | 73.3 | 40.0 | 6.7 | 33.3 | 24.4 (밑줄) |
| ViT STAGE0 | **26.7** | 13.3 | 26.7 | 6.7 | 6.7 | 40.0 | 26.7 | **13.3** | 20.0 | 20.0 |
| ViT STAGE1 | 13.3 | **33.3** | **66.7** | 26.7 | **13.3** | 53.3 | 20.0 | 13.3 | **26.7** | 29.6 |
| ViT STAGE2 | 0.0 | 13.3 | 46.7 | **40.0** | 6.7 | 53.3 | 20.0 | 13.3 | 20.0 | 23.7 |
| ViT STAGE3 | 13.3 | 6.67 | 46.7 | 13.3 | 13.3 | **66.7** | **40.0** | 13.3 | 20.0 | 25.9 |
| ViT MCL | 6.67 | 20.0 | 73.3 | 13.3 | 13.3 | 80.0 | 46.7 | 13.3 | 26.7 | 32.6 (밑줄) |

- 하단 텍스트: Feature suppression is not an issue unique to SimCLR. While CLIP effectively represents object categories, it can overlook fine-grained details such as orientation, count, and spatial relationships. Applying MCL increased the average MMVP accuracy for ViT models from 20.0 to 32.6.

**설명**
- 질문: feature suppression이 SimCLR 같은 unimodal contrastive에만 국한된 문제인가? → 아님. **CLIP**(image-text contrastive)도 객체 카테고리 같은 쉬운 feature만 잘 학습하고 orientation, count, spatial relationship 같은 **fine-grained detail**을 놓침.
- **MMVP benchmark**: CLIP이 구분 못 하는 이미지 쌍으로 구성된 multimodal 시각 패턴 benchmark. 열(O&D, PSF, S&C 등)은 visual pattern 범주별 정확도이며 약어 풀네임은 슬라이드에 없음.
- 설정: 원본 OpenAI pre-trained CLIP이 STAGE0, MCL 방식으로 STAGE1~3까지 총 4단계. 최종 MCL은 **4 stage 표현을 concat**한 것. ResNet, ViT 두 backbone 모두 실험.
- 결과: ResNet 평균 19.3(Stage0) → 24.4(MCL), ViT 20.0 → **32.6**(약 +12.6p). 두 backbone 모두 MCL이 평균 최고(밑줄).
- 관찰: stage별로 잘하는 범주가 다름(ViT Stage1은 S&C 66.7, Stage3은 C&A 66.7). 어떤 단일 stage도 모든 범주에서 우수하지 않고 concat이 장점을 합침. 하지만 MCL이 개별 stage보다 낮은 범주도 있음(예: ResNet O&D 0.0). ViT Stage1 단독(29.6)이 Stage2/3보다 높아 stage가 올라간다고 단조 증가하지 않음.
- 절대 정확도가 30% 안팎으로 낮아 개선폭은 크지만 문제가 해결된 것은 아님.
- 표에 소수 표기가 6.7과 6.67 혼재 ← 표기 불일치(원문 유지).

> 보충: MMVP는 Tong et al., 2024 "Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs"에서 제안된 benchmark로 CLIP 임베딩은 비슷하지만 시각적으로 다른 이미지 쌍을 사용함. 열 약어와 9개 visual pattern의 정확한 대응은 슬라이드에 없어 추정임.

## Summary (p.60)

- Standard Contrastive Learning Learns the easiest discriminative features.
- MCL
  - Selects negatives with similar learned features.
- Result
  - Each stage learns previously suppressed features.
- 하단 굵은 문장: **Contrastive learning may learn enough to separate samples, but not enough to preserve all useful information.**

**설명**
- 표준 contrastive learning(InfoNCE 계열)은 positive/negative를 **구분할 수만 있으면** loss가 줄기 때문에, 가장 쉬운 feature 하나로 충분하면 거기서 멈춤 → 나머지 유용한 feature는 **suppressed**.
- MCL은 이전 stage가 학습한 feature(K-means cluster로 근사)가 비슷한 샘플을 negative로 골라 그 feature로는 구분 불가능하게 만듦 → 모델이 다른 feature를 배우도록 강제. 결과적으로 stage마다 억눌렸던 feature를 학습함.
- 마지막 문장이 핵심 메시지: contrastive objective의 목표는 "샘플 구분"이지 "정보 보존"이 아님 → **sample discrimination ≠ information preservation**.

## Limitation (p.61)

- Training multiple stages and encoders increases the **computational** load.
- Concatenating representations from each stage also **increases** the feature **dimension**.
- There is **no guarantee that K-means clusters accurately** represent actual semantic features.
- **Simply concatenating** representations from each stage may **not** be **optimal**.

**설명**
- (1) stage 수만큼 encoder를 따로 학습해야 하므로 학습 비용이 stage 수에 비례해 증가함.
- (2) concat이라 최종 표현 차원이 (stage 수 × 단일 차원)으로 늘어 downstream 비용이 증가함(슬라이드에는 차원 증가만 명시; overfitting 우려는 추정). baseline과 비교할 때 차원이 달라 공정성 이슈도 있음.
- (3) "이미 학습한 feature"를 **K-means cluster**로 근사하므로 cluster가 실제 semantic feature(색, 텍스처 등)와 일치한다는 보장이 없음.
- (4) 단순 concat은 stage별 feature의 중요도·중복을 고려하지 않아 최적이 아닐 수 있음.

> 보충: learned weighting이나 attention 기반 결합 등이 개선 방향이 될 수 있으나 슬라이드에는 없음.

## (섹션 구분) Are all negative samples equally unrelated? (p.62)

- 남색 배경에 한 줄 "Are all negative samples equally unrelated?" — 다음 주제(X-Sample Contrastive Loss)로 넘어가는 section divider.

## X-Sample Contrastive Loss 논문 (p.63)

- 논문 첫 페이지 캡처: "Published as a conference paper at ICLR 2025"
- 제목: **X-Sample Contrastive Loss: Improving Contrastive Learning with Sample Similarity Graphs**
- 저자: Vlad Sobal(1,2), Mark Ibrahim(1), Randall Balestriero(3), Vivien Cabannes(1), Diane Bouchacourt(1), Pietro Astolfi(1), Kyunghyun Cho(2,4,5), Yann LeCun(1,2)
- 소속: 1 Meta FAIR, 2 New York Univeristy ← 원문 오타("University"), 3 Brown University, 4 Genentech, 5 CIFAR
- 하단 출처: Sobal et al., ICLR 2025, X-SAMPLE CONTRASTIVE LOSS: IMPROVING CONTRASTIVE LEARNING WITH SAMPLE SIMILARITY GRAPHS

**설명**
- 다음 파트에서 다룰 논문 소개 슬라이드. "모든 negative가 똑같이 무관한가?"라는 질문에 대해 **sample 간 similarity graph**를 이용해 binary 대신 soft한 유사도 target을 쓰는 방법을 제안하는 논문임.
- 이 슬라이드에는 방법 내용이 없으므로 구체 수식은 이후 슬라이드 참조.

> 보충: X-Sample은 cross-sample 유사도 정보를 contrastive loss의 target에 반영한다는 의미로 추정됨(제목 기반).

## Are all negative samples equally unrelated? (p.64)

- Should another dog, a cat, and a piano be treated as equally negative?
- 그림: 좌측 큰 카드 **Anchor**(앉은 저먼 셰퍼드). 우측에 4개 카드 일렬 배치: **Positive**(서 있는 저먼 셰퍼드), **Negative**(흰색 말티즈 강아지), **Negative**(태비 고양이), **Negative**(그랜드 피아노).
  - Anchor → Positive: 초록색 굵은 화살표, 라벨 "Target = 1".
  - Anchor → 세 Negative: 빨간 화살표 3개(카드 아래에서 위로 향함), 각각 라벨 "Target = 0".
  - 세 Negative 아래에 빨간 중괄호와 문구 **"Treated equally"**.

**설명**
- 표준 contrastive loss(SimCLR, InfoNCE)의 target은 **binary**: positive는 1, batch 내 나머지는 모두 0. 따라서 다른 개(말티즈), 고양이, 피아노가 anchor와의 의미적 거리에 상관없이 **똑같이 밀려남(push away)**.
- 문제: 말티즈는 같은 "개"라 의미적으로 가깝고, 고양이는 중간, 피아노는 매우 먼데 모두 target 0. 의미 구조(semantic structure)를 훼손하고, 같은 클래스 샘플을 밀어내는 **false negative** 문제와 연결됨.
- 해결 방향: 샘플 간 **graded similarity**(연속값)를 target으로 사용하는 접근(X-Sample, 다음 슬라이드에서 구체화될 것으로 보임).
- MCL과의 연결: MCL은 "negative를 어떻게 고를지"를 바꿨고, 여기서는 "negative의 target 값 자체를 soft하게" 바꾸는 접근. 둘 다 negative를 균질하게 취급하지 말자는 관점임.

## FIGURES

- p.57: Anchor와 Top1~3 nearest sample을 stage별(STAGE0/1/2)로 보여주는 3×4 격자(지배 feature가 color → texture → shape로 변화) — 상
- p.58: Table 2, MCL vs IFM/TS/FD/SimCLR linear evaluation(Trifeature, CelebA, CIFAR-MNIST) — 상
- p.59: Table 3, MCL-tuned CLIP의 MMVP 결과(ResNet/ViT, stage별 + MCL) — 중
- p.64: Anchor/Positive/Negative(개, 고양이, 피아노) 모두 target 0으로 "Treated equally"를 보여주는 도식 — 상

## EXAM POINTS

1. **Stage별 지배 feature**: STAGE0 = color, STAGE1 = texture, STAGE2 = shape. 왜 그렇게 되는가? (이전 stage가 학습한 feature 기준으로 같은 cluster 안에서만 negative를 뽑으므로 그 feature로는 positive/negative 구분이 불가능해져 무력화되기 때문)
2. **Feature suppression 정의와 원인**: standard contrastive learning은 샘플을 구분하기 충분한 가장 쉬운 feature만 학습하고 나머지는 억눌림. "구분할 정도로만 학습하고 정보를 모두 보존하지 않음."
3. **MCL의 성능 주장**: suppressed feature는 개선하면서 이전 feature는 유지 (Table 2 Average 0.93 vs SimCLR 0.83, CIFAR-MNIST의 CIFAR 0.29 → 0.87).
4. **CLIP으로의 확장**: feature suppression은 SimCLR 고유 문제가 아님. CLIP은 orientation/count/spatial relationship 같은 fine-grained detail을 놓침. MCL로 ViT MMVP 평균 20.0 → 32.6.
5. **MCL vs Hard Negative Mining** (p.56 연결): 겉보기 유사하지만 목적이 다름 (어려운 샘플 찾기 vs 이전 학습 feature 재사용 방지).
6. **MCL의 한계 4가지**: 계산량 증가, concat에 따른 차원 증가, K-means cluster의 semantic 정확성 비보장, 단순 concat의 비최적성.
7. **Binary target의 문제(p.64)**: positive = 1, 모든 negative = 0으로 "treated equally" → 의미적으로 가까운 negative(다른 개)와 먼 negative(피아노)를 구분 못 함. 이를 다루는 논문이 X-Sample Contrastive Loss (Sobal et al., ICLR 2025).
8. **왜 concatenation이 필요한가?** 각 stage encoder는 이전 feature를 덜 쓰게 될 수 있으므로 모든 stage 표현을 합쳐야 전체 feature를 보존함.

## VERIFIER REPORT

- p.57 설명: "사라지는 건 아님. 다만 ... 둔감" 논리 모순 문장을 재서술 (상위 stage 단독은 이전 feature에 둔감 -> 그래서 concat).
- p.61 설명: 슬라이드에 없는 "overfitting 위험"을 추정임을 명시.
- EXAM POINTS 1: negative 선택 이유 문장을 명확히 수정.
- 표(Table 2, Table 3) 수치/bold, 그림 설명(p.57, p.64), 저자/소속/연도 인용 전부 원본과 일치 확인.
