<!-- DRAFT | lecture: 04_Contrastive_Learning_II | chunk: 09 | pages 65–72 -->
<!-- VERIFIED | fixes: 3 | verifier: sonnet -->

## Scaling the Teacher, Distilling the Students (p.65)

- Step 1
  - Train a large ViT-g teacher on LVD-142M.
- Step 2
  - Freeze the large teacher.
  - Distill its representation into smaller ViT models.

**그림 (왼쪽 아래, 표 (b) "Averaged metrics on 8 vision tasks")**: 2개의 표. 행은 (Arch, Method) = ViT-g/14 Scratch(회색, teacher), ViT-L/14 Scratch, ViT-L/14 Distill(굵게).
- 위 표 (INet-1k / Segm. / Depth↓ / Classif.)
  - ViT-g/14 Scratch: 86.5 / 73.4 / 1.00 / 92.1
  - ViT-L/14 Scratch: 84.5 / 72.2 / 1.10 / 90.2
  - ViT-L/14 Distill: **86.3 / 73.3 / 1.08 / 91.2**
- 아래 표 (Finegr. / Retriev. / ARSketch / Video)
  - ViT-g/14 Scratch: 78.3 / 75.2 / 77.0 / 69.3
  - ViT-L/14 Scratch: 75.8 / 71.3 / 69.5 / 67.3
  - ViT-L/14 Distill: **77.6 / 76.3 / 74.5 / 67.5**

**그림 (오른쪽 위, "(a) Comparison on individual metrics")**: radar(방사형) chart. 각 꼭짓점이 하나의 벤치마크(iNat18, Cars, Food, INet-1k, NYUd↓, Kitti↓, INet-R, INet-A, Paris-H, Oxford-H, Places205, iNat21 등)이고 축 값이 해당 metric 점수. 3개 다각형: ViT-L/14 Scratch(파란 실선, 가장 안쪽), ViT-L/14 Distill(주황 실선, 중간), ViT-g/14 Scratch(빨간 점선, 가장 바깥쪽 = teacher). Distill 다각형이 Scratch보다 거의 모든 축에서 바깥이고 teacher 점선에 근접함.

**Observation**
- Distilled ViT-L outperforms the same ViT-L trained from scratch.
- On some tasks, the student approaches or exceeds the teacher.

**Train one large model, then transfer its representation to practical models.**

**설명**
- 방법의 2단계 구조: (1) 큰 모델(ViT-g, 약 1B params)을 큐레이션 데이터셋 **LVD-142M** 으로 DINOv2 objective로 사전학습, (2) teacher를 **freeze** 하고 그 출력 표현을 더 작은 ViT(S/B/L)에 **distillation**함.
- 왜 이렇게 설계하나: 작은 모델을 SSL로 직접 scratch 학습하면 capacity 부족 + SSL objective의 학습 신호가 약해서 성능이 떨어짐. 큰 모델이 먼저 좋은 표현 공간을 만들어 두면 작은 모델은 그 공간을 **모방**하면 되므로 더 쉬운 문제가 됨. 즉 "scaling은 teacher에서, 배포 가능성은 student에서" 분리함.
- 표 해석: ViT-L Scratch → Distill로 INet-1k 84.5→86.3(+1.8), Retrieval 71.3→76.3(+5.0), ARSketch 69.5→74.5(+5.0) 등 전반적으로 큰 향상. 일부 지표(Finegr. 77.6 vs teacher 78.3, Segm. 73.3 vs 73.4)에서는 teacher에 근접하고 Retrieval(76.3 vs 75.2)은 **teacher를 초과**함 → 슬라이드의 "student approaches or exceeds the teacher"의 근거.
- 이전 방법과의 차이: 기존 self-distillation(DINO, MoCo 계열)은 같은 크기 모델끼리 EMA teacher를 쓰는 구조였다면, 여기서는 **별도의 frozen 대형 teacher → 소형 student** 의 일반적인 knowledge distillation 형태임.
- 표 (b)의 캡션 "Averaged metrics on 8 vision tasks"는 두 표의 8개 열(INet-1k, Segm., Depth, Classif., Finegr., Retriev., ARSketch, Video)이 각각 task 그룹별 평균 metric이라는 뜻으로 보임 → 불일치가 아니라 8열 = 8 task.

> 보충: DINOv2 (Oquab et al., 2023, Meta AI)에서 distillation 시 student는 DINO/iBOT loss 구조를 유지하고 teacher만 frozen ViT-g로 바꾸는 방식으로 알려져 있음. LVD-142M은 142M장의 자동 큐레이션 이미지 데이터셋(uncurated pool에서 curated 데이터와의 유사도 기반 retrieval + dedup으로 선별).

## One Frozen Encoder, Many Tasks (p.66)

- Global features: Classification, Fine-grained recognition, Image retrieval
- Dense features: Semantic segmentation, Monocular depth estimation, Pixel correspondence

**그림**: 2행 × 4열, 총 8개 작은 plot. x축은 모두 **flops (10^10 ~ 10^12, log scale)**, y축은 task metric. 패널: (1행) Inet-1k(Accuracy), Segmentation(mIoU), Monocular Depth ↓(R-MSE, 낮을수록 좋음), Classification(Accuracy); (2행) Finegrained Classification(Accuracy), Instance Retrieval(mAP), ImageNet-{A,R,Sketch}(Accuracy), Video Understanding(Accuracy). 범례: 주황 삼각형 = **SSL**(기존 self-supervised 모델들), 분홍 삼각형 = **WSL**(weakly-supervised, 텍스트 감독 모델들), 파란 선+삼각형 = **DINOv2**(모델 크기별 점을 이은 선). 가로 점선은 WSL 최고 수준. 거의 모든 패널에서 파란 선이 주황 SSL 점들보다 위에 있고 flops가 커질수록 상승해 WSL 점선에 도달/초과함. Depth 패널은 반대로 파란 선이 아래로 내려감(좋아짐)이며 SSL/WSL보다 훨씬 낮음. Segmentation·Depth·Retrieval에서는 WSL 점선도 크게 상회함.

**The same frozen backbone supports both image-level and pixel-level tasks.**

**설명**
- **Frozen encoder**: backbone을 고정하고 task별로 얇은 head(linear probe, kNN, 간단한 decoder 등)만 학습하는 평가 방식. backbone 표현 자체의 범용성을 측정함.
- **Global features**(이미지 → 벡터, CLS token 기반): classification, fine-grained, retrieval. **Dense features**(patch token 단위, 위치 정보 보존): segmentation, depth, correspondence. 두 종류를 모두 한 모델이 잘해야 "visual foundation model"이라 할 수 있음.
- 해석: 모델 크기(flops)가 커질수록 성능이 증가(scaling) → SSL도 scaling이 통한다는 증거. dense task에서는 WSL(CLIP류)을 크게 앞섬 → 텍스트 supervision은 global semantics에 강하지만 pixel-level 정보는 약하다는 점을 시사함.
- DINOv2에서 image-level loss(DINO) + patch-level loss(iBOT)를 함께 쓴 설계의 결과로 연결됨.

> 보충: 슬라이드에는 범례(SSL/WSL/DINOv2)만 있고 개별 모델 이름은 없음. 원 논문(DINOv2, Oquab et al., 2023)의 그림으로 보이며, WSL 점선은 최고 성능 WSL 모델(OpenCLIP/EVA-CLIP 계열로 추정)을 가리킴.

## What Does DINOv2 Teach Us? (p.67)

- The SSL objective alone is not enough.
  - Data quality determines what the model can learn.
  - Patch-level learning improves dense representations.
  - Large models require stable and efficient training.
  - Distillation makes large-scale SSL models practical.

**Can we continue scaling DINOv2 indefinitely?**

**설명**
- DINOv2 교훈 4가지 요약:
  1. **데이터 품질**: loss만 좋다고 되는 것이 아니고, uncurated 대규모 데이터보다 큐레이션한 LVD-142M이 표현 품질을 결정함 → "무엇을 학습할 수 있는가"는 데이터가 정함.
  2. **Patch-level learning**: image-level(DINO)만으로는 dense feature가 약해서 iBOT식 masked patch prediction을 추가해 dense task 성능을 높임.
  3. **안정적·효율적 학습**: ViT-g 규모에서는 학습 안정성·효율 기법이 필수임.
  4. **Distillation**: 큰 모델을 실사용 가능한 작은 모델로 옮김(p.65).
- 마지막 질문("indefinitely?")은 DINOv3로 넘어가는 hook: 더 오래·더 크게 학습하면 문제가 생김.

> 보충: 3번의 구체 기법(KoLeo regularizer, Sinkhorn-Knopp centering, FSDP, FlashAttention, sequence packing 등)은 슬라이드에 없는 DINOv2 논문 내용임.

## DINOv3: Scaling Reveals a New Problem (p.68)

- Longer training improves global features
  - Image classification continues to improve.
- Dense features gradually degrade
  - Patch similarity becomes noisy.
  - Segmentation performance decreases.

**그림**: 왼쪽에 입력 이미지 2장(위: 풀밭 위의 양 3마리, 아래: 노란 꽃과 울타리). 오른쪽에 학습 iteration별(200k, 400k, 600k, 800k, 1M) 5개 열의 cosine similarity map(viridis 색상: 어두운 보라=낮음, 노랑/초록=높음). 각 이미지의 빨간 점이 query patch. 200k·400k에서는 query 패치와 같은 객체(양, 꽃잎) 주변에 매끈하고 응집된 고유사도 영역이 나타남. 600k→1M으로 갈수록 객체 영역이 흐려지고 배경 곳곳에 **점상 노이즈 고유사도 패치**가 늘어남.

**설명**
- DINOv3(Siméoni et al., 2025)의 핵심 관찰: 학습을 길게(최대 1M iteration) 하면 **global metric(classification)은 계속 오르는데 dense metric(segmentation, patch similarity)은 오히려 나빠짐**.
- 그림 읽는 법: query patch와 모든 다른 patch 사이의 cosine similarity를 map으로 그린 것. 좋은 dense feature면 같은 객체 부위에서만 값이 높아야 함. iteration이 늘수록 무관한 배경에서도 값이 높아지므로 patch feature의 **locality / spatial consistency가 붕괴**함.
- 원인 직관: global objective(CLS-level loss)가 계속 최적화되면서 patch token이 global 정보를 흡수/공유해 patch 간 구분이 약해짐.

> 보충: 슬라이드는 원인을 명시하지 않음. 위 원인 설명은 DINOv3 논문의 설명(global-local trade-off, patch feature의 global 정보 오염)에 근거한 보충임. DINOv3는 ViT-7B 및 대규모(~17억 장) 데이터로 학습함(슬라이드 외 정보).

## Gram Anchoring (p.69)

- Select an earlier model with clean patch features.
- Compute pairwise similarities between its patch features.
- Preserve this similarity structure during longer training.

$$\mathcal{L}_{\text{Gram}} = \left\| \mathbf{X}_S \cdot \mathbf{X}_S^{\top} - \mathbf{X}_G \cdot \mathbf{X}_G^{\top} \right\|_F^2 .$$

**설명**
- 아이디어: 오염된 patch feature **값 자체**를 고정하는 게 아니라, patch들 간의 **상대적 유사도 구조(Gram matrix)** 를 과거의 "깨끗했던" 모델과 맞추도록 정규화함.
- 기호(슬라이드에는 정의 없음, 보충): $\mathbf{X}_S \in \mathbb{R}^{P\times D}$ = **S**tudent(현재 학습 중 모델)의 한 이미지에 대한 P개 patch feature(보통 L2-normalize), $\mathbf{X}_G$ = **G**ram teacher(깨끗한 earlier model)의 patch feature. $\mathbf{X}\mathbf{X}^\top\in\mathbb{R}^{P\times P}$ 는 patch 간 pairwise (cosine) similarity를 담은 **Gram matrix**. $\|\cdot\|_F^2$ = Frobenius norm 제곱 = 두 Gram matrix 원소 차이의 제곱합.
- 왜 feature 자체가 아니라 Gram matrix인가: feature를 직접 맞추면 student 표현 공간이 teacher 공간에 묶여 global 성능 개선 여지가 줄어듦. Gram matrix는 **feature 공간의 직교변환(회전)에 불변**($XR(XR)^\top = XX^\top$)이라 patch 간 **관계 구조**만 보존하고 feature 자체는 자유롭게 학습 가능 → global 성능은 계속 올리면서 dense 구조는 유지.
- 이전 방법과의 차이: 일반 feature distillation(p.65)은 representation 값을 맞추지만 Gram anchoring은 **second-order(similarity) 구조**만 맞춤.

> 보충: style transfer의 Gram matrix loss(Gatys et al., 2016)와 같은 형태. DINOv3 논문에서 Gram teacher는 earlier checkpoint이며, 고해상도 입력으로 얻은 teacher feature를 downsample해 쓰는 refinement 형태가 $\mathcal{L}_{\text{HRef}}$ (다음 슬라이드)임.

## Gram Anchoring (p.70)

**그림 (Figure 10)**: 3행 × 5열 이미지 격자. 행 라벨: original / w/o $\mathcal{L}_{\text{HRef}}$ / w/ $\mathcal{L}_{\text{HRef}}$. 5개 입력 이미지(노란 꽃, 거위 무리, 맥주잔과 초밥·꼬치 음식, 해안의 물개 무리, 풀밭의 양떼). 1행은 원본 입력 이미지(슬라이드 이미지에는 빨간 표시가 보이지 않고, query patch의 빨간 점은 2·3행 cosine map 위에 표시됨). 2행(w/o)은 noisy한 cosine map으로 유사도가 배경 곳곳에 퍼지고 격자 노이즈가 보임. 3행(w/)은 같은 query에 대해 부드럽고 선명하며 해당 객체(꽃잎, 거위 몸통, 물개, 양 등)에 집중됨.

캡션: Figure 10: Qualitative effect of Gram anchoring. We visualize cosine maps before and after using the refinement objective $\mathcal{L}_{\text{HRef}}$. The input resolution of the images is 1024 × 1024 pixels.

**Original**
- Input image
- Red cross indicates the query patch.

**w/o HRef**
- Similarity spreads to irrelevant regions.
- Object boundaries become unclear.
- Dense features contain spatial noise.

**w/ HRef**
- Similarity concentrates on corresponding object parts.
- Background responses decrease.
- Object structure and boundaries become clearer.

**Gram anchoring restores clean and spatially consistent patch similarities.**

**설명**
- p.68에서 본 문제(긴 학습 후 노이즈 낀 patch similarity)가 Gram anchoring으로 해결됨을 질적으로 보여주는 비교 그림.
- 입력 해상도는 1024×1024(캡션)이며, w/ HRef에서는 이 고해상도에서도 경계가 선명한 map이 나옴. (고해상도라 노이즈가 더 눈에 띈다는 해석은 슬라이드 근거 없음)
- 용어: p.69의 loss는 $\mathcal{L}_{\text{Gram}}$, 이 슬라이드는 $\mathcal{L}_{\text{HRef}}$ ← 두 표기가 슬라이드에서 설명 없이 혼용됨. HRef는 high-resolution refinement로 추정(원문 유지).

> 보충: DINOv3 논문(Siméoni et al., 2025)에서 $\mathcal{L}_{\text{HRef}}$ 는 Gram teacher에 고해상도 입력을 넣어 얻은 더 좋은 patch feature를 student 해상도로 맞춘 뒤 Gram loss를 계산하는 방식으로 설명됨.

## (제목 없음) 열린 질문 (p.71)

- Can positive views be generated rather than augmented?
- Can SSL scale into a general-purpose visual foundation model?
- Do we still need carefully designed augmentations?
- Recent studies reconsider how views are generated, matched, and scaled.

**설명**
- 제목 없는 마무리 슬라이드로 열린 질문 3개와 결론 1줄.
  1. **Positive view를 augmentation이 아니라 생성**으로 만들 수 있나? → augmentation 설계에 의존하지 않는 positive 생성 방향.
  2. SSL이 **general-purpose visual foundation model** 로 확장 가능한가? → DINOv2/v3가 부분적 답.
  3. 정교하게 설계한 augmentation이 여전히 필요한가? → SimCLR식 crop+color jitter 설계 이후 augmentation 의존을 줄이려는 흐름.
- 마지막 bullet: view의 **생성 / 매칭 / 확장** 방식 재고가 최근 연구 흐름임. 다음 주 강의 예고 성격.

> 보충: 구체 논문 인용은 슬라이드에 없음. 위 방향 설명은 일반 배경 지식임.

## Next Week.. (p.72)

- Continue on Contrastive Learning ..

**설명**: 다음 주 강의 예고. Contrastive Learning 내용이 이어짐.

## FIGURES

- p.65: ViT-g teacher / ViT-L scratch / ViT-L distill 비교 표 + radar chart — 중요도 상
- p.66: One Frozen Encoder 8-panel (flops vs metric, SSL vs WSL vs DINOv2) — 중요도 중
- p.68: iteration(200k~1M)별 patch cosine similarity map 열화 — 중요도 상
- p.70: Figure 10, Gram anchoring before/after cosine map 비교 — 중요도 상

## EXAM POINTS

1. DINOv2 distillation 2단계 절차(큰 ViT-g를 LVD-142M으로 학습 → freeze 후 작은 ViT에 distill)와 scratch SSL보다 좋은 이유.
2. Distilled ViT-L vs Scratch ViT-L vs teacher 비교: student가 teacher에 근접/초과하는 경우가 있음(예: Retrieval).
3. Global features vs dense features 구분과 각 해당 task, frozen encoder 평가의 의미.
4. DINOv2의 교훈 4가지(데이터 품질, patch-level learning, 안정적 대규모 학습, distillation).
5. DINOv3에서 발견된 문제: 학습이 길어질수록 global(classification)은 개선되지만 dense(patch similarity, segmentation)는 열화됨. 왜 문제인가.
6. Gram anchoring: $\mathcal{L}_{\text{Gram}}=\|X_SX_S^\top - X_GX_G^\top\|_F^2$ 의 기호 의미, 왜 feature 직접 matching이 아니라 Gram matrix를 쓰는가(회전 불변, 관계 구조만 보존).
7. Gram teacher를 "earlier model with clean patch features"로 선택하는 이유와 효과(w/ HRef에서 배경 응답 감소, 경계 선명).
8. 열린 질문: positive view를 생성할 수 있는가, augmentation 설계가 여전히 필요한가.

## VERIFIER REPORT

- p.65 설명: 캡션 "8 vision tasks"와 표 불일치 의심을 정정(8열 = 8 task).
- p.70 그림: 원본 행에 빨간 십자가 있다는 서술 삭제(빨간 점은 cosine map 행에만 있음); 3열(음식) 집중 대상 서술을 일반화.
- p.70 설명: 근거 없는 "고해상도라 노이즈가 더 눈에 띈다" 해석 제거.
