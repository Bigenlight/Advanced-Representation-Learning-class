<!-- DRAFT | lecture: 09_Weakly_Supervised_Learning_II | chunk: 03 | pages 31–42 -->
<!-- VERIFIED | fixes: 8 | verifier: sonnet -->

## Good, but not good enough: SEC 세 loss (p.31–32)

슬라이드 텍스트 (p.31과 p.32는 같은 슬라이드가 두 번 들어 있어):
- CAM limitations
  - Activation maps are small and coarse
  - Focus only on discriminative regions
  - Poor object boundaries
- Three complementary losses
  - 1. Seeding Loss
    - Use CAM as reliable foreground seeds
    - Preserve highly confident object regions
  - 2. Expansion Loss
    - Encourage predictions to expand beyond the seeds
    - Use image-level labels via Global Weighted Rank Pooling (GWRP)
  - 3. Constrain-to-boundary Loss
    - Refine predictions using CRF
    - Encourage masks to align with image boundaries
- (굵은 글씨 박스) SEC (Seed, Expand, Constrain)
  - Seed: Preserve reliable CAM activations.
  - Expand: Grow seeds to cover the entire object using image-level supervision.
  - Constrain: Refine object boundaries with CRF-based regularization.
- 그림 (p.30과 같은 구조도, 오른쪽 위에 작게): 사진(Person, Cow 태그) → Segmentation CNN → class별 예측 mask. 위쪽 갈래: 사진 → Weak Localization → seed 지도(Person 분홍, Cow 초록, Backgr. 남색, 나머지는 회색 체크무늬 = label 없음) → Seeding Loss. 가운데 갈래: 예측 → Global Weighted Rank-Pooling(Person, Cow 태그) → Expansion Loss. 아래 갈래: 사진을 Downscale + 예측 → CRF → 다듬은 mask → Constrain-to-boundary Loss. 세 loss가 ⊕로 합쳐지고, 회색 화살표(gradient)가 Segmentation CNN 쪽으로 거꾸로 흘러가
- [Kolesnikov & Lampert ECCV 2016]

**설명**
- **한 줄**: CAM에서 확실한 곳만 seed로 쓰고(Seed), image label로 그 seed를 물체 전체로 키우고(Expand), CRF로 경계에 맞추는(Constrain) segmentation network 학습법이야.
- **누가 학습되나**: **Segmentation CNN 하나**야. Weak Localization은 seed를 만들어 주는 쪽이고, CRF도 학습되는 network가 아니라 출력을 경계에 맞게 다듬는 연산이야. 단 논문은 CRF를 거쳐서도 gradient를 흘려 (논문 Sec. 3.2, p.30 회색 화살표).
- 슬라이드는 "Use CAM as reliable foreground seeds"라고만 해. 논문 보충: background seed(그림의 남색)는 CAM이 아니라 별도 gradient-based saliency에서 가장 덜 눈에 띄는 곳을 골라 만들어.
- **CAM limitations 세 개**가 p.28의 두 문제를 조금 더 쪼갠 거야.
  - small and coarse, poor object boundaries → p.28의 "1. Resulting masks are not sharp"
  - focus only on discriminative regions → p.28의 "2. Focused on discriminative area only"
- p.30–32는 같은 SEC 슬라이드 묶음이야. p.30의 "⇢ Use boundary-aware propagation"이 이 SEC(특히 Constrain-to-boundary + CRF)를 가리켜.
- **loss 세 개가 각각 한 문제를 맡아.**
  1. **Seeding loss**: seed가 찍힌 pixel에서만 "이 pixel은 이 class"라고 학습해. seed가 없는 pixel(그림의 회색 체크무늬)엔 loss가 없어.
  2. **Expansion loss**: seed 바깥으로 넓히는 힘이야. pixel label이 없으니까 image label을 쓰고, pixel 점수를 image 점수로 모을 때 **GWRP**를 써.
  3. **Constrain-to-boundary loss**: network 출력을 CRF에 넣어서 이미지 경계에 맞춘 버전을 만들고, network 출력이 그걸 닮게 해.
- **GWRP(Global Weighted Rank Pooling)**: pixel 점수를 큰 순서로 줄 세운 다음, 순위가 내려갈수록 작은 weight를 줘서 평균해.
  - max pooling은 1등 pixel만 봐. 그러면 discriminative part 하나만 커도 image 점수가 높아져서 넓힐 이유가 없어.
  - average pooling은 모든 pixel을 똑같이 봐. 그러면 작은 물체도 사진 전체를 덮어야 점수가 높아져.
  - GWRP는 그 중간이야. 상위 몇 pixel만이 아니라 꽤 많은 pixel이 높아야 점수가 높아서, seed 바깥으로 넓히게 돼.
- **CRF(Conditional Random Field)** = 색이 비슷하고 가까운 pixel은 같은 label을 갖게 정리하는 그래프 모델이야. L08 p.25의 CRF는 "image 사이 window 고르기"용(node = image)이었고, 여기는 "pixel 사이 label 정리"용이라 같은 이름의 다른 쓰임이야.

> 보충 (논문 보충, 슬라이드엔 식이 없음): Kolesnikov & Lampert의 세 loss야. $f_{u,c}(X)$ = 위치 $u$에서 class $c$일 확률, $T$ = 사진에 있는 class 집합, $S_c$ = class $c$ seed 위치.
> - Seeding: $L_{seed} = -\frac{1}{\sum_{c\in T}|S_c|}\sum_{c\in T}\sum_{u\in S_c}\log f_{u,c}(X)$ ← seed 위치에서만 cross-entropy
> - GWRP: 점수를 내림차순 $f_{i_1,c}\ge f_{i_2,c}\ge\dots$로 놓고 $G_c = \frac{1}{Z(d_c)}\sum_{j=1}^{n}(d_c)^{j-1}f_{i_j,c}(X)$, $Z(d_c)=\sum_{j=1}^{n}(d_c)^{j-1}$. $d_c=0$이면 max pooling, $d_c=1$이면 average pooling이야.
> - Expansion: 있는 class(background 포함)는 $-\log G_c$로 키우고, 없는 class는 $-\log(1-G_c)$로 낮춰. class 종류마다 pooling 계수 $d$를 다르게 써 (없는 class는 $d=0$ = max pooling, background는 foreground보다 **큰** $d$ = average 쪽에 더 가까워 ($d_{bg}=0.999 > d_+=0.996$)).
> - Constrain: $L_{constrain} = \frac{1}{n}\sum_{u}\sum_{c} Q_{u,c}\log\frac{Q_{u,c}}{f_{u,c}(X)}$ ← CRF 출력 $Q$와 network 출력 $f$의 KL divergence
> - 전체: $L = L_{seed} + L_{expand} + L_{constrain}$

> 보충 (RL 비유, 슬라이드 밖 = 시험 범위 아님): GWRP의 weight $d^{j-1}$은 discount $\gamma^{t}$와 모양이 같아. $d=0$이면 1등만 보는 myopic, $d=1$이면 전부 똑같이 보는 $\gamma=1$이야. Constrain loss는 "CRF로 개선한 출력을 target으로 두고 KL로 distill"하는 구조라서, AlphaZero에서 MCTS로 개선한 policy를 network가 따라가게 하는 것과 닮았어.

---

## Good, but not good enough: Use transfer learning (p.33)

슬라이드 텍스트:
- 1. Resulting masks are not sharp
  - ⇢ Use transfer learning
- 2. Focused on discriminative area only
- "We could also use saliency cues"
- 그림 왼쪽: image labels (person, bicycle가 적힌 자전거 탄 사람 사진) → saliency (같은 사진에서 사람과 자전거 부분이 노랗게 칠해짐)
- 그림 오른쪽: image (person, table, chair 태그) → 빨간 점선 상자 Guide labeller 안에 Saliency → 노란 덩어리, Seeder → 듬성듬성한 class 색 점 → 둘을 합친 label 지도 → Dense classifier loss. 아래 줄: image → Segmenter convnet → 예측 지도 → Dense classifier loss. 점선 빨간 화살표가 loss에서 Segmenter convnet으로 돌아가
- [Oh et al. CVPR 2017]

**설명**
- **한 줄**: 이미 따로 학습된 saliency 모델로 "물체 전체가 어디까지인가"를 빌려 와서, CAM의 class 정보와 합쳐 pseudo mask를 만들어.
- **saliency** = 사진에서 눈에 띄는 물체(foreground)가 어디인지 나타내는 지도야. class 이름은 몰라.
- **왜 transfer learning이야?** saliency 모델은 이 task의 data가 아니라 다른 data로 미리 학습한 모델이야. 그 모델이 아는 "물체의 모양과 범위"를 가져다 쓰는 거라서 transfer야.
- 그림 왼쪽을 보면 image label은 "person, bicycle"만 말해 줘. saliency는 사람과 자전거 전체를 칠해 줘. 대신 어느 쪽이 person인지는 안 말해 줘.
- 그림 오른쪽이 학습 구조야. Guide labeller가 pseudo label을 만들고, Segmenter convnet이 그걸 target으로 학습해 (Dense classifier loss). 학습되는 건 Segmenter convnet이야.

---

## Saliency-guided Weak Supervision (p.34)

슬라이드 텍스트:
- Observation
  - CAM
    - Provides class-specific localization
    - Covers only discriminative regions
  - Saliency
    - Provides complete object extent
    - Class-agnostic foreground estimation
  - Combine CAM and saliency to generate class-specific pseudo masks
- (굵은 글씨 박스) Saliency-guided Weak Supervision
  - Seeder: Generate class-specific seeds from CAM.
  - Saliency: Estimate foreground regions.
  - Guide Labeller: Fuse both cues into pseudo masks.
  - Segmentation CNN: Learn dense pixel-wise predictions.
- 그림: p.33 그림을 작게 다시 넣음
- [Oh et al. CVPR 2017]

**설명**
- **핵심은 두 단서의 장단점이 정확히 반대라는 거야.**

| | CAM | Saliency |
|---|---|---|
| class를 아나 | 안다 (**class-specific**) | 모른다 (**class-agnostic**) |
| 범위 | discriminative part만 | 물체 전체 (**complete object extent**) |

- 그래서 합쳐. CAM이 "여기가 person"이라는 seed를 찍고, saliency가 "그 seed가 속한 덩어리는 여기까지"라고 범위를 정해 줘.
- **어떻게**
  1. **Seeder**: classifier의 CAM에서 class별 seed를 뽑아.
  2. **Saliency**: saliency 모델로 foreground 영역을 추정해.
  3. **Guide Labeller**: 두 단서를 합쳐 class별 pseudo mask를 만들어.
  4. **Segmentation CNN**: 그 pseudo mask를 target으로 pixel마다 class를 예측하게 학습해.
- **함정**: saliency가 class-specific이고 CAM이 class-agnostic이라고 뒤집은 문장이 나오기 좋아. CAM은 class마다 따로 지도를 만들어서 class-specific이야 (p.26 "Designed to Generate Per-Class Maps").

> 보충 (논문 보충): Oh et al.은 이 조합으로 fully supervised 성능의 약 80%를 얻었다고 보고해. 슬라이드엔 수치가 없어.

---

## Good, but not good enough: Enforce equivariance (p.35)

슬라이드 텍스트:
- 1. Resulting masks are not sharp
  - ⇢ Enforce equivariance
- 2. Focused on discriminative area only
- "Activation maps should be equivariant to the input"
- 그림 왼쪽: 같은 비행기 사진을 크기만 바꿔 세 장 → 각각의 CAM과 SEAM 지도. CAM은 크기마다 모양이 달라지고 뭉개져 있고, SEAM은 세 장 모두 비행기 모양을 비슷하게 따라가
- 그림 오른쪽: 버스 사진 → Affine Transform으로 줄인 버스 사진. 두 사진이 각각 CNN → CAM → PCM을 지나는 두 갈래(Siamese Network, 위 = 원본 주황 화살표, 아래 = 변환본 보라 화살표). 원본 갈래의 출력은 $y^o$(PCM 출력)와 $\hat y^o$(CAM), 변환본 갈래는 $\hat y^t$(CAM)와 $y^t$(PCM 출력). 오른쪽 상자 세 개: ECR Loss($y^o$, $\hat y^o$, $\hat y^t$, $y^t$ → L1 두 개), ER Loss($\hat y^o$, $\hat y^t$ → L1), Cls Loss($\hat y^o$, $\hat y^t$ → class 벡터)
- [Wang et al. CVPR 2020]

**설명**
- **equivariance(등변성)** = 입력을 변환하면 출력도 **같은 방식으로** 변환돼야 한다는 성질이야. 식으로 쓰면 $F(A(I)) = A(F(I))$, 여기서 $A$는 affine 변환(크기 조절, 뒤집기, 회전 등)이야.
- **invariance(불변성)** 와 헷갈리면 안 돼. invariance는 입력을 변환해도 출력이 **안 변하는** 거야. 분류는 invariant해야 해. 비행기를 줄여도 답은 "airplane"이야. segmentation은 equivariant해야 해. 비행기를 줄이면 mask도 같이 줄어야 해.
- 그림 왼쪽이 문제를 보여 줘. 같은 비행기인데 크기만 바뀌어도 CAM 모양이 달라져. 원래 그러면 안 되는 거야.
- L08 p.29에서도 "The predicted output should be equivariant to image transformations"라는 prior가 나왔어. 여기선 그걸 CAM에 loss로 걸어.

---

## SEAM: Self-supervised Equivariant Attention Mechanism (p.36)

슬라이드 텍스트:
- CAM Limitation
  - CAMs are often coarse and inconsistent
  - Small input transformations can produce different activation maps
- Key idea: Enforce Equivariance
  - Apply geometric transformations (e.g., affine transform)
  - Predict CAMs for both original and transformed images
  - Enforce consistency after compensating for the transformation
- PCM (Pixel Correlation Module) is a module that propagates CAM activations to other pixels within the same object by utilizing pixel-wise affinity among CNN features.
- Training objective
  - Classification Loss: Preserve image-level classification performance
  - Equivariant Regularization (ER) Loss: Encourage transformation-consistent CAMs
  - Cross-view Consistency (ECR) Loss: Align CAMs across different views ← 약자는 ECR인데 풀네임이 안 맞아. 논문(Wang et al., CVPR 2020) 표기는 **E**quivariant **C**ross **R**egularization
- 그림: p.35 그림을 작게 다시 넣음
- [Wang et al. CVPR 2020]

**설명**
- **한 줄**: 원본 사진과 변환한 사진의 CAM이 "변환만큼만 다르게" 나오도록 classifier를 학습시켜서, CAM을 더 일관되고 완전하게 만들어.
- **"compensating for the transformation"** 이 핵심 표현이야. 두 CAM을 그냥 비교하면 크기가 달라서 안 맞아. 원본 CAM에 같은 변환 $A$를 걸어 준 다음 비교해.
- **어떻게**
  1. 사진 $I$와 변환한 사진 $A(I)$를 **같은 weight**의 CNN 두 갈래(Siamese network)에 넣어.
  2. 각 갈래에서 CAM($\hat y$)을 만들고, PCM으로 다듬은 CAM($y$)도 만들어.
  3. 세 loss로 학습해: Classification + ER + ECR.
- **PCM(Pixel Correlation Module)**: CNN feature가 비슷한 pixel끼리는 같은 물체라고 보고, CAM 값을 그 pixel들로 퍼뜨려. 강하게 켜진 머리에서 덜 켜진 몸통으로 값이 번지는 식이야.
- **loss 세 개의 역할**
  - **Classification loss**: image label 분류 성능을 유지해. CAM은 분류 network에서 나오니까 분류가 무너지면 안 돼.
  - **ER loss**: 원본 CAM과 변환본 CAM이 변환을 보정한 뒤 같아지게 해 (L1).
  - **ECR loss**: 한 갈래의 PCM 출력과 다른 갈래의 원본 CAM을 서로 맞춰 (L1 두 개). 그림에서 $y^o$↔$\hat y^t$, $\hat y^o$↔$y^t$ 짝이야.
- **왜 self-supervised야?** 변환 $A$는 우리가 직접 걸었으니까 정답 관계($F(A(I)) = A(F(I))$)를 사람 label 없이 알아. 그 관계 자체가 supervision이야.

> 보충 (논문 보충, 슬라이드엔 식이 없음): $L_{ER} = \|A(\hat y^o) - \hat y^t\|_1$, $L_{ECR} = \|A(y^o) - \hat y^t\|_1 + \|A(\hat y^o) - y^t\|_1$, 전체 $L = L_{cls} + L_{ER} + L_{ECR}$. 논문은 PCM 출력끼리만 ER을 걸면 모든 pixel이 한 class로 쏠리는 local minimum에 빠져서 ECR처럼 갈래를 엇갈려 비교한다고 설명해. 또 "fully supervised segmentation에선 data augmentation 때 pixel label도 같이 변환되니까 equivariance가 저절로 걸리는데, image-level label은 변환해도 안 바뀌어서 그 제약이 없다"는 게 논문의 출발점이야.

> 보충 (RL 비유, 슬라이드 밖 = 시험 범위 아님): DrQ · RAD 같은 image augmentation RL은 Q 값이 augmentation에 **invariant**하게 만들어. SEAM은 지도가 변환을 **따라가게**(equivariant) 만들어. pick-and-place처럼 출력이 이미지 위의 heatmap인 policy라면 필요한 건 equivariance 쪽이야.

---

## Hide-and-Seek: random hiding (p.37)

슬라이드 텍스트:
- 1. Resulting masks are not sharp
- 2. Focused on discriminative area only
  - Problem
    - CAM activates only the most discriminative regions.
  - Solution (Hide-and-Seek)
    - Randomly hide image patches during training.
    - Encourage learning from the remaining visible parts.
    - Generate more complete localization maps.
  - Effect
    - The classifier is forced to discover other visible object parts.
    - CAM gradually covers a larger portion of the object.
- (굵은 글씨) Random hiding encourages the network to discover complementary object parts instead of relying on a single discriminative region.
- 그림: 개 사진(Training image, 가로 W, 세로 H)을 한 변 S인 격자로 나눔 → Epoch 1, Epoch 2, …, Epoch N마다 다른 칸들이 검게 가려진 사진 → 각각 CNN
- [Wang et al. CVPR 2020] ← slide cite 오류. Hide-and-Seek은 Singh & Lee, "Hide-and-Seek: Forcing a Network to be Meticulous for Weakly-supervised Object and Action Localization", ICCV 2017 (웹 확인). p.35–36의 SEAM cite가 그대로 남은 걸로 보여

**설명**
- **한 줄**: 학습할 때마다 사진의 칸 몇 개를 무작위로 가려서, classifier가 가장 쉬운 부분 하나에만 기대지 못하게 해.
- **문제**: classifier는 개의 얼굴만 봐도 "dog"를 맞혀. 그러면 CAM도 얼굴만 켜져.
- **어떻게**
  1. 사진을 S×S 크기 칸의 격자로 나눠.
  2. 매 epoch마다 칸을 무작위로 골라 가려.
  3. 가려진 사진으로 평범하게 classification을 학습해.
- **왜 돼?** 얼굴 칸이 가려진 epoch엔 얼굴 없이 맞혀야 해. 그래서 다리, 몸통 같은 다른 부분도 쓰게 돼. epoch마다 가려지는 칸이 달라서 결국 여러 부분을 골고루 배워.
- **바뀌는 건 학습 data뿐이야.** network 구조나 loss는 그대로야. 그래서 어떤 classifier에도 붙일 수 있어.
- **random**이 핵심 단어야. 어디를 가릴지 모델을 보고 정하지 않아. p.39의 adversarial erasing과 여기서 갈려.

> 보충 (논문 보충): 가린 칸은 검정이 아니라 dataset 평균 pixel 값으로 채워. test 때는 아무것도 안 가리고 전체 사진을 넣어. 원래 논문은 segmentation이 아니라 weakly-supervised **object localization**(box)과 action localization용이야.

---

## Let us play peek-a-boo! (p.38)

슬라이드 텍스트:
- 1. Resulting masks are not sharp
- 2. Focused on discriminative area only
  - ⇢ Let us play peek-a-boo!
- 그림: 표 형태. 열 = Bounding Box (AlexNet-GAP), Heatmap (AlexNet-GAP), Bounding Box (Our HaS), Heatmap (Our HaS)를 두 묶음. 행 = 원숭이, 개, 뱀, 소금통, 도자기 등. AlexNet-GAP heatmap은 얼굴 같은 한 점만 빨갛고, HaS heatmap은 몸통까지 넓게 퍼져. box도 HaS 쪽이 물체 전체에 더 가까워
- [Wang et al. CVPR 2020] ← p.37과 같은 cite 오류. 실제는 Singh & Lee, ICCV 2017

**설명**
- **peek-a-boo**(까꿍 놀이) = 가렸다 보여 주는 놀이야. Hide-and-Seek을 부르는 별명이야.
- 그림은 같은 network(AlexNet + GAP)를 그냥 학습했을 때와 HaS로 학습했을 때를 비교해.
  - 보통 학습: heatmap이 원숭이 얼굴, 개 머리처럼 **한 점**에 몰려.
  - HaS: heatmap이 몸통까지 **넓게** 퍼져서, 거기서 뽑은 box도 물체 전체에 가까워.
- 그림에서 빨간 box가 정답, 초록 box가 예측이야 (원 논문 그림 기준). HaS 쪽 초록 box가 빨간 box에 더 가까워.

---

## Let us play adversarial peek-a-boo! (p.39)

슬라이드 텍스트:
- 1. Resulting masks are not sharp
- 2. Focused on discriminative area only
  - ⇢ Let us play adversarial peek-a-boo!
- (adversarial instead of random erasures)
- 그림 위: Input Image ($t=1$) → Classification Network (conv1–conv7 → Average Pooling → fc) → Classification Loss ← label "dog". 오른쪽 "Classification Model"에서 Recurrent 화살표가 아래로 돌아옴
- 그림 아래 "Object Regions Mining with Adversarial Erasing": heatmap $H_t$ → 계단 함수 기호(threshold) → 가릴 영역 $F_t$(파란 바탕에 남색 덩어리) → $I_t$에서 빼기(⊖) → 지운 사진 $I_{t+1}$ → 다시 Input Image로
- 그림 맨 아래: $I$: $t=1, 2, 3$에서 점점 더 많이 지워진 개집 사진, $H$: $t=1, 2, 3$의 heatmap(켜지는 곳이 계속 옮겨 감) → The Mined Object Region(지운 영역을 합친 개 모양)
- [Wei et al. 2017]

**설명**
- **한 줄**: classifier가 가장 의존하는 곳(CAM이 제일 센 곳)을 일부러 지우고 다시 학습하는 걸 반복해서, 지운 영역들을 모아 물체 전체를 찾아.
- **어떻게**
  1. 사진 $I_t$로 classification network를 image label("dog")로 학습해.
  2. 그 network의 heatmap(CAM) $H_t$를 구해.
  3. $H_t$를 threshold로 잘라 제일 센 영역 $F_t$를 얻어. 그림의 계단 기호가 threshold야.
  4. $I_t$에서 $F_t$를 지워 $I_{t+1}$을 만들어 ($I_{t+1} = I_t \ominus F_t$).
  5. $I_{t+1}$로 1번부터 다시 해. 그림은 $t = 1, 2, 3$이야.
  6. 지운 영역 $F_1, F_2, F_3$를 합친 게 **mined object region**(foreground)이야. 이걸 segmentation 학습용 pseudo label로 써 (논문 보충: background는 saliency map에서 따로 얻어 합쳐).
- **왜 "adversarial"이야?** 지우는 쪽이 classifier가 제일 믿는 단서를 골라서 빼앗아. classifier 입장에선 방해꾼이야. 그래서 classifier는 다음으로 discriminative한 부분을 찾아야 해. 그림 heatmap이 개 머리 → 몸 → 다른 부분으로 옮겨 가.
- **HaS와 차이**: HaS는 **random**하게 가리고, AE는 **CAM이 가장 센 곳**을 골라 지워. 슬라이드 표현 그대로 "adversarial instead of random erasures"야.
- **L08과 연결**: L08 p.38 Ren et al.의 Concrete DropBlock도 "most relevant" region을 지워서 물체 전체를 보게 했어. 단 Ren은 지울 곳을 network(θ)가 min-max로 같이 학습하고, AE는 CAM을 threshold로 잘라서 단계마다 지우고 다시 학습해.

> 보충 (논문 보충): 원 논문 제목은 "Object Region Mining with Adversarial Erasing: A Simple Classification to Semantic Segmentation Approach" (Wei et al., CVPR 2017)이야. 지운 pixel은 평균 pixel 값으로 채우고, 너무 많이 지워 분류가 안 될 때쯤 멈춰. 최종 방법 이름은 AE-PSL이고, PSL(online Prohibitive Segmentation Learning)은 image-level class 점수가 낮은 class의 segmentation 점수를 "금지(prohibit)"하고, segmentation network 자신의 예측으로 mask를 online으로 갱신하는 단계야.

> 보충 (RL 비유, 슬라이드 밖 = 시험 범위 아님): imitation learning의 shortcut 문제(정답과 같이 나오는 쉬운 단서 하나에 기대는 것)와 같은 모양이야. 제일 센 단서를 가려서 다른 단서도 쓰게 만드는 건 robust adversarial RL에서 adversary가 agent가 기대는 걸 방해하는 것과 닮았어. HaS는 그 방해를 random으로 하는 domain randomization 쪽이야.

---

## Adversarial erasing 결과 그림 (p.40)

슬라이드 텍스트:
- 그림 (논문 Figure 6): 행 = Image, AE-Step1, AE-Step2, AE-Step3, Erased Regions. 열 = 개, 와인병, 의자, 새 두 마리, 두 사람, 조랑말, 병 든 사람, 오토바이, 식탁, 양
- Figure 6. Examples of mined object regions produced by the proposed adversarial erasing approach. The second to fourth rows show the produced heatmaps, where the discriminative regions are highlighted. The images with erased regions are shown in the last row in gray.
- [Wei et al. 2017]

**설명**
- 위에서 아래로 읽어. Step이 진행될수록 heatmap의 빨간 곳이 **다른 부분**으로 옮겨 가.
  - 예: 개는 Step1에서 머리가 제일 세고, Step2엔 머리 아래·가슴 쪽으로 옮겨 가고, Step3엔 몸통과 발 쪽이 켜져. 의자는 Step1·2에서 다리, Step3에서 앉는 판과 등받이 쪽까지 켜져.
- 맨 아래 회색이 단계별로 지운 영역을 합친 거야. 물체의 꽤 넓은 부분을 덮지만 (개의 다리, 의자처럼) 전부는 아니야.
- 단계마다 다른 부분을 찾으니까, 합치면 discriminative part 하나보다 훨씬 넓은 영역이 돼.

---

## Adversarial erasing 결과 표 (p.41)

슬라이드 텍스트:
- 1. Resulting masks are not sharp
- 2. Focused on discriminative area only
  - Let us play adversarial peek-a-boo!
- 그림 왼쪽 (논문 Figure 4): Image / Prediction / Ground Truth 세 열, 다섯 행(식탁의 사람들, 기차, 의자와 식탁, 오토바이, 사람). "Qualitative segmentation results on the VOC 2012 val set. One failure case is shown in the last row."
- 그림 오른쪽 (논문 Table 2): "Comparison of weakly-supervised semantic segmentation methods on VOC 2012 test set." 열 = Methods, Training Set, mIoU
  - Supervision: Box — WSSL (ICCV 2015) 10K 62.2, BoxSup (ICCV 2015) 10K 64.2
  - Supervision: Image-level Labels (* indicates methods implicitly use pixel-level supervision) — MIL-seg* (CVPR 2015) 700K 40.6, SN_B* (PR 2016) 10K 43.2, TransferNet* (CVPR 2016) 70K 51.2, AF-MCG* (ECCV 2016) 10K 55.5
  - Supervision: Image-level Labels — MIL-FCN (ICLR 2015) 10K 24.9, CCNN (ICCV 2015) 10K 35.6, MIL-sppxl (CVPR 2015) 700K 35.8, MIL-bb (CVPR 2015) 700K 37.0, EM-Adapt (ICCV 2015) 10K 39.6, DCSM (ECCV 2016) 10K 45.1, BFBP (ECCV 2016) 10K 48.0, STC (PAMI 2016) 50K 51.2, SEC (ECCV 2016) 10K 51.7, AF-SS (ECCV 2016) 10K 52.7
  - Supervision: Image-level Labels — AE-PSL (ours) 10K **55.7**

**설명**
- **mIoU(mean Intersection over Union)** = class마다 예측 mask와 정답 mask의 IoU(겹친 넓이 ÷ 합친 넓이)를 구해서 평균 낸 값이야. segmentation의 기본 점수야.
- 표에서 볼 것: image-level label만 쓴 방법 중 AE-PSL(55.7)이 제일 높고, 앞에서 본 SEC(51.7)보다 높아.
- 그래도 box를 쓴 방법(62.2, 64.2)보단 낮아. image label → pixel label의 간격이 아직 커. 이게 p.42 마지막 질문과 이어져.
- 왼쪽 그림 마지막 줄은 실패 사례야. 사람의 몸 일부를 놓쳤어.
- 표 수치는 시험 비중이 낮아. "AE-PSL > SEC, box supervision이 image-level보다 높다" 정도만 기억해.

---

## Takeaways (p.42)

슬라이드 텍스트:
- Training a classifier, and hoping to get segmentations out of it ✗
- Using spatial continuity priors will help, but to a limit ~
- The training can be modified to promote covering the object extent ✔
- With careful changes it is possible to obtain significant improvements ~
- ⇒ How far can we go from image-level to pixel-level labels ?

**설명**
- 표시 세 개(✗, ~, ✔)를 그대로 기억해. T/F로 뒤집기 좋아.
  - **✗**: classifier를 학습해 놓고 거기서 segmentation이 저절로 나오길 바라는 건 안 돼. p.23 gray box의 "Asking apples from the oranges tree"와 같은 얘기야.
  - **~**: spatial continuity prior(가까운 pixel은 같은 label이라는 가정, 예: CRF, boundary-aware propagation)는 도움이 되지만 한계가 있어.
  - **✔**: **학습 방식 자체를 바꿔서** 물체 전체를 덮게 하는 게 효과가 있어. HaS, adversarial erasing, SEC의 expansion loss가 여기 속해.
  - **~**: 신중하게 바꾸면 크게 좋아질 수 있어. 하지만 쉽진 않아.
- 마지막 줄은 질문이야. image-level label만으로는 어디까지 갈 수 있나. 다음 파트(Webly p.43–, Boxly p.53–)는 image label 말고 다른 정보(web 이미지, box)를 써서 이 질문에 답해.
