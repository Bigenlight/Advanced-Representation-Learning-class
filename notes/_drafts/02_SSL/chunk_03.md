<!-- DRAFT | lecture: 02_SSL | chunk: 03 | pages 17–24 -->
<!-- VERIFIED | fixes: 5 | verifier: sonnet -->

## Self-Supervised Pretext Tasks (p.17)

- Example: learn to predict image transformations / complete corrupted images
- 그림 4개가 가로로 나열됨 (모두 고양이 사진 기반):
  - **image completion**: 고양이 하반신만 남고 위쪽 절반이 회색 박스에 "?" 로 가려진 이미지
  - **rotation prediction**: 회전된 고양이 이미지. 파란 화살표(기준 상향 방향)와 초록 화살표(회전된 방향) 사이 각도에 "θ=?" 표시
  - **"jigsaw puzzle"**: 고양이 이미지를 4조각(2x2)으로 잘라 섞어 놓은 모습
  - **colorization**: 흑백(grayscale) 고양이 이미지 (원래 색을 예측하는 과제)
- 1. Solving the pretext tasks allow the model to learn good features.
- 2. We can automatically generate labels for the pretext tasks.

**설명**
- **Pretext task**는 데이터 자체에서 만든 "가짜" 지도학습 문제임. 사람이 라벨을 달지 않고 입력을 변형/가림 처리한 뒤 그 변형 정보(회전 각도, 가려진 영역, 조각 순서, 원래 색)를 정답으로 씀.
- 핵심 논리 2가지: (1) **pretext를 풀려면 모델이 물체 형태, 구조, 의미 같은 semantic feature를 이해해야 함** → 좋은 representation이 부산물로 학습됨. (2) **라벨이 공짜로 자동 생성**되므로 대규모 unlabeled data 활용 가능.
- 네 가지 예시는 크게 두 계열: 변환 예측(rotation, jigsaw)은 classification 형태, 복원(completion, colorization)은 reconstruction/생성 형태 (단, Zhang et al.의 colorization은 양자화된 색 bin에 대한 classification으로 정식화하기도 하므로 "복원=regression"은 일반화일 뿐임).
- 어떤 pretext를 고르느냐에 따라 학습되는 feature의 성격이 달라짐. 이후 슬라이드에서 relative position, inpainting을 자세히 다룸.

> 보충: rotation prediction은 Gidaris et al., 2018 (RotNet), jigsaw puzzle은 Noroozi & Favaro, 2016, colorization은 Zhang et al., 2016 (Colorful Image Colorization)이 대표 논문. (슬라이드에는 인용 없음)

## How to Evaluate a Self-Supervised Learning Method? (p.18)

- Learn good feature extractors from self-supervised pretext tasks, e.g., predicting image rotations.
- Attach a shallow network on the feature extractor; train the shallow network on the target task with small amount of labeled data.
- 하단 다이어그램: 왼쪽에서 오른쪽으로 5단계 파이프라인 (화살표로 연결)
  1. **Unlabeled Data**: "Lots of unlabeled data", 데이터베이스 아이콘 + 다양한 이미지 썸네일 격자(동물, 풍경 등)
  2. **Self-Supervised Learning**: Pretext task (e.g., rotation prediction). 새 이미지 → conv layers(보라색 블록) → fc(흰 블록) → 출력 "90°"
  3. **Feature Extractor**: Learned representation (e.g., ConvNet). 점점 작아지는 파란색 conv 블록들
  4. **Supervised Learning (Downstream task)**: Train a shallow network with a small amount of labeled data. 아래에서 "Small amount of labeled data on the target task" 데이터베이스가 위로 화살표를 올려 공급
  5. **Evaluate on the target task**: e.g., classification, detection. 새 이미지 → 네트워크 → "bird", 자동차 이미지에 detection bounding box(빨간 박스)

**설명**
- SSL 방법의 좋고 나쁨은 **pretext 성능이 아니라 downstream task 성능**으로 평가함. pretext는 수단일 뿐이므로.
- 절차: (1) unlabeled data로 pretext 학습 → (2) encoder(conv layers)만 떼어 feature extractor로 사용 (pretext 전용 fc head는 버림) → (3) 그 위에 얕은 네트워크(linear classifier 등)를 붙여 소량의 labeled data로 학습 → (4) target task 성능 측정.
- (슬라이드는 "shallow network를 학습"이라고만 적고 encoder freeze 여부는 명시하지 않음. 아래는 linear probing 관점의 해석.)
- 얕은 네트워크만 학습하는 이유: 성능 차이를 **feature extractor가 가진 representation의 품질** 덕으로 돌리기 위함. 이게 곧 **linear probing / linear evaluation** 프로토콜의 아이디어.
- 소량 라벨만 쓴다는 점이 중요: label-scarce 상황에서 SSL pretraining이 얼마나 도움 되는지를 보여줌.

> 보충: 평가 방식은 (a) frozen encoder + linear probe, (b) 전체 fine-tuning, (c) few-shot / semi-supervised(1%, 10% labels), (d) detection/segmentation 같은 transfer learning 등이 있음. 슬라이드는 (a) 중심 설명.

## Pretext Tasks (p.19)

- 섹션 구분 슬라이드: "Pretext Tasks" (진한 남색 배경, 흰 글씨 제목만 있음)

## Pretext Task (p.20)

- Self-supervised task used for learning representations
- Often, not the "real" task (like image classification) we care about

**설명**
- **Pretext task** 정의: representation 학습을 위해 인위적으로 설계한 self-supervised task. 우리가 실제로 풀고 싶은 것(downstream, 예: image classification)과는 보통 다름.
- 따라서 pretext 정확도 자체는 목적이 아니고, 그 과정에서 얻어지는 **encoder의 feature**가 목적임. (p.16의 pretext vs downstream 구분과 연결)
- 설계 원칙: 풀기 위해 high-level semantic 이해가 필요하되, 낮은 수준의 shortcut(예: 경계선 연속성, chromatic aberration)으로 풀리지 않아야 좋은 feature가 나옴.

> 보충: Doersch et al.은 실제로 patch 사이 경계 pixel 연속성 shortcut을 막기 위해 patch 사이 gap과 jitter를 두었고, chromatic aberration shortcut은 color dropping / projection으로 대응함 (슬라이드 p.23 표의 "Ours-projection", "Ours-color-dropping"과 연결).

## Pretext Task1: Relative Position of Patches (p.21)

- 출처: Doersch et al., ICCV 2015, Unsupervised visual representation learning by context prediction
- 왼쪽 그림: 고양이 사진 위에 가운데 **파란 박스**(고양이 얼굴, 기준 patch)와 그 주변 8개 위치에 **빨간 점선 박스**로 번호 1~8이 매겨져 있음 (1 2 3 / 4 _ 5 / 6 7 8 순서, 가운데는 파란 박스)
- 오른쪽 위 "Example:" 두 patch(파란 테두리 = 고양이 얼굴 중심 patch, 빨간 점선 = 고양이 귀 patch) → 3x3 격자에 중앙 파란 patch, 좌상단에 귀 patch가 놓인 그림
- 오른쪽 아래: X = (얼굴 patch, 귀 patch); **Y = 3** ← 왼쪽 그림의 번호 규칙(1이 좌상단, 3이 우상단)대로면 귀 patch가 좌상단에 놓인 예시 격자와 Y=3이 일치하지 않아 보임 (원 논문 번호 규칙 확인 필요)

**설명**
- **방법**: 이미지에서 기준 patch(중앙)와 그 이웃 8방향 중 하나의 patch를 뽑아 한 쌍 (X)으로 만들고, 이웃 patch가 기준 patch의 **어느 위치(8개 중)** 인지 맞추는 **8-way classification** 문제로 정의함. 정답 Y는 위치 index.
- **왜 유용한가**: 위치를 맞히려면 "고양이 귀는 얼굴 위쪽 좌/우에 있다" 같은 **물체의 구조와 부분 간 공간 관계**를 알아야 함 → object part 수준의 semantic을 학습.
- 라벨(Y)은 patch를 어디서 잘랐는지에서 자동으로 나오므로 사람 라벨 불필요.
- 입력은 쌍(2 patch)이고 출력은 8-class → cross-entropy로 학습하는 일반 classification.

> 보충: 원 논문은 patch 사이에 약 half patch 정도의 gap과 random jitter(최대 7 pixel)를 줘서 단순한 low-level 경계 일치로 풀지 못하게 함. 이런 trivial shortcut 회피가 pretext 설계의 핵심 포인트.

## Pretext Task1: Relative Position of Patches (p.22)

- 출처: Doersch et al., ICCV 2015
- 왼쪽: **Siamese(two-stream) 구조 다이어그램**. Patch 1, Patch 2가 각각 같은 구조의 AlexNet 유사 tower로 입력됨 (점선으로 층끼리 연결 = **weight sharing**):
  - conv1 (11x11,96,4) → pool1 (3x3,96,2) → LRN1 → conv2 (5x5,384,2) → pool2 (3x3,384,2) → LRN2 → conv3 (3x3,384,1) → conv4 (3x3,384,1) → conv5 (3x3,256,1) → pool5 (3x3,256,2) → fc6 (4096)
  - 두 tower의 fc6 출력이 합쳐져(concat) fc7 (4096) → fc8 (4096) → fc9 (8) 로 이어짐. fc9 출력 8개가 3x3 격자의 8개 이웃 위치(화살표로 표시)에 대응
  - 아래: X = (고양이 얼굴, 귀 patch); Y = 3
- 오른쪽: **Nearest Neighbors in features** 시각화. 열 구성: Input | Relative-positioning | Random Initialization | ImageNet AlexNet. 각 행은 쿼리 patch(바퀴, 말 다리, 물결, 고양이 얼굴, 비행기, 사람 얼굴, 자동차 등)에 대해 해당 feature 공간에서 가장 가까운 patch 5개를 보여줌.
  - Relative-positioning: 같은 종류 물체(바퀴, 고양이, 비행기 등)를 잘 찾아옴
  - Random Initialization: 색/질감 등 low-level만 비슷하고 의미는 안 맞음
  - ImageNet AlexNet(supervised): 비교 기준, 가장 의미적으로 일관됨

**설명**
- 구조 설계 이유: 두 patch를 **각각 독립적으로 같은 encoder(shared weights)** 로 embedding한 뒤 마지막에 합쳐서 상대 위치를 판단. 이렇게 하면 encoder는 patch 하나만 보고도 의미 있는 feature(예: "이건 고양이 귀")를 만들어야 하고, 이것이 곧 downstream에 재사용할 feature extractor가 됨. fc7 이후(합쳐지는 부분)는 pretext 전용 head.
- 이전 방식(단순 random init)과의 차이: nearest neighbor 결과는 pretext 학습만으로도 **의미적으로 유사한 patch가 가까이 모인다**는 정성적 증거. 라벨 없이 학습했는데도 supervised ImageNet AlexNet과 질적으로 비슷한 retrieval 경향을 보임 (정량적 동등성을 의미하지는 않음; 슬라이드는 정성적 비교만 제시).
- fc9 (8) = 8-way classification head.

> 보충: LRN = Local Response Normalization (AlexNet 계열 구성요소). 슬라이드 conv 표기는 (kernel, #filters, stride) 형식으로 보임.

## Pretext Task1: Relative Position of Patches (p.23)

- Self-supervised pretraining significantly improves downstream object detection performance, even without human annotations.
- 출처: Doersch et al., ICCV 2015
- **Table 1. Results on VOC-2007** (VOC-2007 Test, 클래스별 AP와 마지막 열 mAP). 캡션: "R-CNN performance with our unsupervised pre-training is 5% MAP better than training from scratch, but still 8% below pre-training with ImageNet label supervision."
- 표 열: aero, bike, bird, boat, bottle, bus, car, cat, chair, cow, table, dog, horse, mbike, person, plant, sheep, sofa, train, tv, mAP

| Method | aero | bike | bird | boat | bottle | bus | car | cat | chair | cow | table | dog | horse | mbike | person | plant | sheep | sofa | train | tv | mAP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DPM-v5 [17] | 33.2 | 60.3 | 10.2 | 16.1 | **27.3** | 54.3 | 58.2 | 23.0 | 20.0 | 24.1 | 26.7 | 12.7 | 58.1 | 48.2 | 43.2 | 12.0 | 21.1 | 36.1 | 46.0 | 43.5 | 33.7 |
| [8] w/o context | 52.6 | 52.6 | 19.2 | 25.4 | 18.7 | 47.3 | 56.9 | 42.1 | 16.6 | 41.4 | 41.9 | 27.7 | 47.9 | 51.5 | 29.9 | 20.0 | 41.1 | 36.4 | 48.6 | 53.2 | 38.5 |
| Regionlets [55] | 54.2 | 52.0 | 20.3 | 24.0 | 20.1 | 55.5 | 68.7 | 42.6 | 19.2 | 44.2 | **49.1** | 26.6 | 57.0 | 54.5 | 43.4 | 16.4 | 36.6 | 37.7 | **59.4** | 52.3 | 41.7 |
| Scratch-R-CNN [2] | 49.9 | 60.6 | 24.7 | 23.7 | 20.3 | 52.5 | 64.8 | 32.9 | 20.4 | 43.5 | 34.2 | 29.9 | 49.0 | 60.4 | 47.5 | 28.0 | 42.3 | 28.6 | 51.2 | 50.0 | 40.7 |
| Scratch-Ours | 52.6 | 60.5 | 23.8 | 24.3 | 18.1 | 50.6 | 65.9 | 29.2 | 19.5 | 43.5 | 35.2 | 27.6 | 46.5 | 59.4 | 46.5 | 25.6 | 42.4 | 23.5 | 50.0 | 50.6 | 39.8 |
| Ours-projection | 58.4 | 62.8 | **33.5** | 27.7 | 24.4 | **58.5** | 68.5 | 41.2 | **26.3** | **49.5** | 42.6 | **37.3** | **55.7** | **62.5** | 49.4 | **29.0** | **47.5** | 28.4 | 54.7 | 56.8 | 45.7 |
| Ours-color-dropping | **60.5** | **66.5** | 29.6 | **28.5** | 26.1 | 56.1 | **70.4** | **44.8** | 24.6 | 45.5 | 45.4 | 35.1 | 52.2 | 60.2 | **50.0** | 28.1 | 46.7 | **42.6** | 54.8 | **58.6** | **46.3** |
| Ours-Yahoo100m | 56.2 | 63.9 | 29.8 | 27.8 | 23.9 | 57.4 | 69.8 | 35.6 | 23.7 | 47.4 | 43.0 | 29.5 | 52.9 | 62.0 | 48.7 | 28.4 | 45.1 | 33.6 | 49.0 | 55.5 | 44.2 |
| Ours-VGG | 63.6 | 64.4 | 42.0 | 42.9 | 18.9 | 67.9 | 69.5 | 65.9 | 28.2 | 48.1 | 58.4 | 58.5 | 66.2 | 64.9 | 54.1 | 26.1 | 43.9 | 55.9 | 69.8 | 50.9 | 53.0 |
| ImageNet-R-CNN [19] | 64.2 | 69.7 | 50 | 41.9 | 32.0 | 62.6 | 71.0 | 60.7 | 32.7 | 58.5 | 46.5 | 56.1 | 60.6 | 66.8 | 54.2 | 31.5 | 52.8 | 48.9 | 57.9 | 64.7 | 54.2 |

**설명**
- **실험 설정**: pretext(relative position)로 AlexNet을 pretrain → 이를 초기화로 R-CNN을 PASCAL VOC 2007 detection에 fine-tune → mAP 측정.
- **결과 해석**: (1) scratch(random init) 40.7(Scratch-R-CNN)/39.8(Scratch-Ours) 대비 self-supervised pretrain은 **45.7~46.3 mAP로 약 +5~6.5%p** 향상 (Scratch-R-CNN 40.7 기준 +5.0~5.6, Scratch-Ours 39.8 기준 +5.9~6.5; 캡션은 "5% better"로 요약). (2) 그러나 ImageNet **label supervision** pretrain(54.2)보다는 약 **8%p 낮음** → 당시 SSL은 supervised에 못 미쳤지만 "라벨 없이도 의미 있는 pretrain이 가능하다"는 증거.
- Ours-projection / Ours-color-dropping은 chromatic aberration shortcut을 막는 두 가지 방식(색 채널 projection, 채널 dropping)으로 보임. Ours-Yahoo100m은 더 큰 unlabeled 데이터(Yahoo 100M images)로 학습한 경우로 보임. ← 표상 Yahoo100m(44.2)이 color-dropping(46.3)보다 낮음: 데이터가 많다고 반드시 좋아지지 않음 (원 논문 설정 차이 확인 필요).
- Ours-VGG(53.0)는 VGG 아키텍처 사용으로 추정되며 ImageNet-R-CNN(54.2)과 거의 근접.

> 보충: mAP = mean Average Precision (클래스별 AP 평균). 표의 DPM-v5, Regionlets 등은 non-deep 또는 이전 detection 방법으로 baseline. 위 표의 일부 셀 값은 이미지 판독 결과이므로 오독 가능성 있음.

## Pretext Task2: Inpainting (p.24)

- 출처: Pathak et al., CVPR 2016, Context Encoders: Feature Learning by Inpainting
- Can we predict missing regions from surrounding context?
- To fill the missing region, the network must understand:
  - Object parts
  - Scene structure
  - Semantic context
- 오른쪽 그림 2장: (왼쪽) 건물 상점가 거리 사진(아래쪽에 자동차)의 중앙부 정사각형 영역이 **흰색으로 가려진** 입력, (오른쪽) 네트워크가 가려진 영역을 채워 넣은 복원 결과(창문 같은 구조가 흐릿한 선으로 생성되고 붉은/분홍 색감이 주변 상점 외벽과 이어짐; 해상도·선명도는 낮음)

**설명**
- **Inpainting pretext**: 이미지 일부(보통 중앙 block)를 가리고, 나머지 **주변 context로부터 가려진 영역을 생성**하게 함. 정답은 원본 이미지의 해당 영역 → 라벨 자동 생성.
- 왜 좋은 representation이 되나: 빈칸을 그럴듯하게 채우려면 (1) **object parts**(창문이 있으면 이어지는 창문 패턴), (2) **scene structure**(건물 층 구조, 도로-건물 배치), (3) **semantic context**(가게 간판, 건물이라는 장면 의미)를 이해해야 하기 때문.
- p.21의 relative position과 차이: relative position은 discriminative(8-way classification)이고 inpainting은 **generative / pixel-level reconstruction** 문제임. 출력이 pixel이라 spatial 정보가 풍부하게 필요함.

> 보충: Context Encoder는 encoder-decoder 구조에 reconstruction (L2) loss와 adversarial loss(GAN)를 함께 사용. L2만 쓰면 결과가 흐릿(blurry)해지는 경향이 있고 adversarial loss가 선명도를 높임. 이는 이후 MAE 같은 masked modeling의 선구 격 아이디어. (슬라이드 p.25에서 "Learn from reconstruction" 이어짐)

## FIGURES

- p.17: pretext task 4종(image completion, rotation prediction, jigsaw, colorization) 예시 (상)
- p.18: SSL 평가 파이프라인 5단계 다이어그램 (unlabeled → pretext → feature extractor → downstream → evaluate) (상)
- p.21: 8개 상대 위치 격자 + X, Y 예시 (상)
- p.22: Siamese AlexNet 구조 + nearest neighbor 비교 (상)
- p.23: VOC-2007 결과 표 (중)
- p.24: inpainting 입력/출력 예시 (중)

## EXAM POINTS

1. **Pretext task vs downstream task 정의 및 차이**: pretext는 라벨 없이 데이터 자체로 만든 surrogate task(representation 학습 목적), downstream은 실제 관심 task(소량 labeled data).
2. **SSL 평가 프로토콜**: pretext로 encoder 학습 → encoder 위에 shallow network 부착 → 소량 labeled data로 downstream 학습 → 성능으로 평가 (linear probing 개념). 왜 shallow인가?
3. **Pretext task가 좋은 feature를 학습하는 이유 2가지**: (1) 풀려면 semantic 이해 필요, (2) 라벨 자동 생성으로 scale 가능.
4. **Relative position of patches (Doersch 2015)**: 8-way classification 정의, 입력 쌍 구조, siamese/shared-weight 구조, 왜 두 patch를 따로 embedding하는지.
5. **Pretext 설계 시 trivial shortcut 문제**: patch 경계 연속성, chromatic aberration 등과 이를 막는 방법 (gap/jitter, color dropping/projection).
6. **VOC-2007 결과 해석**: scratch 대비 +5%p, ImageNet supervised 대비 -8%p → SSL의 가능성과 한계.
7. **Inpainting (Context Encoders, Pathak 2016)**: 목적, 학습되는 지식(object parts, scene structure, semantic context), classification형 pretext와 generative형 pretext 비교.
8. **Discriminative(분류형) vs generative(복원형) pretext task** 분류 및 각 예시.

## VERIFIER REPORT

- p.17 설명: colorization을 regression으로 일반화한 부분에 Zhang et al.은 classification 정식화도 쓴다는 단서 추가.
- p.18 설명: 슬라이드가 encoder freeze 여부를 명시하지 않음을 표기 (linear probing은 해석임).
- p.22 설명: "ImageNet 수준에 근접한 retrieval" 과장 완화 (정성적 비교만 제시).
- p.23 설명: scratch 대비 향상폭을 기준별 수치로 정확히 표기 (+5.0~6.5%p).
- p.24 그림 설명: "파리풍" 추정 제거, 실제 이미지(상점가, 분홍/붉은 색감, 흐릿한 창문 구조)에 맞게 수정.
- p.23 표 전체 수치 및 굵은 글씨 위치, p.22 아키텍처 표기, 인용(Gidaris 2018, Noroozi 2016, Zhang 2016, Doersch ICCV 2015, Pathak CVPR 2016) 대조 결과 오류 없음.
