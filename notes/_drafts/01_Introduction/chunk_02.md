<!-- DRAFT | lecture: 01_Introduction | chunk: 02 | pages 9–16 -->
<!-- VERIFIED | fixes: 3 | verifier: sonnet -->

## What is a Representation? (p.9)

- 제목 아래 bullet: "How Does a Computer Understand an Image?"
- 그림: 풀밭에 엎드려 혀를 내민 골든 리트리버 사진 1장 (배경은 흐릿한 나무/초록). 이후 슬라이드들에서 계속 쓰이는 running example.

**설명**
- 섹션 도입 슬라이드. 질문을 던지는 역할: 사람은 사진을 보면 "개"라고 바로 인식하지만 컴퓨터는 어떻게 이해하는가?
- 이 질문의 답이 곧 **representation(표현)** 개념으로 이어짐. 다음 슬라이드에서 컴퓨터에게 이미지는 숫자 행렬일 뿐이라는 점을 보여줌.

## Raw Pixels Are Not Understanding (p.10)

- bullet: "To a computer, an image is initially just a matrix of numbers."
- 그림: 왼쪽에 개 사진, 코/이마 부근에 빨간 반투명 정사각형 영역 표시. 빨간 점선 두 줄이 이 영역의 모서리에서 오른쪽 격자(표)로 뻗어 "확대"를 표현함.
- 오른쪽 표 (10행 × 7열, 픽셀 값):

| 100 | 100 | 100 | 128 | 128 | 128 | 130 |
|---|---|---|---|---|---|---|
| 100 | 100 | 100 | 128 | 128 | 128 | 130 |
| 100 | 100 | 100 | 128 | 128 | 128 | 130 |
| 100 | 100 | 100 | 128 | 128 | 128 | 130 |
| 100 | 100 | 100 | 128 | 128 | 128 | 130 |
| 100 | 100 | 100 | 128 | 128 | 128 | 130 |
| 100 | 100 | 100 | 128 | 128 | 128 | 130 |
| 100 | 100 | 100 | 128 | 128 | 128 | 130 |
| 100 | 100 | 100 | 128 | 128 | 128 | 150 |
| 100 | 100 | 100 | 128 | 128 | 128 | 150 |

**설명**
- 디지털 이미지는 $H \times W \times C$ 크기의 정수 배열 ($x \in \{0,\dots,255\}^{H\times W\times C}$)임. 슬라이드는 일부 패치만 확대해서 보여줌 (grayscale 값처럼 표시됨).
- 핵심 메시지: **raw pixel 값 자체는 "understanding"이 아님**. 픽셀 하나하나에는 의미(semantics)가 없고, 같은 개라도 조명, 위치, 각도가 조금만 바뀌면 픽셀 값이 완전히 달라짐.
- 즉 pixel space에서의 거리(예: L2)는 의미적 유사성과 잘 대응되지 않음. 이것이 representation이 필요한 이유임.
- 표의 값은 열 단위로 거의 동일(100/128/130)한 단순 예시 값이며, 실제 개 얼굴 패치의 값은 아님 (설명용 toy 데이터로 보임).

> 보충: 예를 들어 이미지를 1픽셀만 오른쪽으로 shift 해도 pixel vector의 L2 거리는 크게 변할 수 있지만 의미는 동일함. 이는 pixel space가 translation 등 nuisance factor에 불변하지 않기 때문임.

## What Information Should be Kept? (p.11)

- 그림 (왼쪽 → 오른쪽 3개 박스, 화살표로 연결):
  - **Input Image**: 개 사진.
  - (파란 화살표) →
  - **Representation (keep the useful information)**: 4개 항목 카드
    - **Fur** — Golden, long, and fluffy
    - **Ears** — Droopy and furry
    - **Tail** — Long and furry
    - **Animal shape** — Quadruped with a dog-like body
  - (초록 화살표) →
  - **Downstream Task / Output**: "Dog" (개 아이콘)
- 아래쪽 회색 박스 **Irrelevant / Discarded Information**: Background(산 풍경), Lighting(해), Noise(점박이 원), Camera angle(카메라), People(사람 실루엣), "..."
  - 점선 화살표: Input Image 쪽에서 회색 박스로 내려가고, Representation 쪽에서 회색 박스 방향으로(버려지는 정보), 회색 박스에서 Downstream Task 쪽 위로 점선 화살표가 있음 (이 정보들은 최종 출력에 영향을 주지 않아야 함을 표현).
- 하단 강조 박스 (별 아이콘): "**A good representation keeps useful information and discards irrelevant information.**" (keeps useful information은 초록, discards irrelevant information은 빨강으로 강조)

**설명**
- 좋은 representation의 정의를 직관적으로 설명한 슬라이드: 과제(여기선 "개 분류")에 **유용한 정보는 유지**하고 **무관한 정보는 버림**.
- 유지되는 정보(fur, ears, tail, shape)는 개를 다른 동물과 구별해 주는 semantic/shape 속성임. 버려지는 정보(background, lighting, noise, camera angle, people)는 같은 개라도 촬영 조건에 따라 달라지는 **nuisance factor**임.
- 이는 **invariance**의 개념과 연결됨: 좋은 representation은 nuisance 변화에 **invariant**하고, task 관련 변화에는 **sensitive**해야 함.
- 무엇이 "유용"한지는 task에 따라 달라짐 (예: 배경 분류 task라면 background가 핵심 정보). 따라서 representation의 좋고 나쁨은 downstream task에 상대적임. 이 점이 이후 슬라이드(task-specific feature 문제)로 이어짐.

> 보충: Information Bottleneck 관점(Tishby et al.)에서는 좋은 표현 $Z$가 입력 $X$에 대한 정보는 최소화하면서($I(X;Z)$ 작게) 타깃 $Y$에 대한 정보는 최대화($I(Z;Y)$ 크게)하는 것으로 형식화함. 슬라이드에는 없는 내용임.

## Representation (p.12)

- bullet: "Representation learning aims to map data points to a simpler, organized space."
- 그림 (상단): 
  - 왼쪽 불규칙한 구불구불한 회색 영역 = **Data space** ($\mathcal{X}$). 그 안의 점 $\mathbf{x}$.
  - 곡선 화살표가 사다리꼴 모양 박스 **$f$ (Encoder)** 를 지나 오른쪽의 파란 원 = **Representation space** ($\mathcal{Z}$) 안의 점 $\mathbf{z}$ 로 이어짐.
  - 우측 박스 **Mapping (Encoding)**:
    $$\mathbf{z} = f(\mathbf{x})$$
    $\mathbf{x} \in \mathcal{X}$ (data space), $\mathbf{z} \in \mathcal{Z}$ (representation space)
- 그림 (하단 파이프라인): Input data (개 사진, $\mathbf{x}\in\mathcal{X}$) → Encoder $f$ → Representation (열벡터 $[z_1, z_2, \vdots, z_D]^\top$, $\mathbf{z} = f(\mathbf{x}) \in \mathcal{Z}$) → Downstream task (박스: Classification, Detection, Retrieval, ...)
- 우측 체크 리스트:
  - Learns useful structure
  - Captures semantics
  - Enables many tasks
  - Often in lower dimension ($D \ll \dim(\mathbf{x})$)
- 하단 강조 박스 (별): "**A representation is a transformation of raw data into a form that makes downstream tasks easier.**"

**설명**
- **Representation learning**의 형식적 정의: 데이터 공간 $\mathcal{X}$ 의 점 $\mathbf{x}$ 를 **encoder** $f: \mathcal{X} \to \mathcal{Z}$ 로 변환해 representation space $\mathcal{Z}$ 의 점 $\mathbf{z}=f(\mathbf{x})$ 로 보내는 것.
  - $\mathbf{x}$: 원본 데이터(이미지 픽셀 등). $f$: 학습 가능한 매핑(신경망 등). $\mathbf{z}$: $D$차원 representation vector (feature, embedding, latent code라고도 부름).
- 그림에서 data space는 복잡하고 불규칙한 모양(manifold처럼 구불구불), representation space는 단순한 원(둥근 영역)으로 그려짐 → "**simpler, organized space**"의 시각화. 즉 데이터 공간에서는 의미적으로 비슷한 점들이 복잡하게 흩어져 있지만, 표현 공간에서는 의미 구조가 정돈되어(비슷한 것끼리 가깝게) 있어 downstream task를 더 쉽게(예: 단순한 모델로) 풀 수 있도록 하는 것이 목표임 (슬라이드는 "makes downstream tasks easier"라고만 표현).
- 같은 $\mathbf{z}$ 를 classification, detection, retrieval 등 여러 downstream task에 재사용 가능 → "Enables many tasks".
- 보통 $D \ll \dim(\mathbf{x})$ (예: 224×224×3 = 150,528 차원 → 2048 차원)로 **차원 축소**가 일어남. "Often"이라고 했듯 항상은 아님 (overcomplete representation, 예: sparse coding 등은 입력보다 차원이 클 수도 있음).
- 이전 슬라이드와 연결: p.11의 "유용한 정보만 유지"를 수학적으로 표현한 것이 encoder $f$ 의 역할임.

## Many Possible Representations (p.13)

- bullet: "The same data can have many different representations."
- 그림: 상단 Input Image (개 사진) 하나에서 세 갈래 화살표가 세 개 박스로 이어짐.
  1. **Color Histogram** — "Captures global color distribution". 막대 그래프: x축 Hue (0°, 60°, 120°, 180°, 240°, 300°, 360°), y축 Frequency, 색상별(빨강, 주황, 노랑, 초록, 청록, 파랑, 보라, 분홍 등) 막대. 옆에 모자이크처럼 픽셀화된(저해상도 색상 블록) 이미지.
  2. **CNN Feature** — "Captures hierarchical visual features". 여러 feature map 이미지(레이어가 깊어질수록 점점 blob 형태)와 "..." 아래에 "High-dimensional vector" 표시: [0.21, −0.17, 0.63, 1.24, −0.91, ..., 0.31], 캡션 "(e.g., 2048-D)".
  3. **CLIP Embedding** — "Aligns image with text semantics". 왼쪽에 텍스트 프롬프트 목록(A dog, A golden retriever, A cat, A car, A tree, ...)과 오른쪽에 similarity 히트맵(행: 프롬프트, 진한 보라 = High, 연한 = Low, 대각선이 진함). 캡션 "Similarity to text prompts".
- 하단 강조 박스 (별): "The same data can have **many different representations**." (many different representations 파란색 굵게)

**설명**
- 같은 이미지라도 어떤 변환을 쓰느냐에 따라 전혀 다른 representation이 나옴. 각각 담는 정보가 다름:
  - **Color histogram**: 전역 색상 분포만 담음. 공간 구조(어디에 무엇이 있는지)와 형태 정보는 버림 → 색이 중요한 task에는 유용하지만 개/고양이 구분에는 부적합.
  - **CNN feature**: 계층적(low-level edge → high-level part/object) 시각 특징을 담은 고차원 벡터(예: ResNet-50의 2048-D pooled feature). 분류 등 비전 task에 널리 쓰임.
  - **CLIP embedding**: 이미지와 텍스트를 같은 임베딩 공간에 정렬(align)해서 텍스트 프롬프트와의 유사도로 zero-shot 인식 가능. 히트맵은 이미지 vs 각 텍스트 프롬프트의 유사도(대각선처럼 맞는 쌍이 높음)를 보여줌.
- 따라서 **"좋은 representation"은 절대적 개념이 아니라 task/목적에 의존**. 어떤 정보를 담고 어떤 정보를 버릴지(p.11)가 표현마다 다름.
- 앞으로 이 강의에서 다룰 representation learning 방법(supervised, self-supervised, contrastive, generative 등)이 결국 "어떤 representation을 어떻게 얻을 것인가"의 서로 다른 답임.

> 보충: CLIP = Contrastive Language-Image Pre-training (Radford et al., 2021, OpenAI). 4억 개 이미지-텍스트 쌍으로 image encoder와 text encoder를 contrastive loss로 공동 학습함. 슬라이드에 논문 인용은 없음.

## Before Deep Learning (p.14)

- bullet: "Who design the features?"
- 그림 (좌측 파이프라인): Image (개 사진) ↓ **Hand-crafted Features** 박스 — 세 개의 서브 박스: **SIFT** (scale-space pyramid와 keypoint, descriptor 격자 그림), **HOG** (격자 셀마다 화살표로 gradient 방향 표시), **SURF** (박스 필터/Hessian 근사와 격자 그림) ↓ **Classifier** (e.g., SVM, Random Forest, k-NN) — 파란 원과 빨간 사각형 점들이 점선 결정경계로 분리된 scatter plot.
- 그림 (우측 상단): **SIFT Keypoints** — 흑백 개 사진 위에 노란 원들(여러 스케일의 keypoint)이 빽빽하게 표시됨. 확대 박스: "Keypoint + orientation + descriptor" (원 안에 방향 화살표).
- 그림 (우측 하단): **HOG (Histogram of Oriented Gradients)** — 검은 배경에 흰색 짧은 선분(셀별 gradient 방향 막대)로 개의 윤곽이 드러난 시각화. 확대 박스: 8방향 화살표 (Gradient orientation histogram).
- 하단 강조 박스 (사람 아이콘): "**Feature Engineering Era:** Humans manually designed features. *Different tasks require different features.*"

**설명**
- 딥러닝 이전(~2012)에는 **feature engineering**: 전문가가 "이런 특징이 유용할 것"이라고 직접 설계한 변환(hand-crafted feature)을 이미지에 적용하고, 그 위에 SVM 같은 얕은(shallow) 분류기를 학습시키는 2단계 파이프라인이 표준이었음. 질문 "Who design the features?" 의 답은 **사람**.
- 각 feature 설명:
  - **SIFT** (Scale-Invariant Feature Transform): 스케일 공간에서 keypoint를 검출하고, 각 keypoint 주변의 gradient 방향 히스토그램으로 orientation과 descriptor를 만듦 → 스케일/회전 불변.
  - **HOG** (Histogram of Oriented Gradients): 이미지를 격자 셀로 나눠 셀별 gradient 방향 히스토그램을 계산 → 윤곽/형태 정보 포착 (보행자 검출 등).
  - **SURF** (Speeded-Up Robust Features): SIFT를 근사하여 integral image와 box filter로 가속한 버전.
- 중요한 점: feature 추출과 classifier 학습이 **분리**되어 있고, feature 부분은 데이터로부터 학습되지 않음(사람의 직관 기반). 
- 마지막 문구 "Different tasks require different features"가 다음 슬라이드의 문제 제기로 연결됨.

> 보충: SIFT는 Lowe (2004), HOG는 Dalal & Triggs (2005), SURF는 Bay et al. (2006). 슬라이드에는 연도/저자가 명시되어 있지 않음.

## Why is This Problematic? (p.15)

- 슬라이드 하단 페이지 번호가 "5"로 표기됨 ← 원본 슬라이드 번호 오기(실제는 15페이지), 내용과는 무관.
- 상단 3개 카드:
  1. **Human-designed** (사람+톱니 아이콘)
     - Features are crafted by experts
     - Require domain knowledge
     - Time-consuming and labor-intensive
  2. **Task-specific** (과녁 아이콘)
     - Features optimized for one task
     - Do not generalize well to others
     - New task = new feature design
  3. **Limited scalability** (상승 그래프 아이콘)
     - Does not scale to complex data
     - Hard to cover all variations
     - Limited performance ceiling
- 중단 그림: 세 이미지 그룹이 "≠" 기호로 구분됨.
  - **Object Recognition**: 개, 빨간 자동차, 비행기, 의자, 자전거, 컵 사진 (6장). 캡션: "Requires specific features for objects and shapes"
  - **Face Recognition**: 다양한 인종/나이의 얼굴 사진 6장. 캡션: "Requires specific features for facial patterns"
  - **Medical Imaging**: X-ray, 뇌 MRI, 망막 이미지, CT, 초음파, 조직 병리 이미지 6장. 캡션: "Requires specific features for medical patterns"
- 하단 경고 박스 (빨간 느낌표 아이콘): "**Every new task requires new features.** *This limits generalization, efficiency, and performance.*"

**설명**
- hand-crafted feature 방식의 문제 3가지를 정리한 슬라이드:
  1. **Human-designed**: 전문가의 domain knowledge와 시행착오에 의존 → 비용이 크고 사람의 편향이 들어감.
  2. **Task-specific**: 한 task에 맞춘 feature는 다른 task에 일반화되지 않음 → task가 바뀌면 feature를 처음부터 재설계해야 함.
  3. **Limited scalability**: 데이터의 모든 변이(variation)를 사람이 일일이 포착할 수 없어 복잡한 데이터에서 성능 상한이 존재함.
- 아래 그림은 task별로 필요한 feature가 다름을 예시: 물체 인식(objects and shapes), 얼굴 인식(facial patterns, 예: 눈·코·입 배치), 의료 영상(medical patterns, 예: 병변 패턴)은 서로 다른 feature가 필요하므로 "≠".
- 결론: 매 task마다 새 feature를 설계해야 하므로 일반화, 효율성, 성능이 모두 제한됨 → "그렇다면 feature 자체를 데이터로부터 **학습**하면 되지 않을까?"라는 다음 슬라이드의 동기가 됨.

## Deep Learning Changes the Game (p.16)

- 3열 구성의 정보 슬라이드.
- **1. From Hand-Crafted to Learned Representations** (좌측 열)
  - 파이프라인 (위→아래): Image (개 사진) → **CNN** (점점 작아지는 파란 3D 블록들, "...") → **Representation** (노란 사각형 칸들의 벡터, "...") → **Classifier** (scatter plot + 막대 그래프(확률 분포처럼), "...")
  - 질문 박스: "**What if the machine could learn its own features?**"
  - 비교 박스: **Feature engineering** (사람 → 톱니 아이콘) ⇒ **Feature learning** (모니터 속 뇌/신경망 → 네트워크 그래프 아이콘)
  - 하단 강조 박스 (별): "**Deep learning learns representations automatically from data,** *discovering features that are useful for the task.*"
- **2. What Does CNN Learn?** (중앙 열)
  - 제목: "AlexNet Feature Visualization (Krizhevsky et al., 2012)"
  - **Layer 1 (Edges)**: 3×10 작은 필터/패턴 그리드 — 방향성 있는 edge, Gabor 유사 패턴, 색 blob.
  - **Layer 3 (Textures)**: 3×10 그리드 — 격자/질감 같은 반복 패턴.
  - **Layer 5 (Object Parts)**: 3×10 그리드 — 눈, 바퀴/원형, 입 모양 등 물체 부분 형태.
  - 하단 박스 (뇌 아이콘): "**Representations become increasingly semantic.**" 화살표: Low-level → High-level
- **3. Representation Reuse** (우측 열)
  - 위→아래 흐름: **ImageNet** (이미지 썸네일 모자이크) → **Pretraining** (Large-scale supervision) → **Feature Extractor (Backbone)** → 세 갈래 → **Multiple Tasks**:
    - **Detection** — Find objects (Bounding boxes) — YOLO 로고
    - **Segmentation** — Pixel-level understanding (Masks) — Mask R-CNN 로고
    - **Tracking** — Track objects across frames (IDs) — DeepSORT 로고
  - 하단 강조 박스 (과녁): "**One representation, many applications.** *Learn once, use everywhere.*"

**설명**
- **(1) Hand-crafted → Learned**: 이전에는 feature extractor(사람이 설계)와 classifier가 분리되어 있었다면, 딥러닝에서는 CNN이 raw image로부터 representation을 **end-to-end로 학습**함. 즉 feature engineering이 feature learning으로 대체됨 → p.15의 세 문제(human-designed, task-specific, limited scalability) 해결을 노림.
- **(2) CNN이 무엇을 학습하나**: AlexNet(Krizhevsky et al., 2012)의 층별 시각화를 보면 **계층적(hierarchical) 표현**을 학습함: Layer 1은 edge/색 blob 같은 저수준 특징, Layer 3은 texture(질감), Layer 5는 object parts(눈, 바퀴 등 부분 형태). 깊어질수록 **low-level → high-level, 점점 semantic**해짐. 이는 사람이 설계한 SIFT/HOG와 달리 데이터로부터 자동으로 이런 계층이 나타난다는 점이 핵심임.
- **(3) Representation reuse (transfer)**: ImageNet 같은 대규모 supervised 데이터로 backbone을 pretraining하고, 이 backbone을 feature extractor로 가져다 detection(YOLO), segmentation(Mask R-CNN), tracking(DeepSORT) 등 여러 downstream task에 재사용(fine-tuning/transfer learning)함. hand-crafted 시대의 "새 task = 새 feature" 문제를 해소하는 "learn once, use everywhere"의 의미.
- 다만 이 슬라이드의 pretraining은 **large-scale supervision**(라벨 필요)에 의존함 → 라벨 비용 문제는 남아 있고, 이후 self-supervised representation learning으로 이어지는 동기가 됨 (강의 흐름상 추정).

> 보충: AlexNet = Krizhevsky, Sutskever, Hinton (2012), ImageNet Classification with Deep CNNs. 2012 ILSVRC 우승으로 딥러닝 시대를 연 모델. 층별 시각화 이미지는 AlexNet 자체 논문이라기보다 Zeiler & Fergus (2014)의 "Visualizing and Understanding Convolutional Networks" 스타일의 시각화와 유사함 (슬라이드는 Krizhevsky et al., 2012로 표기). YOLO (Redmon et al., 2016), Mask R-CNN (He et al., 2017), DeepSORT (Wojke et al., 2017)는 슬라이드에 로고만 있고 인용은 없음.

## FIGURES

- p.11: 입력 이미지 → 유지되는 정보(fur, ears, tail, shape) → Dog, 아래에 버려지는 정보(background, lighting, noise, camera angle, people) 다이어그램 (상)
- p.12: Data space $\mathcal{X}$ → Encoder $f$ → Representation space $\mathcal{Z}$ 매핑 도식 + 하단 encoder 파이프라인 (상)
- p.13: 같은 이미지에서 Color histogram / CNN feature / CLIP embedding 세 가지 representation 비교 (중)
- p.14: Hand-crafted feature 파이프라인(SIFT/HOG/SURF → classifier)과 SIFT keypoint, HOG 시각화 (중)
- p.15: hand-crafted feature의 3가지 문제와 task별 이미지 비교(object / face / medical) (중)
- p.16: 3열 종합 (hand-crafted→learned, AlexNet layer 1/3/5 feature 시각화, ImageNet pretraining 후 representation reuse) (상)
- p.10: 픽셀 패치와 숫자 행렬 (중)

## EXAM POINTS

1. **Representation의 정의**: 원본 데이터 $\mathbf{x}\in\mathcal{X}$ 를 encoder $f$ 로 $\mathbf{z}=f(\mathbf{x})\in\mathcal{Z}$ 로 변환해 downstream task를 더 쉽게 만드는 것. 기호 의미와 $D \ll \dim(\mathbf{x})$ 인 이유를 설명할 수 있어야 함.
2. **Why raw pixel은 understanding이 아닌가?**: 이미지는 숫자 행렬일 뿐이며 조명/각도/위치 변화에 픽셀 값이 크게 바뀌고, 픽셀 값 자체에 의미가 없음.
3. **좋은 representation의 조건**: 유용한 정보는 유지하고(fur, ears, tail, shape) 무관한 정보(background, lighting, noise, camera angle)는 버림 (nuisance에 invariant). "유용함"은 task에 의존함.
4. **같은 데이터, 여러 representation 비교**: color histogram(전역 색 분포, 공간 정보 없음) vs CNN feature(계층적 시각 특징, 고차원 벡터) vs CLIP embedding(이미지-텍스트 정렬, zero-shot)이 각각 무엇을 포착하는지.
5. **Hand-crafted feature (SIFT, HOG, SURF)와 feature engineering era**: 파이프라인(feature 추출 → SVM/RF/kNN 분류기)과 각 feature가 무엇을 계산하는지(예: HOG = gradient orientation histogram).
6. **Hand-crafted feature의 문제점 3가지**: human-designed, task-specific, limited scalability. "Every new task requires new features"의 의미.
7. **딥러닝이 해결한 것**: feature engineering → feature learning (end-to-end). AlexNet 층별 표현(Layer 1 edges → Layer 3 textures → Layer 5 object parts)의 계층성과 점점 semantic해지는 이유.
8. **Representation reuse (transfer)**: ImageNet pretraining된 backbone을 detection/segmentation/tracking에 재사용하는 "learn once, use everywhere" 개념과 그 한계(대규모 라벨 의존).

## VERIFIER REPORT

- p.12 설명: "일부 transformer 임베딩은 입력보다 클 수도 있음"은 근거 없는 주장이라 삭제하고 sparse coding 예시만 남김.
- p.12 설명: "선형 분류기 같은 간단한 모델로도 task를 풀 수 있음"은 슬라이드에 없는 과장이라 "downstream task를 더 쉽게" 목표로 표현을 완화.
- p.15 설명: task별 필요 feature를 슬라이드 캡션(objects and shapes / facial patterns / medical patterns)에 맞게 수정 (물체 인식 "형태/질감" -> 슬라이드 표현).
- 그 외 페이지 9–16의 수식, 표 값(10x7, 마지막 두 행 150), 그림 설명, 인용(Krizhevsky 2012, Zeiler & Fergus 2014, Radford 2021, Lowe 2004, Dalal & Triggs 2005, Bay 2006 등)은 원본과 일치.
