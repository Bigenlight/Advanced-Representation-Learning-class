<!-- DRAFT | lecture: 01_Introduction | chunk: 01 | pages 1–8 -->
<!-- VERIFIED | fixes: 2 | verifier: sonnet -->

## Advanced Representation Learning (고급표현학습) 표지 (p.1)

- 제목: **Advanced Representation Learning (고급표현학습)**
- Instructor: Donghyeon Cho
- The Department of Computer Science
- Hanyang University
- 그림/로고: 배경에 한양대 사자 문양 워터마크, 좌하단 한양대학교(HANYANG UNIVERSITY) 로고, 우하단 한양대 인장(1939)과 연구실 로고(CV 추정). 파란 가로 라인 3줄로 영역 구분.

**설명**
- 강의 전체 표지 슬라이드라 정보는 과목명·담당 교수·소속뿐임.
- 과목명의 핵심어 **Representation Learning**은 raw data(이미지, 영상, 텍스트 등)를 downstream task에 유용한 **feature(표현)** 로 바꾸는 방법을 학습하는 분야를 뜻함. "Advanced"가 붙은 것은 기본 supervised deep learning 이후의 주제(SSL, contrastive, semi-/weakly-supervised, active learning, MAE 등)를 다루기 때문임(p.3, p.4 참고).

## Course Information (p.2)

- Instructor
  - Prof. Donghyeon Cho
  - Office: IT.BT rm 504
  - Tel: 02-2220-2657
  - Web: cvlab.hanyang.ac.kr
  - Email: doncho@hanyang.ac.kr; cdh12242@gmail.com
  - Research area: computer vision, visual intelligence
- Smart-F (Flip Learning)
  - Online + offline (1.5h)
  - Online: basic concept
  - Offline: review + extra
    - You can do paper presentations if you want~!

**설명**
- **Smart-F (Flip Learning)** 방식: 기본 개념은 온라인 강의로 미리 학습하고, 오프라인(대면)은 복습 + 추가 내용(extra)으로 진행하는 flipped classroom 구조임. 슬라이드에는 "Online + offline (1.5h)"로만 적혀 있어 1.5h가 오프라인 시간인지 전체 시간인지는 명시되지 않음.
- 오프라인 시간에 원하면 **논문 발표(paper presentation)** 를 할 수 있음. p.5의 Participation(30%)과 연결될 가능성이 있음 ← 슬라이드에 명시는 없고 추정임.
- 교수 연구 분야가 computer vision / visual intelligence라서 강의 예시도 이미지·비디오 중심(p.6, p.8의 모델 대부분이 vision 모델)임.

> 보충: flipped learning은 "수업 전 개념 학습(영상) → 수업 중 토론·문제풀이"로 순서를 뒤집는 교육 방식임. 일반 강의는 수업 중 개념 설명, 수업 후 과제.

## Prerequisite subjects (p.3)

- Deep learning
  - Not for beginner's course
  - Dealing with "(slightly) Advanced" topics!
  - Course for students majoring in AI
- No textbook!

**설명**
- **딥러닝 기초(CNN, backprop, optimizer 등)를 이미 안다는 전제**로 진행하며 입문 과정이 아님. "(slightly) Advanced"라고 완화했지만 AI 전공 대상임.
- **교재 없음** → 슬라이드와 인용되는 논문이 사실상 교재. 시험 범위는 슬라이드 + 논문 핵심 아이디어로 보는 게 안전함.
- 이 슬라이드에는 페이지 번호가 표시되지 않았으나 PDF 3쪽임.

## Time Table (p.4)

표 (열: Date / Lecture / Details)

| Date | Lecture | Details |
|---|---|---|
| Week 1 (9/3(목)) | Introduction | |
| Week 2 (9/10(목)) | No class (휴강) | ECCV Periods |
| Week 3 (9/17(목)) | Self-Supervised Learning | |
| Week 4 (9/24(목)) | Contrastive Learning I | Chuseok Periods (online) |
| Week 5 (10/1(목)) | Contrastive Learning II | |
| Week 6 (10/8(목)) | Contrastive Learning III | |
| Week 7 (10/15(목)) | Semi-Supervised Learning I | |
| Week 8 (10/22(목)) | Semi-Supervised Learning II | |
| Week 9 (10/29(목)) | Midterm Quiz | |
| Week 10 (11/5(목)) | Weakly-Supervised Learning I | |
| Week 11 (11/12(목)) | Weakly-Supervised Learning II | |
| Week 12 (11/19(목)) | Active Learning I | |
| Week 13 (11/26(목)) | Active Learning II | |
| Week 14 (12/3(목)) | Masked Autoencoder (MAE) | |
| Week 15 (12/10(목)) | Final  Quiz | |
| Week 16 (12/17(목)) | VideoMAE+ | |

**설명**
- 매주 목요일 수업, 총 16주. 파란 헤더 표, 행은 두 가지 연한 색이 번갈아 나타남.
- 흐름: **SSL 개요(3주차) → Contrastive Learning 3주(4~6주차) → Semi-Supervised 2주(7~8주차) → 중간고사(9주차) → Weakly-Supervised 2주 → Active Learning 2주 → MAE(14주차) → 기말고사(15주차) → VideoMAE+(16주차)**.
- 중간·기말 모두 "Quiz"라는 명칭이며(p.5 Grading), 기말(15주차) 이후 16주차에 VideoMAE+ 수업이 있음 → 16주차 내용은 기말 범위 밖일 가능성 ← 슬라이드에 명시 없음, 확인 필요.
- 2주차는 ECCV 기간이라 휴강, 4주차는 추석 기간이라 온라인 진행임.
- 학습 패러다임을 **label 의존도가 낮은 순서(SSL: label 없음 → Semi: 일부 label → Weakly: 약한 label → Active: label을 선택적으로 요청)** 로 훑는 구성으로 해석 가능함(필기자 해석).

> 보충: 슬라이드에 연도는 없음. 9/3이 목요일인 해는 2026년과 일치함.

## Grading (p.5)

- Attendance: 10%
- Participation: 30%
- Mid Quiz: 30%
- Final Quiz: 30%

**설명**
- 합계 10 + 30 + 30 + 30 = **100%**.
- **Participation 비중이 30%로 큼** → 수업 중 참여/논문 발표(p.2) 등이 크게 반영될 수 있음. 시험(Quiz) 비중은 합쳐서 60%.
- 출석은 10%로 상대적으로 낮음.

## Introduction (p.6)

슬라이드 제목: Introduction. 6칸(2행×3열) 인포그래픽과 하단 배너.

- (1행 1열) **Feature Learning**: 산 풍경 이미지 → 점점 작아지는 CNN 블록(파란 3D 직육면체 4개) → 파란 사각형 세로 벡터(feature). 캡션 "Learn useful features".
- (1행 2열) **Self-Supervised Learning**: 산 풍경 이미지(위)와 색/밝기가 변형된 같은 이미지(아래)가 양방향 곡선 화살표로 연결되고 가운데 아래 화살표 → 초록 사각형 세로 벡터(feature). 캡션 "Learn from unlabeled data".
- (1행 3열) **Foundation Models**: 지구본 + 네트워크 그래프가 중앙에 있고 주변에 이미지, 텍스트(T), 오디오 파형, 격자, "..." 아이콘 배치. 캡션 "Generalize to many tasks".
- (2행 1열) **Video Representation Learning**: 달리는 사람 필름 스트립(프레임들) → 파란 사각형 시퀀스. 캡션 "Capture temporal dynamics".
- (2행 2열) **Multimodal Representation Learning**: 이미지 / 텍스트("A dog running in the park") / 오디오 파형 → 각각 파랑·초록·보라 토큰 시퀀스. 캡션 "Align across modalities".
- (2행 3열) **Applications**: Classification, Detection, Segmentation, Generation, "..." (아이콘 포함).
- 하단 배너(전구 아이콘): **"From understanding data to understanding the world."**

**설명**
- 강의 전체 로드맵을 한 장에 요약한 그림. 읽는 순서는 **Feature Learning → Self-Supervised Learning → Foundation Models → Video → Multimodal → Applications**.
- **Feature Learning**: 입력 $x$를 encoder $f_\theta$에 통과시켜 $z=f_\theta(x)$ 형태의 **유용한 feature vector**를 얻는 것이 목표. 사람이 feature를 설계(hand-crafted)하지 않고 데이터로부터 학습함.
- **Self-Supervised Learning (SSL)**: label 없이 데이터 자체에서 supervision signal을 만들어 학습함. 그림에서는 같은 이미지의 서로 다른 변형(augmented view)을 화살표로 이어, "같은 이미지의 두 view는 비슷한 표현을 가져야 한다"는 contrastive/invariance 계열 아이디어를 시각화함(p.8 SimCLR과 연결).
- **Foundation Models**: 대규모 데이터로 한 번 사전학습해 **여러 downstream task로 generalize**하는 모델. 여러 modality를 다룸.
- **Video Representation Learning**: 정지 이미지와 달리 **temporal dynamics(시간적 변화)** 를 담는 표현이 필요함. 16주차 VideoMAE+, p.8의 V-JEPA와 연결.
- **Multimodal Representation Learning**: 서로 다른 modality(이미지/텍스트/오디오)를 **공통 embedding 공간에서 align** 하는 것이 핵심(p.8 CLIP).
- **Applications**: 좋은 표현 하나로 classification, detection, segmentation, generation 등을 수행함.
- 배너 문구는 "개별 데이터를 이해하는 단계에서 세계를 이해하는 단계로"라는 분야의 방향성을 말함. p.7의 5단계 진화와 이어짐.

> 보충: "Foundation model"은 Bommasani et al., 2021 ("On the Opportunities and Risks of Foundation Models")에서 대중화된 용어로, 대규모 데이터로 학습되어 다양한 downstream task에 adapt 가능한 모델을 가리킴.

## What AI has Been Learning (p.7)

- bullet: From rules to world knowledge: the journey of learning better representations
- 그림: 5단계가 오른쪽 화살표(→)로 연결된 가로 흐름. 각 단계는 위쪽 아이콘 박스 + 캡션 + 아래 Example 박스로 구성.

| 단계 | 아이콘 | 캡션 | Example 박스 |
|---|---|---|---|
| 1. Rules | 저울 | Hand-crafted rules and logic | IF raining THEN wet road / IF red light THEN stop / IF speed > 60 THEN ticket / ... |
| 2. Patterns | 2D 축 위의 초록 점들 + 점선 경계 | Learn patterns from data | Discover correlations and structure in data (흩어진 초록 점들) |
| 3. Features | 주황색 큐브 + 회색 필터(Gabor 유사) 패치 격자 | Learn meaningful features | Edges, textures, shapes, parts (필터 패치 이미지) |
| 4. Representations | 보라 노드 그래프 + 색깔별 클러스터 산점도 | Learn useful representations that generalize across tasks | Object-, scene-, and semantic-level embeddings that transfer to many tasks (색깔별 클러스터) |
| 5. World Knowledge | 붉은 지구본 + 이미지/텍스트/오디오/비디오 아이콘 연결 그래프 | Learn rich, general knowledge about the world | Understand objects, relations, events, and concepts across modalities |

**설명**
- AI가 "무엇을 학습 대상으로 삼았는가"의 진화를 5단계로 정리함.
  1. **Rules**: 전문가가 규칙을 직접 작성하는 symbolic / rule-based 시스템. 규칙 밖 상황에 취약하고 확장이 안 됨.
  2. **Patterns**: 데이터에서 통계적 상관·구조를 학습하는 전통 machine learning. 단 입력 feature는 여전히 사람이 설계했음.
  3. **Features**: 모델이 edge·texture·shape·part 같은 **의미 있는 feature를 학습**하는 단계(CNN 초기 층 필터가 Gabor 유사 edge detector를 학습하는 것이 대표적).
  4. **Representations**: feature가 한 task에 묶이지 않고 **여러 task에 transfer 되는 object/scene/semantic 수준의 embedding**으로 발전. 이 강의의 중심 단계임.
  5. **World Knowledge**: 여러 modality에 걸쳐 객체·관계·사건·개념을 이해하는 범용 지식(foundation model, multimodal).
- 핵심 메시지: 발전의 축은 "**사람이 넣어주는 지식(규칙, feature 설계)이 줄고, 데이터에서 배우는 표현의 추상 수준과 일반성이 커지는 것**"임.
- 3단계와 4단계의 차이: 3은 저수준(edge, texture)~중간 수준(parts)의 feature, 4는 task 간 전이가 가능한 고수준 semantic embedding.

> 보충: 1단계는 symbolic AI, 2단계는 classical ML(SVM, 로지스틱 회귀 등), 3단계는 deep CNN의 계층적 feature(Zeiler & Fergus, 2014 시각화 등)에 대응시켜 볼 수 있음.

## Representative Models (p.8)

슬라이드 제목: Representative Models. 6칸(2행×3열) 모델 아키텍처 다이어그램.

- (1행 1열) **AlexNet**: 거리 이미지 입력 → conv 블록들(레이블: 11×11 s=4, 5×5 s=1, 3×3 s=1, max pooling) → FC 4096, 4096 → 1000 출력(보라 사각형).
- (1행 2열) **SimCLR**: 호랑이 이미지 하나에서 두 개의 augmentation → 각각 동일한 encoder $f(\cdot)$ → 파란 사각형 표현 2개 → 점선 양방향 화살표로 "maximize similarity".
- (1행 3열) **CLIP**: 텍스트("A dog running in the park") → text encoder, 강아지 이미지 → image encoder → 두 embedding 사이를 격자 행렬로 표현(일부 칸 초록/보라 색칠, 칸 "..." 표시).
- (2행 1열) **DINOv2**: 다양한 이미지 격자(4×4, 산/꽃/토마토 모양 객체 등) → ViT encoder → 파란 사각형 세로 feature 열.
- (2행 2열) **MAE**: 4열×5행 패치 격자에서 대부분 회색(masked)이고 일부 패치만 보임 → encoder (ViT) → 보이는 패치 토큰(파랑) + mask 토큰(주황)이 섞인 열 → decoder (ViT) → 복원된 패치 이미지들.
- (2행 3열) **V-JEPA**: 오리 영상 프레임 시퀀스 → 일부는 context encoder, 일부는 target encoder → 파란/보라 토큰들 사이 점선 화살표(context 표현에서 target 표현을 예측).

**설명**
- p.7의 진화를 대표 모델로 짚은 슬라이드이며, 각 모델은 이후 강의에서 다룸.
- **AlexNet** (Krizhevsky, Sutskever, Hinton, 2012): ImageNet에서 deep CNN이 hand-crafted feature를 압도한 사건. **supervised learning**(1000-class label)으로 feature를 학습하는 출발점. 11×11 stride 4 첫 conv, 4096-d FC 두 개가 특징.
- **SimCLR** (Chen et al., 2020): 한 이미지에서 만든 두 augmented view의 표현을 **maximize similarity**(positive pair), 다른 이미지의 표현과는 멀어지게 하는 **contrastive learning**. label 불필요 → 4~6주차 주제.
- **CLIP** (Radford et al., 2021): 이미지-텍스트 쌍으로 두 encoder를 학습하여 같은 쌍은 가깝게, 다른 쌍은 멀게(배치 내 $N\times N$ similarity 행렬의 대각선이 positive) 만드는 **multimodal contrastive learning**. 그림의 격자 = 유사도 행렬로 해석됨.
- **DINOv2** (Oquab et al., 2023): 대규모 curated 데이터로 **self-distillation 기반 SSL**로 ViT를 학습, fine-tuning 없이 여러 task에 쓰이는 **범용 visual feature** 제공(foundation model).
- **MAE (Masked Autoencoder)** (He et al., 2022): 패치의 큰 비율(원논문 75%)을 mask하고 **보이는 패치만 encoder**에 넣은 뒤 **decoder가 mask된 패치를 복원**하도록 학습하는 **reconstruction 방식 SSL**. 14주차 주제.
- **V-JEPA** (Bardes et al., 2024): 영상에서 context 영역의 표현으로 target 영역의 **표현(feature space)** 을 예측하는 JEPA(Joint-Embedding Predictive Architecture) 방식. pixel 복원이 아니라 latent 예측인 점이 MAE와의 차이.
- 모델 계열 정리(필기자 정리): **supervised (AlexNet) → contrastive (SimCLR) → vision-language contrastive (CLIP) → self-distillation (DINOv2) → masked reconstruction (MAE) → latent prediction for video (V-JEPA)**.

> 보충: 슬라이드에는 모델 이름과 그림만 있고 저자·연도는 없음. 위 인용 정보와 모델 설명은 일반 지식으로 보충한 것이며 강의에서 다루는 방식과 다를 수 있음.

## FIGURES

- p.4: Time Table 전체 표(16주 강의 일정) — 중
- p.6: 강의 로드맵 인포그래픽(Feature / SSL / Foundation / Video / Multimodal / Applications) — 중
- p.7: What AI has Been Learning 5단계 진화 도식(Rules → Patterns → Features → Representations → World Knowledge) — 상
- p.8: Representative Models 6종 아키텍처 비교(AlexNet, SimCLR, CLIP, DINOv2, MAE, V-JEPA) — 상

## EXAM POINTS

1. **AI 학습 대상의 5단계 진화**: Rules → Patterns → Features → Representations → World Knowledge 순서와 각 단계의 특징·예시 서술.
2. 3단계 Features와 4단계 Representations의 차이: **저·중수준 feature vs. 여러 task로 transfer되는 semantic embedding**.
3. **Representation Learning의 정의**와 목적: raw data를 downstream task에 유용한 feature로 변환(Learn useful features).
4. **Self-Supervised Learning**의 정의: label 없는(unlabeled) 데이터로부터 학습, 같은 이미지의 augmented view 사이 관계 활용.
5. Representative Models를 **학습 방식으로 분류**: supervised(AlexNet), contrastive(SimCLR, CLIP), self-distillation(DINOv2), masked reconstruction(MAE), latent prediction(V-JEPA).
6. **SimCLR vs. CLIP**: 둘 다 contrastive지만 positive pair가 (augmented view, augmented view) vs. (image, text) — modality 차이.
7. **MAE vs. V-JEPA**: pixel 복원(decoder, reconstruction) vs. feature space 예측(context encoder → target encoder).
8. 강의 구성 순서와 성적 비율: SSL → Contrastive → Semi → (Midterm) → Weakly → Active → MAE → (Final) → VideoMAE+; Attendance 10 / Participation 30 / Mid 30 / Final 30.

## VERIFIER REPORT

- p.8 MAE 그림 설명: 패치 격자를 5×5로 적은 것을 4열×5행으로 수정.
- p.2 설명: "오프라인 1.5시간"으로 단정한 부분을 슬라이드 원문("Online + offline (1.5h)") 기준으로 수정하고 시간 귀속이 불명확함을 명시.
- 그 외 페이지 1–8의 bullet/표/수식/인용(AlexNet 2012, SimCLR 2020, CLIP 2021, DINOv2 2023, MAE 2022, V-JEPA 2024, Bommasani 2021, Zeiler & Fergus 2014)은 원본과 일치 확인.
