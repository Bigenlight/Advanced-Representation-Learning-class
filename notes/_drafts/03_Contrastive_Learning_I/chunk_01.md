<!-- DRAFT | lecture: 03_Contrastive_Learning_I | chunk: 01 | pages 1–8 -->
<!-- VERIFIED | fixes: 5 | verifier: sonnet -->

## Advanced Representation Learning (고급표현학습) — 표지 (p.1)

- Advanced Representation Learning (고급표현학습)
- Instructor: Donghyeon Cho
- The Department of Computer Science
- Hanyang University
- 그림: 한양대 로고(좌하단, "한양대학교 HANYANG UNIVERSITY"), 한양대 인장(1939), 연구실 로고(카메라+V 모양, 우하단)

**설명**
- 강의 03(Contrastive Learning I)의 표지 슬라이드. 내용 없음.

## Contents of Table (p.2)

- 1. Toward Contrastive Learning
- 2. Contrastive Training Objectives
  - Siamese Networks
  - Contrastive Loss
  - Triplet Loss
  - Lifted Structured Loss
  - N-pair Loss
  - NCE
  - InforNCE ← 오타: InfoNCE가 맞음 (원문 유지)
  - Soft-Nearest Neighbors Loss
- 3. Common Setup for Contrastive Learning

**설명**
- 이 강의의 전체 흐름: (1) **pretext task → contrastive learning**으로 넘어오는 동기, (2) 여러 **contrastive objective**를 슬라이드 목록 순서대로 훑기(엄밀한 연대순은 아님. 예: NCE(2010)가 Triplet(2015)·Lifted(2016)보다 먼저 나옴), (3) 공통 학습 setup 정리.
- 목록 순서는 대략 **pair 기반(Contrastive) → triplet(anchor, pos, neg) → 여러 negative를 동시에 쓰는 loss(Lifted, N-pair) → 확률적 분류 관점(NCE, InfoNCE) → soft neighbor 관점(SNN)** 으로 이어지는 대략적인 흐름임(pair/triplet은 negative 1개, Lifted·N-pair·InfoNCE 등은 여러 negative를 동시에 사용, NCE·InfoNCE·SNN은 확률/softmax 형태). 슬라이드 자체는 이 흐름을 명시하지 않은 필자의 해석임.
> 보충: NCE = Noise-Contrastive Estimation (Gutmann & Hyvärinen, 2010), InfoNCE = Contrastive Predictive Coding에서 제안 (van den Oord et al., 2018), Soft-Nearest Neighbors Loss는 Frosst, Papernot & Hinton(2019)이 제안했고, Neighbourhood Components Analysis(Goldberger et al., 2004)에서 영감을 받은 것.

## Toward Contrastive Learning (p.3)

- 섹션 구분 슬라이드: "Toward Contrastive Learning" (남색 배경, 흰 글씨)

**설명**
- 섹션 1 시작. 내용 없음.

## From Pretext Tasks to Contrastive Learning (p.4)

- They are all handcrafted Objectives
- 그림: 고양이 사진 하나를 이용한 4가지 pretext task를 가로로 나열, 각 이미지 아래 캡션
  - **image completion**: 위쪽이 회색 박스(물음표 "?")로 가려지고 아래쪽 고양이 몸통만 보이는 이미지 → 가려진 부분을 예측
  - **rotation prediction**: 고양이 이미지가 기울어져 있고, 파란 수직 화살표와 초록 화살표 사이의 각도 "θ=?" 표시 → 회전 각도 예측
  - **"jigsaw puzzle"**: 고양이 이미지를 4개 조각으로 잘라 섞어 놓음 → 원래 배치(순서) 예측
  - **colorization**: 흑백 고양이 이미지 → 색 예측
- 하단 질문: "Can we learn representations without designing a task?"

**설명**
- 지금까지(이전 강의) 본 self-supervised learning의 **pretext task**들은 모두 사람이 **직접 설계한(handcrafted) objective**임. 라벨 없이 데이터 자체에서 supervision을 만들어내지만, "무엇을 예측시킬지"는 연구자가 정해야 함.
- 문제점: (1) task마다 설계 노하우가 필요하고, (2) 특정 task에 맞춘 shortcut(예: jigsaw에서 조각 경계의 chromatic aberration 단서)을 학습해 **representation이 일반화되지 않을 수 있음**, (3) task가 달라질 때마다 새로 설계해야 함.
- 그래서 던지는 질문: **task를 따로 설계하지 않고도 representation을 학습할 수 있나?** → 다음 슬라이드의 "similarity"라는 더 일반적인 관점으로 연결됨.
> 보충: image completion = Context Encoder/inpainting (Pathak et al., 2016), rotation = RotNet (Gidaris et al., 2018), jigsaw = Noroozi & Favaro (2016), colorization = Zhang et al. (2016). 슬라이드에는 출처 미표기.

## A More General Pretext Task? (p.5)

- Good representations place semantically similar samples nearby.
- 그림: 하단 중앙에 기준 고양이 이미지. 위쪽에 p.4의 4가지 변형(image completion, rotation, jigsaw, colorization) 이미지가 배치되고, 기준 고양이와 각각 **초록 양방향 화살표**로 연결됨 → 라벨 "same object". 오른쪽 아래에는 강아지 이미지가 있고, 기준 고양이와 **빨간 양방향 화살표**로 연결됨 → 라벨 "different object".
- 하단 문구: "Representation Learning = Learning Similarity"

**설명**
- 핵심 관점 전환: pretext task들이 사실은 **"같은 물체(원본 고양이)의 다른 view"** 를 만든 것이고, 이들이 서로 가깝게 임베딩되길 바라는 것임. 반대로 **다른 물체(강아지)** 는 멀어야 함.
- 즉 좋은 representation = **semantically similar한 샘플은 embedding 공간에서 가깝고, 다른 샘플은 멂**. 따라서 representation learning을 **similarity를 학습하는 문제**로 일반화할 수 있음 (**Representation Learning = Learning Similarity**).
- 이전 방법과의 차이: 이전엔 "회전각을 맞혀라" 같은 **특정 예측 task**를 풀다가 부산물로 feature를 얻었다면, 여기서는 **embedding 간의 거리/유사도 관계 자체**를 목표로 삼음.

## Similarity as Supervision (p.6)

- Similarity provides a learning signal.
- 그림: p.5와 동일한 구도. 라벨만 변경됨 — 초록 화살표 쪽 "**attract**", 빨간 화살표 쪽 "**repel**".

**설명**
- "같은 물체 / 다른 물체"라는 **similarity 관계가 곧 supervision(학습 신호)** 임. 클래스 라벨이 없어도 "이 둘은 같은 원본에서 왔다"는 정보만 있으면 됨.
- 같은 물체의 view끼리는 **attract(당김)**, 다른 물체와는 **repel(밀어냄)** 하도록 loss를 설계하면 됨.
- 라벨 없이도 데이터 증강(augmentation)으로 positive pair를 자동 생성할 수 있어 **self-supervised** 학습이 가능해짐.

## Metric Learning (p.7)

- Pull Similar Samples Together
- Push Dissimilar Samples Apart
- 그림: p.5–6과 같은 구도에서 표기만 수식 기호로 대체.
  - 기준 고양이 이미지 = $x$, 4가지 변형 이미지(image completion, rotation, jigsaw, colorization) 각각 = $x^+$, 강아지 이미지 = $x^-$
  - 좌하단 범례 박스: $x$ reference / $x^+$ positive / $x^-$ negative
  - $x$와 각 $x^+$는 초록 양방향 화살표, $x$와 $x^-$는 빨간 양방향 화살표

**설명**
- 용어 정의:
  - $x$ : **reference**(anchor) 샘플
  - $x^+$ : **positive** — $x$와 semantically 비슷한 샘플
  - $x^-$ : **negative** — $x$와 다른 샘플
- 목표: embedding 공간에서 $d(f(x), f(x^+))$ 는 작게, $d(f(x), f(x^-))$ 는 크게 만드는 $f$ 학습 → 이것이 **metric learning(distance metric learning)** 의 정의.
- 이후 나오는 contrastive loss(Contrastive, Triplet, N-pair, InfoNCE ...)는 대체로 이 $(x, x^+, x^-)$ 구조를 어떻게 loss로 쓰느냐의 차이임.
> 보충: metric learning은 원래 supervised(라벨로 positive/negative 정의)로 시작했고, self-supervised에서는 augmentation으로 positive를 정의함.

## Siamese Network (p.8)

- Both images are mapped into the same embedding space.
- 그림: 두 개의 평행한 branch가 위아래로 놓임.
  - 위 branch "Sister Network-1": 남성 얼굴 이미지 → 파란 3D 블록(conv feature map) 3개(첫째보다 둘째가 크고 셋째가 가장 작음) → 원 모양 뉴런 열(fully connected) 2개 → 출력 벡터 $f(x1)$
  - 아래 branch "Sister Network- 2": 여성 얼굴 이미지 → 같은 구조 → 출력 벡터 $f(x2)$
  - 두 벡터가 한 개의 파란 박스 "Contrastive Loss function"으로 합쳐지고, 그 오른쪽 빨간 박스에 "No Match" 출력
- 그림 아래 캡션: "Same Network + Shared Weights"
- 수식(슬라이드 하단): z1=f(x1), z2=f(x2)
- "If they are the same person decreases, d(z1,z2), otherwise d(z1,z2) increases ." ← 문장 어순이 어색함 (같은 사람이면 d(z1,z2)가 decrease, 아니면 increase 라는 뜻으로 해석, 원문 유지)

**설명**
- **Siamese network**: **동일한 네트워크 $f$(가중치 공유)** 를 두 입력 $x_1, x_2$에 각각 적용해 $z_1 = f(x_1)$, $z_2 = f(x_2)$ 를 얻고, 둘 사이의 거리 $d(z_1, z_2)$ 를 loss에 넣는 구조.
- **왜 weight sharing인가**: 두 입력이 **같은 embedding 공간**에 매핑되어야 거리 비교가 의미가 있음. 가중치가 다르면 두 branch가 서로 다른 좌표계를 만들어 거리가 무의미해짐. 또 파라미터 수가 절반이고 입력 순서에 대칭적임 ($d(z_1,z_2)=d(z_2,z_1)$).
- 학습 방식: **같은 사람(positive pair)** 이면 $d(z_1, z_2)$ 를 줄이고, **다른 사람(negative pair)** 이면 늘림. 그림의 예는 서로 다른 두 사람이라 출력이 "No Match".
- 예시가 얼굴 인증(face verification)인 이유: 클래스(사람) 수가 많고 새로운 사람이 계속 등장 → 분류기 대신 **거리 기반 비교**가 적합.
> 보충: Siamese network는 Bromley et al. (1993, 서명 검증)과 Chopra, Hadsell & LeCun (2005, 얼굴 검증)이 원조. "Contrastive Loss function" 박스의 구체적 수식은 다음 chunk에서 다룸. 슬라이드 그림상 f(x)는 CNN + FC로 표현됨.

## FIGURES

- p.4: 4가지 pretext task(image completion, rotation, jigsaw, colorization) 예시 — 중요도 중
- p.5: 같은 물체(초록 화살표)/다른 물체(빨간 화살표) similarity 관계도 — 중요도 상
- p.7: reference $x$ / positive $x^+$ / negative $x^-$ 범례가 있는 metric learning 도식 — 중요도 상
- p.8: Siamese network 구조(두 branch, weight sharing, contrastive loss) — 중요도 상

## EXAM POINTS

- Pretext task(rotation, jigsaw, colorization, image completion)의 공통 한계: **handcrafted objective** → task 설계 의존, shortcut 학습 위험. 이를 해결하는 동기가 contrastive learning임.
- "**Representation Learning = Learning Similarity**" 의 의미: 좋은 representation은 semantically similar한 샘플을 가깝게, 다른 샘플을 멀게 배치함.
- **Similarity as supervision**: 같은 물체의 view = positive(attract), 다른 물체 = negative(repel)로 라벨 없이 학습 신호 생성.
- **Metric learning**의 정의와 용어: reference $x$, positive $x^+$, negative $x^-$, pull together / push apart.
- **Siamese network** 구조: same network + **shared weights**, $z_i = f(x_i)$, 거리 $d(z_1,z_2)$를 loss에 사용. Why shared weights? (같은 embedding 공간, 대칭성, 파라미터 절약)
- 이 강의에서 다룰 loss 목록 순서(Contrastive → Triplet → Lifted Structured → N-pair → NCE → InfoNCE → Soft-NN)를 기억.

## VERIFIER REPORT

- p.2 설명: "역사순"은 과장 -> 슬라이드 목록 순서(연대순 아님)로 수정.
- p.2 설명: negative 수가 점점 늘고 확률 모델에 가까워진다는 단정 -> 대략적 흐름 + 필자 해석임을 명시.
- p.2 보충: Soft-NN 출처 오류(Salakhutdinov & Hinton 2007 NCA) -> Frosst, Papernot & Hinton(2019), NCA는 Goldberger et al. 2004로 정정.
- p.7 설명: "모든 contrastive loss가 (x, x+, x-) 구조" 과장 -> "대체로"; 근거 없는 supervised/unsupervised 구분 연결 문장 삭제.
- p.8 그림: 파란 블록 크기 묘사 오류("점점 작아짐") -> 둘째가 가장 크고 셋째가 가장 작음, FC 열 2개로 정정.
- 페이지 누락, 수식 오기, 나머지 인용(Pathak 2016, Gidaris 2018, Noroozi 2016, Zhang 2016, Bromley 1993, Chopra 2005, Gutmann 2010, van den Oord 2018)은 이상 없음.
