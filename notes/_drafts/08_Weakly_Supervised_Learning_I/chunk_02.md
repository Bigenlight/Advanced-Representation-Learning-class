<!-- DRAFT | lecture: 08_Weakly_Supervised_Learning_I | chunk: 02 | pages 14–21 -->
<!-- VERIFIED | fixes: 1 | verifier: opus -->

## Section: Weakly-Supervised Object Detection (p.14)

슬라이드 텍스트:
- Weakly-Supervised Object Detection

**설명**
- 섹션 구분 슬라이드야. 이 섹션(p.14–32)은 고전적 WSOD의 틀인 **MIL pipeline**과 그 개선을 다뤄.
- **object detection** = 사진 속 물체마다 box와 class를 내는 task야.
- **WSOD** = train 때 box 없이 image label만 받고, test 때 box를 내는 설정이야.

---

## Principles for WSOD (p.15)

슬라이드 텍스트:
- Multiple-Instance Learning (MIL)
  - MIL reformulates weakly supervised object detection as a bag classification problem, where each image is a bag of candidate windows, and only the bag label (not the instance labels) is known.
- 그림: Positive bags (motorbike가 있는 사진 3장, 후보 window 여러 개 + 진짜 물체를 감싼 초록 window), Negative bags (motorbike가 없는 사진 3장, 후보 window만)
- bags = images
- instances = windows
- Goals:
  - find true positive instances
  - train window classifier

**설명**
- **window** = 사진 안의 후보 직사각형 영역이야.
- **bag** = instance들을 담은 묶음이야. 여기선 사진 한 장이 bag이고, 그 안의 window들이 instance야.
- **positive image (positive bag)** = 찾는 class가 들어 있는 사진이야. **negative image (negative bag)** = 그 class가 없는 사진이야.
- 아는 건 **bag label뿐**이야. "이 사진에 motorbike가 있다"는 알아도 어느 window가 그건지는 몰라.
- 목표는 두 개를 동시에 푸는 거야.
  1. positive bag에서 진짜 positive window 찾기
  2. window classifier 학습하기
- 이 둘은 서로 물려 있어. classifier가 있어야 window를 고르고, window가 있어야 classifier를 학습해. 그래서 p.16의 번갈아 풀기가 나와.

> 보충: MIL의 표준 가정은 "positive bag엔 positive instance가 **적어도 하나** 있고, negative bag의 instance는 **전부** negative"야. 슬라이드엔 문장으로 안 적혀 있고, p.21의 "all proposals in negative images are treated as negatives"가 뒷부분에 해당해.

> 보충 (RL 비유, 슬라이드 밖 = 시험 범위 아님): 에피소드 끝에 reward 하나만 받고 어느 action이 공인지 찾는 **credit assignment**와 닮았어. bag label 하나로 어느 window가 공인지 찾는 문제야.

---

## Standard MIL Pipeline (p.16)

슬라이드 텍스트:
- Two step alternating optimization
  - Re-train
  - Re-localize
- Challenges
  - Sensitive to initialization (local minimum)
  - Overfitting (locking) to predicted windows
- 그림: 후보 window가 많은 사진 → Initialize → box 하나를 고른 사진 → Object detection model. 사진에서 model로 가는 화살표가 Re-train, model에서 사진으로 돌아오는 화살표가 Re-localize

**설명**
- **어떻게**
  1. **Initialize**: positive image마다 시작 box를 정해 (p.18).
  2. **Re-train**: 지금 고른 box를 정답처럼 써서 detector를 학습해 (p.21).
  3. **Re-localize**: 학습한 detector로 box를 다시 골라 (p.20).
  4. 2–3을 반복해.
- **alternating optimization** = 두 변수를 한꺼번에 못 푸니까, 하나를 고정하고 다른 하나를 푸는 걸 번갈아 하는 거야.
- **Challenge 1: Sensitive to initialization (local minimum)**. 시작 box가 나쁘면 나쁜 답에 갇혀.
- **Challenge 2: Overfitting (locking) to predicted windows**. detector가 자기가 학습한 box를 다시 고르면서 그 box에 잠겨. p.23이 이 고리를 풀어서 설명해.

> 보충 (슬라이드 밖 = 시험 범위 아님): 구조는 K-means나 hard EM과 같아. "어느 window가 물체인가"가 숨은 변수고, detector가 parameter야.

> 보충 (RL 비유, 슬라이드 밖 = 시험 범위 아님): locking은 L07의 confirmation bias, RL의 bootstrapping error 누적과 같은 모양이야. 자기 예측을 target으로 다시 쓰면서 오차가 굳어져.

**시험 관점**
- Re-train과 Re-localize가 각각 뭘 고정하고 뭘 바꾸는지 구분해.
  - Re-train: box 고정, detector 갱신
  - Re-localize: detector 고정, box 갱신

---

## How to Generate Bags? (p.17)

슬라이드 텍스트:
- Sliding windows
  - \>100 ← "dense"와 proposals "~2k"에 비해 숫자가 작아 보여. 지수가 빠졌을 수 있지만 슬라이드로는 실제 값을 확인 못 해 (원문 유지)
  - dense
  - translations, scales and aspect-ratios (4D space)
- Object proposals
  - Selective Search and Edge Boxes
  - ~2k per image
  - Commonly used in WSOD
- 그림: 위 = 노란 window가 일정 간격으로 훑는 사진, 아래 = 파란 proposal이 많이 겹친 사진
- Slide credit: Vittorio Ferrari

**설명**
- bag을 만든다는 건 사진 한 장에서 후보 window를 뽑는다는 뜻이야.
- **sliding window** = 여러 크기·가로세로 비율의 창을 사진 위로 빽빽하게 옮겨 가며 후보를 만드는 방식이야.
  - **4D space** = 가로 위치, 세로 위치, 크기(scale), 가로세로 비율(aspect ratio) 네 축이야.
- **object proposal** = 물체가 있을 법한 영역을 class 구분 없이 미리 뽑아 주는 방식이야.
  - 대표 알고리즘이 Selective Search와 Edge Boxes야.
  - 사진 한 장에 약 2k개고, WSOD에서 많이 써.

**시험 관점**
- WSOD에서 흔히 쓰는 건 sliding window가 아니라 **object proposals**야.

---

## Initialization (p.18)

슬라이드 텍스트:
- Simple strategies:
  - Whole image
  - Whole image minus a margin
- [Song ICML 14] constructs a graph to find initial boxes:
  - 1. relevant (occur in many positive images)
  - 2. discriminative (dissimilar to the boxes in the negative images)
  - 3. complementary (capture multiple modes)
- Instead of initializing MIL with the entire image, Song et al. construct a proposal graph across positive images and select object proposals that are (1) common across positives, (2) discriminative from negatives, and (3) representative of diverse object appearances.
- 그림: 위 두 줄 = 사진 전체 box, margin을 뺀 box. 아래 = 여러 사진의 box를 잇는 graph

**설명**
- p.16의 Challenge 1 때문에 시작 box가 중요해.
- **단순한 방법**: 사진 전체를 box로 쓰거나, 테두리를 조금 뺀 box를 써. 물체가 어디 있든 일단 포함되는 쪽이야.
- **Song ICML 14**: positive 사진들에 걸쳐 proposal graph를 만들어서 시작 box를 골라. 조건이 세 개야.
  1. **relevant**: positive 사진 여러 장에 나오는 box. 물체라면 반복해서 나올 테니까.
  2. **discriminative**: negative 사진의 box와 달라야 해. 배경처럼 어디에나 있는 건 걸러져.
  3. **complementary**: 물체의 여러 modes(서로 다른 모습)를 담아야 해. 한 가지 모습만 고르면 안 돼.

**시험 관점**
- 세 조건의 이름과 뜻을 짝지어. "discriminative = **negative** 이미지의 box와 달라야"가 뒤집히기 쉬워.

---

## Standard MIL Pipeline (p.19)

슬라이드 텍스트:
- 그림만 있음: p.16의 그림에서 왼쪽 Initialize 부분이 흐려지고, Re-train / Re-localize 고리만 진하게 표시

**설명**
- Initialization 얘기는 끝났고, 이제 고리 안의 두 단계를 본다는 표시야. p.20이 Re-localize, p.21이 Re-train이야.

---

## Re-localize (p.20)

슬라이드 텍스트:
- Standard max formulation
  - Pick the highest scoring window as positive
  - Only one positive instance per image
- 수식과 기호 설명:

$$\arg\max_b A(x_b)$$

- $b$: Proposal / $A$: Appearance model / $x_b$: Features
- 그림: 후보 window가 많은 사진 2장 → 노란 box 하나만 남은 사진 2장
- [Nguyen ICCV 09, Bilen CVPR 14, Cinbis CVPR 14, Papadopoulos CVPR 16]

**설명**
- 기호를 풀면 이래.
  - $b$ = proposal 번호야.
  - $x_b$ = proposal $b$의 feature야.
  - $A(\cdot)$ = appearance model이야. feature를 받아서 "이 window가 찾는 물체 같은 정도"를 점수로 내.
- 사진마다 점수가 제일 높은 window **하나**를 positive로 골라.
- **한계**: 사진에 물체가 여럿 있어도 positive는 하나뿐이야. p.22가 이걸 완화해.

---

## Re-train (p.21)

슬라이드 텍스트:
- Hinge and cross-entropy loss
- For positive images:

$$\max_b A(x_b) > \Delta \ \ (\Delta\text{: margin}) \quad \text{or} \quad \log\big(\text{Softmax}(\max_b A(x_b))\big)$$

- For negative images:

$$\max_b A(x_b) < -\Delta \quad \text{or} \quad \log\big(1-\text{Softmax}(\max_b A(x_b))\big)$$

← log(Softmax(…)) 항이 부호를 뒤집은 loss가 아니라 키워야 할 objective 꼴로 적혀 있어. Softmax의 입력도 값 하나로 적혀 있어서 무엇에 대해 normalize하는지 슬라이드만으론 알 수 없어 (원문 유지).

- Difference to supervised object detection (eg Faster-RCNN)
  - Only 1 positive instance is used in each positive image
  - No negative instances from positive image
- During the re-training stage, MIL selects the highest-scoring proposal as the positive instance for each positive image, while all proposals in negative images are treated as negatives. Unlike fully supervised detection, only one positive proposal is used and no negative proposals are sampled from positive images.
- [Nguyen ICCV 09, Bilen CVPR 14, Cinbis CVPR 14, Papadopoulos CVPR 16]

**설명**
- **margin $\Delta$** = 점수가 넘어야 하는 여유 폭이야.
- 슬라이드는 같은 요구를 두 방식으로 적었어.
  - **hinge 쪽**: 점수가 margin을 넘으라는 부등식이야.
  - **cross-entropy 쪽**: 확률의 log를 키우라는 식이야.
- **positive image**: 최고 점수 window의 점수가 $\Delta$보다 커야 해. 즉 "물체다"라고 자신 있게 말하는 window가 **하나는** 있어야 해.
- **negative image**: **최고 점수**조차 $-\Delta$보다 작아야 해. 최고가 낮으면 나머지는 더 낮으니까, 이 식 하나가 "모든 window가 negative"라는 뜻이야. 슬라이드의 "all proposals in negative images are treated as negatives"와 맞아.
- **supervised detector와 다른 점 두 개**
  1. positive 사진에서 positive window를 **1개만** 써.
  2. positive 사진에서 negative window를 **뽑지 않아**.

> 보충 (슬라이드엔 식 없음): 부등식을 hinge loss로 쓰면 positive는 $\max(0,\ \Delta-\max_b A(x_b))$, negative는 $\max(0,\ \Delta+\max_b A(x_b))$ 꼴이야. 조건을 만족하면 loss가 0이고, 못 넘은 만큼 벌을 받아.

> 보충 (슬라이드엔 이유 없음): positive 사진의 나머지 window는 배경일 수도, 같은 class의 다른 물체일 수도 있어. 어느 쪽인지 모르니까 negative로 쓰면 진짜 물체를 negative로 가르칠 위험이 있어.

**시험 관점**
- negative image 식의 부호와 max를 기억해: $\max_b A(x_b) < -\Delta$.
- "MIL samples negative proposals from positive images"는 틀린 문장이야.
- "all proposals in **positive** images are treated as negatives"처럼 positive/negative를 뒤바꾸는 함정을 조심해.
