<!-- DRAFT | lecture: 08_Weakly_Supervised_Learning_I | chunk: 01 | pages 1–13 -->
<!-- VERIFIED | fixes: 1 | verifier: opus -->

## Title (p.1)

슬라이드 텍스트:
- Advanced Representation Learning (고급표현학습)
- Instructor: Donghyeon Cho
- The Department of Computer Science
- Hanyang University

**설명**
- 강의 08 표지야. 주제는 **Weakly-Supervised Learning**이고, 이번 슬라이드는 그중 **object detection** 쪽을 다뤄.
- 중간고사 뒤 첫 강의라 기말 범위의 시작이야.

> 보충: PDF 파일 메타데이터 제목은 `[2026][W8] Weakly-Supervised Learning I.pptx`인데, `exam/PLAYBOOK.md` 0.2의 일정(L01 p.4 기준)엔 W10 11/5로 적혀 있어. 슬라이드 본문엔 주차 표기가 없어서 시험과는 무관해.

---

## Contents of Table (p.2)

슬라이드 텍스트:
- 1. Weakly-Supervised Learning
- 2. Challenge
- 3. Weakly-Supervised Object Detection
- 4. Deep and End-to-End WSOD

**설명**
- 강의 흐름은 네 덩어리야.
  1. weak supervision이 뭐고 다른 학습 설정과 어떻게 다른가 (p.3–8)
  2. 왜 어려운가 (p.9–13)
  3. deep learning 이전의 고전적 방법 = **MIL pipeline**과 그 개선 (p.14–32)
  4. 한 network로 끝까지 학습하는 **WSDDN → OICR → PCL → Ren CVPR 20** (p.33–39)
- WSOD = **W**eakly-**S**upervised **O**bject **D**etection.
- 슬라이드 제목은 "Contents of Table"이야 ← "Table of Contents"가 뒤집힌 표기 (L07에도 같은 표기, 원문 유지).

---

## Section: Motivation to Weakly-Supervised Learning (p.3)

슬라이드 텍스트:
- Motivation to Weakly-Supervised Learning

**설명**
- 섹션 구분 슬라이드야. 왜 weak supervision이 필요한지부터 봐.

---

## Manual Supervision for Object Recognition (p.4)

슬라이드 텍스트:
- Weak supervision: Lower degree (or cheaper) annotation at train time than the required output at test time
- 그림 (왼쪽에서 오른쪽으로 annotation이 비싸져. 축 이름 = annotation time):

| annotation | 그림의 표기 | 시간 |
|---|---|---|
| image-level | {motorbike, person} | 1 sec per class |
| point | {motorbike (point), person (point)} | 2.4 sec per instance |
| bounding box | {motorbike (b-box), person (b-box)} | 10 sec per instance |
| pixel | {motorbike (pixel labels), person (pixel labels)} | 78 sec per instance |

- Berman et al., ECCV 2016, What's the Point: Semantic Segmentation with Point Supervision ← slide 오타. 실제 저자는 Bearman et al.

**설명**
- **annotation** = 사람이 data에 직접 붙이는 정답 표시야.
- **instance** = 사진 속 물체 하나하나야. 양이 세 마리면 instance가 3개야.
- **bounding box (b-box)** = 물체를 감싸는 직사각형이야.
- **pixel labels** = 픽셀마다 class를 매긴 것이야. 이걸 예측하는 task가 segmentation이야.
- **weak supervision의 정의는 상대적이야.** 핵심은 "train 때 받는 annotation이 test 때 내야 할 출력보다 낮은 수준(또는 싸다)"이야.
  - test 때 box를 내야 하는데 train 때 image label만 받으면 weak야.
  - test 때 pixel mask를 내야 하는데 train 때 box만 받으면 이것도 weak야.
  - 그러니까 같은 image label도 classification에선 full supervision이고 detection에선 weak supervision이야.
- 왜 하나? 시간 차이가 커. 슬라이드 숫자로 image label은 1 sec, pixel은 78 sec야.
- **단위 주의**: image label만 "per class"고 나머지는 "per instance"야. 사진에 같은 class 물체가 많을수록 image label이 더 유리해져.

**시험 관점**
- 정의 문장의 "train time"과 "test time"을 뒤바꾸면 바로 함정이야.
- annotation을 싼 순서로 나열하기: image label < point < bounding box < pixel labels.

---

## Related Topics: Semi-Supervised Learning (p.5)

슬라이드 텍스트:
- Semi-supervised: learns from a set of labeled (bounding box) and unlabeled images.
- Weakly semi-supervised: learns from a set of labeled (bounding box) and weakly labeled (image label) images.
- 그림: 사진 두 장(말 탄 사람)의 annotation 상태로 네 설정을 비교
  - (a) Supervised learning: 두 장 다 box + class
  - (b) Weakly supervised learning: 두 장 다 class 이름만 (horse, person), box 없음
  - (c) Weakly semi-supervised learning: 한 장은 box + class, 다른 한 장은 class 이름만
  - (d) Semi-supervised learning: 한 장은 box + class, 다른 한 장은 annotation 없음
- Image credit: [Jeong 19 NeurIPS]

**설명**
- 네 설정의 차이는 **"box가 없는 사진에 뭐가 남아 있나"**야.

| 설정 | 사진 A | 사진 B |
|---|---|---|
| Supervised | box | box |
| Weakly supervised | image label | image label |
| Weakly semi-supervised | box | image label |
| Semi-supervised | box | 없음 |

- L06–L07의 Semi-Supervised Learning은 (d)에 해당해. 나머지 data엔 아무 label도 없어.
- Weakly semi-supervised는 나머지 data에 **image label은 있어**. 여기서 (c)와 (d)가 갈려.

**시험 관점**
- "Semi-supervised learns from labeled and **weakly labeled** images"는 틀린 문장이야. 그건 weakly semi-supervised고, semi-supervised는 **unlabeled**야.

---

## Related Topics: Transfer Learning (p.6)

슬라이드 텍스트:
- Learns detectors for auxiliary categories with regular supervision (bounding box)
- Transfers the knowledge from auxiliary categories to weakly supervised categories
- 그림:
  - 위 줄 "Tasks with Strong Labels": Classification Style Data (person, bike 사진) + box가 있는 detection 사진 (person, bike)
  - 아래 줄 "Tasks with Weak Labels": Classification Style Data (dog 사진) + 후보 box가 여러 개 겹친 사진 (`?`, dog)
  - 두 줄 사이에 "Representation Learning" 화살표(위→아래), 아래 줄의 두 그림 사이에 "MIL" 양방향 화살표
- Image credit: [Hoffman NIPS 14]

**설명**
- **auxiliary category** = 보조로 쓰는 class야. 이 class엔 box annotation이 있어.
- box가 있는 class로 detector를 학습하고, 그 지식을 box가 없는 class로 옮기는 설정이야.
- p.5와 나뉘는 축이 달라.
  - semi 계열: **일부 사진**에만 box가 있어.
  - transfer: **일부 class**에만 box가 있어.
- 그림의 MIL은 p.15에서 정의해.

---

## Related Topics: Self-Supervised Learning (p.7)

슬라이드 텍스트:
- Defines a pretext task and learns representations in an unsupervised way
- Transfers pre-trained weights to a downstream task
- Transfer-learning requires regular annotations
- 그림 왼쪽: patch 위치 맞히기 예시 (Example, Question 1, Question 2)
- 그림 오른쪽: 이미지 X를 $g(X, y=0..3)$으로 0·90·180·270도 돌려서 ConvNet model $F(.)$에 넣는 rotation 예측
- Image credit: [Jeong 19 NeurIPS] ← 그림 내용은 L02의 Relative Position(L02 p.21–23)·Rotation(L02 p.40–43)인데 credit은 p.5와 같은 표기야. 원문 유지

**설명**
- **pretext task** = label 없이 data에서 정답을 자동으로 만드는 가짜 과제야. L02에서 봤어.
- **downstream task** = 진짜 풀고 싶은 과제야.
- 세 번째 줄의 뜻: pretext는 annotation이 필요 없지만, 배운 weight를 downstream으로 옮겨 학습할 땐 **보통의 annotation이 필요해**.
- 그래서 self-supervised는 "pre-training엔 annotation 없음 + 옮길 땐 annotation 있음"이고, weakly supervised는 "처음부터 싼 annotation을 직접 씀"이야.

---

## Weakly supervised learning (p.8)

슬라이드 텍스트:
- Typically uses pretrained networks and fine-tunes them by using weak supervision

**설명**
- **pretrained network** = 다른 data로 미리 학습해 둔 network야.
- **fine-tune** = 그 weight에서 출발해서 새 task로 조금 더 학습하는 거야.
- 즉 weak supervision은 보통 맨바닥에서 시작하지 않아. p.34의 WSDDN도 "finetunes a pre-trained deep network"라고 적혀 있어.

---

## Section: Challenges (p.9)

슬라이드 텍스트:
- Challenges

**설명**
- 섹션 구분 슬라이드야. p.10은 인식 일반의 어려움이고, p.11–13이 weak supervision에서 특히 문제되는 co-occurrence야.

---

## Challenges in Generic Object Recognition (p.10)

슬라이드 텍스트:
- Intra-class variations
  - Appearance
  - Viewpoint
  - Scale
  - Aspect ratio
- Background clutter
- Occlusions
- 그림: 색·각도·크기·배경이 제각각인 오토바이 사진 6장

**설명**
- **intra-class variation** = 같은 class 안에서 생김새가 달라지는 것이야. 같은 motorbike라도 색, 보는 각도, 크기, 가로세로 비율이 다 달라.
- **background clutter** = 배경이 복잡해서 물체가 묻히는 거야.
- **occlusion** = 다른 물체에 가려서 일부만 보이는 거야.
- 이 셋은 box가 있어도 어려운 문제야. weak supervision은 여기에 p.11–13의 문제가 더해져.

---

## Cooccurrence: Parts vs object (p.11)

슬라이드 텍스트:
- Parts vs object
- Question: What is a person?
  - a) Face
  - b) Face + upper body
  - c) Face + whole body ☺
- 그림: 사람 사진 4장. 얼굴과 상반신만 보이는 사진이 많고, 전신이 보이는 건 일부야.

**설명**
- **co-occurrence** = 두 가지가 늘 같이 나타나는 거야.
- image label "person"만 받으면 model은 **어디까지가 person인지** 알 길이 없어.
- 얼굴은 person 사진 대부분에 나와. 그러니까 얼굴만 찾아도 "이 사진에 person이 있나"는 잘 맞혀.
- 정답(☺)은 c인데, label만으로는 a와 c를 구분할 수 없어.
- 이 문제가 p.35의 "Detects discriminative parts rather than whole object extent"로 이어져.

---

## Cooccurrence: Context vs object (p.12)

슬라이드 텍스트:
- Context vs object
- Question: What is a train?
  - a) Rail
  - b) Train + Rail
  - c) Train ☺
- 그림: 기차 사진 4장. 철로가 늘 같이 보여.

**설명**
- train 사진엔 거의 항상 rail이 있어.
- 그래서 rail을 찾아도 "train이 있다"를 맞힐 수 있어. model 입장에선 rail이 train의 일부인지 배경인지 구분할 근거가 없어.
- p.11은 물체의 **일부**를, p.12는 물체 **바깥의 배경**을 물체로 착각하는 문제야.

---

## Cooccurrence: Multiple vs single instance (p.13)

슬라이드 텍스트:
- Multiple vs single instance
- Question: What is a sheep?
  - a) 2 sheeps
  - b) 3 sheeps + grass
  - c) 1 sheep ☺
- 그림: 양이 여러 마리 있는 사진 3장

← "sheeps"는 슬라이드 표기야 (sheep의 복수형은 sheep. 원문 유지)

**설명**
- 양이 늘 무리로 나오면, "양 두 마리를 묶은 덩어리"도 label "sheep"을 잘 설명해.
- image label은 **몇 마리인지** 말해 주지 않아. 한 마리의 경계를 알려 주는 정보가 없어.
- 이 문제가 p.35의 "Groups multiple instances together"로 이어져.

**시험 관점**
- 세 가지 ambiguity를 묶어서 기억해. 뒤에서 방법마다 "무엇을 고치나"의 기준이 돼.

| 이름 | 착각 | 뒤에서 나오는 곳 |
|---|---|---|
| Parts vs object | 일부를 전체로 | p.28 Objectness, p.35, p.38 |
| Context vs object | 배경을 물체로 | p.27 context |
| Multiple vs single instance | 여러 개를 하나로 | p.35, p.37 PCL |

- p.40은 이걸 "ambiguous due to part/whole and co-occurence issues"로 요약해.
