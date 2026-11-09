<!-- DRAFT | lecture: 09_Weakly_Supervised_Learning_II | chunk: 01 | pages 1–17 -->
<!-- VERIFIED | fixes: 10 | verifier: sonnet -->

## Title (p.1)

슬라이드 텍스트:
- Advanced Representation Learning (고급표현학습)
- Instructor: Donghyeon Cho
- The Department of Computer Science
- Hanyang University

**설명**
- 강의 09 표지야. 주제는 **Weakly-Supervised Learning II**이고, L08이 object detection(box)을 다뤘다면 이번엔 **semantic segmentation(pixel)** 쪽이야.
- L08과 같은 질문을 이어 가: "label이 내야 할 출력보다 약할 때, 모자란 정보는 어디서 채우나?"

> 보충: PDF 메타데이터 제목은 `[2026][W9] Weakly-Supervised Learning II.pptx`인데, `exam/PLAYBOOK.md` 0.2 일정(L01 p.4 기준)엔 **W11 11/12**로 적혀 있어. L08도 메타데이터는 W8, 일정은 W10이었어 (두 칸씩 밀림). 슬라이드 본문엔 주차 표기가 없어서 시험과는 무관해.

---

## Contents of Table (p.2)

슬라이드 텍스트:
- 1. Weakly-Supervised Semantic Segmentation
- 2. From Image-Level Labels to Pixel-Level Labels
- 3. Webly Supervised Learning
- 4. Boxly Supervised Learning

**설명**
- 슬라이드 제목은 "Contents of Table"이야 ← "Table of Contents"의 순서가 바뀐 표기. L08도 같은 표기였어. 원문 유지.
- 목차는 네 덩어리야.
  1. **WSSS가 뭐고 어떤 정보를 쓸 수 있나** (p.3–17): priors와 hints
  2. **image label → pixel label** (p.18–42): gray box / white box, CAM과 그 보강
  3. **Webly** (p.43–52): web에서 모은 사진을 supervision으로
  4. **Boxly** (p.53–61): box label에서 pixel label로
- 목차 밖에 p.62–95의 최신 논문 세 편(CD-CLIP, web 기반 incremental learning, WISH)이 더 붙어 있어. L08 노트 규칙대로 "넓히기" 파트로 따로 묶어.

---

## Section: Weakly-Supervised Semantic Segmentation (p.3)

슬라이드 텍스트:
- Weakly-Supervised Semantic Segmentation

**설명**
- 섹션 구분 슬라이드야. 이 섹션(p.3–17)은 방법이 아니라 **문제 설정과 쓸 수 있는 정보의 종류**를 정리해.
- **semantic segmentation** = 사진의 **모든 pixel**에 class 이름을 붙이는 task야. 출력은 사진과 같은 크기의 "class 지도"(mask)야.
- **WSSS (Weakly-Supervised Semantic Segmentation)** = test 땐 pixel마다 class를 내야 하는데, train 땐 그보다 약한 label(image label, point, box …)만 받는 설정이야.
- L08 p.4의 정의를 그대로 쓰면 돼: "Lower degree (or cheaper) annotation at train time than the required output at test time". 이번엔 test 출력이 **pixel**이야.

> 보충: L08 p.4 annotation 시간표를 다시 보면 pixel labels가 78 sec per instance로 제일 비싸 (image label 1 sec per class, point 2.4, b-box 10 sec per instance). segmentation이 weak supervision을 제일 절실하게 원하는 이유야.

---

## Fully Supervised vs Weakly Supervised (p.4)

슬라이드 텍스트:
- Fully Supervised vs Weakly Supervised
- (그림 안 글자) Fully supervised → **Directly** supervised
- (그림 안 글자) Weakly supervised → **Indirectly** supervised
- 그림: 위 줄 = 고양이 두 마리 사진 → 파란 화살표 → 두 고양이 몸 전체가 **분홍색 mask**로 칠해진 사진. 아래 줄 = 같은 사진에 고양이마다 **분홍 점 하나씩**만 찍힌 사진.
- Slide credit: Rodrigo Benenson

**설명**
- **Fully supervised = Directly supervised**: 정답이 출력과 같은 모양이야. pixel mask를 내야 하니까 pixel mask를 그대로 줘.
- **Weakly supervised = Indirectly supervised**: 정답이 출력과 다른 모양이야. 여기선 고양이마다 점 하나만 줘. 나머지 pixel이 고양이인지는 모델이 **간접적으로** 알아내야 해.
- 위 그림에서 **두 고양이가 같은 색 하나**로 칠해져 있어. semantic segmentation은 "어느 class냐"만 나누고, 고양이 1번과 2번을 따로 나누진 않아. 개체까지 나누는 건 **instance segmentation**이야 (p.54, p.87에서 다시 나옴).
- 이 "direct ↔ indirect" 구분이 p.5의 **Hints = "indirect supervision"** 정의로 바로 이어져.

> 보충 (RL 비유, 시험 범위 아님): fully supervised는 행동마다 정답 action을 주는 behavior cloning, weakly supervised는 episode 끝의 reward처럼 정답 모양과 다른 신호만 주는 쪽에 가까워. 신호에서 pixel 단위 정답을 거꾸로 추론해야 해.

---

## There are only two sources of information (p.5)

슬라이드 텍스트:
- There are only two sources of information
- **Priors**: what do I believe to be true independent of any particular image sample
- **Hints**: indirect supervision received for each image
- Slide credit: Rodrigo Benenson

**설명**
- weak label만으로는 pixel 정답이 안 정해져. 모자란 정보를 채울 곳은 **딱 두 군데**라는 게 이 슬라이드의 주장이야.
- **Prior** = 특정 사진과 **상관없이** 원래 참이라고 믿는 것. 데이터 전체, 세상 전체에 대한 믿음이야.
  - 예: "물체는 보통 한 덩어리다", "물체 경계는 사진의 edge와 겹친다".
- **Hint** = **사진마다** 따로 받는 간접 supervision. 사람이 그 사진을 보고 달아 준 약한 label이야.
  - 예: "이 사진엔 cat이 있다", "고양이 위에 점 하나".
- 구분 기준 하나로 외우면 돼: **사진마다 다르면 hint, 사진과 무관하면 prior.**
- 시험 함정: 두 정의를 뒤바꾸는 지문. "Priors: indirect supervision received for each image"는 False야.

> 보충 (RL 비유, 시험 범위 아님): prior ↔ agent 설계에 박아 넣는 inductive bias(물리 법칙, 연속성 가정), hint ↔ episode마다 받는 feedback(reward, 언어 지시, demonstration). 학습 신호가 약할수록 inductive bias를 더 많이 쓰게 되는 것도 같아.

---

## Priors: Size, Location, Shape (p.6)

슬라이드 텍스트:
- Priors
- Size, Location, Shape
- 그림: 숲 사진 두 장에 고양이를 **억지로 합성**한 사진. 왼쪽 = 나무만큼 거대한 고양이가 숲에 앉아 있고, 오른쪽 아래에 보통 크기 고양이. 오른쪽 = 몸 없이 **동그란 고양이 얼굴**만 있음 (큰 것 하나는 나무 사이 공중, 작은 것 하나는 왼쪽 아래 바닥 근처).
- Slide credit: Rodrigo Benenson

**설명**
- 슬라이드는 단어 세 개와 이상한 사진만 보여 줘. 사진이 **prior를 일부러 어긴 예**라서, 보는 순간 "이상하다"고 느끼는 그 감각이 바로 prior야.
  - **Size**: 고양이가 나무만큼 클 리 없어 → 물체마다 그럴듯한 크기가 있어.
  - **Location**: 고양이 얼굴이 공중에 떠 있을 리 없어 → 물체는 그럴듯한 위치(땅 위 등)에 있어.
  - **Shape**: 고양이가 동그란 얼굴만 있을 리 없어 → class마다 전형적인 모양이 있어.
- WSSS에선 이런 믿음을 loss나 후처리에 넣어서 "pixel mask 후보 중 그럴듯한 것"을 고르는 데 써.

> 보충: "prior를 어긴 합성 사진"이라는 읽기는 내 해석이야. 슬라이드엔 설명 문장이 없어. L08 p.29–31에도 Equivariance, Scale, Motion이 WSOD의 "Priors"로 나왔어.

---

## Priors: Number of instances, Contrast (p.7)

슬라이드 텍스트:
- Priors
- Number of instances, Contrast
- 그림: 왼쪽 = 어두운 숲 한가운데 고양이 **한 마리**가 배경과 비슷한 색이라 잘 안 보임. 오른쪽 = 자잘한 무늬가 가득한 배경 위에 큰 고양이 하나, 작은 고양이 하나.
- Slide credit: Rodrigo Benenson

**설명**
- **Number of instances**: 한 사진에 그 class 물체가 보통 몇 개 나오는지에 대한 믿음이야. (슬라이드엔 단어만 있어. 몇 개가 "보통"인지는 안 적혀 있어.)
- **Contrast**: 물체는 보통 배경과 **색·밝기가 달라서** 눈에 띈다는 믿음이야. 왼쪽 사진처럼 배경에 녹아든 물체는 이 prior가 안 통하는 경우야.
- 역시 사진은 설명용 예시고, 슬라이드엔 두 단어만 있어.

> 보충: 사진별로 어느 prior를 보여 주는지는 슬라이드에 안 적혀 있어. 위 매칭(왼쪽 = contrast가 낮은 경우)은 해석이야.

---

## Priors: Boundaries, saliency (p.8)

슬라이드 텍스트:
- Priors
- Boundaries, saliency
- 그림: 왼쪽 두 장 = 말 탄 사람과 서 있는 사람 사진 → 같은 장면의 **경계선(edge) 지도** (흰 바탕에 검은 윤곽선). 오른쪽 두 장 = 경주 오토바이 사진 → **saliency 지도** (오토바이와 탄 사람 pixel이 같이 빨갛고 배경은 파랗게 덮인 heat map).
- Slide credit: Rodrigo Benenson

**설명**
- **Boundaries**: 물체의 경계는 사진의 **edge(색이 급하게 바뀌는 선)** 와 겹친다는 믿음이야. mask를 edge에 맞춰 다듬을 수 있어.
  - (연결, 해석) 뒤의 SEC "Constrain-to-boundary" loss(CRF로 경계 맞추기, "Encourage masks to align with image boundaries", p.31)와 "boundary-aware propagation"(p.30)이 이 prior를 쓰는 예로 읽혀. 슬라이드가 직접 이어 주진 않아.
- **Saliency**: 사진에서 **눈에 확 띄는 영역**이 물체일 가능성이 높다는 믿음이야. 오른쪽 heat map처럼 "어디가 눈에 띄나"를 pixel마다 점수로 낸 게 saliency map이야. 오토바이와 탄 사람이 **class 구분 없이** 같이 빨개.
  - 이 saliency는 **class를 모르는(class-agnostic)** foreground 추정이야. 뒤의 Oh et al. CVPR 2017(p.33–34)이 "Saliency: Provides complete object extent, Class-agnostic foreground estimation"으로 이걸 써.
- **⚠️ saliency라는 단어가 이 강의에서 두 뜻으로 나와.**
  1. 여기(p.8)와 p.33–34: **salient object** = 눈에 띄는 물체 영역. class와 무관.
  2. p.21–22: **gradient-based saliency map** = classifier 점수가 각 pixel에 얼마나 민감한가. 한 class 점수에 대한 gradient라서 class마다 달라 (class-specific은 논문 기준 설명이고, 슬라이드엔 "score"로만 적혀 있어).

> 보충 (RL 비유, 시험 범위 아님): boundary prior는 "결과가 공간적으로 매끄럽고 경계에서만 바뀐다"는 smoothness 가정이야. value function이 비슷한 state에서 비슷하다는 가정과 같은 종류의 inductive bias야.

---

## Priors: Class Distribution (p.9)

슬라이드 텍스트:
- Priors
- Class Distribution
- 그림: 합성 사진 세 장. 세 장 모두 숲에 **고양이 여러 마리 + 자동차 한 대**가 있음 (처음 두 장은 고양이 2 + 차 1, 세 번째 장은 고양이 3 + 차 1). 아래에 큰 글씨로 "⅔ Cats, ⅓ Cars".
- Slide credit: Rodrigo Benenson

**설명**
- **Class distribution**: data 전체에서 class가 **어떤 비율로** 나오는지에 대한 믿음이야. 슬라이드 글자 "⅔ Cats, ⅓ Cars"는 "고양이가 차보다 두 배 많다"는 믿음이야.
- 이런 비율을 알면 모델 예측이 한 class로 쏠리는 걸 막는 데 쓸 수 있어. "이 data에선 cat이 car보다 두 배 많아야 정상"이라는 기준이 생기니까.

> 보충: 비율을 물체 개수로 셀지 pixel 면적으로 셀지는 슬라이드에 없어. 그림의 개수는 사진마다 조금씩 달라서 (고양이 2–3마리, 차 1대) 정확한 ⅔가 그림에서 나오진 않아.
> 보충: L09 p.65–69의 CD-CLIP도 "class distribution"이라는 말을 쓰는데, 거기선 **patch 하나가 class들에 대해 갖는 점수 분포**라는 뜻이야. 여기(p.9)의 data 전체 class 비율과는 다른 개념이야. 이름만 같아.

---

## Priors: Similarity across images, Similarity with external images (p.10)

슬라이드 텍스트:
- Priors
- Similarity across images
- Similarity with external images
- 그림 왼쪽: 고양이 사진이 여러 줄 (생김새가 서로 비슷함) — "versus" — 의자 사진이 여러 줄 (모양이 제각각: 나무 의자, 사무용 의자, 둥근 의자 …).
- 그림 오른쪽: "(a) Input database + sparse annotations" (풍경·도시 사진 여러 장. 몇 장에만 "tree, sky, river, mountain", "sky, river, building, bridge", "sidewalk, road, car, building, tree, sky" 같은 태그나 색칠된 pixel 지도가 붙어 있음) → "(b) Annotation propagation (→ = dense pixel correspondence)" (사진들이 화살표로 서로 연결된 그래프)
- Slide credit: Rodrigo Benenson

**설명**
- **Similarity across images**: 같은 class 물체는 **여러 사진에 걸쳐 비슷하게 생겼다**는 믿음이야. 학습 사진들끼리 공통으로 나오는 모양을 찾으면 그게 물체일 가능성이 높아.
  - 왼쪽 그림의 "versus"는 이 prior가 class마다 잘 통하는 정도가 다르다는 걸 보여 줘. 고양이는 사진마다 비슷하게 생겼는데, 의자는 모양이 제각각이야. (해석, 슬라이드엔 설명 문장 없음)
- **Similarity with external images**: 학습 data **밖의** 사진(큰 database, web 등)과도 닮은 점을 쓸 수 있다는 믿음이야.
  - 오른쪽 그림: 일부 사진에만 태그나 pixel label이 있는 큰 database에서, 닮은 사진끼리 **pixel 단위 대응(dense pixel correspondence)** 을 찾아 label을 옆 사진으로 퍼뜨려(annotation propagation).
- (연결, 해석) 뒤의 Webly Supervised Learning(p.43–52)은 "external images"를 web에서 직접 가져오는 방향으로 읽혀.

> 보충: 두 그림이 두 bullet 중 어느 쪽 예인지는 슬라이드에 안 적혀 있어. 왼쪽 = across images, 오른쪽 = external images로 짝지은 건 해석이야 (오른쪽은 한 database 안에서 퍼뜨리는 그림이라 across images 예로도 읽혀).
> 보충: 오른쪽 그림의 출처 논문은 슬라이드에 안 적혀 있어.

---

## Hints: Image (video) label, Image caption (p.11–12)

슬라이드 텍스트 (p.11과 p.12는 같은 내용):
- Hints
- Image (video) label
  - Person, cat, door
- Image caption
  - Kattie holds the cat next to the door
- 그림: 빨간 머리 여성이 회색 고양이를 안고 있고, 뒤에 문이 있는 사진.
- 하단 cite (p.11–17 공통): [Verbeek CVPR 07, Vezhnevets ICCV 11, Xu CVPR 14, Pinheiro CVPR 15, Pathak ICLR 15, Papandreou ICCV 15, Kolesnikov ECCV 16, Wei CVPR 17, Singh ICCV 17] ← 두 번째 줄 "Singh ICCV 17]"은 페이지 아래로 잘려서 글자 윗부분만 보여.

**설명**
- 여기부터 p.17까지가 **hint의 종류**야. 같은 사진(p.11, 14)에 다른 hint를 달아서 비교하게 돼 있어.
- **Image (video) label**: 사진(또는 영상)에 무엇이 있는지 **class 이름 목록**만 줘. "Person, cat, door". 위치·개수·모양은 없어. 이 강의 2장(p.18–42)의 주인공이야.
- **Image caption**: 사진을 설명하는 **문장**이야. "Kattie holds the cat next to the door". class 이름에 더해 **관계**(holds, next to)와 이름(Kattie)까지 들어 있어. 대신 문장에서 class를 뽑아내야 해.
- caption에서 class를 뽑아 label로 쓰는 아이디어는 p.79–81의 web incremental learning 논문이 실제로 써 ("Captions are converted into multi-label supervision").

> 보충: 하단 cite 목록은 hint 슬라이드마다 똑같이 붙어 있어. 이 중 Kolesnikov ECCV 16 = SEC(p.30–32), Wei CVPR 17 = adversarial erasing(p.39–41), Singh ICCV 17 = Hide-and-Seek으로 보이고, 뒤 슬라이드와 이어져. 특히 p.37–38의 Hide-and-Seek은 슬라이드 cite가 "Wang et al. CVPR 2020"인데, 이 목록의 Singh ICCV 17이 Hide-and-Seek 원 논문 표기와 맞아. (chunk 03에서 확인)

---

## Hints: Click inside object (p.13)

슬라이드 텍스트:
- Hints
- Click inside object
- 그림: 사진 네 장. 말 타는 두 사람 사진(사람마다 하늘색 점, 말마다 주황 점), 소파에 앉은 아이 둘(아이마다 하늘색 점, 소파에 분홍 점, 젖병에 남색 점), 털 위에 누운 고양이(분홍 점 하나), 길 위의 고양이(보라 점 하나).

**설명**
- **Click (point)**: 물체 **안쪽 아무 데나 점 하나**를 찍어. 점 색은 class를 나타내는 것으로 보여 (말 타는 사진: 사람 = 하늘색, 말 = 주황). 색-class 대응은 슬라이드에 설명이 없어.
- image label보다 정보가 많아: 위치를 대략 알려 주고, 물체마다 점을 찍으면 점 개수로 **instance 수**도 알 수 있어 (해석: 말 타는 사진에서 사람 둘 = 하늘색 점 둘).
- 그래도 물체의 **범위(경계)** 는 몰라. 점에서 어디까지가 물체인지는 모델이 채워야 해.
- L08 p.4 기준 point는 2.4 sec per instance로 image label(1 sec per class)과 b-box(10 sec) 사이야.
- 뒤의 WISH(p.87, 91–93)는 point를 "intermediate level of supervision"으로 써.

---

## Hints: Object bound boxes (p.14)

슬라이드 텍스트:
- Hints
- Object bound boxes ← "bounding boxes"의 줄임 표기 (원문 유지)
- 그림: p.11과 같은 사진에 box 세 개. 흰 box "Person"(사람 전체), 분홍 box "Human face"(얼굴), 노란 box "Cat"(고양이).

**설명**
- **Bounding box**: 물체를 감싸는 직사각형이야. 위치와 **대략의 크기·범위**를 알려 줘.
- 그래도 box 안의 **어느 pixel이 물체**인지는 몰라. box 안엔 배경도 섞여 있고, 그림처럼 box끼리 겹치기도 해 (Cat box와 Person box가 겹침).
- "box → pixel mask"가 4장 Boxly Supervised Learning(p.53–60)의 문제야.

> 보충: Person box와 Human face box처럼 **부분(얼굴)과 전체(사람)** 에 box가 따로 달릴 수 있어. L08 p.11의 "Parts vs object" 문제와 같은 구도야.

---

## Hints: Object extreme points (p.15)

슬라이드 텍스트:
- Hints
- Object extreme points
- 그림: 왼쪽 = 오토바이 묘기 사진에 **색 점 네 개**(맨 왼쪽 초록, 맨 위 빨강, 맨 오른쪽 노랑, 맨 아래 파랑). 오른쪽 = 자전거 탄 사람이 초록 mask로 칠해져 있고, 그 윤곽의 맨 위·왼쪽·오른쪽·아래에 흰 테두리 점 네 개.

**설명**
- **Extreme points**: 물체의 **가장 왼쪽, 가장 위, 가장 오른쪽, 가장 아래** 점 네 개를 찍어. (슬라이드엔 정의 문장이 없어. 그림의 점 4개와 표준 정의에 근거한 설명이야.)
- 네 점이면 box가 바로 나와 (왼쪽 점의 x, 위 점의 y …). 거기에 더해 점들이 **물체 윤곽 위**에 있으니까 box보다 모양 정보가 조금 더 있어.
- 오른쪽 그림은 네 점만 찍었는데 사람 모양 mask(초록)가 그려진 결과를 보여 줘. 네 점으로 mask까지 추정할 수 있다는 뜻이야.

> 보충: "box 네 꼭짓점 대신 물체 끝점을 찍으면 더 빠르고 정보도 많다"는 게 extreme point 아이디어야. 이 시간·성능 비교는 슬라이드에 없어서 시험 범위 밖이야.

---

## Hints: Scribbles (p.16)

슬라이드 텍스트:
- Hints
- Scribbles
- 그림: 왼쪽 = 실내에서 의자·소파에 앉은 사람들 사진에 **색깔 낙서 선**이 5개 (분홍 = 왼쪽 뒤 의자, 연한 하늘색 = 뒤쪽 화분, 자홍 = 소파·쿠션, 남색 = 소파 위 작은 물건, 길고 진한 하늘색 = 앞쪽 남자). 앉아 있는 여자에겐 선이 없음. 오른쪽 = 바구니 의자에 웅크린 고양이 몸통 위로 하늘색 선이 한 바퀴 구불구불 그어져 있음.

**설명**
- **Scribble**: 물체 위에 **붓으로 대충 선을 그은** 것이야. 선이 지나간 pixel은 그 class라고 확실히 알 수 있어.
- point보다 더 많은 pixel에 label이 생기지만, 여전히 물체의 **경계**는 안 그어. 선 밖의 pixel은 모델이 퍼뜨려서 채워야 해.
- 선을 모든 물체에 긋진 않아. 왼쪽 사진의 앉은 여자처럼 선이 없는 물체도 있어.

> 보충: 선 색과 class의 대응은 슬라이드에 설명이 없어. 위 짝은 그림을 보고 읽은 거야.

---

## Hints: Eye gaze (p.17)

슬라이드 텍스트:
- Hints
- Eye gaze
- 그림 왼쪽: 태블릿 화면을 보는 사람이 "A woman **crossing** the road from right to left along with a man"이라고 말함 → 세 화살표: 말 → **Language Understanding**, 시선 → **Gaze Estimation**, 화면 → **Scene Understanding** → 합쳐서 길 건너는 사람들 사진에 초록 box 하나.
- 그림 오른쪽: 네 줄. **Descriptions**("a woman in front is crossing the road from left to right side with a travel bag", "a red car in front is parked on right side of road along with other cars") / **Video and Gaze**(거리 영상 + 구석에 보는 사람 얼굴) / **Intermediate Results**(시선이 머문 곳에 빨간 heat map) / **Final Results**(찾은 물체에 box).

**설명**
- **Eye gaze**: annotator가 물체를 **쳐다본 시선**을 hint로 써. 사람이 클릭조차 안 하고 그냥 보기만 해도 "어디 있나"의 단서가 생겨.
- 그림은 시선(어디를 봤나)에 말(무엇을 말했나)과 장면 정보를 합쳐서 그 물체의 box를 찾는 예야.
- 정리하면 hint 목록은 이 순서로 나와: image label → caption → click → box → extreme points → scribbles → eye gaze. **슬라이드는 이 목록을 비용 순서로 정렬하진 않았어.**

> 보충: 그림 출처 논문은 슬라이드에 안 적혀 있어. 하단 cite 목록(Verbeek … Singh)은 image-level label WSSS 논문들이라 이 그림의 출처는 아니야.
> 보충 (Physical AI 연결, 시험 범위 아님): 사람 시선 data를 imitation learning의 보조 신호로 쓰는 연구도 있어. "어디를 보고 행동했나"가 policy 학습의 약한 hint가 되는 구조라서 이 슬라이드와 같은 발상이야.

---

## Chunk 01 정리

| 구분 | 정의 (p.5) | 슬라이드 목록 |
|---|---|---|
| **Priors** | "what do I believe to be true **independent of any particular image sample**" | Size, Location, Shape (p.6) / Number of instances, Contrast (p.7) / Boundaries, saliency (p.8) / Class Distribution (p.9) / Similarity across images, Similarity with external images (p.10) |
| **Hints** | "**indirect supervision** received **for each image**" | Image (video) label, Image caption (p.11–12) / Click inside object (p.13) / Object bound boxes (p.14) / Object extreme points (p.15) / Scribbles (p.16) / Eye gaze (p.17) |

- 뒤 슬라이드와의 연결: boundaries → SEC Constrain (p.30–32) / saliency → Oh et al. (p.33–34) / external images → Webly (p.43–52) / image label → 2장 전체 (p.18–42) / box → Boxly (p.53–60) / caption → web incremental (p.79–81) / tag·point·box 혼합 → WISH (p.86–95).
