<!-- DRAFT | lecture: 09_Weakly_Supervised_Learning_II | chunk: 04 | pages 43–61 -->
<!-- VERIFIED | fixes: 4 | verifier: sonnet -->

## Section: Webly Supervised Learning (p.43)

슬라이드 텍스트:
- Webly Supervised Learning

**설명**
- 섹션 구분 슬라이드야. 목차(p.2)의 3번 파트고, p.43–52가 여기 들어가.
- 앞 파트(p.18–42)는 사람이 단 **image label**만으로 pixel label을 만들려고 했어. 이 파트는 **web에서 공짜로 모은 사진**을 supervision으로 끌어와.

---

## Webly Supervised Learning: 정의 (p.44)

슬라이드 텍스트:
- A learning method that uses data collected from the web as supervision.
  - Collect training images directly from web search engines.
  - Enables large-scale learning without manual annotation.
  - Labels are noisy, ambiguous, and incomplete.
  - Learning must be robust to noisy supervision.
- (굵은 글씨) Webly supervised learning leverages freely available web images as supervision, trading annotation cost for label noise.
- 그림: 세 줄 = saxophone, jasmine, chihuahua (검색어). 왼쪽 칸 = 흰 배경에 물체 하나가 가운데 있는 깔끔한 사진들 (색소폰 3개, 만화 캐릭터 jasmine, 치와와 2마리). 오른쪽 칸 = 배경이 복잡하거나 뜻이 다른 사진들 (악기를 든 장면, 공주 분장을 한 사람과 **jasmine 꽃**, 이불 위 강아지)
- [Shen CVPR 18, Jin CVPR 17, Chen ICCV 15]

**설명**
- **webly supervised** = web + weakly supervised를 합친 말이야. **검색어가 곧 label**이야. "saxophone"으로 검색해서 나온 사진엔 "saxophone" label이 붙어.
- 장점: 사람이 label을 안 달아도 돼서 **아주 많이** 모을 수 있어 ("Enables large-scale learning without manual annotation").
- 단점: label이 **noisy, ambiguous, incomplete**야.
  - **noisy** = 틀린 사진이 섞여. 검색 결과에 엉뚱한 사진이 나와.
  - **ambiguous** = 같은 단어가 여러 뜻이야. 그림의 "jasmine"은 꽃일 수도, 디즈니 공주일 수도 있어.
  - **incomplete** = 사진에 다른 물체도 있는데 검색어 하나만 label이 돼.
- 그래서 핵심 문장이 이거야: **"trading annotation cost for label noise"**. annotation 비용을 아끼는 대가로 label noise를 받아들이는 거래야.
- 결론: 학습은 noisy supervision에 **robust**해야 해.

> 보충 (그림 해석, 슬라이드엔 캡션 없음): 왼쪽은 물체 하나가 가운데 있는 "쉬운" 검색 결과, 오른쪽은 현실 장면이나 다른 뜻이 섞인 "어려운" 결과로 읽혀. 이 왼쪽 같은 사진이 p.47에서 "web images are much cleaner—typically containing a single centered object"의 근거가 돼.

> 보충: cite 확인 — Shen et al., "Bootstrapping the Performance of Webly Supervised Semantic Segmentation", CVPR 2018 / Jin et al., "Webly Supervised Semantic Segmentation", CVPR 2017 / Chen & Gupta, "Webly Supervised Learning of Convolutional Networks", ICCV 2015. 이 파트의 pipeline(p.45–52)은 Shen CVPR 2018 하나야.

---

## Webly Supervised Learning: Pipeline step 1 (p.45)

슬라이드 텍스트:
- Pipeline
  - **1. Train an image classifier (Model-T) on clean image-level labels.** (굵게 강조)
  - 2. Use Model-T to filter noisy web images.
  - 3. Train Web-SEC (pixel labeler) using the filtered web images.
  - 4. Apply Web-SEC to generate pixel labels for the target images.
  - 5. Use Model-T to produce class activation maps (CAMs) on the target images.
  - 6. Fuse the two label sources and train the final segmentation model.
- 그림: 왼쪽 타원 "Target domain"(군인, 사람들 단체 사진, 소 등 VOC 같은 현실 사진) 안에 **Model-T**, 오른쪽 타원 "Web domain"(스쿠터, 흰 말, 소, 비행기, 아이 = 물체 하나짜리 사진) 안에 **Model-W**. Model-T → Model-W 화살표 = **Filter**, Model-W → Model-T 화살표 = **Enhance**
- [Shen CVPR 18]

**설명**
- p.45–50은 같은 6단계 목록을 한 단계씩 굵게 칠하면서 넘어가. 지금은 1단계야.
- **domain** = data가 나온 곳(분포)이야.
  - **target domain** = 우리가 실제로 segmentation을 하고 싶은 dataset이야 (그림은 PASCAL VOC 같은 현실 사진).
  - **web domain** = 검색으로 모은 사진이야.
- **Model-T** = target domain의 **clean image-level labels**(사람이 단 image label)로 학습한 **image classifier**야. T는 Target이야.
- 그림의 **Model-W**는 web domain 쪽 모델이야. 글의 **Web-SEC**가 이 자리에 해당해 (p.50 "Learn a pixel-level object labeler (Web-SEC) from the filtered web images"). ← 슬라이드는 둘을 직접 연결하진 않아. 그림상 web 쪽 모델이 pixel labeler라서 같은 것으로 읽은 거야 (추정).
- 그림의 두 화살표가 pipeline 전체야.
  - **Filter**(T → W): Model-T가 web 사진을 걸러 줘 (2단계).
  - **Enhance**(W → T): web에서 배운 지식이 target 쪽 label을 좋게 해 (4·6단계).
- 즉 **양방향**이야. target이 web을 걸러 주고, web이 target을 도와줘.

> 보충: 논문은 이걸 "bi-directional transfer"라고 불러. 제목의 **bootstrapping**은 "작은 것에서 시작해 스스로 끌어올린다"는 뜻이야.

---

## Pipeline step 2: Filter (p.46)

슬라이드 텍스트:
- (같은 6단계 목록) **2. Use Model-T to filter noisy web images.** (굵게)
- 그림: "noisy web images"(비행기 사진 더미) → 화살표 → "clean web images"(더 얇은 더미)
- [Shen CVPR 18]

**설명**
- web 사진엔 검색어와 안 맞는 사진이 섞여 있어(p.44). 그래서 먼저 걸러.
- **누가 거르나**: **Model-T**야. clean label로 학습한 classifier라서, "이 사진이 정말 airplane인가?"를 판단할 수 있어.
- 그림에서 더미가 얇아지는 게 "버린 사진이 있다"는 뜻이야.

> 보충 (RL 비유, 시험 범위 아님): 깨끗한 data로 학습한 classifier로 큰 noisy data를 걸러서 쓰는 건 **filtered behavior cloning**이랑 닮았어. 성공 판정기(또는 reward model)로 demo를 걸러 놓고 남은 것만 imitate하는 거야.

---

## Pipeline step 3: Web-SEC 학습 (p.47)

슬라이드 텍스트:
- (같은 6단계 목록) **3. Train Web-SEC (pixel labeler) using the filtered web images.** (굵게)
- 그림 왼쪽: "noisy web images" → "clean web images" → "fine masks (web images)"(비행기 모양 빨간 mask)
- 그림 가운데: "image" / "Web-SEC" 두 열. 개 사진 → 개 모양 보라 mask, 아이 사진 → 아이 모양 분홍 mask
- 오른쪽 글:
  - Unlike CAM-based methods, Web-SEC actually learns a pixel classifier.
  - The key observation is that web images are much cleaner—typically containing a single centered object—so approximate segmentation masks can be generated automatically.
  - These masks are then used to train a pixel-level segmentation network, which learns generic object shapes and can later be transferred to the target dataset.
- [Shen CVPR 18]

**설명**
- **pixel labeler** = 사진을 넣으면 pixel마다 class를 매겨 주는 network야. 즉 segmentation network야.
- **핵심 관찰**: web 사진은 대개 **물체 하나가 가운데** 있어. 그래서 사람이 안 그려도 **대략적인 mask를 자동으로** 만들 수 있어.
- 그 자동 mask(그림의 "fine masks")로 **pixel-level segmentation network**를 학습해. 이게 **Web-SEC**야.
- **CAM 계열과 차이** (시험 포인트):
  - CAM(p.24–27)은 **classifier**의 activation을 읽어서 위치를 추정해. 분류용으로 학습된 network에서 위치를 "빌려 오는" 거야.
  - Web-SEC는 **진짜 pixel classifier**를 학습해 ("Web-SEC actually learns a pixel classifier").
- Web-SEC가 배우는 건 **generic object shapes**(물체 모양 일반)야. 나중에 target dataset으로 **transfer**해.

> 보충: 이름의 SEC는 p.31–32의 SEC(Seed, Expand, Constrain)와 같은 이름이야. 둘의 관계는 슬라이드가 설명하지 않아.

---

## Pipeline step 4: Web-SEC를 target에 적용 (p.48)

슬라이드 텍스트:
- (같은 6단계 목록) **4. Apply Web-SEC to generate pixel labels for the target images.** (굵게)
- 그림: p.45와 같은 Target domain / Web domain 그림. 아래쪽 **Enhance** 화살표(Model-W → Model-T)에 주황 박스로 강조
- [Shen CVPR 18]

**설명**
- web에서 학습한 Web-SEC를 **target 사진**(VOC 사진)에 돌려. 그러면 target 사진마다 pixel label이 하나 생겨.
- 강조된 **Enhance** 화살표가 이 단계야. web 지식이 target 쪽으로 넘어와.
- 이 label이 p.50 그림의 "masks with web knowledge"야.

---

## Pipeline step 5: Model-T로 CAM (p.49)

슬라이드 텍스트:
- (같은 6단계 목록) **5. Use Model-T to produce class activation maps (CAMs) on the target images.** (굵게)
- 그림: "VOC training images (image-level labels)"(식탁에 둘러선 사람들 단체 사진) → "initial masks"(사람 = 분홍 덩어리, 식탁 = 주황 덩어리. 경계가 뭉툭함)
- [Shen CVPR 18]

**설명**
- 이번엔 **Model-T**가 target 사진의 **CAM**을 만들어. CAM은 p.24–27의 그 CAM이야.
- 결과가 **initial masks**야. 그림을 보면 사람들이 한 덩어리로 뭉치고 경계가 흐려. p.28의 CAM 한계("not sharp")가 그대로 보여.
- 여기까지 target 사진마다 label이 **두 개** 생겼어.
  1. Web-SEC가 만든 mask (4단계)
  2. Model-T의 CAM으로 만든 initial mask (5단계)

---

## Pipeline step 6: Fuse → 최종 학습 (p.50)

슬라이드 텍스트:
- (같은 6단계 목록. 단 3번 줄이 바뀜) 1. Train an image classifier (Model-T) on clean image-level labels. / 2. Use Model-T to filter noisy web images. / Learn a pixel-level object labeler (Web-SEC) from the filtered web images. ← 번호 "3." 빠짐 (원문 유지) / 4. … / 5. … / **6. Fuse the two label sources and train the final segmentation model.** (굵게)
- 그림: "VOC training images (image-level labels)" → "initial masks". 위쪽 "masks with web knowledge". 두 mask가 합쳐져 → "enhanced masks"
- [Shen CVPR 18]

**설명**
- 두 label을 **합쳐서(fuse)** "enhanced masks"를 만들어. 이걸 정답처럼 써서 **final segmentation model**을 학습해.
- 그림에서 web knowledge mask는 사람 모양을 더 잘 잡고, initial mask는 식탁(주황) 같은 class 정보를 갖고 있어. 합친 enhanced mask엔 둘 다 들어가.
- 3번 줄이 "Learn a pixel-level object labeler (Web-SEC) from the filtered web images"로 바뀌었어. 뜻은 p.45의 3번과 같아.
- **누가 어떤 label을 만드나** (시험 함정 정리):

| 모델 | 무엇으로 학습 | 하는 일 |
|---|---|---|
| Model-T | target의 clean **image-level** labels | ② web 사진 filter, ⑤ target 사진의 CAM → initial masks |
| Web-SEC (그림의 Model-W로 추정) | filter된 **web** 사진 + 자동 mask | ④ target 사진의 pixel label (masks with web knowledge) |
| final segmentation model | ⑥ 두 label을 합친 enhanced masks | test 때 쓰는 모델 |

> 보충 (논문): 두 label을 합치는 구체적인 규칙은 슬라이드에 없어. 이름(Model-T, Web-SEC)과 단계 순서만 외워.

---

## 결과 표 (p.51–52)

슬라이드 텍스트:
- Table 1: Comparison with methods using other supervisions. (열: Method / val / test / Extra Supervision)
  - Chen et al. [2]: 63.7 / 66.4 / Fully supervised (강조)
  - Lin et al. [16]: 63.1 / - / Scribble
  - Dai et al. [6]: 62.0 / 64.6 / Bounding box+MCG
  - Oh et al. [21]: 55.7 / 56.7 / Bounding box
  - Bearman et al. [1]: 46.1 / - / Point
  - Wei et al. [29]: 55.0 / 55.7 / Supervised saliency
  - STC [30]: 49.8 / 51.2 / Supervised saliency
  - EM-Adapt [22]: 33.8 / 39.6 / -
  - CCNN [23]: 35.3 / 35.6 / -
  - SEC [13]: 50.7 / 51.7 / -
  - Hong et al. [10]: 58.1 / 58.7 / -
  - Ours-VGG16: 58.8 / 60.2 / Web
  - Ours-Res50: 63.0 / 63.9 / Web (강조)
- 그림: input / prediction 세 쌍 (사람과 고양이, 검은 양 떼, 말과 사람 둘)
- [Shen CVPR 18]
- p.51과 p.52는 같은 슬라이드야 (원문 그대로 두 장).

**설명**
- 숫자는 PASCAL VOC 2012 segmentation의 mIoU(%)야 (논문 기준. 표에 dataset 이름은 없어).
- 읽는 법: 위 칸은 **추가 annotation**(scribble, box, point, saliency)을 쓴 방법이고, 아래 칸 "-"는 image label만 쓴 방법이야. Ours는 추가 annotation 대신 **Web**을 써.
- 메시지: image label + web만으로 **Ours-Res50 63.0(val)** 이 나와. 강조된 fully supervised 줄 **63.7(val)** 에 가까워. image label만 쓰는 SEC(50.7)보다 훨씬 높아.
- 수치는 시험 비중이 낮아. "web이 추가 annotation 없이 fully supervised에 가까워진다" 정도만 기억해.

> 보충: Ours-VGG16(58.8) → Ours-Res50(63.0)은 같은 방법에 backbone만 바꾼 거야. p.61의 "ride the wave of better network architectures"의 예로 볼 수 있어. 표의 fully supervised 줄(Chen et al.)이 어떤 backbone인지는 슬라이드에 없어서, Ours-Res50과 1:1로 비교하진 마.

---

## Section: Boxly Supervised Learning (p.53)

슬라이드 텍스트:
- Boxly Supervised Learning

**설명**
- 섹션 구분 슬라이드야. 목차(p.2)의 4번 파트고, p.53–60이 여기 들어가.
- 이번엔 image label보다 정보가 많은 **bounding box**를 weak label로 써.

---

## From box-level labels to pixel-level instance labels (p.54)

슬라이드 텍스트:
- From box-level labels to pixel-level instance labels
- 그림: 고양이 두 마리 사진에 노란 점선 box 두 개(각각 "Cat") → 파란 화살표 → 고양이 두 마리가 각각 분홍 mask로 칠해진 사진(각각 "Cat")
- 왼쪽 아래: Box-level labels at training time
- 오른쪽 아래: Pixel-level per instance labels at test time

**설명**
- **bounding box** = 물체를 감싸는 직사각형이야. 물체의 위치와 대략적인 크기는 알려 주지만, box 안의 어느 pixel이 물체인지는 안 알려 줘.
- **instance** = 물체 하나하나야. **per instance labels** = 같은 class라도 물체마다 따로 mask를 낸다는 뜻이야. 그림에서 고양이 두 마리가 각각 따로 칠해져.
- **weak supervision 정의와 연결**: L08 p.4의 정의는 "Lower degree (or cheaper) annotation at train time than the required output at test time"이었어. 여기선 train = **box**, test = **pixel-level instance mask**야. box가 mask보다 싸니까 weak supervision이야.

> 보충: L08 p.4 annotation 시간 표에서 b-box는 10 sec per instance, pixel labels는 78 sec per instance였어. box가 mask보다 훨씬 싸.

---

## What we want vs What we have (p.55)

슬라이드 텍스트:
- 그림: 말 탄 사람 사진 / **What we want:** 말(자주색), 사람(분홍), 뒤 차들(회색)이 실제 모양대로 칠해진 mask / **What we have:** 같은 색의 **직사각형** 덩어리 (말 = 큰 자주 사각형, 사람 = 그 위에 겹친 분홍 사각형, 차 = 회색 띠)
- [Khoreva CVPR 17]

**설명**
- 원하는 건 물체 모양 그대로의 mask야. 가진 건 box뿐이야.
- 가장 단순한 방법은 **box 안을 통째로 그 class로 칠하는 것**이야. 이걸 "Rectangles"라고 불러 (p.57 그래프의 곡선 이름).
- 이 label은 **noisy**해. 말 다리 사이 배경, 사람 주변 공간까지 전부 "말"이나 "사람"으로 칠해져.
- 그림에선 작은 box(사람)가 큰 box(말) 위에 덮여 칠해져 있어.

> 보충: cite 확인 — Khoreva et al., "Simple Does It: Weakly Supervised Instance and Semantic Segmentation", CVPR 2017.

---

## Convnets are robust to noise. Let us use that! (p.56)

슬라이드 텍스트:
- Convnets are robust to noise. Let us use that !
- 그림 (왼쪽 → 오른쪽): Example input rectangles (말 탄 사람 사진 위 직사각형) → Output after 1 training round (뭉툭한 말 모양) → After 5 rounds → After 10 rounds (다리까지 보이는 말 모양) → Ground truth
- Initially, training is performed using a noisy mask, but the CNN's predictions are then used as pseudo-labels for retraining.
- Segmentation networks are robust to noisy supervision.
- Approximate masks are sufficient for training.
- Iteratively refine pseudo masks during training.
- Better masks lead to better segmentation.
- [Khoreva CVPR 17]

**설명**
- **한 줄**: 직사각형 mask로 시작해서, network 예측을 다음 round의 label로 쓰면서 mask를 점점 다듬어.
- **robust to noise**인 이유(직관, 슬라이드엔 이유 설명 없음): label이 틀린 pixel(box 안 배경)은 사진마다 제각각이야. 반면 물체 모양은 사진끼리 공통이야. 그래서 network는 공통 패턴(물체 모양)을 먼저 배우고, 틀린 pixel까지 그대로 따라 하진 않아. 그 결과 예측이 입력 직사각형보다 물체에 더 가까워.
- **어떻게** (recursive training):
  1. box를 칠한 직사각형 mask로 segmentation network를 학습해.
  2. 학습한 network로 training 사진의 mask를 다시 예측해.
  3. 그 예측을 **pseudo-label**(모델이 만든 가짜 정답)로 써서 다시 학습해.
  4. 2–3을 반복해. 그림에서 1 → 5 → 10 round로 갈수록 말 다리가 살아나.
- **"Better masks lead to better segmentation"**: label mask가 좋아질수록 segmentation도 좋아지고, 좋아진 segmentation이 다시 더 좋은 label을 만들어.
- **이전 강의와 연결**:
  - L07 self-training(pseudo labeling)과 같은 구조야. 차이는 단위가 **pixel**이고, box라는 약한 label이 있다는 거야.
  - L08 Standard MIL의 Re-train ↔ Re-localize 반복과도 닮았어. 거기선 box를 다시 골랐고, 여기선 mask를 다시 그려.
  - 위험도 같아. 틀린 예측을 label로 쓰면 **confirmation bias**(L07)로 오류가 굳을 수 있어.

> 보충 (논문, 슬라이드엔 없음): 논문은 그냥 반복하면(naive recursive training) 잘 안 된다고 해. 그래서 round마다 예측을 손봐. box 밖으로 나간 예측은 background로 되돌리고(box enforcing), 너무 작아진 예측은 처음 box로 되돌리고(outlier reset), CRF로 경계를 다듬어. box가 "label은 이 안에만 있다"는 제약으로 계속 쓰이는 거야.

> 보충 (RL 비유, 시험 범위 아님): round마다 직전 모델의 예측을 다음 target으로 쓰는 건 **fitted Q iteration**의 target 갱신과 같은 구조야. 오류가 쌓이는 위험(bootstrapping error)도 같아. box enforcing은 target을 data가 허락하는 범위 안에 묶어 두는 장치라서, offline RL의 support constraint와 동기가 닮았어.

---

## Training rounds 그래프: Rectangles (p.57)

슬라이드 텍스트:
- Convnets are robust to noise. Let us use that !
- (p.56과 같은 round별 그림 줄)
- 왼쪽 아래 그림: 직사각형 mask (회색 띠 + 자주 사각형 + 분홍 사각형)
- 그래프: x축 Training rounds (0–15), y축 mIoU (50.0–70.0)
  - 맨 위 실선: Fully supervised (~70)
  - 점선: Previous best (any input) (~62)
  - 아래 파선: Previous best (rectangles input) (~52.3)
  - 초록 곡선 Rectangles: round 1에 ~50 → round 4–5에 ~60 → round 10 근처 ~61에서 평평
- [Khoreva CVPR 17] (그래프 값은 눈대중)

**설명**
- **mIoU**(mean Intersection over Union) = segmentation 성능 지표야. class마다 "예측 pixel 집합과 정답 pixel 집합의 IoU(겹친 넓이 ÷ 합친 넓이)"를 구해서 평균 내. 클수록 좋아.
- 읽는 법: 초록 곡선이 round가 늘수록 올라가. round 1엔 이전 방법(rectangles input, ~52.3)보다 살짝 낮지만(~50), 반복하면 **이전 best(rectangles input)를 크게 넘고** 이전 best(any input)에 가까워져.
- 그래도 fully supervised(~70)까진 거리가 있어. 직사각형이라는 출발점이 너무 거칠어서야.

---

## 20% Bbox (p.58)

슬라이드 텍스트:
- Convnets are robust to noise. Let us use that !
- (같은 round별 그림 줄)
- 왼쪽 아래 그림: box 영역 대부분이 베이지색, 그 **가운데에만 작은** 분홍·자주 사각형
- 그래프: p.57 곡선들 + 연초록 곡선 **20% Bbox** (round 1에 ~56.5 → ~63에서 평평, Previous best (any input) 위로 올라감)
- [Khoreva CVPR 17]

**설명**
- 입력 label을 바꾼 버전이야. box 전체를 칠하지 않고 **box 가운데 일부만** class로 칠해.
- 그래프: 20% Bbox는 **시작점부터 높고**(~56.5 vs ~50), 끝점도 Rectangles보다 높아. 이전 best(any input)도 넘어.
- 이유(직관): box 가장자리는 배경일 확률이 높고, box 가운데는 물체일 확률이 높아. 확실한 곳만 칠하면 틀린 label이 줄어. 즉 **"Better masks lead to better segmentation"** 의 예야.

> 보충 (논문, 슬라이드엔 설명 없음): 논문에서 이 입력은 box 넓이의 **안쪽 20%** 만 class로 칠하고, 나머지 box 영역은 **ignore**(loss를 안 거는 영역)로 둬. 그림의 베이지색이 ignore 영역이야. §2.4의 CAM seeds처럼 "확실한 곳만 label을 단다"는 생각과 같아.

---

## GrabCut+ · MCG · MCG ∩ GrabCut+ (p.59)

슬라이드 텍스트:
- Convnets are robust to noise. Let us use that !
- 왼쪽 그림: 직사각형 mask → 파란 아래 화살표 → 말과 사람 모양에 가깝게 다듬어진 mask
- 그래프: p.58 곡선들 + 위쪽 세 곡선
  - **MCG ∩ Grabcut+** (주황, round 1에 큰 점 ~65.7, ~66 근처에서 약간 내려감, round 7까지)
  - **MCG** (진한 파랑, ~62.6 → ~65 → ~64.8, round 5까지)
  - **Grabcut+** (하늘색, ~63.4 → ~64.5 → ~64, round 5까지)
  - 20% Bbox, Rectangles, Fully supervised, Previous best 두 줄은 그대로
- [Khoreva CVPR 17]

**설명**
- 이번엔 box를 **고전 segmentation 알고리즘**에 넣어서 더 좋은 초기 mask를 만들어.
  - **GrabCut** = box를 주면 box 안에서 전경(물체)과 배경을 색 분포로 나누는 고전(딥러닝 아닌) 방법이야. **Grabcut+** 는 그 개량판이야.
  - **MCG**(Multiscale Combinatorial Grouping) = 사진에서 "물체일 것 같은 segment 후보"를 뽑는 고전 방법이야.
  - **MCG ∩ Grabcut+** = 두 방법이 **둘 다** 물체라고 한 pixel만 쓰는 거야 (∩ = 교집합).
- 그래프 읽는 법:
  - **Grabcut+와 MCG ∩ Grabcut+는 round 1부터**, MCG는 round 2부터 20% Bbox의 최종값(~63)보다 높아. 세 곡선 모두 round 2 이후엔 20% Bbox보다 위야.
  - 곡선이 거의 평평하고 짧아. 처음 label이 좋으면 반복할 게 별로 없어.
  - 제일 높은 건 MCG ∩ Grabcut+이야. 큰 점(round 1)이 거의 최고점이야.
- 정리: 성능을 올리는 손잡이가 두 개야.
  1. **입력 label 설계**(Rectangles < 20% Bbox < GrabCut+ · MCG < MCG ∩ GrabCut+)
  2. **recursive training**(거친 입력일수록 효과가 커)

> 보충 (논문, 슬라이드엔 없음): GrabCut+ 는 GrabCut의 pairwise term(이웃 pixel끼리 같은 label이 되게 하는 항)에 색 차이 대신 HED라는 경계 검출기의 경계를 쓴 버전이야. M ∩ G+ 에서 두 방법이 의견이 다른 pixel은 ignore로 둬.

---

## We reach ~95% of full supervision by using GrabCut++ (p.60)

슬라이드 텍스트:
- We reach ~95% of full supervision by using GrabCut++
- 그림: 다섯 줄(자전거 선수들, 당나귀, 회의 테이블, 사람과 소, 말과 사람) × 여섯 열: Image / Ground truth / Rectangles / 20% Bbox / M ∩ G+ / Fully supervised
- [Khoreva CVPR 17]

**설명**
- 열을 왼쪽에서 오른쪽으로 보면 입력 label이 좋아질수록 결과가 Ground truth에 가까워져.
  - Rectangles: 물체가 네모나게 부풀어. 소 줄에선 소 안에 다른 색이 섞여.
  - 20% Bbox: 조금 나아졌지만 여전히 뭉툭해.
  - M ∩ G+: 자전거 바퀴, 당나귀 다리 같은 얇은 부분이 살아나. Fully supervised와 거의 비슷해.
- 제목의 메시지: box만으로 **fully supervised의 약 95%** 성능에 도달해.
- 제목은 "GrabCut++"인데 그림 범례와 그래프(p.59)는 "Grabcut+", "M ∩ G+"야 ← 슬라이드 표기 불일치 (원문 유지). 논문 기준으로 ~95%를 낸 입력은 **M ∩ G+**(MCG ∩ GrabCut+)야.

> 보충: 논문 abstract도 "reach 95% of the quality of the fully supervised model, for both semantic labelling and instance segmentation"이라고 적어. 숫자 자체는 비중이 낮아.

---

## Summary (p.61)

슬라이드 텍스트:
- Many priors and hints have been explored
- Exploit all available priors, explore new information sources
- Do not underestimate the power of transfer learning (and webly supervision)
- Current techniques are quite effective, and ride the wave of better network architectures

**설명**
- 강의 전반부(p.3–60) 전체의 정리야. 줄마다 앞 내용과 이어져.
  - **priors and hints**: p.5–17의 두 정보원이야. prior = 사진과 상관없이 믿는 것(크기, 모양, 경계 …), hint = 사진마다 받는 간접 supervision(image label, box, scribble …).
  - **explore new information sources**: web 사진(p.43–52), box(p.53–60), saliency(p.33–34)처럼 새 정보원을 끌어와.
  - **transfer learning (and webly supervision)**: Web-SEC는 web에서 배운 물체 모양을 target으로 옮겨(p.47). p.33의 "Use transfer learning"도 같은 얘기야.
  - **ride the wave of better network architectures**: 방법이 같아도 backbone이 좋아지면 성능이 올라 (예: p.51 Ours-VGG16 58.8 → Ours-Res50 63.0).
- 이 뒤(p.62–)는 목차 밖의 최근 논문(Beyond CAM)이야.
