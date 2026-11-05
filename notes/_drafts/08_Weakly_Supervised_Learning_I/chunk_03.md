<!-- DRAFT | lecture: 08_Weakly_Supervised_Learning_I | chunk: 03 | pages 22–32 -->
<!-- VERIFIED | fixes: 3 | verifier: opus -->

## Re-Localization and Re-Training: Relaxing max operator (p.22)

슬라이드 텍스트:
- More robust optimization: Relaxing max operator
- Hedge your bets on multiple proposals
- Re-train
- For positive images

$$\log \sum_b \exp A(x_b^+) > \Delta$$

- For negative images

$$\log \sum_b \exp A(x_b^-) < -\Delta$$

- Less sensitive to initialization
- Leverages multiple positive samples during training
- Relax hard max into soft-max (log-sum-exp) to learn from multiple high-scoring proposals instead of a single proposal.
- 그림: 맨 위 = 동물이 여러 마리 있는 사진 2장의 후보 box. 가운데 "Max" = 고른 box 하나 (노랑). 맨 아래 "Soft-max" = 동물마다 강조된 box 여러 개 (주황·빨강)
- [Bilen BMVC 14]

**설명**
- **문제**: hard max는 window 하나에 전부 걸어. 그 하나가 틀리면 학습 전체가 틀린 쪽으로 가.
- **핵심**: max를 **log-sum-exp**로 바꿔서 점수 높은 window 여러 개에 나눠 걸어. 이게 "Hedge your bets"야.
- 기호: $x_b^+$는 positive image의 proposal $b$의 feature, $x_b^-$는 negative image의 것이야.
- p.21과 비교하면 $\max_b A(x_b)$ 자리에 $\log\sum_b\exp A(x_b)$가 들어갔어. margin 구조는 그대로야.
- **좋아지는 점 두 개**
  1. Less sensitive to initialization: 시작 box 하나에 덜 묶여.
  2. Leverages multiple positive samples during training: p.20의 "Only one positive instance per image"가 풀려.

> 보충 (슬라이드엔 유도 없음): log-sum-exp는 max의 부드러운 근사야. window가 $B$개면 $\max_b a_b \le \log\sum_b e^{a_b} \le \max_b a_b + \log B$야.
>
> 왜 여러 window가 학습되나? $a_b$로 미분하면 $\partial/\partial a_b \log\sum_{b'} e^{a_{b'}} = \text{softmax}(a)_b$야. hard max는 gradient가 최고 window 하나로만 가는데, log-sum-exp는 $e^{a_b}$ 비율(= softmax 비중)대로 나눠 가져.
>
> 예시 (설명용 숫자): 점수가 3, 3, 0이면 max = 3이고 gradient 비중은 1, 0, 0이야 (동점이면 하나를 골라). log-sum-exp = 약 3.72고 gradient 비중은 약 0.49, 0.49, 0.02야.

> 보충 (RL 비유, 슬라이드 밖 = 시험 범위 아님): soft Q-learning의 soft value $V = \alpha\log\sum_a \exp(Q/\alpha)$와 같은 연산이야. hard max 대신 부드러운 max를 쓰는 동기도 닮았어.

---

## Re-Localization and Re-Training: Self-paced learning (p.23)

슬라이드 텍스트:
- More robust optimization: Self-paced learning
- In conventional WSOD, the initial localization results are crucial.
- If an incorrect bounding box is selected initially,
  - → the detector is trained using that box,
  - → leading to a vicious cycle (locking) where that same box is repeatedly selected.
- Idea: Inspired from Curriculum Learning, Instead of training on all samples—including difficult ones—from the start, let's train progressively, beginning with easy samples.
- Easiness measures for WSOD:
  - Selection of samples via confidence of max scoring window
  - Selection of window space by allowing smaller windows
  - Selection of samples via intercategory competition
- Learn from easy images first, then progressively incorporate harder samples as localization becomes more reliable.
- 그림 위: 물체가 크게 잘 보이는 오토바이 사진과 복잡한 사진
- 그림 가운데: (d) iter 0, (e) iter 1, (f) iter 5. 초록 box가 사진 전체에서 시작해 점점 작아지며 사람 주변으로 좁혀져. 밑에 Image credit: [Bilen IJCV 14] (Object and Action Classification with Latent Window Parameters, IJCV 2014)
- 그림 아래: 왼쪽은 초록 TV box + 차 일부에 빨간 Car box, 오른쪽은 차 전체에 초록 Car box
- Image credit: [Sangineto PAMI 17]

**설명**
- **locking의 고리**: 처음에 틀린 box를 고름 → 그 box로 학습 → 같은 box를 또 고름. 이게 "vicious cycle"이야.
- **curriculum learning** = 쉬운 샘플부터 어려운 샘플 순으로 학습하는 방식이야.
- **self-paced learning** = 그 순서를 사람이 정하지 않고, 학습 중인 model의 상태로 정하는 방식이야. 슬라이드는 "Inspired from Curriculum Learning"이라고만 적었어.
- **easiness 기준 세 가지**
  1. 최고 점수 window의 confidence가 높은 샘플부터 써.
  2. window 공간을 넓혀 가. 작은 window를 점점 허용해. 가운데 그림의 iter 0 → 5가 이 모습으로 읽혀.
  3. intercategory competition으로 샘플을 골라. 슬라이드엔 방법 설명이 없어.

> 보충 (슬라이드 밖 = 시험 범위 아님): L07의 FreeMatch에서 threshold를 점점 올리는 것도 "처음엔 쉽게, 나중엔 엄격하게"라는 curriculum 감각이 닮았어. 메커니즘은 달라.

**시험 관점**
- 순서는 **easy → hard**야. "starts from difficult samples"는 틀린 문장이야.
- 해결하는 문제는 initialization 민감성과 locking이야.

---

## Re-Localization and Re-Training: Multifold MIL (p.24)

슬라이드 텍스트:
- More robust re-localization: Multifold MIL
  - Problem: Detector overfits into the given proposal
  - Solution: Train using positive examples in all folds but k, and all negative examples
- Idea: the detector performing localization for an image must be one that has not been trained on that image.
- -> Use cross-validation style training: localize each image with a detector that has never seen that image during training.
- 그림: 사진 2장과 Object detector 1, 2. 파란 실선 Re-train은 위 사진 → detector 2, 아래 사진 → detector 1로 엇갈려 가. 주황 점선 Re-localize는 detector 1 → 위 사진, detector 2 → 아래 사진으로 돌아와.
- Image credit: [Sangineto PAMI 17] ← slide credit이 틀린 듯. Multifold MIL은 Cinbis et al., TPAMI 2017이야.

**설명**
- **문제**: detector가 자기가 학습한 window를 외워. 그래서 다음 re-localize 때 같은 window가 또 최고 점수를 받아. 이게 locking이야.
- **fold** = data를 나눈 조각이야. **cross-validation** = 조각 하나를 빼고 학습한 뒤 뺀 조각으로 평가하는 방식이야.
- **핵심**: 어떤 사진의 box를 고르는 detector는 **그 사진으로 학습한 적이 없어야** 해.
- **어떻게** (fold $k$에 대해)
  1. positive 사진을 fold로 나눠.
  2. fold $k$를 뺀 나머지 positive 예시와 **모든 negative 예시**로 detector를 학습해.
  3. 그 detector로 fold $k$의 사진을 re-localize해.
- 그림을 읽으면, 위 사진은 detector 2를 학습시키고 box는 detector 1에게 받아. 자기로 학습한 detector에게는 box를 받지 않아.

> 보충 (RL 비유, 슬라이드 밖 = 시험 범위 아님): Double Q-learning이 "고르는 쪽과 평가하는 쪽을 나눠서 자기 편향을 끊는" 동기와 닮았어. L07 DivideMix의 두 network도 같은 동기야. 메커니즘은 각각 달라.

**시험 관점**
- 함정 지문: "the detector localizes an image **it was trained on**"은 틀렸어. 반대야.
- 빠지는 건 **positive** fold $k$뿐이야. negative는 전부 써.

---

## Re-Localization and Re-Training: Pairwise similarity (p.25)

슬라이드 텍스트:
- Pairwise similarity
  - Similarity between selected windows across positive images
    - Less overfitting
    - Expensive to optimize
      - All positive images must be connected (N² similarities)
    - Ignores intra-class variation
      - The assumption that all objects must be similar is unrealistic.
      - However, since this method assumes that similar = good, it does not sufficiently reflect diversity.
- Enforce consistency across positive images by jointly selecting similar object proposals, reducing localization ambiguity but increasing computational cost and limiting intra-class diversity.
- 그림: fully connected CRF. nodes: images / states: windows / unary: objectness scores / pairwise: similarity between windows
- [Chum CVPR 07, Deselaers ECCV 10, Siva ICCV 11, Bilen CVPR 15]

**설명**
- **핵심**: 사진마다 따로 box를 고르지 말고, positive 사진들에서 **서로 닮은 box를 같이** 골라.
- 진짜 물체라면 여러 사진에서 비슷하게 생겼을 거라는 가정이야.
- **CRF(Conditional Random Field)** = 변수들을 graph로 잇고, 각 변수의 점수와 이웃 사이 점수의 합으로 답을 고르는 model이야.
  - 그림에서 node = 사진, state = 그 사진에서 고를 window야.
  - **unary** = 변수 하나의 점수야. 여기선 objectness score야.
  - **pairwise** = 두 변수 사이의 점수야. 여기선 두 window의 유사도야.
  - **fully connected** = 모든 사진 쌍이 이어졌다는 뜻이야. 그래서 $N^2$ 유사도가 필요해.
- **장점**: Less overfitting. 자기 점수만 보지 않고 다른 사진과의 일치를 보니까.
- **단점 두 개**
  1. Expensive to optimize: 모든 positive 사진을 이어야 해 ($N^2$).
  2. Ignores intra-class variation: "similar = good"이라고 가정해서, 같은 class의 다양한 모습을 놓쳐.

**시험 관점**
- 장점 1개 + 단점 2개를 묶어서 외워. 단점이 p.26의 동기야.

---

## Re-Localization and Re-Training: Sub-categories / Exemplars (p.26)

슬라이드 텍스트:
- Pairwise similarity
  - Sub-categories
    - Clustering via probabilistic latent Semantic Analysis (pLSA)
      - (+) Modeling intra-class variations
      - (-) Sensitive to number of clusters
  - Exemplars
    - Pick representative samples
    - (+) No need to set number of clusters
- Instead of enforcing similarity across all positive images, model intra-class variation using either sub-category clustering or representative exemplars.
- 그림: 맨 위 비행기 사진 여러 장. 아래는 전투기 box 묶음(주황)과 여객기 box 묶음(보라)이 각각 가운데 대표 샘플로 이어지고, 대표들이 회색 타원 "clusters" 안에 있음
- [Wang ECCV 14, Chum CVPR 07, Bilen CVPR 15]

**설명**
- p.25는 모든 물체가 닮았다고 가정해서 다양성을 놓쳤어. 여기선 "**비슷한 것끼리만** 비슷하면 된다"로 고쳐.
- 그림으로 보면 airplane class 안에 전투기 무리와 여객기 무리가 있어.
- **Sub-categories**: class 안을 cluster로 나눠.
  - **pLSA** = 원래 문서의 숨은 주제를 찾는 확률 model이야. 여기선 clustering에 썼어.
  - 장점: intra-class variation을 model링해. 단점: **cluster 개수**에 민감해.
- **Exemplars**: 대표 샘플을 골라서 그 주변으로 묶어.
  - 장점: cluster 개수를 정할 필요가 없어.

**시험 관점**
- 짝 맞추기: Sub-categories ↔ "Sensitive to number of clusters", Exemplars ↔ "No need to set number of clusters".

---

## Re-Localization and Re-Training: Additional cues — context (p.27)

슬라이드 텍스트:
- Additional cues: context
  - Background provides contextual cues for recognition
  - Better separation of foreground and background
  - Additive: select a ROI that is semantically compatible with its context
  - Contrastive: select a ROI that is outstanding from its context
- Context provides valuable cues beyond the object appearance. It can either reinforce semantically compatible objects (additive) or help distinguish foreground from background (contrastive).
- 그림 왼쪽: 자동차 사진에서 FG(안쪽 box)와 BG(바깥) 구분 [Russakovsky ECCV 12]
- 그림 오른쪽: ROI extraction 뒤 두 model [Kantorov ECCV 16]
  - context-aware additive model: "person in ROI?" ⊕ "person in context?" → person?
  - context-aware contrastive model: "person in ROI?" ⊖ "person in context?" → "person in ROI but not in context?"

**설명**
- **ROI(Region of Interest)** = 지금 보고 있는 후보 영역이야. **context** = 그 영역 바깥 주변이야.
- **foreground** = 물체, **background** = 배경이야.
- 물체 자체의 생김새(appearance) 말고 주변 정보도 쓰자는 생각이야. 쓰는 방식이 두 가지야.
  - **Additive**: 주변과 **어울리는** ROI를 골라. ROI 점수와 context 점수를 더해.
  - **Contrastive**: 주변에서 **튀는** ROI를 골라. ROI 안엔 있고 바깥엔 없어야 점수가 높아.

**시험 관점**
- Additive = semantically **compatible**, Contrastive = **outstanding**. 뒤바꾸기 쉬운 짝이야.
- 여기서 contrastive는 L03–L05의 contrastive learning(InfoNCE)과 **다른 말**이야. ROI와 context의 차이를 본다는 뜻이야.

---

## Re-Localization and Re-Training: Additional cues — Objectness (p.28)

슬라이드 텍스트:
- Additional cues: Objectness
  - Quantify how likely a window is to contain an object of any class
    - Generates a small set of accurate object proposals
    - Steers re-localization towards objects and away from background
    - Pushes towards whole objects instead of sub-regions

$$\arg\max_b\ \lambda A(x_b) + (1-\lambda)\,\text{Obj}(b)$$

- Obj(b): proposal의 objectness ← 슬라이드에 한국어 조사가 섞인 표기 (원문 유지)
  - EdgeBoxes
  - Selective Search
  - BING
- Objectness provides a class-independent prior that guides WSOD toward complete objects rather than discriminative object parts or background regions.
- 그림: 오토바이가 있는 사진에 색이 다른 box 여러 개 + 0–1 color bar. 오토바이 전체 box는 진한 빨강(높음), 오토바이 일부 box는 주황, 하늘 box는 남색(낮음)에 가까워
- Slide credit: Vittorio Ferrari

**설명**
- **objectness** = 어떤 window에 **class와 상관없이** 물체가 들어 있을 정도야.
- **prior** = 학습 data를 보기 전에 넣어 주는 사전 지식이야.
- 식을 풀면 p.20의 re-localize에 항 하나를 더한 거야.
  - $\lambda A(x_b)$ = 찾는 class처럼 생긴 정도야.
  - $(1-\lambda)\text{Obj}(b)$ = 물체다운 정도야.
  - $\lambda = 1$이면 p.20과 같고, $\lambda$가 작을수록 objectness를 더 믿어.
- **왜 돼?** class 점수는 "가장 class다운 부위"를 좋아해. objectness는 물체 전체를 감싼 box를 좋아해. 그래서 p.11의 parts 문제와 p.12의 context 문제를 둘 다 완화해.
- 예시 (설명용 숫자): $\lambda = 0.5$, window 1은 $A = 0.9$, Obj $= 0.2$라 0.55고, window 2는 $A = 0.7$, Obj $= 0.8$이라 0.75야. 그러면 window 2가 뽑혀.

**시험 관점**
- objectness는 **class-independent**야. "class-specific"이면 틀린 문장이야.
- $\lambda$가 앞에 붙는 항은 $A$, $(1-\lambda)$가 붙는 항은 Obj야.

---

## Re-Localization and Re-Training: Priors — Equivariance (p.29)

슬라이드 텍스트:
- Priors: Equivariance
  - The predicted output should be equivariant to image transformations
- 그림: 윗줄 사진과 좌우가 뒤집힌 아랫줄 사진, 각각의 Before / After heat map
- [Bilen BMVC 14]

**설명**
- **equivariance** = 입력을 변환하면 출력도 **같이** 변환돼야 한다는 성질이야. L02 p.71–79에서 봤어.
  - invariance: $f(t(x)) = f(x)$ (출력이 그대로)
  - equivariance: $f(t(x)) = \phi(t, f(x))$ (출력이 따라 변해)
- detection에선 사진을 뒤집으면 예측 위치도 뒤집혀야 해. box는 위치를 내는 출력이니까 equivariance가 맞아.
- 그림을 읽으면, Before에선 같은 차의 강도가 원본과 뒤집은 사진에서 꽤 달라. After에선 차이가 줄어.

**시험 관점**
- "invariant"로 바꾼 함정을 조심해. 슬라이드 표현은 **equivariant**야.

---

## Re-Localization and Re-Training: Priors — Scale (p.30)

슬라이드 텍스트:
- Priors: Scale
  - Curriculum learning (bigger objects down to smaller ones)
  - Weight object proposals according to estimated size
  - Requires training a size estimator from a small set
- introduces an object-size prior, encouraging the detector to localize large objects first and progressively learn smaller ones, thereby reducing localization errors caused by object parts.
- 그림: 위 RE-TRAINING = 사진들을 "Size Order"로 정렬. 아래 RE-LOCALIZATION = proposal들에 "Size Weighting"을 적용. 가운데 곡선은 세로축 $w$, 가로축 $S_p$이고 $S_e$에서 가장 높아
- [Shi ECCV 16]

**설명**
- **어떻게**
  1. **Re-training**: 사진을 물체 크기 순서로 정렬해서 **큰 물체부터** 학습해 (curriculum).
  2. **Re-localization**: 추정한 크기에 가까운 proposal에 weight를 더 줘.
- 곡선을 읽으면 $S_p$는 proposal 크기, $S_e$는 추정된 크기로 보여 ← 슬라이드에 기호 정의는 없어.
- **왜 돼?** 크기를 알면 물체의 일부만 감싼 작은 box가 밀려나. 슬라이드도 "reducing localization errors caused by object parts"라고 적었어.
- **비용**: size estimator를 학습할 작은 set이 따로 필요해.

**시험 관점**
- 순서는 **bigger → smaller**야. 뒤집은 지문을 조심해.
- p.23(easy → hard)과 p.30(big → small)은 둘 다 curriculum이야.

---

## Re-Localization and Re-Training: Priors — Motion (p.31)

슬라이드 텍스트:
- Priors: Motion
  - 1. extracts candidate spatio-temporal tubes based on motion segmentation
  - 2. selects one tube per video jointly over all videos
  - 3. uses objectness to reduce object candidates
  - 4. applies a domain adaptation method to address the video/image gap
  - (+) Motion cues for object boundaries
  - (-) Noisy data
- Motion provides strong temporal cues for object localization. By extracting motion-based object tubes from videos and transferring them to image detectors, WSOD can obtain much more reliable pseudo bounding boxes than using images alone.
- 그림: Training Videos → Candidate Tubes → Joint-Selection of 1 Tube per Video → Sample Frames from Tubes → Sampled Frames → Train an Object Detector → Object Detector
- Image credit: [Prest CVPR 12]

**설명**
- **spatio-temporal tube** = 동영상에서 시간을 따라 이어지는 box 묶음이야.
- **motion segmentation** = 움직임을 기준으로 영역을 나누는 거야.
- **domain adaptation** = 분포가 다른 data 사이의 차이를 메우는 방법이야. 여기선 video와 image의 차이야.
- **왜 돼?** 움직이는 물체는 배경과 따로 움직여. 그래서 사진 한 장으로는 안 보이는 물체 경계가 보여.
- 고른 tube의 frame이 **pseudo bounding box** 역할을 해. L07의 pseudo label과 같은 생각이야. 모델이 만든 box를 정답처럼 쓰는 거야.

**시험 관점**
- 4단계 순서: tube 추출 → 영상마다 tube 1개 선택 → objectness로 후보 축소 → domain adaptation.
- "one tube per video"가 숫자 함정이야.

---

## Standard MIL Pipeline (p.32)

슬라이드 텍스트:
- Two step alternating optimization
  - Re-train
  - Re-localize
- Challenges
  - Sensitive to initialization (local minimum)
  - Overfitting (locking) to predicted windows
- 그림: p.16과 같음

**설명**
- p.16과 같은 슬라이드야. 고전 MIL 파트를 닫는 표시로 다시 나왔어.
- 지금까지의 개선을 두 Challenge와 pipeline 단계로 묶으면 이래.

| 개선 | 페이지 | 건드리는 곳 | 해결하려는 것 |
|---|---|---|---|
| Song ICML 14 graph | p.18 | Initialize | 나쁜 시작 box |
| log-sum-exp | p.22 | Re-train | 시작 민감성, positive 하나 제한 |
| Self-paced | p.23 | 샘플·window 순서 | locking, 시작 민감성 |
| Multifold | p.24 | Re-localize | overfitting (locking) |
| Pairwise / Sub-categories / Exemplars | p.25–26 | Re-localize | overfitting, 다양성 |
| Context | p.27 | Re-localize | 전경/배경 구분 |
| Objectness | p.28 | Re-localize | part, background |
| Equivariance / Scale / Motion priors | p.29–31 | 사전 지식 | 잘못된 위치·크기 |

- 표의 "건드리는 곳"은 슬라이드 설명에서 읽어 낸 분류야. 슬라이드 제목은 전부 "Re-Localization and Re-Training"으로 묶여 있어.
- 다음 섹션은 이 고리를 network 하나의 학습으로 합치는 쪽이야.
