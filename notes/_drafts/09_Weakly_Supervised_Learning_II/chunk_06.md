<!-- DRAFT | lecture: 09_Weakly_Supervised_Learning_II | chunk: 06 | pages 75–96 -->
<!-- VERIFIED | fixes: 15 | verifier: sonnet -->

> 이 청크는 목차(p.2) 밖에 붙은 세미나 논문 두 편과 다음 주 예고야.
> - p.75–85: Liu et al., "Learning from the Web: Language Drives Weakly-Supervised Incremental Learning for Semantic Segmentation", ECCV 2024 (p.76 cite 그대로, 웹으로 확인함. arXiv 2407.13363)
> - p.86–95: WISH. **슬라이드엔 cite가 없어.** 웹으로 확인한 원 논문은 Kweon & Yoon, "WISH: Weakly Supervised Instance Segmentation using Heterogeneous Labels", CVPR 2025야.
> - p.96: Next Week.. Active learning

---

## Section question: Can a segmentation model learn new classes from the web without storing its original training images? (p.75)

슬라이드 텍스트:
- Can a segmentation model learn new classes from the web without storing its original training images?
- 그림: 남색 배경에 질문 한 줄만 있는 section 구분 슬라이드

**설명**
- 이 질문에 단어가 세 개 들어 있어. 이 세 개가 논문의 설정 전부야.
  1. **learn new classes**: 이미 배운 model에 class를 **더** 배우게 해. 이게 **incremental learning**이야.
  2. **from the web**: 새 class의 학습 사진을 사람이 모은 dataset이 아니라 **web 검색**으로 얻어.
  3. **without storing its original training images**: 처음 학습에 쓴 사진은 **안 남겨 둬**. 그래서 옛 class를 복습할 사진이 없어.
- **incremental learning** = model을 한 번 학습하고 끝내지 않고, step마다 새 class를 추가로 배우는 설정이야. 예: step 0에서 class 1–10을 배우고, step 1에서 class 11–20을 더 배워.

---

## (title) Learning from the Web (p.76)

슬라이드 텍스트:
- 논문 첫 페이지: "Learning from the Web: Language Drives Weakly-Supervised Incremental Learning for Semantic Segmentation"
- Chang Liu¹, Giulia Rizzoli², Pietro Zanuttigh², Fu Li¹, and Yi Niu¹ (¹ School of Artificial Intelligence, Xidian University, China / ² Department of Information Engineering, University of Padova, Italy)
- Liu et.al., Learning from the Web: Language Drives Weakly-Supervised Incremental Learning for Semantic Segmentation, ECCV 2024

**설명**
- 제목을 쪼개 읽으면 설정이 보여.
  - **Weakly-Supervised**: 새 class엔 pixel label 없이 image-level label만 있어.
  - **Incremental Learning**: class를 step마다 추가로 배워.
  - **for Semantic Segmentation**: 최종 출력은 pixel마다 class를 칠한 map이야.
  - **Language Drives**: web 검색어, label, 필터, 복습 전부를 **언어(class 이름과 caption)**로 해 (p.80 마지막 줄).
- 이 설정을 이 분야에선 줄여서 **WILSS**라고 불러 (p.79 "Conventional WILSS"). 제목의 Weakly-supervised Incremental Learning for Semantic Segmentation이야.

---

## Can the Web Replace the Training Dataset? (p.77)

슬라이드 텍스트:
- Can a segmentation model learn **new classes** from the **web without storing** its original training images?
- Conventional web supervision
  - Web images supplement a curated training dataset.
  - Target classes are fixed.
  - Previous training images remain available.
- New scenario
  - Only the initial model is retained.
  - **Original training** images are **no** longer **accessible**.
  - **New** and **old** classes must be learned **from web** images.
- 그림 (오른쪽 위) "Conventional Web Supervision": Curated Dataset + Web Images → Segmentation Model, 밑에 "Web data supplements curated data"
- 그림 (오른쪽 아래) "Web-Based Incremental Learning": Initial Model + Class Name + Open Web → (사진들) → Updated Segmentation Model, 밑에 "No original training images" (dataset 아이콘에 빨간 X)

**설명**
- **curated dataset** = 사람이 직접 고르고 정리한 dataset이야 (PASCAL VOC 같은 것). 사진이 깨끗하고 분포가 고르지.
- 비교 포인트는 두 가지야.

| | Conventional web supervision | New scenario (이 논문) |
|---|---|---|
| web의 역할 | curated dataset을 **보충**(supplement) | curated dataset을 **대체** |
| class | 고정 (Target classes are fixed) | step마다 **새 class 추가** |
| 옛 학습 사진 | 남아 있음 | **없음**. initial model만 남아 |
| 무엇을 web에서 배우나 | (보충분) | **새 class + 옛 class 둘 다** |

- 앞 §3 Webly(Shen CVPR 2018, p.43–52)는 "Conventional" 쪽이야. web 사진이 target dataset 학습을 **도와** 줬어. 이 논문은 web 사진이 dataset **자리를 차지**해.

> 보충 (RL 비유, 슬라이드 밖 = 시험 범위 아님): 옛 replay buffer를 지운 채로 새 task를 배우는 continual RL과 같은 구도야. 남은 건 옛 policy(initial model)뿐이야.

---

## Can the Web Replace the Training Dataset? — Three challenges (p.78)

슬라이드 텍스트:
- Three challenges
  - 1. Web images have a **different distribution**.
  - 2. Search queries provide **noisy** single-class labels.
  - 3. Learning new classes causes **forgetting** of old classes.
- The web must provide both new knowledge and memory of old knowledge.

**설명**
- 세 문제와 이 논문의 처방을 미리 짝지으면 이래 (처방은 p.80–83).

| 문제 | 무슨 뜻 | 처방 |
|---|---|---|
| 1. different distribution | web 사진은 dataset 사진과 생김새 분포가 달라 (상품 사진, 그림, 흰 배경 등) | **Domain filtering**: dataset 같은 사진만 남김 (p.81) |
| 2. noisy single-class labels | 검색어 "dog"로 받은 사진엔 label이 dog 하나뿐이야. 같이 찍힌 cat·sofa는 label에 없고, 검색 결과가 틀릴 수도 있어 | **Caption → multi-label** (p.81), **Caption filtering** (p.80, 83) |
| 3. forgetting | 새 class만 보고 학습하면 옛 class를 잊어 | **Caption 기반 rehearsal**: 옛 사진 대신 caption을 저장하고 web에서 비슷한 사진을 다시 받아 복습 (p.82–83) |

- **catastrophic forgetting** = 새 data로만 학습하면 옛 class 성능이 급격히 무너지는 현상이야. 슬라이드 표현은 "Learning new classes causes forgetting of old classes."
- 핵심 문장: "The web must provide both **new knowledge** and **memory of old knowledge**." web이 새 class 학습 재료이자 옛 class 기억 장치여야 한다는 뜻이야.

> 보충 (RL 비유, 시험 범위 아님): forgetting ↔ 새 task로 fine-tuning한 policy가 옛 task에서 무너지는 policy drift / catastrophic forgetting in continual RL.

---

## From Curated Data to Open-Web Learning (p.79)

슬라이드 텍스트:
- 그림 (a) WILSS: Step t, Dataset → Data Sample → network → $\ell_{cls}$ ← "Dog"
- 그림 (b) Ours: Step t, Web Source → Web Image (개 사진) + Rehearsed Web Image (비행기·새·소 사진, 점선) → network → $\ell_{seg}$ ← Pseudo Labeling (색칠된 mask), $\ell_{cls}$ ← "Dog"
- Conventional WILSS
  - A new class is introduced at step t.
  - Curated dataset images are available.
  - Only image-level labels are given for new classes.
  - The localizer generates pseudo segmentation labels.
- Proposed Setting
  - New images are downloaded from the **web**.
  - Old-class web images are also retrieved for rehearsal.
  - **Caption** models provide richer image-level supervision.
  - **No original** image is **stored** after the initial step.
- Web images replace curated incremental data, while captions replace reliable manual image labels.

**설명**
- **(a) Conventional WILSS**: step t에 새 class(예: Dog)가 들어와. 사진은 **curated dataset**에서 오고, label은 사람이 단 **image-level label**("Dog")이야. pixel label은 없으니까 **localizer**(image label로 학습해서 물체 위치를 대략 칠해 주는 부분)가 pseudo segmentation label을 만들어.
- **(b) Ours**: 사진은 **web**에서 받아. 새 class 사진뿐 아니라 옛 class 사진(Rehearsed Web Image)도 web에서 다시 받아 와. label은 사람 대신 **caption model**이 줘.
- 그림 차이를 한 줄로: (a)는 loss가 $\ell_{cls}$ 하나만 그려져 있고, (b)는 $\ell_{cls}$ + $\ell_{seg}$(pseudo label과 비교)가 있어.
- 마지막 줄이 이 논문의 두 가지 바꿔치기야.
  - curated incremental data → **web images**
  - reliable manual image labels → **captions**
- **caption** = 사진을 보고 만든 설명 문장이야. 예: "a dog and a cat laying on a couch". **caption model** = 사진을 넣으면 caption을 써 주는 model이야.

> 보충: $\ell_{cls}$ = image-level classification loss (어떤 class가 있나), $\ell_{seg}$ = pixel 단위 segmentation loss야. 슬라이드엔 기호만 있고 정의는 없어.

---

## Language-Driven Web Learning (p.80)

슬라이드 텍스트:
- ① **Retrieve** web images
  - Query the web using new class names.
  - Retrieve rehearsal images using stored captions.
- ② **Filter noisy** web images
  - Domain filtering removes out-of-distribution images.
  - Caption filtering removes semantically irrelevant images.
- ③ **Generate supervision**
  - Captions are converted into multi-label supervision.
  - The previous model produces pixel-level pseudo-labels.
- ④ **Update** the segmentation **model**
  - Learn new classes from web images.
  - Rehearse old classes without storing original images.
- Web retrieval → Filtering → Caption labeling → Pseudo labeling → Incremental training
- Language is used for querying, labeling, filtering, and rehearsal.
- 그림 (framework):
  - 위 (주황 영역) Old classes: airplane, bird, cat. "Data at initial step"(비행기·백로·고양이 사진) → **Caption Model** → caption 목록 ("an airplane taking off from an airport", "a great egret standing on a rock", …, "a white and black cat yawning") → **Caption Filtering** → "Web images at initial step"(다시 받은 비행기·새·고양이 사진)
  - 아래 (파란 영역) New classes: dog, sofa. "Class of queried 'Dog'" → **Web-crawler** → **Domain Filtering** → "Web images at step t"(개와 고양이가 소파에 있는 사진) → caption "a dog and a cat laying on a couch" → "cat, dog, sofa" → $\ell_{cls}$
  - 오른쪽 (회색 영역): 범례 Frozen(파랑) / Trainable(주황). **Model at Step t−1**(frozen) → Pseudo label(비행기·새·고양이 mask, ⋮) → $\ell_{seg}$. **Model at Step t**(trainable) 안에 **Localizer** → 색칠된 mask, $\ell_{seg}$·$\ell_{cls}$와 연결

**설명**
- **out-of-distribution (OOD)** = 학습 dataset의 분포 밖에 있는 사진이야. 예: VOC는 일상 사진인데 web에선 흰 배경 상품 사진, 만화가 섞여 와.
- 네 단계를 그림과 맞춰 읽으면 이래.
  1. **Retrieve**: 새 class는 **class 이름**("Dog")으로 검색해. 옛 class는 저장해 둔 **caption**으로 검색해.
  2. **Filter**: 새 class 사진은 **Domain filtering**(dataset 분포와 다른 사진 제거), 옛 class 복습 사진은 **Caption filtering**(뜻이 다른 사진 제거)을 거쳐 (그림 배치 기준).
  3. **Generate supervision**: caption 단어를 **여러 개의 image-level label**로 바꿔 ("couch" → sofa까지 포함해서 "cat, dog, sofa"). pixel 단위 pseudo label은 **이전 step model**(frozen)이 만들어 줘.
  4. **Update**: step t model을 학습해. 새 class는 web 사진에서, 옛 class는 web에서 다시 받은 사진으로 복습해.
- 마지막 줄이 제목의 "Language Drives"야: 언어가 **querying, labeling, filtering, rehearsal** 네 군데 다 쓰여.
- 그림에서 눈여겨볼 것: caption "a dog and a cat laying on a **couch**"에서 label "**sofa**"가 나와. caption 단어를 class 이름으로 옮기는 단계가 있다는 뜻이야.
- **frozen** = 학습 중 weight를 안 바꾸는 model이야. step t−1 model은 frozen이라 target(pseudo label)만 만들어.

> 보충 (그림 해석): step t model 안의 Localizer는 caption label($\ell_{cls}$)로 학습되고, 그 출력이 새 class의 pseudo mask 역할을 하는 것으로 그려져 있어. 옛 class pseudo label은 step t−1 model이 만들어. 슬라이드 본문엔 "The previous model produces pixel-level pseudo-labels."까지만 적혀 있어.

> 보충 (RL 비유, 시험 범위 아님): frozen 옛 model이 target을 만들어 주는 건 **policy distillation**과 닮았어. 옛 model의 출력을 (hard) pseudo label target으로 써서 잊지 않게 하는 self-distillation 느낌이야.

---

## Learning New Classes from the Web (p.81)

슬라이드 텍스트:
- 1. Select Dataset-Like Web Images
  - Query the web using a new class name.
  - Remove web images with a **large domain gap**.
  - **Keep** images that **resemble** the original dataset.
- 2. Recover Multi-Label Supervision
  - A search query provides only one label.
  - A caption describes all visible objects.
  - **Caption** words are **converted** into **image-level labels**.
- 식: $(p_{ds}, p_{web}) = (p(\mathbf{x}\in\mathcal{X}),\ p(\mathbf{x}\in\mathcal{X}^{web})) = M_D(|\mathcal{F}(\mathbf{x})|)$
- 식: $p_{ds}/p_{web} > 1$
- 그림: p.80 framework의 아래쪽 부분 확대 (Class of queried "Dog" → Web-crawler → Domain Filtering → Web images at step t, "a dog and a cat laying on a couch")
- The class name retrieves images; the caption determines their labels.

**설명**
- **domain gap** = 두 사진 집합의 생김새 분포 차이야.
- **1. Domain filtering**: 사진 $\mathbf{x}$마다 "원래 dataset $\mathcal{X}$ 사진 같을 확률" $p_{ds}$와 "web 사진 $\mathcal{X}^{web}$ 같을 확률" $p_{web}$를 판별기 $M_D$로 구해. $p_{ds}/p_{web} > 1$, 즉 **dataset 쪽일 확률이 더 크면 남겨**.
  - 기호 풀이: $\mathcal{X}$ = 원래 dataset 사진 집합, $\mathcal{X}^{web}$ = web 사진 집합. (논문 보충, 슬라이드엔 기호만) $M_D$ = domain을 맞히는 판별기(discriminator), $|\mathcal{F}(\mathbf{x})|$ = 사진의 Fourier 변환 크기(amplitude)야.
  - 왜 원래 사진이 없는데 dataset과 비교해? 판별기는 **initial step**에서 원래 사진이 있을 때 학습해 두는 것으로 보여 (추정, 논문 확인은 안 함, 시험 범위 밖).
- **2. Multi-label 복구**: 검색어 "Dog"는 label을 **하나**만 줘. 그런데 사진엔 고양이와 소파도 있어. caption은 "a dog and a cat laying on a couch"처럼 **보이는 물체를 다** 말해 줘. 그래서 caption 단어를 image-level label 여러 개(cat, dog, sofa)로 바꿔.
- 마지막 줄이 역할 분담이야: "The class name **retrieves** images; the caption **determines** their labels." 검색은 class 이름이, label은 caption이 맡아.

> 보충 (논문, 슬라이드엔 기호만): $M_D$는 논문에서 Fourier 기반 domain discriminator야. 사진의 Fourier amplitude spectrum은 색감·질감 같은 "스타일"을 많이 담아서 domain 차이를 잡기 좋다는 게 직관이야. 슬라이드엔 $\mathcal{F}$, $M_D$의 정의가 없으니 시험에선 "domain discriminator로 dataset 같은 web 사진만 남긴다" 수준이면 충분해.

---

## The Web as External Memory — Store Captions, Not Images (p.82)

슬라이드 텍스트:
- Can Captions Replace Stored Training Images?
- **Store Captions**, Not Images
  - Original training images are unavailable after the initial step.
  - The method stores their captions instead.
  - Stored captions are used to retrieve rehearsal images from the web.
- 식: $\mathcal{X}_r^{web} = \{\mathbf{x} = \mathcal{D}^{web}(q') \mid q' = M_{CAP}(\mathbf{x}) : \mathbf{x}\in\mathcal{X}\}$
- Old image → Stored caption q' → Web search → Rehearsal image
- q': caption of an old training image

**설명**
- **rehearsal** = 새 class를 배울 때 옛 class 사진을 같이 섞어 보여 줘서 잊지 않게 하는 거야. 보통은 옛 사진 일부를 **memory**에 저장해 둬야 해.
- 이 논문은 옛 사진 대신 **caption을 저장**해. 순서는 이래.
  1. initial step에서 옛 사진 $\mathbf{x}\in\mathcal{X}$마다 caption model $M_{CAP}$로 caption $q'$를 만들어 저장해.
  2. 옛 사진은 버려.
  3. 나중 step에서 $q'$로 web을 검색해($\mathcal{D}^{web}(q')$) 비슷한 사진을 받아. 이게 rehearsal 사진 집합 $\mathcal{X}_r^{web}$야.
- 식의 기호: $\mathcal{D}^{web}(\cdot)$ = 검색어를 넣으면 web 사진을 돌려주는 검색, $M_{CAP}$ = caption model, $\mathcal{X}_r^{web}$ = rehearsal(r)용 web 사진 집합. ← 식에서 $\mathbf{x}$가 "받아 온 web 사진"과 "옛 사진" 두 뜻으로 같이 쓰였어 (슬라이드 표기 그대로).
- 왜 class 이름 대신 caption으로 검색해? caption이 장면을 더 자세히 말하니까 옛 사진과 **더 닮은** 사진이 와 ("Caption queries retrieve semantically richer images than class-name queries", p.84).
- 그래서 제목이 "The Web as **External Memory**"야. 사진은 web에 있고, 내가 들고 있는 건 그걸 찾아올 **열쇠(caption)**뿐이야.

> 보충 (RL 비유, 시험 범위 아님): rehearsal ↔ **replay buffer**. 차이는 buffer에 transition 대신 "다시 찾아올 검색어"만 넣는다는 거야. 사진 저장(개인정보·용량) 문제를 피하는 대신 다시 받은 사진이 원본과 같진 않아.

---

## The Web as External Memory — Verify the Retrieved Image (p.83)

슬라이드 텍스트:
- Verify the Retrieved Image
  - 1. Generate a new caption q'' for the retrieved image.
  - 2. Compare the nouns in q' and q''.
  - 3. Keep the image if they belong to a similar semantic family.
- 그림: q': "a **bird** perched on a plant." → (web 검색) → 플라밍고 사진 → q'': "a **flamingo** standing on a branch". 위에 "Vectorization to the bird". syn: synonyms, depth 목록: 1. Wading bird, 11 / 2. Aquatic bird, 10 / 3. Bird, 9 / 4. Vertebrate, 8 / 5. Chordate, 7 / 6. Animal, 6 / 7. Organism, 5 / 8. Living thing, 4 / 9. Object, 2 / 10. Entity, 0. 두 벡터 $v_1$, $v_2$ (칸이 검정/흰색), $\text{score} = 1 - \cos(v_1, v_2)$ → "Bird"
- Captions serve as retrieval keys, and the web serves as external memory.

**설명**
- 이게 **Caption filtering**이야. caption으로 검색했다고 해도 엉뚱한 사진이 올 수 있으니까 한 번 더 확인해.
  1. 받아 온 사진에 caption을 **새로** 달아 ($q''$).
  2. 저장해 둔 $q'$의 명사와 $q''$의 명사를 비교해.
  3. 둘이 **비슷한 의미 계열(semantic family)**이면 남겨.
- 그림 예: $q'$는 "bird", $q''$는 "flamingo"야. 그림 해석 (슬라이드엔 설명 없음): 목록은 flamingo → Wading bird → Aquatic bird → **Bird** → … → Entity로 올라가는 상위어 사슬로 보이고, $v_1$·$v_2$는 $q'$·$q''$의 명사를 이 계층 위에 vectorize한 것으로 보여. 결과 "Bird"는 둘을 같은 계열로 판정했다는 뜻으로 읽혀. 슬라이드에 글로 적힌 건 "Keep the image if they belong to a similar semantic family."와 $\text{score} = 1 - \cos(v_1, v_2)$까지야.
- 목록의 숫자는 depth(계층 깊이)로, 클수록 구체적인 개념이야 (Entity가 0으로 제일 일반적).
- 벡터 $v_1$, $v_2$는 각 명사를 이 계층 위에 표시한 것이고, $\text{score} = 1 - \cos(v_1, v_2)$는 **거리**야. 작을수록 비슷해. ← 슬라이드엔 남기는 기준값이 없어.
- 한 줄: "Captions serve as **retrieval keys**, and the web serves as **external memory**."

> 보충: 이런 상위어(hypernym) 계층은 보통 WordNet 같은 어휘 사전에서 와. 슬라이드엔 사전 이름이 없어. "Object, 2" 다음이 "Entity, 0"이라 depth 1이 목록에서 빠져 있어 (그림 원문 그대로).

---

## Do Filtering and Captions Actually Help? (p.84)

슬라이드 텍스트:
- Table 4: Ablation study on the web learning strategies: Pascal-VOC Overlapped. (Domain filtering / Caption Labeling, 10-10과 15-5 설정)
  - 둘 다 없음: 10-10 All 62.1 / 15-5 All 67.8
  - 둘 다 있음: 10-10 All **65.0** / 15-5 All **68.4**
- Table 5: Ablation study on the rehearsal strategies on Pascal-VOC Overlapped. (Caption query / Caption Filtering)
  - 둘 다 없음: 10-10 All 63.6 / 15-5 All 69.8
  - 둘 다 있음: 10-10 All **65.3** / 15-5 All **73.4**
- Learning New Classes
  - Domain filtering reduces the web-to-dataset gap.
  - Caption labeling corrects incomplete single-class supervision.
  - The two components are complementary.
  - Combined use improves the 10–10 result by 2.9 points.
- Web rehearsal ablation
  - Caption queries retrieve semantically richer images than class-name queries.
  - Caption filtering removes irrelevant retrievals.
  - Querying and filtering work best together.
  - The combined method improves 15–5 performance by 3.6 points.
- (왼쪽 아래, 잘린 글상자) Homogeneous Supervision / - Every training image uses the same weak label type. / - Tags are cheap but provide no localization. / - Boxes are informative but expensive. / - Points provide an intermediate level of supervision. / Heterogeneous Supervisi… ← p.87 본문과 같은 글이 슬라이드 아래에 잘린 채로 남아 있어 (슬라이드 편집 잔여물로 보여. p.84 내용 아님)

**설명**
- **ablation study** = 부품을 하나씩 빼 보고 성능이 얼마나 떨어지는지 보는 실험이야.
- 설정 이름 읽는 법: "10-10" = 처음에 class 10개를 배우고, 다음 step에 10개를 더 배워. "15-5" = 15개 → 5개 더. 열은 10-10이면 1-10(옛)/11-20(새)/All, 15-5면 1-15(옛)/16-20(새)/All이야.
- 숫자 확인: 62.1 → 65.0 = **+2.9** (Table 4, 10-10 All), 69.8 → 73.4 = **+3.6** (Table 5, 15-5 All). 슬라이드 문장과 맞아.
- 부품 하나만 쓴 행 (All, 10-10 / 15-5): Table 4 domain filtering만 63.1 / 68.0, caption labeling만 63.6 / 68.2. Table 5 caption query만 64.9 / 72.6, caption filtering만 63.5 / 71.9. All 기준으론 둘 다 쓸 때가 최고야. ← 단 Table 4의 15-5 새 class(16-20)는 caption labeling만 쓴 46.7이 둘 다 쓴 45.9보다 높아.
- 메시지는 "두 부품은 서로 보완적(complementary)이고 같이 쓸 때 제일 좋다" 하나야. 표 수치는 시험 비중이 낮아.

> 보충: "Overlapped"는 incremental segmentation 쪽 용어로, step t 사진에 옛 class나 아직 안 배운(미래) class가 (배경으로) 같이 찍혀 있어도 되는 설정이야. 슬라이드엔 정의가 없어 (시험 범위 밖).

---

## Web Images Can Provide Both Learning and Rehearsal (p.85)

슬라이드 텍스트:
- The proposed method improves the challenging 10–1 result substantially.
- Caption-based rehearsal is particularly valuable in multi-step learning.
- 그림 Fig. 4: Qualitative results in the overlapped 10-1 setting. (열: Image, GT, WILSON (VOC+WEB), OURS (VOC+WEB), WILSON (WEB+WEB), OURS (WEB+WEB). 사이클 선수, 의자 위 개, 고양이 사진)
- Table 3: Comparison on single-step multi-class settings on PASCAL-VOC. (열: Method, Train, Memory, M, 15-1 (1-15, 16-20, All), 10-1 (1-10, 11-20, All))
  - WILSON [7] VOC - - : 15-1 All 0.6 / 10-1 All 0.1
  - RaSP [37] VOC - - : 13.2 / 1.3
  - RaSP VOC VOC 100: 56.0 / 36.9
  - WILSON VOC ImageNet 100: 65.9 / 52.3
  - RaSP VOC ImageNet 100: 66.6 / **54.4**
  - WILSON VOC WEB 100: 63.6 / 49.9
  - Ours VOC WEB 100: 65.1 / 52.0
  - WILSON WEB WEB 100: 67.2 / 42.9
  - Ours WEB WEB 100: **67.4** / **54.4**
- The web is not only a source of new data. It can serve as an external semantic memory.

**설명**
- "10-1" = 처음 10개를 배운 뒤 class를 **하나씩** 10번 추가해. step이 많을수록 잊기 쉬워서 "challenging"이야.
- 표 열 읽는 법: **Train** = 새 class 학습 사진 출처 (VOC = curated, WEB = web), **Memory** = 복습 사진 출처, **M** = memory 열의 숫자(100)야. ← M의 정확한 뜻은 슬라이드에 없어.
- forgetting이 얼마나 심한지는 첫 두 줄이 보여 줘. memory 없이(-) 학습하면 10-1 All이 **0.1**, **1.3**이야. 옛 class는 물론 앞 step에서 배운 새 class까지 거의 다 잊어서 0 근처야 (WILSON: 1-10 0.0, 11-20 0.2).
- 같은 WEB+WEB 조건에서 10-1 All이 WILSON 42.9 → Ours **54.4**야. 이게 "improves the challenging 10–1 result substantially"야.
- 결론 문장: web은 새 data의 출처일 뿐 아니라 **external semantic memory**도 될 수 있어.
- ← 표 caption은 "single-step multi-class"인데, 15-1·10-1은 class 하나씩 여러 step 추가하는 설정이라 이름이 반대로 보여 (원문 유지, 의심).

---

## Section question: Under a fixed annotation budget, should every image receive the same type of weak label? (p.86)

슬라이드 텍스트:
- Under a fixed annotation budget, should every image receive the same type of weak label?
- 그림: 남색 배경 section 구분 슬라이드

**설명**
- **annotation budget** = label 다는 데 쓸 수 있는 총 비용(시간·돈)이야.
- 질문: 예산이 정해져 있을 때, 모든 사진에 **같은 종류**의 weak label(전부 tag, 또는 전부 box)을 달아야 할까? 이 논문 답은 "아니, **섞는** 게 낫다"야.
- 이 파트부터 task가 바뀌어. 앞은 **semantic** segmentation이었고, WISH는 **instance** segmentation이야.
  - **semantic segmentation**: pixel마다 class만 칠해. 사람 두 명이 붙어 있으면 한 덩어리 "person"이야.
  - **instance segmentation**: 물체 **하나하나**를 따로 mask로 내. 사람 1, 사람 2가 구분돼.

---

## Must Every Image Use the Same Weak Label? (p.87)

슬라이드 텍스트:
- Homogeneous Supervision
  - Every training image uses the same weak label type.
  - Tags are cheap but provide no localization.
  - Boxes are informative but expensive.
  - Points provide an intermediate level of supervision.
- Heterogeneous Supervision
  - Different images may use **different weak label** types.
  - **Cheap** labels provide **broad** data coverage.
  - **Strong** labels provide **accurate** localization.
  - A **single model** learns from **all label types**.
- 그림: 왼쪽 "Homogeneous" — Tag $(\mathbf{I}, \mathbf{Y}^t)$ 줄(Bird Chair / Person / Boat 글자), Point $(\mathbf{I}, \mathbf{Y}^p)$ 줄(점), Box $(\mathbf{I}, \mathbf{Y}^b)$ 줄(box). 오른쪽 "Heterogeneous" — 사진마다 tag·point·box가 섞인 한 열, Hetero $(\mathbf{I}, \mathbf{Y}^{\text{Hetero}})$ → $\mathbf{Y}^t\ \mathbf{Y}^p\ \mathbf{Y}^b$
- 식: $\mathbf{Y}_i^{\text{Hetero}} = \mathbf{Y}_i^t$ or $\mathbf{Y}_i^p$ or $\mathbf{Y}_i^b$.
- (왼쪽 아래 작은 글씨) Class matching identifies the semantic category. / Prompt matching identifies the approximate instance location. / Mask matching transfers SAM's segmentation knowledge. / Hungarian matching assigns predictions to weakly labeled instances. ← p.90 본문과 같은 글이 남아 있어 (편집 잔여물로 보여)

**설명**
- weak label 세 종류 (instance segmentation 기준):
  - **tag** = 사진에 어떤 class가 있는지만 (image-level label). 제일 싸고, 위치 정보가 **없어**.
  - **point** = 물체마다 안쪽에 점 하나. 중간 정도 정보와 비용이야.
  - **box** = 물체마다 bounding box. 정보가 많고 비싸.
- **Homogeneous** = 모든 사진이 같은 종류의 label이야 (기존 WSIS).
- **Heterogeneous** = 사진 $i$마다 label 종류가 달라도 돼. 식이 그 뜻이야: 사진 $i$의 label $\mathbf{Y}_i^{\text{Hetero}}$는 tag, point, box **중 하나**야.
- 역할 분담: **싼 label = 넓은 coverage**(많은 사진), **강한 label = 정확한 위치**. 그리고 **model은 하나**로 전부에서 배워.

---

## WISH: A Common Language for Weak Labels (p.88)

슬라이드 텍스트:
- SAM's prompt space serves as a common representation for heterogeneous supervision.
- Predict prompt latent from object queries
  - Q: Mask2Former object queries
  - H_prompt: prompt prediction head
  - hat(z): predicted prompt latent
- 식: $\hat{\mathbf{Z}} = \mathcal{H}_{\text{prompt}}(\mathbf{Q})$,
- Weak labels to prompt latent
  - Point prompt → point latent
  - Box prompt → box latent
  - Tag → converted point prompt
- 그림 (architecture): Image $\mathbf{I}$ → Encoder → Feature $\mathbf{F}$ → Decoder; Input queries $\mathbf{Q}_0$ → Transformer decoder → $\mathbf{Q}$ → 세 head $\mathcal{H}_{cls}$(Instance class prediction logit $\hat{\mathbf{y}}_{cls}$), $\mathcal{H}_{prompt}$(Prompt laent prediction $\hat{\mathbf{Z}}$), $\mathcal{H}_{mask}$(Pixel-wise mask prediction $\hat{\mathbf{M}}$) → **Multi-stage matching (Section 4.2)** ← $\mathbf{C}$(Instance class supervision), $\mathbf{Z}$(Prompt latent supervision), $\mathbf{M}_{SAM}$(SAM mask supervision). 왼쪽 아래 "Section 4.3" 상자: $\mathbf{F}$ → $\mathcal{H}_{cam}$ → CAM $\mathbf{A}$ → $\hat{\mathbf{y}}^{cam}_{cls}$, $\mathcal{L}_{cam}$ ↔ Tag $\mathbf{Y}^t$. Heterogeneous Weak Labels → $\mathbf{Y}^{t\to p}$, $\mathbf{Y}^p\ \mathbf{Y}^b$ (Point/Box) → **SAM prompt encoder**(얼음 표시 = frozen) → $\mathbf{Z}$, → **SAM mask decoder**(frozen, SAM img feature 입력) → $\mathbf{M}_{SAM}$ ← 그림 글자 "Prompt **laent**"는 latent의 오타
- SAM maps points and boxes into a shared prompt space.
- WISH predicts the corresponding prompt latent from each object query.
- Weak labels with different formats become comparable in one space.
- The model learns what instance each query should represent.

**설명**
- 먼저 깔 용어
  - **SAM (Segment Anything Model)** = 사진과 **prompt**(점이나 box)를 주면 그 물체의 mask를 내 주는 큰 pretrained segmentation model이야. prompt를 숫자 벡터로 바꾸는 **prompt encoder**와, mask를 그리는 **mask decoder**가 있어.
  - **prompt latent** = prompt encoder가 점·box를 바꾼 벡터야. 이 벡터들이 사는 공간이 **prompt space**야.
  - **Mask2Former / object query** = Mask2Former는 학습되는 벡터 $N$개(object queries)를 갖고, query 하나가 물체 하나를 맡아서 그 class와 mask를 내는 segmentation model이야. $\mathbf{Q}_0$가 처음 query, $\mathbf{Q}$가 transformer decoder를 거친 query야.
- 문제: tag, point, box는 생김새가 달라서 하나의 loss로 비교하기 어려워.
- 핵심: **SAM의 prompt space를 공용어(common language)로 써.** point는 point latent로, box는 box latent로, tag는 먼저 point prompt로 바꾼 다음 point latent로 보내 (p.91). 그러면 모든 weak label이 **같은 공간의 벡터** $\mathbf{Z}$가 돼.
- WISH는 query마다 head를 하나 더 달아 ($\mathcal{H}_{\text{prompt}}$) "이 query가 맡은 물체의 prompt latent" $\hat{\mathbf{Z}}$를 예측해. 이 예측을 weak label의 $\mathbf{Z}$와 맞추면 "**이 query가 어느 instance를 맡아야 하는지**"를 배워.
- query마다 head가 셋이야: class($\hat{\mathbf{y}}_{cls}$), prompt latent($\hat{\mathbf{Z}}$), mask($\hat{\mathbf{M}}$). 각각을 weak label에서 온 $\mathbf{C}$, $\mathbf{Z}$, $\mathbf{M}_{SAM}$과 맞춰 (p.89).
- SAM 부분(prompt encoder, mask decoder)엔 얼음 표시가 있어. **frozen**이라 학습 안 하고 target만 만들어.

> 보충 (논문, 슬라이드엔 없음): 논문 abstract 기준으로 WISH는 SAM의 prompt latent space를 weak label 공통 표현으로 쓰고, multi-stage matching으로 tag에 없는 spatial 정보를 보완해. 다른 방법과의 대비는 확인 못 했어 (시험 범위 아님).

---

## How Is Each Prediction Matched to a Weak Label? (p.89)

슬라이드 텍스트:
- 1. Class Matching
  - Does query i predict the class of weak label j?
  - 식: $\text{Cost}^{i,j}_{\text{cls}} = \hat{\mathbf{y}}^i_{\text{cls}}(\mathbf{c}^j)$
- 2. Prompt Matching
  - Does the predicted query represent a similar point or box prompt?
  - 식: $\text{Cost}^{i,j}_{\text{prompt}} = \text{KLD}(\hat{\mathbf{Z}}^i, \mathbf{Z}^j)$
- 3. Mask Matching
  - Select the SAM candidate mask closest to the predicted instance mask.
  - 식: $\text{Cost}^{i,j}_{\text{mask}} = \min_{n\in\{1,2,3\}} d(\hat{\mathbf{M}}^i, \mathbf{M}^{j,n}_{\text{SAM}})$
- 식 (오른쪽 아래): $\text{Cost}^{i,j} = \alpha\,\text{Cost}^{i,j}_{\text{cls}} + \beta\,\text{Cost}^{i,j}_{\text{prompt}} + \gamma\,\text{Cost}^{i,j}_{\text{mask}}$
- 그림: p.88 architecture의 matching 부분 확대

**설명**
- **matching이 왜 필요해?** query는 $N$개고 사진 속 weak label은 몇 개뿐이야. 어느 query가 어느 물체(weak label $j$)를 맡을지 먼저 짝을 정해야 그 짝끼리 loss를 걸 수 있어.
- 짝을 정하는 점수(cost)는 세 가지를 더한 거야.
  1. **Class**: query $i$가 weak label $j$의 class $\mathbf{c}^j$에 주는 예측 확률 $\hat{\mathbf{y}}^i_{\text{cls}}(\mathbf{c}^j)$. "**무엇**인가"를 봐.
  2. **Prompt**: 예측한 prompt latent $\hat{\mathbf{Z}}^i$와 weak label의 prompt latent $\mathbf{Z}^j$ 사이의 KL divergence. 작을수록 같은 점·box를 가리켜. "**어디**인가"를 봐.
  3. **Mask**: SAM은 prompt 하나에 mask 후보를 **3개** 줘 ($n = 1, 2, 3$). 그중 예측 mask $\hat{\mathbf{M}}^i$와 **제일 가까운** 걸 골라 거리를 cost로 써. "**어느 SAM mask가 제일 맞나**"를 봐.
- 전체 cost = $\alpha$·class + $\beta$·prompt + $\gamma$·mask. $\alpha, \beta, \gamma$는 세 항의 비중이야.
- ← 슬라이드 식 그대로면 $\text{Cost}_{\text{cls}}$는 확률이라 **클수록 좋은** 값이야. 최소 cost 짝을 찾는 matching이라면 보통 음수 부호(−확률)를 붙여 써. 슬라이드엔 부호가 없어 (원문 유지, 의심).
- $d(\cdot,\cdot)$는 두 mask 사이 거리야. 슬라이드엔 정의가 없어.
- **KLD** = KL(Kullback–Leibler) divergence, 두 분포가 얼마나 다른지 재는 값이야. ← 슬라이드엔 $\hat{\mathbf{Z}}$를 어떻게 분포로 보는지 정의가 없어 (원문 유지).
- 표기 함정: 여기 $\beta$는 cost 가중치야. p.92의 $\beta_t, \beta_p, \beta_b$(label 종류별 비용)와 다른 뜻이야.

---

## How Is Each Prediction Matched to a Weak Label? — 요약 (p.90)

슬라이드 텍스트:
- Class matching identifies the **semantic category**.
- Prompt matching identifies the approximate instance **location**.
- Mask matching transfers **SAM's** segmentation **knowledge**.
- Hungarian matching assigns predictions to weakly labeled instances.
- WISH matches an instance by asking what it is, where it is, and which SAM mask fits it best.

**설명**
- **Hungarian matching** = 예측 $N$개와 정답 $M$개를 **1:1로** 짝지을 때, 짝들의 cost 합이 **최소**가 되는 조합을 찾는 알고리즘이야. DETR, Mask2Former 계열이 정답 순서가 없는 물체 집합을 학습할 때 써.
- p.89의 $\text{Cost}^{i,j}$ 표(query × weak label)를 만들고, Hungarian matching이 짝을 정해. 짝이 안 된 query는 "물체 없음"으로 처리돼 (Mask2Former 일반 동작, 보충).
- 한 줄로: **what it is**(class) · **where it is**(prompt) · **which SAM mask fits it best**(mask).

> 보충 (RL 비유, 시험 범위 아님): 여러 agent와 여러 task를 1:1로 배정하는 **assignment problem**이 Hungarian 알고리즘의 원래 문제야. multi-agent task allocation에서도 똑같이 써.

---

## How Can a Class Tag Become an Instance Prompt? (p.91)

슬라이드 텍스트:
- 1. Generate Class Activation Maps
  - **CAM** provides rough **class-specific localization**.
  - **Local maxima** are extracted from each class activation map.
  - 식: $\mathbf{A} = \mathcal{H}_{\text{cam}}(\mathbf{F})$
- 2. Convert CAM Peaks into Point Prompts
  - Each selected **CAM peak** becomes a positive point prompt.
  - **SAM** generates an **instance mask** from each point.
  - Overlapping masks are grouped to identify individual instances.
  - 식: $\mathbf{Y}_i^{t\to p} = \{(\mathbf{X}_i^1, \mathbf{c}_i^1), \dots, (\mathbf{X}_i^{k'_i}, \mathbf{c}_i^{k'_i})\}$
- 3. Improve CAMs with Model Predictions
  - Each selected CAM peak becomes a positive point prompt.
  - SAM generates an instance mask from each point.
  - Overlapping masks are grouped to identify individual instances.
  - 식: $\mathcal{L}_{\text{self}} = \mathcal{L}_{\text{bce}}(\mathbf{A}, \hat{\mathbf{M}}_c)$
  ← 3번 아래 bullet 세 줄이 2번 bullet과 똑같아. 복사 실수로 보여 (원문 유지). 3번의 실제 내용은 식 $\mathcal{L}_{\text{self}}$와 p.94 "Self-Enhancement"("Current instance masks refine the CAM. Improved CAMs generate better point prompts for tag supervision.")야.
- 그림: p.88 architecture의 왼쪽 부분 확대 (Section 4.3: $\mathcal{H}_{cam}$, CAM $\mathbf{A}$, $\mathcal{L}_{cam}$, Tag $\mathbf{Y}^t$ → $\mathbf{Y}^{t\to p}$)
- CAM supplies missing spatial cues, and SAM turns those cues into instance-level masks.

**설명**
- tag엔 위치가 없어서 prompt space로 바로 못 보내. 그래서 tag를 **point prompt로 바꿔**.
  1. **CAM**: feature $\mathbf{F}$에 CAM head $\mathcal{H}_{\text{cam}}$를 달아 class별 activation map $\mathbf{A}$를 만들어. tag로 학습해 ($\mathcal{L}_{cam}$, 그림). 이게 p.24–26의 CAM이야.
  2. **Local maxima**: class map마다 **봉우리(국소 최대값)**를 뽑아. 봉우리 하나 = 물체 하나일 가능성이 커.
  3. **Point prompt**: 봉우리 하나하나를 positive point prompt로 SAM에 넣어 instance mask를 받아. mask가 많이 겹치는 점들은 같은 물체로 묶어.
  4. 결과 $\mathbf{Y}_i^{t\to p}$: 사진 $i$의 (점 위치 $\mathbf{X}$, class $\mathbf{c}$) 쌍 $k'_i$개. tag가 point label처럼 바뀌었어 ($t\to p$).
- **Self-enhancement** ($\mathcal{L}_{\text{self}}$): 학습이 진행되면 model이 예측한 instance mask $\hat{\mathbf{M}}_c$(class $c$)를 target으로 CAM $\mathbf{A}$를 BCE(binary cross-entropy)로 다시 학습시켜. CAM이 좋아지면 봉우리도 좋아지고, point prompt도 좋아져.
- 한 줄: "**CAM** supplies missing **spatial cues**, and **SAM** turns those cues into **instance-level masks**."
- p.28의 CAM 문제(discriminative part만 켜짐, mask가 sharp하지 않음)와 연결: point prompt를 만들 땐 CAM의 **봉우리 위치**만 쓰고 mask 모양은 SAM이 그려. CAM 자체는 $\mathcal{L}_{cam}$, $\mathcal{L}_{self}$로 계속 학습·개선돼. CAM의 약점(경계가 거칠다)을 SAM이 메워 주는 구도야.

> 보충 (논문 supplement): 겹치는 봉우리는 SAM mask가 겹치면 하나로 합친다고 적혀 있어 (겹침 기준의 세부 수식은 확인 못 함). 슬라이드 표현은 "Overlapping masks are grouped"까지야.

---

## Which Labels Should We Buy? (p.92)

슬라이드 텍스트:
- Tags are cheap, so many images can be labeled.
- Boxes are expensive, so fewer images can be labeled.
- Strong supervision and broad data coverage must be balanced.
- WISH searches for effective heterogeneous combinations.
- 식: $N_t\beta_t + N_p\beta_p + N_b\beta_b = \zeta$.
  - $N_t$: number of tag-labeled images
  - $N_p$: number of point-labeled images
  - $N_b$: number of box-labeled images
  - $\zeta$: fixed total annotation budget
- Table 3. Results of WISH framework under heterogeneous settings on PASCAL VOC val set. The top row provides the results of tag-point while the bottom row shows the results of tag-box.
  - tag-point: $N_t$ 10,582 / 7,936 / 5,290 / 2,644 / 0, $N_p$ 0 / 441 / 882 / 1,323 / 1,764, AP 46.0 / 46.9 / **47.3** / 47.2 / 44.7
  - tag-box: $N_t$ 10,582 / 7,936 / 5,290 / 2,644 / 0, $N_b$ 0 / 220 / 441 / 661 / 882, AP 46.0 / 47.2 / **48.3** / 47.9 / 45.1
- A balanced mixture can outperform either weak label alone under the same annotation cost.

**설명**
- 식 = **예산 제약**이야. label 종류별 (사진 수 × 장당 비용)의 합이 고정 예산 $\zeta$와 같아. $\beta_t, \beta_p, \beta_b$는 label 종류별 장당 비용이야 ← 슬라이드 목록엔 $\beta$ 설명이 빠져 있어.
- 표 읽는 법: 같은 예산 안에서 tag를 줄이고 point(또는 box)를 늘려 가며 AP를 봤어.
  - 양 끝(tag만 46.0, point만 44.7, box만 45.1)보다 **가운데 섞은 쪽**(47.3, 48.3)이 높아.
  - 그래서 "A balanced mixture can outperform either weak label alone under the same annotation cost."
- 설명용 계산 (표에서 거꾸로 구한 비율, 슬라이드에 적힌 값 아님): tag 2,646장을 빼면 point 441장 또는 box 약 220장을 살 수 있어. point 1장 ≈ tag 6장, box 1장 ≈ tag 12장 값이라는 설정으로 보여.
- **AP (Average Precision)** = detection·instance segmentation의 대표 성능 지표야. 높을수록 좋아.
- 표기 함정: 이 $\beta$는 비용이고, p.89의 $\beta$는 matching cost 가중치야.

---

## Why Does Heterogeneous Supervision Work? (p.93)

슬라이드 텍스트:
- Tags Provide Coverage
  - Many images can be labeled.
  - The model observes diverse appearances and contexts.
  - Localization is weak.
- Points and Boxes Provide Localization
  - Fewer images can be annotated.
  - Object locations are explicitly provided.
  - Boxes provide stronger spatial constraints.
- Heterogeneous Labels Provide Both
  - Tags improve data coverage.
  - Points and boxes anchor instance localization.
  - SAM's prompt space transfers information across label types.
- Cheap labels provide breadth, while strong labels provide spatial precision.

**설명**
- 정리 표:

| | Tags | Points / Boxes | Heterogeneous |
|---|---|---|---|
| 사진 수 | 많음 | 적음 | 둘 다 |
| 주는 것 | **coverage**: 다양한 생김새·맥락 | **localization**: 물체 위치 명시 (box가 더 강한 제약) | 둘 다 |
| 약점 | localization is weak | 사진이 적음 | — |

- 세 번째 줄의 마지막 bullet이 WISH의 고리야: **SAM's prompt space transfers information across label types.** 공용어가 있어서 tag 사진에서 배운 것과 box 사진에서 배운 것이 한 model 안에서 섞여.
- 한 줄: "Cheap labels provide **breadth**, while strong labels provide **spatial precision**."

> 보충 (RL 비유, 시험 범위 아님): 싼 data를 많이(coverage) + 비싼 data를 조금(precision) 섞는 건, 대량의 저품질 play data + 소량의 전문가 demo를 같이 쓰는 로봇 학습 data mix와 닮았어.

---

## Does Each Component Help? (p.94)

슬라이드 텍스트:
- Prompt Matching
  - Aligns object queries with weak annotations in SAM's latent space.
  - Helps the model learn instance identity without mask ground truth.
- Adaptive SAM Mask Matching
  - SAM returns three masks for each prompt.
  - WISH selects the candidate closest to the current prediction.
  - Adaptive selection outperforms using a fixed mask level.
- Self-Enhancement
  - Current instance masks refine the CAM.
  - Improved CAMs generate better point prompts for tag supervision.
- Table 4. Ablation study for WISH framework on PASCAL dataset. We verify the impact of $\mathcal{L}_{prompt}$, $\mathcal{L}_{mask}$, and $\mathcal{L}_{self}$ on performance.
  - A: $L_{prompt}$ ✗, $L_{mask}$ Adaptive, $L_{self}$ ✓ → 43.6
  - B: ✓, Lv1, ✓ → 36.2
  - C: ✓, Lv2, ✓ → 40.7
  - D: ✓, Lv3, ✓ → 42.8
  - E: ✓, Adaptive, ✗ → 41.5
  - F (WISH): ✓, Adaptive, ✓ → **46.0** (mAP)
- Prompt matching, adaptive mask selection, and self-enhancement are complementary.

**설명**
- 부품 세 개와 역할:
  - **Prompt matching** ($\mathcal{L}_{prompt}$): query를 SAM latent space에서 weak label과 맞춰. **mask 정답 없이도** "이 query는 이 물체"라는 instance identity를 배워.
  - **Adaptive SAM mask matching** ($\mathcal{L}_{mask}$): SAM이 주는 mask 후보 3개(Lv1/Lv2/Lv3, 세밀도가 다른 3단계) 중 **현재 예측과 제일 가까운 것**을 골라 target으로 써. 한 level로 고정하는 것보다 좋아 (36.2 / 40.7 / 42.8 < 46.0).
  - **Self-enhancement** ($\mathcal{L}_{self}$): 지금 instance mask로 CAM을 고치고 → 좋아진 CAM이 tag 사진의 point prompt를 더 좋게 만들어.
- 어느 하나를 빼도 46.0보다 떨어져 (A 43.6, E 41.5). 그래서 "complementary".
- ← Table 4의 46.0은 p.92 표의 tag-only 값(46.0)과 같아서 tag-only 설정의 실험으로 보여 (추정, 슬라이드에 명시 없음). 섞은 설정의 최고치는 p.92의 48.3이야.

> 보충: SAM의 3개 mask는 보통 "전체 / 부분 / 더 작은 부분"처럼 세밀도가 다른 후보야. 점 하나로는 "셔츠"를 원하는지 "사람 전체"를 원하는지 모호해서 SAM이 3개를 같이 줘. Lv1–Lv3의 정확한 대응은 슬라이드에 없어.

---

## From One Weak Label to a Portfolio of Labels (p.95)

슬라이드 텍스트:
- Conventional WSIS
  - Chooses one weak label type for the entire dataset.
  - Treats tags, points, and boxes as separate settings.
- WISH
  - Combines different label types in one model.
  - Uses SAM's prompt space as a common representation.
  - Matches predictions using class, prompt, and mask information.
  - Achieves better cost–performance trade-offs.
- The optimal supervision strategy may be a portfolio of weak labels, not a single annotation type.

**설명**
- **WSIS** = Weakly Supervised Instance Segmentation (WISH 논문 제목의 task). 기존 방법은 dataset 전체에 label 종류를 **하나**만 정하고, tag 설정·point 설정·box 설정을 **따로따로** 다뤘어.
- WISH와의 차이 네 줄이 곧 요약이야: ① 한 model에 여러 label 종류 ② SAM prompt space가 공용 표현 ③ class·prompt·mask로 matching ④ 같은 비용에서 더 좋은 성능.
- **portfolio** = 투자에서 여러 자산을 섞어 담는 것처럼, label도 한 종류에 몰지 말고 **섞어 사라**는 뜻이야.

---

## Next Week.. (p.96)

슬라이드 텍스트:
- Active learning ..

**설명**
- 다음 강의(W12 11/19)는 **Active learning**이야.
- 연결 고리: WISH는 "예산 안에서 **어떤 종류의** label을 살까"를 물었어. active learning은 다음 강의에서 "**어느 사진**에 label을 달까"를 다룰 거야 (예고, 내용은 다음 강의에서 확인).
