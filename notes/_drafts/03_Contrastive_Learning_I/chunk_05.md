<!-- DRAFT | lecture: 03_Contrastive_Learning_I | chunk: 05 | pages 33–40 -->
<!-- VERIFIED | fixes: 3 | verifier: sonnet -->

## 섹션 구분: Are positive and negative pairs always constructed correctly? (p.33)

남색 배경 단일 문장 슬라이드: "Are positive and negative pairs always constructed correctly?"

**설명**
- 직전 p.32 (Positive Pair Construction: "Contrastive learning incorporates multiple positive and negative samples.")에서 이어지는 **문제 제기** 슬라이드임.
- contrastive learning은 "같은 이미지의 두 augmentation = positive, 다른 이미지 = negative"라는 **가정**에 기반하는데, 이 가정이 항상 맞는지 의문을 던짐. 이후 p.34~ 에서 positive pair 쪽 문제를 GenView로 다룸.
> 보충: negative 쪽 문제로는 같은 클래스인데 negative로 취급되는 **false negative** (sampling bias) 문제가 있음. 이 chunk는 positive 쪽(false positive / 낮은 diversity)에 집중함.

## GenView 논문 표지 (p.34)

- 논문 첫 페이지 캡처: **"GenView: Enhancing View Quality with Pretrained Generative Model for Self-Supervised Learning"**
- 저자: Xiaojie Li, Yibo Yang, Xiangtai Li, Jianlong Wu, Yue Yu, Bernard Ghanem, Min Zhang
- 소속: Harbin Institute of Technology (Shenzhen), Peng Cheng Laboratory, KAUST, S-Lab (Nanyang Technological University)
- 코드: https://github.com/xiaojieli0903/genview
- 좌상단 "Check for updates" 아이콘.
- 하단 출처: "Li et al., ECCV 2024, GenView: Enhancing View Quality with Pretrained Generative Model for Self-Supervised Learning"

**설명**
- 이 절의 중심 논문은 **GenView (Li et al., ECCV 2024)**. 핵심 아이디어는 handcrafted augmentation 대신 **pretrained generative model**(diffusion)로 positive view를 만들고, 생성 품질에 따라 loss 가중치를 주는 것임.
- 즉 "view의 quality" 자체를 개선하는 접근이라 제목이 *Enhancing View Quality*.

## Can a Generative Model Create Better Positive Views? (p.35)

**그림 (논문 overview figure)**: 제목 "Adaptive View Generation".
- 왼쪽 "Original image": 개 한 마리가 앉아 있는 사진에 색 박스 3개 (파랑=벽돌 건물 배경 영역, 노랑=개 얼굴 영역, 분홍=개 얼굴을 포함하는 약간 다른 영역).
- 맨 위 아치형 화살표: Original image → **"Image Conditioned Generative Model"**(눈송이 아이콘 = frozen) → 오른쪽 "GenView" 이미지(같은 개가 풀밭에서 전신이 보이는 다른 pose로 생성됨; 초록 박스=개 머리, 하늘색 박스=다리/몸통 일부).
- 가운데 4개 pair (a)~(d), 각 pair는 위(비교 대상 view)·아래(노랑 crop = anchor view) 두 이미지:
  - (a) 위: 파랑 crop(벽돌 벽만, 개 없음), 아래: 개 얼굴. 가중치 **1.0** (Standard Contrastive Loss)
  - (b) 위: 분홍 crop(개 얼굴, 아래 노랑과 거의 동일), 아래: 개 얼굴. 가중치 **1.0** (Standard)
  - (c) 위: 초록 박스에서 얻은 생성 이미지 속 개 얼굴(다른 pose/배경), 아래: 개 얼굴. 가중치 **1.2** (초록 강조) (Quality-Driven Contrastive Loss)
  - (d) 위: 하늘색 박스에서 얻은 생성 이미지 속 개 다리/몸통 일부(품질 낮음), 아래: 개 얼굴. 가중치 **0.8** (빨강 강조) (Quality-Driven)
- (a),(b)는 회색 박스 "Standard Contrastive Loss", (c),(d)는 주황 박스 "Quality-Driven Contrastive Loss"로 묶임.

**설명**
- 질문: **생성 모델이 더 좋은 positive view를 만들 수 있는가?** 답이 GenView.
- 표준 방식은 원본에서 crop (a),(b)로 view를 만들고 **모든 pair에 동일 가중치 1.0**을 줌. (a)는 개가 없는 crop이라 anchor와 의미가 다른 **false positive**, (b)는 거의 똑같은 view라 배울 게 적음.
- GenView는 생성 모델로 의미(개)는 유지하면서 pose/배경이 다른 view (c)를 만들고, 생성이 어색한 (d)는 낮은 가중치 0.8로 줄이며 좋은 pair (c)는 1.2로 올림. 즉 **view generation + quality-based re-weighting**의 2단 구성.
- 가중치 1.0 / 1.2 / 0.8은 개념 설명용 수치 예시로 보임.

## Limitations of Handcrafted Views (p.36)

- 그림: p.35의 왼쪽 일부 재사용. Original image(박스 3개)와 pair (a), (b), 둘 다 가중치 1.0, "Standard Contrastive Loss".
- 본문: "Standard augmentations have two limitations"
  - (a) **False Positive Pair**: A strong crop may remove the main object.
  - (b) **Limited Diversity**: A weak crop may produce two nearly identical views.
- 하단 굵은 문장: **A good positive pair should preserve semantics while introducing diversity**

**설명**
- **(a) False Positive**: random resized crop 범위가 넓으면 한쪽 view에 객체가 아예 없음(배경 벽돌만). 그런데 loss는 두 view를 같은 embedding으로 당기므로 **의미 없는 배경 ↔ 개**를 positive로 학습 → noisy supervision.
- **(b) Limited Diversity**: crop이 약하면 두 view가 거의 동일 → invariance를 배울 게 적어 학습 신호가 약하고, 표현이 pose/viewpoint/배경 변화에 robust해지지 못함.
- 두 문제는 **trade-off**: augmentation을 세게 하면 (a), 약하게 하면 (b). handcrafted 방식은 이미지별로 강도를 조절할 수 없음(전 이미지 동일 augmentation 분포).
- 좋은 positive의 조건 = **semantic 보존 + diversity** (두 목표를 동시에 만족).
> 보충: SimCLR에서 random crop + color distortion이 핵심 augmentation으로 알려져 있지만(Chen et al., 2020), crop은 object가 없는 view를 만들 수 있음이 지적돼 왔음.

## With Generative Model.. (p.37)

- 그림: p.35의 (a)~(d) 4개 pair 확대 (가중치 1.0, 1.0, 1.2, 0.8; Standard Contrastive Loss / Quality-Driven Contrastive Loss).
- 본문: "Generative model can overcome.."
  - (c) Pose and background change while the object's semantics are preserved.
  - (d) The quality of the generated results is relatively low -> assigned a low weight of 0.8.
- 하단 굵은 문장: **GenView's goal is to vary elements such as pose, viewpoint, and background while preserving object identity**

**설명**
- (c): 생성 모델은 원본 이미지의 **semantic(CLIP image embedding)** 에 conditioning해서 만들므로 개라는 identity는 유지하되 pose/배경/viewpoint는 달라짐 → **semantic 보존 + diversity** 동시 달성. crop으로는 불가능한 변화(자세 변경 등)임.
- (d): 생성 모델도 완벽하지 않음. 객체가 잘리거나 이상한 부분만 생성되면 오히려 false positive가 되므로, 그런 pair는 **quality를 측정해 낮은 weight(0.8)** 를 줌. 생성의 약점을 loss 쪽에서 보완하는 설계.
- GenView 목표 한 줄 요약: **pose/viewpoint/background는 바꾸고 object identity는 유지**.

## GenView Framework — 두 구성요소 (p.38)

슬라이드 문구: "GenView has two components". 하단 라벨 두 개: **Generate a diverse positive view** (왼쪽 초록 영역) / **Give more weight to better pairs** (오른쪽 파란 영역).

**그림 (전체 파이프라인 다이어그램)**
- 입력 이미지 **X_i** (방 안의 개 2마리).
- **위쪽 경로**: X_i --(t, handcrafted random augmentation)--> **P_i** (augmented view) → **Encoder f** → representation.
- **아래쪽 경로 (초록 박스, "Adaptive View Generation")**: X_i → **CLIP Image Encoder C** (눈송이=frozen) →
  - 출력 1: 임베딩 **c_i** (벡터 하나).
  - 출력 2: 여러 층의 feature **Z_i** (적층된 큐브) → **A_i** (foreground 확률/attention 맵, 회색 heatmap) → **B(·)** (binarize, 흑백 마스크) → **l_i^ada** (adaptive noise level; 마스크의 foreground 비율에서 결정).
  - **noisy(c_i, l_i^ada)** 블록에서 c_i에 noise를 섞음 → **Stable Diffusion G** (눈송이=frozen) → 생성 이미지 **P_i^+** (같은 개 2마리, 다른 pose/배경; 경로에 t^+ 표기) → **Encoder f**.
- **오른쪽 파란 영역 ("Quality-Driven Contrastive Loss")**:
  - 두 view P_i, P_i^+ 각각이 frozen **E**(눈송이)를 통과해 feature map **F_i**, **F_i^+** 생성.
  - F_i → foreground mask **M_i^f**, background mask **M_i^b** (흑백 heatmap 2장)와 각각 ⊗(곱, pooling) → **z_i^f**, **z_i^b**. F_i^+ → **M_i^{f+}**, **M_i^{b+}** → **z_i^{f+}**, **z_i^{b+}**.
  - 점수: $q_i = s_i^f - s_i^b$ (s^f: foreground 간 유사도, s^b: background 간 유사도로 보이며 슬라이드에는 정의 없음)
  - 가중치: $w_i = \mathrm{softmax}(q_i)$
  - 최종 loss: $\tilde{\mathcal{L}}_{\mathrm{SSL},*} = w_i \cdot \mathcal{L}_{\mathrm{SSL},*}$ (주황 타원; 두 Encoder 출력에서 오는 화살표가 이 loss로 들어감).

**설명**
- **구성요소 1: Adaptive View Generation** — frozen **CLIP image encoder + Stable (un)CLIP diffusion**으로 원본 이미지와 의미가 같되 모습이 다른 positive를 **이미지별 적응적 noise**로 생성.
- **구성요소 2: Quality-Driven Contrastive Loss** — 생성된 pair의 품질을 **foreground 일관성(s^f) − background 일관성(s^b)** 같은 점수 q로 평가하고 **softmax로 정규화한 weight w_i**를 SSL loss에 곱함. 의미가 유지된(foreground는 비슷, background는 다른) pair일수록 높은 가중치라고 해석됨 (s^f, s^b 정의가 슬라이드에 없어 q의 형태 q = s^f - s^b 로부터의 추정).
- 세 pretrained 모델(CLIP image encoder C, Stable Diffusion G, E)은 모두 **frozen**(눈송이 아이콘) → 학습 대상은 SSL encoder f뿐.
> 보충: 생성을 offline/별도 단계로 수행할 수 있는지는 슬라이드에 명시 없음 (추정).
- 기존 SSL(SimCLR, MoCo, BYOL, DINO 등)에 plug-in 가능한 **framework**임 (loss는 "SSL,*"로 표기되어 어떤 SSL loss든 대체 가능).
> 보충: s^f, s^b의 정확한 정의와 E의 종류는 슬라이드에 없으므로 정식 노트에서 확인 필요. 마스크 M^f/M^b는 E의 feature map 위 foreground/background 분할로 이해하면 됨.
> 보충: softmax는 **batch 내** 샘플들 사이에서 정규화한다고 보는 것이 자연스러움. ← 슬라이드에 정규화 범위 명시 없음

## GenView Framework — 두 positive view 생성 (p.39)

본문: "GenView generates two positive views"
- 그림: p.38 왼쪽 부분 확대. 위: X_i --(**random predefined augmentations**, 기호 t 빨간 동그라미)--> P_i. 아래: CLIP image encoder → c_i / Z_i → A_i → B(·) → l_i^ada → noisy(c_i, l_i^ada) → Stable Diffusion G → t^+ → P_i^+.
- 우측 텍스트:
  - Generate Positive Views
  - (Up) Apply random augmentation
  - (Bottom)
    - Extract CLIP image embedding
    - Add adaptive noise
      - based on the foreground ratio
    - Stable unCLIP generates an image
- 하단 수식:

$$\mathbf{X}_i^{+} = \mathcal{G}\big(\mathbf{z}_T,\ \mathrm{noisy}(\mathbf{c}_i,\ l_i^{\mathrm{ada}}),\ w\big)$$

  - 빨간 표시: **c_i** = "conditional image embedding", **l_i^ada** = "perturbation strength".

**설명**
- 수식 기호:
  - $\mathcal{G}$: frozen **Stable unCLIP** 생성 모델(diffusion).
  - $\mathbf{z}_T$: 시작 **Gaussian noise latent** (T = 최대 diffusion step; 여기서부터 denoising하며 이미지 생성).
  - $\mathbf{c}_i$: 원본 X_i의 **CLIP image embedding** → 생성의 **조건(condition)**. 이 embedding이 의미(semantic)를 담아서 객체 identity가 유지됨.
  - $\mathrm{noisy}(\mathbf{c}_i, l_i^{\mathrm{ada}})$: 조건 embedding에 noise를 섞어 **perturbation** 부여. 조건이 흔들리므로 생성 결과가 원본에서 멀어짐 = **diversity 확보**.
  - $l_i^{\mathrm{ada}}$: **perturbation strength (noise level)**. 이미지마다 다르게 정해짐(adaptive) — foreground ratio 기반 (p.40).
  - $w$: **classifier-free guidance scale**로 보임 (슬라이드에 설명 없음). ← 슬라이드에서 w 정의 안 됨 (p.38의 loss 가중치 $w_i$와 기호가 비슷해 혼동 주의)
- 결국 positive pair 구성: **(P_i, P_i^+)** = (handcrafted augmentation view, generated view). 즉 **한쪽은 기존 augmentation, 다른 쪽은 생성 view**로 pair를 만들어 둘의 장점을 섞음.
> 보충: Stable unCLIP은 Stable Diffusion을 text 대신 **CLIP image embedding(unCLIP 방식, DALL·E 2 계열 아이디어)** 으로 conditioning하도록 fine-tune한 모델이며, 이미지 variation 생성이 가능함. 슬라이드에는 "Stable Diffusion"과 "Stable unCLIP"이 혼재.
> 보충: 슬라이드 제목은 "two positive views"이며, 그림의 위쪽(augmentation)과 아래쪽(generated) 두 view를 가리킨다고 이해함.

## Adaptive View Generation (p.40)

본문: "How much variation should we introduce?"

**그림 (이미지 격자 표)**
- 열: 왼쪽 3열 "Lower foreground proportion"(풀밭의 작은 새, 바다 위 작은 새, 하늘의 새떼 등 객체가 작은 이미지), 오른쪽 3열 "Higher foreground proportion"(새 클로즈업, 개 클로즈업, 요트 등 객체가 큰 이미지).
- 행: "Real image" / "Lower noise level" / "Higher noise level" (각 행에 생성 결과).
- 파란 박스: Lower foreground × Lower noise level 3장 (객체가 유지된 생성). 초록 박스: Higher foreground × Higher noise level 3장 (객체는 유지되며 pose·배경이 다양하게 변화).
- Lower foreground × Higher noise level 행은 객체가 작아지거나 사라지는 경향, Higher foreground × Lower noise는 원본과 매우 유사.
- 하단 설명:
  - **Small foreground** — Use less noise. The object may easily disappear.
  - **Large foreground** — Use more noise. More variation can be introduced safely.

**설명**
- **핵심 아이디어**: 변화량(noise level)을 이미지의 **foreground 비율**에 따라 조절함.
  - foreground가 작은 이미지: 조건 embedding에 noise를 많이 주면 객체 정보가 날아가 생성 이미지에 객체가 없어짐 → **semantic drift / false positive**. 그래서 noise를 **적게**.
  - foreground가 큰 이미지: 객체 정보가 embedding에 충분히 담겨 있어 noise를 많이 줘도 객체가 남음 → **더 큰 diversity를 안전하게** 도입.
- p.38~39의 $A_i \to B(\cdot) \to l_i^{ada}$ 흐름이 바로 이것: CLIP feature에서 foreground 맵 A_i를 얻고, 이진화(B)로 foreground 비율을 계산하여 $l_i^{ada}$를 결정 (비율이 높을수록 $l_i^{ada}$ 큼).
- 고정 noise level을 모든 이미지에 쓰는 것의 한계(작은 객체 이미지는 망가지고 큰 객체 이미지는 변화가 부족)를 해결.
> 보충: 이 범위에서 $l_i^{ada}$와 foreground ratio의 정확한 함수식은 제시되지 않음 (다음 chunk 또는 논문 확인 필요).

## FIGURES

- p.35: GenView overview 그림 (원본 + (a)~(d) pair + 가중치 1.0/1.0/1.2/0.8, Standard vs Quality-Driven Loss) — 중요도 상
- p.36: handcrafted view의 한계 (false positive / limited diversity) — 중요도 중 (p.35와 겹침)
- p.37: (a)~(d) 확대, 생성 view의 장점/단점 — 중요도 중
- p.38: GenView 전체 프레임워크 다이어그램 (Adaptive View Generation + Quality-Driven Contrastive Loss) — 중요도 상
- p.39: Adaptive view generation 경로 확대 + $\mathbf{X}_i^+$ 수식 — 중요도 상
- p.40: foreground ratio × noise level 격자 그림 — 중요도 상

## EXAM POINTS

1. **handcrafted augmentation의 두 한계**: (a) False Positive Pair(strong crop이 main object 제거), (b) Limited Diversity(weak crop은 거의 동일한 view). 좋은 positive pair = **semantic 보존 + diversity**.
2. **GenView의 두 구성요소**: Adaptive View Generation(다양한 positive view 생성) + Quality-Driven Contrastive Loss(좋은 pair에 큰 가중치). 각각의 역할 설명.
3. **생성 수식** $\mathbf{X}_i^{+}=\mathcal{G}(\mathbf{z}_T,\mathrm{noisy}(\mathbf{c}_i,l_i^{\mathrm{ada}}),w)$ 의 각 기호 의미 ($\mathbf{c}_i$=CLIP image embedding(조건), $l_i^{ada}$=perturbation strength, $\mathcal{G}$=Stable unCLIP, $\mathbf{z}_T$=초기 noise).
4. **Why adaptive noise?** foreground 비율이 작으면 noise 적게(객체 소실 방지), 크면 많이(안전하게 diversity 증가). foreground ratio는 CLIP feature → A_i → B(·)로 구함.
5. **Quality-driven loss**: $q_i = s_i^f - s_i^b$, $w_i=\mathrm{softmax}(q_i)$, $\tilde{\mathcal{L}}=w_i\cdot\mathcal{L}_{SSL}$. 왜 낮은 품질 pair에 낮은 가중치를 주는가? (생성 실패로 인한 false positive 영향 감소)
6. **생성 view vs handcrafted view 비교**: 생성 view는 pose·viewpoint·background를 바꿀 수 있어 diversity가 큼 vs. 생성 품질이 낮을 수 있음 → loss re-weighting으로 보완.
7. **positive pair 구성**: GenView는 (augmentation view, generated view) pair를 사용하고, CLIP/Stable Diffusion/E는 모두 frozen이라 학습 대상은 SSL encoder뿐.
8. **p.33 질문의 의도**: contrastive learning의 pair 구성이 항상 올바르지 않음 (false positive 등) → 이를 개선하는 방법 논의.

## VERIFIER REPORT
- p.38 설명: "두 pretrained 모델(CLIP, Stable Diffusion, E)"은 세 개이므로 "세"로 수정, 슬라이드에 없는 "offline/별도 단계 생성 가능" 주장은 보충(추정)으로 격하.
- p.38 설명: q 가중치 해석(foreground 비슷/background 다름)을 슬라이드에 정의가 없는 추정임을 명시.
- p.36 설명: "trivial solution에 가깝고"는 과장이라 "학습 신호가 약하고"로 완화.
- 페이지 누락, 수식·기호, 그림 설명, 인용(GenView, Li et al., ECCV 2024; SimCLR, Chen et al., 2020)은 이상 없음.
