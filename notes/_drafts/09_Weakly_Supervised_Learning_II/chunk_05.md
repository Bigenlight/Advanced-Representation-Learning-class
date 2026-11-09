<!-- DRAFT | lecture: 09_Weakly_Supervised_Learning_II | chunk: 05 | pages 62–74 -->
<!-- VERIFIED | fixes: 8 | verifier: sonnet -->

## Section: Beyond CAM (p.62)

슬라이드 텍스트:
- Beyond CAM
- CAM activates only the most discriminative regions
- CAM boundaries are coarse and inaccurate

**설명**
- 섹션 구분 슬라이드야. p.2 목차(1–4)엔 없는 부분이고, 여기서부터 p.95까지 **최신 논문 세 편**을 소개해. 목차 밖이어도 슬라이드에 있으니까 **시험 범위**야.
- 앞(p.28–41)에서 본 CAM의 두 약점을 다시 적은 거야.
  1. **most discriminative regions만 켜져.** 개라면 얼굴만 빨갛고 몸통은 안 켜져 (p.28 "Focused on discriminative area only").
  2. **경계가 거칠고 부정확해** (p.28 "Resulting masks are not sharp").
- "Beyond CAM" = CAM에서 출발은 하지만, CAM 하나로는 못 푸는 이 두 문제를 **다른 정보원(CLIP의 class 분포, DINO feature, web, SAM 등)** 으로 메우는 방법들이라는 뜻이야.

---

## Paper title (p.63)

슬라이드 텍스트:
- Leveraging Class Distributions in CLIP for Weakly Supervised Semantic Segmentation
- Ziqian Yang, Xinqiao Zhao, Xiaolei Wang, Quan Zhang, Jimin Xiao (Xi'an Jiaotong-Liverpool University, University of Liverpool)
- 출처 줄: "Yang et.al., Leveraging Class Distributions in CLIP for Weakly Supervised Semantic Segmentation, CVPR 2026"
- 그림: CVF Open Access 판 논문 첫 페이지 캡처 (제목과 저자만)

**설명**
- 첫 번째 세미나 논문 **CD-CLIP**이야. 이름은 p.66 제목("An overview of our framework CD-CLIP")에 처음 나와.
- 이름 풀이는 슬라이드엔 없고, 논문 abstract에 "CD-CLIP (Class Distribution based CLIP)"이라고 적혀 있어.
- **CLIP** = 사진과 글을 같은 embedding 공간에 놓도록 학습한 image–text 모델이야 (L05). 사진 patch feature와 "a photo of a dog" 같은 class 이름 text feature의 cosine similarity로 "이 patch가 어느 class 같나"를 점수로 낼 수 있어.

> 보충: 학회 표기 확인함 — CVPR 2026 (CVF open access, pp. 34714–34723). 슬라이드 표기와 같아.

---

## Can CLIP Recover the Whole Object from Image-Level Labels? (p.64)

슬라이드 텍스트:
- Problem 1. **Incomplete activation**
  - CAM focuses on the most discriminative part.
  - Other object regions remain inactive.
- Problem 2. **Inaccurate boundaries**
  - Affinity propagation may spread activation.
  - Foreground activation can cross class boundaries.
- How can we expand CAMs without losing object boundaries?
- 그림: Original Image(골든 리트리버) → Initial CAM(머리만 빨갛게, "Discriminative region only"). 오른쪽 ① Incomplete Activation: 개 실루엣 중 머리만 빨강, "Missing object regions". ② Boundary Leakage: 개 전체가 빨간데 빨강이 잔디(배경)까지 번짐, "Activation crosses boundaries"

**설명**
- 문제가 **두 개이고 방향이 반대**야.
  1. **덜 켜짐 (under-activation)**: 머리만 켜지고 몸통은 꺼져 있어.
  2. **너무 번짐 (boundary leakage)**: 덜 켜진 걸 고치려고 **affinity propagation**(비슷한 patch끼리 활성값을 나눠 주기)을 하면, 이번엔 활성이 물체 경계를 넘어 배경이나 다른 class까지 퍼져.
- **affinity** = 두 patch(또는 pixel)가 "같은 물체에 속할 것 같은 정도"를 숫자로 적은 표야. patch가 N개면 N×N 행렬이야.
- 그래서 질문이 "**넓히되 경계는 지키려면?**"이야. 뒤에서 넓히기는 **CDA**, 경계 지키기는 **SBE**가 맡아 (p.74).

---

## Target Score Is Not Enough (p.65)

슬라이드 텍스트:
- Conventional affinity
  - Uses feature or attention similarity between patches.
  - Similar appearance does not always imply the same class.
  - Incorrect affinity causes under-activation or boundary leakage.
- CD-CLIP observation
  - Every **patch** has scores **over multiple classes**.
  - Patches of the same semantic class show **similar class distributions**.
  - The full distribution can be more informative than one target-class score.
- A patch is characterized not only by its target-class score, but by its distribution over all classes.
- 그림 (a) "CAMs refined by MHSA Affinity": 같은 사진(오토바이 탄 사람)에 대해 class마다 CAM 8장('Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle', 'Motorbike', 'Person'(초록 글씨), 'Pottedplant'). 사진마다 빨간 점선 상자(머리·헬멧 쪽 patch)와 검은 점선 상자(몸통 쪽 patch)가 같은 자리에 표시돼 있음
- 그림 (b) "Class Distributions": 검은 상자 patch와 빨간 상자 patch의 막대그래프(세로축 Probability, 가로축 20 classes). 초록 막대(Person)는 검은 쪽이 높고 빨간 쪽은 낮음. 두 분포를 JS에 넣으면 "Highly similar!"

**설명**
- **기존 affinity**는 patch feature나 attention이 비슷하면 "같은 물체"라고 봐. 그런데 **겉모습이 비슷하다고 같은 class는 아니야** (p.65). 이 affinity가 틀리면 두 문제가 다 생겨: 이어야 할 patch를 못 이으면 under-activation, 끊어야 할 patch를 이으면 boundary leakage.
- (a)의 'MHSA Affinity' = ViT의 **MHSA(Multi-Head Self-Attention)** 에서 나온 patch끼리의 attention 값을 affinity로 쓴 거야. 이걸로 CAM을 다듬어도 'Person' CAM에서 빨간 상자(그림상 머리·헬멧 쪽)와 검은 상자(몸통 쪽)의 활성 정도가 달라 보여 (그림 해석).
- **CD-CLIP의 관찰**: patch 하나는 target class 점수 하나만 갖는 게 아니라 **모든 class에 대한 점수 묶음**을 갖고 있어.
  - (b)에서 두 patch의 'Person' 점수(초록 막대)는 많이 달라. 그것만 보면 다른 물체처럼 보여.
  - 그런데 막대 전체의 모양(분포)은 거의 같아. 그래서 JS divergence로 재면 "Highly similar!"야.
- 핵심 문장: "A patch is characterized not only by its target-class score, but by its distribution over all classes." (p.65)

> 보충 (논문): 그림 설명에 "The selected patches in the same target class are marked in 'red box' and 'black box'"라고 돼 있고, 본문에 "high 'person' score for black patch vs. low score for red patch"라고 적혀 있어. 두 patch 모두 사람(같은 target class)이야.

> 보충 (RL 비유, 슬라이드 밖 = 시험 범위 아님): 두 state가 비슷한지 볼 때 greedy action 하나의 Q값만 비교하지 않고 **policy 분포 π(·|s) 전체**를 비교하는 것과 닮았어. 한 칸 점수는 달라도 전체 모양이 같으면 같은 상황으로 보는 거야.

---

## An overview of our framework CD-CLIP (p.66)

슬라이드 텍스트:
- An overview of our framework CD-CLIP
- How can we determine whether two patches belong to the same semantic region?
- 그림 (전체 구조도):
  - 왼쪽 위: Super Classes(빨강) + Target Classes(초록) → Prompt → **Text Encoder**(눈꽃 = Frozen) → Text Features (Super Class 빨강 칸, Target Class 초록 칸)
  - 가운데: Image → **Visual Encoder**(Frozen) → (No gradient) Image Features → Text Features와 **S(Cosine similarity)** → Attention Maps
  - 오른쪽 위 노란 상자 **Class Distribution Aware (CDA)**: Attention Maps에서 patch 두 개의 막대그래프(Target Classes 주황 + Super Classes 파랑) → **JS** → MHSA에서 온 Initial Affinity를 고쳐서 → CDA Affinity
  - 가운데 아래: Image Features → Class logits(빨강 = super, 초록 = target) → 점선 상자 안 **Super CAM**, **CAM** (CDA Affinity 화살표가 이 상자로 들어감) → Pseudo label → $\mathcal{L}_{seg}$
  - 오른쪽: Image Features → **Decoder** → Seg ($\mathcal{L}_{seg}$와 $\mathcal{L}_b$ 둘 다로 감)
  - 아래 초록 상자 **Super-class Boundary Exploration (SBE)**: Image → **DINO Encoder**(Frozen) → (No gradient) DINO Features → Super CAM과 **P(Class Average Pooling)** → Super-Class Prototypes → DINO Features와 **S** → Prototype CAM → Pseudo label → $\mathcal{L}_b$ (Class Label도 $\mathcal{L}_b$로 들어감)
  - 범례: S = Cosine similarity, P = Class Average Pooling, JS = J-S Divergence, 검은 화살표 = Forward, 빗금 화살표 = No gradient, 파란 점선 = Gradient, 빨간 화살표 = Loss, 눈꽃 = Frozen

**설명**
- 두 모듈 이름이 그림에 풀려 있어: **CDA = Class Distribution Aware**, **SBE = Super-class Boundary Exploration**.
- 흐름을 세 줄로 줄이면 이래.
  1. CLIP으로 patch마다 class 점수(분포)를 내고 CAM을 만들어.
  2. **CDA**: 분포가 비슷한 patch끼리 affinity를 다시 매기고, 그 affinity로 CAM을 퍼뜨려서 **덜 켜진 곳을 채워**.
  3. **SBE**: DINO feature로 super-class 영역을 깔끔하게 잡아서 **경계 supervision**을 줘.
- encoder 세 개(Text, Visual, DINO)에 전부 **눈꽃(Frozen)** 이 붙어 있어. 학습 신호(빨간 Loss 화살표)는 **Decoder**의 출력 Seg로 가.
- **super-class** = target class 여러 개를 묶은 상위 class야. 논문 예시는 'vehicle', 'animal'이고, 'cat'은 'animal'에 속해 (논문 보충, 슬라이드엔 예시 없음).
- 아래 질문 "How can we determine whether two patches belong to the same semantic region?"이 CDA의 질문이야.

> 보충 (논문 본문, 시험 범위 아님): Visual Encoder는 frozen CLIP ViT-B/16, DINO Encoder는 frozen DINOv2 ViT-B/14, Decoder는 WeCLIP식 가벼운 transformer decoder야. 그림의 파란 점선 "Gradient"는 CAM을 만들 때 class logit의 gradient로 feature map 가중치를 정한다는 표시야 (논문: "$W^c_k$ … derived from the gradient between class logits and feature maps"). encoder를 학습시키는 gradient가 아니야.

---

## Each Patch Has a Class Distribution (p.67)

슬라이드 텍스트:
- A patch is described by its distribution over classes, not only by one target-class score.
- $\mathbf{S_a} = \text{Min-max}(\text{CosSim}(\mathbf{F}, \mathbf{T}_{all}))$,
- Target classes provide specific semantics.
- Super-classes provide broader semantic relationships.
- Each patch is represented by its scores over all classes.
- 그림: p.66 구조도에서 Text Features · Image Features · S · Attention Maps 부분을 빨간 네모로 강조

**설명**
- 기호 풀이:
  - $\mathbf{F}$ = CLIP Visual Encoder가 낸 **patch별 image feature**야.
  - $\mathbf{T}_{all}$ = **모든 class**(target class + super-class)의 text feature야. "all"이 핵심이야.
  - $\text{CosSim}$ = cosine similarity. patch 하나와 class 하나의 방향이 비슷할수록 커.
  - $\text{Min-max}$ = 값을 0–1 범위로 맞추는 정규화야: $(x-\min)/(\max-\min)$.
  - $\mathbf{S_a}$ = 결과. patch마다 class 개수만큼 점수가 있는 표야 (그림의 Attention Maps). 한 patch의 줄 하나가 그 patch의 "class 점수 묶음"이야.
- 왜 super-class까지 넣어?
  - target class는 "구체적인 뜻"(dog, cat)을 줘.
  - super-class는 "더 넓은 관계"(animal)를 줘. 점수 칸이 늘어나니까 분포가 더 풍부해져.

> 보충 (논문): 논문에선 $\mathbf{T}_{all} \in \mathbb{R}^{|C+N| \times d}$ (C = target class 수, N = super-class 수)로 적어.

---

## From Class Distributions to Patch Affinity (p.68)

슬라이드 텍스트:
- From Class Distributions to Patch Affinity
- How are similar class distributions used to complete the CAM?
- 그림: p.66 구조도에서 CDA(노란 상자) 부분을 빨간 네모로 강조

**설명**
- 다음 장(p.69)으로 넘어가는 질문 슬라이드야. "분포가 비슷하다"는 정보를 **affinity → CAM 채우기**로 어떻게 연결하나가 질문이야.

---

## From Class Distributions to Patch Affinity (p.69)

슬라이드 텍스트:
- 1. Compare two **patch distributions**
  - Small JS divergence: similar class distributions
  - Large JS divergence: different class distributions
- 2. **Correct** the original **affinity**
  - Distribution similarity filters unreliable relationships in the original affinity.
- 3. **Propagate** CAM activation
  - The head is strongly activated in the initial CAM.
  - The body has a similar class distribution.
  - CDA propagates activation from the head to the body.
- CDA connects semantically similar patches and propagates activation to missing object regions.
- 수식:
  - $\mathbf{S_d} = \dfrac{(1 - \mathbf{D_{JS}}(\text{Softmax}(\mathbf{S_a}) \,\|\, \text{Softmax}(\mathbf{S_a})^{\mathrm{T}}))}{\tau}$,
  - $\mathbf{A}_d = \text{Softmax}(\mathbf{S_d}) \cdot \mathbf{A}_{init}$,
  - $\mathbf{M}^c_{re} = \mathbf{M}^c_{init} \cdot (\mathbf{A}_d)^t$.
- 그림: CDA 상자 확대 + Attention Maps에서 두 막대그래프로 가는 화살표 + CDA Affinity에서 Super CAM / CAM 상자로 내려가는 화살표 + Pseudo label → $\mathcal{L}_{seg}$ ← Decoder

**설명**
- **어떻게** (세 단계)
  1. **비교**: $\text{Softmax}(\mathbf{S_a})$로 patch마다 점수를 확률 분포로 바꾸고, patch 두 개씩 **JS divergence**를 재. 작으면 비슷한 분포, 크면 다른 분포야.
  2. **고치기**: $1 - D_{JS}$는 "비슷할수록 큰 값"이야. 이걸 $\tau$(temperature)로 나눈 게 $\mathbf{S_d}$(분포 유사도)고, softmax를 씌워서 MHSA에서 온 원래 affinity $\mathbf{A}_{init}$에 곱해. 분포가 다른 patch 쌍의 관계는 작아져. 이게 "filters unreliable relationships"야. 결과가 **CDA affinity** $\mathbf{A}_d$야. (슬라이드·논문 모두 '·' 연산을 따로 정의하진 않아. '필터' 설명상 patch 쌍별(element-wise) 곱으로 읽는 게 자연스럽고, 3단계의 $(\mathbf{A}_d)^t$는 행렬 거듭제곱이야.)
  3. **퍼뜨리기**: 처음 CAM $\mathbf{M}^c_{init}$에 $\mathbf{A}_d$를 $t$번 곱해. 한 번 곱할 때마다 활성값이 "비슷한 patch"로 한 칸씩 퍼져. 머리(강하게 켜짐)와 몸통(분포가 비슷함)이 이어져 있으니까 머리의 활성이 몸통으로 넘어가.
- **JS divergence(Jensen–Shannon divergence)** 정의 (기초 정의, 슬라이드엔 기호만 있음):
  $$D_{JS}(P\|Q) = \tfrac12 D_{KL}(P\|M) + \tfrac12 D_{KL}(Q\|M), \quad M = \tfrac12(P+Q)$$
  - $D_{KL}(P\|Q) = \sum_k P_k \log \frac{P_k}{Q_k}$ = 분포 $P$가 $Q$와 얼마나 다른지. 순서를 바꾸면 값이 달라져 (비대칭).
  - JS는 두 분포의 **평균 $M$** 을 기준으로 KL을 양쪽에서 재서 반씩 더해. 그래서 **대칭**이고 값이 **유한**해 (log 밑이 2면 0–1).
  - 같은 분포면 0, 겹치는 데가 없으면 최대야.
- **기호 풀이**
  - $\mathbf{S_a}$: p.67의 patch×class 점수표.
  - $\text{Softmax}(\mathbf{S_a})$와 그 전치 $(\cdot)^{\mathrm{T}}$: (슬라이드엔 정의 없음, 해석) "모든 patch 쌍"을 비교한다는 표기야. 그래서 $\mathbf{S_d}$는 patch×patch 표야.
  - $\tau$: temperature. 작을수록 비슷한 쌍과 아닌 쌍의 차이가 softmax 뒤에 더 뾰족해져.
  - $\mathbf{A}_{init}$: MHSA attention으로 만든 원래 affinity.
  - $\mathbf{M}^c_{init}$, $\mathbf{M}^c_{re}$: class $c$의 처음 CAM과 다듬은(refined) CAM.
  - $t$: 퍼뜨리는 횟수 (논문: "refining iterations"). 행렬을 $t$제곱해.

> 보충 (설명용 toy 예시, 슬라이드 수치 아님): class 4개 [person, bicycle, motorbike, sky]. 머리 patch = [0.40, 0.30, 0.20, 0.10], 몸통 patch = [0.25, 0.35, 0.25, 0.15], 하늘 patch = [0.05, 0.05, 0.05, 0.85]. log 밑 2로 재면 $D_{JS}$(머리, 몸통) ≈ 0.02, $D_{JS}$(머리, 하늘) ≈ 0.46이야. person 점수만 보면 머리 0.40 vs 몸통 0.25로 꽤 다르지만, 분포 전체로 보면 머리–몸통은 거의 같고 머리–하늘은 확실히 달라.

> 보충 (논문): $\mathbf{A}_{init} = (\text{Sinkhorn}(\mathbf{A}) + \text{Sinkhorn}(\mathbf{A})^{\mathrm{T}})/2$로 MHSA attention $\mathbf{A}$를 정규화하고 대칭으로 만든 거야. 논문은 JS를 "a finite and symmetric metric"이라고 적고, "$\tau$ is the temperature to control the softness"라고 설명해.

> 보충 (RL 비유, 슬라이드 밖 = 시험 범위 아님): $\mathbf{M}\cdot\mathbf{A}^t$는 확률 분포에 **transition matrix를 $t$번 곱하는 것**(Markov chain $t$-step)과 같은 모양이야. affinity가 transition matrix 역할을 하고, 활성값이 "비슷한 patch"로 random walk하듯 퍼져. 단 $\mathbf{A}_d$의 줄 합이 정확히 1이라고 슬라이드가 말하진 않아서, 모양이 닮았다는 정도로만 봐.

---

## Complete CAMs Can Cross Boundaries (p.70)

슬라이드 텍스트:
- CDA expands activation—but where should it stop?
  - 1. Construct a **super-class prototype**
    - Use the **CLIP** super-class **CAM** as a semantic seed and average the corresponding DINO features.
    - $\mathbf{f}^n_s = \text{CAP}(\mathbf{M}^n_s \odot \mathbf{F}_d)$
  - 2. Generate a **prototype CAM**
    - Compare the prototype with every **DINO** patch to recover a complete and structurally coherent super-class region.
    - $\mathbf{M}^n_p = \text{ReLU}(\text{CosSim}(\mathbf{f}^n_s, \mathbf{F}_d))$
- 그림: p.66 구조도의 아랫부분 (DINO Encoder → DINO Features → P → Super-Class Prototypes → S → Prototype CAM → Pseudo label → $\mathcal{L}_b$ ← Seg, Class Label) + 범례

**설명**
- CDA로 넓혔더니 이번엔 **어디서 멈추나**가 문제야 (p.64의 Problem 2, boundary leakage). 그래서 SBE가 나와.
- **prototype** = 한 그룹을 대표하는 feature 벡터 하나야. 보통 그 그룹에 속한 feature들의 평균이야.
- **어떻게**
  1. **super-class prototype 만들기**: CLIP의 super-class CAM $\mathbf{M}^n_s$로 "animal이 있는 자리"를 표시하고(semantic seed), 그 자리의 **DINO feature** $\mathbf{F}_d$를 평균 내. 결과 $\mathbf{f}^n_s$가 super-class $n$의 prototype이야.
  2. **prototype CAM 만들기**: prototype을 **모든 DINO patch**와 cosine similarity로 비교하고, 음수는 ReLU로 0으로 만들어. 결과 $\mathbf{M}^n_p$가 super-class $n$의 영역 지도야.
- 기호 풀이
  - $\odot$: 원소별 곱. CAM을 마스크처럼 써서 그 자리 feature만 남겨.
  - $\text{CAP}$: Class Average Pooling (그림 범례 P). 남긴 feature를 평균 내서 벡터 하나로 만들어.
  - $\mathbf{F}_d$: DINO feature. $n$: super-class 번호.
- 왜 DINO야? DINO feature는 물체의 **구조**가 잘 갈려서, prototype과 비교하면 "complete and structurally coherent" 영역이 나와 (p.70). 반대로 DINO는 class 이름을 몰라. 그래서 이름표는 CLIP이 붙여 줘 (p.71).
- **DINO** = label 없이 self-distillation으로 학습한 ViT야 (L04). DINO attention이 물체 모양을 잘 따라간다는 게 L04에서 본 특징이야.

> 보충 (논문): 논문은 "DINO features cannot provide class-specific" 정보라서 CLIP의 super-class CAM을 class-specific mask로 쓴다고 설명해. 또 DINO feature가 "superior representational capacity for super-class region segmentation"을 보인다고 해.

---

## CLIP gives the prototype a semantic identity, and DINO expands it into a coherent region (p.71)

슬라이드 텍스트:
- CLIP gives the prototype a semantic identity, and DINO expands it into a coherent region
- 그림: p.66과 같은 CD-CLIP 전체 구조도

**설명**
- 제목 한 문장이 SBE의 요약이야. **역할 분담**을 외워 둬.
  - **CLIP**: prototype에 **이름(semantic identity)** 을 줘. "이건 animal이야."
  - **DINO**: 그 prototype을 **빈틈없고 모양이 맞는 영역(coherent region)** 으로 넓혀 줘.
- 뒤집은 문장("DINO gives the identity, CLIP expands the region")은 틀린 거야.

---

## Training objective (p.72)

슬라이드 텍스트:
- Boundary-enhanced loss
  - Apply boundary loss only when there are two or more distinct superclasses.
- $\mathcal{L} = \mathcal{L}_{seg} + \lambda_1 \mathcal{L}_b.$
- $\mathcal{L}_b = \begin{cases} \text{Dice}(\mathbf{P}, \mathbf{Y}_s) & \text{if } |\mathbf{C}_p| \ge 2, \\ 0, & \text{otherwise.} \end{cases}$

**설명**
- 전체 loss = segmentation loss + $\lambda_1$ × boundary-enhanced loss.
  - $\mathcal{L}_{seg}$: decoder 출력 Seg를 **CDA로 다듬은 CAM에서 나온 pseudo label**에 맞춰 (p.66 그림).
  - $\mathcal{L}_b$: Seg를 **prototype CAM에서 나온 super-class pseudo label $\mathbf{Y}_s$** 에 맞춰.
  - $\lambda_1$: 두 loss의 비율.
- 기호 풀이 (논문 확인)
  - $\mathbf{P}$: segmentation prediction (decoder 출력).
  - $\mathbf{Y}_s$: prototype CAM $\mathbf{M}^n_p$로 만든 pseudo label.
  - $|\mathbf{C}_p|$: 그 사진에 **있는 super-class 개수**. image label의 class를 super-class로 바꿔서 세 (예: 'cat' → 'animal').
- **Dice loss**: 슬라이드의 $\text{Dice}(\mathbf{P}, \mathbf{Y}_s)$는 이미 **Dice loss**야 (논문: "Dice(·, ·) represents the Dice loss"). 바탕이 되는 Dice coefficient는 $\frac{2|P \cap Y|}{|P| + |Y|}$ (두 영역이 겹칠수록 1)이고, loss는 보통 $1 -$ coefficient 꼴이야. 겹칠수록 loss가 작아.
- **왜 super-class가 2개 이상일 때만?** 경계 supervision은 **서로 다른 super-class가 맞닿은 곳**에서 의미가 있어 (p.73 "Boundary supervision is most meaningful when different semantic super-classes coexist."). super-class가 하나뿐이면 갈라 줄 경계가 없으니까 0으로 꺼.

> 보충 (논문 본문 값, 시험 범위 아님): $\lambda_1 = 0.6$으로 썼어. 논문은 이 조건을 "To ensure accurate supervision while minimizing computational cost"라고 설명해.

---

## Does Each Component Help? (p.73)

슬라이드 텍스트:
- CDA significantly improves the CAM mIoU from 70.1 to 80.3.
- SBE provides a greater improvement to the final segmentation performance than the CAM itself.
- This is because SBE contributes to boundary learning within the decoder, rather than to activation expansion.
- Boundary supervision is most meaningful when different semantic super-classes coexist.
- 표 "Table 3. Ablation study of two modules in our method, 'M' denotes the mIoU (%) of CAM performance. 'Seg.' denotes the mIoU (%) of segmentation performance."

  | # | CAA | RFM | CDA | SBE | M | Seg. |
  |---|---|---|---|---|---|---|
  | 0 | No CAM refinement | | | | 70.1 | 68.8 |
  | 1 | ✓ | | | | 73.2 | 72.8 |
  | 2 | | ✓ | | | 77.4 | 76.1 |
  | 3 | | | ✓ | | 80.3 | 80.2 |
  | 4 | ✓ | | ✓ | | 76.8 | 76.4 |
  | 5 | | ✓ | ✓ | | 80.1 | 80.2 |
  | 6 | | | ✓ | ✓ | 80.8 | 81.6 |
- 그림 "Figure 4. Visualization comparison of CAMs between other modules and our CDA method. The experimental dataset is PASCAL VOC 2012 train split." (열: CLIP, +CAA, +RFM, +CDA)

**설명**
- **mIoU(mean Intersection over Union)** = class마다 예측 영역과 정답 영역의 IoU(겹친 넓이 ÷ 합친 넓이)를 구해서 평균 낸 값이야. segmentation의 기본 점수야.
- 두 열의 차이가 포인트야.
  - **M** = CAM 자체의 mIoU. **CDA**는 여기를 크게 올려: 70.1 → 80.3 (#0 → #3).
  - **Seg.** = 최종 segmentation의 mIoU. **SBE**를 더하면 M은 조금(80.3 → 80.8) 오르지만 Seg.는 더 올라(80.2 → 81.6) (#3 → #6).
  - 다른 모듈과 합친다고 항상 오르는 건 아니야: CAA+CDA(#4)는 CDA 단독(#3)보다 낮아 (M 76.8 < 80.3, Seg. 76.4 < 80.2). RFM+CDA(#5)는 CDA 단독과 거의 같아 (80.1 / 80.2). 슬라이드는 이유를 안 적었고, 논문은 "When combined with the CAA, the performance increase is limited (Setting #4), as the CAA module contains the regional constraint mask"라고 설명해.
- 이유: SBE는 CAM을 넓히는 게 아니라 **decoder가 경계를 배우게** 해 ("contributes to boundary learning within the decoder, rather than to activation expansion").
- CAA, RFM은 비교용 다른 CAM 보정 모듈이야. 슬라이드엔 이름 풀이가 없어.
- 수치는 시험 비중이 낮아. "CDA → CAM, SBE → 최종 Seg"라는 방향만 기억해.

> 보충 (논문): CAA = Class-aware Attention-based Affinity (CLIP-ES에서 온 모듈), RFM = Frozen CLIP CAM Refinement module (WeCLIP에서 온 모듈)이야.

---

## Final Results and Takeaway (p.74)

슬라이드 텍스트:
- CD-CLIP addresses two complementary problems:
  - 1. CDA: CAM completeness
    - Class distributions → better patch affinity → complete CAM
  - 2. SBE: Boundary accuracy
    - DINO structure → super-class prototypes → boundary supervision
- Better WSSS requires both knowing which patches belong together and knowing where their relationship should stop.
- 그림: 정성 비교. 행 = Image, ToCo, WeCLIP, Ours, GT. 열 7개, 왼쪽 2열은 "Single Class", 오른쪽 5열은 "Multiple Classes"

**설명**
- 한 장 요약이야. 두 문제가 **서로 보완(complementary)** 관계야.
  - **CDA → completeness**: 어떤 patch끼리 **같이 가야 하나** (which patches belong together).
  - **SBE → boundary accuracy**: 그 관계가 **어디서 멈춰야 하나** (where their relationship should stop).
- 그림은 "Single Class"(앞 2열)와 "Multiple Classes"(뒤 5열)로 나뉘어 있어. Multiple Classes 열엔 고양이와 의자, 식탁과 사람들 같은 사진이 있어 (논문 본문: "'dining table' with 'people' and 'chair' with 'cat'"). (해석, 슬라이드가 직접 연결하진 않음: class가 여러 개인 장면에서 경계를 가르는 게 p.73 "Boundary supervision is most meaningful when different semantic super-classes coexist"가 말하는 상황이야.)
- p.64와 짝지어 보면 깔끔해: Problem 1 Incomplete activation ↔ CDA, Problem 2 Inaccurate boundaries ↔ SBE.

> 보충: ToCo(CVPR 2023), WeCLIP(CVPR 2024)은 비교 대상인 이전 WSSS 방법이야 (논문 표 기준). 이 강의에선 이름만 나와.
