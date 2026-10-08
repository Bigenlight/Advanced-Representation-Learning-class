<!-- DRAFT | lecture: 03_Contrastive_Learning_I | chunk: 06 | pages 41–48 -->
<!-- VERIFIED | fixes: 2 | verifier: sonnet -->

## GenView Framework — Apply SSL loss (p.41)

- 제목 bullet: "Apply SSL loss"
- 그림: GenView 전체 파이프라인 중 **Quality-Driven Contrastive Loss** 블록(회색-파랑 배경 박스). 왼쪽에 원본 이미지 $\mathbf{P}_i$(강아지 사진)와 생성된 positive view $\mathbf{P}_i^+$(다른 배경/포즈의 강아지 사진)가 위·아래로 놓임. 각각 **Encoder $f$** (회색 박스)로 들어가 representation을 만들고, 이 두 representation이 가운데의 빨간 타원으로 강조된 식 $\tilde{\mathcal{L}}_{\mathrm{SSL},*} = w_i \cdot \mathcal{L}_{\mathrm{SSL},*}$ 로 모임.
- 같은 이미지는 동시에 눈송이(❄) 아이콘의 **frozen encoder $E$** 로도 들어가 feature map $\mathbf{F}_i$, $\mathbf{F}_i^+$ (파란 3겹 큐브) 생성 → foreground mask $\mathbf{M}_i^f$ / background mask $\mathbf{M}_i^b$ (흑백 heatmap) 와 $\otimes$ (곱) → 벡터 $\mathbf{z}_i^f, \mathbf{z}_i^b, \mathbf{z}_i^{f+}, \mathbf{z}_i^{b+}$ → 오른쪽 주황 박스 $q_i = s_i^f - s_i^b$ → 주황 화살표로 $w_i = \mathrm{softmax}(q_i)$ 가 왼쪽 loss 식으로 되돌아옴.
- 오른쪽 텍스트:
  - Train the SSL Encoder
    - Input the two positive views into the target encoder
    - Calculate the base SSL loss using the two representations:
      - Various loss functions can be used

**설명**
- 이 단계는 GenView에서 **실제로 SSL encoder를 학습시키는 단계**임. 원본 $\mathbf{P}_i$ 와 생성된 view $\mathbf{P}_i^+$ 를 positive pair로 보고 target encoder $f$ 에 넣어 representation을 얻고, base SSL loss $\mathcal{L}_{\mathrm{SSL},*}$ 를 계산함.
- $*$ 는 어떤 SSL loss든 들어갈 수 있다는 placeholder임 (InfoNCE/SimCLR, MoCo, BYOL, SimSiam, SwAV 등) → GenView는 **loss-agnostic, plug-in 방식**의 view construction 기법. p.44 표에서 여러 프레임워크에 붙여 쓴 것이 이를 뒷받침함.
- 차이점: 기존 방식은 한 이미지에 augmentation(crop, color jitter 등)을 두 번 적용해 positive pair를 만들었는데, GenView는 **생성모델로 만든 view**를 pair로 쓰고, 그 pair의 **품질 $q_i$ 로 loss를 가중**($w_i$)함. 가중치 계산은 gradient가 흐르지 않는 frozen encoder $E$ 쪽에서 이뤄짐(품질 평가는 학습 대상이 아님).
- $w_i = \mathrm{softmax}(q_i)$ 는 배치 내 샘플들에 대한 softmax로 보는 것이 자연스러움(배치 전체에서 품질 높은 pair에 상대적으로 큰 가중). 슬라이드에는 정규화 범위가 명시되지 않음.

> 보충: 이 슬라이드의 GenView는 Stable unCLIP과 frozen CLIP을 활용해 positive view를 생성하는 방법으로, 앞 슬라이드(p.37~40)에서 다룬 내용의 학습 단계에 해당함. 논문 세부(저자/정확한 제목)는 앞 chunk 참조.

## GenView Framework — Evaluate Pair Quality (p.42)

- 제목 bullet: "Evaluate Pair Quality"
- 그림: p.41과 동일한 파이프라인이되 **빨간 타원 2개**가 frozen encoder $E$ 이후 부분(위쪽: $\mathbf{P}_i$ 쪽 $\mathbf{F}_i \to \mathbf{M}_i^f, \mathbf{M}_i^b \to \mathbf{z}_i^f, \mathbf{z}_i^b$; 아래쪽: $\mathbf{P}_i^+$ 쪽 $\mathbf{F}_i^+ \to \mathbf{M}_i^{f+}, \mathbf{M}_i^{b+} \to \mathbf{z}_i^{f+}, \mathbf{z}_i^{b+}$)을 강조. $\mathbf{z}$ 들에서 나온 파란 곡선 화살표들이 오른쪽 $q_i = s_i^f - s_i^b$ 박스로 모이고, 여기서 $w_i = \mathrm{softmax}(q_i)$ 가 loss 식 $\tilde{\mathcal{L}}_{\mathrm{SSL},*} = w_i \cdot \mathcal{L}_{\mathrm{SSL},*}$ 로 이어짐.
- 오른쪽 텍스트:
  - A separate frozen encoder evaluates the quality of the pair
    - First, extract the feature maps of the two views
    - Next, divide them into foreground and background, respectively
    - Calculate the similarity for the foreground and background, between the two views.
  - The conditions for a good positive pair are as follows:
    - Similar foreground
    - Different background
  - Give more weight to high-quality positive pairs
- 수식: $q_i = s_i^f - s_i^b$

**설명**
- 파이프라인: (1) frozen encoder $E$ 로 두 view의 feature map $\mathbf{F}_i, \mathbf{F}_i^+$ 추출 → (2) mask $\mathbf{M}^f$(foreground), $\mathbf{M}^b$(background)를 feature map에 곱해($\otimes$) pooling하면 foreground 벡터 $\mathbf{z}^f$ 와 background 벡터 $\mathbf{z}^b$ 를 얻음 → (3) 두 view 간 유사도를 foreground끼리 $s_i^f = \mathrm{sim}(\mathbf{z}_i^f, \mathbf{z}_i^{f+})$, background끼리 $s_i^b = \mathrm{sim}(\mathbf{z}_i^b, \mathbf{z}_i^{b+})$ 로 계산.
- **왜 $q_i = s_i^f - s_i^b$ 인가**: 좋은 positive pair는 **semantics(object, foreground)는 같고 nuisance(background)는 달라야** 함. 그래야 encoder가 background shortcut이 아니라 object 정보에 invariant해지도록 학습됨. 따라서 $s^f$ 는 크고 $s^b$ 는 작을수록 $q$ 가 커짐 → 품질 점수. 생성 view가 semantic drift로 object가 바뀌었다면 $s^f$ 가 낮아 $q$ 가 작아지고, background가 거의 그대로면(변화 없음, 정보량 적음) $s^b$ 가 커서 역시 $q$ 가 작아짐.
- $w_i = \mathrm{softmax}(q_i)$ 로 고품질 pair에 큰 loss 가중을 줘 noisy pair의 영향을 줄임 (p.45에서 말하는 semantic drift 완화 장치).
- 기호 정리: $i$ = 샘플 인덱스, $s$ = similarity(보통 cosine), 위첨자 $f/b$ = foreground/background, $+$ = positive(생성) view.

> 보충: foreground/background mask는 frozen CLIP 계열 feature map에서 만든 것으로 보이며(그림의 흑백 heatmap), 정확한 mask 생성 방식(attention/clustering 등)은 슬라이드에 없음.

## What Does GenView Generate? (p.43)

- 그림: 2개 데이터셋 × (원본 행 / 생성 view 행) 구성의 이미지 그리드. 세로축 라벨 왼쪽: **ImageNet**(위 블록), **CIFAR-10**(아래 블록, 점선으로 구분). 오른쪽 라벨: 각 블록에서 위 행 **Original images**, 아래 행 **Generated views**. 각 행 8장.
  - ImageNet: 사람+개(잔디 → 산 배경 앞 개와 사람), 빨간 배(다른 각도/배경), 식탁(다른 방 구도), 스키어(다른 포즈/배경), 새(다른 자세/배경), 물가의 개 2마리(다른 구도), 개(얼굴 클로즈업), 낚시한 남자(다른 사람/구도).
  - CIFAR-10: 분홍 밴(다른 위치/배경), 배, 비행기, 보트, 흰 SUV(다른 시점), 소방차/트럭, 사슴/동물(다른 포즈), 고양이 얼굴.
- 하단 텍스트: GenView changes: • Background • Object pose • Viewpoint / The main semantics remain unchanged.

**설명**
- 생성 view는 **같은 클래스/semantic의 object를 유지**하면서 배경, 포즈, 시점이 크게 달라짐. 일반 augmentation(crop, flip, color jitter)이 못 만드는 **의미 보존 + 큰 변형**을 얻는 것이 핵심.
- 이는 앞서 p.42의 "similar foreground, different background" 조건에 부합하는 pair임. 기존 augmentation은 픽셀 수준 변형이라 background/context shortcut이 남지만, 생성 view는 background 자체가 바뀜.
- 주의: 일부 예(예: 낚시 사진의 인물이 바뀜, 개 얼굴 클로즈업)는 semantic이 완전히 같다고 보기 어려움 → p.45의 semantic drift 한계와 연결. 그래서 quality weighting이 필요함.

## Does Better View Construction Help? (p.44)

- 제목 bullet: "ImageNet-1K linear evaluation with ResNet-50"
- 표 (Table 1: Linear evaluation on IN-1K. *: our reproduction.) — 좌측 표(기존 방법 비교, 열: Method / Architecture / Epochs / Top-1):

| Method | Architecture | Epochs | Top-1 |
|---|---|---|---|
| InstDisc [94] | ResNet-50 | 200 | 56.5 |
| SimCLR [16] | ResNet-50 | 200 | 66.8 |
| PCL [47] | ResNet-50 | 200 | 67.6 |
| Adco [67] | ResNet-50 | 200 | 68.6 |
| InfoMin [84] | ResNet-50 | 200 | 70.1 |
| NNCLR [23] | ResNet-50 | 200 | 70.7 |
| LEVEL [39] | ResNet-50 | 200 | 72.8 |
| Barlow Twins [102] | ResNet-50 | 300 | 71.4 |
| CLIP [68] | ResNet-50 | - | 74.3 |

- 우측 표 (GenView 적용 전/후; 굵은 글씨 = GenView 적용 결과):

| Method | Architecture | Epochs | Top-1 |
|---|---|---|---|
| MoCov2 [33] | ResNet-50 | 200 | 67.5 |
| MoCov2 + C-Crop [66] | ResNet-50 | 200 | 67.8 |
| MoCov2 + GenView | ResNet-50 | 200 | **70.0** |
| SwAV [12]* | ResNet-50 | 200 | 70.5 |
| SwAV + GenView | ResNet-50 | 200 | **71.7** |
| SimSiam [18] | ResNet-50 | 200 | 70.0 |
| SimSiam + GenView | ResNet-50 | 200 | **72.2** |
| BYOL [30]* | ResNet-50 | 200 | 71.8 |
| BYOL + GenView | ResNet-50 | 200 | **73.2** |
| MoCov3 [19] | ResNet-50 | 100 | 68.9 |
| MoCov3 + GenView | ResNet-50 | 100 | **72.7** |
| MoCov3 [19] | ResNet-50 | 300 | 72.8 |
| MoCov3 + GenView | ResNet-50 | 300 | **74.8** |
| MoCov3 [19] | ViT-S | 300 | 73.2 |
| MoCov3 + GenView | ViT-S | 300 | **74.5** |
| MoCov3 [19] | ViT-B | 300 | 76.7 |
| MoCov3 + GenView | ViT-B | 300 | **77.8** |

- 표 아래 텍스트:
  - GenView improves different SSL frameworks.
    - The benefit comes from better pair construction,
    - not simply from adding more images.
- 하단 굵은 문장: **Contrastive learning depends on the quality of its positive views.**

**설명**
- 평가 프로토콜은 **linear evaluation**: backbone을 freeze하고 그 위에 linear classifier만 학습해 ImageNet-1K Top-1 정확도를 측정 → representation 자체의 품질을 보는 지표.
- 주요 관찰: (1) MoCov2/SwAV/SimSiam/BYOL/MoCov3 등 **서로 다른 SSL 프레임워크 모두에서 향상** (+1.1 ~ +3.8%p). 특히 MoCov3 ResNet-50 100 epoch는 68.9 → 72.7 (+3.8)로 가장 큼, ViT-B도 76.7 → 77.8. (2) MoCov2 + C-Crop(67.8)은 다른 view-construction 기법인데 개선이 +0.3에 그친 반면 GenView는 +2.5 → 더 나은 view construction이 효과적. (3) MoCov3 100 epoch + GenView(72.7)가 GenView 없는 300 epoch(72.8)에 거의 근접 → 데이터/학습 효율 향상 시사.
- "not simply from adding more images"는 생성 이미지로 데이터 양이 늘어난 효과가 아니라 **pair의 질(foreground 유지·다양한 background)** 때문이라는 주장임. ← 슬라이드 표만으로는 이를 직접 입증하지 못함(데이터 양을 맞춘 ablation은 논문에 있을 가능성; 슬라이드에 없음).
- 왼쪽 표는 기존 SSL 방법들의 baseline 성능 참고용이고, CLIP(74.3)은 대규모 image-text 데이터로 학습되어 조건이 다름(Epochs 칸 "-").

> 보충: 좌측 표의 숫자는 슬라이드에서 그대로 옮김. 방법 인용번호([94] 등)는 논문 reference 번호.

## Limitations (p.45)

- It relies on prior knowledge from Frozen CLIP and Stable unCLIP.
  - Therefore, it is **not a fully independent SSL** method trained solely on the target dataset.
- While image generation is performed offline, it **requires additional computation and storage space**.
- Determining noise levels based on the foreground ratio is a relatively simple **heuristic**.
- Quality weighting mitigates the impact of semantic drift but **does not eliminate it entirely**.

**설명**
- **외부 prior 의존**: frozen CLIP(mask/품질 평가)과 Stable unCLIP(view 생성)은 대규모 외부 데이터로 사전학습된 모델 → 순수 self-supervised(target dataset만 사용)라 보기 어렵고, 다른 SSL과의 비교가 완전히 공정하지 않을 수 있음.
- **비용**: 생성이 offline이라 학습 중 오버헤드는 없지만 사전 생성 연산량과 저장 공간이 필요함.
- **heuristic**: p.40의 adaptive view generation(foreground 비율이 작으면 noise 적게, 크면 많이)은 단순 규칙이라 최적이라는 보장이 없음.
- **semantic drift 잔존**: $w_i$ 가중은 나쁜 pair의 영향을 줄일 뿐 제거하지는 못함 (p.43의 일부 예처럼 의미가 변한 view가 여전히 섞일 수 있음).

## 섹션 구분: 질문 슬라이드 (p.46)

- 한 줄: "Even with good positive pairs, does contrastive learning preserve every useful feature?" (전환 질문, 남색 배경 — 좋은 positive pair가 있어도 feature가 다 보존되는지 묻는 다음 주제 feature suppression 도입)

## Learning the Unlearned: Mitigating Feature Suppression in Contrastive Learning (p.47)

- 논문 첫 페이지 캡처 슬라이드. 제목: **Learning the Unlearned: Mitigating Feature Suppression in Contrastive Learning**, 저자: Jihai Zhang(1), Xiang Lan(2), Xiaoye Qu(3), Yu Cheng(1), Mengling Feng(2), Bryan Hooi(2). 소속: (1) The Chinese University of Hong Kong, (2) National University of Singapore, (3) Shanghai AI Laboratory. 우상단에 "Check for updates" 배지.
- 하단 인용: Zhang et al., ECCV 2024, Learning the Unlearned: Mitigating Feature Suppression in Contrastive Learning

**설명**
- 다음 주제인 **feature suppression**을 다루는 논문 소개 슬라이드임. 이 논문은 contrastive learning(SimCLR, CLIP 등)에서 일부 feature가 학습되면서 다른 유용한 feature가 억제되는 현상을 분석하고 완화하는 방법을 제안함 (방법 세부는 이 chunk 범위 밖; 다음 chunk 참조).
- 앞 주제(GenView)가 "좋은 positive pair 구성"이라는 **데이터/입력 측** 개선이었다면, 이 주제는 "pair가 좋아도 학습 결과 representation이 정보를 잃는" **representation 측** 문제임.

## Feature Suppression (p.48)

- 제목 bullet: "The model learns some features but ignores other useful features"
- 그림 (좌: SimCLR): 회색 배경 위 초록색 격자무늬 texture의 두 이미지 — 왼쪽은 오각형(**Shape: Pentagon**), 오른쪽은 원(**Shape: Circle**). 각 이미지에서 점선 파란 화살표가 아래 파란 평행사변형 **SimCLR Space** 안의 두 점(서로 매우 가까움)으로 이어지고 **Similarity=0.935**. 위쪽에는 초록 화살표(Pentagon 이미지에서 위로, shape 라벨을 가리킴)와 빨간 화살표(Circle 이미지에서 Pentagon 라벨 쪽으로)로 "shape 기준으로는 서로 다르다"는 점을 표시.
- 그림 (우: CLIP): 토끼 사진 2장 — 왼쪽은 바구니 옆 부활절 달걀과 토끼, 오른쪽은 풀밭의 회색 토끼. 각각 주황 평행사변형 **CLIP Space** 의 매우 가까운 두 점으로 매핑되고 **Similarity=0.933**. 위쪽 캡션 라벨은 왼쪽 "a rabbit facing left", 오른쪽 "a rabbit facing right"이며, 오른쪽 이미지에서 초록 화살표는 "facing right"(맞는 캡션)로, 빨간 화살표는 "facing left"(틀린 캡션) 쪽으로 향함 → 방향 캡션을 구분해야 하는데 표현은 거의 같다는 것을 표시.
- 텍스트:
  - **SimCLR**: Same color and texture / Different shape / Cosine similarity: 0.935
  - **CLIP**: Same object / Different orientations / Cosine similarity: 0.933
  - **High similarity does not always mean the same semantics.**

**설명**
- **Feature suppression**: 입력에 있는 유용한 정보 중 일부만 representation에 남고 나머지는 사라지는 현상(다음 슬라이드 p.49 정의). 이 슬라이드는 두 예시로 이를 보임.
  - **SimCLR**: color/texture만으로 두 이미지가 구분 안 될 만큼 cosine similarity 0.935 → shape 정보가 억제됨. SimCLR의 augmentation(color jitter, crop 등)과 InfoNCE는 "쉬운 shortcut feature(color, texture)"만으로도 instance discrimination이 가능해 shape를 굳이 안 배움.
  - **CLIP**: 같은 object(rabbit)지만 방향(left/right)이 다른 이미지가 0.933으로 거의 같은 표현 → CLIP은 object 종류는 잘 잡지만 **orientation, 위치 같은 세부 속성**을 놓침 (caption이 보통 object 이름 위주라 loss가 그 정보를 요구하지 않음).
- 결론: **cosine similarity가 높다고 모든 downstream에 필요한 semantics가 같은 것은 아님.** downstream task가 shape나 orientation을 필요로 하면 성능 저하.
- 직관: contrastive loss는 pair를 구분하는 데 **충분한 feature만** 배우면 loss가 줄어들기 때문에, 더 쉬운 feature가 경쟁에서 이기고 나머지는 suppress됨 (shortcut/competition 관점).

> 보충: 이 현상은 "feature competition", "simplicity bias"와 관련해 논의되며, p.47의 논문이 이를 mitigate하는 방법을 제안함. 슬라이드의 마지막 문장은 다음 p.49에서 정의로 이어짐.

## FIGURES

- p.41: GenView 전체 파이프라인(Quality-Driven Contrastive Loss: frozen encoder → fg/bg mask → $q_i$ → $w_i$ → 가중 loss) — 중요도 상
- p.42: 같은 다이어그램에서 품질 평가 부분 강조 + $q_i = s^f - s^b$ — 중요도 상
- p.43: ImageNet/CIFAR-10 원본 vs 생성 view 그리드 (배경/포즈/시점 변화) — 중요도 중
- p.44: Linear evaluation 표(프레임워크별 GenView 효과) — 중요도 상
- p.48: Feature suppression 예시(SimCLR 오각형/원 0.935, CLIP 토끼 좌/우 0.933) — 중요도 상

## EXAM POINTS

- **GenView의 pair quality 점수** $q_i = s_i^f - s_i^b$ 와 가중치 $w_i = \mathrm{softmax}(q_i)$, 최종 loss $\tilde{\mathcal{L}}_{\mathrm{SSL},*} = w_i \cdot \mathcal{L}_{\mathrm{SSL},*}$ 를 쓰고 각 기호 의미 설명하기.
- **Why**: 좋은 positive pair의 조건이 왜 "similar foreground, different background"인가? (nuisance에 invariant, background shortcut 방지, semantic drift 감지)
- 품질 평가를 하는 encoder가 왜 **frozen**이고 target encoder $f$ 와 분리되어 있는지.
- GenView가 **loss-agnostic**한 이유와, MoCov2/SwAV/SimSiam/BYOL/MoCov3에 적용 시 향상된다는 실험 결과 요약(예: MoCov3 R50 100ep 68.9 → 72.7).
- GenView의 **한계 4가지**: 외부 prior(CLIP, Stable unCLIP) 의존, 추가 연산/저장, foreground ratio 기반 noise heuristic, semantic drift 완전 제거 불가.
- **Feature suppression 정의**와 SimCLR(color/texture만 학습, shape 억제)·CLIP(object는 알지만 orientation 못 구분) 예시. 
- "High similarity does not always mean the same semantics"의 의미 (cosine similarity와 downstream 필요 정보의 불일치).
- 비교: 기존 augmentation 기반 view vs. 생성 모델 기반 view (큰 변형 + semantics 보존 vs. 픽셀 수준 변형).

## VERIFIER REPORT

- p.44 설명: GenView 향상 폭 범위 "+1.2 ~ +3.8%p" → "+1.1 ~ +3.8%p" 로 수정 (MoCov3 ViT-B 76.7 → 77.8 = +1.1).
- p.48 그림 설명(CLIP): 캡션 라벨 위치와 화살표 대응이 틀려 수정 (초록 화살표 = 오른쪽 이미지 → "facing right", 빨간 화살표 = 오른쪽 이미지 → "facing left").
- 표 수치(p.44), 수식, 인용(ECCV 2024, 저자/소속)은 원본과 일치 확인.
