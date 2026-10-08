<!-- DRAFT | lecture: 04_Contrastive_Learning_II | chunk: 06 | pages 41–48 -->
<!-- VERIFIED | fixes: 5 | verifier: sonnet -->

## DINO: Self-Distillation with No Labels — BYOL vs DINO 비교 (p.41)

- DINO (2021)
- 표 (BYOL vs DINO)

| BYOL | DINO |
|---|---|
| Feature matching | Distribution matching |
| MSE Loss | Cross Entropy Loss |
| Predictor | Softmax probability |
| EMA Teacher | EMA Teacher |
| No labels | No labels |

- 요약 문장: Unlike BYOL, which matches feature vectors, DINO performs self-distillation by matching the probability distributions produced by a momentum teacher network.
- 출처: Caron et al., ICCV 2021, *Emerging Properties in Self-Supervised Vision Transformers*

**설명**
- **DINO** (self-**di**stillation with **no** labels)는 student 네트워크가 teacher 네트워크의 출력을 따라가도록 학습하는 self-distillation 방식의 SSL임. teacher는 student 가중치의 **EMA(momentum)** 로 갱신됨 (gradient로 학습하지 않음).
- BYOL과의 공통점: (1) EMA teacher 사용, (2) negative sample 없음, (3) label 없음. 둘 다 **collapse를 negative 없이 피해야 하는** 계열임.
- 차이점:
  - **BYOL**: online 네트워크의 **predictor** 출력과 target(EMA) 네트워크의 projection 벡터를 **L2 정규화 후 MSE**로 맞춤 → **feature(벡터) matching**. collapse 방지는 predictor의 비대칭 + EMA stop-gradient에 의존.
  - **DINO**: 마지막에 projection head 출력을 **softmax**로 K차원 확률분포로 바꾸고, teacher 분포 $P_t$와 student 분포 $P_s$ 사이의 **cross-entropy**를 최소화 → **distribution matching**. 슬라이드 표는 BYOL의 Predictor에 DINO의 Softmax probability를 대응시킬 뿐이며, 실제 DINO에는 predictor가 없고 collapse 방지는 centering + sharpening이 담당함 (아래 보충 참고).
- 슬라이드에는 수식이 없지만 DINO 손실은 아래와 같음.

> **보충 (DINO 논문, Caron et al. 2021)**:
> $$\min_{\theta_s} \sum_{x\in\{x_1^g,x_2^g\}}\ \sum_{\substack{x'\in V\\ x'\neq x}} H\big(P_t(x),\,P_s(x')\big),\quad H(a,b)=-a\log b$$
> $P_s(x)^{(i)}=\frac{\exp(g_{\theta_s}(x)^{(i)}/\tau_s)}{\sum_k \exp(g_{\theta_s}(x)^{(k)}/\tau_s)}$ ($P_t$는 $\tau_t$ 사용). $V$ = 2개 global crop + 여러 local crop. teacher는 global crop만 입력받고 student는 모든 crop을 입력받음 (**local-to-global correspondence**).
> **Collapse 방지**: teacher 출력에 **centering** ($g_t(x)\leftarrow g_t(x)-c$, $c$는 배치 평균의 EMA)을 적용 → 한 차원이 지배하는 collapse 억제, 그리고 teacher에 **sharpening**(낮은 $\tau_t$) 적용 → uniform collapse 억제. 두 효과가 서로 균형을 맞춤.
> teacher 갱신: $\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$.
> 참고: BYOL(Grill et al., 2020)은 predictor + MSE, DINO는 predictor 없이 centering/sharpening + CE.

---

## DINO: Self-Attention 시각화 (Figure 1) (p.42)

- DINO (2021)
- 그림: 2행 × 4쌍(총 8쌍)의 [원본 이미지 | attention map] 배열. 원본은 새, 칫솔, 국회의사당(Palace of Westminster) 같은 건물, 기린 두 마리, 요트(보트), 문 앞의 자전거, 의자 위의 개, 활주로의 비행기. 각 원본 오른쪽에 보라색 배경 위 노랑–초록 하이라이트의 heatmap이 붙어 있음 (밝을수록 attention 높음).
- 캡션: "Figure 1: **Self-attention from a Vision Transformer with 8 × 8 patches trained with no supervision.** We look at the self-attention of the [CLS] token on the heads of the last layer. This token is not attached to any label nor supervision. These maps show that the model automatically learns class-specific features leading to unsupervised object segmentations."
- 출처: Caron et al., ICCV 2021

**설명**
- ViT의 마지막 layer에서 **[CLS] token이 각 patch token에 주는 self-attention**을 시각화한 것임. heatmap이 배경(나뭇가지, 풍경)이 아니라 **전경 객체의 형태(새 윤곽, 기린 몸통, 배의 돛, 자전거, 비행기 날개)** 를 따라 켜짐.
- 이 모델은 **label 없이** DINO로만 학습했는데도 객체 경계를 attention이 잡아냄 → **emergent property**: 별도 segmentation supervision 없이 **unsupervised object segmentation / object discovery**가 가능해짐.
- 의미: 학습된 표현이 단순 instance 구분이 아니라 **semantic layout**을 담고 있음을 정성적으로 보여주는 증거임. supervised ViT에서는 이런 깨끗한 attention map이 잘 나오지 않는다는 점이 논문의 주요 주장임.
- "8 × 8 patches"는 patch size 8인 ViT (ViT-S/8 등)를 의미함 → 작은 patch일수록 해상도 높은 attention map 가능.

> **보충**: 논문은 [CLS] attention map을 threshold해 얻은 mask의 segmentation 품질을 정량 평가(PASCAL VOC12 Jaccard 등)하고, 별도로 DINO feature를 nearest-neighbor propagation에 쓰는 DAVIS-2017 video object segmentation에서도 경쟁력 있는 성능을 보고함 (DAVIS는 attention mask가 아닌 feature 기반).

---

## DINO: ImageNet Linear / k-NN 성능 (Table 2) (p.43)

- DINO (2021)
- 표 캡션: "Table 2: **Linear and k-NN classification on ImageNet.** We report top-1 accuracy for linear and k-NN evaluations on the validation set of ImageNet for different self-supervised methods. We focus on ResNet-50 and ViT-small architectures, but also report the best results obtained across architectures. * are run by us. We run the k-NN evaluation for models with official released weights. The throughput (im/s) is calculated on a NVIDIA V100 GPU with 128 samples per forward. Parameters (M) are of the feature extractor."
- 왼쪽 표 (ResNet-50 비교; Param 23M, im/s 1237 동일):

| Method | Arch. | Param. | im/s | Linear | k-NN |
|---|---|---|---|---|---|
| Supervised | RN50 | 23 | 1237 | 79.3 | 79.3 |
| SCLR [11] | RN50 | 23 | 1237 | 69.1 | 60.7 |
| MoCov2 [13] | RN50 | 23 | 1237 | 71.1 | 61.9 |
| InfoMin [54] | RN50 | 23 | 1237 | 73.0 | 65.3 |
| BarlowT [66] | RN50 | 23 | 1237 | 73.2 | 66.0 |
| OBoW [21] | RN50 | 23 | 1237 | 73.8 | 61.9 |
| BYOL [23] | RN50 | 23 | 1237 | 74.4 | 64.8 |
| DCv2 [9] | RN50 | 23 | 1237 | 75.2 | 67.1 |
| SwAV [9] | RN50 | 23 | 1237 | **75.3** | 65.7 |
| **DINO** | RN50 | 23 | 1237 | **75.3** | **67.5** |

- 오른쪽 표 상단 (ViT-S, Param 21, im/s 1007):

| Method | Arch. | Param. | im/s | Linear | k-NN |
|---|---|---|---|---|---|
| Supervised | ViT-S | 21 | 1007 | 79.8 | 79.8 |
| BYOL* [23] | ViT-S | 21 | 1007 | 71.4 | 66.6 |
| MoCov2* [13] | ViT-S | 21 | 1007 | 72.7 | 64.4 |
| SwAV* [9] | ViT-S | 21 | 1007 | 73.5 | 66.3 |
| **DINO** | ViT-S | 21 | 1007 | **77.0** | **74.5** |

- 오른쪽 표 하단 "Comparison across architectures":

| Method | Arch. | Param. | im/s | Linear | k-NN |
|---|---|---|---|---|---|
| SCLR [11] | RN50w4 | 375 | 117 | 76.8 | 69.3 |
| SwAV [9] | RN50w2 | 93 | 384 | 77.3 | 67.3 |
| BYOL [23] | RN50w2 | 93 | 384 | 77.4 | – |
| **DINO** | ViT-B/16 | 85 | 312 | 78.2 | 76.1 |
| SwAV [9] | RN50w5 | 586 | 76 | 78.5 | 67.1 |
| BYOL [23] | RN50w4 | 375 | 117 | 78.6 | – |
| BYOL [23] | RN200w2 | 250 | 123 | 79.6 | 73.9 |
| **DINO** | ViT-S/8 | 21 | 180 | 79.7 | **78.3** |
| SCLRv2 [12] | RN152w3+SK | 794 | 46 | 79.8 | 73.1 |
| **DINO** | ViT-B/8 | 85 | 63 | **80.1** | 77.4 |

(DINO 행은 슬라이드에서 분홍색 음영으로 강조됨. 표 내 "SCLR"은 SimCLR의 표기임 ← 논문 표 원문의 약어이므로 원문 유지.)
- 출처: Caron et al., ICCV 2021

**설명**
- **평가 프로토콜 두 가지**: (1) **Linear probing** = backbone을 freeze하고 그 위에 linear classifier만 학습 후 top-1 정확도 측정. (2) **k-NN** = 추가 학습 없이 feature 공간에서 k개 최근접 이웃 투표로 분류 → 표현 공간 자체의 **semantic clustering 품질**을 직접 반영.
- **ResNet-50 비교(왼쪽)**: DINO는 linear 75.3으로 SwAV와 동률 최고, **k-NN은 67.5로 단독 1위**. BYOL(74.4 / 64.8), SimCLR(69.1 / 60.7)보다 높음. 즉 같은 backbone이라도 distribution matching 방식이 linear, 특히 k-NN에서 이점을 보임.
- **ViT-S 비교(오른쪽 위)**: BYOL/MoCov2/SwAV를 ViT-S에 적용(직접 실행, *)하면 71~73%대인데 DINO는 **77.0 / 74.5**. 특히 **k-NN 격차가 큼(74.5 vs 최고 66.6, 약 +8)** → DINO가 ViT와 특히 궁합이 좋음을 시사.
- **아키텍처 간 비교(오른쪽 아래)**: ViT-B/8 DINO가 linear 80.1로 최고, ViT-S/8은 파라미터 21M만으로 linear 79.7, k-NN 78.3 (표 내 k-NN 최고). 훨씬 큰 ResNet(RN152w3+SK 794M)보다 파라미터 효율이 좋음.
- **Supervised와의 격차**: ResNet-50에서 supervised 79.3 vs DINO 75.3(linear)로 아직 차이가 있지만, ViT-S/8·ViT-B/8에서는 supervised 수준에 근접/상회(ViT-S 기준 supervised 79.8과 비교).
- ← 약어 표기는 원 논문 그대로임 (SCLR = SimCLR, DCv2 = DeepCluster-v2, SCLRv2 = SimCLRv2).

> **보충**: patch size가 작을수록(/16 → /8) 토큰 수가 늘어 정확도는 오르지만 throughput(im/s)은 크게 떨어짐 (312 → 63 for ViT-B). 표의 im/s 열이 이 trade-off를 보여줌.

---

## DINO: Why is DINO important? (p.44)

- DINO (2021)
  - Why is DINO important?
    - Produces **semantic representations** without labels
    - Learns object-centric attention in Vision Transformers
    - Emergent segmentation and object discovery
    - Foundation of many modern SSL methods
- 출처: Caron et al., ICCV 2021

**설명**
- **semantic representation**: label 없이도 k-NN만으로 높은 ImageNet 정확도가 나옴(p.43) → feature 공간이 의미 단위로 군집화됨.
- **object-centric attention**: [CLS] attention이 객체를 따라가는 현상(p.42). ViT 구조 + self-distillation 조합에서 나타나는 emergent property임. contrastive(SimCLR 류)나 supervised ViT에서는 이만큼 뚜렷하지 않다는 것이 논문의 메시지.
- **Foundation of many modern SSL methods**: DINO의 EMA teacher + centering/sharpening 구조는 이후 iBOT, **DINOv2**, 그리고 다양한 self-distillation 기반 foundation model의 기반이 됨.

> **보충**: DINOv2 (Oquab et al., 2023)는 대규모 curated 데이터(LVD-142M)와 patch-level 손실(iBOT)을 더해 범용 visual feature를 만든 후속작임.

---

## (섹션 구분) Can positive views be generated rather than augmented? (p.45)

- 섹션 구분 슬라이드: "Can positive views be generated rather than augmented?" — positive view를 augmentation이 아닌 **생성(generation)** 으로 만들 수 있는가라는 질문 제기.

---

## Learning Vision from Models Rivals Learning Vision from Data (p.46)

- 논문 표지 캡처 (CVF Open Access 워터마크 포함: "This CVPR paper is the Open Access version, provided by the Computer Vision Foundation. Except for this watermark, it is identical to the accepted version; the final published version of the proceedings is available on IEEE Xplore.")
- 제목: **Learning Vision from Models Rivals Learning Vision from Data**
- 저자: Yonglong Tian$^{1,\dagger}$, Lijie Fan$^{2,\dagger,*}$, Kaifeng Chen$^1$, Dina Katabi$^2$, Dilip Krishnan$^1$, Phillip Isola$^2$
- 소속: $^1$Google Research, $^2$MIT CSAIL, $^\dagger$equal contribution
- Github Repo: https://github.com/google-research/syn-rep-learn
- 출처: Tian et al., CVPR 2024

**설명**
- 이 논문이 제안한 방법이 **SynCLR**임. 핵심 아이디어: **실제 이미지 데이터 없이, 생성 모델(LLM + text-to-image diffusion)이 만든 synthetic data만으로** visual representation을 contrastive learning으로 학습함.
- 제목의 의미: "모델(generative model)로부터 배운 표현"이 "실제 데이터로부터 배운 표현"에 필적(rival)함.

> **보충**: SynCLR 파이프라인 (논문 기준) — (1) LLM(Llama-2)으로 synthetic caption 대량 생성, (2) 각 caption으로 text-to-image 모델(Stable Diffusion)로 여러 장의 이미지 생성, (3) 같은 caption에서 나온 이미지들을 positive로 하는 **multi-positive contrastive loss** + masked image modeling 류 보조 손실로 학습. 이 pipeline의 caption 생성 등 세부 내용은 이후 슬라이드(chunk 범위 밖)에서 이어질 것으로 보임.

---

## From Image Augmentation to Image Generation (p.47)

- Can we generate positive views instead of augmenting an image?
- 그림: 좌우 두 패널 비교.
  - **SimCLR (왼쪽)**: 눈 덮인 숲의 여우 사진 1장 → 화살표 $t_1$, $t_2$ (서로 다른 augmentation) → 두 개의 변형 이미지(하나는 확대·주황 색감의 얼굴 crop, 다른 하나는 푸른 색조 crop) → 각각 encoder를 거쳐 파란 원(embedding) 2개 → 대괄호로 묶어 **Positive pair**. 하단 문구: "Same image, different augmentations".
  - **SynCLR (오른쪽)**: 텍스트 박스 "A red fox in a snowy forest" → **Image Generator** 박스 → 서로 다른 구도/포즈/배경의 여우 이미지 4장 (2×2) → 각각 embedding 파란 원 4개(농도 다름) → 대괄호로 묶어 **Positive set**. 하단 문구: "Same caption, different generated images".
- 하단 bullet:
  - SimCLR preserves image identity.
  - SynCLR preserves caption-level semantics.
  - SimCLR creates positive pairs through image augmentation, whereas SynCLR generates a positive set from shared caption semantics.

**설명**
- **SimCLR**: 하나의 이미지 $x$에 두 augmentation $t_1, t_2$를 적용해 $(t_1(x), t_2(x))$를 **positive pair**로 삼음. 따라서 invariance의 정의는 "**같은 원본 이미지(instance identity)**"이며, 변하는 것은 crop/color jitter 같은 low-level 변환뿐임. positive가 항상 2개(pair).
- **SynCLR**: 하나의 **caption**에서 text-to-image generator로 여러 이미지를 생성하고 그 전부를 positive로 취급 → **positive set**. 같은 caption이지만 **포즈·구도·배경·질감이 실제로 다른 이미지**들이므로, augmentation으로는 만들 수 없는 high-level 변화(viewpoint, 배치, 배경)에 대한 invariance를 학습함.
- 설계 이유: augmentation은 픽셀 수준 변환이라 객체 자체의 변이(자세, 장면)를 주지 못함. 생성 모델은 caption이 정한 **semantic 수준에서 동일하되 시각적으로 다양한** positive를 공급 → 더 의미 중심의 표현 학습.
- 대비 요약: 단위 = image identity vs caption semantics, positive 개수 = pair vs set, positive 생성 수단 = augmentation vs generator.

> **보충**: SynCLR는 positive가 여러 개이므로 SupCon(Khosla et al., 2020)과 유사한 **multi-positive InfoNCE** 형태의 손실을 사용함 (caption = class label 역할).

---

## What Defines a Positive Pair? (p.48)

- 그림: 3개 행 (점선으로 구분), 각 행에 골든 리트리버 이미지 4장 — (a) 자전거 타는 개, (b) 선글라스·모자 쓰고 자전거 타는 개, (c) 스시가 놓인 검은 지붕 장식 안의 개, (d) 스시 있는 오두막(doghouse) 안의 개. (a)(b)는 자전거 장면, (c)(d)는 스시/집 장면.
  - **SimCLR 행**: 4장 각각이 자기만의 둥근 테두리로 묶여 **class 1, class 2, class 3, class 4** (이미지마다 별개 클래스).
  - **Supervised CE 행**: 4장 전부가 하나의 큰 테두리 안에 **class 1**.
  - **SynCLR 행**: (a)(b)가 **class 1**, (c)(d)가 **class 2**로 두 묶음.
- 오른쪽 텍스트:
  - **SimCLR**
    - Each individual image becomes a separate class.
    - Semantically similar generated images are pushed apart.
  - **Supervised Learning**
    - All golden retrievers belong to one class.
    - Actions and backgrounds are ignored.
  - **SynCLR**
    - Images generated from the same caption form a class.
    - Captures objects, actions, and relationships.
- 하단 문구: **A caption defines the granularity of a visual class.**

**설명**
- "무엇을 positive로 볼 것인가"는 곧 **클래스(그룹)의 입도(granularity)를 정의**하는 문제임.
  - **SimCLR (instance discrimination)**: 이미지 1장 = 1 class. 의미상 매우 비슷한 이미지(자전거 타는 개 두 장)도 서로 **negative**가 되어 밀려남(pushed apart) → **false negative** 문제. 입도가 가장 세밀함.
  - **Supervised CE**: label이 "golden retriever" 하나뿐이면 4장 전부 같은 class. 자전거 타기·스시 장면 같은 **행동/배경/관계 정보가 무시**됨 → 입도가 너무 거침.
  - **SynCLR**: caption(예: "a golden retriever riding a bike" vs "a golden retriever in a sushi doghouse")이 같은 이미지끼리 한 class → **객체 + 행동 + 관계**까지 반영하는 **중간 입도**. 자전거 장면 2장은 가깝게, 스시/집 장면 2장은 별도 class로 분리.
- 결론: caption이 class의 granularity를 조절하는 손잡이이며, 텍스트의 풍부함으로 인해 supervised label보다 세밀하고 instance discrimination보다 의미적으로 일관된 positive 구조를 만들 수 있음.
- ← 슬라이드 그림에서 SimCLR 행의 "class 1~4"는 실제 SimCLR가 class를 쓰는 것이 아니라 **instance 단위 negative 구조를 설명하기 위한 비유**임 (원문 유지).

---

## FIGURES

- p.42: DINO [CLS] self-attention map 8쌍 (원본 vs heatmap, unsupervised object segmentation emergent) — 중요도 상
- p.43: ImageNet linear / k-NN 비교 표 (ResNet-50 및 ViT 아키텍처 간 비교) — 중요도 중 (핵심 수치만 발췌 가능)
- p.41: BYOL vs DINO 비교 표 — 중요도 상
- p.47: SimCLR(augmentation, positive pair) vs SynCLR(generator, positive set) 비교 다이어그램 — 중요도 상
- p.48: SimCLR / Supervised CE / SynCLR의 class 입도 비교 그림 (골든 리트리버 예시) — 중요도 상

## EXAM POINTS

1. **BYOL vs DINO 차이**: feature matching(MSE, predictor) vs distribution matching(cross-entropy, softmax). 공통점은 EMA teacher, no labels, negative 없음.
2. **DINO 손실 구조**: teacher/student가 각각 softmax 분포 출력(온도 $\tau_t, \tau_s$), $H(P_t, P_s)=-P_t\log P_s$ 최소화, teacher는 stop-gradient + EMA 갱신, multi-crop(global → teacher, global+local → student). (보충 지식 포함)
3. **Collapse 방지 방법 (DINO)**: centering + sharpening의 역할 구분 (centering → 한 차원 지배 방지, sharpening → uniform 방지). (보충)
4. **DINO의 emergent property**: ViT [CLS] self-attention이 label 없이도 객체 경계를 분할 → unsupervised segmentation / object discovery. 왜 중요한가 서술 가능해야 함.
5. **평가 지표**: linear probing vs k-NN 정의 및 차이, DINO가 ResNet-50에서 75.3 / 67.5, ViT-S에서 77.0 / 74.5로 BYOL·SwAV·MoCov2 대비 우위 (특히 k-NN).
6. **SimCLR vs SynCLR positive 정의**: 같은 이미지의 두 augmentation(positive pair, image identity) vs 같은 caption에서 생성한 여러 이미지(positive set, caption-level semantics).
7. **Why 질문**: 왜 generated positive가 augmentation보다 유리한가? (pose·구도·배경 같은 high-level 변이에 대한 invariance, false negative 완화)
8. **Class granularity 비교**: SimCLR(이미지당 1 class, 너무 세밀, 비슷한 이미지도 push apart) vs Supervised CE(너무 거침, 행동/배경 무시) vs SynCLR(caption 기반 중간 입도, 객체·행동·관계 반영).

## VERIFIER REPORT

- p.42 보충: DAVIS-2017은 attention mask threshold가 아니라 feature nearest-neighbor propagation 기반이므로 서술 수정 (attention mask는 PASCAL VOC12 등으로 평가).
- p.43 설명: ViT-S k-NN 격차 "+8 이상"을 실제 수치(74.5 vs 66.6, 약 +8)로 정정.
- p.46 보충: "다음 슬라이드(p.49)" 오기 수정 (다음 슬라이드는 p.47, caption 생성은 chunk 범위 밖).
- p.41 설명: "softmax가 predictor 역할을 한다"는 과장된 해석을 슬라이드 표의 대응 관계 + 실제 collapse 방지(centering/sharpening)로 정정.
- p.43 설명: 의미 불명확한 약어 주석을 정리 (DCv2 = DeepCluster-v2, SCLRv2 = SimCLRv2).
- 그 외 표 수치(p.43 전체), 그림 설명(p.42/47/48), 저자·연도·학회 인용은 원본과 일치 확인.
