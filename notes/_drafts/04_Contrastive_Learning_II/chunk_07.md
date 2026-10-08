<!-- DRAFT | lecture: 04_Contrastive_Learning_II | chunk: 07 | pages 49–56 -->
<!-- VERIFIED | fixes: 3 | verifier: sonnet -->

## SynCLR: Data Generation Pipeline — Iterative Caption Generation (p.49)

- Iterative Caption Generation (Generating Synthetic Captions with Llama-2)
  - Sample three examples as the initial context.
  - Llama-2 generates a new caption from the context.
  - Add the generated caption to the context.
  - Repeat the process to generate diverse captions.

**그림**: 왼쪽에 둥근 회색 박스 "Context"가 있고, 안에 in-context prompt가 들어 있음. 내용은 "Generate an image caption with an object category and an environment category:" 뒤에 예시 3개와 마지막 빈 쿼리 줄이 이어지는 형태.
- `tiger, forest => Two tigers are running together in the forest.`
- `groom, wedding ceremony => father of groom congratulating him after the wedding ceremony.`
- `fountain, park => Water cascades from an ornate fountain, surrounded by autumn-hued trees in a serene park.`
- `red fox, yard =>` (빈칸, 여기를 채워야 함)

Context 박스에서 파란 화살표("Llama 2")가 오른쪽의 "Generated Caption" 박스로 나감. 박스 내용은 "wild red fox sitting on a partially snow covered front yard of a house in the suburbs of a small city". 이 박스에서 초록색 곡선 화살표가 Context 박스의 마지막 줄(`red fox, yard =>`) 쪽으로 되돌아옴. 범례는 **Blue: Generate a new caption**, **Green: Add it back to the context**.

**설명**
- SynCLR는 **real image 없이** 학습 데이터를 만드는 방법이라서, 먼저 "무엇을 그릴지"를 정하는 **caption 생성** 단계가 필요함. 이 슬라이드는 그 첫 단계임.
- 방식은 **in-context learning** 기반 caption 합성: LLM(Llama-2)에게 `(object category, environment category) => caption` 형식의 few-shot 예시 3개를 주고, 새로운 (category 쌍)에 대한 caption을 이어서 쓰게 함. 예시 속 `red fox, yard`가 새 쿼리고, 생성 결과가 "wild red fox sitting on ..."임.
- 생성된 caption을 다시 context에 넣는 **iterative** 구조(초록 화살표)가 핵심임. context 예시가 계속 바뀌므로 매번 prompt가 달라지고, 그래서 같은 형식의 caption만 반복되는 것을 막고 **diversity**를 확보함. 예시를 3개 샘플링해서 시작하는 것도 같은 이유(예시 조합이 달라지면 출력 스타일도 달라짐).
- 왜 이렇게 설계했나: contrastive learning에서 "클래스" 역할을 하는 것이 caption이므로(p.48), caption의 **다양성과 coverage**가 곧 학습되는 visual concept의 다양성이 됨. 사람이 라벨을 달 필요 없이 caption 자체를 scalable하게 만들 수 있음.
- 이전 방법과의 차이: SimCLR는 이미지 하나가 곧 클래스이고 augmentation으로 positive를 만들었음. SynCLR는 **text(caption)가 먼저 존재**하고 그 caption에서 이미지를 만듦.

> 보충: 이 pipeline은 Tian et al., 2024, "Learning Vision from Models Rivals Learning Vision from Data" (SynCLR)의 방식임. 원 논문에서는 ImageNet-21K, Places-365 등의 concept 목록에서 object/environment 쌍을 만들어 LLM에 넣었고, 이렇게 얻은 약 **150M개** caption을 사용함(p.50의 숫자와 일치). Llama-2는 Meta의 open LLM(Touvron et al., 2023).

## SynCLR: Data Generation Pipeline — From a Caption to Positive Views (p.50)

- From a Caption to Positive Views
  - Use one synthetic caption as the condition.
  - Start from different random noise.
  - Generate four images for each caption.
  - Use a low CFG scale to increase diversity.
- Same caption, shared semantics
  - Different appearance and composition
  - Same objects and relationships
  - Treated as a positive set
- 하단 문구: Different noise produces visual diversity, while the caption preserves common semantics.
- 하단 문구: 150M captions X 4 = 600 M synthetic images

**그림**: 오른쪽에 세 줄의 이미지 행이 있고, 각 행은 같은 caption으로 생성된 이미지 4장과 그 아래 caption으로 구성됨.
- 1행: "A plate of paella, a mixed rice dish with chicken, beans, and seafood" — 서로 다른 구도/접시/조명의 paella 4장.
- 2행: "An industrial power plant with its smokestacks belching black smoke." — 굴뚝에서 연기가 나오는 발전소, 각도와 날씨가 다른 4장.
- 3행: "A fluffy, black and white junco bird perches on a snow-covered fence, overlooking a dark forest." — 눈 덮인 환경의 junco 새, 배경과 자세가 다른 이미지들.

같은 행 안에서 이미지는 모두 달라 보이지만 공통 의미(객체, 관계)는 유지됨.

**설명**
- p.49에서 만든 caption 하나를 **text-to-image diffusion model의 condition**으로 넣고, **서로 다른 random noise**에서 출발해 이미지를 4장 생성함. 같은 condition + 다른 noise이므로 **semantic은 공유하되 appearance/composition은 다름**. 이 4장이 하나의 **positive set**이 됨.
- **Low CFG (classifier-free guidance) scale**을 쓰는 이유: guidance가 크면 caption에 강하게 맞추느라 샘플들이 서로 비슷한 mode로 몰려서 diversity가 줄어듦. 낮추면 조건 충실도는 조금 떨어지지만 이미지 간 변이가 커져서, positive 간 "다양한 view" 효과(= augmentation 역할)가 커짐.
- 핵심 직관: **hand-designed augmentation(crop, color jitter)** 대신 **생성 모델이 만드는 semantic variation**을 positive로 사용함. crop 기반 positive는 같은 사진의 일부이므로 낮은 수준의 변형인 반면, 여기서는 구도, 자세, 배경까지 달라지는 high-level 변형이 가능함.
- 규모: 150M captions × 4 = **600M synthetic images**. (계산 일치)

> 보충: 원 논문(SynCLR)은 Stable Diffusion 1.5를 사용했고, 기본값보다 낮은 CFG scale이 representation 품질에 유리하다고 보고한 것으로 알고 있음. 슬라이드에는 구체 수치가 없으므로 정확한 값은 원 논문 확인 필요.

## Multi-Positive Contrastive Learning (p.51)

- 왼쪽: 박스 "Caption c — A tiger in the forest".
- 가운데: 같은 caption에서 생성된 이미지 4장 $x_1, x_2, x_3, x_4$ (숲속의 호랑이; 정면 걷기, 바위에 엎드림, 뒷모습 걷기, 풀숲에서 얼굴 내밀기).
- 오른쪽: "2D feature space" 박스.
  - $z_1$ = **Anchor** (진한 파란 큰 원), $z_2, z_3, z_4$ = **Positive** (밝은 파란 원).
  - 오른쪽에 점선 타원 2개 안에 회색 점들 = "Negatives from other captions".
- 수식:

$$\mathbf{q}_i = \frac{\exp(\boldsymbol{a}\cdot\boldsymbol{b}_i/\tau)}{\sum_{j=1}^{K}\exp(\boldsymbol{a}\cdot\boldsymbol{b}_j/\tau)}$$

$$\mathbf{p}_i = \frac{\mathbb{1}_{\text{match}(\boldsymbol{a},\boldsymbol{b}_i)}}{\sum_{j=1}^{K}\mathbb{1}_{\text{match}(\boldsymbol{a},\boldsymbol{b}_j)}}$$

$$\mathcal{L}(\boldsymbol{a}) = H(\mathbf{p},\mathbf{q}) = -\sum_{i=1}^{K}\mathbf{p}_i\log\mathbf{q}_i$$

- For each anchor:
  - Same-caption images are positives.
  - Different-caption images are negatives.
  - Multiple positives share caption-level semantics.

**설명**
- 기호: $\boldsymbol{a}$ = anchor 이미지의 embedding(정규화된 feature), $\boldsymbol{b}_i$ = batch 안의 $i$번째 후보 embedding($K$개), $\tau$ = **temperature**, $\mathbf{q}_i$ = anchor가 $b_i$를 "정답"으로 고를 모델의 softmax 확률, $\mathbb{1}_{\text{match}(a,b_i)}$ = $a$와 $b_i$가 **같은 caption에서 나왔으면 1, 아니면 0**인 indicator, $\mathbf{p}_i$ = 그 indicator를 정규화한 **target 분포**.
- $\mathbf{p}$는 positive가 $m$개면 각 positive에 $1/m$, negative에는 0을 주는 uniform 분포임. 따라서 loss $H(\mathbf{p},\mathbf{q})$는 **cross-entropy**이고, 결과적으로 $-\frac{1}{m}\sum_{i\in \text{pos}}\log q_i$ 형태임. 분모는 모든 후보(positive + negative)를 포함하므로 positive는 가까워지고 negative와는 멀어짐.
- **왜 이렇게 설계했나**: 기존 InfoNCE(SimCLR)는 anchor당 positive가 **정확히 1개**였음(같은 이미지의 다른 augmentation). SynCLR는 같은 caption에서 이미지가 4장 나오므로 positive가 여러 개이고, 이를 자연스럽게 일반화한 것이 **multi-positive** 형태임. $m=1$이면 $\mathbf{p}$가 one-hot이 되어 기존 InfoNCE와 정확히 일치함.
- 이전 방법과의 차이: **SupCon**(Khosla et al., 2020)도 multi-positive이지만 positive 정의가 **class label**임. 여기서는 label이 아니라 **caption(생성 조건)**이 positive를 정의함. 그림처럼 2D feature space에서 같은 caption의 $z_2,z_3,z_4$는 anchor $z_1$ 쪽으로 끌려오고, 다른 caption의 회색 점들은 밀려남.
- 중요한 점: positive들이 **서로 다른 이미지**이므로 positive pair가 "같은 원본 이미지"에서 나올 필요가 없음(p.54와 연결).

> 보충: 이 형태의 loss는 StableRep(Tian et al., NeurIPS 2023)에서 먼저 제안된 multi-positive contrastive loss와 같은 계열임. SynCLR 원 논문은 여기에 patch-level masking 목적을 함께 쓴다고 알고 있으나, 슬라이드에는 해당 내용이 없음.

## Does Learning from Models Work? — ImageNet linear evaluation (p.52)

- ImageNet linear evaluation

**표 (Table 6)**: 열 순서는 method, text, img, # imgs, backbone 뒤에 ImageNet, Aircraft, Cars, DTD, Flowers, Pets, SUN397, Caltech-101, Food-101, VOC2007, Average.

| method | text | img | # imgs | backbone | ImageNet | Aircraft | Cars | DTD | Flowers | Pets | SUN397 | Caltech-101 | Food-101 | VOC2007 | Average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| StableRep | real | syn | 100M | ViT-B/16 | 75.7 | 59.2 | 83.5 | 80.1 | 97.3 | 88.3 | 74.3 | 94.7 | 85.1 | 87.9 | 83.4 |
| CLIP | real | real | 400M | ViT-B/16 | 80.2 | 59.5 | 86.7 | 79.2 | 98.1 | 93.1 | 78.4 | 94.7 | 92.8 | 89.2 | 85.7 |
| CLIP | real | real | 400M | ViT-L/14 | 83.9 | 69.4 | 90.9 | 82.1 | 99.2 | 95.1 | 81.8 | 96.5 | 95.2 | 89.6 | 88.9 |
| OpenCLIP | real | real | 400M | ViT-B/16 | 78.9 | 61.1 | 92.3 | 81.9 | 98.2 | 91.5 | 77.9 | 95.2 | 90.9 | 88.0 | 86.3 |
| OpenCLIP | real | real | 400M | ViT-L/14 | 82.3 | 67.1 | 94.0 | 83.6 | 98.8 | 92.5 | 81.0 | 96.4 | 93.4 | 88.8 | 88.4 |
| OpenCLIP | real | real | 2B | ViT-L/14 | 83.4 | 71.7 | 95.3 | 85.3 | 99.0 | 94.2 | 82.2 | 97.5 | 94.1 | 88.9 | 89.8 |
| DINO v2* | - | real | 142M | ViT-B/14 | **83.9**† | 79.4 | 88.2 | 83.3 | 99.6 | 96.2 | 77.3 | 96.1 | 92.8 | 88.2 | **89.0** |
| DINO v2* | - | real | 142M | ViT-L/14 | **85.7**† | 81.5 | 90.1 | 84.0 | 99.7 | 96.6 | 78.7 | 97.5 | 94.3 | 88.3 | 90.1 |
| SynCLR | syn | syn | 600M | ViT-B/16 | 80.7 | 81.7 | 93.8 | 79.9 | 99.1 | 93.6 | 76.2 | 95.3 | 91.6 | 89.4 | **89.0** |
| SynCLR | syn | syn | 600M | ViT-L/14 | 83.0 | 85.6 | 94.2 | 82.1 | 99.2 | 94.1 | 78.4 | 96.1 | 93.4 | 90.3 | **90.4** |

(슬라이드에서 OpenCLIP은 400M 두 줄 + 2B 한 줄로 표기되며 backbone은 ViT-B/16, ViT-L/14, ViT-L/14 순서임. 값은 이미지에서 직접 옮김.)

캡션: "Table 6. **Comparison on ImageNet linear evaluation and fine-grained classificaton.** ← 원문 오타(classification). SynCLR achieves comparable results with OpenAI's CLIP and DINO v2 models, despite *only* using synthetic data. *DINO v2 modes are distilled from a ViT-g model, thus advantageous in this comparison. † we rerun only using `cls` token instead of concatenating multiple layers presented in the original DINO v2 paper [59]." ← "modes"는 원문 표기(models 오타로 보임).

**설명**
- 열 의미: `text`/`img`는 **학습에 쓴 text와 image가 real인지 synthetic인지**를 뜻함. StableRep은 real caption + synthetic image, CLIP/OpenCLIP은 real text + real image, DINOv2는 text 없음 + real image, **SynCLR는 text도 image도 모두 synthetic**. 즉 SynCLR는 유일하게 **real data를 전혀 쓰지 않는** 설정임.
- 평가 방식은 **linear evaluation**: backbone을 freeze하고 위에 linear classifier만 학습함. 그래서 점수는 representation 자체의 질을 반영함. ImageNet 외에 Aircraft, Cars, DTD, Flowers, Pets, SUN397, Caltech-101, Food-101, VOC2007이 **fine-grained / 다양한 domain transfer** 평가임.
- 결과 해석:
  - ImageNet top-1에서는 SynCLR ViT-B/16(80.7)이 CLIP ViT-B/16(80.2)과 비슷하고 OpenCLIP ViT-B(78.9)보다 높음. ViT-L/14는 83.0으로 CLIP(83.9), DINOv2(85.7)보다는 약간 낮지만 근접함.
  - **Average(10개 데이터셋 평균)**: SynCLR ViT-B 89.0 = DINOv2 ViT-B 89.0, ViT-L 90.4로 이 표에서 가장 높음(DINOv2 ViT-L 90.1, OpenCLIP 2B 89.8, CLIP 88.9).
  - SynCLR는 Aircraft(81.7, 85.6)에서 특히 강함(다른 모든 행보다 높음; DINOv2 ViT-L 81.5, OpenCLIP 2B 71.7). Cars(93.8, 94.2)는 OpenCLIP(ViT-L/14 94.0, 2B 95.3)과 비슷한 수준이고 최고는 아님. 이들은 fine-grained이므로, caption이 세밀한 object/attribute 정보를 담는다는 p.48 주장과 연결할 수 있음(단 이 표만으로 인과를 증명하지는 못함).
- 주의: DINOv2(*)는 **ViT-g에서 distillation**되었으므로 비교에서 유리하다고 캡션이 밝힘. † 표시는 원 논문처럼 여러 layer를 concat하지 않고 `cls` token만 써서 재실행한 값임.
- 이 슬라이드의 결론: "Learning from models"(= 생성 모델이 만든 synthetic data로 학습)가 real data 학습과 **견줄 만한 수준**에 도달함.

> 보충: 슬라이드 제목은 원 논문 제목 "Learning Vision from Models Rivals Learning Vision from Data"(Tian et al., 2024)에서 온 질문임. StableRep은 Tian et al., NeurIPS 2023(Stable Diffusion 합성 이미지 + multi-positive contrastive)으로 SynCLR의 직계 선행 연구임. OpenCLIP은 LAION 등 공개 데이터로 CLIP을 재현한 구현(Ilharco et al.).

## Does Learning from Models Work? — Semantic segmentation (p.53)

- Semantic segmentation
- 하단 문구: Synthetic pretraining also produces strong dense features.

**표 (Table 7)**:

| method | pre-train data | distill | ViT-B | ViT-L |
|---|---|---|---|---|
| StableRep | hybrid, 100M | | 49.4 | - |
| MoCo v3 | real, IN1K-1M | | 47.3 | 49.1 |
| BEiT | real, IN1K-1M+DALLE | | 47.1 | 53.3 |
| MAE | real, IN1K-1M | | 48.1 | 53.6 |
| iBOT | real, IN1K-1M | | 50.0 | - |
| CLIP | real, WIT-400M | | 52.6 | - |
| BEiT v2 | real, WIT-400M, IN1K | ✓ | 53.1 | 56.7 |
| DINO v2 | real, LVD-142M | ✓ | **54.4**† | 57.5† |
| SynCLR | synthetic, 600M | | **54.3** | **57.7**† |

캡션: "Table 7. **ADE20K semantic segmentation** (mIoU) using UperNet, with single scale at 512x512 resolution. † use patch size of 14x14, thus adapt to 518x518 resolution."

**설명**
- 평가: **ADE20K** semantic segmentation, 지표는 **mIoU**, decoder는 **UperNet**, 단일 스케일 512×512 해상도임. image-level classification과 달리 **pixel/patch-level(dense) feature** 품질을 측정함.
- `distill` 열 체크(✓)는 큰 모델에서 distillation으로 얻은 backbone임을 뜻함(BEiT v2, DINO v2). SynCLR는 distill 없이 직접 학습했는데도 ViT-B 54.3으로 DINOv2(54.4)와 거의 같고, **ViT-L 57.7로 DINOv2(57.5)보다 높음**.
- MoCo v3, MAE, iBOT, CLIP 등 real data 기반 방법들(47–53%대)보다 높음. 따라서 "synthetic pretraining도 dense prediction에 쓸 만한 feature를 준다"가 결론임.
- 의미: global contrastive 학습만으로 dense task 성능이 나오는지가 의문일 수 있는데, SynCLR는 patch-level 목적을 일부 함께 쓰고(보충 참고) diffusion이 만든 풍부한 장면 구성이 도움이 된 것으로 해석 가능함.
- 표기: ViT-B의 SynCLR 54.3에는 † 표시가 없고 ViT-L 57.7에만 있음 ← 캡션의 †(patch 14, 518 해상도)가 행별로 어떻게 적용되는지 슬라이드만으로 불명확함.

> 보충: BEiT(Bao et al., 2022), BEiT v2(Peng et al., 2022), MAE(He et al., 2022), iBOT(Zhou et al., 2022), MoCo v3(Chen et al., 2021)는 모두 ViT 기반 self-supervised 방법. WIT-400M은 CLIP이 학습한 OpenAI의 WebImageText 4억 쌍, LVD-142M은 DINOv2의 curated 데이터셋임(p.57 참고).

## What changed? (p.54)

- Hand-designed augmentation → Generative semantic views
- Positive pairs do not need to originate from the same image.
- Captions provide controllable semantic classes.
- Generative models can serve as a scalable data source.
- Representation learning can be performed without real images.

**설명**
- 한 문장 요약: **positive를 만드는 주체가 사람이 설계한 augmentation에서 생성 모델로 바뀜**.
- 항목별 정리:
  - **Hand-designed augmentation → Generative semantic views**: SimCLR/MoCo류는 crop, color jitter 등 사람이 정한 변환이 "무엇이 invariant해야 하는지"를 정의함. SynCLR에서는 caption + 다른 noise로 생성된 이미지들이 view 역할을 하며, 변형이 pixel 수준이 아니라 **의미 수준**임.
  - **Positive pairs do not need to originate from the same image**: 기존 방식은 같은 이미지에서 나온 두 view만 positive였음. 여기서는 완전히 다른 이미지도 같은 caption이면 positive임.
  - **Captions provide controllable semantic classes**: caption을 바꾸면 **visual class의 granularity를 사람이 조절**할 수 있음(p.48의 "A caption defines the granularity of a visual class"와 연결).
  - **Generative models can serve as a scalable data source**: caption과 이미지를 모두 모델이 만들기 때문에 데이터 양을 늘리는 비용이 낮음.
  - **Representation learning can be performed without real images**: 학습 단계에서 real image 없이도 경쟁력 있는 representation이 가능함(p.52–53의 결과가 근거).

> 보충: 한계로, 생성 모델 자체가 real data로 학습되었으므로 "real data 없이"는 representation 학습 단계에만 해당하며, 생성 모델의 bias와 coverage를 그대로 물려받을 수 있음(슬라이드에는 없는 내용).

## (섹션 구분) How can DINO become a general-purpose visual foundation model? (p.55)

- 섹션 구분 슬라이드: "How can DINO become a general-purpose visual foundation model?" (파란 배경, 다음 주제인 DINOv2로의 전환 질문)

## DINOv2: A Visual Foundation Model (p.56)

- Learn general-purpose visual features from images alone.
  - No labels, No captions
  - Frozen features for many downstream tasks
  - Strong global and patch-level representations
- 하단: **PCA visualization**
- DINO demonstrated emergent representations.
- DINOv2 aims to make them general-purpose features.
- 출처: Oquab et al., TMLR 2024, DINOv2: Learning Robust Visual Features without Supervision

**그림**: 가로로 4개 블록이 있고, 각 블록은 "원본 이미지 4장(세로) + 오른쪽에 대응하는 PCA 색상 맵(검은 배경)" 쌍으로 구성됨.
- 블록 1: 독수리, 비행기, 독수리, 비행기. 날개/몸통/머리 등 같은 부위가 비슷한 색(초록, 보라, 분홍 등)으로 칠해짐. 새와 비행기 사이에서도 날개 부위의 색이 대응됨.
- 블록 2: 코끼리, 사람, 코끼리, 가네샤 조각상. 코끼리 부위별로 색이 구분되고 조각상에서도 대응되는 색이 나타남.
- 블록 3: 흰 말, 말 떼, 말 스케치, 초원의 동물들. 사진과 스케치처럼 domain이 달라도 말의 부위가 일관된 색으로 나옴.
- 블록 4: 자동차 모형, 버스, 그림체의 빨간 버스, 사막의 SUV. 차 종류와 스타일이 달라도 바퀴/차체 등이 대응되는 색으로 칠해짐.
- 배경은 검게 마스킹되어 전경(foreground) 객체의 patch feature만 시각화됨.

**설명**
- DINOv2의 목표: **label이나 caption 없이 이미지만으로** 여러 downstream task에 **frozen feature 그대로** 쓸 수 있는 **general-purpose visual feature**를 학습하는 것. CLIP이 text supervision으로 foundation model이 된 것과 대비되는 **순수 image-only SSL** 노선임.
- "Strong global and patch-level representations": 이미지 전체(`cls` token) 수준의 분류, retrieval뿐 아니라 **patch token**을 이용한 segmentation, depth estimation 등 dense task에도 쓸 수 있다는 뜻임(p.53 결과와 연결).
- **PCA visualization** 읽는 법: 이미지의 patch feature들에 PCA를 적용해 주성분을 RGB로 매핑한 것임(원 논문 방식으로는 첫 성분으로 전경을 분리하고 전경 patch에 대해 다시 PCA한 상위 3개 성분을 RGB로 사용; 슬라이드에는 설명 없음). **같은 색 = 비슷한 feature**. 서로 다른 이미지, 심지어 다른 category나 domain(사진 vs 스케치, 새 vs 비행기)에서도 **의미적으로 대응되는 부위가 같은 색**으로 나타남. 즉 feature가 **part-level semantic correspondence**를 포함하고 있음을 보여줌.
- "DINO demonstrated emergent representations. DINOv2 aims to make them general-purpose features.": DINO(v1)는 label 없이도 attention map에 object segmentation 같은 성질이 **emerge**한다는 점을 보였음(연구 결과 중심). DINOv2는 그 성질을 **데이터와 모델 규모 확장, 학습 안정화**로 실제 범용 backbone이 되도록 밀어붙인 것임(다음 p.57에서 비교).
- 이전 방법과의 차이: DINO는 ImageNet-1K에서의 연구용 방법이었고, DINOv2는 큰 curated 데이터와 큰 모델로 **foundation model family**를 목표로 함.

> 보충: Oquab et al., 2024(TMLR)가 DINOv2 원 논문이고, 이름의 DINO는 Caron et al., 2021 "Emerging Properties in Self-Supervised Vision Transformers" (self-**di**stillation with **no** labels)임. 원 논문의 PCA 시각화는 첫 성분으로 전경을 분리한 뒤 나머지 성분을 RGB로 표시하는 방식으로 알고 있으나, 슬라이드에는 설명이 없음.

## FIGURES

- p.49: 반복 caption 생성 pipeline 도식(Context 박스 → Llama 2 → Generated Caption → context에 다시 추가). 중요도: 상
- p.50: 같은 caption에서 서로 다른 noise로 생성한 이미지 4장 예시(paella, power plant, junco). 중요도: 상
- p.51: Multi-positive contrastive 도식(Caption → 이미지 4장 → 2D feature space에서 anchor/positive/negative) + 수식 $\mathbf{q}_i,\mathbf{p}_i,\mathcal{L}$. 중요도: 상
- p.52: Table 6 (ImageNet linear eval + fine-grained transfer 비교표). 중요도: 상
- p.53: Table 7 (ADE20K segmentation 비교표). 중요도: 중
- p.56: DINOv2 PCA visualization (patch feature가 category와 domain을 넘어 part 단위로 대응). 중요도: 상

## EXAM POINTS

1. **SynCLR pipeline 3단계**: (1) Llama-2로 in-context iterative caption 생성(150M) → (2) caption당 서로 다른 noise로 이미지 4장 생성(600M, low CFG) → (3) multi-positive contrastive learning. 각 단계의 목적을 서술할 수 있어야 함.
2. **Low CFG scale을 쓰는 이유**: guidance가 낮으면 diversity가 커져 positive 간 view variation이 풍부해짐. 높으면 caption에 과하게 맞추느라 이미지가 비슷해짐.
3. **Multi-positive loss 수식**: $\mathbf{q}_i$(softmax over similarity/$\tau$), $\mathbf{p}_i$(same-caption indicator의 정규화), $\mathcal{L}=H(\mathbf{p},\mathbf{q})=-\sum_i \mathbf{p}_i\log\mathbf{q}_i$. positive가 1개면 InfoNCE로 환원됨을 설명할 것.
4. **SimCLR vs Supervised vs SynCLR의 positive 정의 비교**(p.48과 묶임): SimCLR는 이미지 1장 = 1 class, supervised는 label class, SynCLR는 caption 하나로 만든 이미지 집합 = 1 class. SynCLR는 positive가 같은 원본 이미지일 필요가 없음.
5. **"Learning from models" 결과 해석**: synthetic-only인 SynCLR가 ImageNet linear eval에서 CLIP과 비슷하고, ImageNet 포함 10개 데이터셋 Average 90.4(ViT-L)로 표에서 가장 높으며, ADE20K에서도 DINOv2와 동급임. DINOv2는 ViT-g distillation의 이점이 있다는 점 유의.
6. **Why caption-based class**: caption이 visual class의 **granularity**를 조절하고, 객체뿐 아니라 action과 relationship까지 담을 수 있음.
7. **"What changed?" 한 줄 요약**: hand-designed augmentation → generative semantic view. 한계도 함께 말할 수 있어야 함(생성 모델의 bias, coverage 의존).
8. **DINOv2의 목표와 PCA 시각화의 의미**: label/caption 없는 image-only SSL로 frozen general-purpose feature(global + patch-level)를 학습. PCA 색 대응은 서로 다른 category/domain에서도 part-level semantic correspondence가 있음을 보임. DINO(emergent properties) → DINOv2(general-purpose)의 차이를 설명할 것.

## VERIFIER REPORT

- p.52 설명: Cars를 "특히 강함"으로 서술한 과장을 수정 (OpenCLIP 2B 95.3이 더 높음; Aircraft만 두드러짐으로 정정).
- EXAM POINTS 5: "fine-grained 평균(Average)"를 "ImageNet 포함 10개 데이터셋 Average"로 정정.
- p.56 설명: PCA 시각화 설명을 원 논문 방식(전경 분리 후 상위 3성분 RGB)과 일치시킴.
- 표(p.52, p.53) 수치, 수식(p.51), 그림 설명, 인용 연도는 슬라이드와 대조 후 이상 없음.
