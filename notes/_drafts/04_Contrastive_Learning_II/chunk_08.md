<!-- DRAFT | lecture: 04_Contrastive_Learning_II | chunk: 08 | pages 57–64 -->
<!-- VERIFIED | fixes: 2 | verifier: sonnet -->

## What Changed from DINO to DINOv2? (p.57)

표 (DINO vs DINOv2):

| | DINO | DINOv2 |
|---|---|---|
| 데이터 | ImageNet-1K | Curated 142M images |
| 학습 수준 | Image-level self-distillation | Image-level and patch-level learning |
| 모델 크기 | ViT-S/B | Up to ViT-g, about 1B parameters |
| 표현의 성격 | Semantic representation | General-purpose visual features |
| 위상 | Research method | Foundation model family |

핵심 식 (슬라이드 중앙):

$$\text{DINOv2} = \text{DINO} + \text{iBOT} + \text{KoLeo} + \text{Curated Data} + \text{Scaling}$$

하단 문장: "The core of DINOv2 is not a single, entirely new loss function; rather, it is an integrated system that combines existing self-supervised learning (SSL) components and stabilizes training to enable scaling up both data and model size."

**설명**
- DINOv2의 핵심 메시지는 **새로운 loss가 아니라 "기존 SSL 부품 + 데이터 + 스케일"을 안정적으로 합친 system**이라는 점임.
- 5개 행 모두 "연구용 방법 → 범용 foundation model"로의 이동을 나타냄. 특히 **데이터(1.28M → 142M, 약 100배)** 와 **모델(ViT-S/B → ViT-g 약 1B)** 이 동시에 커짐.
- 구성 요소별 역할
  - **DINO**: CLS token 기반 image-level self-distillation (global semantic).
  - **iBOT**: masked patch 기반 patch-level self-distillation (local/spatial).
  - **KoLeo**: batch 내 feature를 embedding 공간에 고르게 퍼뜨리는 regularizer.
  - **Curated Data (LVD-142M)**: 웹 이미지를 dedup + retrieval로 정제한 데이터.
  - **Scaling**: 큰 teacher를 학습하고 작은 모델로 distill (p.65 이후 내용).
- 방법이 "scale에서 안정적으로 학습되는가"가 연구 포인트였기 때문에, loss보다 **engineering/recipe** 쪽 기여가 큼.

> 보충: DINOv2 논문(Oquab et al., TMLR 2024)은 이 외에도 untied head(DINO head와 iBOT head 분리), Sinkhorn-Knopp centering, 고해상도 adaptation, efficient stochastic depth/FSDP 등 학습 효율화 기법을 포함함. 슬라이드의 수식 "DINO + iBOT + KoLeo"는 이를 요약한 표현임.

## Curating 142M Unlabeled Images (p.58)

그림: 왼쪽에서 오른쪽으로 흐르는 3단계 파이프라인 다이어그램. 상단 회색 원통 "Uncurated Data"(여러 음식/자동차 이미지), 하단 주황색 원통 "Curated Data"(음식 이미지 1장). 점선으로 구분된 세 단계 **Embedding → Deduplication → Retrieval**.
- Embedding: 모든 이미지가 embedding(옆의 격자 아이콘)으로 변환됨.
- Deduplication: 중복된 이미지 2장에 빨간 X 표시가 붙어 제거되고 하나만 남음.
- Retrieval: curated 이미지의 embedding을 기준으로(아래→위 화살표) 비슷한 uncurated 이미지만 남기고 나머지(흐릿한/빨간 X)는 버림.
- 오른쪽 끝: "Augmented Curated Data" 원통 — 회색(검색으로 가져온 uncurated 이미지) + 주황색(원래 curated 이미지)이 합쳐진 형태.

텍스트:
- 1. Start with curated and uncurated datasets
  - a. Curated datasets provide diverse reference concepts.
    - i. ImageNet-22k, ImageNet-1k train, Google Landmarks,
  - b. Uncurated web images provide scale.
    - i. 1.2B images from webs
- 2. Remove duplicates (Pizzi et al., CVPR 2022)
  - a. Map images into an embedding space.
  - b. Remove near-duplicate images.
- 3. Retrieve similar images
  - a. Match uncurated images to curated examples (use self-supervised ViT-H/16).
  - b. Rebalance the data around useful visual concepts.
- LVD-142M: 142 million curated images without manual labels
- Scaling requires more data, but useful scaling requires curated data.

**설명**
- 문제의식: 웹 크롤링 이미지는 양은 많지만 **분포가 편향(long-tail, 중복, 저품질)** 되어 있어, 그냥 키우면 scaling 효과가 떨어짐. 그래서 "더 많은 데이터"가 아니라 "**쓸모 있는 데이터**"가 필요함.
- 설계 직관: **curated set을 "원하는 개념의 anchor"로 쓰고**, 웹 이미지 중 그 anchor와 가까운 것만 끌어와 데이터셋을 키움 (**label 없이 curation**). 즉 label은 안 쓰지만 curated 데이터셋의 "분포"를 간접적으로 사용함.
- 3단계
  1. Embedding: self-supervised ViT-H/16으로 이미지 embedding 추출 (label-free라 caption/라벨 불필요).
  2. Dedup: 근접 중복 제거 → 학습 중 같은 이미지가 반복되어 생기는 memorization/bias와 benchmark leakage 완화.
  3. Retrieval: curated 이미지를 query로 uncurated 이미지에서 nearest neighbor를 가져와 **개념 분포를 rebalance**.
- 결과물 **LVD-142M** = 142M 이미지의 curated 데이터셋 (슬라이드·논문 모두 약어 풀이를 명시하지 않음).
- 이전(DINO)과 차이: DINO는 ImageNet-1K(약 1.28M)를 그대로 썼지만, DINOv2는 **데이터 자체를 설계 대상**으로 삼음.

> 보충: 논문에서 retrieval은 두 가지 방식을 씀. 대형 curated set(ImageNet-22k 등)은 **sample-based retrieval**(각 query의 k개 이웃을 가져옴), 작은 curated set은 **cluster-based retrieval**(uncurated 이미지를 k-means로 군집화한 뒤 cluster 단위로 샘플링). p.59 표의 "sample"/"cluster"/"as is"가 이를 뜻함. 또한 슬라이드의 "ImageNet-22k, ImageNet-1k train, Google Landmarks"는 일부 예시이며 실제로는 더 많은 데이터셋이 포함됨.

## LVD-142M 구성 표 (Table 15, Table 18) (p.59)

슬라이드는 제목 없이 DINOv2 논문의 두 표를 그대로 붙인 것임.

**왼쪽: Table 15 — Composition of our LVD-142M dataset.** (열: Task, Dataset/Split, Images, Retrieval, Retrieved, Final)

| Task | Dataset / Split | Images | Retrieval | Retrieved | Final |
|---|---|---|---|---|---|
| classification | ImageNet-22k / – | 14,197,086 | as is | – | 14,197,086 |
| classification | ImageNet-22k / – | 14,197,086 | sample | 56,788,344 | 56,788,344 |
| classification | ImageNet-1k / train | 1,281,167 | sample | 40,997,344 | 40,997,344 |
| fine-grained classif. | Caltech 101 / train | 3,030 | cluster | 2,630,000 | 1,000,000 |
| fine-grained classif. | CUB-200-2011 / train | 5,994 | cluster | 1,300,000 | 1,000,000 |
| fine-grained classif. | DTD / train1 | 1,880 | cluster | 1,580,000 | 1,000,000 |
| fine-grained classif. | FGVC-Aircraft / train | 3,334 | cluster | 1,170,000 | 1,000,000 |
| fine-grained classif. | Flowers-102 / train | 1,020 | cluster | 1,060,000 | 1,000,000 |
| fine-grained classif. | Food-101 / train | 75,750 | cluster | 21,670,000 | 1,000,000 |
| fine-grained classif. | Oxford-IIIT Pet / trainval | 3,680 | cluster | 2,750,000 | 1,000,000 |
| fine-grained classif. | Stanford Cars / train | 8,144 | cluster | 7,220,000 | 1,000,000 |
| fine-grained classif. | SUN397 / train1 | 19,850 | cluster | 18,950,000 | 1,000,000 |
| fine-grained classif. | Pascal VOC 2007 / train | 2,501 | cluster | 1,010,000 | 1,000,000 |
| segmentation | ADE20K / train | 20,210 | cluster | 20,720,000 | 1,000,000 |
| segmentation | Cityscapes / train | 2,975 | cluster | 1,390,000 | 1,000,000 |
| segmentation | Pascal VOC 2012 (seg.) / trainaug | 1,464 | cluster | 10,140,000 | 1,000,000 |
| depth estimation | Mapillary SLS / train | 1,434,262 | as is | – | 1,434,262 |
| depth estimation | KITTI / train (Eigen) | 23,158 | cluster | 3,700,000 | 1,000,000 |
| depth estimation | NYU Depth V2 / train | 24,231 | cluster | 10,850,000 | 1,000,000 |
| depth estimation | SUN RGB-D / train | 4,829 | cluster | 4,870,000 | 1,000,000 |
| retrieval | Google Landmarks v2 / train (clean) | 1,580,470 | as is | – | 1,580,470 |
| retrieval | Google Landmarks v2 / train (clean) | 1,580,470 | sample | 6,321,880 | 6,321,880 |
| retrieval | AmsterTime / new | 1,231 | cluster | 960,000 | 960,000 |
| retrieval | AmsterTime / old | 1,231 | cluster | 830,000 | 830,000 |
| retrieval | Met / train | 397,121 | cluster | 62,860,000 | 1,000,000 |
| retrieval | Revisiting Oxford / base | 4,993 | cluster | 3,680,000 | 1,000,000 |
| retrieval | Revisiting Paris / base | 6,322 | cluster | 3,660,000 | 1,000,000 |
| **합계** | | | | | **142,109,386** |

캡션: "Table 15: Composition of our LVD-142M dataset. We report the list of datasets and associated splits used to build the dataset, how they were included (as is without retrieval or via sample-based or cluster-based retrieval). For retrievals, we indicate the actual number of retrieved images and the final number included in the dataset. We chose to include as many datasets as possible in the pretraining data in order to cover as many domains as possible. We kept a few datasets aside in order to evaluate performance outside of the pretraining domain. More details about dataset usages can be found in Table 18."

**오른쪽: Table 18 — List of datasets used.** (열: Dataset, Pretraining (as is), Retrieving pretraining data, Eval., Task, Citation)

| Dataset | Pretraining (as is) | Retrieving data | Eval. | Task | Citation |
|---|---|---|---|---|---|
| ImageNet-1k | ✗ | ✓ | ✓ | Classif. | Russakovsky et al., 2015 |
| ImageNet-22k | ✓ | ✓ | ✗ | – | Deng et al., 2009 |
| ImageNet-V2 | ✗ | ✗ | ✓ | Classif. | Recht et al., 2019 |
| ImageNet-ReaL | ✗ | ✗ | ✓ | Classif. | Beyer et al., 2020 |
| ImageNet-A | ✗ | ✗ | ✓ | Classif. | Hendrycks et al., 2021b |
| ImageNet-C | ✗ | ✗ | ✓ | Classif. | Hendrycks & Dietterich, 2019 |
| ImageNet-R | ✗ | ✗ | ✓ | Classif. | Hendrycks et al., 2021a |
| ImageNet-Sk. | ✗ | ✗ | ✓ | Classif. | Wang et al., 2019 |
| Food-101 | ✗ | ✓ | ✓ | Classif. | Bossard et al., 2014 |
| CIFAR-10 / CIFAR-100 | ✗ | ✓ | ✓ | Classif. | Krizhevsky et al., 2009 |
| SUN397 | ✗ | ✓ | ✓ | Classif. | Xiao et al., 2010 |
| StanfordCars | ✗ | ✓ | ✓ | Classif. | Krause et al., 2013 |
| FGVC-Aircraft | ✗ | ✓ | ✓ | Classif. | Maji et al., 2013 |
| VOC 2007 | ✗ | ✓ | ✓ | Classif. | Everingham et al., 2010 |
| DTD | ✗ | ✓ | ✓ | Classif. | Cimpoi et al., 2014 |
| Oxford Pets | ✗ | ✓ | ✓ | Classif. | Parkhi et al., 2012 |
| Caltech101 | ✗ | ✓ | ✓ | Classif. | Fei-Fei et al., 2004 |
| Flowers | ✗ | ✓ | ✓ | Classif. | Nilsback & Zisserman, 2008 |
| CUB200 | ✗ | ✓ | ✓ | Classif. | Welinder et al., 2010 |
| iNaturalist 2018 / 2021, Places-205 | ✗ | ✗ | ✓ | Classif. | Van Horn et al., 2018 / 2021, Zhou et al., 2014 |
| UCF101, Kinetics-400, SSv2 | ✗ | ✗ | ✓ | Video | Soomro et al., 2012; Kay et al., 2017; Goyal et al., 2017 |
| GLD v2 | ✓ | ✓ | ✗ | – | Weyand et al., 2020 |
| R-Paris, R-Oxford | ✗ | ✓ | ✓ | Retrieval | Radenović et al., 2018a |
| Met | ✗ | ✓ | ✓ | Retrieval | Ypsilantis et al., 2021 |
| Amstertime | ✗ | ✓ | ✓ | Retrieval | Yildiz et al., 2022 |
| ADE20k, Cityscapes, VOC 2012 | ✗ | ✓ | ✓ | Seg. | Zhou et al., 2017; Cordts et al., 2016; Everingham et al., 2010 |
| Mapillary SLS | ✓ | ✗ | ✗ | – | Warburg et al., 2020 |
| NYU-Depth V2, KITTI, SUN-RGBD | ✗ | ✓ | ✓ | Depth | Silberman et al., 2012; Geiger et al., 2013; Song et al., 2015 |
| DollarStreet, Casual Conv. | ✗ | ✗ | ✓ | Fairness | De Vries et al., 2019; Hazirbas et al., 2021 |

**설명**
- 이 슬라이드는 설명용이 아니라 **레퍼런스용 표**임. 핵심은 "LVD-142M이 어떤 curated 데이터셋을 anchor로 retrieval해서 만들어졌는가"를 보여주는 것.
- Table 15 읽는 법
  - Retrieval 열: **as is**(그대로 포함), **sample**(sample-based: 이미지별 이웃 검색), **cluster**(cluster-based: 군집 단위 검색).
  - Retrieved는 실제로 검색된 수, Final은 최종 포함된 수. cluster 방식은 데이터셋당 **최대 1M으로 cap**이 걸림 (예: Food-101은 21.67M이 검색되지만 Final은 1,000,000).
  - 가장 큰 비중은 ImageNet-22k(as-is 14.2M + sample 56.8M)와 ImageNet-1k(sample 41.0M)임. 합계 **142,109,386장**.
- Table 18 읽는 법: 각 데이터셋이 (1) 그대로 pretraining에 쓰였는지, (2) retrieval의 query(anchor)로 쓰였는지, (3) 평가용인지를 ✓/✗로 표시. **Eval-only 데이터(ImageNet-A/C/R/Sk., iNaturalist, video 등)** 는 pretraining 영역 밖의 일반화 성능을 보기 위해 일부러 제외함.
- 포인트: **평가 benchmark의 train split을 retrieval anchor로 쓴다**는 점이 논란거리일 수 있으나, 라벨은 안 쓰고 이미지 분포만 활용하며 test split은 제외함.

> 보충: 표의 세부 수치는 시험 대상이 아님. 중요한 것은 "합계 142M", "sample vs cluster retrieval", "cluster는 1M cap"이라는 구조 정도임.

## Detecting Near-Duplicate Images (p.60)

텍스트:
- Near-duplicate detection with SSCD
  - Encode each image into a compact descriptor.
  - Compare descriptors using cosine similarity.
  - Retrieve the nearest neighbors for each image.
  - Remove images whose similarity exceeds a predefined threshold.

그림: 좌하단의 작은 플로우차트. 위에서 아래로 세 박스: 수식 $\text{sim}(z_i, z_j) > \tau$ → (화살표) "Near-duplicate" 분홍색 박스 → (화살표) "Remove" 외곽선 박스.

출처: Pizzi et al., CVPR 2022, A self-supervised descriptor for image copy detection

**설명**
- **SSCD** (Self-Supervised descriptor for image Copy Detection, Pizzi et al., CVPR 2022)는 이미지를 compact한 descriptor $z$로 만드는 모델로, 원래 **이미지 복제(copy)·편집 탐지**용으로 SimCLR 계열 contrastive 학습 + entropy regularizer로 학습함.
- 절차: 각 이미지를 $z_i$로 인코딩 → cosine similarity $\text{sim}(z_i,z_j)=\frac{z_i^\top z_j}{\|z_i\|\|z_j\|}$ → nearest neighbor 검색 → $\text{sim} > \tau$이면 near-duplicate로 판정하고 제거.
- 왜 필요한가: 웹 이미지는 크기 조정, crop, 색 변화, 워터마크 등으로 중복이 많음. 중복을 방치하면 특정 이미지가 과대표현되어 학습이 편향되고, 평가 set과의 중복(leakage)도 생김. **pixel 단위 hash로는 못 잡는 변형된 중복**을 embedding 유사도로 잡는 것이 포인트임.
- 1.2B 규모에서는 정확한 전수 비교 대신 **approximate nearest neighbor (Faiss 등)** 가 필요함.

> 보충: 논문(DINOv2)에서는 SSCD copy detection으로 uncurated 데이터 내 중복과 benchmark test/val 이미지와의 중복을 모두 제거했다고 서술함. threshold $\tau$ 값은 슬라이드에 없음 ("predefined threshold").

## Image-Level and Patch-Level Learning (p.61)

텍스트:
- Image-level objective (Caron et al., 2021).
  - Match the student and teacher **CLS-token** distributions.
    - Use cross-entropy so that the student follows the teacher's CLS.
  - Learn global semantic representations.

$$\mathcal{L}_{DINO} = -\sum p_t \log p_s$$

- Patch-level objective (Zhou et al., 2022a).
  - Mask some student patches.
  - Use the visible teacher patches as targets.
  - Learn local and spatial representations.
  - Separate the DINO head and the iBOT head.

$$\mathcal{L}_{iBOT} = -\sum_i p_{ti} \log p_{si}$$

"$i$ is the position of the masked patch."

하단 출처: Caron et al., ICCV 2019, Emerging properties in self-supervised vision transformers. ← 출처 연도 오타 의심 (DINO는 ICCV 2021). / Zhou et al., ICLR 2022, ibot: Image bert pre-training with online tokenizer.

**설명**
- **Image-level (DINO)**: student와 teacher가 같은 이미지의 서로 다른 augmented view를 받고, 각각의 **CLS token** 출력에 head(MLP + softmax)를 적용해 확률분포 $p_s, p_t$를 만듦. student가 teacher 분포를 따라가도록 **cross-entropy** $-\sum p_t\log p_s$를 최소화. teacher는 gradient가 없는 **EMA(exponential moving average) of student**이며 stop-gradient. → 이미지 전체의 **global semantic**을 학습.
  - $p_t$: teacher softmax 출력(target), $p_s$: student softmax 출력(prediction). 합은 prototype(출력 차원) 위에서 취함.
- **Patch-level (iBOT)**: BERT의 masked language modeling을 vision에 적용한 것. student 입력에서 일부 patch를 **mask**하고, 그 위치 $i$에서 student가 낸 patch token 분포 $p_{si}$가 **teacher가 (mask 없이) 본 같은 위치의 patch token 분포** $p_{ti}$를 따라가도록 cross-entropy. → **local/spatial** 정보 학습, dense task(segmentation, depth)에 유리.
  - 슬라이드의 "Use the visible teacher patches as targets"는 "teacher는 masking 되지 않은(visible) 전체 patch를 보고, 그 중 student가 가려진 위치 $i$의 token이 target이 된다"로 읽어야 함. ← 표현이 모호함 (teacher patch가 visible하다는 의미이지 student의 visible patch가 target인 것이 아님).
- **Separate heads**: DINOv2는 image-level과 patch-level objective에 **서로 다른 head(untied)** 를 씀. head를 공유하면 두 목표가 서로 간섭하고, 작은 scale에선 patch-level이 좋아도 큰 scale에선 image-level이 손해를 보는 현상이 있어, 분리가 scale에서 더 낫다고 보고함.
- DINO와의 차이: DINO는 CLS(전역)만 학습해서 dense 표현이 약했고, DINOv2는 **iBOT을 추가해 global + local을 동시에** 학습.

> 보충: 최종 loss는 $\mathcal{L} = \mathcal{L}_{DINO} + \mathcal{L}_{iBOT} + 0.1\,\mathcal{L}_{KoLeo}$ 형태 (Oquab et al., 2024). 슬라이드의 $\sum p_t\log p_s$는 간략화한 표기이며, 실제 DINO/iBOT은 teacher에 temperature 조절 + centering(DINOv2에서는 SK centering)을 적용한 softmax를 씀. iBOT에서 "online tokenizer"란 별도 tokenizer(dVAE) 없이 teacher network 자체가 tokenizer 역할을 한다는 뜻임.

## Preventing Feature Collapse and Crowding (p.62)

텍스트:
- KoLeo regularizer (Sablayrolles et al., 2019).
  - Image-level features are distributed across the entire embedding space.
  - It pushes apart representations of different images so that they do not cluster too closely together.

$$\mathcal{L}_{koleo} = -\frac{1}{n}\sum_{i=1}^{n}\log(d_{n,i}),$$
$$d_{n,i} = \min_{j\neq i}\|x_i - x_j\|$$

- Sinkhorn-Knopp centering (Caron et al., 2020).
  - Normalization/centering of teacher output.
  - The teacher predictions are re-normalized so that assignments are more evenly distributed across prototypes.
  - This avoids the case where many images are mapped to only a small subset of prototypes.
  - As a result, the student receives more informative and less degenerate training targets.

출처: Sablayrolles et al., ICLR 2019, Spreading vectors for similarity search.

**설명**
- 문제: self-distillation은 **collapse**(모든 입력이 같은 출력 / 소수 prototype만 사용)와 **crowding**(서로 다른 이미지의 feature가 좁은 영역에 몰림)의 위험이 있음. 이 슬라이드의 두 기법이 각각 대응함.
- **KoLeo** (Kozachenko-Leonenko entropy estimator 기반)
  - $x_i$: $\ell_2$-normalized feature, $d_{n,i}$: batch 내에서 $x_i$와 **가장 가까운 이웃**까지의 거리.
  - $-\log d_{n,i}$를 최소화 = 최근접 이웃 거리를 **키움** → 가까운 점들을 서로 밀어내어 feature가 hypersphere 위에 **균일하게(uniformly)** 퍼지게 함. 가장 가까운 이웃만 보므로 연산이 저렴하고, $\log$ 때문에 이미 멀리 있는 점에는 gradient가 작음(가까운 점에 집중).
  - 직관: batch entropy를 높이는 효과 (Kozachenko-Leonenko entropy 추정치와 동일 형태). 논문의 원래 용도는 similarity search에서 vector를 균일하게 spread시켜 nearest neighbor 검색 품질을 올리는 것.
  - **CLS(image-level) feature에만** 적용되고, batch 내에서(필요 시 작은 그룹 단위로) 계산함.
- **Sinkhorn-Knopp (SK) centering** (SwAV, Caron et al., 2020)
  - DINO의 단순 mean centering(+sharpening) 대신, teacher의 prototype 할당 행렬을 **SK 알고리즘으로 정규화**해 batch 내 각 prototype의 사용량이 균등(equipartition)해지도록 함. 행/열을 번갈아 정규화하는 iterative normalization.
  - 효과: 소수 prototype에만 샘플이 쏠리는 **assignment collapse**를 막고, student가 더 informative한 target을 받음. 구체 예시는 p.63.
- 비교: KoLeo는 **student feature 공간의 분산(uniformity)** 을, SK centering은 **teacher target 분포의 균형**을 담당 → 서로 보완적.

> 보충: DINOv2 논문은 DINO의 centering을 SwAV의 SK batch normalization으로 교체한 것으로 서술함 (iBOT 헤드에도 적용). KoLeo 가중치는 0.1. ← 슬라이드는 "Caron et al., 2020"으로만 표기, SwAV 논문(Unsupervised learning of visual features by contrasting cluster assignments, NeurIPS 2020)임.

## Sinkhorn-Knopp centering (p.63)

그림: 세 개의 패널 (왼쪽→오른쪽 흐름).
1. **Teacher outputs** (왼쪽): 이미지 6장 (개, 고양이, 새, 산, 자동차, 꽃) → 각각 $x_1 \dots x_6$ 노드로 인코딩 → 4개의 prototype $P_1\dots P_4$ 사각형으로 향하는 다수의 화살표(완전 연결 형태). 하단 라벨 "raw similarities / assignments".
2. **Before SK** (가운데): "Prototype assignment matrix Q (6×4)" 히트맵과 그 아래 "Prototype usage (sum over batch)" 막대그래프. 빨간 박스: "Few prototypes dominate (assignment collapse)".

   | | $P_1$ | $P_2$ | $P_3$ | $P_4$ |
   |---|---|---|---|---|
   | $x_1$ | 0.80 | 0.15 | 0.03 | 0.02 |
   | $x_2$ | 0.70 | 0.20 | 0.07 | 0.03 |
   | $x_3$ | 0.65 | 0.25 | 0.07 | 0.03 |
   | $x_4$ | 0.10 | 0.80 | 0.07 | 0.03 |
   | $x_5$ | 0.15 | 0.70 | 0.10 | 0.05 |
   | $x_6$ | 0.05 | 0.60 | 0.25 | 0.10 |

   막대그래프: $P_1$ 약 2.5, $P_2$ 약 1.5, $P_3$ 약 0.6, $P_4$ 약 0.4.
3. 화살표 "Sinkhorn-Knopp normalization" 박스를 거쳐 → **After SK** (오른쪽): 같은 6×4 행렬과 막대그래프. 초록 박스: "Balanced prototype usage".

   | | $P_1$ | $P_2$ | $P_3$ | $P_4$ |
   |---|---|---|---|---|
   | $x_1$ | 0.40 | 0.30 | 0.20 | 0.10 |
   | $x_2$ | 0.25 | 0.25 | 0.25 | 0.25 |
   | $x_3$ | 0.20 | 0.30 | 0.30 | 0.20 |
   | $x_4$ | 0.25 | 0.20 | 0.30 | 0.25 |
   | $x_5$ | 0.15 | 0.25 | 0.30 | 0.30 |
   | $x_6$ | 0.25 | 0.20 | 0.25 | 0.30 |

   막대그래프: 네 prototype 모두 약 1.5.

하단 문장: "The final weights of the projection head act as prototype-like directions."

**설명**
- **Prototype assignment matrix $Q$**: 행 = batch 내 샘플, 열 = prototype, 원소 $Q_{ik}$ = 샘플 $i$가 prototype $k$에 할당될 확률(teacher softmax 출력). 각 행의 합 = 1.
- **Before SK**: 샘플 $x_1,x_2,x_3$(개/고양이/새)가 모두 $P_1$에, $x_4,x_5,x_6$는 $P_2$에 몰림 → $P_3, P_4$는 거의 안 쓰임 = **assignment collapse (dead prototype)**. 이 상태에서 student가 teacher를 따르면 정보량이 낮은 target만 학습하게 됨.
- **After SK**: 행 합 = 1을 유지하면서 **열 합이 균등(batch size/prototype 수 = 6/4 = 1.5)** 해지도록 정규화. 한 prototype에 쏠린 확률을 다른 prototype으로 재분배 → 모든 prototype을 사용 (equipartition).
- SK 알고리즘: $Q$에 대해 (1) 열 방향 정규화(각 prototype 총량 = $B/K$), (2) 행 방향 정규화(각 샘플 합 = 1)를 번갈아 수 iteration 반복하는 optimal-transport 기반 정규화.
- "final weights of the projection head act as prototype-like directions": head의 마지막 linear layer의 가중치 열 벡터가 곧 prototype 벡터 역할을 하고, feature와 prototype의 내적(유사도)이 logit이 됨.
- 그림의 수치 오류 의심
  - Before SK 막대그래프의 $P_2$ 값 ← 행렬 열 합(0.15+0.20+0.25+0.80+0.70+0.60 = 2.70)과 달리 막대는 약 1.5로 그려져 있음 (2.7이 맞음).
  - Before SK 막대그래프의 $P_4$ 값 ← 행렬 열 합(0.02+0.03+0.03+0.03+0.05+0.10 = 0.26)과 달리 막대는 약 0.4로 그려져 있음 ($P_1$ 열 합 2.45, $P_3$ 열 합 0.59는 막대와 대체로 일치).
  - After SK 행렬의 열 합 ← $P_3$ = 1.60, $P_4$ = 1.40으로 정확히 1.5는 아님 (막대그래프는 모두 1.5로 그려짐). 개념 설명용 도식으로 이해할 것.

> 보충: SK는 SwAV(Caron et al., NeurIPS 2020)에서 clustering assignment를 균등 분할 제약 하에 구하기 위해 도입됨. DINO는 이 대신 teacher logit에 **running mean center + sharpening(낮은 temperature)** 를 써서 collapse를 피했고, DINOv2는 SK로 바꿈.

## DINOv2 전체 파이프라인 요약 (p.64)

제목 없는 슬라이드 (그림 요약). 구성:
- **Section 3. Data Curation Pipeline** (파란 박스, 위→아래 화살표): "Large-scale web images (1.2B)" → "Deduplication" → "SSL embedding retrieval" → "**LVD-142M**" (이미지 모자이크 아이콘).
- 굵은 화살표로 아래 박스로 연결.
- **Section 4. Self-supervised Pre-training** (주황 박스): "Student / Teacher" (teacher 옆 눈송이 아이콘 = frozen/stop-gradient 표시) 에서 두 갈래로 분기.
  - 왼쪽: **CLS token → DINO → global features** (지구본 아이콘)
  - 오른쪽: **Patch token → iBOT → local features** (격자 아이콘)
  - 하단 점선 박스에 세 개 추가 요소: **+ KoLeo**, **+ SK centering**, **+ high-res adaptation**
- 최하단 요약: "DINOv2 = curated large-scale data + image-level and patch-level self-distillation."
- 우측 bullet: "Only the last short training epoch is trained at 518×518"

**설명**
- 지금까지(p.57~63)의 내용을 한 장으로 정리한 슬라이드. 흐름은 **(Section 3) 데이터 만들기 → (Section 4) self-supervised pretraining** (섹션 번호는 DINOv2 논문의 섹션 번호).
- 학습 구조: student/teacher self-distillation에서 **CLS token은 DINO head → global**, **patch token은 iBOT head → local**. 여기에 KoLeo와 SK centering이 collapse/crowding을 막고, **high-res adaptation**이 해상도를 올림.
- **high-res adaptation**: pixel-level task(segmentation, depth)는 고해상도가 유리하지만 전 과정을 고해상도로 학습하면 비싸므로, **마지막 짧은 구간만 518×518**로 학습함 (본 학습은 224×224). 비용 대비 효율이 좋음.
- 결론: DINOv2의 차별점 = **좋은 데이터(curation) + 두 수준의 distillation + 안정화 기법 + scale**.

> 보충: teacher의 눈송이 아이콘은 teacher에 gradient가 흐르지 않음을 상징 (EMA로만 갱신). 518 = 14(patch size) × 37.

## FIGURES

- p.58: LVD-142M curation 파이프라인 (Embedding → Deduplication → Retrieval, curated/uncurated 원통 도식) — 중요도 상
- p.63: Sinkhorn-Knopp centering before/after (assignment matrix + prototype usage 막대그래프) — 중요도 상
- p.64: DINOv2 전체 파이프라인 (Data curation + Self-supervised pretraining 구조도) — 중요도 상
- p.57: DINO vs DINOv2 비교 표 — 중요도 중
- p.59: LVD-142M 구성 표 (Table 15) — 중요도 중 (요약 수치만 필요, 전체 표는 불필요)
- p.60: near-duplicate 판정 플로우 ($\text{sim}(z_i,z_j)>\tau$ → Remove) — 중요도 중

## EXAM POINTS

1. **DINOv2 = DINO + iBOT + KoLeo + Curated Data + Scaling**: 새 loss가 아니라 기존 SSL 요소를 통합하고 안정화해 data/model을 scale한 system임. DINO와의 차이 5가지(데이터, 학습 수준, 모델 크기, 표현, 위상)를 말할 수 있어야 함.
2. **LVD-142M curation 3단계**: (1) curated + 1.2B uncurated 준비 → (2) SSCD 기반 dedup → (3) self-supervised ViT-H/16 embedding retrieval로 curated 이미지와 유사한 이미지 선택. **"scaling에는 more data가 아니라 curated data가 필요"** 이유 서술 (why 질문).
3. **Near-duplicate 제거**: SSCD descriptor + cosine similarity + threshold $\tau$ (Pizzi et al., CVPR 2022). 왜 중복을 제거하는지(bias, memorization, leakage).
4. **Image-level vs patch-level loss**: $\mathcal{L}_{DINO}=-\sum p_t\log p_s$ (CLS, global) vs $\mathcal{L}_{iBOT}=-\sum_i p_{ti}\log p_{si}$ ($i$ = masked patch 위치, local). 두 head를 분리(untied)하는 이유.
5. **KoLeo regularizer**: $-\frac1n\sum\log d_{n,i}$, $d_{n,i}=\min_{j\ne i}\|x_i-x_j\|$. 최근접 이웃 거리를 키워 feature를 균일하게 퍼뜨림 (crowding 방지). 수식 의미를 설명할 수 있어야 함.
6. **Sinkhorn-Knopp centering**: teacher의 prototype assignment를 균등 분포로 정규화해 assignment collapse 방지. Before/After 그림 해석 (소수 prototype 독점 → 사용량 균등, 열 합 = $B/K$).
7. **collapse vs crowding 구분**: SK centering = teacher target의 prototype 쏠림 방지, KoLeo = student feature의 밀집 방지. 각각 무엇을 막는지 비교.
8. **high-res adaptation**: 마지막 짧은 epoch만 518×518로 학습하는 이유 (dense task 성능 + 학습 비용 절감).

## VERIFIER REPORT
- p.58 설명: LVD-142M을 "Learning with Diverse data"로 풀이한 부분은 근거 없어 제거 (약어 풀이 미명시로 수정).
- p.63 설명: Before SK 막대그래프 P4(약 0.4)가 행렬 열 합(0.26)과 불일치함을 수치 의심 목록에 추가.
- 그 외 표(Table 15/18) 수치, 수식, 인용 연도, 그림 설명은 원본과 일치 확인.
