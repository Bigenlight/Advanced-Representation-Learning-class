<!-- DRAFT | lecture: 01_Introduction | chunk: 03 | pages 17–24 -->
<!-- VERIFIED | fixes: 3 | verifier: sonnet -->

## What is the Good Representations? (p.17)

- A good representation can be reused across many tasks.
- 그림 (위아래 2단 구조, 가운데 아래 방향 굵은 화살표로 연결):
  - 위 (빨간 박스) **Task-specific model**: 3개 열이 점선으로 구분됨. 각 열은 입력 이미지 → 화살표 → 모델 박스. (1) Object Detection (도로 장면 + bounding box) → Model A (task-specific), (2) Semantic Segmentation (색칠된 도로 장면 마스크) → Model B (task-specific), (3) Tracking (여러 박스/궤적이 표시된 장면) → Model C (task-specific).
  - 아래 (초록 박스) **General-purpose representation** (*Learn once*): 여러 이미지(자동차, 건물, 음식, 동물 등 다양한 사진 격자) → **Representation / Feature Extractor** (e.g., ResNet, ViT, CLIP) → 3갈래로 분기 → Detection, Segmentation, Tracking (+ 세로 점 "⋮"로 더 많은 task 암시).

**설명**
- 위쪽은 task마다 모델을 **처음부터 따로** 학습하는 전통적 방식. 아래쪽은 **한 번 학습한(learn once)** feature extractor를 여러 downstream task에 **재사용**하는 방식임.
- 즉 "좋은 representation"의 가장 실용적인 정의는 **task-agnostic하게 재사용 가능한 feature**임. task별 head만 얹거나 가볍게 fine-tune하면 됨 → 데이터/연산 비용 절감.
- 예시로 든 ResNet(CNN), ViT(Transformer), CLIP(image-text contrastive)은 모두 이런 범용 backbone 역할을 함.
> 보충: 이 관점이 pretrain → transfer (fine-tuning / linear probing) 패러다임의 동기. representation의 품질은 보통 downstream task 성능(linear probe 정확도 등)으로 평가함. ResNet: He et al., 2016. ViT: Dosovitskiy et al., 2020. CLIP: Radford et al., 2021.

## What Makes a Good Representation? (p.18)

- A good representation should:
  - Compress information
  - Support prediction
  - Generalize across tasks

**설명**
- 좋은 representation의 조건을 3가지로 요약함: **압축(Compression)**, **예측 지원(Prediction)**, **task 간 일반화(Generalization)**.
- 앞의 두 개(Compression, Prediction)가 이 강의의 핵심 축이고, 세 번째(Generalize)는 앞 두 조건을 만족했을 때 따라오는 결과로 이후 슬라이드(p.21)에서 "Better Representations"의 성질(Generalize better, Transfer across tasks)로 다시 나옴.
- 다음 슬라이드 p.19, p.20에서 각각 Compression, Prediction을 자세히 설명함.

## Compression (p.19)

- Captures just the essential characteristics of the data.
  - Less Memory
    - Smaller representations are easier to store.
  - Invariance
    - Remove nuisance factors that do not affect semantics.
  - Occam's Razor
    - Among equally useful representations, prefer the simpler one.
- (강조 문구, 하단 중앙) **Compression is not about losing information. It is about removing irrelevant information.**

**설명**
- Compression = 데이터의 **본질적 특성만** 남기는 것. 세 가지 이점:
  - **Less Memory**: representation 차원이 작으면 저장/전송/검색이 쉬움.
  - **Invariance**: 의미(semantics)와 무관한 **nuisance factor**(조명, 시점, 배경, 노이즈 등)를 제거 → 같은 의미의 입력이 비슷한 representation을 가짐.
  - **Occam's Razor**: 같은 유용성이면 **더 단순한** representation을 선호 (일반화에 유리, overfitting 억제).
- 핵심 메시지: 압축은 "정보 손실"이 아니라 "**무관한 정보의 제거**"임. 즉 task에 relevant한 정보는 보존하고 irrelevant한 정보만 버림.
> 보충: 정보이론적으로는 Information Bottleneck (Tishby et al., 2000) 관점과 연결됨. 입력 $X$의 정보는 최대한 버리되 target $Y$에 대한 정보 $I(Z;Y)$는 유지하는 $Z$를 찾는 것: $\min_{Z}\, I(X;Z) - \beta I(Z;Y)$. 슬라이드에는 없는 내용.

## Prediction (p.20)

- Why do we need representations?
  - To predict what we cannot directly observe.
- 그림: 3개 패널 (좌→우), 각 패널 아래에 "예시" 박스.
  - **1. Future**: 위에 "Current frame"(도로 주행 영상 프레임 시퀀스, 가운데 프레임이 파란 테두리로 강조) → 아래 화살표 → "Future frame"(미래 프레임들). 예시: Video prediction, V-JEPA.
  - **2. Past**: "Today"(도로 위 차량 이미지) → 화살표 → "Yesterday?"(점선 테두리의 흐릿한 회색 이미지, 추정해야 할 과거). 예시: State estimation, Temporal reasoning.
  - **3. Missing Data**: "Corrupted image"(산과 호수 풍경, 일부 패치가 회색/검정으로 가려짐) → 화살표 → "Inpainting"(복원된 이미지). 예시: MAE, Image completion.
- (강조 문구) **Prediction means inferring unobserved properties of the world.**

**설명**
- representation이 필요한 이유는 **직접 관측되지 않는 것을 예측**하기 위해서임. 관측되지 않는 대상은 세 종류:
  1. **Future**: 현재까지 관측으로 미래 프레임/상태를 예측 (video prediction, V-JEPA).
  2. **Past**: 현재 관측에서 과거 상태를 추정 (state estimation, temporal reasoning).
  3. **Missing data**: 가려지거나 손상된 부분을 복원 (inpainting, MAE).
- 예측을 잘 하려면 데이터의 **구조/인과/규칙성**을 이해해야 하므로, 예측 과제 자체가 좋은 representation을 학습시키는 신호가 됨.
- 슬라이드의 "예시" 라벨은 한국어(오타 아님, 슬라이드가 한영 혼용).
> 보충: V-JEPA = Video Joint-Embedding Predictive Architecture (Bardes et al., 2024, Meta), pixel이 아니라 latent space에서 가려진 영역을 예측함. MAE = Masked Autoencoders (He et al., 2022), 가려진 patch의 pixel을 복원.

## Good Representations = Compression + Prediction (p.21)

- 그림: 두 원이 겹치는 **벤 다이어그램**. 왼쪽 파란 원 = Compression, 오른쪽 초록 원 = Prediction, 교집합(보라) = Better Representations.
  - **Compression** (파란 원): "Capture essential information while removing redundancy and nuisances."
    - **Autoencoder**: Remove redundant information
    - **Contrastive Learning**: Achieve invariance to transformations
    - **Clustering**: Quantize continuous data into discrete categories
  - **Prediction** (초록 원): "Enable inference of unobserved or future properties of the world."
    - **Future Prediction**: Predict future states or observations
    - **Imputation**: Predict missing data (e.g., inpainting)
    - **Pretext Tasks**: Predict abstract properties of the data
  - **Better Representations** (교집합, 체크 리스트):
    - Generalize better
    - Transfer across tasks
    - Support decision making
    - More sample efficient

**설명**
- 이 슬라이드가 p.17~20의 결론: **좋은 representation = Compression ∩ Prediction**. 두 원리를 **모두** 만족할 때 더 잘 일반화되고, task 간 전이되며, 의사결정을 지원하고, sample efficiency가 높아짐.
- 대표 학습 방법을 원리별로 분류:
  - Compression 계열: Autoencoder (중복 제거), Contrastive Learning (변환에 대한 invariance), Clustering (연속 → 이산 category로 quantize).
  - Prediction 계열: Future Prediction, Imputation, Pretext Tasks.
- 하나의 원리만 쓰면 한계가 있음: 압축만 하면 task에 필요한 정보까지 버릴 수 있고(또는 trivial solution), 예측만 하면 nuisance까지 모델링하려 함(예: pixel 단위 복원). 둘의 균형이 핵심 → 이후 강의에서 각 방법을 이 틀로 해석함.
- 교집합 항목 중 **More sample efficient**: 좋은 representation 위에서는 적은 라벨로도 downstream 학습이 가능하다는 의미.

## Beyond Compression and Prediction (p.22)

- Good representations often exhibit additional desirable properties.
  1. Disentangled
  2. Interpretable
  3. Actionable
  4. Transferable
- These properties are often consequences of learning representations that are both ← 문장이 "both"에서 끊김 (원문 유지; 의도는 "both compressive and predictive"로 추정)
- 그림: 2x2 패널.
  - **1 Disentangled**: 개 사진에서 Identity / Pose / Lighting / Background 4가지 요인이 각각 별도의 코드(색깔 다른 점/사각형/삼각형/별 묶음)로 인코딩됨. 하단에 VAE, $\beta$-VAE, Disentangled Representation Learning 라벨의 latent 분포 산점도.
  - **2 Interpretable**: 신경망 그림에서 Neuron 1, 2, 3이 각각 특정 시각 개념(털 질감, 눈, 꼬리 등 개 부위 이미지)에 대응. 아래에 사람 머리+전구, 막대그래프, 돋보기 아이콘.
  - **3 Actionable**: Representation → Decision → Action 흐름의 박스, 로봇 팔/자율주행 차량/로봇/뇌 아이콘.
  - **4 Transferable**: 하나의 큐브 모양 "One Representation"에서 여러 갈래 화살표로 다양한 task(차트/클러스터링, 개 검출, segmentation, 검색 등)로 연결.

**설명**
- Compression + Prediction을 만족하면 **종종 부수적으로** 따라오는 바람직한 성질 4가지:
  1. **Disentangled**: 변동 요인(identity, pose, lighting, background 등)이 representation의 서로 다른 축/차원에 분리되어 인코딩됨. VAE, $\beta$-VAE가 대표 방법.
  2. **Interpretable**: 개별 뉴런/차원이 사람이 이해할 수 있는 개념과 대응.
  3. **Actionable**: representation에서 바로 decision/action(로봇 제어, 자율주행)을 수행할 수 있음.
  4. **Transferable**: 하나의 representation을 여러 downstream task에 재사용(p.17과 연결).
- 중요: 이 성질들은 **직접 목표로 삼는 것이 아니라 흔히 결과(consequence)**로 나타난다("often consequences")는 것이 슬라이드의 주장. 항상 보장되는 것은 아님.
> 보충: $\beta$-VAE (Higgins et al., 2017)는 VAE의 KL 항에 가중치 $\beta>1$을 줘서 disentanglement를 유도함: $\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \beta\, D_{KL}(q(z|x)\|p(z))$. 단, 비지도 disentanglement가 inductive bias 없이는 보장되지 않는다는 이론 결과도 있음 (Locatello et al., 2019).

## Types of Representation Learners (p.23)

표 (Learning Method / Learning Principle / Short Summary):

| Learning Method | Learning Principle | Short Summary |
|---|---|---|
| Autoencoding | Compression | Remove redundant information |
| Contrastive | Compression | Achieve invariance to viewing transformations |
| Clustering | Compression | Quantize continuous data into discrete categories |
| Future prediction | Prediction | Predict the future |
| Imputation | Prediction | Predict missing data |
| Pretext tasks | Prediction | Predict abstract properties of your data |

**설명**
- p.21 벤 다이어그램의 6가지 방법을 표로 정리한 것. 위 3개는 **Compression**, 아래 3개는 **Prediction** 원리로 분류됨.
- 방법별 직관:
  - **Autoencoding**: 입력을 bottleneck latent로 압축 후 복원 → 중복 제거.
  - **Contrastive**: 같은 sample의 augmentation(viewing transformation)은 가깝게, 다른 sample은 멀게 → 변환에 **invariant**한 representation.
  - **Clustering**: 연속 feature를 이산 cluster/codebook으로 quantize.
  - **Future prediction / Imputation / Pretext tasks**: 미래, 결측, 인위적으로 만든 추상적 속성(예: 회전 각도, patch 순서)을 예측.
- 같은 방법이 두 원리에 걸칠 수도 있음(분류는 대표적 해석 기준).
> 보충: 슬라이드 분류에서 contrastive를 "compression"에 넣은 것은 augmentation invariance가 nuisance 제거(압축)라는 관점 때문. 일부 문헌은 contrastive를 mutual information 최대화(InfoNCE)로 해석하기도 함.

## Next Week.. (p.24)

- Self-Supervised Learning and Pretext Tasks
- 다음 주 주제 예고 슬라이드.

**설명**
- 다음 강의는 **Self-Supervised Learning**과 **Pretext Tasks** (Prediction 계열 방법의 구체 사례)를 다룸.

## FIGURES

- p.17: Task-specific model vs. General-purpose representation(Learn once) 비교 도식 (중요도 상)
- p.20: Prediction의 3 유형 (Future / Past / Missing Data) 패널 (중요도 중)
- p.21: Compression ∩ Prediction 벤 다이어그램 (중요도 상)
- p.22: Disentangled / Interpretable / Actionable / Transferable 2x2 패널 (중요도 중)
- p.23: Types of Representation Learners 표 (중요도 상, 표로 재작성 가능)

## EXAM POINTS

- **좋은 representation의 조건** 3가지: Compress information, Support prediction, Generalize across tasks.
- **Compression의 정의와 이점**: 본질 정보만 보존; Less memory / Invariance(nuisance 제거) / Occam's Razor. "압축은 정보 손실이 아니라 무관한 정보 제거".
- **Prediction의 정의**: 관측되지 않은 world의 속성(future, past, missing data)을 추론하는 것. 각 유형별 예시(V-JEPA, MAE 등).
- **Good Representation = Compression + Prediction** 벤 다이어그램: 각 원에 속하는 방법(Autoencoder/Contrastive/Clustering vs Future prediction/Imputation/Pretext) 분류하기.
- **Representation Learner 6종 표**: 방법 - 원리 - 한 줄 요약을 매칭하는 문제.
- **추가 바람직한 성질 4가지**: Disentangled, Interpretable, Actionable, Transferable와 각각의 의미; 이들이 Compression+Prediction의 결과라는 점.
- Why 질문: task-specific model 대신 general-purpose representation을 쓰는 이유 (Learn once, reuse across tasks).
- Why 질문: 압축만 또는 예측만으로는 부족하고 둘을 결합해야 하는 이유.

## VERIFIER REPORT

- p.18 설명: Generalize 조건이 "p.21, p.22"에서 다시 나온다는 서술 정정 (p.21에만 해당; p.22는 Transferable 등 별도 성질).
- p.22 설명: "부수적으로 따라온다"를 슬라이드의 "often consequences"에 맞게 "종종"으로 완화 (과장 수정).
- p.22 설명: "결과로 나타난다" 단정을 "흔히 결과"로 완화하고 항상 보장되는 것은 아님을 명시.
- 누락 페이지/bullet/수식, 그림 설명, 인용(ResNet, ViT, CLIP, V-JEPA, MAE, Tishby, β-VAE, Locatello) 및 수식은 이상 없음.
