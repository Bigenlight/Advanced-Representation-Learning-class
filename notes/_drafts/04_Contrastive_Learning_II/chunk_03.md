<!-- DRAFT | lecture: 04_Contrastive_Learning_II | chunk: 03 | pages 17–24 -->
<!-- VERIFIED | fixes: 1 | verifier: sonnet -->

## Augmentation Strategies — Population Based Augmentation (PBA) (p.17)

- Population Based Augmentation (PBA)
  - Evolves augmentation policies during training
  - Uses **population**-based training (PBT)
  - **Continuously updates** augmentation strategies
  - **Key Idea**: Learn augmentations **online** via evolutionary optimization.
  - In contrast to AutoAugment, PBA learns a schedule instead of a fixed policy
- 그림 (Ho et al.의 비교 도식), 위아래 두 블록:
  - (a) AutoAugment: "RNN controller"가 "sampled policy"를 내보냄(아래 화살표). 컨트롤러는 "trained with Proximal Policy Optimization over 15,000 child models" (컨트롤러 자신으로 돌아오는 곡선 화살표). sampled policy는 "augmentation policy f_{c1..c5}(x)" 박스(여러 장 겹친 sub-policy g^i_{op,mag,prob}(x))이고, 오른쪽 화살표로 "fixed augmentation policy f^i_{c1..c5}(x)" (sub-policy g^i(x) 포함)로 이어짐 = 탐색 후 고정된 정책.
  - (b) Population Based Augmentation: "augmentation policy f_p(x) where p_init = (op, mag=0, prob=0) x 30" 박스에서 화살표로 "single run of Population Based Training on 16 child models" 를 거쳐 "augmentation policy **schedule** f_{p0...pT}(x, t)" 출력 (시간 t에 따라 변하는 스케줄).
- 출처: Ho et al., ICML 2019, Population Based Augmentation: Efficient Learning of Augmentation Policy Schedules

**설명**
- 문제의식: AutoAugment는 RNN controller를 PPO(RL)로 학습하며 child model을 약 **15,000개** 학습시켜야 해서 탐색 비용이 매우 큼. 결과물도 학습 전체에 걸쳐 쓰는 **fixed policy**.
- PBA는 **Population Based Training (PBT)** 을 사용함: 여러 child model(여기서는 **16개**)을 병렬로 학습시키면서 주기적으로 성능 나쁜 모델이 좋은 모델의 가중치와 정책 파라미터를 복사(exploit)하고 정책 파라미터를 약간 변형(explore)함. 이 과정에서 정책이 학습 도중 계속 바뀌므로 결과가 **시간 t에 따른 schedule** `f_{p_0..p_T}(x, t)`가 됨.
- 초기값은 모든 연산의 magnitude=0, prob=0 (= augmentation 없음)에서 시작해 점점 강해지는 형태가 되는데, 이는 학습 초기에는 약한 augmentation, 후기에는 강한 augmentation이 유리하다는 직관과 맞음.
- 핵심 차이: AutoAugment = **offline search → fixed policy**, PBA = **online evolutionary optimization → schedule**. 연산량이 크게 줄어듦(RL 15,000 모델 vs PBT 16 모델 1회 실행).

> 보충: PBT는 Jaderberg et al., 2017 (DeepMind) 의 하이퍼파라미터 최적화 방법. PBA 논문은 CIFAR-10 등에서 AutoAugment와 비슷한 성능을 약 1000배 적은 연산으로 달성했다고 보고함 (논문 주장, 수치는 슬라이드에 없음).

## Augmentation Strategies — Unsupervised Data Augmentation (UDA) (p.18)

- Unsupervised Data Augmentation (UDA)
  - Applies augmentations to **unlabeled** data
  - Enforces prediction **consistency** between original and augmented images
  - Minimizes **KL** divergence between predictions
  - **Key Idea**: Learn from unlabeled data through consistency regularization.
- 그림 (Xie et al.의 UDA 구조도): 아래에서 위로 흐르는 두 갈래.
  - 왼쪽: "Labeled Data" → x, y* → 모델 M, p_θ(y|x) → "Supervised Cross-entropy Loss".
  - 오른쪽: "Unlabeled Data" → x → (a) 모델 M, 출력 p_θ̃(y|x) (θ̃ 는 고정/stop-gradient된 파라미터 사본) → "Unsupervised Consistency Loss"; (b) x → "Augmentations" 상자(Back translation, RandAugment, TF-IDF word replacement 가 점선으로 연결) → x̂ → 모델 M, p_θ(y|x̂) → "Unsupervised Consistency Loss".
  - 두 loss(Supervised Cross-entropy Loss, Unsupervised Consistency Loss)가 합쳐져 "Final Loss".
- 출처: Xie et al., NeurIPS 2020, Unsupervised Data Augmentation for Consistency Training

**설명**
- Semi-supervised 방법: labeled data에는 일반 cross-entropy, unlabeled data에는 원본 x와 augmented x̂의 예측 분포가 같아지도록 **consistency loss** 를 줌.
- 수식 (슬라이드에 없음, 보충): $\mathcal{L} = \mathbb{E}_{x\in L}[-\log p_\theta(y^*|x)] + \lambda\, \mathbb{E}_{x\in U}\, \mathbb{E}_{\hat x\sim q(\hat x|x)} \left[ D_{KL}\big(p_{\tilde\theta}(y|x)\,\|\,p_\theta(y|\hat x)\big)\right]$. 원본 x 쪽 예측은 gradient를 흘리지 않는 target 역할(그림에서 θ̃로 표시됨).
- 왜 효과적인가: 라벨 없이도 "입력에 작은 변화가 있어도 출력은 같아야 한다"는 **smoothness / cluster assumption**을 강제함. 이때 augmentation 품질이 중요해서 단순 noise 대신 RandAugment(이미지), back-translation(텍스트) 같은 **강한(advanced) augmentation**을 사용함이 UDA의 포인트.
- 앞선 AutoAugment/PBA가 "좋은 augmentation을 찾는 법"이라면 UDA는 "augmentation을 unlabeled data 활용에 쓰는 법"임.
- 슬라이드 본문에는 "original and augmented images"라 되어 있으나 그림에는 텍스트용 augmentation(Back translation, TF-IDF word replacement)도 있음 ← 이미지/텍스트 혼용 표현, 원문 유지.

## Image Mixture — Mixup (p.19)

- Mixup
  - It runs **global-level** mixture by creating a weighted pixel-wise combination of two existing images.
- 수식:
  $$\tilde{x} = \lambda x_i + (1-\lambda)x_j,$$
  $$\tilde{y} = \lambda y_i + (1-\lambda)y_j,$$
- 표 (a) Test errors for the CIFAR experiments (낮을수록 좋음):

| Dataset | Model | ERM | mixup |
|---|---|---|---|
| CIFAR-10 | PreAct ResNet-18 | 5.6 | **4.2** |
| CIFAR-10 | WideResNet-28-10 | 3.8 | **2.7** |
| CIFAR-10 | DenseNet-BC-190 | 3.7 | **2.7** |
| CIFAR-100 | PreAct ResNet-18 | 25.6 | **21.1** |
| CIFAR-100 | WideResNet-28-10 | 19.4 | **17.5** |
| CIFAR-100 | DenseNet-BC-190 | 19.0 | **16.8** |

- 그래프 (b) "CIFAR-10 Test Error": x축 epoch(0~200), y축 error(0~20). 파랑=DenseNet-190 baseline, 주황=DenseNet-190 mixup. epoch 100 근처 LR decay로 급락, 이후 mixup 곡선이 baseline보다 계속 낮음(약 2.7 vs 3.8 근처). 캡션: "Test error evolution for the best ERM and mixup models."
- 출처: Zhang et al., ICLR 2018, mixup: Beyond Empirical Risk Minimization

**설명**
- 기호: $x_i, x_j$ 는 두 입력 이미지, $y_i, y_j$ 는 one-hot label, $\lambda\in[0,1]$ 은 혼합 비율. 이미지와 **라벨을 같은 비율로** 선형 보간함.
- 보통 $\lambda \sim \mathrm{Beta}(\alpha,\alpha)$ 에서 샘플링함 (보충, 슬라이드에는 없음).
- 직관: 학습 데이터 사이의 **선형 동작(linear behavior)** 을 유도하는 regularizer이며, ERM(Empirical Risk Minimization)이 학습 샘플만 외우는 경향(memorization)을 줄이고 decision boundary를 부드럽게 함. 이는 Vicinal Risk Minimization 관점으로 해석됨 (보충).
- **global-level**: 이미지 전체의 모든 픽셀을 섞음 → 다음 슬라이드의 CutMix(region-level)와 대비.
- 결과: 모든 모델/데이터셋에서 ERM 대비 error 감소, CIFAR-100에서 개선폭이 더 큼.

## Image Mixture — CutMix (p.20)

- Cutmix
  - Cutmix does **region-level** mixture by generating a new example by combining a local region of one image with the rest of the other image.
- 수식: $I_{\text{cutmix}} \leftarrow \mathbf{M}_b \odot I_1 + (1-\mathbf{M}_b)\odot I_2$, where $\mathbf{M}_b \in \{0,1\}^I$
- 그림/표 (열: ResNet-50, Mixup [48], Cutout [3], CutMix):
  - 이미지 행: ResNet-50 = 코기 원본 이미지; Mixup = 코기와 고양이가 반투명하게 겹친 이미지; Cutout = 왼쪽 위 사각 영역이 검정으로 가려진 코기; CutMix = 왼쪽 위 영역은 고양이 패치, 나머지는 코기.
  - Label 행: Dog 1.0 / Dog 0.5, Cat 0.5 / Dog 1.0 / Dog 0.6, Cat 0.4.

| | ResNet-50 | Mixup | Cutout | CutMix |
|---|---|---|---|---|
| ImageNet Cls (%) | 76.3 (+0.0) | 77.4 (+1.1) | 77.1 (+0.8) | **78.6 (+2.3)** |
| ImageNet Loc (%) | 46.3 (+0.0) | 45.8 (-0.5) | 46.7 (+0.4) | **47.3 (+1.0)** |
| Pascal VOC Det (mAP) | 75.6 (+0.0) | 73.9 (-1.7) | 75.1 (-0.5) | **76.7 (+1.1)** |

- 출처: Yun et al., ICCV 2019, CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features

**설명**
- 기호: $\mathbf{M}_b\in\{0,1\}^{I}$ 는 이미지 크기 $I$(=W×H)의 **binary mask** (1이면 $I_1$ 픽셀, 0이면 $I_2$ 픽셀), $\odot$ 은 element-wise 곱.
- 라벨은 mixup처럼 **면적 비율**로 섞음 (예: 패치가 이미지의 40%이면 Cat 0.4, Dog 0.6) (슬라이드 그림에서는 0.6/0.4로 확인됨).
- 왜 설계했나: Mixup은 픽셀을 투명하게 섞어 **부자연스러운 이미지**가 되고 localization에 오히려 손해(Loc -0.5, Det -1.7). Cutout은 영역을 지워 정보를 버리는 낭비(검정 영역은 학습에 기여 못함). CutMix는 지운 영역을 **다른 이미지의 실제 패치로 채워** 정보 손실 없이 정규화 + localizable feature 학습.
- 결과: ImageNet classification +2.3, localization +1.0, VOC detection +1.1 로 세 가지 모두 개선되는 유일한 방법.

> 보충: 원 논문에서 패치 비율 $\lambda\sim\mathrm{Beta}(\alpha,\alpha)$, 박스 면적이 $(1-\lambda)$가 되도록 샘플링함.

## Image Mixture — MoCHi (Mixing of Contrastive Hard Negatives) (p.21)

- MoCHi (Mixing of Contrastive Hard Negatives)
  - Identify the **hardest negative samples** based on similarity to the query.
  - Generate **synthetic hard negatives** by interpolating hard negative embeddings.
  - Create even harder negatives by mixing the **query** with hard negatives.
  - Improve representation learning with more informative negative samples.
  - **Key Idea**: Generate harder negatives by mixing embeddings in the feature space.
- 그림: 2D 산점도(feature space). 범례: 회색 x = "NEGATIVES (MEMORY)", 파란 역삼각형 = "SYNTHETIC NEGATIVES", 빨간 사각형 = "QUERY". 회색 x들이 전체에 흩어져 있고, 오른쪽의 query(빨강) 주변에 가까운 회색 x들이 모여 있으며 그 주위에 파란 역삼각형(합성 negative)이 촘촘히 생성되어 있음. 왼쪽 아래에 "HARD NEGATIVE MIXING" 라벨 상자.
- 출처: Kalantidis et al., NeurIPS 2020, Hard Negative Mixing for Contrastive Learning

**설명**
- 배경: contrastive learning(MoCo 등)에서 negative가 많아도 대부분은 query와 멀어 **easy negative**라 gradient 기여가 작음. 성능은 **hard negative**(query와 유사한 negative)에 좌우됨.
- 절차: (1) memory bank/queue의 negative를 query와의 similarity로 정렬해 상위 $N$개를 hard negative로 선택, (2) 그중 두 개를 convex combination으로 섞어 합성 negative 생성 (feature-level mixup), (3) query와 hard negative를 섞은 더 어려운 합성 negative도 생성, 이후 L2 normalize.
- 보충 수식: $\tilde h = \dfrac{h'}{\|h'\|},\ h' = \alpha n_i + (1-\alpha) n_j$ (hard negative끼리, $\alpha\in(0,1)$), $\ \tilde s = \dfrac{s'}{\|s'\|},\ s'=\beta q + (1-\beta) n_j$ (query와 mix, $\beta<0.5$로 제한). 슬라이드에는 수식 없음.
- 핵심: 입력 이미지가 아닌 **embedding space에서** 섞기 때문에 추가 encoder forward 비용이 거의 없음. Mixup/CutMix가 pixel-space 혼합인 것과 대비.
- 이 슬라이드는 "Image Mixture" 섹션에 속하지만 실제로는 이미지가 아니라 feature 혼합임.

## Parallel Augmentation (p.22)

- (섹션 구분 슬라이드) Parallel Augmentation

## SimCLR (1) — 프레임워크와 augmentation, encoder (p.23)

- SimCLR is a simple framework for contrastive learning of visual representations.
  - maximizing agreement between differently augmented views of the same sample
  - a contrastive loss in the latent space
- 그림: 아래에서 위로 흐르는 대칭 구조. 맨 아래 원본 $x$ → 왼쪽 화살표(라벨 $t\sim\mathcal{T}$) → $\tilde x_i$, 오른쪽 화살표(라벨 $t'\sim\mathcal{T}$) → $\tilde x_j$. 각각 $f(\cdot)$ (같은 encoder) → $h_i$, $h_j$ ("← Representation →" 로 표시), 다시 $g(\cdot)$ → $z_i$, $z_j$. 맨 위에 $z_i \leftrightarrow z_j$ 양방향 화살표 "Maximize agreement".
- 오른쪽 설명:
  1. Randomly sample a minibatch of N samples and each sample is applied with two different data augmentation operations, resulting in 2N augmented samples in total.
     $$\tilde{\mathbf{x}}_i = t(\mathbf{x}),\quad \tilde{\mathbf{x}}_j = t'(\mathbf{x}),\quad t,t'\sim\mathcal{T}$$
  2. Given one positive pair, other 2(N-1) data points are treated as negative samples. Representation is produced by a base encoder $f()$.
     $$\mathbf{h}_i = f(\tilde{\mathbf{x}}_i),\quad \mathbf{h}_j = f(\tilde{\mathbf{x}}_j)$$
- 출처: Chen et al., ICML 2020, A Simple Framework for Contrastive Learning of Visual Representations

**설명**
- 기호: $\mathcal{T}$ = augmentation 함수들의 family(보충: random crop+resize, color distortion, Gaussian blur 등), $t,t'$ 는 거기서 독립 샘플링한 두 변환, $f$ = base encoder(보충: ResNet, 출력 $h$ = avgpool 이후 representation, downstream에 쓰이는 값).
- 한 이미지에서 만든 두 view $(\tilde x_i,\tilde x_j)$ 가 **positive pair**, 같은 배치의 나머지 $2(N-1)$ 개 view가 모두 **negative**. 별도 memory bank 없이 **in-batch negatives**만 사용 → 구조가 단순함(MoCo와의 차이점).
- 이 슬라이드의 제목이 "Parallel Augmentation" 섹션 아래에 있는 이유: 한 샘플에 두 augmentation을 **병렬로** 적용해 두 view를 얻기 때문으로 보임 (추정).
- "2N-2" 라고 쓰는 대신 "2(N-1)" 로 표기. 총 2N개 중 anchor 자신과 positive를 뺀 수와 일치하므로 오타 아님.

## SimCLR (2) — projection head와 NT-Xent loss (p.24)

- (p.23과 동일한 제목·bullet·그림)
- 오른쪽 설명:
  3. Loss operates on an extra projection layer of the representation $g()$. Loss is defined using cosine similarity **sim**().
     $$\mathbf{z}_i = g(\mathbf{h}_i),\quad \mathbf{z}_j = g(\mathbf{h}_j)$$
     $$\mathcal{L}_{\text{SimCLR}}^{(i,j)} = -\log\frac{\exp(\text{sim}(\mathbf{z}_i,\mathbf{z}_j)/\tau)}{\sum_{k=1}^{2N}\mathbb{1}_{[k\neq i]}\exp(\text{sim}(\mathbf{z}_i,\mathbf{z}_k)/\tau)}$$
  - SimCLR needs a **large batch size** to incorporate enough negative samples to achieve good performance.
- 출처: Chen et al., ICML 2020

**설명**
- 기호: $g(\cdot)$ = projection head(보충: MLP, 보통 hidden layer 1개+ReLU), $\tau$ = **temperature**, $\text{sim}(u,v)=\dfrac{u^\top v}{\|u\|\|v\|}$ = cosine similarity, $\mathbb{1}_{[k\neq i]}$ = $k=i$ 항(자기 자신)을 분모에서 제외하는 indicator.
- 구조: 분자는 positive pair의 유사도, 분모는 anchor $i$ 를 제외한 모든 $2N-1$ 개(positive 1 + negative $2N-2$)와의 유사도 합. 즉 **(2N-1)-way softmax cross-entropy에서 positive를 맞추는 문제** = InfoNCE / **NT-Xent** (normalized temperature-scaled cross entropy). 최종 loss는 $(i,j)$ 와 $(j,i)$ 모두, 즉 2N개 sample 전부를 anchor로 삼아 평균냄 (p.25 텍스트 요지).
- 왜 $g$ 를 추가하나: contrastive loss를 $h$ 에 직접 걸면 augmentation invariance 때문에 downstream에 유용한 정보(색, 방향 등)가 사라질 수 있어, loss는 $z$ 공간에서 계산하고 **downstream에는 $h$를 사용**(보충: SimCLR 논문의 ablation 결과).
- $\tau$ 가 작을수록 hard negative에 큰 가중을 줌 (보충).
- **large batch size** 필요: negative 수 = $2(N-1)$ 이 batch size에 비례하므로 큰 batch (논문은 4096~8192)와 많은 연산자원이 필요. 이 한계를 MoCo(queue), BYOL 등이 해결하려 함 (보충).

## FIGURES

- p.17: AutoAugment vs PBA 비교 도식 (RNN controller + PPO 15,000 child vs PBT 16 child → schedule), 중요도 중
- p.18: UDA 구조도 (supervised CE + unsupervised consistency, augmentation 종류), 중요도 중
- p.19: mixup 수식 + CIFAR 결과 표/test error 곡선, 중요도 중
- p.20: Mixup vs Cutout vs CutMix 비교 이미지 + 성능 표, 중요도 상
- p.21: MoCHi 2D 산점도 (query, memory negatives, synthetic negatives), 중요도 중
- p.23: SimCLR 프레임워크 도식 ($x\to t,t'\to f\to h\to g\to z$, maximize agreement), 중요도 상
- p.24: SimCLR loss (NT-Xent) 수식, 중요도 상

## EXAM POINTS

1. **AutoAugment vs PBA**: AutoAugment는 RL(PPO) controller + 약 15,000 child model로 **fixed policy**를 탐색, PBA는 PBT(16 child model, 1회 실행)로 **schedule**을 online 학습. 왜 PBA가 더 효율적인지 설명.
2. **UDA**: unlabeled data에 strong augmentation을 적용해 원본-증강 예측 간 **KL divergence consistency loss** 최소화. Supervised CE와 합산하는 final loss 구조 설명.
3. **Mixup 수식**: $\tilde x=\lambda x_i+(1-\lambda)x_j$, $\tilde y=\lambda y_i+(1-\lambda)y_j$. global-level, 라벨도 같이 섞음. 효과(ERM 대비 regularization).
4. **CutMix vs Mixup vs Cutout**: region-level mask $\mathbf{M}_b$ 로 패치 교체, 라벨은 면적 비율. 정보 손실(Cutout)과 부자연스러운 이미지(Mixup) 문제를 어떻게 해결하는지, 표의 Cls/Loc/Det 결과 비교.
5. **MoCHi**: hard negative 선택 → feature space에서 보간 → query와 mix. 왜 hard negative가 중요한가, pixel-space mixing과의 차이.
6. **SimCLR 파이프라인**: $x\to(t,t')\to f\to h\to g\to z$, positive pair/2(N-1) negatives 구성. 각 기호 역할.
7. **NT-Xent(InfoNCE) loss 수식**: 분모에서 $k\neq i$ indicator의 의미, $\tau$ 역할, cosine similarity, 모든 2N개를 anchor로 평균.
8. **Why 질문**: SimCLR이 왜 large batch size를 필요로 하는가(negative 수 = batch에 비례)? projection head $g$ 를 왜 쓰고 downstream에는 $h$를 쓰는가?

## VERIFIER REPORT

- p.21 MoCHi 보충 수식 표기 수정: 정규화 전 벡터를 h', s'(틸다 없음), 정규화 후를 h̃, s̃ 로 통일하고 α 범위 (0,1) 명시. 나머지 페이지(17-24)의 수치, 표, 그림 설명, 인용(저자/학회/연도)은 원본과 일치.
