<!-- DRAFT | lecture: 05_Contrastive_Learning_III | chunk: 06 | pages 41–48 -->
<!-- VERIFIED | fixes: 3 | verifier: sonnet -->

## Supervised Contrastive Learning — 실험 결과 (p.41)

- 제목: Supervised Contrastive Learning
- bullet: Supervised Contrastive Learning (SupCon)
- 표 (Table 2: ResNet-50 Top-1 classification accuracy, 여러 데이터셋):

| Dataset | SimCLR[3] | Cross-Entropy | Max-Margin [32] | SupCon |
|---|---|---|---|---|
| CIFAR10 | 93.6 | 95.0 | 92.4 | **96.0** |
| CIFAR100 | 70.7 | 75.3 | 70.5 | **76.5** |
| ImageNet | 70.2 | 78.2 | 78.0 | **78.7** |

- 캡션 (Table 2): Top-1 classification accuracy on ResNet-50 [17] for various datasets. cross-entropy training, unsupervised representation learning (SimCLR [3]), max-margin classifiers [32], SupCon(ours)를 비교함. 모든 baseline은 재구현 + hyperparameter tuning했고, margin classifier만 published result를 인용함. CIFAR-10/100 결과는 PyTorch 구현, ImageNet은 TensorFlow 구현에서 나온 값임.
- **Experimental Results**
  - SupCon consistently outperforms SimCLR and Cross-Entropy.
  - Improves classification accuracy across CIFAR-10, CIFAR-100, and ImageNet.
  - Learns more discriminative representations than conventional supervised learning.
- **Key Takeaway**: Contrastive objectives remain beneficial even with class labels.
- 출처: Khosla et al., NeurIPS 2020, Supervised Contrastive Learning

**설명**
- 이 표는 같은 backbone(ResNet-50)에서 **loss만 바꿔** 비교한 결과임. SimCLR는 label 없이 self-supervised로 학습한 representation(이후 linear eval 가정)이고, Cross-Entropy는 일반 supervised baseline, SupCon은 label을 이용해 같은 class 샘플을 전부 positive로 묶는 contrastive loss임.
- 수치 해석: CIFAR10에서 SupCon 96.0 vs CE 95.0 (+1.0), CIFAR100 76.5 vs 75.3 (+1.2), ImageNet 78.7 vs 78.2 (+0.5). 즉 **모든 데이터셋에서 1등이지만 ImageNet에서는 격차가 작음** (다음 슬라이드에서 "only by a small amount"라고 직접 언급).
- SimCLR가 CE보다 크게 낮은 것(ImageNet 70.2 vs 78.2)은 label 정보를 안 쓴 대가이고, SupCon은 label을 쓰면서도 contrastive 구조(embedding 공간에서 같은 class는 당기고 다른 class는 밀어냄)를 유지해 CE보다 더 **discriminative**한 representation을 얻는다는 주장임.
- Key Takeaway의 의미: "label이 있으면 contrastive는 필요 없다"가 아니라, label이 있어도 **contrastive objective가 CE보다 representation 품질을 개선**할 수 있다는 것.
- 비교 포인트: SimCLR는 positive가 같은 이미지의 두 augmentation(instance 단위)뿐이라 같은 class의 다른 이미지가 negative로 취급되는 **false negative** 문제가 있음. SupCon은 label로 이를 해소함.

> 보충: SupCon loss (Khosla et al., 2020)는 anchor $i$에 대해 같은 label의 모든 샘플 $P(i)$를 positive로 두는 $\mathcal{L}^{sup}=\sum_i \frac{-1}{|P(i)|}\sum_{p\in P(i)}\log\frac{\exp(z_i\cdot z_p/\tau)}{\sum_{a\ne i}\exp(z_i\cdot z_a/\tau)}$ 형태임. (이 chunk의 슬라이드에는 수식이 없고 앞 chunk 내용으로 추정됨.) Max-Margin [32]은 Elsayed et al., 2018 "Large margin deep networks for classification"으로 추정됨.

## Supervised Contrastive Learning — 추가 관찰 (p.42)

- 제목: Supervised Contrastive Learning
- Supervised Contrastive Learning (SupCon)
  - does outperform the base cross entropy, but only by a small amount.
  - outperforms the cross entropy on robustness benchmark (ImageNet-C, which applies common naturally occuring perturbations such as noise, blur and contrast changes to the ImageNet dataset). ← "occuring" 오타 (occurring), 원문 유지
  - is less sensitive to hyperparameter changes.
- 출처: Khosla et al., NeurIPS 2020, Supervised Contrastive Learning

**설명**
- 앞 슬라이드의 정확도 향상이 **작다**는 점을 솔직히 인정하는 슬라이드임. 그래서 SupCon의 진짜 장점은 정확도보다 다른 곳에 있다는 논지로 이어짐.
- **Robustness**: ImageNet-C는 ImageNet 검증셋에 noise, blur, contrast 변화 등 자연적으로 일어날 법한 corruption을 가한 benchmark임. SupCon이 CE보다 corruption 하에서 더 좋은 성능을 보인다는 것인데, 이는 representation이 low-level 변화에 더 robust할 가능성을 시사함 (슬라이드는 결과만 주장하고 원인은 명시하지 않음).
- **Hyperparameter sensitivity**: optimizer, augmentation, learning rate 등을 바꿔도 성능 변동이 CE보다 작음. 실무적으로 튜닝 부담이 줄어듦.
- 정리: SupCon = 소폭의 정확도 이득 + robustness + hyperparameter 안정성.

> 보충: Khosla et al.은 ImageNet-C에서 mean Corruption Error(mCE)로 CE 대비 개선을 보고했고, 다양한 optimizer/augmentation 조합에서 top-1 변동폭이 작다고 보였음 (논문 내용, 슬라이드에는 수치 없음).

## 섹션 구분: Can the memory bank provide more than negatives? (p.43)

- 섹션 구분 슬라이드 (진한 파랑 배경, 흰 글씨): "Can the memory bank provide more than negatives?"

**설명**
- memory bank를 지금까지 **negative 공급원**으로만 썼는데, positive도 얻을 수 있는지 묻는 질문. 다음 슬라이드들은 memory bank에서 **positive를 가져오는** NNCLR, All4One을 다룸.

## All4One 논문 (p.44)

- 논문 첫 페이지 캡처: "This ICCV paper is the Open Access version, provided by the Computer Vision Foundation. Except for this watermark, it is identical to the accepted version; the final published version of the proceedings is available on IEEE Xplore." (CVF 로고)
- 제목: **All4One: Symbiotic Neighbour Contrastive Learning via Self-Attention and Redundancy Reduction**
- 저자: Imanol G. Estepa (Universitat de Barcelona), Ignacio Sarasúa (NVIDIA), Bhalaji Nagarajan (Universitat de Barcelona), Petia Radeva (Universitat de Barcelona, Computer Vision Center, Cerdanyola)
- 출처 표기: Estepa et al., ICCV 2023, All4One: Symbiotic Neighbour Contrastive Learning via Self-Attention and Redundancy Reduction

**설명**
- 이 슬라이드는 논문 표지 소개용임. 제목의 키워드가 방법을 요약함: **Neighbour Contrastive Learning**(NNCLR 계열, memory bank의 neighbor 사용), **Self-Attention**(neighbor들을 Transformer로 요약해 centroid 생성), **Redundancy Reduction**(Barlow Twins류의 feature-level loss). "Symbiotic"은 세 loss가 서로 보완한다는 의미임.

> 보충: All4One은 NNCLR(Dwibedi et al., ICCV 2021, "With a Little Help from My Friends: Nearest-Neighbor Contrastive Learning of Visual Representations")와 Barlow Twins(Zbontar et al., ICML 2021)의 아이디어를 결합·확장한 방법임.

## Positive Samples from a Memory Bank (p.45)

- 제목: Positive Samples from a Memory Bank
- 그림 (좌우 두 패널, 가운데 회색 세로선으로 구분):
  - 왼쪽 "Nearest Neighbour CLR": 초록색 원 "NN"들이 흩어져 있고, 노란 원 $P^2$ (두 번째 view의 prediction)가 있음. 그중 가장 가까운 NN 하나와 $P^2$ 사이에 양방향 파란 화살표 "Pull". 나머지 NN은 사용 안 됨.
  - 오른쪽 "Centroid based CLR": 점선 세로선으로 좌우 분리. 왼쪽(초록) NN 여러 개에서 보라색 화살표가 모여 $C^1$로, 오른쪽(빨강) NN 여러 개와 $P^2$(노랑)에서 화살표가 모여 $C^2$로 향함. $C^1$과 $C^2$ 사이에 양방향 파란 화살표 "Pull".
- 우측 텍스트:
  - NNCLR
    - Store representations in a support set.
    - Retrieve the nearest neighbor.
    - Use one neighbor as the positive sample.
  - All4One
    - Retrieve K nearest neighbors.
    - Combine their information.
    - Compute the contrastive loss once.
- 하단 굵은 글씨: **NNCLR uses one neighbor. All4One summarizes multiple neighbors into one representation.**

**설명**
- **NNCLR**: 과거 batch의 representation을 **support set**(memory bank, queue 형태)에 저장해 두고, query(view 1의 embedding)와 가장 가까운 이웃 1개를 꺼내 그것을 positive로 사용함. 장점은 같은 이미지의 augmentation보다 **더 다양한(semantic하게 같은 class의 다른 이미지)** positive를 얻는 것.
- **한계**: nearest neighbor 1개는 noisy할 수 있음(다른 class일 수도 있음). 한 개에 의존하면 잘못 고른 경우 loss 신호가 크게 오염됨.
- **All4One**: K개의 neighbor를 모두 꺼내 하나의 표현(centroid)으로 **요약**한 뒤 contrastive loss를 한 번만 계산함. K개 정보가 합쳐지므로 개별 neighbor의 noise에 덜 민감하고, neighbor 간 context를 활용할 수 있음. K번 loss를 따로 계산하는 것이 아니라 **한 번의 비교**라는 점이 효율적임.
- 그림에서 오른쪽 패널의 centroid는 view 1 쪽 neighbor들 → $C^1$, view 2 쪽 neighbor들 → $C^2$로 각각 만든 뒤 둘을 pull함.

## All4One Overview (p.46)

- 제목: All4One Overview
- 다이어그램 (아래에서 위로 흐름):
  - 입력 $X^1$ → **encoder**(초록 박스) → $Z^1$ → **predictor**(주황 박스) → 두 갈래: $P^1_{nn}$ (Neighbour Contrast로), $P^1_{cc}$ (위쪽 KNN으로).
  - 입력 $X^2$ → **momentum encoder**(파란 박스) → $Z^2$. encoder에서 momentum encoder로 점선 화살표 "EMA".
  - $Z^2$ → KNN(보라 원, support set 조회) → $nn_1$ → Neighbour Contrast 쪽으로(초록 화살표, $P^1_{nn}$과 비교). 동시에 KNN에서 $nn_1 \ldots nn_k$가 위로 나가 transformer encoder로 입력됨.
  - $P^1_{cc}$ → KNN(분홍 원) → transformer encoder(보라 박스)로 입력, $nn_1\, nn_2 \ldots nn_k$ (view 2 쪽)도 transformer encoder로 입력 → 출력 $C^1$, $C^2$ → "Centroid Contrast".
  - $Z^1$과 $Z^2$ 사이 빨간 화살표: "Feature Contrast".
- 우측 텍스트:
  - Neighbor Contrast: One nearest neighbor / Direct positive alignment
  - Centroid Contrast: Multiple neighbors / Contextual neighbor representation
  - Feature Contrast: Original augmented features / Redundancy reduction
- 수식: $L_{All4One} = \sigma L_{NNCLR} + \kappa L_{Centroid} + \eta L_{Red}$
- 하단 굵은 글씨: **All4One combines three losses defined on three different representations.**

**설명**
- 구조는 BYOL/NNCLR 계열의 **비대칭 online–momentum** 구조임: online branch(encoder + predictor)가 view 1, momentum encoder(EMA로 업데이트, gradient 없음)가 view 2를 처리함.
- 세 loss가 **서로 다른 representation**에 정의됨:
  1. **Neighbour Contrast** ($L_{NNCLR}$): predictor 출력 $P^1_{nn}$ vs 1-NN $nn_1$ — 직접적인 positive alignment.
  2. **Centroid Contrast** ($L_{Centroid}$): K개 neighbor를 Transformer로 요약한 $C^1$, $C^2$ 간 contrast — contextual한 neighbor 정보.
  3. **Feature Contrast** ($L_{Red}$): 원래 augmentation된 feature $Z^1$, $Z^2$ 간 redundancy reduction (Barlow Twins식, feature 차원 간 중복 제거) — collapse 방지와 feature 다양성.
- $\sigma, \kappa, \eta$는 각 loss의 가중치 hyperparameter임.
- 설계 이유: neighbor 기반 loss만 쓰면 neighbor가 틀렸을 때 representation이 흔들리고 dimensional collapse 위험도 있음. feature-level loss로 원본 view 정보를 직접 정렬하고 차원 중복을 줄여 보완함 ("symbiotic").
- 슬라이드 그림에서 $P^1_{cc}$의 KNN이 view 1 쪽 neighbor를, momentum 쪽 KNN이 view 2 쪽 neighbor를 공급한다고 해석됨 (그림상 두 갈래의 입력이 transformer encoder로 들어감). ← 그림의 화살표만으로는 어느 KNN이 $C^1$/$C^2$에 각각 대응되는지 모호함

> 보충: 논문에서 $L_{Red}$는 Barlow Twins의 cross-correlation matrix를 identity에 가깝게 만드는 loss(대각 → 1, 비대각 → 0)임. Support set은 NNCLR처럼 FIFO queue로 유지됨.

## Neighbor Contrast (p.47)

- 제목: Neighbor Contrast
- 다이어그램: p.46과 동일, 단 predictor → $P^1_{nn}$ → Neighbour Contrast ← $nn_1$ ← KNN 구간이 빨간 박스로 강조됨. 우상단에 NNCLR 그림(NN들 중 하나와 $P^2$ 사이 Pull) 축소 표시.
- 수식: $L_i^{NNCLR} = -\log\left(\dfrac{\exp(nn_i^1 \cdot p_i^2/\tau)}{\sum_{k=1}^{N}\exp(nn_i^1\cdot p_k^2/\tau)}\right)$
- 수식 주석 (화살표):
  - $nn_i^1$: nearest neighbor found in the support set
  - $p_i^2$: online prediction from the second view
  - $p_k^2$: another prediction in the batch
- 설명 텍스트: NNCLR does not directly compare two views generated from the same image
  - instead, it retrieves the representation closest to the first view from a support set
  - uses that neighbor as the positive sample for the second view.
- 하단 굵은 글씨: **The nearest neighbor replaces one augmented view in the positive pair.**

**설명**
- 수식은 **InfoNCE**임. 분자: sample $i$의 nearest neighbor $nn_i^1$과 같은 sample의 두 번째 view prediction $p_i^2$의 유사도(내적 / temperature $\tau$). 분모: batch 내 다른 sample $k$의 prediction $p_k^2$와의 유사도 합 → $p_k^2$ ($k\neq i$)가 **negative**.
- 핵심 변화: 보통 positive pair는 (view 1, view 2)인데, 여기서는 **view 1 대신 view 1과 가장 가까운 support set 내 embedding($nn^1_i$)**을 씀. 즉 "nearest neighbor가 positive pair의 한쪽 augmented view를 대체함".
- 효과: 같은 이미지 augmentation만 positive로 쓸 때보다 **intra-class 변동(다른 instance, 다른 pose)**을 positive로 학습하게 되어 더 풍부한 invariance를 얻음.
- 변수 표기: 슬라이드 주석은 $p_i^2$를 "online prediction from the second view"라 하지만, 그림(p.46)에서 predictor는 view 1 branch($Z^1$ → $P^1_{nn}$)에 붙어 있음. ← 수식의 $p^2$ (view 2 prediction)와 그림의 $P^1_{nn}$ (view 1 prediction) 표기가 불일치. NNCLR 원 논문은 대칭 loss라 두 view를 번갈아 쓰기 때문으로 추정됨. 원문 유지.
- 수식의 $N$은 batch size(분모 합의 범위)로 이해됨.

> 보충: NNCLR 원논문(Dwibedi et al., 2021)의 loss는 $-\log\frac{\exp(NN(z_i,Q)\cdot z_i^+/\tau)}{\sum_k \exp(NN(z_i,Q)\cdot z_k^+/\tau)}$ 이고 $Q$가 support set임. NN 연산은 non-differentiable하므로 gradient는 $z_i^+$ 쪽으로만 흐름.

## Centroid Contrast (p.48)

- 제목: Centroid Contrast
- 다이어그램: p.46과 동일, 단 상단 Centroid Contrast 영역(KNN → transformer encoder → $C^1$, $C^2$)이 빨간 박스로 강조됨. 우상단에 "Centroid based CLR" 그림(p.45 오른쪽 패널) 축소, 캡션 "K neighbors, one contrastive comparison".
- 수식: $L_i^{centroid} = -\log\left(\dfrac{\exp(c_i^1\cdot c_i^2/\tau)}{\sum_{n=1}^{n}\exp(c_i^1\cdot c_n^2/\tau)}\right)$ ← 분모 합의 상한이 소문자 $n$, 인덱스도 $n$으로 겹쳐 쓰여 있음 (슬라이드 원문 유지, 보통 $\sum_{n=1}^{N}$). 분자는 $c_i^1\cdot c_i^2$, 분모는 $c_i^1\cdot c_n^2$ 형태임(슬라이드 이미지에서 확인됨).
- 설명 텍스트:
  - For each query, K neighbors are retrieved.
    - When the sequence of neighbors is fed into a small Transformer encoder, information from each neighbor is integrated via self-attention.
    - The first output representation from the Transformer is used as the centroid.
  - Subsequently, the centroids generated from the two views are treated as a positive pair to compute the InfoNCE loss.
- 하단 굵은 글씨: **The centroid is produced by self-attention. It is not an average or a cluster center.**

**설명**
- 과정: (1) query마다 support set에서 K개 neighbor 검색 → (2) 그 K개 neighbor 시퀀스를 (그림상 $P^1_{cc}$ → KNN 경로와 momentum 쪽 $nn_1\ldots nn_k$ 경로가 모두 입력으로 들어감) 작은 **Transformer encoder**에 넣음 → (3) self-attention으로 neighbor 간 정보가 섞임 → (4) **첫 번째 출력 토큰**(ViT의 CLS token 역할)을 **centroid** $c$로 사용.
- 두 view에서 각각 $c^1$, $c^2$를 만들고, 같은 sample의 $(c_i^1, c_i^2)$를 positive, batch 내 다른 sample의 centroid $c_n^2$를 negative로 하는 **InfoNCE**를 계산함 ("K neighbors, one contrastive comparison").
- 강조점: centroid는 **단순 평균이 아니고 clustering의 cluster center도 아님**. 학습되는 attention으로 neighbor 별 중요도를 가중해 만든 요약이므로, 틀린 neighbor는 낮은 weight로 억제하고 관련 있는 neighbor 정보를 문맥적으로 결합할 수 있음. 이름만 centroid이고 실제로는 **learned aggregation**임.
- NNCLR 대비 차이: 1개 neighbor → K개 neighbor의 context 활용, 개별 neighbor noise에 robust.

> 보충: Transformer 입력은 neighbor 임베딩 시퀀스에 query embedding(첫 토큰)을 둔 구성으로 알려져 있음 (논문 세부 사항, 슬라이드에는 명시 안 됨).

## FIGURES

- p.41: SupCon vs SimCLR/CE/Max-Margin accuracy 표 (CIFAR10/100, ImageNet) — 중요도 중
- p.45: Nearest Neighbour CLR vs Centroid based CLR 개념도 (1 neighbor pull vs K neighbors → centroid pull) — 중요도 상
- p.46: All4One 전체 구조도 (encoder, predictor, momentum encoder EMA, KNN, transformer encoder, 세 contrast 경로) + $L_{All4One}$ 수식 — 중요도 상
- p.47: Neighbor Contrast 강조 구조도 + NNCLR 수식 주석 — 중요도 중
- p.48: Centroid Contrast 강조 구조도 + centroid InfoNCE 수식 — 중요도 중

## EXAM POINTS

1. SupCon이 CE보다 좋은 점: 정확도 향상은 작지만(CIFAR10 96.0 vs 95.0, ImageNet 78.7 vs 78.2), **ImageNet-C robustness 우수, hyperparameter에 덜 민감**. (서술형 why)
2. SimCLR vs SupCon: label 사용 여부, 같은 class 샘플을 positive로 취급해 false negative 해소. Key takeaway: label이 있어도 contrastive objective가 유익함.
3. **NNCLR vs All4One 비교**: 1-NN을 positive로 사용(직접 alignment) vs K-NN을 self-attention으로 centroid로 요약해 loss 1회 계산.
4. NNCLR loss 수식 해석: 분자 $nn_i^1\cdot p_i^2/\tau$, 분모 batch 내 다른 $p_k^2$가 negative. "nearest neighbor가 positive pair의 한쪽 augmented view를 대체"한다는 의미.
5. **All4One 총 loss**: $L = \sigma L_{NNCLR} + \kappa L_{Centroid} + \eta L_{Red}$, 세 loss가 각각 neighbor / centroid / feature representation에 정의됨. 각각의 역할(direct alignment / contextual neighbor / redundancy reduction)을 설명할 수 있어야 함.
6. **Centroid 생성 방식**: 평균·cluster center가 아니라 Transformer encoder의 self-attention 출력 중 첫 번째 토큰. 왜 단순 평균보다 나은가(neighbor 중요도 학습, noise 억제).
7. All4One 구조: online(encoder+predictor) vs momentum encoder(EMA), support set에서 KNN 검색.
8. "memory bank가 negative 말고 positive도 제공할 수 있는가?" — NNCLR/All4One의 동기.

## VERIFIER REPORT

- p.42 설명: ImageNet-C 결과를 "representation이 low-level 변화에 더 불변"이라 단정한 과장을 완화 (슬라이드는 결과만 제시).
- p.48 설명: Transformer 입력에 "query prediction 자신 P^1_cc"를 K개와 함께 시퀀스로 넣는다는 슬라이드에 없는 단정을 그림 기준 서술(두 KNN 경로가 입력)로 수정.
- p.48 수식 주석: 분모 $c_i^1\cdot c_n^2$를 "불확실"로 적었으나 이미지상 확인되므로 확정 표기로 수정.
- 표(p.41), 수식(p.46/47), 인용(Khosla 2020, Estepa ICCV 2023, Dwibedi 2021, Zbontar 2021), 그림 설명은 원본과 일치.
