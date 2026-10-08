<!-- DRAFT | lecture: 05_Contrastive_Learning_III | chunk: 01 | pages 1–8 -->
<!-- VERIFIED | fixes: 2 | verifier: sonnet -->

## Advanced Representation Learning (고급표현학습) — 표지 (p.1)

- Advanced Representation Learning (고급표현학습)
- Instructor: Donghyeon Cho
- The Department of Computer Science, Hanyang University
- 그림: 한양대 로고, 한양대 인장(1939), 연구실 로고(카메라 모양 + V)

**설명**
- 강의 표지 슬라이드. 이번 강의는 Contrastive Learning 시리즈의 세 번째(III)로, memory bank / clustering / supervised dataset 활용 방식의 contrastive 계열 방법을 다룸.

## Contents of Table (p.2)

- 1. Learning from Memory Bank
  - MoCo
  - CURL
- 2. Clustering
  - DeepCluster
  - SwAV
- 3. Working with Supervised Datasets
  - CLIP
  - Supervised Contrastive Learning

**설명**
- 오늘 강의의 3개 큰 축 구성임.
  - **Memory Bank 계열**: 큰 negative 집합을 싸게 확보하는 방법 (MoCo, CURL).
  - **Clustering 계열**: instance 단위 대신 cluster(pseudo-label) 단위로 contrast (DeepCluster, SwAV).
  - **Supervised dataset 활용**: label/text 같은 supervision을 contrastive에 결합 (CLIP, SupCon).
- 세 축은 모두 "negative를 어떻게 구성하느냐 / positive를 무엇으로 정의하느냐"의 변주로 볼 수 있음.

> 보충: MoCo = He et al., CVPR 2020 (Momentum Contrast); CURL = Laskin et al., ICML 2020 (RL용 contrastive, MoCo식 momentum encoder 사용); DeepCluster = Caron et al., ECCV 2018; SwAV = Caron et al., NeurIPS 2020; CLIP = Radford et al., ICML 2021; SupCon = Khosla et al., NeurIPS 2020.

## Learning from Memory Bank — 섹션 구분 (p.3)

- 섹션 구분 슬라이드: "Learning from Memory Bank" (진한 파랑 배경, 흰 글씨)

## Memory Bank (p.4)

- **Large numbers of negative samples** improve contrastive learning.
- Computing embeddings for all negatives is computationally expensive.
- **Store** previously computed embeddings in memory.
- **Reuse** stored embeddings as negative samples during training.
- **Key Idea**: Trade stale representations for efficient large-scale contrastive learning.
- Mini-batch Negatives → Limited but Fresh
- Memory Bank Negatives → Large but Slightly Stale

**설명**
- Contrastive loss(InfoNCE 등)는 negative가 많을수록 성능이 좋아짐. negative가 많을수록 분모의 합이 전체 데이터 분포를 더 잘 근사하고, 더 어려운(hard) negative가 포함될 확률이 높아지기 때문임.
- 문제: SimCLR식 **mini-batch negative**는 batch size에 묶임. 큰 batch는 GPU 메모리 한계로 비쌈 (negative마다 forward + backward 필요).
- 해결: 과거 iteration에서 이미 계산한 embedding을 **memory에 저장**해 두고 negative로 **재사용**함. 이번 iteration에서 negative를 다시 forward할 필요가 없으므로 비용이 batch size와 분리(decouple)됨.
- 대가는 **staleness**: 저장된 embedding은 과거 encoder 파라미터로 만든 값이라 현재 encoder와 불일치함 → "Large but Slightly Stale".
- 정리
  - Mini-batch negatives: 수 적음, 항상 현재 encoder 기준(fresh)
  - Memory bank negatives: 수 많음, 과거 encoder 기준(stale)
- 이 trade-off가 이후 MoCo(momentum encoder + queue)로 이어지는 동기임.

## Instance Discrimination with Memory Bank (p.5)

- **Instance contrastive learning**
  - **Each** image is considered a **unique class**.
  - Positive pair: the same image.
  - Negative pairs: all other images.
  - → Unlike conventional classification, instance discrimination treats every image as a unique class and learns to distinguish it from all other images.
- 출처: Wu et al., CVPR 2018, Unsupervised Feature Learning via Non-Parametric Instance-level Discrimination
- 그림(파이프라인): 왼쪽에 표범 이미지 → 굵은 화살표 → **CNN backbone**(2048D 세로 직사각형) → 화살표 "low dim" → 128D 박스 → 화살표 "L2 norm" → 128D 박스 → **Non-param Softmax** 박스 → 오른쪽으로 부채꼴 화살표가 1-th image, 2-th image, ..., i-th image, ..., (n-1)-th image, n-th image 썸네일로 향함. 아래 중괄호가 backbone~L2 norm 전체를 묶어 $f_\theta(x)$로 표기.

**설명**
- **Instance discrimination**: 데이터셋의 N개 이미지 각각을 서로 다른 클래스로 보는 N-way 분류 문제로 pretext task를 정의함. label이 필요 없고 데이터 자체가 supervision이 됨.
- Positive는 "같은 이미지"(실제 구현에선 같은 이미지의 augmentation), negative는 나머지 모든 이미지.
- 일반 classification과의 차이: 클래스 수가 N(ImageNet이면 약 128만)이라 매우 많고, 클래스당 샘플이 1개뿐임.
- 그림 해석: 이미지 → $f_\theta$(CNN, 2048D 출력 → 128D 투영 → L2 normalize) → 단위 구면 위 128D 벡터 → 이를 i번째 이미지(클래스)에 대한 확률로 변환하는 non-parametric softmax. "Non-param"인 이유는 마지막 분류기의 weight가 학습 파라미터가 아니라 각 이미지의 feature 자체이기 때문임(p.7).

> 보충: 이 논문의 직관은 supervised network의 softmax 출력에서, 표범 사진에 대해 시각적으로 비슷한 클래스(예: jaguar, cheetah)에는 높은 확률이, 무관한 클래스에는 매우 낮은 확률이 나온다는 관찰임(label 없이도 클래스 간 유사도가 학습됨). 클래스를 instance까지 극단적으로 쪼개 학습하면 의미적 유사도를 배울 수 있다는 아이디어.

## Instance Discrimination with Memory Bank — NCE와 Memory Bank (p.6)

- **Instance contrastive learning**
  - **Unfeasible** to train a softmax layer with these many heads.
  - → Instead of calculating all classes, approximated by using **Noise Contrastive Estimation (NCE)**, which samples only some negatives.
  - **Memory bank** enables efficient large-scale instance discrimination.
    - Storing sample representation in the database from past iterations.
- 출처: Wu et al., CVPR 2018
- 그림: p.5와 동일한 파이프라인(CNN backbone 2048D → low dim 128D → L2 norm 128D → Non-param Softmax → 1-th ~ n-th image)에, 오른쪽 끝에 **Memory Bank** 세로 박스가 추가됨. 박스는 위에서부터 $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3, \dots, \mathbf{v}_{n-3}, \mathbf{v}_{n-2}, \mathbf{v}_{n-1}, \mathbf{v}_n$ 행으로 구성되고, 각 행이 Non-param Softmax의 각 이미지 출력과 대응됨.

**설명**
- N이 백만 단위일 때 N-way softmax는 (1) 파라미터 $N \times d$가 크고 (2) 정규화 상수(분모)를 매번 N개 전부로 계산해야 해서 비현실적임.
- 해결 두 가지.
  - **NCE**: N-way 분류를 "데이터 분포에서 온 샘플 vs noise 분포에서 온 샘플"의 **이진 분류**로 바꾸고, noise(negative)는 M개만 샘플링함 (p.8에서 구체화).
  - **Memory bank**: 각 이미지 i의 최신 feature $\mathbf{v}_i$를 bank에 저장하고, softmax의 "클래스 weight" 역할로 재사용함. 매 iteration 갱신되는 것은 batch에 포함된 이미지의 항목뿐임.
- 그림에서 memory bank의 각 행이 softmax 출력 각 노드와 대응되는 것이 핵심: bank = 분류기 weight 행렬 역할.

## Instance Discrimination with Memory Bank — Non-parametric Softmax (p.7)

- Non-parametric classifier predicts the probability of a sample $v$ belonging to class $i$:
  - embedding: $\mathbf{v} = f_\theta(x)$
  - Probability: $$P(C=i \mid \mathbf{v}) = \frac{\exp(\mathbf{v}_i^\top \mathbf{v}/\tau)}{\sum_{j=1}^{n}\exp(\mathbf{v}_j^\top \mathbf{v}/\tau)}$$
- **Memory Bank** for storing sample representation in the database from past iterations.
  - Memory bank: $V = \{\mathbf{v}_i\}$
- 출처: Wu et al., CVPR 2018

**설명**
- 기호
  - $x$: 입력 이미지, $f_\theta$: CNN encoder, $\mathbf{v}=f_\theta(x)$: L2-normalized 128D embedding ($\|\mathbf{v}\|=1$).
  - $\mathbf{v}_i$: memory bank에 저장된 i번째 이미지의 embedding (= i번 클래스의 prototype/weight 역할).
  - $\tau$: temperature. 작을수록 분포가 sharp해지고 hard negative에 더 큰 가중이 감.
  - $n$: 전체 이미지 수(= 클래스 수).
- 일반 parametric softmax는 $\exp(\mathbf{w}_i^\top \mathbf{v})$로 학습되는 weight $\mathbf{w}_i$를 쓰지만, 여기서는 $\mathbf{w}_i$를 **$\mathbf{v}_i$로 직접 대체**함 → 별도 파라미터가 없으므로 non-parametric.
- 장점
  - train/test에서 같은 metric(cosine similarity)을 씀 → test 시 kNN으로 바로 평가 가능.
  - L2 norm 덕분에 $\mathbf{v}_i^\top\mathbf{v}$는 cosine similarity가 되어 $[-1,1]$ 범위, $\tau$로 스케일 조절.
- 목적: 이미지 $x_i$의 embedding $\mathbf{v}$가 자기 항목 $\mathbf{v}_i$와는 가깝고, 다른 $\mathbf{v}_j$와는 멀게 학습됨. 식 구조는 InfoNCE의 softmax와 유사함(단, 여기서는 bank 항목 전체를 후보로 하는 N-way 분류).

> 보충: 논문에서는 bank 항목을 학습 중 현재 feature로 덮어쓰기(갱신)하고 다시 L2 normalize함(슬라이드에는 없음). 논문 기본값: $\tau=0.07$, 128D, $M=4096$ negatives, 가중 $\lambda=0.5$ (기억 기반, 확인 필요).

## Instance Discrimination with Memory Bank — Monte Carlo 근사와 NCE Loss (p.8)

- Access to the representations of all the samples
  - too expensive in practice → **Monte Carlo approximation** using a random subset of M
- $$P(i\mid\mathbf{v}) = \frac{\exp(\mathbf{v}^\top\mathbf{f}_i/\tau)}{\sum_{j=1}^{N}\exp(\mathbf{v}_j^\top\mathbf{f}_i/\tau)} \simeq \frac{\exp(\mathbf{v}^\top\mathbf{f}_i/\tau)}{\frac{N}{M}\sum_{k=1}^{M}\exp(\mathbf{v}_{j_k}^\top\mathbf{f}_i/\tau)}$$
- Introduced an extra term for positive samples in the loss function based on the **proximal optimization method**.
- $$\mathcal{L}_{\text{instance}} = -\mathbb{E}_{P_d}\Big[\log h(i,\mathbf{v}_i^{(t-1)}) - \lambda\|\mathbf{v}_i^{(t)}-\mathbf{v}_i^{(t-1)}\|_2^2\Big] - M\,\mathbb{E}_{P_n}\Big[\log\big(1-h(i,\mathbf{v}'^{(t-1)})\big)\Big]$$
- $$h(i,\mathbf{v}) = \frac{P(i\mid\mathbf{v})}{P(i\mid\mathbf{v})+M P_n(i)}\quad\text{where the noise distribution is uniform } P_n = 1/N$$
- 빨간 박스 주석 1 ($h(i,\mathbf{v})$ 정의에 연결): "Given feature v and ID i, what is the probability that it originated from the actual data source?"
- 빨간 박스 주석 2 ($\|\mathbf{v}_i^{(t)}-\mathbf{v}_i^{(t-1)}\|_2^2$ 항에 연결): "gradually vanish as the learned embedding converges"
- 출처: Wu et al., CVPR 2018
- ← 슬라이드 표기 주의: 첫 식에서 분자는 $\mathbf{v}^\top\mathbf{f}_i$, 분모 합은 $\mathbf{v}_j^\top\mathbf{f}_i$로, p.7의 $\mathbf{v}_i^\top\mathbf{v}$와 역할(bank 항목 vs 현재 feature)이 뒤바뀐 형태임. 논문 원문 표기 그대로이며 $\mathbf{f}_i$=이미지 i의 현재 feature, $\mathbf{v}$=bank 항목으로 읽어야 함. ← $\mathbf{v}$ 기호가 p.7(현재 embedding)과 p.8(bank 항목)에서 다른 의미로 쓰임.

**설명**
- **Monte Carlo 근사**: 분모 $Z=\sum_{j=1}^N \exp(\cdot)$ 는 N개 전부 필요함. 무작위 M개를 뽑아 평균을 내고 $N$배로 스케일하면 불편(unbiased) 추정량: $Z \approx \frac{N}{M}\sum_{k=1}^{M}\exp(\mathbf{v}_{j_k}^\top\mathbf{f}_i/\tau)$ ($j_k$ = k번째 샘플된 인덱스). 계산량이 $O(N)\to O(M)$.
- **NCE 해석**: 문제를 이진 분류로 바꿈. positive(데이터 분포 $P_d$에서 온 (feature, ID) 쌍) vs negative(noise 분포 $P_n$에서 온 쌍). noise는 uniform $P_n=1/N$, negative를 데이터보다 $M$배 더 많이 뽑는다고 가정.
- $h(i,\mathbf{v})=\dfrac{P(i\mid\mathbf{v})}{P(i\mid\mathbf{v})+MP_n(i)}$: feature $\mathbf{v}$와 ID $i$ 쌍이 **실제 데이터에서 왔을 사후확률**(posterior). 분자 = 데이터 모델 확률, 분모의 $MP_n(i)$ = noise가 그 쌍을 만들 확률(M배 가중).
- Loss의 두 항
  - 첫 항: positive에 대해 $\log h$를 크게 → 자기 ID와 feature가 맞다고 분류하도록.
  - 둘째 항 $-M\,\mathbb{E}_{P_n}[\log(1-h)]$: noise 샘플 $\mathbf{v}'$에 대해 $h$를 작게 → negative를 noise로 분류하도록.
- **Proximal regularization** $\lambda\|\mathbf{v}_i^{(t)}-\mathbf{v}_i^{(t-1)}\|_2^2$: (t)는 현재, (t-1)은 이전 iteration. 이전 값에서 크게 벗어나지 않게 하는 proximal term으로, 각 클래스(이미지)가 epoch당 한 번만 방문돼 학습이 불안정하고 진동하기 때문에 도입함. embedding이 수렴하면 $\mathbf{v}^{(t)}\approx\mathbf{v}^{(t-1)}$이므로 이 항은 **점차 0으로 사라짐**(빨간 박스 주석).
- 기호 정리: $P_d$ = 데이터 분포, $P_n$ = noise 분포, $\lambda$ = proximal 가중, $M$ = negative 수, $\mathbf{v}'$ = noise에서 샘플된 bank 항목.
- 부호: loss 전체에 마이너스가 붙어 최소화하므로 proximal 항은 결과적으로 loss에 $+\lambda\|\cdot\|^2$ 페널티로 작용함.

> 보충: 논문은 normalizing constant $Z$를 첫 배치에서 Monte Carlo로 추정해 상수로 고정하는 방식을 씀. 슬라이드의 $N/M$ 스케일이 그 근사에 해당함.

## FIGURES

- p.5: Instance Discrimination 파이프라인 (표범 이미지 → CNN 2048D → 128D → L2 norm → Non-param Softmax → 1~n번 이미지) — 상
- p.6: 같은 파이프라인 + 우측 Memory Bank($\mathbf{v}_1\dots\mathbf{v}_n$) 추가 버전 — 상
- p.4: Mini-batch negatives(Limited but Fresh) vs Memory bank(Large but Slightly Stale) 대비 문구 — 중 (그림은 아니나 표로 정리할 가치)
- p.8: 식 주석 박스(h의 의미, proximal 항 소멸) — 중

## EXAM POINTS

- **Memory bank의 동기와 trade-off**: mini-batch negative(적고 fresh) vs memory bank(많고 stale). 왜 많은 negative가 필요하며 왜 stale이 문제인지 서술.
- **Instance discrimination 정의**: 각 이미지 = 1 class, positive = 같은 이미지, negative = 나머지 모두. 일반 classification과의 차이.
- **Non-parametric softmax 식**: $P(C=i\mid\mathbf{v})=\exp(\mathbf{v}_i^\top\mathbf{v}/\tau)/\sum_j\exp(\mathbf{v}_j^\top\mathbf{v}/\tau)$ 의 각 기호($\tau$, $\mathbf{v}_i$, L2 norm)와 "non-parametric"인 이유(classifier weight를 memory bank feature로 대체).
- **왜 full softmax가 불가능한가 → NCE / Monte Carlo 근사**: 분모 N개 합 비용, M개 샘플 + $N/M$ 스케일로 근사하는 과정.
- **$h(i,\mathbf{v})$의 의미**: feature-ID 쌍이 실제 데이터 분포에서 왔을 확률(NCE 이진 분류 posterior), noise $P_n=1/N$.
- **Proximal regularization 항**: 왜 넣는가(epoch당 1회 방문으로 인한 불안정/진동 완화), 왜 수렴하면 사라지는가.
- **Memory bank의 Pros/Cons** (p.9, 다음 chunk): 많은 negative·낮은 비용 vs stale embedding → "큰 dictionary를 유지하면서 staleness를 줄일 수 있는가?"가 MoCo로 연결.

## VERIFIER REPORT

- p.5 보충: 표범 사진의 softmax 응답 관찰(비슷한 클래스 jaguar/cheetah에 높은 확률, 무관한 클래스엔 낮은 확률)을 부정확한 서술에서 수정.
- p.7 설명: "InfoNCE의 softmax와 동일" 과장 표현을 "유사"로 완화.
- 그 외 페이지 1–8의 수식, 기호, 그림 설명, 인용(연도/학회) 대조 결과 이상 없음.
