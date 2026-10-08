<!-- DRAFT | lecture: 05_Contrastive_Learning_III | chunk: 03 | pages 17–24 -->

## MoCo-V2: Ablation of MoCo baselines (p.17)

슬라이드 텍스트:
- 제목: MoCo-V2 / • **MoCo v2 (2020)**
- 출처: Chen et al., 2020, Improved Baselines with Momentum Contrastive Learning
- Key takeaways: Non-linear projection head and strong data augmentation are crucial for contrastive learning.

**표 (Table 1. Ablation of MoCo baselines)**, ResNet-50 기준 (i) ImageNet linear classification, (ii) VOC object detection fine-tuning (5회 평균). "MLP": MLP head 사용, "aug+": 추가 blur augmentation, "cos": cosine learning rate schedule.

| case | MLP | aug+ | cos | epochs | ImageNet acc. | AP50 | AP | AP75 |
|---|---|---|---|---|---|---|---|---|
| supervised (회색) | | | | | 76.5 | 81.3 | 53.5 | 58.8 |
| MoCo v1 | | | | 200 | 60.6 | 81.5 | 55.9 | 62.6 |
| (a) | ✓ | | | 200 | 66.2 | 82.0 | 56.4 | 62.6 |
| (b) | | ✓ | | 200 | 63.4 | 82.2 | 56.8 | 63.2 |
| (c) | ✓ | ✓ | | 200 | 67.3 | **82.5** | 57.2 | 63.9 |
| (d) | ✓ | ✓ | ✓ | 200 | 67.5 | 82.4 | 57.0 | 63.6 |
| (e) | ✓ | ✓ | ✓ | **800** | **71.1** | **82.5** | **57.4** | **64.0** |

- 그림 요소: MoCo v1 ~ (c) 행을 빨간 사각형으로 강조 (MLP, aug+ 효과를 보여주려는 부분). (e) 행과 일부 수치는 굵게 표시.

**설명**
- 이 표는 MoCo v1에 SimCLR의 두 가지 레시피(**MLP projection head**, **stronger augmentation(blur 추가)**)와 학습 스케줄 트릭(**cosine LR**)을 하나씩 얹었을 때의 **ablation**임.
- MLP head만 추가(a): 60.6 → 66.2 (**+5.6%p**), augmentation만 추가(b): 60.6 → 63.4 (+2.8%p), 둘 다(c): 67.3. 즉 **MLP head 효과가 augmentation보다 크고**, 둘은 대체로 가산적(additive)으로 작동함.
- cosine LR(d)은 +0.2%p로 미미, 학습을 **200 → 800 epochs**로 늘리면(e) 71.1까지 상승. 즉 contrastive 방법은 **오래 학습할수록 이득**이 큼.
- **ImageNet linear acc.는 크게 오르지만 VOC detection(transfer)은 상대적으로 변화폭이 작음** (AP50 81.5 → 82.5). 그래도 supervised(81.3 AP50, 53.5 AP)를 이미 넘어서는 점이 중요: unsupervised pretrain이 detection transfer에서는 supervised를 능가함.
- 왜 MLP head가 도움이 되나? head가 **contrastive loss에 특화된 정보(augmentation invariance)를 담는 용도**로 쓰이고, 그 앞단 representation(backbone 출력)은 downstream에 더 유용한 정보를 보존하게 됨 (SimCLR의 논지).
> 보충: MoCo v2는 SimCLR(Chen et al., 2020)의 MLP head(2-layer, 2048-d hidden, ReLU)와 augmentation(Gaussian blur, 강한 color distortion)을 MoCo v1(He et al., 2019/2020)에 그대로 이식한 **짧은 tech report** 형태의 논문임. 새 알고리즘이 아니라 **improved baseline** 제안이 목적.
> 보충: linear classification은 backbone을 freeze하고 그 위에 linear layer만 학습하는 평가 프로토콜.

---

## MoCo-V2: MoCo vs. SimCLR (p.18)

슬라이드 텍스트:
- 제목: MoCo-V2 / • **MoCo v2 (2020)**
- 출처: Chen et al., 2020
- Key takeaways:
  - Non-linear projection head and strong data augmentation are crucial for contrastive learning.
  - Decoupling mini-batch size with negative sample size allows MoCo-V2 to outperform SimCLR with smaller batch size (256 vs. 8192).

**표 (Table 2. MoCo vs. SimCLR)**: ImageNet linear classifier accuracy (ResNet-50, 1-crop 224×224), unsupervised pre-training feature 기반. "aug+" in SimCLR includes blur and stronger color distortion. SimCLR ablation은 [2]의 Fig. 9에서 가져옴.

| case | MLP | aug+ | cos | epochs | batch | ImageNet acc. |
|---|---|---|---|---|---|---|
| MoCo v1 [6] | | | | 200 | 256 | 60.6 |
| SimCLR [2] | ✓ | ✓ | ✓ | 200 | 256 | 61.9 |
| SimCLR [2] | ✓ | ✓ | ✓ | 200 | 8192 | 66.6 |
| **MoCo v2** | ✓ | ✓ | ✓ | 200 | 256 | **67.5** |
| *results of longer unsupervised training follow:* | | | | | | |
| SimCLR [2] | ✓ | ✓ | ✓ | 1000 | 4096 | 69.3 |
| **MoCo v2** | ✓ | ✓ | ✓ | 800 | 256 | **71.1** |

(MoCo v2 행은 회색 배경 + 굵은 글씨로 강조)

**설명**
- 동일한 200 epochs, 동일한 구성요소(MLP, aug+, cos)에서 **batch 256**으로 비교하면 SimCLR는 61.9인 반면 MoCo v2는 67.5 → 같은 batch 크기에서 **MoCo가 5.6%p 우세**. SimCLR는 batch를 8192로 키워야 66.6으로 올라가며, 이래도 MoCo v2(256)보다 낮음.
- 이유: SimCLR의 negative는 **같은 mini-batch 안의 다른 샘플들**(2N-2개)이라 **negative 수 = batch size에 종속**됨. 반면 MoCo는 **queue(dictionary)**에 과거 key를 쌓아두므로 **negative 수(예: 65536)가 batch size와 분리(decoupling)**됨 → 작은 batch로도 많은 negative 확보.
- 긴 학습에서도 MoCo v2(800 epochs, batch 256) 71.1 > SimCLR(1000 epochs, batch 4096) 69.3. **더 짧은 epoch, 훨씬 작은 batch로 더 좋은 성능**.
> 보충: queue 안의 key는 momentum encoder로 만들어 일관성(consistency)을 유지하고, 가장 오래된 batch를 dequeue하는 FIFO 구조임. 8192 batch는 TPU 수준의 자원을 필요로 하는 설정임.

---

## MoCo-V2: Memory and time cost (p.19)

슬라이드 텍스트:
- 제목: MoCo-V2 / • **MoCo v2 (2020)**
- 출처: Chen et al., 2020
- Key takeaways:
  - Non-linear projection head and strong data augmentation are crucial for contrastive learning.
  - Decoupling mini-batch size with negative sample size allows MoCo-V2 to outperform SimCLR with smaller batch size (256 vs. 8192).
  - All with much smaller memory footprint! ("end-to-end" means SimCLR here)

**표 (Table 3. Memory and time cost)**, 8 V100 16G GPUs, PyTorch 구현. †: based on our estimation.

| mechanism | batch | memory / GPU | time / 200-ep. |
|---|---|---|---|
| **MoCo** | 256 | **5.0G** | **53 hrs** |
| end-to-end | 256 | 7.4G | 65 hrs |
| end-to-end | 4096 | 93.0G† | n/a |

**설명**
- "end-to-end"는 슬라이드 설명대로 **SimCLR 방식**(query/key 모두 같은 encoder로 gradient를 흘리고 batch 내 샘플을 negative로 사용하는 방식)을 의미함.
- 같은 batch 256에서도 MoCo가 메모리 5.0G vs 7.4G, 시간 53h vs 65h로 더 효율적. 이유: MoCo의 key encoder는 **momentum update로 gradient가 필요 없고**(no backprop), queue의 negative는 이미 계산된 feature라 **추가 forward/backward가 불필요**함.
- end-to-end에서 batch를 4096으로 키우면 GPU당 약 **93GB(추정치)** 필요 → 16GB V100 8장으로는 현실적으로 불가능(시간은 n/a). 즉 **큰 batch 의존 방식은 하드웨어 진입장벽이 높고**, MoCo는 일반 8-GPU 머신에서 재현 가능함.
- 정리: MoCo v2는 **정확도(p.18) + 메모리/시간(p.19)** 양쪽 모두에서 SimCLR 대비 이득.

---

## CURL: 개요 (p.20)

슬라이드 텍스트:
- 제목: CURL
- • **CURL (2020)**
  - Applies contrastive learning to **reinforcement learning**.
  - Learns visual representations from **raw** observations.
  - Uses random crop **augmentation** to create positive pairs.
  - Employs a momentum encoder and contrastive loss, following **MoCo**.
- • **Key Idea**: Improve sample-efficient reinforcement learning through contrastive representation learning.
- 출처: Srinivas et al., 2020, CURL: Contrastive Unsupervised Representations for Reinforcement Learning

**설명**
- **CURL**은 MoCo 스타일의 contrastive learning을 **강화학습(RL)**의 pixel 기반 관측에 적용한 방법. 목적은 **sample efficiency** 향상 (적은 환경 상호작용으로 높은 성능).
- 문제 배경: pixel 입력 RL은 고차원 이미지에서 control에 필요한 feature를 reward 신호만으로 학습해야 해서 샘플이 매우 많이 필요함. contrastive auxiliary task로 **좋은 representation을 별도 신호로 학습**시켜 이를 완화.
- positive pair: 같은 observation에 **random crop**을 두 번 적용한 두 view (같은 원본 → positive, 다른 observation의 crop → negative). 이는 SimCLR/MoCo의 instance discrimination을 그대로 가져온 것.
- MoCo를 따르므로 **query encoder + momentum key encoder + InfoNCE loss** 구조 사용.
> 보충: CURL은 contrastive loss의 similarity로 **bilinear product** $q^\top W k$ (학습되는 $W$)를 사용하고, 입력은 연속된 frame을 stack한 형태이며 stack 전체에 같은 crop을 적용함 (논문 기준). 슬라이드에는 없는 내용.
> 보충: 논문에서는 DeepMind Control Suite와 Atari 벤치마크에서 평가했고, pixel 기반 방법이 state 기반 방법에 근접한 sample efficiency를 보였다고 주장함.

---

## CURL: 구조 (p.21)

슬라이드 텍스트:
- 제목: CURL / • **CURL (2020)**
- 그림 아래 bullet:
  - Sample observations from the replay buffer.
  - Generate two augmented views of each observation.
  - Encode them using the query encoder and momentum key encoder.
  - Train the encoder with contrastive learning.
  - Simultaneously optimize the RL policy/value network using the learned representation.
- 출처: Srinivas et al., 2020

**그림 서술**: 큰 파란 둥근 박스 안에 왼쪽부터 오른쪽으로 데이터 흐름이 그려짐.
- 왼쪽 위: **Replay buffer**(원통형 DB 아이콘) → 점선 화살표 → **Observation**(겹쳐진 걷는 로봇(walker) 이미지 프레임 stack).
- Observation에서 **Data Aug** 라벨의 세로 점선이 위/아래 두 갈래로 나뉨: 위쪽은 $o_q$(augmented view), 아래쪽은 $o_k$.
- $o_q$ → **Query Encoder**(흰/파란 conv 블록 3D 도형) → 출력 $q = f_{\theta_q}(o_q)$.
- $o_k$ → **Key Encoder**(회색 conv 블록) → 출력 $k = f_{\theta_k}(o_k)$.
- 오른쪽 두 박스: 위쪽 초록색 **Reinforcement Learning** 박스(q로부터 화살표 입력), 아래쪽 빨간색 **Contrastive Unsupervised Learning** 박스(q와 k가 함께 입력). 즉 query encoder의 출력 $q$는 RL 모듈과 contrastive 모듈 **양쪽**으로 보내지고, key $k$는 contrastive 모듈로만 감.

**설명**
- 흐름: replay buffer에서 observation을 샘플 → 서로 다른 random crop 2개 $o_q, o_k$ 생성 → $q = f_{\theta_q}(o_q)$, $k = f_{\theta_k}(o_k)$ → contrastive loss로 encoder 학습, 동시에 $q$를 RL(policy/value network) 입력으로 사용.
- key encoder $\theta_k$는 query encoder의 **momentum(EMA) 업데이트**: $\theta_k \leftarrow m\theta_k + (1-m)\theta_q$ (MoCo와 동일 아이디어, 슬라이드에는 수식 없음).
- **핵심 설계**: contrastive 학습은 RL 학습과 **동시에(jointly, auxiliary loss 형태)** 진행됨. 즉 pretrain 후 freeze가 아니라 RL 업데이트와 같이 encoder를 학습 → RL 데이터 분포 변화에 representation이 계속 적응.
- 이전(MoCo/SimCLR) 대비 차이: 이미지 분류용 pretraining이 아니라 **downstream이 RL 정책 학습**이며 positive의 source가 replay buffer라는 점.
> 보충: replay buffer는 off-policy RL(예: SAC, Rainbow DQN)에서 과거 경험 (state, action, reward, next state)을 저장해 재사용하는 메모리. CURL은 SAC/Rainbow 위에 얹힘.

---

## Clustering (p.22)

섹션 구분 슬라이드: "Clustering" (진한 남색 배경, 중앙 흰 글씨). 이후 clustering 기반 self-supervised 방법(DeepCluster 등) 소개.

---

## DeepCluster: 개념 (p.23)

슬라이드 텍스트:
- 제목: DeepCluster
- • Unlike contrastive learning, it does not create positive/negative pairs, but uses clustering results as pseudo labels.
- 출처: Caron et al., 2018, Deep Clustering for Unsupervised Learning of Visual Features

**그림 서술**: 왼쪽에서 오른쪽으로 흐르는 파이프라인.
- **Input**: 박스 안에 4장의 이미지(꽃, 오래된 자동차, 바다 위 새, 들판 동물).
- → 화살표 → **Convnet**: 점점 작아지는 conv 블록들(3D 직육면체) 박스.
- Convnet 출력에서 두 갈래 화살표: 위쪽은 **Classification**(막대 3개 bar chart: 파랑 낮음/초록 높음/노랑 중간)로 향하고, 아래쪽은 **Clustering**(파랑/초록/노랑 점들이 점선 경계로 나뉜 scatter)로 향함.
- Clustering → 위 방향 화살표 라벨 **Pseudo-labels** → Classification 박스. 즉 cluster 결과가 classification의 타깃이 됨.
- Classification에서 Convnet 쪽으로 회색 굵은 화살표 **Backprop** → classification loss의 gradient가 convnet으로 역전파.

**설명**
- DeepCluster는 positive/negative pair를 만들지 않고, convnet feature를 **k-means로 clustering**해서 얻은 cluster ID를 **pseudo label**로 삼아 일반 supervised classification처럼 학습함.
- **contrastive와의 차이**: contrastive는 instance 단위로 "같은 이미지 vs 다른 이미지"를 구분(instance discrimination)하는 반면, DeepCluster는 **cluster(그룹) 단위 의미적 유사성**을 활용. 따라서 비슷한 이미지(같은 클래스 등)가 같은 cluster에 묶여 같은 label을 받을 수 있어, contrastive의 **false negative 문제**를 구조적으로 피함.
- 학습 신호는 feature → cluster → label → feature 개선의 **self-reinforcing** 구조.
> 보충: Caron et al., 2018 (ECCV)의 DeepCluster는 AlexNet/VGG를 사용. 임의 초기화된 convnet도 어느 정도 유용한 구조적 prior를 가지고 있어 초기 clustering이 완전히 무의미하지는 않다는 관찰이 출발점임.

---

## DeepCluster: 알고리즘 (p.24)

슬라이드 텍스트:
- • **DeepCluster (2018)**
  - Cluster image features using the current representation.
  - Treat cluster assignments as pseudo labels.
  - Train the network to predict the assigned clusters.
  - Repeat clustering and representation learning iteratively.
- • **Key Idea**: Alternate between clustering and classification.
- Process
  - Extract image representations.
  - Cluster the representations using k-means.
  - Assign pseudo labels from cluster IDs.
  - Update the network using the pseudo labels.
  - Repeat until convergence.
- 출처: Caron et al., 2018

**설명**
- 알고리즘은 **EM과 유사한 교대 최적화**: (1) 현재 network로 전체 데이터셋 feature 추출 → (2) **k-means**로 cluster 할당 → (3) cluster ID를 pseudo label로 지정 → (4) classification loss(cross-entropy)로 network 업데이트 → 반복.
- 수식으로 쓰면(슬라이드엔 없음) 대략 $\min_{\theta, W, C}\ \frac{1}{N}\sum_n \ell\!\left(g_W(f_\theta(x_n)),\ y_n\right)$, 여기서 $y_n$은 $\min_{C,\{y_n\}} \sum_n \lVert f_\theta(x_n) - C y_n\rVert^2$ (k-means)로 정해지는 cluster 할당, $C$는 centroid 행렬.
- **epoch마다 clustering을 새로 수행**하므로 pseudo label이 학습이 진행되며 점차 의미 있게 개선됨 (representation 개선 → 더 좋은 cluster → 더 좋은 label).
> 보충: 실제 DeepCluster 논문에는 degenerate solution을 막기 위한 장치가 있음. (1) **빈 cluster**는 비어있지 않은 cluster centroid를 약간 perturb해 재할당, (2) 특정 cluster 쏠림을 피하려고 **cluster 크기에 대한 uniform sampling** 사용, (3) 마지막 classification layer는 cluster ID가 epoch마다 permutation되므로 **매 epoch 재초기화**. 또한 clustering 때문에 **전체 데이터셋 forward가 매 epoch 필요**하여 online 방식보다 비효율적임.

---

## FIGURES

- p.17: Table 1, MoCo v1 → (e) ablation 표(MLP/aug+/cos 효과), 중요도 상
- p.18: Table 2, MoCo v2 vs SimCLR (batch size 256 vs 8192 비교), 중요도 상
- p.19: Table 3, memory/time cost 비교, 중요도 중
- p.21: CURL 구조 다이어그램 (replay buffer → augmentation → query/key encoder → RL + contrastive 이중 경로), 중요도 상
- p.23: DeepCluster 파이프라인 다이어그램 (convnet → clustering → pseudo-labels → classification → backprop), 중요도 상

## EXAM POINTS

1. **MoCo v2가 MoCo v1에서 바꾼 것**: SimCLR의 **MLP projection head + 강한 augmentation(blur)** (+cosine LR) 도입, momentum encoder/queue는 유지. ablation 수치 (60.6 → 67.3(c) → 71.1(800 ep))와 어느 요소의 효과가 큰지(MLP > aug).
2. **Why MoCo v2 (batch 256)가 SimCLR (batch 8192)를 이김**: queue로 **negative 수를 batch size와 decoupling**. SimCLR는 negative가 batch에 종속되어 큰 batch(많은 메모리) 필요.
3. **메모리/시간 비교**: MoCo 5.0G/53h vs end-to-end(SimCLR) 7.4G/65h (batch 256), batch 4096이면 약 93G(추정). "end-to-end = SimCLR"임을 기억.
4. **CURL의 목적과 구조**: contrastive learning을 RL에 적용해 **sample efficiency** 향상. 같은 observation의 random crop 2개 = positive, query encoder + momentum key encoder + InfoNCE, **RL loss와 contrastive loss를 동시에 최적화**.
5. **CURL 그림 해석**: $q$는 RL과 contrastive 양쪽에 쓰이고 $k$는 contrastive에만 쓰임 (key encoder는 EMA, gradient 없음).
6. **DeepCluster 알고리즘 순서**: feature 추출 → k-means → pseudo label → classification으로 network 업데이트 → 반복 (**clustering과 classification의 교대**).
7. **Contrastive vs DeepCluster 비교**: pair(positive/negative) 대신 **cluster를 pseudo label**로 사용, instance 단위가 아닌 **group 단위** 학습, false negative 문제 회피 / 단점은 매 epoch clustering 비용과 degenerate solution(빈 cluster, 쏠림) 위험.
8. 인용 정보 암기: MoCo v2 (Chen et al., 2020), CURL (Srinivas et al., 2020), DeepCluster (Caron et al., 2018).
