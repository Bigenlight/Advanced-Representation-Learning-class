<!-- DRAFT | lecture: 05_Contrastive_Learning_III | chunk: 02 | pages 9–16 -->
<!-- VERIFIED | fixes: 5 | verifier: sonnet -->

## Instance Discrimination with Memory Bank (p.9)

- Memory Bank: Pros and Cons
- Pros
  - Large number of negative samples
  - Low computational cost
- Cons
  - Stored embeddings become stale
  - Current queries are compared with outdated features
- Question: Can we keep a large dictionary while reducing feature staleness?
- 출처 표기: Wu et al., CVPR 2018, Unsupervised Feature Learning via Non-Parametric Instance-level Discrimination

**설명**
- **Memory bank**는 데이터셋 전체 샘플의 embedding을 저장해 두고 negative를 여기서 뽑아 쓰는 방식임. 매 step마다 모든 샘플을 encoder에 다시 통과시킬 필요가 없어서 **계산 비용이 낮고**, mini-batch 크기와 무관하게 **negative 수를 크게** 가져갈 수 있음 (SimCLR는 batch 크기가 곧 negative 수라 큰 batch가 필요한 것과 대비됨).
- 단점은 **staleness**임. bank의 항목 $v_i$는 해당 샘플이 마지막으로 sampling된 시점(보통 한 epoch 전)의 encoder로 계산된 값이라, 현재 encoder가 만든 query $q$와 **서로 다른 시점의 network로 만든 feature**를 비교하게 됨. 학습 초반처럼 encoder가 빨리 변할수록 불일치가 커짐.
- 그래서 던지는 질문이 "큰 dictionary는 유지하되 key의 일관성(consistency)을 높일 수 있나?"이고, 이에 대한 답이 다음 슬라이드의 **MoCo**임.

> 보충: Wu et al. (2018)은 해당 mini-batch 샘플의 bank 항목을 현재 feature로 덮어쓰고, 학습 안정화를 위해 proximal regularization을 사용함(항목 갱신 자체는 샘플이 뽑힐 때만 일어남). 근본적으로 bank 항목이 "과거 encoder 출력"이라는 점은 그대로임.

## MoCo (p.10)

- **Momentum Contrast (MoCo)**
  - Maintains a dynamic dictionary of negative keys.
  - Stores recent representations in a **FIFO queue.**
  - Encodes keys using a **momentum encoder.**
  - Enables a large and consistent dictionary for contrastive learning.
- **Key Idea**: Replace the memory bank with a momentum-updated queue.
- 출처 표기: He et al., CVPR 2020, Momentum Contrast for Unsupervised Visual Representation Learning

**설명**
- MoCo (He et al., 2020)는 contrastive learning을 **dictionary look-up** 문제로 봄: query $q$가 dictionary 안의 key들 중 자기 positive key $k^+$와 매칭되도록 학습.
- 좋은 dictionary의 조건은 **(1) 크기가 클 것, (2) key들이 일관된 encoder로 만들어졌을 것**임. memory bank는 (1)은 만족하나 (2)가 깨지고, end-to-end(SimCLR 식)는 (2)는 만족하나 (1)이 batch 크기에 묶임.
- 해결: **FIFO queue**에 최근 mini-batch의 key를 쌓고(가장 오래된 batch는 dequeue), key는 **momentum encoder**로 생성함. queue 안의 key들은 서로 비슷한 시점의 encoder(천천히 변함)로 만들어졌기 때문에 일관성이 유지됨. 또 오래된 key를 자연스럽게 버리므로 stale한 항목이 쌓이지 않음.

## MoCo (p.11)

- **Momentum Contrast (MoCo)**
- 그림: 아래에서 위로 흐르는 두 갈래 구조. 왼쪽 가지는 $x^{\text{query}} \to$ encoder(초록 박스) $\to q$. 오른쪽 가지는 $x^{\text{key}}_0, x^{\text{key}}_1, x^{\text{key}}_2, \dots \to$ momentum encoder(파란 박스) $\to$ queue $\to k_0, k_1, k_2, \dots$. 두 가지가 위에서 **similarity**로 합쳐지고 최종적으로 **contrastive loss**로 이어짐 (화살표는 $q$와 queue의 key들에서 similarity 노드로 향함).
- Key differences to SimCLR:
  - Keep a running **queue** of keys (negative samples).
  - Compute gradients and update the encoder **only through the queries.**
  - Decouple min-batch size with the number of keys: can support **a large number of negative samples.** ← "min-batch"는 mini-batch의 오타로 보임
  - The key encoder is **slowly progressing** through the momentum update rules:

$$\theta_k \leftarrow m\theta_k + (1-m)\theta_q$$

$$\mathcal{L}_{\text{MoCo}} = -\log \frac{\exp(\mathbf{q}\cdot\mathbf{k}^+/\tau)}{\sum_{i=1}^{N}\exp(\mathbf{q}\cdot\mathbf{k}_i/\tau)}$$

- "k+ is differently augmented query" ← 정확히는 같은 이미지를 다른 augmentation으로 변환해 key encoder에 통과시킨 key임 (query 자체가 아님). 또 분모 합 $i=1..N$ 에 $k^+$ 가 포함되어야 함(논문은 $K+1$개 key, $k_0=k^+$).
- 출처 표기: He et al., CVPR 2020

**설명**
- **기호**: $q = f_q(x^q)$ 는 query encoder 출력, $k = f_k(x^k)$ 는 key encoder 출력, $k^+$ 는 $q$와 같은 이미지에서 나온 positive key, $k_i$ 는 queue의 key들(negative), $\tau$ 는 temperature, $\theta_q,\theta_k$ 는 각 encoder의 파라미터, $m\in[0,1)$ 은 momentum 계수.
- **Loss**: **InfoNCE**로, positive 1개 + negative 여럿에 대한 $(1+K)$-way softmax cross-entropy(정답=positive)임. $\mathbf{q}\cdot\mathbf{k}$는 L2 normalize된 벡터면 cosine similarity.
- **Momentum update**: $\theta_k$ 는 backprop으로 학습하지 않고 $\theta_q$ 의 **exponential moving average (EMA)** 로만 갱신됨. $m$이 1에 가까울수록(논문 기본 $m=0.999$) key encoder가 천천히 변해 queue 안 key들이 일관됨.
- **Gradient 경로**: key 쪽은 `detach`(stop-gradient)되어 gradient는 query encoder로만 흐름. 그래서 queue의 수만 개 key에 대해 backprop 할 필요가 없음 → 메모리·연산 절약, **dictionary 크기와 batch 크기를 분리(decouple)** 할 수 있음.
- **SimCLR와 차이**: SimCLR는 같은 batch 내 다른 샘플을 negative로 쓰고 양쪽 view 모두 gradient를 받음 → negative 수가 batch 크기에 묶여 큰 batch가 필요함. MoCo는 queue로 이를 우회함.

> 보충: 논문 ablation에 따르면 $m=0$(momentum 없음)에서는 학습이 수렴하지 못하고, $m$이 작으면(0.9) 성능이 떨어지며, 0.99~0.999 근처에서 잘 됨. 즉 **느린 진화가 핵심**임. 정확한 수치는 원 논문 확인 필요.

## MoCo (p.12)

- Codes. (Algorithm 1: Pseudocode of MoCo in a PyTorch-like style)
- 의사코드 (슬라이드에 그림으로 제시):

```
# f_q, f_k: encoder networks for query and key
# queue: dictionary as a queue of K keys (CxK)
# m: momentum
# t: temperature

f_k.params = f_q.params # initialize
for x in loader: # load a minibatch x with N samples
    x_q = aug(x) # a randomly augmented version
    x_k = aug(x) # another randomly augmented version

    q = f_q.forward(x_q) # queries: NxC
    k = f_k.forward(x_k) # keys: NxC
    k = k.detach() # no gradient to keys

    # positive logits: Nx1
    l_pos = bmm(q.view(N,1,C), k.view(N,C,1))

    # negative logits: NxK
    l_neg = mm(q.view(N,C), queue.view(C,K))

    # logits: Nx(1+K)
    logits = cat([l_pos, l_neg], dim=1)

    # contrastive loss, Eqn.(1)
    labels = zeros(N) # positives are the 0-th
    loss = CrossEntropyLoss(logits/t, labels)

    # SGD update: query network
    loss.backward()
    update(f_q.params)

    # momentum update: key network
    f_k.params = m*f_k.params+(1-m)*f_q.params

    # update dictionary
    enqueue(queue, k) # enqueue the current minibatch
    dequeue(queue) # dequeue the earliest minibatch
# bmm: batch matrix multiplication; mm: matrix multiplication; cat: concatenation.
```

- 빨간 박스 주석(슬라이드 라벨):
  - "Generate a positive pair by sampling data augmentation functions" → `q = f_q.forward(x_q)`, `k = f_k.forward(x_k)` 두 줄(박스 대상; 바로 위 `x_q = aug(x)`, `x_k = aug(x)`가 실제 augmentation 샘플링)
  - "No gradient through the key" → `k = k.detach()`
  - "Use the running queue of keys as the negative samples" → `l_neg = mm(q, queue)` 구간
  - "InfoNCE loss" → `logits`, `labels`, `CrossEntropyLoss` 구간
  - "Update f_k through momentum" → `f_k.params = m*f_k.params+(1-m)*f_q.params`
  - "Update the FIFO negative sample queue" → `enqueue / dequeue`
- 출처 표기: He et al., CVPR 2020

**설명**
- 한 step 흐름: (1) 같은 이미지 $x$에서 augmentation 두 번 → $x_q, x_k$. (2) $f_q$로 $q$ ($N\times C$), $f_k$로 $k$ ($N\times C$), $k$는 detach. (3) positive logit은 $q$와 자기 $k$의 내적($N\times1$), negative logit은 $q$와 queue 전체($C\times K$)의 내적($N\times K$). (4) 둘을 concat하여 $N\times(1+K)$ logits, **label은 항상 0번(positive가 0번째 열)** 이므로 일반 CrossEntropy로 InfoNCE를 구현함. (5) $f_q$만 SGD로 업데이트, (6) $f_k$는 EMA, (7) 현재 batch의 $k$를 queue에 넣고 가장 오래된 batch를 뺌.
- 순서 주의: queue 갱신은 loss 계산 **후**이므로 현재 batch의 $k$가 자기 negative로 들어가지 않음.

> 보충: 논문은 batch 내 BN이 정보 leak(shortcut)을 만들 수 있어 **Shuffling BN**(multi-GPU에서 key encoder 샘플을 섞어 BN 통계 계산)을 사용함. 슬라이드 pseudocode에는 생략됨.

## MoCo (p.13)

- Comparisons.
- 그림 (Figure 2): 세 구조를 나란히 비교한 개념도. 모두 위쪽에 contrastive loss, 그 아래 $q\cdot k$, 그 아래 $q$, $k$, 더 아래 encoder, 맨 아래 입력 $x^q, x^k$.
  - (a) end-to-end: encoder q와 encoder k 두 개(둘 다 초록). $q$와 $k$ 양쪽에 "gradient" 화살표가 있음(양쪽 모두 backprop으로 업데이트).
  - (b) memory bank: query 쪽은 encoder, 입력 $x^q$. key 쪽은 입력 없이 **memory bank**(회색)에서 **sampling**하여 $k$를 얻음. gradient는 query 쪽에만 표시.
  - (c) MoCo: query는 encoder(초록), key는 **momentum encoder**(파랑)에 $x^k$를 통과시켜 $k$ 생성. gradient는 query 쪽에만 표시.
- 캡션: Figure 2. Conceptual comparison of three contrastive loss mechanisms (empirical comparisons are in Figure 3 and Table 3). Here we illustrate one pair of query and key. The three mechanisms differ in how the keys are maintained and how the key encoder is updated. (a): The encoders for computing the query and key representations are updated end-to-end by back-propagation (the two encoders can be different). (b): The key representations are sampled from a memory bank [61]. (c): MoCo encodes the new keys on-the-fly by a momentum-updated encoder, and maintains a queue (not illustrated in this figure) of keys.
- 출처 표기: He et al., CVPR 2020

**설명**
- 세 방식의 차이는 **"key를 어떻게 만들고 key encoder를 어떻게 갱신하느냐"** 임.
  - (a) end-to-end: key가 항상 최신 encoder 출력이라 일관적이나 dictionary 크기 = mini-batch 크기라 GPU 메모리에 제약됨 (SimCLR 계열).
  - (b) memory bank: dictionary는 크지만 key가 서로 다른 시점의 encoder 출력이라 inconsistent.
  - (c) MoCo: key를 즉석에서 만들되 encoder가 천천히 변하므로 일관성 유지 + queue로 크기 확장.

## MoCo (p.14)

- Comparisons.
- 표:

| | End-to-End | Memory Bank | MoCo |
|---|---|---|---|
| Query | Current encoder | Current encoder | Current encoder |
| Key | Current encoder | Memory bank | Momentum encoder |
| Negative Source | Mini-batch | Memory bank | FIFO queue |
| Dictionary Size | Small | Large | Large |
| Feature Freshness | Fresh | Stale | Consistent |
| Encoder Update | Backpropagation | Backpropagation | Momentum Update |

- 표 아래 문장: MoCo is essentially a hybrid of end-to-end learning and memory banks: it keeps a large dictionary like a memory bank while generating more consistent keys using a momentum encoder.
- 출처 표기: He et al., CVPR 2020

**설명**
- 표의 핵심은 **Dictionary Size × Feature Freshness** 의 trade-off: end-to-end는 (작음, 신선), memory bank는 (큼, stale), MoCo는 (큼, consistent)로 두 장점을 결합함.
- "Fresh"와 "Consistent"는 다름: MoCo의 key는 현재 encoder가 아니라 약간 과거의 momentum encoder 출력이므로 완전히 fresh하지는 않지만, queue 내 key들이 **서로 비슷한 encoder**로 만들어져 일관됨.
- Encoder Update 열: MoCo 열의 "Momentum Update"는 **key encoder**에 해당하며 query encoder는 backprop으로 학습됨. ← 표만 보면 MoCo의 encoder 전체가 momentum으로 갱신되는 것처럼 읽힐 수 있음(오해 소지).

## MoCo (p.15)

- Comparisons.
- 그림/표 (논문 캡처, 3개 표):
  - **Table 1. Comparison under the linear classification protocol on ImageNet.** 열: method / architecture / #params (M) / accuracy (%).
    - Exemplar R50w3x 211M 46.0; RelativePosition R50w2x 94M 51.4; Jigsaw R50w2x 94M 44.6; Rotation Rv50w4x 86M 55.4; Colorization R101* 28M 39.6; DeepCluster VGG 15M 48.4; BigBiGAN R50 24M 56.6, Rv50w4x 86M 61.3.
    - "methods based on contrastive learning follow:" InstDisc R50 24M 54.0; LocalAgg R50 24M 58.8; CPC v1 R101* 28M 48.7; CPC v2 R170* wider 303M 65.9; CMC R50 L+ab 47M 64.1†, R50w2x L+ab 188M 68.4†; AMDIM small 194M 63.5†, large 626M 68.1†.
    - **MoCo**: R50 24M **60.6**; RX50 46M 63.9; R50w2x 94M 65.4; R50w4x 375M **68.6**.
    - 캡션: The figure visualizes the table. All are reported as unsupervised pre-training on the ImageNet-1M training set, followed by supervised linear classification trained on frozen features, evaluated on the validation set. The parameter counts are those of the feature extractors. We compare with improved re-implementations if available (referenced after the numbers).
  - **Table 2. Object detection fine-tuned on PASCAL VOC trainval07+12.** Evaluation is on test2007: AP50 (default VOC metric), AP (COCO-style), AP75, averaged over 5 trials. All are fine-tuned for 24k iterations (~23 epochs). In the brackets are the gaps to the ImageNet supervised pre-training counterpart. In green are the gaps of at least +0.5 point.
    - (a) Faster R-CNN, R50-dilated-C5 (AP50 / AP / AP75): random init. 64.4 / 37.9 / 38.6; super. IN-1M 81.4 / 54.0 / 59.1; MoCo IN-1M 81.1 (-0.3) / 54.6 (+0.6) / 59.9 (+0.8); MoCo IG-1B 81.6 (+0.2) / 55.5 (+1.5) / 61.2 (+2.1).
    - (b) Faster R-CNN, R50-C4: random init. 60.2 / 33.8 / 33.1; super. IN-1M 81.3 / 53.5 / 58.8; MoCo IN-1M 81.5 (+0.2) / 55.9 (+2.4) / 62.6 (+3.8); MoCo IG-1B 82.2 (+0.9) / 57.2 (+3.7) / 63.7 (+4.9).
  - **Table 3. Comparison of three contrastive loss mechanisms** on PASCAL VOC object detection, fine-tuned on trainval07+12 and evaluated on test2007 (averages over 5 trials). All models are implemented by us (Figure 3), pre-trained on IN-1M, and fine-tuned using the same settings as in Table 2.

| pre-train | R50-dilated-C5 AP50 | AP | AP75 | R50-C4 AP50 | AP | AP75 |
|---|---|---|---|---|---|---|
| end-to-end | 79.2 | 52.0 | 56.6 | 80.4 | 54.6 | 60.3 |
| memory bank | 79.8 | 52.9 | 57.9 | 80.6 | 54.9 | 60.6 |
| **MoCo** | **81.1** | **54.6** | **59.9** | **81.5** | **55.9** | **62.6** |

- 출처 표기: He et al., CVPR 2020

**설명**
- **Table 1**: MoCo R50(24M)이 같은 크기 R50의 InstDisc(54.0), LocalAgg(58.8)보다 높은 60.6을 달성. 모델을 키우면(R50w4x) 68.6으로 당시 최고 수준(CMC 68.4†, AMDIM large 68.1†보다 약간 높음)임. 다만 contrastive 계열이 항상 우세한 것은 아님(예: CPC v1 48.7은 Rotation 55.4보다 낮음). 표가 보여주는 것은 MoCo/CMC/AMDIM 같은 최신 contrastive 방식이 pretext-task 방식(Jigsaw, Rotation, Colorization 등)보다 대체로 높다는 점임.
- **Table 2**: 핵심 메시지는 **unsupervised pre-training이 downstream detection에서 ImageNet supervised pre-training과 대등하거나 능가**한다는 점(특히 AP75에서 큰 향상, 더 큰 데이터인 Instagram 1B(IG-1B)에서 이득 증가). random init 대비는 큰 폭 향상.
- **Table 3**: 같은 설정에서 세 mechanism을 직접 비교한 ablation. **MoCo > memory bank > end-to-end** 이고, 이는 "큰 dictionary + 일관성"의 중요성을 실증함. (슬라이드에 해석 문장은 없고 위는 정리자의 해석임.)

> 보충: end-to-end가 가장 낮은 것은 batch 크기 제약(작은 dictionary) 때문으로 논문이 논함.

- Table 1의 † 표기는 슬라이드 캡션에 설명이 없음 ← 원 논문에서는 "ImageNet 라벨로 supervised된 FastAutoAugment를 pre-training에 사용"한 결과를 가리키는 표기임(정확한 문구는 원 논문 확인 필요).

## MoCo-V2 (p.16)

- **MoCo v2 (2020)**
  - Adopts the **MLP projection head** from SimCLR.
  - Uses **stronger data augmentation** for contrastive learning.
  - Improves representation quality and transfer performance.
  - Retains the efficient momentum encoder and queue mechanism.
- **Key Idea**: Combine MoCo with SimCLR's effective training strategies.
- 출처 표기: Chen et al., 2020, Improved Baselines with Momentum Contrastive Learning

**설명**
- MoCo v2 (Chen, Fan, Girshick, He, 2020)는 새 구조 제안이 아니라 **SimCLR에서 효과가 검증된 기법을 MoCo에 이식**한 것임: (1) projection head를 linear FC에서 **2-layer MLP (with ReLU)** 로 교체, (2) **stronger augmentation** (예: Gaussian blur 추가) 사용.
- MoCo의 queue + momentum encoder는 유지 → SimCLR처럼 큰 batch가 필요 없어 **일반적인 8-GPU 머신에서도 학습 가능**하다는 장점을 유지하면서 성능은 SimCLR를 능가한다고 보고됨.

> 보충: v2 논문은 MLP head, augmentation 외에 cosine learning rate schedule, 더 긴 학습 등도 사용함. 정확한 수치는 원 논문 확인 필요.

## FIGURES

- p.11: MoCo 구조도 (encoder / momentum encoder / queue / contrastive loss) + 수식 — 중요도 상
- p.12: MoCo pseudocode (Algorithm 1) 및 주석 박스 — 중요도 상
- p.13: end-to-end vs memory bank vs MoCo 개념 비교 Figure 2 — 중요도 상
- p.14: 세 방식 비교 표 — 중요도 상
- p.15: Table 1 (ImageNet linear classification), Table 3 (세 mechanism 비교) — 중요도 중

## EXAM POINTS

1. **Memory bank의 장단점**: 장점 = 큰 negative 수, 낮은 계산 비용 / 단점 = **stale embedding** (과거 encoder로 만든 feature와 현재 query 비교). 왜 stale이 문제인지 설명할 수 있어야 함.
2. **MoCo의 핵심 아이디어**: "memory bank → momentum-updated queue". 큰 dictionary + 일관성(consistency)을 동시에 달성하는 방법을 설명.
3. **Momentum update 수식** $\theta_k \leftarrow m\theta_k + (1-m)\theta_q$ : 각 기호 의미, $m$이 1에 가까워야 하는 이유(key encoder가 천천히 변해 queue 안 key들이 일관됨), key encoder는 gradient 없이 EMA로만 갱신됨.
4. **MoCo loss (InfoNCE)** $-\log \frac{\exp(q\cdot k^+/\tau)}{\sum_{i}\exp(q\cdot k_i/\tau)}$ : positive 1개 + queue의 negative들에 대한 $(K+1)$-way softmax CE. pseudocode에서 label=0, `cat([l_pos, l_neg])` 구성.
5. **SimCLR vs MoCo 비교**: negative 출처(batch vs queue), gradient 경로(양쪽 vs query만), batch 크기와 negative 수의 decoupling, 큰 batch 필요 여부.
6. **End-to-end / Memory bank / MoCo 비교표** (key 생성 방식, negative source, dictionary 크기, freshness, 업데이트 방식)를 재현하고 MoCo가 hybrid인 이유를 서술.
7. **왜 key에 stop-gradient(detach)인가**: queue의 많은 key에 backprop 하지 않아 메모리/연산 절약, dictionary 크기를 batch와 분리 가능.
8. **MoCo v2**: MoCo에 SimCLR의 **MLP projection head + stronger augmentation**을 추가한 것, queue/momentum encoder는 유지. 그 효과와 의의.

## VERIFIER REPORT
- p.9 보충: Wu et al. 2018이 "moving average"로 bank를 갱신한다는 부정확한 서술을 "샘플이 뽑힐 때 현재 feature로 덮어쓰기 + proximal regularization"으로 수정.
- p.11 보충: m 관련 서술을 논문 ablation에 맞게 수정(m=0 수렴 실패, 0.9 성능 저하, 0.99~0.999 양호).
- p.12: "Generate a positive pair" 빨간 박스 대상을 슬라이드대로 q/k forward 두 줄로 정정(augmentation 줄은 박스 밖).
- p.15 설명: "contrastive 계열이 pretext보다 우세"라는 과장을 완화(CPC v1 48.7 < Rotation 55.4), CMC/AMDIM large 비교 수치 명시.
- p.15 † 표기: 의미를 FastAutoAugment(ImageNet 라벨 supervised) 사용으로 보강.
- 수식, 표 수치(Table 1-3), 그림 설명(p.11, 13, 14), 인용 논문명/연도는 원본과 일치 확인.
