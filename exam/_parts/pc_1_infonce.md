# PC1. InfoNCE family (SimCLR · MoCo · CLIP) — Pseudo code 핵심 정리 & 문제

> 교수님: *"study the pseudo code"* → 구현 디테일이 아니라 **"논문의 core idea가 코드 어느 줄에 박혀 있나"** 를 묻는다고 보고 정리함. 세 방법 모두 loss의 뼈대는 **InfoNCE = positive 1개 vs negatives의 softmax cross-entropy** 이고, 차이는 **negative를 어디서 가져오나 / label을 어떻게 만드나 / gradient가 어디로 흐르나** 뿐임.

## A. Annotated pseudo code

### SimCLR (Chen et al., ICML 2020) — L04 p.25

**Transcription** (Algorithm 1, 수식형 pseudo code — 슬라이드 렌더 그대로)

```
Algorithm 1  SimCLR's main learning algorithm.

input: batch size N, constant τ, structure of f, g, T.
for sampled minibatch {x_k}_{k=1}^{N} do
    for all k ∈ {1, ..., N} do
        draw two augmentation functions t∼T, t′∼T
        # the first augmentation
        x̃_{2k−1} = t(x_k)
        h_{2k−1} = f(x̃_{2k−1})                    # representation
        z_{2k−1} = g(h_{2k−1})                     # projection
        # the second augmentation
        x̃_{2k} = t′(x_k)
        h_{2k} = f(x̃_{2k})                        # representation
        z_{2k} = g(h_{2k})                         # projection
    end for
    for all i ∈ {1, ..., 2N} and j ∈ {1, ..., 2N} do
        s_{i,j} = z_iᵀ z_j / (‖z_i‖ ‖z_j‖)         # pairwise similarity
    end for
    define ℓ(i,j) as  ℓ(i,j) = −log [ exp(s_{i,j}/τ) / Σ_{k=1}^{2N} 1_[k≠i] exp(s_{i,k}/τ) ]
    L = 1/(2N) Σ_{k=1}^{N} [ ℓ(2k−1, 2k) + ℓ(2k, 2k−1) ]
    update networks f and g to minimize L
end for
return encoder network f(·), and throw away g(·)
```

$$\ell(i,j) = -\log\frac{\exp(s_{i,j}/\tau)}{\sum_{k=1}^{2N}\mathbb{1}_{[k\neq i]}\exp(s_{i,k}/\tau)},\qquad \mathcal{L} = \frac{1}{2N}\sum_{k=1}^{N}\big[\ell(2k-1,2k)+\ell(2k,2k-1)\big]$$

슬라이드 callout (박스 → 대상 줄)
- 빨간 박스 "**Generate a positive pair by sampling data augmentation functions**" → `draw two augmentation functions t∼T, t′∼T`, $\tilde x_{2k-1}=t(x_k)$, $\tilde x_{2k}=t'(x_k)$
- 파란 박스 "**InfoNCE loss: Use all non-positive samples in the batch as x -**" → $\ell(i,j)$ 정의 ← "x -"는 $x^-$ (negative) 위첨자가 깨진 표기로 보임 (원문 유지)
- 분홍 박스 "**Iterate through and use each of the 2N sample as reference, compute average loss**" → $\mathcal L$ 줄 ← "each of the 2N sample**s**" 문법 오타 (원문 유지)
- 읽기 주의: $k$가 **두 가지 의미**로 재사용됨 — (i) 이미지 index: $x_k$, `for all k`, $\mathcal L$의 $\sum_{k=1}^{N}$ (k번째 이미지의 두 view = $2k-1, 2k$) / (ii) 분모의 dummy index $\sum_{k=1}^{2N}$ (2N개 view 전체를 훑음). 오타는 아니고 논문 표기 그대로.

**핵심 줄 (Key lines)**

| line | what it does | which core idea it implements | 빈칸 후보 |
|---|---|---|---|
| `draw two augmentation functions t∼T, t′∼T` → $\tilde x_{2k-1}=t(x_k)$, $\tilde x_{2k}=t'(x_k)$ | 같은 $x_k$에서 두 view 생성 → index $(2k-1, 2k)$가 positive pair | **augmentation이 곧 supervision** — "maximizing agreement between differently augmented views of the same sample" (p.23) | augmentation, $t'$ |
| $h=f(\tilde x)$ `# representation`, $z=g(h)$ `# projection` | base encoder $f$ → $h$, projection head $g$ → $z$ | loss는 **projection head 출력 $z$** 위에서 계산 (p.24) | representation / projection |
| $s_{i,j}=z_i^\top z_j/(\lVert z_i\rVert\lVert z_j\rVert)$ | $2N\times 2N$ 모든 쌍의 cosine similarity | in-batch 전체를 비교 대상으로 씀 (memory bank·queue 없음) | cosine, pairwise similarity |
| $\ell(i,j)$ with $\mathbb{1}_{[k\neq i]}$ | anchor $i$ 기준 $(2N-1)$-way softmax CE, 정답 = positive $j$ | **InfoNCE / NT-Xent**, negatives = 같은 batch의 나머지 **$2N-2$개** ("Use all non-positive samples in the batch") | $\mathbb{1}_{[k\neq i]}$, $\tau$, $2N$ |
| $\mathcal L=\frac{1}{2N}\sum_{k=1}^N[\ell(2k-1,2k)+\ell(2k,2k-1)]$ | 두 view가 번갈아 anchor → $2N$개 평균 | **symmetric**: 2N개 sample 모두 reference로 한 번씩 | $2N$, $2k-1$ |
| `return encoder network f(·), and throw away g(·)` | downstream엔 $f$(즉 $h$)만 사용 | $z$는 augmentation에 invariant → 정보 손실, $g$가 흡수 → **$h$에 정보 보존** (p.28) | $g(\cdot)$, $f(\cdot)$ |

**Pseudo code ↔ core idea 한 줄**: 같은 $x_k$에서 뽑은 $t,t'$ 두 view를 positive로, **같은 batch의 나머지 $2N-2$개를 negative**로 놓고 $\mathbb{1}_{[k\neq i]}$ InfoNCE를 **양방향 평균** → 학습 끝나면 **$g$ 버리고 $f$만** 남김. negative가 batch에서만 나오니 **large batch 필수** (p.24, p.29).

- Implementation trivia (low priority): large batch → backprop 메모리 커서 **TPU 분산 학습** (p.29).

---

### MoCo (He et al., CVPR 2020) — L05 p.12

**Transcription** (Algorithm 1: Pseudocode of MoCo in a PyTorch-like style — 슬라이드 렌더 그대로)

```python
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
```
슬라이드 하단 주석: *bmm: batch matrix multiplication; mm: matrix multiplication; cat: concatenation.*

슬라이드 callout (박스/화살표 → 대상 줄)
- 좌 "**Generate a positive pair by sampling data augmentation functions**" → `q = f_q.forward(x_q)` / `k = f_k.forward(x_k)` 박스 ← 화살표가 `aug(x)` 두 줄이 아니라 **forward 두 줄**을 가리킴 (실제 augmentation은 `x_q = aug(x)`, `x_k = aug(x)`). 원문 유지
- 좌 "**No gradient through the key**" → `k = k.detach()` 줄 밑줄
- 우 "**Use the running queue of keys as the negative samples**" → `l_neg` 박스
- 우 "**InfoNCE loss**" → `logits` / `labels` / `loss` 박스 ← 박스 윗변이 `# logits: Nx(1+K)` 주석 줄을 가로질러 취소선처럼 보임 (내용은 그대로 주석)
- 우 "**Update f_k through momentum**" → momentum update 박스
- 좌 "**Update the FIFO negative sample queue**" → `enqueue` / `dequeue` 박스

**핵심 줄 (Key lines)**

| line | what it does | which core idea it implements | 빈칸 후보 |
|---|---|---|---|
| `k = k.detach()  # no gradient to keys` | key 쪽으로 gradient 차단 → `update(f_q.params)`로 **query encoder만 SGD** | "Compute gradients and update the encoder only through the queries" (p.11) — key encoder는 backprop 안 함 | `detach` |
| `l_pos = bmm(q.view(N,1,C), k.view(N,C,1))` | 각 query와 **자기 key만** 내적 → $N\times1$ | positive = 같은 $x$의 다른 augmentation을 key encoder에 통과시킨 key | `bmm`, `Nx1` |
| `l_neg = mm(q.view(N,C), queue.view(C,K))` | 각 query vs **queue의 K개 key** → $N\times K$ | "running queue of keys as the negative samples" → negative 수 $K$가 **batch size와 decouple** (p.11) | `queue`, `K`, `NxK` |
| `logits = cat([l_pos, l_neg], dim=1)` + `labels = zeros(N)` + `CrossEntropyLoss(logits/t, labels)` | positive를 **0번 column**에 붙이고 $(1+K)$-way CE | **InfoNCE를 일반 CE로 구현**: 모든 query의 정답 class = 0 | `zeros`, `0-th`, `1+K` |
| `f_k.params = m*f_k.params+(1-m)*f_q.params` | key encoder = query encoder의 **EMA** | **momentum encoder** → key encoder가 "slowly progressing" → queue 속 key들이 **consistent** (p.11, p.14) | `m`, `(1-m)` |
| `enqueue(queue, k)` / `dequeue(queue)` | 현재 batch의 **key**를 넣고 **가장 오래된(earliest)** batch를 뺌 | **FIFO queue** = memory bank를 대체하는 large & consistent dictionary (p.10) | `k`, `earliest` |

**Pseudo code ↔ core idea 한 줄**: `k.detach()` + momentum 줄 = **key encoder는 backprop 없이 EMA로만 천천히 변함(consistent keys)**, `l_neg`의 `queue` + 끝의 enqueue/dequeue = **batch size와 무관한 large dictionary**, `zeros(N)` = InfoNCE를 $(1+K)$-way CE로 구현. → "Replace the memory bank with a momentum-updated queue" (p.10).

- 순서 포인트: queue 갱신(enqueue/dequeue)은 loss·update **이후** → 현재 batch의 key는 이번 step에선 자기 query의 positive로만 쓰이고, negative로는 **다음 batch부터** 쓰임.
- Implementation trivia (low priority): 논문은 BN을 통한 batch 내 정보 leak을 막는 **Shuffling BN**을 쓰지만 pseudo code에선 생략 (슬라이드에 없음).

---

### CLIP (Radford et al., PMLR 2021) — L05 p.35

**Transcription** (numpy-like pseudo code — 슬라이드 렌더 그대로, callout 박스 없음)

```python
# image_encoder - ResNet or Vision Transformer
# text_encoder  - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l]       - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t             - learned temperature parameter

# extract feature representations of each modality
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T)  #[n, d_t]

# joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)

# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)

# symmetric loss function
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss   = (loss_i + loss_t)/2
```

- 모호한 점: `loss_i`(axis=0) / `loss_t`(axis=1)가 각각 image→text인지 text→image인지는 슬라이드가 `cross_entropy_loss`의 axis 의미를 정의하지 않아서 **해석이 갈림** (원본 코드 그대로). → 시험 대비로는 **"axis만 다른 두 CE를 평균 = symmetric"** 까지만 확실히 기억.

**핵심 줄 (Key lines)**

| line | what it does | which core idea it implements | 빈칸 후보 |
|---|---|---|---|
| `I_f = image_encoder(I)`, `T_f = text_encoder(T)` | modality마다 **별도 encoder** (ResNet/ViT, CBOW/Text Transformer) | "Jointly learns image and text representations" (p.32) | `image_encoder`, `text_encoder` |
| `I_e = l2_normalize(np.dot(I_f, W_i), axis=1)` (T_e 동일) | learned projection $W_i, W_t$로 서로 다른 차원 $d_i, d_t$ → 공통 $d_e$, 그다음 L2 normalize | **shared (joint multimodal) embedding space**; normalize했으니 dot = cosine | `l2_normalize`, `W_i`, `d_e` |
| `logits = np.dot(I_e, T_e.T) * np.exp(t)` | $n\times n$ image–text cosine similarity matrix × scale | **N×N similarity matrix** (p.33) + **learned temperature** ($e^{t}$ 곱함, MoCo처럼 `/t` 아님) | `np.exp(t)`, `[n, n]`, cosine |
| `labels = np.arange(n)` | $i$번째 이미지의 정답 = $i$번째 text → **diagonal** | matched (image, caption) pair = positive, 같은 row/column의 나머지 $n-1$개 = **in-batch negatives** | `arange` |
| `loss_i ... axis=0`, `loss_t ... axis=1`, `loss = (loss_i + loss_t)/2` | 같은 logits·labels로 두 방향 CE 후 평균 | **symmetric contrastive loss** (p.33) | `axis`, `symmetric`, `/2` |

**Pseudo code ↔ core idea 한 줄**: 두 modality를 각자 encode → **learned projection + L2 norm으로 shared space** → $n\times n$ cosine matrix에서 **diagonal(`arange(n)`)을 정답**으로 axis 0/1 양방향 CE 평균 = **symmetric contrastive loss**. positive가 augmentation이 아니라 **자연어 caption**이라는 게 SimCLR와의 핵심 차이 (p.32).

- Implementation trivia (low priority): `t`는 log-scale parameter라 `np.exp(t)`로 항상 양수 scale을 보장 (> 보충, 슬라이드엔 "learned temperature parameter"만).

---

## B. Cross-method comparison

| | **SimCLR** (L04 p.25) | **MoCo** (L05 p.12) | **CLIP** (L05 p.35) |
|---|---|---|---|
| Positive | 같은 $x_k$의 두 augmented view ($2k-1$, $2k$) | 같은 $x$의 두 aug: query $q$ ↔ 자기 key $k$ (`l_pos`) | matched (image, caption) pair = **diagonal** |
| Negatives 출처 | **in-batch**: $2N-2$개 (anchor·positive 제외) | **queue**: $K$개 (`l_neg`; 현재 batch의 다른 key는 이번 step엔 안 씀) | **in-batch**: 방향당 $n-1$개 (같은 row/column의 off-diagonal) |
| softmax 크기 (anchor당) | $2N-1$ ($\mathbb 1_{[k\neq i]}$) | $1+K$ (`Nx(1+K)`) | $n$ |
| `labels` 구성 | 명시적 labels 없음 — 분자에 $s_{i,j}$로 positive를 직접 지정 | `zeros(N)` — positive를 **0번 column**에 concat | `np.arange(n)` — positive = **diagonal** |
| Symmetric? | **Yes** — $\ell(2k-1,2k)+\ell(2k,2k-1)$, $\frac{1}{2N}$ | **No** — query→keys 한 방향 | **Yes** — axis=0 + axis=1, `/2` |
| Similarity / temperature | cosine $s_{i,j}$, constant $\tau$ (`/τ`) | `q·k` 내적, `logits/t` | cosine (`l2_normalize`), `* np.exp(t)` (**learned** t) |
| Encoder 구성 | 하나의 $f$ (+ $g$)를 두 view가 공유 | 같은 구조 2개: $f_q$, $f_k$ (`f_k.params = f_q.params # initialize`) | **서로 다른** 2개: image encoder, text encoder (+ $W_i$, $W_t$) |
| Gradient 받는 것 | $f$와 $g$ ("update networks f and g") | **$f_q$만** (`k.detach()`, `update(f_q.params)`) | image/text encoder, $W_i$, $W_t$, $t$ 모두 |
| Momentum-updated | 없음 | **$f_k$** (EMA of $f_q$) | 없음 |
| 학습 후 버리는 것 | **$g(\cdot)$ 버림**, $f$ 사용 | 슬라이드 명시 없음 (> 보충: 논문은 $f_q$를 downstream에 사용) | 없음 — 추론 때 두 encoder 다 씀 (image + text prompt, p.33) |
| 큰 negative 수를 얻는 방법 | **large batch** (p.29) | **queue** — batch size와 decouple | **large batch** (in-batch) ← 슬라이드엔 명시 없음 (> 보충: 논문 batch 32,768) |

---

## C. Question pool (14 questions)

> 빈칸 6 · MCQ 4 · T/F 4. killer: **Q3, Q9, Q14** (그래도 전부 core line 이해 문제).

### Q1 [Blank]
SimCLR's Algorithm 1 ends with the line: "return encoder network (1) ______, and throw away (2) ______."
**Answer**: (1) f(·) (2) g(·) (accept: (1) f, the base encoder (2) g, the projection head) — loss는 projection head 출력 $z=g(h)$ 위에서 계산하지만 학습 후엔 **$g$를 버리고 $f$의 출력 $h$를 downstream에 씀**. $z$는 augmentation에 invariant하게 학습돼서 정보를 잃고, $g$가 그 부담을 흡수해서 $h$에 정보가 더 남기 때문. 함정 변형: "keep g, throw away f" (L04 p.25, p.28)

### Q2 [T/F]
In SimCLR's Algorithm 1, the indicator $\mathbb{1}_{[k\neq i]}$ removes only the anchor $i$ itself from the denominator of $\ell(i,j)$, so the denominator contains the positive term plus $2N-2$ negative terms.
**Answer**: True — 분모 = $\sum_{k=1}^{2N}\mathbb{1}_{[k\neq i]}\exp(s_{i,k}/\tau)$ → 자기 자신(similarity 항상 1)만 빠지고 **$2N-1$개 = positive 1 + negatives $2(N-1)$**. "other 2(N-1) data points are treated as negative samples" (p.23), "Use all non-positive samples in the batch as $x^-$" (p.25). 함정 변형: "includes $i$ itself" / "$2N-1$ negatives" → False (L04 p.23, p.25)

### Q3 [Blank]
Complete SimCLR's final loss in Algorithm 1, in which every augmented view in the mini-batch is used as the reference (anchor) exactly once:
$$\mathcal{L} = \frac{1}{(1)\,\underline{\qquad}}\sum_{k=1}^{N}\big[\ell(2k-1,\,2k) + \ell(2k,\ (2)\,\underline{\qquad})\big]$$
**Answer**: (1) 2N (2) 2k−1 (accept: (1) $2N$ (2) $2k-1$) — $\ell(i,j)$는 anchor $i$의 분모로 normalize하니까 **비대칭** → 두 view를 번갈아 anchor로 써야 함 (callout "Iterate through and use each of the 2N sample as reference, compute average loss"). $N$쌍 × 2방향 = $2N$개 항이라 $\frac{1}{2N}$로 평균. 함정: $\frac1N$, 한 방향($\ell(2k-1,2k)$)만 쓰기 (L04 p.25)

### Q4 [Blank]
In MoCo's pseudo code, the line `k = k.______()  # no gradient to keys` ensures that the contrastive loss does not backpropagate into the key encoder.
**Answer**: detach (accept: —) — callout "No gradient through the key". 그래서 `loss.backward()` 다음 `update(f_q.params)` → **query encoder만 SGD**, key encoder는 momentum update로만 바뀜. SimCLR와의 key difference: "Compute gradients and update the encoder only through the queries" (L05 p.11). (L05 p.12)

### Q5 [MCQ]
Which line of MoCo's pseudo code implements "Use the running queue of keys as the negative samples"?
(A) `l_pos = bmm(q.view(N,1,C), k.view(N,C,1))`
(B) `l_neg = mm(q.view(N,C), queue.view(C,K))`
(C) `enqueue(queue, k)`
(D) `f_k.params = m*f_k.params+(1-m)*f_q.params`
**Answer**: (B) — 각 query($N\times C$)를 queue의 $K$개 key($C\times K$)와 곱해 **negative logits $N\times K$**. (A)는 각 query와 자기 key만 곱한 **positive** logits $N\times1$. (C)는 queue를 **갱신**하는 줄(callout "Update the FIFO negative sample queue")이지 negative를 계산하는 줄이 아님. (D)는 key encoder의 **momentum update** (L05 p.12)

### Q6 [Blank]
In MoCo's pseudo code:
```python
logits = cat([l_pos, l_neg], dim=1)        # Nx(1+K)
labels = ______(N)  # positives are the 0-th
loss = CrossEntropyLoss(logits/t, labels)
```
**Answer**: zeros (accept: `torch.zeros`) — `l_pos`를 **맨 앞(0번 column)**에 concat했으니 모든 query의 정답 class가 0 → 일반 CE가 곧 $(1+K)$-way **InfoNCE**. 함정: CLIP은 `np.arange(n)` (positive가 diagonal) (L05 p.12; cf. L05 p.35)

### Q7 [T/F]
In MoCo's pseudo code, the key network is updated by `f_k.params = m*f_q.params + (1-m)*f_k.params`.
**Answer**: False — 슬라이드는 `f_k.params = m*f_k.params+(1-m)*f_q.params` → $\theta_k \leftarrow m\theta_k + (1-m)\theta_q$. $m$은 **$f_k$ 자기 자신**에 곱함 → $m$이 1에 가까우면 key encoder가 "slowly progressing"해서 queue 안 key들이 **consistent**. 뒤바꾸면 $f_k$가 매 step $f_q$를 거의 복사하게 돼 consistency가 깨짐. query encoder는 momentum이 아니라 `loss.backward(); update(f_q.params)`로 갱신 (L05 p.11–12)

### Q8 [Blank]
MoCo's pseudo code ends each iteration with:
```python
# update dictionary
enqueue(queue, (1)____)  # enqueue the current minibatch
dequeue(queue)  # dequeue the (2)________ minibatch
```
**Answer**: (1) k (2) earliest (accept: (1) the keys (2) oldest) — dictionary = **key encoder가 만든 key들의 FIFO queue**. 현재 batch의 key를 넣고 가장 오래된 batch를 뺌 → 가장 낡은 key가 빠지니 staleness ↓, queue 크기 $K$ 유지. 함정: `enqueue(queue, q)` (query를 넣음), "latest/most recent" 제거 (L05 p.10, p.12)

### Q9 [T/F]
In MoCo's pseudo code, `enqueue(queue, k)` and `dequeue(queue)` are executed at the end of the iteration, after the loss is computed, so the negatives in `l_neg` for the current queries come only from the queue, not from the other keys of the current mini-batch.
**Answer**: True — `l_neg = mm(q, queue)`는 **queue만** 사용, `l_pos`는 `bmm`이라 각 query가 **자기 key 하나**와만 곱해짐 → 같은 batch의 다른 key는 이번 step에 negative로 안 쓰이고, enqueue된 뒤 **다음 batch부터** negative가 됨. 그래서 negative 수 $K$가 batch size와 **decouple** ("Decouple min-batch size with the number of keys", p.11). SimCLR(in-batch negatives)와 대비 (L05 p.11–12)

### Q10 [Blank]
In CLIP's pseudo code:
```python
# joint multimodal embedding [n, d_e]
I_e = (1)____________(np.dot(I_f, W_i), axis=1)
T_e = (1)____________(np.dot(T_f, W_t), axis=1)
# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)
# symmetric loss function
labels = np.(2)______(n)
```
**Answer**: (1) l2_normalize (2) arange (accept: (1) L2 normalization, normalize) — (1) learned projection $W_i, W_t$로 공통 차원 $d_e$로 보낸 뒤 **L2 normalize** → `np.dot`이 곧 **cosine similarity** (주석 "scaled pairwise cosine similarities"). (2) `arange(n)` = $[0,1,\dots,n-1]$ → $i$번째 이미지의 정답은 $i$번째 text, 즉 **diagonal이 positive**. MoCo의 `zeros(N)`과 헷갈리지 말 것 (L05 p.35)

### Q11 [T/F]
In CLIP's pseudo code, `t` is a fixed temperature hyperparameter, and the logits are divided by it in the same way as MoCo's `CrossEntropyLoss(logits/t, labels)`.
**Answer**: False — CLIP 주석은 `# t - learned temperature parameter`이고 코드는 `logits = np.dot(I_e, T_e.T) * np.exp(t)` → **학습되는 parameter**, 나누는 게 아니라 **$e^{t}$를 곱함** ($e^{t}=1/\tau$). MoCo는 `logits/t`로 나눔. (L05 p.35; cf. L05 p.12)

### Q12 [MCQ]
The following snippet (variable names changed) comes from the lecture's pseudo code. Which method is it from, and what are the positives?
```python
logits = np.dot(A_e, B_e.T) * np.exp(t)      # [n, n]
labels = np.arange(n)
loss = (ce(logits, labels, axis=0) + ce(logits, labels, axis=1)) / 2
```
(A) SimCLR — the positives are two augmented views of the same image.
(B) MoCo — the positive is always placed in column 0 of the logits.
(C) CLIP — the positives are the diagonal entries (matched image–text pairs).
(D) CLIP — the positives are all off-diagonal entries, since the two inputs come from different modalities.
**Answer**: (C) — `np.exp(t)`(learned temperature) + `arange(n)` + axis 0/1 평균(symmetric)은 CLIP의 시그니처. `arange(n)`이라 정답이 **diagonal** = matched (image, caption) pair, off-diagonal $n-1$개는 negative. (A) SimCLR는 수식형 Algorithm이고 $2N\times2N$ 행렬에서 $\mathbb 1_{[k\neq i]}$로 자기 자신을 뺌. (B) MoCo는 `zeros(N)` + $N\times(1+K)$ logits. (D) off-diagonal은 negative (L05 p.33, p.35)

### Q13 [MCQ]
Based on the three pseudo codes, which statement about parameter updates is correct?
(A) SimCLR updates both f and g by backpropagation; MoCo updates f_q by backpropagation and f_k only by the momentum (EMA) rule; CLIP learns both encoders together with W_i, W_t, and t.
(B) SimCLR updates only the encoder f; the projection head g stays fixed at its random initialization.
(C) MoCo updates both f_q and f_k with `loss.backward()`; the momentum line only smooths the queue.
(D) In CLIP, the text encoder is a momentum (EMA) copy of the image encoder, as in MoCo.
**Answer**: (A) — SimCLR "update networks f and g to minimize $\mathcal L$" (L04 p.25). MoCo "# SGD update: query network" → `update(f_q.params)`, "# momentum update: key network" (L05 p.12). CLIP은 $W_i$, $W_t$, $t$ 모두 "learned", "Jointly learns image and text representations" (L05 p.32, p.35). (B) $g$도 학습함 — 학습 **후에** 버리는 것. (C) `k.detach()` 때문에 $f_k$로는 gradient가 안 감, momentum 줄은 **key encoder 갱신**. (D) CLIP은 서로 다른 encoder 두 개(ResNet/ViT vs CBOW/Text Transformer), momentum 줄 없음

### Q14 [MCQ]
For a single anchor (query) in one direction, how many **negatives** appear in the softmax denominator of each pseudo code? (SimCLR: mini-batch of N images → 2N views; MoCo: mini-batch N, queue size K; CLIP: mini-batch of n pairs)
(A) SimCLR $2N-1$ · MoCo $K$ · CLIP $n$
(B) SimCLR $2N-2$ · MoCo $K$ · CLIP $n-1$
(C) SimCLR $2N-2$ · MoCo $K+(N-1)$ · CLIP $n-1$
(D) SimCLR $N-1$ · MoCo $K$ · CLIP $2(n-1)$
**Answer**: (B) — SimCLR: 분모 $2N-1$항 = positive 1 + **$2N-2$** negatives ($\mathbb 1_{[k\neq i]}$, "2(N-1)"). MoCo: logits $N\times(1+K)$ = positive 1 + **$K$** queue negatives. CLIP: 한 row(또는 column) $n$개 = diagonal 1 + **$n-1$** negatives. (A) positive까지 negative로 셈. (C) MoCo가 현재 batch의 다른 key도 negative로 쓴다고 착각 — `l_neg`는 queue만 씀. (D) SimCLR는 이미지당 view가 2개라 $N-1$이 아니고, CLIP은 방향(axis)별로 따로 softmax라 $2(n-1)$이 아님 (L04 p.23, p.25; L05 p.12, p.35)

<!-- WRITTEN 2026-10-22 | author: opus | PC1 -->

<!-- VERIFIED 2026-10-22 | checked: 18 | fixed: 3 | deleted: 0 | reviewer: opus -->
<!-- scope: 14 questions + 3 method blocks (transcription/callouts/key-line table vs rendered L04 p.25, L05 p.12, L05 p.35) + section B. All 3 transcriptions match the slides token-for-token; all 14 answer keys correct. -->
<!-- A/SimCLR "읽기 주의": said k has 3 different meanings; x_k, `for all k` and the k in L's sum are the same image index (2 meanings, not 3) -> corrected. -->
<!-- B/CLIP "large batch": not stated on any slide -> tagged as 보충 (paper value). -->
<!-- Q3: stem said "each of the 2N augmented samples", which handed over blank (1)=2N -> reworded to "every augmented view in the mini-batch"; key unchanged. -->
