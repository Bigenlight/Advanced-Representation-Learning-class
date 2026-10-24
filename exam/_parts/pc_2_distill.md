# PC2. Self-distillation & soft targets (DINO · X-CLR · BYOL · DeepCluster) — Pseudo code 핵심 정리 & 문제

> 이 그룹 공통 질문: **"정답(target)을 어디서 만들어 오나?"**
> DINO = EMA teacher의 softmax 분포 / X-CLR = frozen text encoder의 similarity graph / BYOL = EMA target network의 projection / DeepCluster = k-means cluster ID.
> 공통점: **target 쪽으로는 gradient가 안 흐름** (`detach()` / `sg` / 이산 cluster ID). pseudo code 문제는 거의 다 "그 target을 만드는 줄"과 "gradient를 막는 줄"을 물어볼 거라고 보면 됨.

---

## A. Annotated pseudo code

### DINO (Caron et al., ICCV 2021) — L04 p.40

**Transcription** (slide 오른쪽, 캡션: **Algorithm 1** DINO PyTorch pseudocode w/o multi-crop.)

```python
# gs, gt: student and teacher networks
# C: center (K)
# tps, tpt: student and teacher temperatures
# l, m: network and center momentum rates
gt.params = gs.params
for x in loader: # load a minibatch x with n samples
    x1, x2 = augment(x), augment(x) # random views

    s1, s2 = gs(x1), gs(x2) # student output n-by-K
    t1, t2 = gt(x1), gt(x2) # teacher output n-by-K

    loss = H(t1, s2)/2 + H(t2, s1)/2
    loss.backward() # back-propagate

    # student, teacher and center updates
    update(gs) # SGD
    gt.params = l*gt.params + (1-l)*gs.params
    C = m*C + (1-m)*cat([t1, t2]).mean(dim=0)

def H(t, s):
    t = t.detach() # stop gradient
    s = softmax(s / tps, dim=1)
    t = softmax((t - C) / tpt, dim=1) # center + sharpen
    return - (t * log(s)).sum(dim=1).mean()
```

코드 아래 수식 (slide 그대로):

$$c \leftarrow mc + (1-m)\frac{1}{B}\sum_{i=1}^{B} g_{\theta_t}(x_i),$$

slide 왼쪽 그림: $x \to x_1 \to$ **student $g_{\theta_s}$** $\to$ softmax $\to p_1$ / $x \to x_2 \to$ **teacher $g_{\theta_t}$** $\to$ **centering** $\to$ softmax $\to$ **sg** $\to p_2$ / student → teacher 화살표 "ema" / loss: $-p_2\log p_1$.

- ← **`l`은 소문자 L** (헤더 주석 `# l, m: network and center momentum rates`). slide 폰트(Courier)에서 숫자 `1`과 거의 똑같이 생겨서 `1*gt.params + (1-1)*gs.params`로 읽히기 쉬움 → 그렇게 읽으면 teacher가 영원히 안 변하는 말이 안 되는 식. `l` = network momentum ($\lambda$).
- ← 수식은 소문자 $c$, 코드는 대문자 `C` (같은 center). 수식은 batch $B$개 평균, 코드는 `cat([t1, t2])` = 두 view 합친 $2n$개 평균. 표기 차이일 뿐 의미 같음 (원문 유지).
- ← 그림은 $x_1$→student, $x_2$→teacher 한 방향만 + loss $-p_2\log p_1$. 코드는 두 view를 **둘 다** student/teacher에 넣고 **교차 대칭** loss → 그림은 단순화.
- slide엔 "$\tau_t < \tau_s$" 같은 부등식·수치가 없음. sharpening 근거는 `# center + sharpen` 주석이 전부.

**핵심 줄 (Key lines)**

| line | what it does | which core idea it implements | 빈칸 후보 |
|---|---|---|---|
| `gt.params = l*gt.params + (1-l)*gs.params` | teacher = student parameter의 **EMA**. teacher는 SGD 안 받음 | **Self-distillation without labels** — teacher를 외부 pretrained 모델이 아니라 **student 자신의 EMA**로 만듦 (p.39 "Update the teacher using an EMA of the student parameters") | `l`, `gs.params`, "EMA" |
| `loss = H(t1, s2)/2 + H(t2, s1)/2` | view 1의 teacher 분포 ↔ view 2의 student 분포 (교차), 대칭 평균. `H(t1, s1)` 같은 같은-view 쌍은 없음 | student가 **다른 view**에 대한 teacher 출력을 맞히게 → augmentation-invariant | `s2`, `s1`, `/2` |
| `t = t.detach() # stop gradient` | teacher 출력으로 gradient 차단 | teacher는 backprop 대상 아님 (그림의 **sg**) | `detach`, "stop gradient" |
| `t = softmax((t - C) / tpt, dim=1) # center + sharpen` | teacher logit에서 center 빼기(**centering**) + 낮은 teacher temperature로 나누기(**sharpening**) | negative 없이 **collapse 방지**. **teacher 쪽에만** 적용 (student는 `softmax(s / tps)`만) | `C`, `tpt`, "center", "sharpen" |
| `C = m*C + (1-m)*cat([t1, t2]).mean(dim=0)` | center = **teacher 출력** batch 평균의 EMA ($c \leftarrow mc + (1-m)\frac1B\sum g_{\theta_t}(x_i)$) | centering의 기준값. DINO는 EMA가 **두 개** (teacher params + center) | `t1, t2`, `m`, "teacher" |
| `return - (t * log(s)).sum(dim=1).mean()` | teacher 분포 $P_t$ → student 분포 $P_s$의 **cross-entropy** $-\sum_k P_t^{(k)}\log P_s^{(k)}$ | **Distribution matching** (BYOL의 feature matching·MSE와 대비, p.41) | "cross-entropy", `log(s)` |

**Pseudo code ↔ core idea 한 줄**: student(`gs`)는 SGD로, teacher(`gt`)는 student의 EMA로 갱신하고, student가 **다른 view**에 대한 teacher의 **centered + sharpened softmax 분포**를 CE로 따라감 → label 없는 self-distillation.

> 보충 (논문, slide엔 없음): centering만 하면 분포가 **uniform** 쪽으로, sharpening만 하면 **한 차원이 지배**하는 쪽으로 무너지려는 경향 → 반대 방향 두 효과가 균형을 잡아서 negative 없이 collapse 회피. 논문 값 $\tau_s=0.1$, $\tau_t$ 0.04→0.07 warmup. 원 논문은 multi-crop (student는 모든 crop, teacher는 global crop만) — slide 제목 "w/o multi-crop".

Implementation trivia (low priority): `dim=1` = K개 output 차원 방향 softmax, `.mean(dim=0)` = batch 축 평균이라 center는 K차원 벡터 (`# C: center (K)`), 출력 shape `n-by-K`.

---

### X-CLR (Sobal et al., ICLR 2025) — L03 p.67

**Transcription** (slide 왼쪽 코드)

```python
def training_step(captions, images):
    # encode captions a frozen text encoder
    txt_emb = txt_enc(captions).detach() # N x D_txt
    # encode images with the model we are training
    img_emb = img_enc(images)  # N x D_img

    # calculate similarities
    tgt_sims =
        softmax(txt_emb @ txt_emb.T / tao_s, dim=1)
    img_sims =
        softmax(img_emb @ img_emb.T / tao, dim=1)
    # the loss is the cross-entropy between the two
    return CE(tgt_sims, img_sims, dim=1).mean()
```

slide 오른쪽: **Target graph** — Similarity from text or metadata / **Learned graph** — Similarity between image representations / **Training objective** — Match the learned graph to the target graph.

slide 아래 수식:

$$\mathcal{L}_{\mathbb{X}\text{-CLR}} = \frac{1}{2N_b}\sum_{i=1}^{2N_b} H(s_i, p_i)$$

$$p_{i,j} = \frac{\exp(\text{sim}(z_i,z_j)/\tau)}{\sum_{k=1}^{2N}\mathbf{1}_{k\neq i}\exp(\text{sim}(z_i,z_k)/\tau)} \qquad s_{i,j} = \frac{\exp(\boldsymbol{G}^{(\text{soft})}_{i,j}/\tau_s)}{\sum_{k=1}^{2N_b}\exp(\boldsymbol{G}^{(\text{soft})}_{i,k}/\tau_s)} \qquad \boldsymbol{G}^{(\text{soft})}_{i,j} = \text{sim}\big(f_\text{text}(c_i), f_\text{text}(c_j)\big)$$

- ← 주석 "encode captions **a** frozen text encoder" — "with" 빠진 문법 오타 (원문 유지).
- ← `tao`, `tao_s` = $\tau$, $\tau_s$ (tau 철자, 원문 유지).
- ← **대각항 불일치 (slide에서 직접 확인)**: 수식 $p_{i,j}$ 분모엔 $\mathbf{1}_{k\neq i}$ (자기 자신 제외)가 있는데 코드 `img_sims = softmax(img_emb @ img_emb.T / tao, dim=1)`엔 masking이 없어서 대각(자기 자신) 포함. 반면 target 쪽은 수식 $s_{i,j}$도 코드 `tgt_sims`도 **둘 다 mask 없음** → 불일치는 **learned graph($p$ ↔ `img_sims`) 쪽만**. (notes 03 §7.4의 플래그를 이렇게 좁혀서 읽으면 됨)
- ← 크기 불일치: 코드는 이미지당 1개, `N x D_txt` (N개). 수식은 2 view 기준 $2N_b$개, 게다가 $p_{i,j}$ 분모 상한만 $2N$ (나머지는 $2N_b$).
- ← 수식은 $\text{sim}(\cdot)$, 코드는 normalize 없이 raw dot product `@` → pseudo code 단순화.
- `CE(tgt_sims, img_sims, dim=1)`: 이미 softmax된 두 분포를 넣는 표기 = $H(s_i, p_i) = -\sum_j s_{i,j}\log p_{i,j}$. 인자 순서 **(target, prediction)** 이 수식 $H(s_i,p_i)$와 같음 (DINO의 `H(t, s)`도 target 먼저).

**핵심 줄 (Key lines)**

| line | what it does | which core idea it implements | 빈칸 후보 |
|---|---|---|---|
| `txt_emb = txt_enc(captions).detach()` | **frozen** text encoder로 caption 임베딩, gradient 차단 | target graph는 **외부 지식(text/metadata)** 에서 오고 학습 내내 **고정**. 학습되는 건 image encoder뿐 (p.66 "A fixed text encoder…") | `detach`, "frozen", "text" |
| `tgt_sims = softmax(txt_emb @ txt_emb.T / tao_s, dim=1)` | text–text similarity N×N → row-wise softmax = **target graph** $s_i$ | **Binary graph → soft similarity graph** (p.65). `tao_s`($\tau_s$)가 target을 얼마나 soft하게 할지 결정 (p.68) | `txt_emb.T`, `tao_s` |
| `img_sims = softmax(img_emb @ img_emb.T / tao, dim=1)` | image–image similarity → **learned graph** $p_i$ | batch 안 다른 샘플들과 비교하는 InfoNCE식 softmax 그대로 | `img_emb.T`, `tao` |
| `return CE(tgt_sims, img_sims, dim=1).mean()` | target graph ↔ learned graph의 **cross-entropy** | **InfoNCE의 one-hot label을 soft label로 교체**. "Match the learned graph to the target graph" | "CE", "cross-entropy" |

**Pseudo code ↔ core idea 한 줄**: InfoNCE의 one-hot 정답 행 대신 **frozen text encoder가 준 soft similarity 분포**(`tgt_sims`)를 정답으로 놓고, image similarity 분포(`img_sims`)가 그걸 CE로 따라가게 함 → "개는 피아노보다 고양이에 가깝다"를 학습.

- $\tau_s$ (p.68): **low** → strongest positive에 집중 → standard contrastive learning과 비슷 / **high** → related sample에 weight ↑ → target graph가 softer.
- **vs CLIP (L05 p.35)**: CLIP은 image–text **cross** similarity `np.dot(I_e, T_e.T) * np.exp(t)` + one-hot `labels = np.arange(n)`, text encoder도 같이 학습. X-CLR은 text–text(target)와 image–image(learned)를 **따로** 계산 → 두 modality를 공통 공간으로 맞출 필요 없음(`D_txt`와 `D_img`가 달라도 됨), text는 supervision 제공자일 뿐.

Implementation trivia (low priority): shape 주석 `N x D_txt`, `N x D_img`.

---

### BYOL (Grill et al., NeurIPS 2020) — L04 p.32 (step list)

**Transcription** (slide bullet 그대로)

- Create two augmented views: $\mathbf{v} = t(\mathbf{x});\ \mathbf{v}' = t'(\mathbf{x})$ with augmentations sampled $t \sim \mathcal{T}, t' \sim \mathcal{T}'$;
- Then they are encoded into representations, $\mathbf{y}_\theta = f_\theta(\mathbf{v}), \mathbf{y}' = f_\xi(\mathbf{v}')$;
- Then they are projected into latent variables, $\mathbf{z}_\theta = g_\theta(\mathbf{y}_\theta), \mathbf{z}' = g_\xi(\mathbf{y}')$;
- The online network outputs a prediction $q_\theta(\mathbf{z}_\theta)$;
- Both $q_\theta(\mathbf{z}_\theta)$ and $\mathbf{z}'$ are L2-normalized, giving us $\bar{q}_\theta(\mathbf{z}_\theta) = q_\theta(\mathbf{z}_\theta)/|q_\theta(\mathbf{z}_\theta)|$ and $\bar{\mathbf{z}}' = \mathbf{z}'/|\mathbf{z}'|$;
- The loss $\mathcal{L}^{\text{BYOL}}_\theta$ is MSE between L2-normalized prediction $\bar{q}_\theta(\mathbf{z})$ and $\bar{\mathbf{z}}'$;
- The other symmetric loss $\tilde{\mathcal{L}}^{\text{BYOL}}_\theta$ can be generated by switching $\mathbf{v}'$ and $\mathbf{v}$; that is, feeding $\mathbf{v}'$ to online network and $\mathbf{v}$ to target network.
- The final loss is $\mathcal{L}^{\text{BYOL}}_\theta + \tilde{\mathcal{L}}^{\text{BYOL}}_\theta$ and only parameters $\theta$ are optimized.

**BYOL does not use negative pairs.**
**BYOL directly boostrapps the latent representation.** ← "boostrapps" 오타 (bootstraps), 원문 유지

- ← $|\cdot|$은 L2 norm $\|\cdot\|_2$ 의미. loss 줄의 $\bar q_\theta(\mathbf z)$는 앞 줄 $\bar q_\theta(\mathbf z_\theta)$와 첨자 불일치 (원문 유지).
- p.32 step list엔 **stop-gradient와 EMA 식이 없음** → p.30 그림과 같이 볼 것: target 끝에 $\text{sg}(z')$, 오른쪽 $\theta\!:\ \xi \leftarrow \tau\xi + (1-\tau)\theta$ "Exponential Moving Average (EMA)". 여기 $\tau$는 **momentum**이지 temperature 아님. p.31: Target network = "Provides a stable learning target / Updated by EMA / No backpropagation".

**핵심 줄 (Key lines)**

| line | what it does | which core idea it implements | 빈칸 후보 |
|---|---|---|---|
| $\mathbf y_\theta = f_\theta(\mathbf v),\ \mathbf y' = f_\xi(\mathbf v')$ (+ $g$ 줄) | 같은 구조 두 network: **online θ** vs **target ξ** | "Two neural networks interact and learn from each other" (p.30) | $\xi$, "target" |
| The online network outputs a prediction $q_\theta(\mathbf z_\theta)$ | **online에만** predictor | **비대칭** — online이 target projection을 **예측** ("Learn by prediction", p.34) | "predictor", "online" |
| L2-normalized … MSE between $\bar q_\theta$ and $\bar{\mathbf z}'$ | 정규화된 prediction과 target projection의 MSE | **negative 없이** positive pair만 맞추는 **feature matching** (p.41 "MSE Loss") | "L2", "MSE" |
| switching $\mathbf v'$ and $\mathbf v$ | v′를 online, v를 target에 넣어 대칭 loss | 두 view가 번갈아 online/target 역할 | "symmetric" |
| only parameters $\theta$ are optimized (+ p.30 EMA) | ξ는 gradient로 안 바뀜, $\xi \leftarrow \tau\xi + (1-\tau)\theta$ | **stable target** = 과거 자기 자신을 target으로 bootstrapping | "θ", "EMA" |

**Pseudo code ↔ core idea 한 줄**: online(θ)이 **predictor**로 EMA target(ξ)의 정규화된 projection을 **MSE로 예측** — contrastive가 아니라 regression이라 negative가 필요 없음.

> 보충: 정규화 벡터 MSE는 $\|\bar a - \bar b\|_2^2 = 2 - 2\langle \bar a, \bar b\rangle = 2 - 2\cos(a,b)$ → SimCLR loss에서 분모(negative 항)를 뺀 꼴. downstream엔 online encoder $f_\theta$만 씀.

Implementation trivia (low priority): $\mathcal T$ vs $\mathcal T'$ — 두 view의 augmentation 분포를 다르게 둘 수 있음.

---

### DeepCluster (Caron et al., ECCV 2018) — L05 p.24 (Process)

**Transcription** (slide 그대로)

- **DeepCluster (2018)**
  - Cluster image features using the current representation.
  - Treat cluster assignments as pseudo labels.
  - Train the network to predict the assigned clusters.
  - Repeat clustering and representation learning iteratively.
- **Key Idea**: Alternate between clustering and classification.

Process
- Extract image representations.
- Cluster the representations using k-means.
- Assign pseudo labels from cluster IDs.
- Update the network using the pseudo labels.
- Repeat until convergence.

(L05 p.23: "Unlike contrastive learning, it does not create positive/negative pairs, but uses clustering results as pseudo labels." 그림: Input → Convnet → Clustering → **Pseudo-labels** → Classification, Classification에서 Convnet으로 **Backprop**.)

**핵심 줄 (Key lines)**

| line | what it does | which core idea it implements | 빈칸 후보 |
|---|---|---|---|
| Extract image representations. | **현재** convnet으로 데이터 feature 추출 | label의 원천이 network 자신 ("using the current representation") | "current representation" |
| Cluster the representations using k-means. | feature 공간에서 k-means | **clustering** step | "k-means" |
| Assign pseudo labels from cluster IDs. | cluster ID를 class label처럼 씀 | pos/neg pair 대신 **pseudo label** (p.23) | "cluster IDs", "pseudo labels" |
| Update the network using the pseudo labels. | 그 label을 맞히도록 classification 학습 (backprop) | **classification** step ("Train the network to predict the assigned clusters") | "pseudo labels" |
| Repeat until convergence. | 두 step을 번갈아 반복 | **Key Idea: Alternate between clustering and classification** | "alternate", "repeat" |

**Pseudo code ↔ core idea 한 줄**: k-means(clustering)로 pseudo label 만들고 → 그 label로 classification 학습 → 좋아진 feature로 다시 clustering… 를 **교대**로 반복 (EM 같은 alternating optimization).

- target에 gradient가 안 흐르는 이유: pseudo label은 k-means가 준 **이산 cluster ID**라 미분 경로 자체가 없음 → classification 단계에선 고정 정답.

> 보충 (논문, slide엔 없음): trivial solution 방지 — 빈 cluster는 다른 centroid를 perturb해서 재할당, 큰 cluster 쏠림 막으려 cluster 크기 기준 uniform sampling, cluster ID가 매번 뒤섞이니 마지막 classifier 재초기화.

Implementation trivia (low priority, slide 없음): 논문 k = 10,000, Sobel filter 입력.

---

## B. Cross-method comparison

| | Target (정답이 뭐냐) | Gradient 멈추는 곳 | EMA 되는 것 | Collapse 방지 | Loss | Page |
|---|---|---|---|---|---|---|
| **DINO** | EMA teacher의 **centered + sharpened softmax** $P_t$ (K차원 soft 분포) | `t = t.detach()` (teacher 출력) | **teacher params** (`l`) + **center `C`** (`m`) — 2개 | **centering + sharpening** (+ EMA teacher) | **CE** `-(t * log(s))` | L04 p.39–41 |
| **X-CLR** | **frozen text encoder**의 text–text similarity softmax $s_i$ (batch 샘플에 대한 soft 분포) | `txt_enc(captions).detach()` (text 쪽 전체) | 없음 (text encoder는 EMA가 아니라 **frozen**) | target이 image encoder와 무관하게 고정 + batch softmax의 contrastive 구조 유지 | **CE** `CE(tgt_sims, img_sims)` | L03 p.65–68 |
| **BYOL** | **target network projection** $\bar{\mathbf z}'$ (연속 벡터) | $\text{sg}(z')$ (target branch) | **target params ξ** ($\tau$) | **predictor (online만) + EMA target** + sg, negative 없음 | **L2-normalized MSE** ($2-2\cos$) | L04 p.30–34 |
| **DeepCluster** | **k-means cluster ID** (hard pseudo label) | 이산 ID라 미분 경로 없음 (고정 정답) | 없음 (EMA 아님, clustering을 반복 재계산) | slide 언급 없음 (보충: 빈 cluster 재할당, uniform sampling) | classification (**CE**) | L05 p.23–24 |
| *MoCo (대비용)* | momentum encoder의 **positive key** $k^+$ + queue negatives, one-hot `labels = zeros(N)` | `k = k.detach()` (key) | **key encoder** `f_k` (`m`) | **negatives** (queue) | InfoNCE = `CrossEntropyLoss(logits/t, labels)` | L05 p.11–12 |
| *Mean Teacher (대비용)* | EMA teacher의 **class prediction** (다른 perturbation) | teacher (no backprop) | **teacher params** ($\beta$) | labeled data의 supervised CE가 anchor | supervised CE + consistency ("distance between predictions"; 논문은 MSE) | L06 p.23–24 |

- **EMA 방향은 항상 "느린 쪽 ← 빠른 쪽"**: `gt.params = l*gt.params + (1-l)*gs.params` (DINO) / $\xi \leftarrow \tau\xi + (1-\tau)\theta$ (BYOL) / `f_k.params = m*f_k.params+(1-m)*f_q.params` (MoCo) / $\theta' \leftarrow \beta\theta' + (1-\beta)\theta$ (Mean Teacher). teacher/target/key가 student/online/query를 천천히 따라감, 반대 아님.
- **DINO만 EMA가 2개**: teacher parameter EMA + teacher **output** 평균(center)의 EMA. center는 parameter가 아니라 output 통계라는 점이 함정 포인트.
- **weight EMA vs prediction EMA**: DINO·BYOL·MoCo·Mean Teacher = **weight(parameter) EMA**. Temporal Ensembling(L06 p.20)만 sample별 **prediction EMA**.
- **detach 위치는 항상 target 쪽**: DINO = teacher 출력, X-CLR = text 임베딩, MoCo = key, BYOL = $z'$. 단 DINO teacher는 EMA로 계속 변하고, X-CLR text encoder는 frozen이라 **target graph가 학습 내내 고정**.
- **CE vs MSE**: target이 확률분포/label이면 **CE** (DINO, X-CLR, DeepCluster, MoCo), target이 feature 벡터면 **MSE** (BYOL). p.41: "BYOL Feature matching / MSE Loss ↔ DINO Distribution matching / Cross Entropy Loss".
- **soft vs hard target**: DINO(teacher softmax)·X-CLR(text graph) = **soft 분포**, DeepCluster(cluster ID)·MoCo(positive index 0) = **hard label**, BYOL = 연속 벡터.
- **같은 문자, 다른 뜻**: BYOL p.30의 $\tau$ = **EMA momentum** / DINO `tps`·`tpt`, X-CLR `tao`·`tao_s` = **softmax temperature**.
- **DINO vs Mean Teacher**: 둘 다 "teacher = student의 weight EMA". Mean Teacher는 label이 있어서 supervised loss가 collapse를 막아주지만, DINO는 label이 전혀 없어서 **centering + sharpening**이 따로 필요.

---

## C. Question pool (15문제: T/F 5 · MCQ 4 · Blank 6)

★ = killer (그래도 핵심 줄에 대한 문제, trivia 아님)

### Q1 [T/F]
In DINO's pseudo code, centering and sharpening are applied to the student output, while the teacher output goes through a plain softmax with temperature `tps`.
**Answer**: False — 반대. **teacher**: `t = softmax((t - C) / tpt, dim=1) # center + sharpen`, **student**: `s = softmax(s / tps, dim=1)`만. `tps` = student temperature, `tpt` = teacher temperature (헤더 주석). 그림에서도 centering 박스는 teacher 쪽에만 있음. (L04 p.40)

### Q2 [T/F]
In DINO's pseudo code, the teacher starts as a copy of the student (`gt.params = gs.params`), receives no gradient (`t.detach()`), and is afterwards updated only as an exponential moving average of the student's parameters.
**Answer**: True — SGD는 `update(gs) # SGD`로 student만. teacher는 `gt.params = l*gt.params + (1-l)*gs.params`로만 변함. Mean Teacher(L06 p.23)와 같은 weight-EMA teacher지만 DINO는 label을 안 씀. (L04 p.39–40)

### Q3 [T/F]
In X-CLR's `training_step`, the text encoder is frozen (its output is detached), and only the image encoder is trained so that its image–image similarity distribution matches the text–text similarity distribution.
**Answer**: True — `txt_emb = txt_enc(captions).detach()` (주석 "a frozen text encoder"), `img_emb = img_enc(images)` (주석 "the model we are training"). target graph = text similarity, learned graph = image similarity, "Match the learned graph to the target graph". CLIP처럼 두 encoder를 같이 학습하는 게 아님. (L03 p.66–67)

### Q4 [T/F]
In BYOL, the final loss $\mathcal{L}^{\text{BYOL}}_\theta + \tilde{\mathcal{L}}^{\text{BYOL}}_\theta$ is minimized with respect to both the online parameters $\theta$ and the target parameters $\xi$.
**Answer**: False — "only parameters $\theta$ are optimized" (p.32). target $\xi$는 $\xi \leftarrow \tau\xi + (1-\tau)\theta$ **EMA**로만 갱신, "No backpropagation" (p.30–31). $\tilde{\mathcal L}$은 v와 v′를 바꿔 넣은 대칭 loss일 뿐 ξ를 학습시키는 항이 아님. (L04 p.30–32)

### Q5 [T/F]
In DeepCluster, k-means clustering is performed once at the beginning of training, and the network is then trained on these fixed pseudo labels until convergence.
**Answer**: False — Process 마지막 줄 "Repeat until convergence", Key Idea "**Alternate** between clustering and classification". 매번 **현재** representation으로 다시 clustering → feature 개선 → 더 좋은 cluster → 더 좋은 pseudo label. (L05 p.24)

### Q6 [MCQ]
In DINO's pseudo code, which line updates the **teacher** network at every iteration of the training loop?
(A) `update(gs) # SGD`
(B) `gt.params = l*gt.params + (1-l)*gs.params`
(C) `C = m*C + (1-m)*cat([t1, t2]).mean(dim=0)`
(D) `gt.params = gs.params`
**Answer**: (B) — teacher = student의 EMA, `l` = network momentum (소문자 L, 숫자 1 아님). (A) SGD로 **student**만 갱신. (C) teacher parameter가 아니라 **center** 갱신 (teacher 출력 평균의 EMA). (D) 루프 **밖**에서 한 번만 하는 초기화 (teacher를 student 복사로 시작). (L04 p.40)

### Q7 [MCQ]
Which method does the following pseudo code come from?
```python
txt_emb = txt_enc(captions).detach()
img_emb = img_enc(images)
tgt_sims = softmax(txt_emb @ txt_emb.T / tao_s, dim=1)
img_sims = softmax(img_emb @ img_emb.T / tao, dim=1)
return CE(tgt_sims, img_sims, dim=1).mean()
```
(A) CLIP
(B) X-CLR
(C) DINO
(D) SupCon
**Answer**: (B) — 단서: text–text로 **target graph**, image–image로 **learned graph**, text는 detach(frozen), soft target과 CE. (A) CLIP은 image–text **cross** similarity `np.dot(I_e, T_e.T) * np.exp(t)` + one-hot `labels = np.arange(n)`, text encoder도 학습 (L05 p.35). (C) DINO는 student/teacher + `(t - C) / tpt` (L04 p.40). (D) SupCon은 같은 class = positive인 **binary** target (L03 p.65, L05 p.38–40). (L03 p.67)

### Q8 [MCQ] ★
In DINO's loop, the loss is `loss = H(t1, s2)/2 + H(t2, s1)/2`. Why is the teacher output of view 1 (`t1`) paired with the student output of view 2 (`s2`) rather than with `s1`?
(A) So that the student learns to match the teacher's output distribution for a *different* augmented view of the same image, making the representation invariant to augmentation.
(B) So that gradients can flow into the teacher network through `t1`.
(C) Because `s2` serves as a negative sample for `t1`.
(D) Because the center `C` is computed only from the student outputs.
**Answer**: (A) — student가 다른 view의 teacher 분포를 따라가야 view-invariant feature가 학습됨. 그림도 $x_1$→student, $x_2$→teacher, loss $-p_2\log p_1$. (B) `t = t.detach() # stop gradient` → teacher엔 gradient 없음. (C) DINO는 negative를 안 씀 — 같은 이미지의 두 view는 positive 관계. (D) center는 **teacher** 출력 `cat([t1, t2])`로 계산. (L04 p.40)

### Q9 [MCQ]
Which pairing of method and **learning target** is correct?
(A) DINO — the k-means cluster ID of each image
(B) X-CLR — a softmax over text–text similarities computed by a frozen text encoder
(C) BYOL — the teacher's centered and sharpened softmax distribution
(D) DeepCluster — the L2-normalized projection of an EMA target network
**Answer**: (B) — `tgt_sims = softmax(txt_emb @ txt_emb.T / tao_s, dim=1)` (L03 p.67). (A)는 **DeepCluster** (L05 p.24), (C)는 **DINO** (L04 p.40), (D)는 **BYOL** (L04 p.30–32). 이름만 섞어 놓은 함정.

### Q10 [Blank]
In DINO's pseudo code, the teacher output is processed as `t = softmax((t - C) / tpt, dim=1)`, and the comment on this line reads `# (1) ______ + (2) ______`: subtracting `C` does (1), and dividing by the low teacher temperature `tpt` does (2).
**Answer**: (1) center (accept: centering) (2) sharpen (accept: sharpening) — `(t - C)` = teacher logit에서 running mean을 빼는 **centering**, `/ tpt` = 낮은 temperature로 분포를 뾰족하게 하는 **sharpening**. 둘 다 **teacher** 쪽에만 있음. negative 없이 collapse를 막는 DINO의 핵심 장치. (L04 p.40)

### Q11 [Blank] ★
In DINO's loop, the center is updated as `C = m*C + (1-m)*cat([t1, t2]).mean(dim=0)`. So the center is an exponential moving average of the batch mean of the ______ network's outputs.
**Answer**: teacher (accept: `gt`, $g_{\theta_t}$) — `t1, t2` = teacher output (`# teacher output n-by-K`), 수식도 $g_{\theta_t}$. student 출력 `s1, s2`는 center에 안 들어감. center는 teacher 출력에서 빼는 값이니 teacher 통계로 만드는 게 자연스러움. 함정 두 개: "student"라고 쓰기, 그리고 center를 parameter EMA(`l` 줄)와 헷갈리기 — center는 **output** 평균의 EMA. (L04 p.40)

### Q12 [Blank]
In X-CLR's `training_step`, `tgt_sims` is a softmax over similarities between (1) ______ embeddings from a frozen encoder, `img_sims` is a softmax over image–image similarities, and the returned loss is the (2) ______ between these two distributions.
**Answer**: (1) text (accept: caption, `txt_emb`) (2) cross-entropy (accept: CE) — Target graph = "Similarity from text or metadata", Learned graph = "Similarity between image representations", 코드 주석 "the loss is the cross-entropy between the two". 핵심은 InfoNCE의 one-hot target을 **soft target**으로 바꾼 것. MSE 아님. (KL은 고정 target이면 CE와 gradient가 같지만 slide 표현은 cross-entropy.) (L03 p.67)

### Q13 [Blank]
In BYOL, the online prediction $q_\theta(\mathbf{z}_\theta)$ and the target projection $\mathbf{z}'$ are both (1) ______-normalized, and the loss $\mathcal{L}^{\text{BYOL}}_\theta$ is the (2) ______ between them; no negative pairs are used.
**Answer**: (1) L2 (2) MSE (accept: mean squared error, squared L2 distance) — p.32 step 5–6. 정규화 벡터 간 MSE = $2 - 2\cos$ → positive 쪽 유사도만 키움. CE가 아님: CE는 DINO ("MSE Loss vs Cross Entropy Loss", p.41). (L04 p.32)

### Q14 [Blank]
DeepCluster's process: extract image representations → cluster the representations using (1) ______ → assign pseudo labels from (2) ______ → update the network using the pseudo labels → repeat until convergence.
**Answer**: (1) k-means (2) cluster IDs (accept: cluster assignments) — pos/neg pair를 안 만들고 clustering 결과를 **pseudo label**로 씀 (p.23). Key Idea: "Alternate between clustering and classification". (L05 p.23–24)

### Q15 [Blank] ★
`.detach()` appears once in DINO's pseudo code and once in X-CLR's `training_step`. In both cases it is applied to the side that provides the learning target: the (1) ______ network's output in DINO, and the (2) ______ encoder's embeddings in X-CLR.
**Answer**: (1) teacher (2) text (accept: frozen text, caption) — 공통 원리: **target 쪽은 상수 취급**, 학습은 student / image encoder만. 차이: DINO teacher는 gradient 대신 **EMA로 계속 변함**, X-CLR text encoder는 **frozen**이라 target graph가 학습 내내 고정. 함정: "student"/"image"를 쓰면 정반대. (L04 p.40, L03 p.67)

<!-- WRITTEN 2026-10-22 | author: opus | PC2 -->

<!-- VERIFIED 2026-10-22 | checked: 20 | fixed: 3 | deleted: 0 | reviewer: opus -->
<!-- scope: 15 questions + 4 method blocks (transcription/flags/key-line table vs rendered L04 p.40, L03 p.67, L04 p.32, L05 p.24; cross-checked L04 p.30-31/39/41, L03 p.65-66/68, L05 p.23, L06 p.23-24) + section B. All transcriptions match; l-vs-1 flag in DINO confirmed at zoom (header "# l, m"); X-CLR diagonal-mask flag confirmed (only p_ij has 1_{k!=i}); all 15 answer keys correct. -->
<!-- Q6: "during training" let (D) gt.params = gs.params (initialization) be argued as a teacher update -> "at every iteration of the training loop". -->
<!-- Q11 (killer): stem quoted c <- mc + (1-m)(1/B) sum g_{theta_t}(x_i); the theta_t subscript gave away "teacher" -> formula removed, code line kept. -->
<!-- Q15 (killer): stem quoted `t = t.detach()` and `txt_enc(captions).detach()`, which gave away both blanks -> code removed; student must recall where detach sits. -->
