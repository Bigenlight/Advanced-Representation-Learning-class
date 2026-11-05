# Pseudo code · code card — 기말 범위 (W10–W14)

> **이 파일은 뭐야**: 기말 범위(Weakly-Supervised I·II, Active Learning I·II, MAE)의 코드 조각 모음이야. 강의가 끝날 때마다 블록을 이어 붙여.
> **왜 새 파일이야**: 기말이 누적인지 아직 몰라서, 중간고사용 `pseudocode.md`는 건드리지 않고 따로 만들었어. 누적으로 확정되면 합치면 돼.
> **L08 스캔 결과**: 41p 전수 스캔에서 **Algorithm·코드 박스 0개**야. 수식은 p.20, 21, 22, 28, 38에만 있어.
> **그래도 카드를 만든 이유**: 중간고사 코드 문제 4개 중 3개가 **슬라이드에 코드가 없는 방법**이었어 (`PLAYBOOK.md` 7장). 그래서 슬라이드의 pipeline 문장과 수식을 코드로 옮겼어.
> **코드는 전부 dummy tensor로 돌려 봤어.** `A`, `iou`, `train_detector` 같은 helper는 카드에서 뺐고, 핵심 줄만 남겼어.
> **참고**: `midterm_recall.md`가 weakly-supervised 후보로 꼽은 **CAM은 L08에 없어.** W11(Weakly-Supervised II)에서 확인해.

**연습 방법**: 조각을 보고 ① 빈칸을 채우고 ② 아래 순서로 영어 2–3문장을 직접 써 봐. 빈칸은 가볍고 무게는 설명에 있어.

1. 어떤 입력이 들어가나
2. target은 어디서 오나
3. 어떤 loss를 쓰나
4. 누가 update되나

**용어 먼저**

- **proposal / window**: 사진 안에서 "여기 물체가 있을지도 몰라" 하고 뽑은 후보 네모 칸이야. MIL에선 이게 instance야.
- **bag**: 후보 칸을 담은 사진 한 장이야. label은 bag 단위로만 있어.
- **IoU(Intersection over Union)**: 두 네모가 얼마나 겹치는지 나타내는 값이야. 겹친 넓이 ÷ 합친 넓이고, 0–1 사이야.

---

## L08 Weakly-Supervised Learning I — 한눈에 보기

| 카드 | 방법 (p.N) | 우선순위 | 꼭 기억할 핵심 줄 | 빈칸 | core idea 한 줄 |
|---|---|---|---|---|---|
| WS-1 | Standard MIL (p.16, 20–21) | **1** | `b_star = scores.argmax()` | `argmax` | 1등 window 하나를 positive로 정하고 다시 학습해 |
| WS-2 | WSDDN (p.34, 36) | **1** | `fc_det(x).softmax(dim=0)` | `0` | class 비교 × proposal 비교를 곱하고, 합해서 image label로 학습해 |
| WS-3 | OICR (p.36–37) | **1** | `top = s_prev[:, c].argmax()` | `no_grad` | 이전 stage 점수로 pseudo label을 만들어 다음 stage를 학습해 |
| WS-4 | Relaxing max (p.22) | 2 | `torch.logsumexp(s, dim=0)` | `logsumexp` | hard max를 부드럽게 해서 여러 proposal에 걸어 |
| WS-5 | Multifold MIL (p.24) | 2 | `(fold != k)` | `!=` | 이미지를 학습한 적 없는 detector로 re-localize해 |
| WS-6 | Objectness (p.28) | 2 | `lam * appearance + (1 - lam) * obj` | `1 - lam` | class와 무관한 "물체다움"을 점수에 섞어 |
| WS-7 | PCL (p.37) | 2 | `c = deg.argmax()` | `argmax` | 연결이 가장 많은 proposal을 cluster 중심으로 뽑아 |
| WS-8 | Ren et al. (p.38) | 2 | `theta += lr * g_th` | `+=` | detector는 loss를 줄이고 DropBlock은 loss를 키워 |

---

## 1순위: 계열 대표 방법

### WS-1. Standard MIL — re-localize + re-train

```python
scores = A(feats_pos).squeeze(1)        # A(x_b) for every proposal of a positive image
b_star = scores.______()                # re-localize
s_pos = A(feats_pos[b_star]).squeeze()  # only one positive instance per image
s_neg = A(feats_neg).squeeze(1).max()   # best window of a negative image
loss = F.relu(delta - s_pos) + F.relu(delta + s_neg)  # hinge with margin delta
loss.backward()                         # re-train
```

**Answer**: `argmax` (same meaning: `scores.argmax(dim=0)`, `torch.argmax(scores)`)

**Process (EN)**: This code implements one re-localize and re-train round of the standard MIL pipeline. It scores every proposal of a positive image with the appearance model A and keeps the top window as the only positive instance, then applies a hinge loss that pushes this window above Δ and the best window of a negative image below −Δ, and this loss updates A. The blank is `argmax`, which re-localizes the object by selecting the highest-scoring window.

**풀이**: 이 줄이 "이 사진의 물체는 여기야"라고 정해. 없으면 어떤 window가 positive인지 정해지지 않아서 학습할 instance-level target이 사라져.

- **왜 negative에도 max야?** 슬라이드 식이 $\max_b A(x_b) < -\Delta$야. 1등 window까지 $-\Delta$ 아래면 나머지도 전부 아래니까, "모든 window가 negative"라는 뜻이 돼.
- **다른 빈칸 후보**: `delta`(margin), `.max()`(negative 쪽), `delta - s_pos`의 부호.

**근거**: p.16 "Two step alternating optimization", p.20 $\arg\max_b A(x_b)$ · "Only one positive instance per image", p.21 hinge 식.

**단순화**: 슬라이드 코드가 아니라 식을 옮긴 거야. positive 사진 1장, negative 사진 1장만 넣었고, re-train을 gradient 한 번으로 줄였어. 실제로는 전체 데이터로 detector를 다시 학습해. p.21엔 cross-entropy 형태도 같이 있어.

### WS-2. WSDDN — two-stream head

```python
s_cls = fc_cls(x).softmax(dim=1)        # x: [R, D] region features -> [R, C]
s_det = fc_det(x).softmax(dim=______)   # localization stream
s = s_cls * s_det                       # proposal scores [R, C]
y = s.sum(dim=0)                        # image-level scores [C]
loss = F.binary_cross_entropy(y.clamp(0, 1), labels)
loss.backward()
```

**Answer**: `0` (accept: the proposal dimension, `-2`)

**Process (EN)**: This code implements the two-stream head of WSDDN. It takes the RoI-pooled region features, turns them into a softmax over classes and a softmax over proposals, multiplies the two into proposal scores, and sums over proposals to get image-level scores, then trains the whole network end to end with a binary cross-entropy against the image labels, the only target available. The blank is `0`, the proposal dimension, which makes the localization stream compare regions with each other for each class.

**풀이**: `dim=0`이 class마다 proposal끼리 경쟁시켜. `dim=1`로 바꾸면 두 stream이 둘 다 class만 비교해서, 어느 region이 그 class를 설명하는지 가리지 못해.

- **왜 sum 뒤에 softmax가 없어?** 한 사진에 class가 여러 개 있을 수 있어서야. 그래서 class마다 따로 binary loss를 걸어.
- **왜 $y$가 0–1 사이야?** `s_det`는 proposal 방향으로 합이 1이고 `s_cls`는 1 이하라서, `y`는 `s_cls`의 가중 평균이야. `clamp`는 반올림 오차만 막는 안전장치고 개념은 아니야.
- **다른 빈칸 후보**: `dim=1`(classification stream), `sum`, `dim=0`(sum 쪽), `*`.

**근거**: p.34 "Two stream architecture", "Classification: Associates each region with a class score", "Localization: Compares regions for each class". p.36 그림의 "Softmax over classes", "Softmax over proposals", "Element-wise product", "Sum over proposals".

**단순화**: 슬라이드엔 식이 없어서 구조는 p.36 그림에서, loss는 논문(Bilen & Vedaldi, CVPR 2016)에서 가져왔어. 논문 loss는 class별 binary log-loss의 **합**이고, 여기 `binary_cross_entropy`는 **평균**이라 상수배 차이가 나. 논문 label은 {−1, 1}이고 여기선 {0, 1}을 썼어 (같은 loss야). RoI pooling과 backbone은 뺐어.

### WS-3. OICR — one refinement stage

```python
with torch.______():                    # build the pseudo label
    top = s_prev[:, c].argmax()         # highest scoring proposal of class c
    near = iou(boxes, boxes[top:top + 1]).squeeze(1) > t  # plus surrounding ones
target = torch.full((R,), C)            # C = background, the (C+1)-th class
target[near] = c
loss = F.cross_entropy(head_k(x), target)  # stage-k instance classifier
loss.backward()
```

**Answer**: `no_grad` (accept: `inference_mode`)

**Process (EN)**: This code implements one refinement stage of OICR. For a class c in the image, it labels the previous stage's top-scoring proposal and the proposals overlapping it as class c and every other proposal as background, then applies cross-entropy to the stage-k classifier, whose gradient also reaches the shared proposal features. The blank is `no_grad`, which treats these pseudo labels as a fixed target, so no gradient flows back through them.

**풀이**: 이 블록은 "target은 고정이고, gradient는 `head_k(x)` 쪽으로만 흘러"를 코드로 드러내. 다만 이 조각에서 빼도 결과는 같아. label이 `argmax`와 IoU 비교로 만들어져서 원래 gradient가 안 흐르거든.

- **target은 어디서 와?** 어느 class인지는 **image-level label**이, 어느 proposal인지는 **이전 stage** 점수가 정해. WSDDN → pseudo label → instance classifier → 더 나은 pseudo label로 이어져 (p.36).
- **누가 update돼?** `head_k`와 공유 proposal feature야. 이전 stage가 멈추는 건 아니야. 실제로는 모든 stage의 loss를 더해서 한 network로 같이 학습해 ($L = L_b + \sum_k L_r^k$, 논문 Eq. 5). 그래서 "Online"이고, Standard MIL의 alternating과 달라.
- **positive 사진 안에 negative가 생겨.** IoU가 threshold 이하인 proposal은 background가 돼. WSDDN엔 없던 신호야 (p.35 한계 참고).
- **다른 빈칸 후보**: `argmax`, `t`(IoU threshold), `C`(background label), `s_prev`.

**근거**: p.36 "Initially uses WSDDN to set instance labels", "Iteratively refines them by using an instance classifier". p.37 "highest scoring proposal + surrounding ones as positives".

**단순화**:
- `no_grad`는 **우리가 넣은 표시**야. OICR 논문은 detach를 말하지 않아. 후속 PCL 논문은 supervision이 "only computed in the forward process"이고 gradient를 계산하지 않는다고 적어 (Sec. III-C).
- 원래는 사진에 있는 class마다 반복하는데, 여기선 class `c` 하나만 썼어.
- background는 논문처럼 마지막 index `C`((C+1)번째 class)로 뒀어. 그래야 `s_prev[:, c]`가 `[R, C]`인 WSDDN 점수와 `[R, C+1]`인 refinement 점수에서 같은 class를 가리켜.
- IoU threshold와 background 규칙은 슬라이드가 아니라 논문(Tang et al., CVPR 2017)에서 왔어. 논문의 loss 가중치는 뺐어.

---

## 2순위: 같은 계열의 변형

### WS-4. Relaxing max — log-sum-exp

```python
def relaxed_max(s):                     # s: scores of all proposals of one image
    return torch.______(s, dim=0)
s_pos = A(feats_pos).squeeze(1)
s_neg = A(feats_neg).squeeze(1)
loss = F.relu(delta - relaxed_max(s_pos)) + F.relu(delta + relaxed_max(s_neg))
loss.backward()
```

**Answer**: `logsumexp`

**Process (EN)**: This code implements the relaxed max of MIL training. It scores all proposals of a positive and a negative image, replaces the hard max with log-sum-exp, then applies the same hinge loss, which updates A. The blank is `logsumexp`, which spreads the gradient over several high-scoring proposals instead of sending it all to one window.

**풀이**: 한 window에 올인하지 않고 여러 proposal에 나눠 걸어 ("Hedge your bets"). hard max로 돌아가면 첫 선택이 틀렸을 때 그 window만 계속 학습해서 초기값에 민감해져.

**근거**: p.22 $\log\sum_b \exp A(x_b^+) > \Delta$, $\log\sum_b \exp A(x_b^-) < -\Delta$, "Less sensitive to initialization", "Leverages multiple positive samples during training".

**단순화**: 슬라이드 식을 코드로 옮긴 거고, `relaxed_max`라는 이름은 우리가 붙였어.

### WS-5. Multifold MIL

```python
for k in range(K):
    train_ids = (fold ______ k).nonzero().squeeze(1)              # positives
    det = train_detector(feats[train_ids, box[train_ids]], negs)  # + all negatives
    for i in (fold == k).nonzero().squeeze(1):                    # held-out fold
        box[i] = det(feats[i]).squeeze(1).argmax()                # re-localize
```

**Answer**: `!=` (같은 뜻: `fold.ne(k)`)

**Process (EN)**: This code implements the training loop of Multifold MIL. For each fold k, it trains a detector on the current positive windows of all folds but k and on all negative examples, then re-localizes the images of fold k with that detector. The blank is `!=`, which keeps fold k out of training, so each image is localized by a detector that has never seen it.

**풀이**: fold k를 학습에서 빼는 줄이야. 빼지 않으면 detector가 자기가 외운 box를 다시 고르는 **locking**이 그대로 남아.

- **나뉘는 건 positive뿐이야.** negative는 매번 전부 써. "all folds but k, and all negative examples"라는 문장이 함정 포인트야.

**근거**: p.24 "Train using positive examples in all folds but k, and all negative examples", "localize each image with a detector that has never seen that image during training". ← slide 크레딧은 [Sangineto PAMI 17]인데, Multifold MIL은 실제로 Cinbis et al., TPAMI 2017이야.

**단순화**: `train_detector`는 작은 linear classifier로 대신했어. 실제로는 이 loop 전체를 여러 번 반복하고, 매번 fold를 새로 나눠. fold loop 뒤엔 window 전체로 detector를 다시 학습하고 hard-negative mining을 해 (논문 Algorithm 1의 2(c), 2(d)). 여기선 `box`를 바로 덮어써서 뒤 fold 학습에 앞 fold의 새 window가 들어가. 논문은 이 순서를 못박지 않았어.

### WS-6. Objectness — class-independent prior

```python
appearance = A(feats).squeeze(1)        # A(x_b): class-specific score
obj = objectness(boxes)                 # Obj(b): class-independent score
score = lam * appearance + (______) * obj
b_star = score.argmax()                 # re-localize with both cues
```

**Answer**: `1 - lam`

**Process (EN)**: This code implements re-localization with an objectness cue. It mixes the class-specific appearance score A(x_b) with the class-independent objectness score Obj(b), then picks the proposal with the highest combined score as the new positive window. The blank is `1 - lam`, the weight of objectness, which steers the choice toward whole objects and away from background and object parts.

**풀이**: 두 점수의 비중을 나누는 줄이야. objectness 항이 없으면 appearance만 남아서, 배경이나 물체 일부(discriminative part) 쪽으로 쏠리기 쉬워.

**근거**: p.28 $\arg\max_b\ \lambda A(x_b) + (1-\lambda)\,\mathrm{Obj}(b)$, "Steers re-localization towards objects and away from background", "Pushes towards whole objects instead of sub-regions".

**단순화**: `objectness`는 dummy 함수야. 슬라이드는 EdgeBoxes, Selective Search, BING을 예로 들어.

### WS-7. PCL — cluster centres

```python
adj = iou(top_boxes, top_boxes) > t     # undirected graph of top-ranking proposals
alive = torch.ones(len(top_boxes), dtype=torch.bool)
centres = []
while alive.any():
    deg = (adj & alive[None, :]).sum(1) * alive  # connections among remaining ones
    c = deg.______()                    # next cluster centre
    centres.append(c.item())
    alive &= ~adj[c]                    # remove the centre and its neighbours
```

**Answer**: `argmax` (accept: `argmax(dim=0)`)

**Process (EN)**: This code implements the cluster-centre selection of PCL. It builds an undirected graph over the top-ranking proposals, then repeatedly picks the vertex with the most connections as a cluster centre and removes it with its neighbours, so one image can yield several centres, each leading to one pseudo object. The blank is `argmax`, which selects the most connected vertex.

**풀이**: 연결이 가장 많은 proposal을 중심으로 뽑는 줄이야. 이걸 반복해서 중심이 여러 개 나오니까, 사진 안의 물체 여러 개를 따로 잡을 수 있어. OICR처럼 pseudo object 하나로 끝나지 않아.

**근거**: p.37 "builds an undirected graph of top- ranking proposals", "iteratively selects vertices which have most connections to be the cluster centres", "each cluster as an independent pseudo object".

**단순화**: 슬라이드엔 edge 기준과 제거 단계가 없어. `IoU > t`로 연결하는 규칙(Eq. 4)과 중심·이웃을 지우는 줄(Algorithm 3)은 논문(Tang et al., PAMI 2018)에서 왔어. 논문은 사진에 있는 class마다 graph를 따로 만들고, top-ranking proposal은 점수를 k-means로 나눠서 뽑아. cluster는 graph 이웃이 아니라, 모든 proposal을 가장 많이 겹치는 중심에 붙여서 따로 만들어 (Algorithm 4). 이 뒷부분과 loss는 뺐어.

### WS-8. Ren et al. — adversarial update

```python
loss = L_img(w, theta) + L_roi(w, theta)
g_w, g_th = torch.autograd.grad(loss, [w, theta])
with torch.no_grad():
    w -= lr * g_w                       # detector
    theta ______ lr * g_th              # Concrete DropBlock
```

**Answer**: `+=`

**Process (EN)**: This code implements the adversarial update of Ren et al. It computes one loss from the image-level and ROI-level terms, takes a gradient step that lowers the loss for the detector w, then a step that raises the loss for the Concrete DropBlock θ. The blank is `+=`, gradient ascent on θ, which makes the module learn to drop the most discriminative regions so that the detector must use the whole object.

**풀이**: θ를 loss가 **커지는** 쪽으로 움직이는 줄이야. `-=`로 바꾸면 DropBlock도 detector를 돕는 쪽이 돼서, 가장 눈에 띄는 부분을 가리는 방해꾼이 사라져.

**근거**: p.38 $w^*, \theta^* = \arg\min_w \max_\theta \sum_I \mathcal{L}_{\text{img}}(w,\theta) + \mathcal{L}_{\text{roi}}(w,\theta)$, "Adversarial optimization", "drop the most relevant regions". ← slide의 Σ 아래 첨자는 잘려 있어서 $I$로 읽었어.

**단순화**: 슬라이드엔 $w$, $\theta$ 정의가 없어. $w$ = detector parameter, $\theta$ = Concrete DropBlock parameter는 논문(Ren et al., CVPR 2020)에서 확인했어. `L_img`, `L_roi`는 테스트에선 toy 함수고, optimizer 대신 손으로 gradient step을 적었어. 논문은 update 순서를 적지 않았어. 공식 코드는 $w$를 먼저 update하고 forward를 다시 돌린 뒤 `-weight * loss`로 $\theta$를 update해. 여기선 gradient를 한 번만 구한 동시 step으로 줄였어.

---

## 방법 간 비교

### `max` vs `logsumexp` vs `softmax(dim=0)`

"proposal 여러 개를 어떻게 하나로 합치나?"가 이 강의 코드의 공통 질문이야.

| 연산 | 모양 변화 | gradient가 가는 곳 | 어디서 |
|---|---|---|---|
| `s.max()`, `s.argmax()` | `[B]` → 값 1개 / index 1개 | 1등 window 하나 (`argmax`는 index라서 gradient 없음) | Standard MIL (p.20–21) |
| `torch.logsumexp(s, dim=0)` | `[B]` → 값 1개 | 점수의 softmax 비율로 여러 window | Relaxing max (p.22) |
| `s_cls * s.softmax(dim=0)` 후 `sum(dim=0)` | `[R, C]` → `[R, C]` → `[C]` | 모든 proposal | WSDDN (p.34, 36) |

설명용 숫자로 보면, 점수가 `A = [3, 2.9, -1]`일 때 gradient가 이렇게 달라. 슬라이드 수치가 아니라 직접 돌린 값이야.

- hard max: `[1, 0, 0]`
- log-sum-exp: `[0.52, 0.47, 0.01]` (값은 3.65, 상한 $\max + \log 3$ = 4.10)

### softmax는 어느 dim으로?

`x`가 `[R, C]`(행 = proposal, 열 = class)일 때야.

| stream | 무엇을 비교하나 | 슬라이드 표현 | dim |
|---|---|---|---|
| classification | region 하나 안에서 class끼리 | "Associates each region with a class score" | `dim=1` |
| localization | class 하나 안에서 region끼리 | "Compares regions for each class" | `dim=0` |

**번호를 외우지 말고 shape를 봐.** p.34 그림과 논문은 행 = class, 열 = region(`C × |R|`)이라, 그 배치면 번호가 뒤집혀.

---

## 함정 Top 5

1. **WSDDN softmax 축**: classification = class 방향, localization = proposal 방향이야. 뒤바꾼 문장이 나오기 쉽고, 코드에선 tensor shape에 따라 번호가 달라져.
2. **`logsumexp` ≠ `softmax`**: p.22의 "soft-max"는 log-sum-exp야. 값 하나를 내는 **max의 대체**고, softmax는 그 **gradient**에서 나와.
3. **Multifold의 "all but k"**: positive만 fold로 나눠. negative는 전부 쓰고, re-localize는 빠진 fold k에만 해.
4. **OICR의 target**: instance label을 사람이 붙이지 않고, 같은 stage의 출력도 아니야. class는 **image-level label**에서, proposal은 **이전 stage**의 점수에서 와. IoU가 threshold 이하인 proposal은 background라서, positive 사진 안에서도 negative가 생겨.
5. **min–max 방향**: $w$는 **minimize**, $\theta$는 **maximize**야. Concrete DropBlock은 **가장 relevant한** 부분을 가려.
