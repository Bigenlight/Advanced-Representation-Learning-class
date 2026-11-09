# Pseudo code · code card — 기말 범위 (W10–W14)

> **이 파일은 뭐야**: 기말 범위(Weakly-Supervised I·II, Active Learning I·II, MAE)의 코드 조각 모음이야. 강의가 끝날 때마다 블록을 이어 붙여.
> **왜 새 파일이야**: 기말이 누적인지 아직 몰라서, 중간고사용 `pseudocode.md`는 건드리지 않고 따로 만들었어. 누적으로 확정되면 합치면 돼.
> **L08 스캔 결과**: 41p 전수 스캔에서 **Algorithm·코드 박스 0개**야. 수식은 p.20, 21, 22, 28, 38에만 있어.
> **그래도 카드를 만든 이유**: 중간고사 코드 문제 4개 중 3개가 **슬라이드에 코드가 없는 방법**이었어 (`PLAYBOOK.md` 7장). 그래서 슬라이드의 pipeline 문장과 수식을 코드로 옮겼어.
> **코드는 전부 dummy tensor로 돌려 봤어.** `A`, `iou`, `train_detector` 같은 helper는 카드에서 뺐고, 핵심 줄만 남겼어.
> **L09 스캔 결과**: 96p 전수 스캔에서도 **Algorithm·코드 박스 0개**야. 수식은 전부 그림 안에 있어. 카드는 WS-9–WS-19 (11장, 1순위 4장)이고 이 파일 뒤쪽 `## L09` 블록에 있어.
> **참고**: `midterm_recall.md`가 weakly-supervised 후보로 꼽은 **CAM 계산은 L09의 WS-9**야 (L08엔 CAM이 없어).

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

---

## L09 Weakly-Supervised Learning II — 한눈에 보기

> **L09 스캔 결과**: 96p 전수 스캔에서 **Algorithm·코드 박스 0개**야. 수식은 전부 그림 안에 있어서 글자로는 안 잡혀. 수식이 있는 쪽: p.21–22 ($\partial S_c/\partial I$), p.27 (CAM weighted averaging, $>\tau_{loc}$), p.67, 69, 70, 72 (CD-CLIP), p.81–83 (web learning), p.87–89, 91–92 (WISH). p.36·p.39 그림엔 loss 이름과 $I_t, H_t, F_t$ 같은 기호만 있어.
> **고른 기준**: 1순위는 슬라이드 목차 계열마다 대표 하나야. gray box(WS-10), white box CAM(WS-9), CAM 넓히기(WS-11), boxly(WS-12). 2순위는 같은 계열의 변형, webly, 넓히기 논문 중 수식이 있는 두 편이야.
> **이어 돌린 카드**: WS-9(CAM) → WS-13(seed) → WS-14(SEC)는 앞 카드 출력을 뒤 카드 입력으로 넣어서 같이 돌렸어. WS-11(AE)은 3 step, WS-12(Boxly)는 3 round를 이어서 돌렸어 (L08 OICR 교훈).
> **CAM은 여기서 처음 나와.** `midterm_recall.md`가 weakly-supervised 후보로 꼽은 "CAM 계산"이 WS-9야.

| 카드 | 방법 (p.N) | 우선순위 | 꼭 기억할 핵심 줄 | 빈칸 | core idea 한 줄 |
|---|---|---|---|---|---|
| WS-9 | CAM (p.24–27) | **1** | `torch.einsum('k,khw->hw', fc.weight[c], feat)` | `weight` | 분류에 쓰던 FC weight로 feature map을 가중합하면 class 위치 지도가 나와 |
| WS-10 | Gradient saliency · SmoothGrad (p.21–22) | **1** | `torch.stack(grads).mean(dim=0)` | `mean` | noise를 넣고 구한 입력 gradient를 평균해서 잡음을 줄여 |
| WS-11 | Adversarial Erasing (p.39–41) | **1** | `F_t = heat > tau` → `x[:, F_t] = mean_pixel` | `>` | 가장 센 부분을 지우고 다시 학습해서 다른 부분을 찾게 해 |
| WS-12 | Boxly: recursive training (p.55–59) | **1** | `label = pred` | `pred` | 채운 box로 시작하고, network 예측을 다음 round의 label로 써 |
| WS-13 | Seeds (p.27) | 2 | `seed = torch.full_like(cls, 255)` | `255` | 확실한 곳만 seed로 쓰고 애매한 중간은 비워 둬 |
| WS-14 | SEC (p.30–32) | 2 | `f.flatten(1).sort(dim=1, descending=True)` | `sort` | seed로 시작해서, GWRP로 image label까지 넓히고, CRF로 경계에 맞춰 |
| WS-15 | SEAM (p.35–36) | 2 | `(A(cam_o) - cam_t).abs().mean()` | `A` | 입력을 변환하면 CAM도 똑같이 변환되게 맞춰 |
| WS-16 | Hide-and-Seek (p.37–38) | 2 | `torch.rand(B, 1, S, S) > p_hide` | `rand` | 학습 때 patch를 무작위로 가려서 다른 부분도 보게 해 |
| WS-17 | Webly: Model-T filter (p.44–47) | 2 | `conf = p[torch.arange(N), query]` | `query` | 검색어 label을 깨끗한 data로 학습한 Model-T로 걸러 |
| WS-18 | CD-CLIP: CDA (p.65–69) | 2 | `S_d = (1 - D_js) / tau` | `1` | class 분포가 비슷한 patch끼리 affinity를 살려서 CAM을 퍼뜨려 |
| WS-19 | WISH: matching cost (p.88–90, 94) | 2 | `.min(dim=-1).values` | `min` | class · prompt · mask cost를 더해서 Hungarian으로 짝지어 |

---

## L09 1순위: 계열 대표 방법

### WS-9. CAM — GAP classifier + class activation map

```python
feat = backbone(img)                     # F: [K, h, w], last conv feature maps
logits = fc(feat.mean(dim=(1, 2)))       # GAP -> FC -> class scores [C]
loss = F.multilabel_soft_margin_loss(logits[None], labels[None])  # image labels only
loss.backward()                          # train backbone + fc as a classifier
cam = torch.einsum('k,khw->hw', fc.______[c], feat)  # M_c = sum_k w_{k,c} F_k
cam = F.relu(cam) / F.relu(cam).max()    # [h, w], scaled to 0..1
cam = F.interpolate(cam[None, None], size=img.shape[-2:], mode='bilinear')[0, 0]  # resize
fg = cam > tau                           # B_c: confident foreground
```

**Answer**: `weight` (`fc.weight[c]` = class c의 FC weight 벡터 $w_{\cdot,c}$, shape `[K]`)

**Process (EN)**: This code trains a classifier with global average pooling on image-level labels and then reads out the class activation map of class c. The feature maps are averaged by GAP and scored by the FC layer, and a multi-label loss against the image labels, the only target, updates the backbone and the FC layer; afterwards the FC weights of class c re-weight the K feature maps, and the map is resized and thresholded into a foreground mask. The blank is `weight`, because the CAM is the weighted sum of the last feature maps with the FC weights of class c, so no extra gradient computation is needed.

**풀이**: 분류에 쓰던 weight를 위치 지도에 그대로 다시 써. $M_c(x,y)=\sum_k w_{k,c}\,F_k(x,y)$야. GAP와 FC가 둘 다 선형이라 순서를 바꿔도 돼서, class 점수 = CAM의 평균 + bias야 (직접 돌려서 확인). 그래서 CAM 값이 큰 곳 = class 점수를 많이 올린 곳이야.

- **gradient가 필요 없어.** CAM은 forward에서 나온 feature map과 weight만 써 ("without the need for separate gradient calculations", p.25). backward가 필요한 건 gray box(WS-10)야.
- **p.24의 1×1 conv 관점**: 슬라이드는 GAP **다음에** 1×1 conv(= FC)를 붙여. 둘 다 선형이라 1×1 conv를 GAP 앞으로 옮겨도 값이 같아. 그때 conv 출력 `[C, h, w]`가 class별 CAM이고 그 평균이 class 점수야 (직접 돌려서 확인. 순서 바꾸기는 슬라이드에 없는 보충).
- **해상도가 거칠어.** CAM은 마지막 conv 해상도(p.29 그림의 14×14)라서 resize한 뒤 threshold해 (p.27 M3). "Resulting masks are not sharp"(p.28)가 여기서 나와.
- **누가 update돼?** target은 image label뿐이고, update되는 건 backbone과 `fc`야. CAM 계산 줄은 학습된 weight를 읽기만 해.
- **다른 빈칸 후보**: `mean`(GAP), `dim=(1, 2)`, `c`, `tau`($\tau_{loc}$), `relu`.

**근거**: p.24 "Step 1) eliminate the FC layer", "Step 2) add GAP (Global Average Pooling)", "Step 3) add a 1×1 convolution that performs a linear combination of the features". p.25 "since the network itself is designed to preserve spatial information, activation maps can be obtained directly without the need for separate gradient calculations", 그림 $w_1\cdot + w_2\cdot + \dots + w_n\cdot = $ Class Activation Map. p.27 그림 "M1: Global Average Pooling", "M2: Weighted Averaging" ($w_{1,k}F_1 + \dots + w_{C,k}F_C = M_k$), "M3: Thresholding" (resize → $>\tau_{loc}$ → $B_k$).

**단순화**:
- 슬라이드 코드가 아니라 p.24–27 그림을 옮긴 거야.
- **기호 주의**: p.27 그림은 $k$가 class, $C$가 channel 수야. 이 카드는 `K` = channel 수, `c` = class로 바꿔 썼어.
- 원 논문(Zhou et al., CVPR 2016)은 ImageNet 단일 label이라 softmax를 써. WSSS는 사진에 class가 여러 개라 multi-label loss로 바꿨어.
- `relu` 후 최댓값으로 나누는 정규화는 흔한 구현이고 슬라이드엔 없어. optimizer step은 뺐어.
- `relu(cam).max()`가 0이면 0으로 나누니까 실제 구현은 작은 `eps`를 더해. CAM 읽는 줄은 학습이 끝난 뒤 `torch.no_grad()` 안에서 해.

### WS-10. Gradient saliency · SmoothGrad — gray box

```python
grads = []
for _ in range(n):                                  # p.22: repeat 50 times
    x = (img + sigma * torch.randn_like(img)).requires_grad_()  # add small random noise
    score = classifier(x[None])[0, c]               # S_c
    g, = torch.autograd.grad(score, x)              # dS_c / dI: [3, H, W]
    grads.append(g)
sal = torch.stack(grads).______(dim=0).abs().amax(dim=0)  # combine the n gradients, then |.| max over RGB
```

**Answer**: `mean` (accept: `sum`, 지도 모양은 같고 크기만 n배야. `max`·`median`은 안 돼. 슬라이드 단어가 "Average"야)

**Process (EN)**: This code computes a SmoothGrad saliency map for class c with an already trained classifier. It adds small random noise to the image n times, takes the gradient of the class score S_c with respect to each noisy input, and averages these gradients before taking the absolute value and the maximum over color channels; there is no label target and no loss, and no weight is updated because the gradient goes to the input, not to the network. The blank is `mean`, the averaging step that cancels the pixel-level noise of single gradients and gives a cleaner saliency map.

**풀이**: gradient 하나는 pixel 단위 민감도라 "salt-and-pepper"처럼 지저분해 (p.21). noise를 넣은 여러 장의 gradient를 평균하면 우연한 잡음은 상쇄되고 일관된 부분만 남아 (p.22).

- **`n = 1`, `sigma = 0`이면 Simonyan의 vanilla gradient saliency(p.21)야.** 직접 돌려서 같은 값인 걸 확인했어. SmoothGrad는 그 위에 "noise + 평균"만 더한 거야.
- **누가 update돼?** 아무것도 안 돼. `torch.autograd.grad(score, x)`는 입력 `x`에 대한 gradient만 구하고 weight엔 손대지 않아. classifier는 Step 1에서 이미 학습된 거야.
- **gray box의 한계는 그대로야.** 분류기를 억지로 localization에 쓰는 거라 "classification에 필요한 부분만" 밝혀 (p.23 "Asking apples from the oranges tree"). 평균으로 잡음은 줄어도 이 한계는 안 없어져.
- **다른 빈칸 후보**: `randn_like`(noise), `autograd.grad`, `abs`, `amax`(RGB 중 최대), `requires_grad_`.

**근거**: p.21 "Step 2) Calculate how much the score changes when the input is slightly modified", $\frac{\partial S_c}{\partial I}\big|_{I_0}$, "it is extremely noisy", "salt-and-pepper". p.22 "Step 2) Add small random noise to the images", "Step 3) Calculate the gradient for each", "Step 4) Average", 그림 "Step 4: Repeat 50 times step 2&3, average result", "calculating the gradient multiple times and averaging it removes noise".

**단순화**: `abs().amax(dim=0)`(RGB 채널 중 절댓값 최대)는 Simonyan et al.(2014)의 saliency map 정의를 따랐어. 슬라이드엔 없어. 절댓값을 평균 전에 할지 뒤에 할지는 구현마다 달라. noise 크기 `sigma`도 슬라이드엔 없어.

### WS-11. Adversarial Erasing — erase, retrain, repeat

```python
mined = torch.zeros(H, W, dtype=torch.bool)    # union of erased regions
x = img.clone()                                # I_1
for t in range(T):                             # T = 3 steps in the p.39 figure
    net = train_classifier(x, labels)          # retrain on the erased image
    heat = cam(net, x, c)                      # H_t: [H, W], scaled to 0..1
    F_t = heat ______ tau                      # F_t: most discriminative region
    mined |= F_t                               # accumulate the mined object region
    x[:, F_t] = mean_pixel                     # erase it: I_{t+1}
```

**Answer**: `>` (같은 뜻: `>=`, `heat.gt(tau)`)

**Process (EN)**: This code implements the object-region mining loop of adversarial erasing. At each step a classification network is retrained on the current (erased) image with a classification loss against the image-level label, the only target, and only this network is updated; then the heatmap of class c is thresholded into the most discriminative region, which is added to the mined region and erased from the image with the mean pixel value, so the next network must find other object parts. The blank is `>`, because the region to erase is where the heatmap is high, not low.

**풀이**: CAM이 가장 센 곳(예: 강아지 얼굴)을 지우면 분류기는 남은 부분(몸통, 다리)으로 맞혀야 해. 그 부분이 다음 heatmap에서 켜지고, 그걸 또 지워. 지운 영역을 모두 합친 `mined`가 물체 전체에 가까운 영역이 돼 (p.39 "The Mined Object Region").

- **target은?** step마다 분류 target은 image label 하나야. pixel target은 없고, 지운 영역의 합집합이 결과물이야.
- **이어 돌린 결과**: 설명용 toy 사진(물체 120 pixel, 가운데가 특히 밝음)으로 3 step 돌리면 누적 mined pixel이 10 → 84 → 106이야. 한 번 지운 pixel은 끝까지 지워진 채로 남아.
- **Hide-and-Seek(WS-16)와 차이**: HaS는 **무작위** patch를 가리고, AE는 **CAM이 높은 곳**을 골라 지워 ("adversarial instead of random erasures", p.39). HaS는 network 하나를 학습하면서 매 iteration 새로 가리고, AE는 지우고 다시 학습하는 step을 반복해.
- **너무 많이 돌리면?** 이 코드에선 heat를 최댓값으로 나누니까, 물체를 다 지운 뒤에도 (heat가 전부 0이 아니면) 어딘가는 `tau`를 넘어. 그러면 배경을 지우기 시작해. 그래서 step 수를 정해 둬 (슬라이드 그림은 t = 1, 2, 3).
- **다른 빈칸 후보**: `train_classifier`(retrain), `|=`(합집합), `mean_pixel`, `tau`.

**근거**: p.39 "Let us play adversarial peek-a-boo!", 그림 "Object Regions Mining with Adversarial Erasing" ($H_t$ → threshold → $F_t$, $I_t \ominus F_t \to I_{t+1}$, "Recurrent", t = 1, 2, 3, "The Mined Object Region"), "(adversarial instead of random erasures)". p.40 Figure 6 "AE-Step1–3", "Erased Regions", "the discriminative regions are highlighted". ← slide 크레딧은 [Wei et al. 2017]이고, 논문은 Wei et al., CVPR 2017 "Object Region Mining with Adversarial Erasing: A Simple Classification to Semantic Segmentation Approach"야.

**단순화**:
- 슬라이드 그림을 코드로 옮긴 거야. 실제로는 사진 한 장이 아니라 학습 data 전체로 step마다 classification network를 다시 학습해.
- 지운 자리를 mean pixel 값으로 채우는 건 논문 방식이야. `mean_pixel`은 `[3, 1]`(RGB 평균)이야.
- 논문은 mined region으로 segmentation label을 만들어서 segmentation network를 따로 학습해 (p.41 표의 AE-PSL). 그 뒷부분은 뺐어.
- `train_classifier`, `cam`은 helper야. `cam`은 WS-9의 CAM 계산이야.

### WS-12. Boxly — recursive training from boxes

```python
label = box_lbl.clone()                  # round 0: filled boxes, [B, H, W], 0 = background
for r in range(R):
    net = train_segnet(imgs, label)      # learn from the noisy masks
    with torch.no_grad():
        pred = net(imgs).argmax(dim=1)   # [B, H, W] predicted classes
    pred[box_lbl == 0] = 0               # outside every box -> background
    label = ______                       # next round's training label
```

**Answer**: `pred` (같은 뜻: `pred.clone()`, `pred.detach()`)

**Process (EN)**: This code implements recursive training for box-supervised segmentation. The first labels are the boxes filled with their class, a noisy mask; in every round a segmentation network is trained with a per-pixel loss on the current labels, its predictions become the next labels, and pixels outside every box are reset to background. The blank is `pred`, the network's own prediction, which replaces the noisy label in the next round because convnets are robust to noisy supervision.

**풀이**: 첫 label은 box를 통째로 칠한 거라 box 안 배경도 물체로 잘못 들어가. 그래도 network는 다수의 pixel에 맞춰 학습해서 예측이 label보다 깔끔해져 ("Segmentation networks are robust to noisy supervision", p.56). 그 예측을 다음 label로 쓰면 round마다 mask가 물체 모양에 가까워져 (p.56 그림: Output after 1 training round → After 5 rounds → After 10 rounds).

- **box 밖은 왜 background로 되돌려?** box는 "물체가 이 밖엔 없다"는 확실한 정보야. 예측이 box 밖으로 번지면 그 정보로 잘라 내. 이 줄이 없으면 틀린 예측이 round마다 다음 label로 넘어가 쌓여 (L07 confirmation bias와 같은 구조).
- **이어 돌린 결과**: 설명용 toy(물체가 box보다 작은 사진 2장)로 3 round 돌리면 box 안 pixel 정확도가 0.44(채운 box) → 0.96이 돼.
- **시작 label이 좋을수록 좋아.** p.58–59 그래프에서 Rectangles보다 20% Bbox, Grabcut+, MCG, MCG ∩ Grabcut+로 시작한 쪽이 높아. p.60 제목: "We reach ~95% of full supervision by using GrabCut++" (그림 열 이름은 M ∩ G+).
- **L08과 같은 구조야.** 모델이 고른 결과를 다음 학습의 target으로 쓰는 건 L08 MIL의 re-localize ↔ re-train과 같아. 여기선 단위가 box가 아니라 pixel이야.
- **다른 빈칸 후보**: `argmax`, `0`(background), `box_lbl == 0`, `no_grad`.

**근거**: p.55 "What we want: / What we have:" 그림. p.56 "Initially, training is performed using a noisy mask, but the CNN's predictions are then used as pseudo-labels for retraining.", "Segmentation networks are robust to noisy supervision.", "Approximate masks are sufficient for training.", "Iteratively refine pseudo masks during training.", "Better masks lead to better segmentation." p.57–59 그래프(Training rounds × mIoU). p.60.

**단순화**: box 밖을 background로 되돌리는 줄은 논문(Khoreva et al., CVPR 2017 "Simple Does It") 보충이야. 슬라이드엔 "Iteratively refine"까지만 있어. 논문은 round 사이에 다른 후처리도 더 해. 겹친 box 처리는 뺐고, `train_segnet`은 helper야. 실제로는 학습 data 전체로 round마다 학습해.

---

## L09 2순위: 같은 계열의 변형 + 넓히기

### WS-13. Seeds — CAM to confident labels

```python
cam = cams * labels[:, None, None]             # [C, H, W] in 0..1, absent classes -> 0
score, cls = cam.max(dim=0)                    # strongest class per pixel
seed = torch.full_like(cls, ______)            # value for pixels that are not seeds (ignored by the loss)
seed[score > tau_fg] = cls[score > tau_fg] + 1 # foreground seeds: 1..C
seed[score < tau_bg] = 0                       # background seeds: 0
```

**Answer**: `255` (accept: loss의 `ignore_index`와 같은 상수면 뭐든 돼. `255`, `IGNORE`, `ignore_label`. 255는 PASCAL VOC 관례야)

**Process (EN)**: This code turns class activation maps into seed labels for a segmentation network. It keeps only the classes in the image label, takes the strongest class at each pixel, marks pixels with a high score as foreground seeds of that class and pixels with a low score as background seeds, and leaves everything else unlabeled; there is no loss or update here, because these seeds are the target for the next network. The blank is `255`, the ignore label, so that uncertain pixels between the two thresholds are left out of the segmentation loss.

**풀이**: seed는 "확실한 곳"만이야. CAM이 높으면 foreground, 낮으면 background야 (p.27). 그 사이를 비워 두는 건 슬라이드에 없는 구현 관례야. 애매한 pixel까지 억지로 정하면 틀린 label이 많이 섞이니까.

- **`255`는 슬라이드에 없는 구현 관례(PASCAL VOC)야.** 시험에선 "loss가 무시하는(ignore) 값"이라고 설명하면 돼.

- **`labels`를 곱하는 이유**: image label에 없는 class는 CAM이 켜져도 seed가 되면 안 돼. class는 image label이, 위치는 CAM이 정해.
- **WS-9 → WS-13 → WS-14로 이어져.** CAM이 seed가 되고, seed가 SEC seeding loss의 target이 돼. `255`는 WS-14의 `ignore_index=255`와 짝이야.
- **threshold 두 개**: `tau_fg`를 높이면 foreground seed는 정확해지지만 작아져. `tau_bg`를 낮추면 background seed가 줄어. 어느 쪽이든 비워 둔 영역(255)이 커져.
- **다른 빈칸 후보**: `tau_fg`, `tau_bg`, `+ 1`(0번을 background로 비우려고 class 번호를 한 칸 밀어), `labels`.

**근거**: p.27 "The Class Maps are Thresholded, to Generate Training Data", "High value scores → Foreground areas", "Low value scores → Background areas", "(This confident regions are commonly named "seeds")" (slide 문법 원문 유지), 그림 M3 "$>\tau_{loc}$". p.31 "Use CAM as reliable foreground seeds".

**단순화**: background threshold와 ignore 값 255는 흔한 구현 관례고 슬라이드엔 없어. 슬라이드 그림은 class 하나의 threshold $\tau_{loc}$만 보여 줘.

### WS-14. SEC — Seed, Expand, Constrain

```python
f = seg_net(img).softmax(dim=0)                    # [C+1, H, W], channel 0 = background
L_seed = F.nll_loss(f.log()[None], seed[None], ignore_index=255)  # seeds only
p, _ = f.flatten(1).______(dim=1, descending=True) # [C+1, HW]
w = d ** torch.arange(p.shape[1])                  # rank weights 1, d, d^2, ...
G = (p * w).sum(dim=1) / w.sum()                   # GWRP: image-level scores [C+1]
L_expand = F.binary_cross_entropy(G, present)      # image labels (+ background)
L_constrain = (Q * (Q.log() - f.log())).sum(dim=0).mean()  # KL(Q || f), Q = CRF(img, f) > 0
loss = L_seed + L_expand + L_constrain
```

**Answer**: `sort` (같은 뜻: `torch.sort(…)`)

**Process (EN)**: This code computes the three SEC losses for one image. The segmentation network predicts per-pixel class probabilities; the seeding loss is a cross-entropy only on the CAM seeds, the expansion loss compares image-level scores pooled by GWRP with the image labels, and the constrain-to-boundary loss is a KL divergence to the CRF-refined output, and the sum of the three updates the segmentation network. The blank is `sort`, because global weighted rank pooling first ranks the pixels of each class from the highest score down and then gives them decaying weights.

**풀이**: GWRP는 점수가 높은 pixel부터 weight 1, d, d², …를 줘. d = 0이면 1등 pixel만 보는 max pooling이고, d = 1이면 전부 같은 weight인 average pooling(GAP)이야 (둘 다 직접 확인). 그 사이 값이라 확실한 pixel 위주로 보되 seed 밖으로도 넓어지게 해.

- **세 loss의 target**: Seeding = CAM seed(pixel 일부), Expansion = image label(사진 단위), Constrain = CRF 출력 Q(pixel 전체)야. 셋 다 segmentation network 하나를 update해.
- **왜 GAP나 max가 아니야?** max는 1등 pixel 하나에만 gradient를 줘서 물체를 작게 잡고, GAP는 모든 pixel을 똑같이 밀어서 크게 잡아. GWRP는 그 사이야 (논문 보충).
- **설명용 숫자**: 한 class의 pixel 점수가 `[0.9, 0.8, 0.1, 0.0]`이면 d = 0 → 0.9, d = 0.5 → 0.71, d = 1 → 0.45야 (직접 계산).
- **background도 image label처럼 다뤄.** `present`의 0번(background)은 늘 1이야. 논문은 background를 label 집합 밖의 별도 항 $-\log G_{bg}$로 쓰는데, BCE에서 target 1인 항과 같은 꼴이야.
- **Constrain은 KL(Q‖f)야.** Q = CRF(줄인 image, f)가 앞이야. **카드는 Q를 상수(target)로 단순화했지만, 원 논문은 CRF를 통해서도 gradient를 흘려** (end-to-end, p.30 그림의 회색 화살표가 CRF를 지나가).
- **다른 빈칸 후보**: `ignore_index=255`(seed만), `descending=True`, `d`, `present`, `Q`.

**근거**: p.30 "Use boundary-aware propagation", 그림 (Weak Localization → Seeding Loss, Global Weighted Rank-Pooling → Expansion Loss, Downscale → CRF → Constrain-to-boundary Loss). p.31–32 "1. Seeding Loss: Use CAM as reliable foreground seeds, Preserve highly confident object regions", "2. Expansion Loss: Encourage predictions to expand beyond the seeds, Use image-level labels via Global Weighted Rank Pooling (GWRP)", "3. Constrain-to-boundary Loss: Refine predictions using CRF, Encourage masks to align with image boundaries", "SEC (Seed, Expand, Constrain)".

**단순화**:
- 세 loss의 식과 GWRP 식은 논문(Kolesnikov & Lampert, ECCV 2016) 보충이야. 슬라이드엔 이름과 역할만 있어.
- 논문은 d를 셋으로 나눠: 사진에 있는 class $d_+=0.996$, 없는 class $d_-=0$ (max pooling, "되도록 적은 pixel만 예측"), background $d_{bg}=0.999$. 여기선 d 하나로 줄였어 (없는 class에도 같은 d가 걸려).
- 논문 expansion loss는 있는 class와 없는 class를 따로 평균하는데, 여기선 `binary_cross_entropy` 평균 하나로 줄였어.
- `Q = crf(img, f).detach()`는 카드 밖에서 만들었어. **Q를 상수로 둔 건 우리 단순화야.** 원 논문은 CRF의 gradient까지 계산해서 end-to-end로 학습해 (Sec. 3.2). 줄인 해상도에서 CRF를 도는 부분(p.30 그림의 Downscale)도 뺐어.
- `Q`에 정확히 0이 있으면 `0 * log 0`이 nan이 돼. 실제 CRF 출력은 0보다 커서 괜찮고, toy로 돌릴 땐 `F.kl_div`를 쓰면 안전해.

### WS-15. SEAM — equivariant regularization

```python
cam_o = cam_net(img)                         # ŷ^o: [C, h, w], CAM of the original image
cam_t = cam_net(A(img))                      # ŷ^t: CAM of the transformed image
L_cls = cls_loss(cam_o, labels) + cls_loss(cam_t, labels)  # image labels, both branches
L_er = (______(cam_o) - cam_t).abs().mean()  # ER loss (L1)
loss = L_cls + L_er                          # (+ ECR loss with PCM, not shown)
```

**Answer**: `A` (accept: 이미지에 건 것과 같은 기하 변환이면 이름은 상관없어. `transform`, `T`, `flip`, `affine` 등)

**Process (EN)**: This code implements the equivariant regularization of SEAM with a siamese pair of the same CAM network. It computes the CAM of the original image and the CAM of the affine-transformed image, applies the image-label classification loss to both, and adds an L1 loss between the transformed original CAM and the CAM of the transformed image, which updates the shared network. The blank is `A`: the CAM of the original image must be transformed in the same way before the comparison, because activation maps should be equivariant, not invariant, to the input transformation.

**풀이**: 사진을 뒤집으면 CAM도 같이 뒤집혀야 해 ("Activation maps should be equivariant to the input", p.35). 그래서 원본 CAM에 같은 변환 A를 걸어 놓고 비교해. A 없이 비교하면 서로 다른 위치끼리 맞추라는 엉뚱한 loss가 돼.

- **target은?** 사람이 준 건 image label뿐이야(`L_cls`). ER loss의 target은 다른 branch의 CAM이야. 두 branch가 같은 network라 서로를 맞춰.
- **equivariance ↔ invariance**: invariance는 "입력이 바뀌어도 출력이 그대로"고, equivariance는 "입력이 바뀐 만큼 출력도 같이 바뀜"이야. 분류 점수는 invariant해야 하고, 위치 지도는 equivariant해야 해. L08 p.29 "Priors: Equivariance"와 같은 생각이야.
- **ECR과 PCM**: PCM(Pixel Correlation Module)은 pixel끼리의 affinity로 CAM을 같은 물체 안으로 퍼뜨려 (p.36). ECR loss는 한 branch의 PCM 결과와 다른 branch의 CAM을 (A로 위치를 맞춘 뒤) L1로 맞춰 (p.35 그림). 이 카드에선 뺐어.
- **다른 빈칸 후보**: `abs`(L1), `cam_t`, `labels`.

**근거**: p.35 "Enforce equivariance", "Activation maps should be equivariant to the input", 그림 (Siamese Network, Affine Transform, $\hat y^o, \hat y^t, y^o, y^t$, ER Loss = L1($\hat y^o$, $\hat y^t$), ECR Loss, Cls Loss). p.36 "Apply geometric transformations (e.g., affine transform)", "Predict CAMs for both original and transformed images", "Enforce consistency after compensating for the transformation", "Equivariant Regularization (ER) Loss: Encourage transformation-consistent CAMs", "Cross-view Consistency (ECR) Loss: Align CAMs across different views" ← slide 이름은 "Cross-view Consistency"인데 논문 이름은 Equivariant Cross Regularization이야.

**단순화**: ER loss 식 $\lVert A(\hat y^o) - \hat y^t\rVert_1$은 논문(Wang et al., CVPR 2020 "Self-supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation") 보충이야. 테스트에선 A로 좌우 flip을 썼어. A는 이미지와 CAM 둘 다에 걸 수 있어야 해 (해상도가 다르면 크기를 맞춰서). `cls_loss`는 CAM을 GAP해서 multi-label loss를 거는 helper야.

### WS-16. Hide-and-Seek — random patch hiding

```python
keep = (torch.______(B, 1, S, S) > p_hide).float()     # one keep/hide flag per grid patch
mask = F.interpolate(keep, size=(H, W), mode='nearest')  # [B, 1, H, W], 1 = visible
x = img * mask + mean_pixel * (1 - mask)               # hidden patches -> mean pixel value
loss = F.cross_entropy(classifier(x), labels)          # training only; test uses the full image
loss.backward()
```

**Answer**: `rand` (`randn`은 정규분포라 `> p_hide`가 "확률 p_hide로 숨김"이 안 돼)

**Process (EN)**: This code implements one training step of Hide-and-Seek. It splits each training image into an S×S grid, hides patches at random by filling them with the mean pixel value, and trains the classifier on the hidden image with the image-level label as the target, which updates the classifier; at test time the full image is used. The blank is `rand`, because the hidden patches are chosen randomly in every iteration, which forces the network to discover other visible object parts instead of relying on one discriminative region.

**풀이**: 얼굴이 가려진 iteration에선 몸통으로 맞혀야 해. 매번 다른 patch가 가려지니까 network가 물체 여러 부분을 다 쓰게 돼. 그래서 CAM이 물체를 더 넓게 덮어 ("CAM gradually covers a larger portion of the object", p.37).

- **왜 mean pixel로 채워?** 평균값으로 채우면 첫 conv layer의 기대 activation이 학습(일부 가림)과 test(안 가림)에서 같아져. 0(검정)으로 채우면 학습 때만 보는 값 분포가 생겨서 test와 어긋나 (논문 보충).
- **AE(WS-11)와 차이**: HaS는 **무작위**, AE는 **CAM이 높은 곳**을 골라 지워. HaS는 network 하나를 학습하면서 iteration마다 새로 가리고, AE는 지우고 다시 학습하는 step을 반복해.
- **test 땐 안 가려.** 가리는 건 학습 때만이야.
- **다른 빈칸 후보**: `p_hide`, `mean_pixel`, `nearest`, `1 - mask`.

**근거**: p.37 "Problem: CAM activates only the most discriminative regions.", "Solution (Hide-and-Seek): Randomly hide image patches during training. Encourage learning from the remaining visible parts. Generate more complete localization maps.", "Effect: The classifier is forced to discover other visible object parts. CAM gradually covers a larger portion of the object.", "Random hiding encourages the network to discover complementary object parts instead of relying on a single discriminative region." p.38 "Let us play peek-a-boo!" ← slide 크레딧은 [Wang et al. CVPR 2020]인데, Hide-and-Seek은 실제로 Singh & Lee, ICCV 2017 "Hide-and-Seek: Forcing a Network to be Meticulous for Weakly-supervised Object and Action Localization"이야.

**단순화**: grid 분할과 mean pixel 채우기는 논문 방식이야. 숨기는 확률 `p_hide`는 슬라이드에 없고, 논문(Singh & Lee, ICCV 2017)은 0.5를 썼어. `mean_pixel`은 `[1, 3, 1, 1]`(RGB 평균)이야.

### WS-17. Webly — Model-T filters noisy web images

```python
with torch.no_grad():
    p = model_T(web_imgs).sigmoid()           # [N, C], Model-T: multi-label classifier
conf = p[torch.arange(N), ______]             # one score per web image: [N]
web_clean = web_imgs[conf > tau]              # drop noisy web images -> train Web-SEC on the rest
```

**Answer**: `query` (각 web 이미지를 검색할 때 쓴 class 번호, shape `[N]`. 이름은 상관없어: `labels`, `web_labels`, `search_cls` 등)

**Process (EN)**: This code implements the filtering step of the webly supervised pipeline. Each web image comes with a noisy label, the class name used to search for it, so Model-T, a classifier trained on clean image-level labels of the target dataset, scores every web image without gradients, and only images whose score for their search class passes the threshold are kept to train the pixel labeler Web-SEC; nothing is updated in this step. The blank is `query`, the search class, which is the only noisy label a web image has.

**풀이**: web 이미지의 label은 검색어뿐이라 "Labels are noisy, ambiguous, and incomplete"야 (p.44). 깨끗한 target data로 학습한 Model-T가 "이 사진이 정말 그 class야?"를 판정해서 걸러 내 (p.46 그림: noisy web images → clean web images).

- **6단계 중 어디?** 2단계 "Use Model-T to filter noisy web images"야. 이렇게 거른 web 이미지로 3단계에서 Web-SEC를 학습해. web 이미지는 대개 가운데에 물체 하나만 있어서 대략적인 mask를 자동으로 만들 수 있어 (p.47).
- **Model-T는 두 번 나와.** 2단계에선 web 이미지를 거르고, 5단계에선 target 이미지의 CAM을 만들어. 6단계에서 Web-SEC의 pixel label과 CAM을 합쳐서 최종 segmentation model을 학습해.
- **Web-SEC vs CAM**: "Unlike CAM-based methods, Web-SEC actually learns a pixel classifier" (p.47).
- **다른 빈칸 후보**: `no_grad`, `tau`, `sigmoid`.

**근거**: p.44 "A learning method that uses data collected from the web as supervision", "Labels are noisy, ambiguous, and incomplete". p.45–50 "Pipeline" 1–6. p.45 그림 (Target domain / Web domain, Model-T, Model-W, Filter, Enhance). p.46 그림. p.47 bullets.

**단순화**: 슬라이드엔 거르는 기준이 없어. 여기선 "검색 class의 Model-T 점수 > tau"로 단순화했어. 실제 기준은 논문(Shen et al., CVPR 2018 "Bootstrapping the Performance of Webly Supervised Semantic Segmentation")을 따라. 논문의 Model-T는 target data의 image-level label로 학습한 weakly supervised network고, web 이미지 중 label이 틀린 것과 segment하기 어려운 것을 걸러 쉬운 이미지만 남긴대 (검수자의 웹 요약, 원문 기준은 미확인). Model-T는 사진 하나에 class가 여러 개일 수 있어서 `sigmoid`로 썼어.

### WS-18. CD-CLIP — class-distribution-aware affinity (CDA)

```python
P = S_a.softmax(dim=-1)                              # [N, C'] class distribution of each patch
M = 0.5 * (P[:, None] + P[None, :])                  # [N, N, C'] pairwise mixtures
D_js = 0.5 * (kl(P[:, None], M) + kl(P[None, :], M)) # [N, N] JS divergence
S_d = (______ - D_js) / tau                          # similar distributions -> large score
A_d = S_d.softmax(dim=-1) * A_init                   # correct the original (MHSA) affinity
cam_re = cam_init @ torch.matrix_power(A_d, t)       # propagate: M_re = M_init (A_d)^t
```

**Answer**: `1` (accept: `0`, `ln 2` 같은 아무 상수. 뒤의 softmax가 상수 shift를 지워서 결과가 같아. 중요한 건 `- D_js`의 부호고, 슬라이드 식이 $1 - D_{JS}$라 `1`을 정답으로 둬)

**Process (EN)**: This code implements the class-distribution-aware affinity of CD-CLIP. Each patch is described by a softmax distribution over CLIP target classes and super-classes, the pairwise Jensen–Shannon divergence between these distributions becomes a similarity that filters the original attention affinity, and the initial CAM is propagated through the corrected affinity t times; there is no loss or update here, and the refined CAM later serves as the pseudo label for the segmentation decoder. The blank is `1`, the slide's constant in 1 − D_JS: what matters is that the divergence enters with a minus sign, so a small JS divergence (similar class distributions) gives a large affinity score.

**풀이**: 머리는 CAM이 세고 몸통은 약해도, 둘의 class 분포(모든 class에 대한 점수 모양)는 비슷해. 분포가 비슷한 patch끼리의 연결은 살리고 다른 patch끼리의 연결은 줄인 뒤 CAM을 퍼뜨리면, 머리의 activation이 몸통으로 번져 (p.69).

- **JS(Jensen–Shannon) divergence**: 두 분포의 평균 M에 대한 KL을 둘 다 구해서 평균한 거야. 대칭이고, 0 이상이고, 자연로그 기준 ln 2 이하야. 같은 분포면 0이야 (직접 확인).
- **왜 target class 점수 하나가 아니야?** "Similar appearance does not always imply the same class" (p.65). 한 class 점수만 보면 점수가 비슷한 다른 물체가 섞여. 분포 전체를 비교하면 더 잘 가려 ("The full distribution can be more informative than one target-class score").
- **`1`은 사실 상수라 softmax가 지워 줘.** 중요한 건 `- D_js`(거리를 점수로 뒤집기)야. 슬라이드 식이 $1 - D_{JS}$라 `1`을 정답으로 뒀어. 로그 밑이 2면 $D_{JS}\in[0,1]$이라 $1 - D_{JS}$가 [0, 1] 유사도가 되는데, 슬라이드엔 밑이 없어.
- **`tau`**: 작을수록 softmax가 뾰족해져서 가장 비슷한 patch에 affinity가 몰려. temperature 역할이야.
- **SBE와 짝**: CDA는 CAM을 **넓히고**(completeness), SBE는 어디서 **멈출지**(boundary)를 정해 (p.74).
- **다른 빈칸 후보** (의미가 더 큰 순): `-`(부호), `softmax(dim=-1)`, `tau`, `t`(전파 횟수), `*`, `0.5`.

**근거**: p.65 "Every patch has scores over multiple classes", "Patches of the same semantic class show similar class distributions". p.67 $S_a = \text{Min-max}(\text{CosSim}(F, T_{all}))$. p.69 "Small JS divergence: similar class distributions", "Large JS divergence: different class distributions", "Correct the original affinity", "Propagate CAM activation", $S_d = \frac{1 - D_{JS}(\text{Softmax}(S_a)\,\Vert\,\text{Softmax}(S_a)^\top)}{\tau}$, $A_d = \text{Softmax}(S_d)\cdot A_{init}$, $M^c_{re} = M^c_{init}\cdot(A_d)^t$.

**단순화**: 슬라이드 식 $A_d$의 `·`는 "Distribution similarity filters unreliable relationships in the original affinity"(p.69)에 맞춰 원소별 곱으로 읽었어. 논문 원문은 확인 못 했어. `A_init`은 CLIP의 MHSA(Multi-Head Self-Attention) affinity야 (p.66 그림 "Initial Affinity ← MHSA"). `kl`은 helper야.

### WS-19. WISH — multi-stage matching cost

```python
c_cls = -p_cls[:, cls_j]                                  # [Q, J] class matching
c_prompt = kld(Z_hat[:, None], Z[None])                   # [Q, J] prompt matching (KLD)
c_mask = dist(M_hat[:, None, None], M_sam[None]).______(dim=-1).values  # [Q, J, 3] -> [Q, J]
cost = alpha * c_cls + beta * c_prompt + gamma * c_mask
q_idx, j_idx = linear_sum_assignment(cost.detach().numpy())  # Hungarian matching: one query per weak label
```

**Answer**: `min` (`torch.min(…, dim=-1).values` 형태. `max`·`mean`은 shape은 맞지만 슬라이드의 $\min_n$이 아니야)

**Process (EN)**: This code computes the matching cost between object queries and weak labels in WISH. For each query and weak label it adds a class cost from the predicted class probability, a prompt cost as the KL divergence between the predicted prompt latent and the SAM prompt latent of the label, and a mask cost from the closest of SAM's three candidate masks, then Hungarian matching assigns one query to each weak label so that the matched pairs can supervise the model; nothing is updated in this step. The blank is `min`, which selects the SAM candidate mask closest to the predicted instance mask, the adaptive mask selection.

**풀이**: SAM은 prompt 하나에 mask를 3개 줘. 어느 수준이 맞는지 미리 정하지 않고 지금 예측에 가장 가까운 걸 고르니까, 물체마다 알맞은 mask가 골라져 ("Adaptive selection outperforms using a fixed mask level", p.94).

- **세 cost가 묻는 것**: class = "무엇인가", prompt = "어디 있나", mask = "어느 SAM mask가 맞나" (p.90 "what it is, where it is, and which SAM mask fits it best").
- **왜 prompt latent로 비교해?** tag, point, box는 모양이 달라서 바로 비교할 수 없어. SAM prompt encoder가 point와 box를 같은 latent 공간에 넣어 주니까 거기서 비교해 (p.88). tag는 CAM peak를 point prompt로 바꿔서 써 (p.91).
- **Hungarian matching**: 총 cost가 가장 작은 1:1 짝을 찾아. query 수 Q가 weak label 수 J보다 많아도 돼.
- **다른 빈칸 후보**: `kld`, `linear_sum_assignment`, `-`(class cost 부호), `None`(broadcast 축).

**근거**: p.88 "SAM's prompt space serves as a common representation for heterogeneous supervision", $\hat Z = \mathcal{H}_{prompt}(Q)$. p.89 $\text{Cost}^{i,j}_{cls} = \hat y^i_{cls}(c^j)$, $\text{Cost}^{i,j}_{prompt} = \text{KLD}(\hat Z^i, Z^j)$, $\text{Cost}^{i,j}_{mask} = \min_{n\in\{1,2,3\}} d(\hat M^i, M^{j,n}_{SAM})$, $\text{Cost}^{i,j} = \alpha\text{Cost}^{i,j}_{cls} + \beta\text{Cost}^{i,j}_{prompt} + \gamma\text{Cost}^{i,j}_{mask}$. p.90 "Hungarian matching assigns predictions to weakly labeled instances". p.94 "SAM returns three masks for each prompt.", "WISH selects the candidate closest to the current prediction."

**단순화**:
- 슬라이드의 class cost엔 마이너스가 없어. Hungarian은 cost가 **최소**인 짝을 찾으니까, 확률이 클수록 cost가 작아지게 `-`를 붙였어 (DETR 계열 관례). 논문의 정확한 부호는 확인 못 했어.
- `kld`(latent를 softmax로 분포로 바꾼 뒤 KL)와 mask 거리 `dist`는 helper야.
- 논문: Kweon & Yoon, CVPR 2025 "WISH: Weakly Supervised Instance Segmentation using Heterogeneous Labels". 슬라이드엔 cite가 없어서 웹으로 확인했어.

---

## L09 방법 간 비교

### Gray box vs White box

| | Gray box (WS-10) | White box: CAM (WS-9) |
|---|---|---|
| 무엇을 쓰나 | 입력에 대한 gradient $\partial S_c/\partial I$ | 마지막 feature map + FC weight |
| 계산 | backward 필요 (SmoothGrad는 n번) | forward만 ("without the need for separate gradient calculations") |
| network | 아무 분류기나 그대로 | 구조를 바꿔: FC 제거 → GAP → 1×1 conv |
| 지도 모양 | pixel 단위, 잡음이 많아 | 마지막 conv 해상도, 거칠어 |
| 공통 한계 | 분류에 필요한 부분만 밝혀 (p.23) | discriminative area만 덮어 (p.28) |

### 물체 전체로 넓히기: 어디를 가리나

| 방법 | 가리는 곳 | 누가 정하나 | 반복 |
|---|---|---|---|
| Hide-and-Seek (WS-16, p.37) | 무작위 grid patch | 난수 | 학습 iteration마다 새로 |
| Adversarial Erasing (WS-11, p.39) | CAM이 높은 곳 | 직전 step의 분류기 | 지우고 다시 학습 (t = 1, 2, 3) |
| Ren et al. (L08 WS-8, L08 p.38) | 가장 relevant한 region | 같이 학습되는 Concrete DropBlock θ | min–max로 동시에 학습 |

### GAP vs GWRP vs max

| pooling | 하는 일 | 어디서 | gradient가 가는 곳 |
|---|---|---|---|
| GAP | 모든 pixel 평균 (d = 1) | CAM 분류기 (WS-9, p.24) | 모든 pixel에 똑같이 |
| GWRP | 순위대로 weight 1, d, d², … | SEC expansion (WS-14, p.31) | 점수가 높은 pixel 위주 |
| max | 1등 pixel만 (d = 0) | L08 Standard MIL의 max와 같은 연산 | pixel 하나 |

설명용 숫자 `[0.9, 0.8, 0.1, 0.0]`: max 0.9, GWRP(d = 0.5) 0.71, GAP 0.45.

### target은 어디서 오나

| 카드 | 학습 target | update되는 것 |
|---|---|---|
| WS-9 CAM | image label | backbone + `fc` |
| WS-10 SmoothGrad | 없음 (분석만) | 없음 |
| WS-11 AE | step마다 image label. 지운 영역의 합이 결과물 | step마다 새 classifier |
| WS-12 Boxly | 채운 box → 그 뒤엔 network 자기 예측 (box 밖은 background) | segmentation net |
| WS-13 Seeds | 없음 (target을 만드는 쪽) | 없음 |
| WS-14 SEC | seed · image label · CRF 출력 | segmentation net |
| WS-15 SEAM | image label + 다른 branch의 CAM | 공유 CAM net |
| WS-16 HaS | image label | classifier |
| WS-17 Webly filter | 없음 (거르기) | 없음 |
| WS-18 CDA | 없음 (CAM 다듬기) | 없음 |
| WS-19 WISH | 없음 (짝짓기). 짝이 다음 loss의 target을 정해 | 없음 |

---

## L09 함정 Top 8

1. **CAM은 gradient를 안 써.** white box는 forward의 feature map과 FC weight로 바로 나와 (p.25). gradient를 쓰는 건 gray box야 (p.21–22).
2. **CAM weight = 그 class의 FC weight야.** feature map의 단순 평균이 아니야. p.27 그림은 $k$가 class, $C$가 channel 수라서 기호가 카드와 달라.
3. **seed는 확실한 곳만이야.** 높으면 foreground, 낮으면 background, 중간은 비워 둬 (p.27).
4. **AE는 높은 곳을 지워.** HaS는 **무작위**로 가려 ("adversarial instead of random erasures", p.39). HaS는 학습 때만 가리고 test 땐 사진 전체를 써.
5. **SEC 세 loss의 target**: Seeding = CAM seed, Expansion = image label(GWRP), Constrain = CRF. GWRP는 d = 0이면 max, d = 1이면 GAP야.
6. **SEAM은 equivariance야.** 원본 CAM에 변환 A를 걸어서 변환된 이미지의 CAM과 비교해. "변환해도 CAM이 그대로"(invariance)가 아니야.
7. **Boxly**: 첫 label은 box를 채운 noisy mask고, 그다음부터는 network 예측이야. box 밖은 늘 background야.
8. **CD-CLIP은 JS divergence가 작을수록 affinity가 커** ($1 - D_{JS}$). **WISH mask cost는 SAM mask 3개 중 min**이야.
