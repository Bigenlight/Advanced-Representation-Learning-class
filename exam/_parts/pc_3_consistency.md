# PC3. Consistency regularization (Π-model · Temporal Ensembling) — Pseudo code 핵심 정리 & 문제

> 범위: L06 p.16–25. Pseudo code 슬라이드는 **Π-model Algorithm 1 (p.18)**, **Temporal Ensembling Algorithm 2 (p.20)** 두 장. Mean Teacher는 pseudo code 슬라이드가 없어서(p.23–29) 비교 대상(foil)으로만 다룸.
> 공통 core idea (p.16): "Predictions should remain **consistent under small perturbations**" — $\mathcal L_u = D\big(f(x), f(\mathcal A(x))\big)$. 두 알고리즘 차이는 결국 **target $\tilde z_i$를 어디서 가져오냐** 하나임.

## A. Annotated pseudo code

### Π-model (Sajjadi et al., ICLR 2016 ← 슬라이드 표기, 아래 flag 참고) — L06 p.18

**Transcription** (렌더한 슬라이드 그대로):

```
Algorithm 1  Π-model pseudocode.
Require: x_i  = training stimuli
Require: L    = set of training input indices with known labels
Require: y_i  = labels for labeled inputs i ∈ L
Require: w(t) = unsupervised weight ramp-up function
Require: f_θ(x) = stochastic neural network with trainable parameters θ
Require: g(x) = stochastic input augmentation function
  for t in [1, num_epochs] do
    for each minibatch B do
      z_{i∈B} ← f_θ(g(x_{i∈B}))                         ▷ evaluate network outputs for augmented inputs
      z̃_{i∈B} ← f_θ(g(x_{i∈B}))                         ▷ again, with different dropout and augmentation
      loss ← − (1/|B|) Σ_{i∈(B∩L)} log z_i[y_i]          ▷ supervised loss component
                + w(t) (1/(C|B|)) Σ_{i∈B} ||z_i − z̃_i||²  ▷ unsupervised loss component
      update θ using, e.g., ADAM                         ▷ update network parameters
    end for
  end for
  return θ
```

$$loss \leftarrow -\frac{1}{|B|}\sum_{i\in(B\cap L)}\log z_i[y_i] \;+\; w(t)\,\frac{1}{C|B|}\sum_{i\in B}\|z_i-\tilde z_i\|^2$$

- 슬라이드 citation `Sajjadi et al., ICLR 2016, Regularization With Stochastic Transformations and Perturbations for Deep Semi-Supervised Learning` ← 의심: "Algorithm 1 Π-model pseudocode"와 Π-model이라는 이름 자체는 Laine & Aila의 Temporal Ensembling 논문(ICLR 2017, Algorithm 1·2가 한 논문)에서 나온 것이고, Sajjadi et al.은 NIPS 2016. **시험은 슬라이드 표기 기준**으로 기억.
- $C$는 Require에 정의 안 됨 ← Algorithm 2의 $\mathbf 0_{[N\times C]}$로 보면 **class 수**. 그래서 $\frac{1}{C|B|}\sum\|\cdot\|^2$ = element 평균 **MSE** (p.17 $\mathcal L_u^{\Pi}=\sum \mathrm{MSE}(f_\theta(\mathbf x), f'_\theta(\mathbf x))$와 같은 것).
- $z_i$, $\tilde z_i$ 두 줄이 **글자상 완전히 똑같은 건 오타 아님** — $f_\theta$, $g$가 둘 다 **stochastic**이라 값이 다름. 차이는 ▷ 주석에만 있음.

**핵심 줄 (Key lines)**

| line | what it does | which core idea it implements | 빈칸 후보 |
|---|---|---|---|
| `z_{i∈B} ← f_θ(g(x_{i∈B}))` | augmented input을 dropout 있는 network에 1번째 forward | perturbation 두 출처 = **stochastic augmentation $g$** + **model stochasticity (dropout)** (p.16) | "stochastic input ______ function" → augmentation |
| `z̃_{i∈B} ← f_θ(g(x_{i∈B}))` ▷ again, with different dropout and augmentation | **같은 $f_\theta$ (같은 θ)** 로 2번째 forward, noise만 다름 → 이게 target | **핵심 trick**: 같은 input의 두 stochastic view → 출력이 같아야 함. 대가는 **forward 2회** (→ TE의 동기, p.19) | "again, with different ______ and ______" → dropout, augmentation |
| `− (1/\|B\|) Σ_{i∈(B∩L)} log z_i[y_i]` | supervised CE, **labeled 샘플만** | label은 아는 데서만 씀 | $B\cap L$ |
| `+ w(t) (1/(C\|B\|)) Σ_{i∈B} \|\|z_i − z̃_i\|\|²` | 두 출력의 **squared difference (MSE)**, **minibatch 전체** (labeled + unlabeled) | consistency regularization = label 없이 계산되는 $\mathcal L_u$ → unlabeled에서 학습 신호 (H1 smoothness) | $i\in B$ / squared difference / MSE |
| `Require: w(t) = unsupervised weight ramp-up function` | unsupervised 항 가중치를 epoch $t$에 따라 키움 | 초반 예측은 믿을 수 없으니 작게 시작 (p.10 $\mu(t)$ "increasing in time") | ramp-up |

**Pseudo code ↔ core idea 한 줄**: 똑같이 생긴 두 줄 `z ← f_θ(g(x))`, `z̃ ← f_θ(g(x))`가 Π-model 그 자체 — **같은 네트워크·다른 noise 두 번 forward → 출력 차이를 MSE로 줄임**, 이 항은 label이 없어도 되니 $i\in B$ 전체에 걸림.

- Implementation trivia (**low priority**): optimizer는 "e.g., ADAM"; supervised 항 분모가 $|B\cap L|$이 아니라 $|B|$.

---

### Temporal Ensembling (Laine et al., ICLR 2017) — L06 p.20

**Transcription** (렌더한 슬라이드 그대로, caption 포함):

```
Algorithm 2  Temporal ensembling pseudocode. Note that the updates of Z and z̃ could equally
well be done inside the minibatch loop; in this pseudocode they occur between epochs for clarity.
Require: x_i  = training stimuli
Require: L    = set of training input indices with known labels
Require: y_i  = labels for labeled inputs i ∈ L
Require: α    = ensembling momentum, 0 ≤ α < 1
Require: w(t) = unsupervised weight ramp-up function
Require: f_θ(x) = stochastic neural network with trainable parameters θ
Require: g(x) = stochastic input augmentation function
  Z ← 0_[N×C]                                          ▷ initialize ensemble predictions
  z̃ ← 0_[N×C]                                          ▷ initialize target vectors
  for t in [1, num_epochs] do
    for each minibatch B do
      z_{i∈B} ← f_θ(g(x_{i∈B}, t))                      ▷ evaluate network outputs for augmented inputs
      loss ← − (1/|B|) Σ_{i∈(B∩L)} log z_i[y_i]          ▷ supervised loss component
                + w(t) (1/(C|B|)) Σ_{i∈B} ||z_i − z̃_i||²  ▷ unsupervised loss component
      update θ using, e.g., ADAM                         ▷ update network parameters
    end for
    Z ← αZ + (1 − α)z                                   ▷ accumulate ensemble predictions
    z̃ ← Z/(1 − α^t)                                     ▷ construct target vectors by bias correction
  end for
  return θ
```

$$Z \leftarrow \alpha Z + (1-\alpha)\,z, \qquad \tilde z \leftarrow \frac{Z}{1-\alpha^t}$$

- `z_{i∈B} ← f_θ(g(x_{i∈B}, t))` ← 의심(표기): 슬라이드에서 $t$는 **$g$의 두 번째 인자** (괄호가 $g(x_{i\in B}, t)$로 닫힘). 그런데 Require는 $g(x)$를 인자 1개로 정의했고 Algorithm 1에는 $t$가 없음 → 의미 설명 없음. 원문 유지, 시험 포인트 아님. (notes 06은 $f_\theta(g(x_i), t)$로 옮겼는데 **슬라이드 기준으로는 $t$가 $g$ 안**에 있음.)
- `Z ← αZ + (1 − α)z`의 $z$ ← 표기 축약: 루프 안에서는 $z_{i\in B}$만 정의되지만 여기선 **이번 epoch 동안 모은 N개 샘플 전체의 예측 ($N\times C$)** 을 뜻함.
- p.10 Notations와 표기 충돌 (low priority): p.10에선 $L$ = number of unique labels, $\alpha$ = MixUp parameter, $\mathbf z$ = logits인데, Algorithm에선 $L$ = labeled index **set**, $\alpha$ = **ensembling momentum**, $\log z_i[y_i]$를 쓰니 $z_i$는 사실상 softmax **확률** 벡터. 알고리즘 문맥 기준으로 읽을 것.

**핵심 줄 (Key lines)**

| line | what it does | which core idea it implements | 빈칸 후보 |
|---|---|---|---|
| `Z ← 0_[N×C]`, `z̃ ← 0_[N×C]` | **전체 training set N개 × C class** 크기 메모리 할당 | "EMA of the model prediction **per training sample**" → 한계: "Requires storing predictions for every sample", "Difficult to scale" (p.19, p.22) | $N\times C$ |
| `z_{i∈B} ← f_θ(g(x_{i∈B}, t))` (**한 줄뿐**) | minibatch당 forward **1회**. Π의 `z̃ ← f_θ(…)` 줄이 **삭제됨** | "Only **one forward pass** per iteration", "Lower computational cost" (p.22) | one forward pass |
| `+ w(t) (1/(C\|B\|)) Σ_{i∈B} \|\|z_i − z̃_i\|\|²` | loss 식은 Π와 **글자 하나 안 다름** — 하지만 $\tilde z_i$는 network가 아니라 **메모리에서 읽은 상수** | "use the **historical predictions of the same sample** as the consistency target" (p.19) | $\tilde z_i$ |
| `Z ← αZ + (1 − α)z` (**`end for` 다음 = epoch 끝**) | 샘플별 예측의 **EMA** 누적 | **핵심 trick**: 여러 epoch(다른 aug·dropout·weight)의 예측 평균 = **temporal ensemble** → "More stable target than Π-model"; 대신 "Updated only **once per epoch**" (p.22) | $\alpha Z + (1-\alpha)z$ / ensembling momentum |
| `z̃ ← Z/(1 − α^t)` | **bias correction**: $Z$가 0에서 시작해서 생기는 0쪽 편향 보정 | 초반 epoch에도 제 스케일의 target 확보 | $1-\alpha^t$ / bias correction |

**왜 $1-\alpha^t$?** — 예측이 $t$ epoch 동안 일정한 $z$라고 가정하면

$$Z_t = (1-\alpha)\sum_{k=0}^{t-1}\alpha^k z = (1-\alpha^t)\,z \;\;\Rightarrow\;\; \tilde z = \frac{Z_t}{1-\alpha^t} = z$$

→ 나누면 정확히 $z$로 복원. $t$는 **epoch index** (EMA가 epoch마다 1번이니까). Adam의 bias correction과 같은 원리.

**Pseudo code ↔ core idea 한 줄**: Π-model 코드에서 **`z̃ ← f_θ(g(x))` 줄을 지우고**, 대신 루프 밖에 **`Z ← αZ + (1−α)z` (prediction EMA) + `z̃ ← Z/(1−α^t)` (bias correction)** 두 줄을 붙인 것 = target을 "지금 한 번 더 forward"에서 "**과거 예측의 per-sample EMA**"로 바꿈.

- Caption 포인트: $Z,\tilde z$ 업데이트는 "could equally well be done **inside** the minibatch loop" — 안에 넣어도 **각 샘플은 epoch당 1번만 등장**하니 샘플별 target은 여전히 epoch당 1번 갱신 (→ 한계 "once per epoch"는 그대로).
- 첫 epoch엔 $\tilde z=\mathbf 0$ (초기화값)이 target → $w(t)$가 **ramp-up**이라 초반 가중치가 작아서 버팀. 두 Require가 짝으로 설계된 것.

> 보충 (low priority, 슬라이드엔 없음): 논문 기본값 $\alpha=0.6$, $w(t)$는 0에서 Gaussian 곡선으로 ramp-up.

---

## B. Cross-method comparison

| | **Π-model** (Alg.1, p.17–18) | **Temporal Ensembling** (Alg.2, p.19–22) | **Mean Teacher** (p.23–29, pseudo code 슬라이드 없음) |
|---|---|---|---|
| Forward passes per step | **2** (같은 θ, 다른 dropout + aug) | **1** | 2 (student θ with noise $\eta$, teacher θ' with $\eta'$) — backprop은 **student만** |
| Target이 오는 곳 | 같은 network의 2번째 stochastic forward $\tilde z_i$ | **per-sample 메모리** $\tilde z = Z/(1-\alpha^t)$ (과거 예측) | **teacher network** (EMA weight)의 prediction |
| 무엇을 EMA? | 없음 | **predictions $Z$** (momentum $\alpha$) | **weights** $\theta' \leftarrow \beta\theta' + (1-\beta)\theta$ (decay $\beta$) |
| EMA 줄 위치 / target 갱신 시점 | — / 매 step (새로 forward) | 루프 **밖** (`end for` 뒤) / **once per epoch** | 매 iteration / "Teacher updated **every iteration**" (p.25) |
| Bias correction | 없음 | **있음** $Z/(1-\alpha^t)$ | 슬라이드에 없음 |
| Memory cost | 추가 없음 | **$N\times C$** prediction 저장 → "Difficult to scale to large datasets" | weight 사본 1개, "**No per-sample prediction memory**", "Scales well to large datasets" (p.25) |
| Target 쪽 gradient | 슬라이드 미명시 | 메모리에서 읽은 **상수** | teacher는 "**No backpropagation**", "Parameters updated **only by EMA**" (p.24) |
| Unsupervised loss | $w(t)\frac{1}{C\lvert B\rvert}\sum_{i\in B}\lVert z_i-\tilde z_i\rVert^2$ (**MSE**, $B$ 전체) | Π와 **동일한 식** | student–teacher prediction 간 distance; "**MSE performs better than KL**" (p.28) |
| Supervised loss | CE, $i\in B\cap L$만 | 동일 | classification cost (labeled만, student) |
| Unsup. weight | $w(t)$ **ramp-up** | $w(t)$ **ramp-up** | 슬라이드 미명시 (일반 SSL objective의 $\mu(t)$ = "increasing in time", p.10) |
| 한 줄 | 두 번 돌려서 맞춰라 | 과거 내 예측의 평균에 맞춰라 | 내 weight 평균(teacher)의 예측에 맞춰라 |

> Mean Teacher를 Algorithm 2 스타일로 옮긴 **참고용 sketch** (슬라이드 아님, p.24 pipeline 기반):
> `for each minibatch B: z ← f_θ(x, η); z̃ ← f_θ'(x, η'); loss ← CE(labeled) + λ·‖z − z̃‖²; update θ; θ' ← βθ' + (1−β)θ` — **EMA 줄이 minibatch 루프 안**에 있고 대상이 **$Z$가 아니라 $\theta'$** 라는 게 TE와의 diff. (MoCo의 `f_k.params = m*f_k.params+(1-m)*f_q.params`, L05 p.12와 같은 모양)

---

## C. Question pool (15 questions: Blank 6 · MCQ 4 · T/F 5, ★ = killer)

### Q1 [Blank]
In the Π-model pseudocode (Algorithm 1), the network is evaluated twice per minibatch: $z_{i\in B} \leftarrow f_\theta(g(x_{i\in B}))$ and $\tilde z_{i\in B} \leftarrow f_\theta(g(x_{i\in B}))$. The comment on the second line says it is evaluated "again, with different (1) ______ and (2) ______."
**Answer**: (1) dropout (2) augmentation (순서 바뀌어도 OK; accept: dropout masks / stochastic augmentation) — 두 줄이 글자상 똑같은 게 핵심. **같은 θ**, noise만 다름 → 출력이 달라지고, 그 차이를 줄이는 게 consistency. p.17 원문도 "different stochastic augmentation and dropout masks". (p.17–18)

### Q2 [Blank]
In both Algorithm 1 (Π-model) and Algorithm 2 (Temporal Ensembling), the supervised loss component is $-\frac{1}{|B|}\sum_{i\in(\_\_\_\_\_\_)}\log z_i[y_i]$, i.e., cross-entropy is computed only for the minibatch samples whose labels are known.
**Answer**: $B\cap L$ (accept: B ∩ L) — $L$ = "set of training input indices with known labels". 반면 unsupervised 항은 $\sum_{i\in B}$ (전체) → 이 비대칭이 unlabeled data를 쓰는 방법. (p.18, p.20)

### Q3 [Blank]
In both pseudocodes, the unsupervised loss component is multiplied by $w(t)$, which the `Require:` line defines as the "unsupervised weight ______ function."
**Answer**: ramp-up (accept: ramp up, rampup) — 초반엔 예측이 부정확하고, 특히 TE는 첫 epoch target이 $\tilde z=\mathbf 0$이라 가중치를 작게 시작해서 점점 키움. p.10 $\mu(t)$ "The unsupervised loss weight, **increasing in time**"과 같은 역할. 함정: ramp-down / decreasing. (p.18, p.20, p.10)

### Q4 [Blank]
Temporal Ensembling initializes the ensemble predictions as $Z \leftarrow \mathbf 0_{[\_\_\_\_\_\_]}$. This per-sample storage is why TE "requires storing predictions for every sample" and is difficult to scale to large datasets. (Use $N$ = number of training samples, $C$ = number of classes.)
**Answer**: $N\times C$ (accept: N × C, [N, C]) — 샘플 하나당 C차원 예측 벡터를 **전체 training set에 대해** 들고 있음 ("EMA ... **per training sample**"). Mean Teacher가 "No per-sample prediction memory"로 해결한 게 바로 이 줄. (p.20, p.22, p.25)

### Q5 [Blank]
At the end of each epoch, Temporal Ensembling constructs the targets as $\tilde z \leftarrow Z / ($(1) ______$)$. The comment says this step constructs target vectors by (2) ______.
**Answer**: (1) $1-\alpha^t$ (accept: 1 − α^t) (2) bias correction — $Z$가 $\mathbf 0$에서 시작해 초반 EMA가 0쪽으로 편향됨. 예측이 일정한 $z$면 $Z_t=(1-\alpha^t)z$라서 $1-\alpha^t$로 나누면 정확히 $z$ 복원 (Adam bias correction과 같은 원리). $t$ = epoch index. (p.20)

### Q6 [Blank]
Mean Teacher's core idea is to "replace the EMA of (1) ______ with an EMA of (2) ______," so the teacher is updated as $\theta' \leftarrow \beta\theta' + (1-\beta)\theta$.
**Answer**: (1) predictions (2) model weights (accept for (2): weights / model parameters / parameters) — TE의 $Z\leftarrow\alpha Z+(1-\alpha)z$ (prediction EMA)를 $\theta'\leftarrow\beta\theta'+(1-\beta)\theta$ (weight EMA)로 바꾼 것. 순서 뒤집으면 틀림 — 이게 단골 함정. (p.23, p.29)

### Q7 [MCQ]
The following excerpt is taken from the pseudocode of which method?
```
for each minibatch B do
    z_{i∈B} ← f_θ(g(x_{i∈B}, t))
    loss ← −(1/|B|) Σ_{i∈(B∩L)} log z_i[y_i] + w(t) (1/(C|B|)) Σ_{i∈B} ||z_i − z̃_i||²
    update θ using, e.g., ADAM
end for
Z ← αZ + (1 − α)z
```
(A) Π-model
(B) Temporal Ensembling
(C) Mean Teacher
(D) MoCo
**Answer**: (B) — forward **1줄**, $\tilde z_i$는 계산 안 하고 그냥 씀(메모리), 루프 밖에서 **predictions $Z$의 EMA**. (A) Π-model은 `z̃ ← f_θ(g(x))` forward 줄이 하나 더 있고 $Z$가 없음. (C) Mean Teacher는 **weights** $\theta'\leftarrow\beta\theta'+(1-\beta)\theta$를 매 iteration EMA하고 teacher가 forward함. (D) MoCo는 key encoder **weights** momentum update(`f_k.params = m*f_k.params+(1-m)*f_q.params`) + queue, loss는 InfoNCE (L05 p.12). (L06 p.20)

### Q8 [MCQ] ★
In Algorithm 2 (Temporal Ensembling), which line actually performs the averaging of the model's predictions over past epochs, i.e., the step that makes the consistency target a *temporal ensemble*?
(A) $z_{i\in B} \leftarrow f_\theta(g(x_{i\in B}, t))$
(B) $Z \leftarrow \alpha Z + (1-\alpha)z$
(C) $\tilde z \leftarrow Z/(1-\alpha^t)$
(D) $\tilde z \leftarrow \mathbf 0_{[N\times C]}$
**Answer**: (B) — ▷ "accumulate **ensemble predictions**". 매 epoch 예측을 EMA로 누적 → 다른 aug·dropout·weight 상태의 예측 평균 = ensemble. (A)는 현재 예측 1회 forward일 뿐. (C)는 bias correction — 평균을 내는 게 아니라 zero-init 편향만 **rescale** (예측이 일정하면 그냥 $z$ 반환). (D)는 초기화. (p.20)

### Q9 [MCQ]
Going from Algorithm 1 (Π-model) to Algorithm 2 (Temporal Ensembling), what is the key change in how the target $\tilde z_i$ is obtained?
(A) $\tilde z_i$ is no longer computed by a second forward pass; it is read from a per-sample memory of EMA predictions that is updated after each epoch.
(B) $\tilde z_i$ is computed by a second network whose weights are an EMA of $\theta$.
(C) $\tilde z_i$ is replaced by the ground-truth one-hot label $y_i$ for every sample.
(D) $\tilde z_i$ is computed by a second forward pass on the original input without augmentation.
**Answer**: (A) — "Instead of generating two predictions every iteration → use the **historical predictions of the same sample**" → forward 1회. (B)는 Mean Teacher. (C) unlabeled엔 $y_i$가 없음; 실제로 supervised 항은 $B\cap L$에만 걸리고 consistency 항이 $B$ 전체에 걸림. (D)는 어느 알고리즘에도 없음 (원본 vs strong aug를 맞추는 건 "noisy samples as learning targets" 쪽, p.30). (p.19–20)

### Q10 [MCQ] ★
Which statement correctly contrasts Temporal Ensembling (TE) with Mean Teacher (MT)?
(A) TE keeps an EMA of the model weights, whereas MT keeps an EMA of per-sample predictions.
(B) TE stores an $N\times C$ prediction memory whose targets are updated once per epoch, whereas MT stores no per-sample predictions and updates its teacher every iteration.
(C) In both methods, the consistency loss is backpropagated into whatever produces the target.
(D) TE needs two forward passes per input, whereas MT needs only one.
**Answer**: (B) — TE 한계 "per training sample / once per epoch / difficult to scale" ↔ MT 장점 "No per-sample prediction memory / Teacher updated every iteration / Scales well". (A) 정반대 (TE = prediction EMA, MT = weight EMA). (C) TE target은 메모리 상수, MT teacher는 "No backpropagation". (D) 반대 — TE가 1회, MT는 student + teacher 두 번 forward (backprop은 student만). (p.22–25)

### Q11 [T/F]
In the Π-model pseudocode, $z_i$ and $\tilde z_i$ are produced by two different networks with separate parameters (a student and a teacher).
**Answer**: False — 둘 다 **같은 $f_\theta$ (같은 θ)**. 다른 건 dropout mask와 augmentation뿐. student/teacher 두 network(θ, θ')는 Mean Teacher. (p.17–18, p.24)

### Q12 [T/F]
In both the Π-model and Temporal Ensembling pseudocode, the supervised loss is computed only over labeled samples ($i\in B\cap L$), whereas the unsupervised loss is computed over all samples in the minibatch ($i\in B$), including unlabeled ones.
**Answer**: True — 두 알고리즘 loss 식이 완전히 같음. consistency 항은 label이 필요 없으니 $B$ 전체 → unlabeled data가 여기서 학습 신호를 줌. (p.18, p.20)

### Q13 [T/F]
In the Π-model pseudocode, the unsupervised loss component is the cross-entropy between $z_i$ and $\tilde z_i$.
**Answer**: False — **squared difference** $\frac{1}{C|B|}\sum_{i\in B}\|z_i-\tilde z_i\|^2$ = **MSE** (p.17 $\mathcal L_u^{\Pi}=\sum\mathrm{MSE}(\cdot,\cdot)$, 그림의 "squared difference" 블록). cross-entropy는 **supervised** 항($-\log z_i[y_i]$). (p.17–18)

### Q14 [T/F] (low priority — caption 디테일)
According to the caption of Algorithm 2, the updates of $Z$ and $\tilde z$ must be performed between epochs; moving them inside the minibatch loop would break Temporal Ensembling.
**Answer**: False — caption: "the updates of $Z$ and $\tilde z$ **could equally well be done inside the minibatch loop**; in this pseudocode they occur between epochs **for clarity**." 루프 안에 넣어도 각 샘플은 epoch당 한 번만 보이니 샘플별 target은 여전히 **once per epoch** 갱신 (p.22 한계는 그대로). (p.20, p.22)

### Q15 [T/F] ★
In Temporal Ensembling, if the ensembling momentum is set to $\alpha = 0$, the target $\tilde z_i$ used during epoch $t+1$ is simply the network's prediction for sample $i$ from epoch $t$ (no averaging over earlier epochs).
**Answer**: True — Require가 $0\le\alpha<1$이라 $\alpha=0$ 허용. $Z\leftarrow 0\cdot Z+1\cdot z=z$, $1-0^t=1$ → $\tilde z=z$ (직전 epoch 예측 하나). $\alpha$가 클수록 더 먼 과거까지 평균 → 더 안정적인 ensemble target. 어느 경우든 forward는 1회. (p.20)

<!-- WRITTEN 2026-10-22 | author: opus | PC3 -->

<!-- VERIFIED 2026-10-22 | checked: 15 | fixed: 1 | deleted: 0 | reviewer: opus -->
<!-- A/B: Alg.1 (p.18), Alg.2 (p.20) transcriptions re-rendered at 200dpi and match line by line (incl. g(x_{i∈B}, t), 0 ≤ α < 1, Z/(1−α^t), caption); flags (Sajjadi citation, t inside g, notes 06 discrepancy) confirmed real; section B consistent with p.19–29 -->
<!-- fix Q8: stem reworded ("actually performs the averaging ...") so option (C) z̃ ← Z/(1−α^t), the line that assigns the target, cannot be argued as "the line that makes the target" -->
