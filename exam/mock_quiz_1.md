# Mock Quiz 1 — Advanced Representation Learning (Lectures 01–07)

> 20 questions · T/F + MCQ · 실전처럼 답 먼저 다 적고 맨 아래 정답 확인. 강의 태그 `[Lnn]`는 출처 강의.

### 1. [T/F] `[L01]`
To a computer, an image is initially just a matrix of numbers.
> True

**채점** ✅

### 2. [MCQ] `[L01]`
Which of the following best describes a representation, according to the lecture?
(A) A transformation of raw data into a form that makes downstream tasks easier
(B) A set of hand-crafted rules that maps inputs to outputs
(C) A lossless copy of the raw pixel values stored in a smaller file format
(D) The final classifier output (e.g., the label "Dog")
> A

**채점** ✅

### 3. [T/F] `[L02]`
Self-supervised learning obtains "labels" from the data itself by using a "semi-automatic" process, for example by predicting part of the data from other parts.
> True

**채점** ✅

### 4. [T/F] `[L02]`
A self-supervised learning method is mainly judged by how accurately it solves its pretext task.
> False

**채점** ✅

### 5. [MCQ] `[L02]`
According to the SSL paradigm, which statement correctly describes the **downstream task**?
(A) It is defined from the data itself and needs no manual annotation.
(B) It is the task used to generate pseudo labels automatically.
(C) It is the application you care about, and its dataset is small and labeled.
(D) It is always trained on a larger dataset than the pretext task.
> C

**채점** ✅

### 6. [T/F] `[L03]`
In a Siamese network, the two inputs are processed by the same network with shared weights, so that both images are mapped into the same embedding space.
> True

**채점** ✅

### 7. [T/F] `[L03]`
A key strength of the contrastive loss (Chopra et al., 2005) is that it explicitly enforces the relative ordering between positive and negative distances.
> False?

**채점** ✅ (불안) — contrastive loss의 한계가 "상대 순서 강제 X" → 그래서 triplet 등장

### 8. [MCQ] `[L03]`
Which of the following is NOT listed as a problem of the triplet loss?
(A) The number of possible triplets grows as $N^3$ for $N$ samples.
(B) It requires a pretrained generative model to create positive samples.
(C) Easy triplets dominate, so most triplets are useless.
(D) Randomly chosen negatives often already satisfy the margin, so selecting hard negatives is crucial.
> A?

**채점** ❌ 정답 **(B)** — 생성 모델로 positive 만들기는 GenView 얘기. (A) N³ 폭발은 진짜 triplet 문제

> 아... 어쩐지.. **사실 이거 반대로 품**. Triplet이 일으키는 문제는 뭘지를 골랐는데 여러개가 문제인 것 같아서 고민한 것. B가 아닌건 알았음.

### 9. [T/F] `[L04]`
Debiased contrastive loss estimates the true negative distribution as (random sampling distribution − positive contribution) divided by the probability of observing a different class.
> ? T
(몰라서 필기 봤음. 사실상 틀린거임.)

**채점** 📖 필기 봄 (정답 True) — $p^- = (p - \eta^+p^+)/\eta^-$, η⁻ = 다른 class일 확률

### 10. [T/F] `[L04]`
In DINO, the teacher is trained by backpropagation to match the student's output distribution, and the student is updated as an EMA of the teacher.
> ? F
(몰라서 필기 봤음. 사실상 틀린거임.)

**채점** 📖 필기 봄 (정답 False) — 반대로 씀. **student**가 backprop, **teacher**가 student의 EMA

### 11. [MCQ] `[L04]`
Why does a large batch size help contrastive methods based on InfoNCE such as SimCLR?
(A) It provides a large number of diverse in-batch negative samples.
(B) It removes the need for data augmentation.
(C) It allows the target network to be updated by EMA.
(D) It guarantees that no false negatives are sampled.
> A?

**채점** ✅ (불안) — 큰 batch = in-batch negative가 많고 다양

### 12. [T/F] `[L05]`
In instance discrimination (Wu et al.), each image is considered a unique class: the positive pair is the same image and the negative pairs are all other images.
> T?

**채점** ✅ (불안) — 사진 한 장 = class 하나

### 13. [T/F] `[L05]`
Compared with mini-batch negatives, memory bank negatives are limited in number but always fresh.
> ?? F?

**채점** ✅ **(불안) — mini-batch = 적지만 fresh ↔ memory bank = 많지만 stale**

### 14. [MCQ] `[L05]`
How is the key encoder updated in MoCo?
(A) By backpropagating the contrastive loss through the keys
(B) $\theta_q \leftarrow m\theta_q + (1-m)\theta_k$
(C) $\theta_k \leftarrow m\theta_k + (1-m)\theta_q$
(D) By copying the query encoder weights at every iteration
> ?? C
(아예 몰라서 바로 필기봤어. 음... 근데 합리적인 생각으로.. 이 정도 수준으로 자세한 문제는 안나올거랴.. 이 분량에서는 이 문제가 쉽지는 않아서.)

**채점** 📖 필기 봄 (정답 C) — 갱신되는 key가 왼쪽: $\theta_k \leftarrow m\theta_k + (1-m)\theta_q$, m = 0.999

### 15. [T/F] `[L06]`
Active learning selects the most valuable unlabeled samples to be collected next, helping us act smartly with a limited budget.
> ? T
(몰라서 필기 봤음. 사실상 틀린거임.)

**채점** 📖 필기 봄 (정답 True) — 슬라이드 문장 그대로

### 16. [T/F] `[L06]`
In the SSL objective $\mathcal L = \mathcal L_s + \mu(t)\mathcal L_u$, the unsupervised loss weight $\mu(t)$ is decreasing in time.
> ?? 알려줘!

**채점** ❌ 무응답 → 정답 **False** — μ(t)는 **increasing** (초반 예측은 못 믿으니 작게 시작해 점점 키움, ramp-up)

> 아잇!! 지금 보니까 당연히 True지!! 당연히 unsupervised 데이터가 가면갈수록 많아지잖아!! 그게 대부분의 semi-supervised learning의 기본이잖아!

### 17. [MCQ] `[L06]`
The statement "Small perturbations should not change the label" belongs to which SSL hypothesis?
(A) Cluster assumption
(B) Smoothness assumption
(C) Low-density separation
(D) Manifold assumption
> C

**채점** ❌ 정답 **(B) Smoothness** — "흔들어도 label 같음" = H1. (C) Low-density는 "boundary는 데이터 드문 곳"

> 앗, 생각해보니까 이것도 있었지... 맞다.

### 18. [T/F] `[L07]`
Pseudo labeling can be viewed as the hard-label approximation of entropy regularization.
> T?

**채점** ✅ (불안) — confident 예측을 argmax hard label로 = entropy minimization의 hard 버전

### 19. [T/F] `[L07]`
Minimizing prediction entropy on unlabeled data moves the decision boundary toward high-density regions of the data.
> ?? F?
(몰라서 필기 봤음. 사실상 틀린거임.)

**채점** 📖 필기 봄 (정답 False) — boundary는 **low**-density 쪽으로 밀려남 (17번 H3와 같은 얘기)

### 20. [MCQ] `[L07]`
Which of the following is NOT one of the methods proposed by Arazo et al. to reduce confirmation bias?
(A) MixUp with soft labels
(B) Oversampling labeled data
(C) Dropout regularization
(D) Updating the teacher based on how much the student improves
> C?
(몰라서 필기 봤음. 사실상 틀린거임. 근데 설마 저자 이름으로 물어보지는 않을거임.)

**채점** ❌ (필기 보고도) 정답 **(D)** — Arazo = MixUp+soft label · labeled oversampling · dropout/aug. (D)는 Meta Pseudo Labels

---

<details>
<summary><b>정답 & 해설 (클릭)</b></summary>

**1.** `[L01]` True — 슬라이드 원문 그대로. raw pixel ≠ understanding (p.10)

**2.** `[L01]` (A) — p.12 정의 원문 (p.12)

**3.** `[L02]` True — 슬라이드 정의 그대로 (p.12)

**4.** `[L02]` False — pretext 성능엔 관심 없고 downstream task에서 learned feature가 얼마나 유용한지로 평가 (p.18, p.46)

**5.** `[L02]` (C) — (A), (B)는 pretext task 설명, downstream은 "You do not have large datasets" (p.16)

**6.** `[L03]` True — "Same Network + Shared Weights", "Both images are mapped into the same embedding space." (p.8)

**7.** `[L03]` False — 반대. 슬라이드 한계로 "Does not explicitly enforce relative ordering"이 명시됨 → 이게 triplet loss의 동기. (p.11)

**8.** `[L03]` (B) — triplet loss의 문제는 Triplet Explosion($N^3$)과 easy triplet 지배. 생성 모델 의존은 GenView의 한계. (p.13)

**9.** `[L04]` True — $p_x^- = (p - \eta^+p_x^+)/\eta^-$. (p.8–9)

**10.** `[L04]` False — 반대. **student**가 teacher 분포를 따라가도록 학습, **teacher**가 student의 EMA. (p.39)

**11.** `[L04]` (A) — large batch = 많고 다양한 in-batch negatives. 오히려 random negative가 많아도 false negative는 생길 수 있음. (p.5, p.24)

**12.** `[L05]` True — 슬라이드 정의 그대로 (p.5)

**13.** `[L05]` False — 반대. Mini-batch = limited but fresh, memory bank = large but slightly stale (p.4)

**14.** `[L05]` (C) — key encoder는 query encoder의 momentum(EMA)으로만 갱신, key로는 gradient 안 흐름. (B)는 query/key 뒤바꾼 함정 (p.11–12)

**15.** `[L06]` True — 슬라이드 원문 그대로 ("Select most valuable unlabeled samples to be collected next / Helps us act smartly with a limited budget"). (p.5)

**16.** `[L06]` False — $\mu(t)$는 **increasing** in time (초반 예측이 부정확하니 점점 키움). (p.10)

**17.** `[L06]` (B) — H1 Smoothness: nearby samples → similar predictions, small perturbations → same label. Consistency regularization의 직접 근거. (p.11)

**18.** `[L07]` True — confident soft prediction을 argmax hard label로 바꾼 것 (p.7)

**19.** `[L07]` False — **low**-density regions로 이동, better-separated clusters (p.6)

**20.** `[L07]` (D) — 그건 Meta Pseudo Labels의 key idea (p.21, p.24)

</details>

---

## 채점 결과 (2026-10-29)

- **실력 점수: 11 / 20** (필기 안 보고 맞힌 것만. 필기 보고 맞힌 것까지 치면 16 / 20)
- ❌ 틀림: 8, 16(무응답), 17, 20 · 📖 필기 봄: 9, 10, 14, 15, 19 · ✅인데 불안(?): 7, 11, 12, 13, 18
- 시험 전 다시 볼 것: 📖·❌ 9개 → flow.html 맨 위 **한눈에 요약**에서 해당 카드 (3-2-2 Triplet, 4-1-2 Debiased · 4-3-4 DINO, 5-1-3 MoCo, 6-1-2 목표식 · 6-1-3 가정 H1–H4, 7-1-1 Pseudo-Label · 7-2-1 Arazo · 7-2-2 Meta PL)

### 오답 해설

**8. Triplet loss의 문제가 아닌 것 → (B)**
- triplet은 샘플 3개 조합이라 N개면 **N³개로 폭발**(Triplet Explosion)하고, 대부분 너무 쉬워서(easy) 쓸모없어 → 그래서 **hard negative mining**이 중요해. (A)(C)(D)는 전부 슬라이드에 있는 문제점이야 (L03 p.13).
- (B) "생성 모델로 positive 만들기"는 triplet이 아니라 **GenView**(L03 파트 4).
- 외우기: triplet 문제 = **"너무 많고(N³), 대부분 너무 쉽다"**.

**16. μ(t)는 시간이 갈수록 감소? → False**
- $\mathcal L = \mathcal L_s + \mu(t)\mathcal L_u$에서 $\mathcal L_u$의 target은 결국 모델 자기 예측이야. 학습 초반엔 모델이 엉터리라 그 예측을 믿으면 안 되니까, μ를 0 근처에서 시작해서 **점점 키워**(ramp-up) (L06 p.10).
- RL 비유: critic이 엉망인 초반엔 bootstrapped target을 약하게 믿는 것과 같은 발상.
- 외우기: **μ(t)는 increasing**.

**17. "작은 perturbation은 label을 바꾸면 안 된다" → (B) Smoothness**
- **H1 Smoothness**: 가까운 입력은 같은 예측 → 살짝 흔들어도 label 같음 ← 이 문제. consistency regularization의 근거.
- **H2 Cluster**: 같은 cluster면 같은 label
- **H3 Low-density separation**: decision boundary는 데이터가 **드문 곳**을 지나야 함
- **H4 Manifold**: 데이터는 low-dimensional manifold 위
- 외우기: **"흔들어도 같다" = Smoothness**.

**20. Arazo et al.의 confirmation bias 해법이 아닌 것 → (D)**
- Arazo = **MixUp + soft label**, **labeled oversampling**(batch마다 labeled 최소 몇 개), **dropout·data augmentation** → (C) dropout은 포함돼 (L07 p.21).
- (D) "student가 얼마나 좋아졌는지로 teacher를 업데이트"는 **Meta Pseudo Labels**의 핵심 (L07 p.24).

### 필기 보고 맞힌 것 (한 줄 정리)
- **9** Debiased: random 분포에서 같은 class 몫을 빼고 η⁻(다른 class일 확률)로 나눔 → True.
- **10** DINO: 문장이 반대. "학습하는 쪽 = student, 천천히 따라오는 쪽(EMA) = teacher" — BYOL·MoCo·Mean Teacher 다 같음 → False.
- **14** MoCo: key encoder는 backprop 없이 query의 EMA로만. 식은 **갱신되는 쪽(k)이 왼쪽** → (C). (B)는 q/k 뒤바꾼 함정.
- **15** Active learning: 가치 큰 unlabeled 샘플을 골라 label → 적은 예산으로 똑똑하게 → True.
- **19** Entropy minimization: 예측을 확신 있게 → boundary가 **low**-density로 → False.

### 메모에 대한 답
- **14 "이 정도로 자세한 건 안 나올 듯"** → 반은 동의. 식을 통째로 쓰라는 문제는 어렵겠지만, **momentum encoder가 MoCo의 core idea 자체**라서 "key encoder는 gradient가 아니라 query의 EMA로 갱신" 정도의 개념형(T/F, q/k 뒤바꾸기)은 나올 확률이 높아. pseudo code(L05 p.12)에도 이 줄이 그대로 있어.
- **20 "저자 이름으로는 안 물을 것"** → 동의. 다만 같은 내용이 "Which is NOT a way to reduce confirmation bias?"처럼 이름 없이 나올 수 있으니, **(D)가 Meta Pseudo Labels 아이디어**라는 구분만 챙겨.
- **9·10·15·19 "필기 봄 = 사실상 틀림"** → 그 기준대로 실력 점수에선 오답 처리했어.

