# Mock Quiz 1 — Advanced Representation Learning (Lectures 01–07)

> 20 questions · T/F + MCQ · 실전처럼 답 먼저 다 적고 맨 아래 정답 확인. 강의 태그 `[Lnn]`는 출처 강의.

### 1. [T/F] `[L01]`
To a computer, an image is initially just a matrix of numbers.

### 2. [MCQ] `[L01]`
Which of the following best describes a representation, according to the lecture?
(A) A transformation of raw data into a form that makes downstream tasks easier
(B) A set of hand-crafted rules that maps inputs to outputs
(C) A lossless copy of the raw pixel values stored in a smaller file format
(D) The final classifier output (e.g., the label "Dog")

### 3. [T/F] `[L02]`
Self-supervised learning obtains "labels" from the data itself by using a "semi-automatic" process, for example by predicting part of the data from other parts.

### 4. [T/F] `[L02]`
A self-supervised learning method is mainly judged by how accurately it solves its pretext task.

### 5. [MCQ] `[L02]`
According to the SSL paradigm, which statement correctly describes the **downstream task**?
(A) It is defined from the data itself and needs no manual annotation.
(B) It is the task used to generate pseudo labels automatically.
(C) It is the application you care about, and its dataset is small and labeled.
(D) It is always trained on a larger dataset than the pretext task.

### 6. [T/F] `[L03]`
In a Siamese network, the two inputs are processed by the same network with shared weights, so that both images are mapped into the same embedding space.

### 7. [T/F] `[L03]`
A key strength of the contrastive loss (Chopra et al., 2005) is that it explicitly enforces the relative ordering between positive and negative distances.

### 8. [MCQ] `[L03]`
Which of the following is NOT listed as a problem of the triplet loss?
(A) The number of possible triplets grows as $N^3$ for $N$ samples.
(B) It requires a pretrained generative model to create positive samples.
(C) Easy triplets dominate, so most triplets are useless.
(D) Randomly chosen negatives often already satisfy the margin, so selecting hard negatives is crucial.

### 9. [T/F] `[L04]`
Debiased contrastive loss estimates the true negative distribution as (random sampling distribution − positive contribution) divided by the probability of observing a different class.

### 10. [T/F] `[L04]`
In DINO, the teacher is trained by backpropagation to match the student's output distribution, and the student is updated as an EMA of the teacher.

### 11. [MCQ] `[L04]`
Why does a large batch size help contrastive methods based on InfoNCE such as SimCLR?
(A) It provides a large number of diverse in-batch negative samples.
(B) It removes the need for data augmentation.
(C) It allows the target network to be updated by EMA.
(D) It guarantees that no false negatives are sampled.

### 12. [T/F] `[L05]`
In instance discrimination (Wu et al.), each image is considered a unique class: the positive pair is the same image and the negative pairs are all other images.

### 13. [T/F] `[L05]`
Compared with mini-batch negatives, memory bank negatives are limited in number but always fresh.

### 14. [MCQ] `[L05]`
How is the key encoder updated in MoCo?
(A) By backpropagating the contrastive loss through the keys
(B) $\theta_q \leftarrow m\theta_q + (1-m)\theta_k$
(C) $\theta_k \leftarrow m\theta_k + (1-m)\theta_q$
(D) By copying the query encoder weights at every iteration

### 15. [T/F] `[L06]`
Active learning selects the most valuable unlabeled samples to be labeled next, helping us act smartly with a limited budget.

### 16. [T/F] `[L06]`
In the SSL objective $\mathcal L = \mathcal L_s + \mu(t)\mathcal L_u$, the unsupervised loss weight $\mu(t)$ is decreasing in time.

### 17. [MCQ] `[L06]`
The statement "Small perturbations should not change the label" belongs to which SSL hypothesis?
(A) Cluster assumption
(B) Smoothness assumption
(C) Low-density separation
(D) Manifold assumption

### 18. [T/F] `[L07]`
Pseudo labeling can be viewed as the hard-label approximation of entropy regularization.

### 19. [T/F] `[L07]`
Minimizing prediction entropy on unlabeled data moves the decision boundary toward high-density regions of the data.

### 20. [MCQ] `[L07]`
Which of the following is NOT one of the methods proposed by Arazo et al. to reduce confirmation bias?
(A) MixUp with soft labels
(B) Oversampling labeled data
(C) Dropout regularization
(D) Updating the teacher based on how much the student improves

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

**15.** `[L06]` True — 슬라이드 원문 그대로. (p.5)

**16.** `[L06]` False — $\mu(t)$는 **increasing** in time (초반 예측이 부정확하니 점점 키움). (p.10)

**17.** `[L06]` (B) — H1 Smoothness: nearby samples → similar predictions, small perturbations → same label. Consistency regularization의 직접 근거. (p.11)

**18.** `[L07]` True — confident soft prediction을 argmax hard label로 바꾼 것 (p.7)

**19.** `[L07]` False — **low**-density regions로 이동, better-separated clusters (p.6)

**20.** `[L07]` (D) — 그건 Meta Pseudo Labels의 key idea (p.21, p.24)

</details>
