# Mock Quiz 3 — Advanced Representation Learning (Lectures 01–07)

> 20 questions · T/F + MCQ · 실전처럼 답 먼저 다 적고 맨 아래 정답 확인. 강의 태그 `[Lnn]`는 출처 강의.

### 1. [T/F] `[L01]`
Representation learning aims to map data points to a simpler, organized space, and the representation is often of lower dimension than the input.

### 2. [MCQ] `[L01]`
Which representation of a dog image "aligns image with text semantics"?
(A) Color Histogram
(B) HOG descriptor
(C) CNN Feature
(D) CLIP Embedding

### 3. [T/F] `[L02]`
In colorization, multiple colors can be plausible and regression averages them, so the task is solved as classification over 313 color classes.

### 4. [T/F] `[L02]`
In a Jigsaw-pretrained network, linear classifiers on the last (highest) layer give the best transfer performance, because higher layers generalize best.

### 5. [MCQ] `[L02]`
Why does the Jigsaw pretext task use only a subset of the possible permutations?
(A) 9! = 362,880 permutations are too many classes, so a subset of 64 permutations that maximize pairwise Hamming distance is used.
(B) Most permutations make the image look natural, so they are removed as easy negatives.
(C) A subset of 64 permutations that minimize pairwise Hamming distance makes the puzzle harder.
(D) Only permutations that keep the center tile fixed are needed for 8-way classification.

### 6. [T/F] `[L03]`
Pretext tasks such as rotation prediction, jigsaw puzzles, and colorization are handcrafted objectives, which motivates viewing representation learning more generally as learning similarity.

### 7. [T/F] `[L03]`
The InfoNCE loss uses a binary cross-entropy loss to classify each sample independently as data or noise.

### 8. [MCQ] `[L03]`
In the common setup for modern contrastive learning without labels, how are the positive and negatives obtained for an anchor?
(A) Positive = a sample of the same class; negatives = samples of other classes, using ground-truth labels.
(B) Positive = another augmented view of the same sample; negatives = other samples in the batch.
(C) Positive = the nearest neighbor in the current embedding; negatives = the farthest samples.
(D) Positive = a sample from the same K-means cluster; negatives = samples from other clusters.

### 9. [T/F] `[L04]`
In Barlow Twins, the diagonal elements of the cross-correlation matrix are pushed toward 1 (invariance), and the off-diagonal elements are pushed toward 0 (redundancy reduction).

### 10. [T/F] `[L04]`
In BYOL, both the online and target networks are updated by gradient descent through backpropagation.

### 11. [MCQ] `[L04]`
Which method generates harder negatives by mixing embeddings in the feature space?
(A) Mixup
(B) CutMix
(C) MoCHi
(D) UDA

### 12. [T/F] `[L05]`
At inference, CLIP performs zero-shot classification by encoding candidate text prompts and predicting the prompt with the highest image–text similarity, instead of using a learned classifier.

### 13. [T/F] `[L05]`
SwAV requires offline clustering of the entire dataset and explicit pairwise comparisons between features.

### 14. [MCQ] `[L05]`
Which of the following is NOT true about CURL?
(A) It applies contrastive learning to reinforcement learning.
(B) It uses random crop augmentation to create positive pairs.
(C) It employs a momentum encoder and contrastive loss, following MoCo.
(D) It first pre-trains the encoder with contrastive learning, then freezes it before training the RL policy.

### 15. [T/F] `[L06]`
VAT replaces the ground-truth label used in adversarial training with the model's own current prediction, so the VAT loss can be applied to unlabeled samples as well.

### 16. [T/F] `[L06]`
Temporal Ensembling uses an exponential moving average of the model weights as the consistency target.

### 17. [MCQ] `[L06]`
Compared with Mean Teacher, how does the "noisy samples as learning targets" approach form its consistency target?
(A) It uses self-consistency between the original and strongly augmented inputs with the same model.
(B) It uses an EMA of the student weights as the teacher.
(C) It averages the predictions of two temporal teachers.
(D) It uses the EMA of each sample's past predictions.

### 18. [T/F] `[L07]`
In Meta Pseudo Labels, the teacher is updated so that the student trained on its pseudo labels achieves a lower supervised loss on labeled data.

### 19. [T/F] `[L07]`
The key contribution of MixMatch is proposing a single new component, rather than combining existing SSL techniques.

### 20. [MCQ] `[L07]`
In ReMixMatch's Distribution Alignment, how is the prediction for an unlabeled sample adjusted?
(A) It is replaced by a one-hot label if its maximum probability exceeds a fixed threshold
(B) It is rescaled to further boost classes that already appear too often in the pseudo labels and suppress rare classes
(C) It is rescaled toward the labeled dataset's class distribution, suppressing over-represented classes and boosting under-represented classes
(D) It is averaged with the prediction of a second network trained in parallel

---

<details>
<summary><b>정답 & 해설 (클릭)</b></summary>

**1.** `[L01]` True — "simpler, organized space", "Often in lower dimension" (p.12)

**2.** `[L01]` (D) — Color Histogram = global color distribution, CNN Feature = hierarchical visual features (p.13)

**3.** `[L02]` True — regression → 313 color classes classification (p.37)

**4.** `[L02]` False — "Higher layers do not generalize": res4에서 최고, 마지막 res5에서 하락 → representation이 pretext에 특화 (p.49)

**5.** `[L02]` (A) — Hamming distance를 maximize (minimize 아님). (D)는 relative position과 헷갈리게 한 것 (p.30)

**6.** `[L03]` True — "They are all handcrafted Objectives" → "Representation Learning = Learning Similarity". (p.4–5)

**7.** `[L03]` False — 그건 NCE. InfoNCE는 **categorical cross-entropy**로 unrelated noise samples 중 positive를 식별. (p.23)

**8.** `[L03]` (B) — label 없을 땐 augmentation view가 positive, batch 내 다른 이미지가 negative. (A)는 label 있는 경우. (p.31–32)

**9.** `[L04]` True — target이 identity matrix. (p.35–36)

**10.** `[L04]` False — target은 online의 **EMA**로 갱신, **No backpropagation**. (p.30–31)

**11.** `[L04]` (C) — Mixup은 global pixel-wise, CutMix는 region-level pixel 혼합, UDA는 consistency regularization. (p.19–21)

**12.** `[L05]` True — CLIP Inference의 Key Idea (p.33)

**13.** `[L05]` False — SwAV는 online으로 assignment 계산, explicit pairwise comparison도 offline clustering도 필요 없음 (p.26, p.30)

**14.** `[L05]` (D) — CURL은 contrastive learning과 RL policy/value network를 **simultaneously** 최적화함 (p.20–21)

**15.** `[L06]` True — AT 식의 GT $q(y|\mathbf x^l)$ 자리에 current model prediction $p_{\hat\theta}(y|\mathbf x)$를 넣음 → "The VAT loss applies to both labeled and unlabeled samples." (슬라이드 화살표 라벨 "If GT is known,"은 원문 유지 ← 문맥상 "unknown"으로 읽어야 자연스러움) (p.34)

**16.** `[L06]` False — TE는 sample별 **prediction**의 EMA. weight EMA는 Mean Teacher. (p.19, p.23)

**17.** `[L06]` (A) — "Uses self-consistency between original and augmented inputs instead of an EMA teacher." (B)=MT, (C)=PS-MT, (D)=TE. (p.31)

**18.** `[L07]` True — outer optimization, evaluation is performed using correct labels (p.25)

**19.** `[L07]` False — "Rather than proposing a new component", 여러 SSL 기법을 **unify**한 holistic framework (p.30)

**20.** `[L07]` (C) — pseudo label이 true class 분포를 안 따를 수 있어서, labeled dataset의 class 분포(target prior) 쪽으로 rescale → "suppresses over-represented classes and boosts under-represented classes". (B)는 방향을 뒤집은 함정, (A)는 FixMatch식 threshold (p.37)

</details>
