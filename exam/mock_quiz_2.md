# Mock Quiz 2 — Advanced Representation Learning (Lectures 01–07)

> 20 questions · T/F + MCQ · 실전처럼 답 먼저 다 적고 맨 아래 정답 확인. 강의 태그 `[Lnn]`는 출처 강의.

### 1. [T/F] `[L01]`
A good representation keeps all information in the input image, including background, lighting, and camera angle, so that no information is lost.

### 2. [MCQ] `[L01]`
In the order "What AI has been learning," which sequence is correct?
(A) Rules → Patterns → Features → Representations → World Knowledge
(B) Rules → Features → Patterns → World Knowledge → Representations
(C) Patterns → Rules → Features → Representations → World Knowledge
(D) Features → Rules → Patterns → Representations → World Knowledge

### 3. [T/F] `[L02]`
To evaluate an SSL method, a shallow network is attached on the learned feature extractor and trained on the target task with a small amount of labeled data.

### 4. [T/F] `[L02]`
In Context Encoders, the reconstruction (L2) loss makes the inpainted region sharp, while the adversarial loss alone causes blurry results.

### 5. [MCQ] `[L02]`
Which pretext task is matched with the learned concept **"Global Spatial Structure"**?
(A) Relative Position
(B) Inpainting
(C) Rotation
(D) Jigsaw

### 6. [T/F] `[L03]`
The N-pair loss compares one positive against multiple negatives simultaneously, and its form is equivalent to the cross entropy loss of an N-way softmax classifier.

### 7. [T/F] `[L03]`
Hard negatives are samples that are close to the anchor but belong to a different class, and they provide weaker learning signals than easy negatives.

### 8. [MCQ] `[L03]`
Why is the `max` over negatives in the Lifted Structured Loss replaced by log-sum-exp?
(A) Log-sum-exp ignores all negatives except the hardest one, making the gradient sparse.
(B) Log-sum-exp gives every negative the same uniform weight, like an average.
(C) Log-sum-exp removes the need for a positive pair in the loss.
(D) `max` is not smooth; log-sum-exp is a smooth approximation of max that gives dense gradients with larger weights on harder negatives.

### 9. [T/F] `[L04]`
In SimCLR, the loss is computed on the projection head output, and after training the projection head g is thrown away and the representation h is used.

### 10. [T/F] `[L04]`
Like AutoAugment, Population Based Augmentation (PBA) learns a single fixed augmentation policy before training.

### 11. [MCQ] `[L04]`
Which statement correctly describes RandAugment?
(A) It uses an RNN controller trained with reinforcement learning to maximize validation accuracy.
(B) It evolves augmentation schedules using population-based training.
(C) It randomly selects augmentations and uses only the number of operations (N) and one global magnitude (M).
(D) It enforces KL-divergence consistency between original and augmented unlabeled data.

### 12. [T/F] `[L05]`
MoCo v2 adopts SimCLR's MLP projection head and stronger data augmentation while retaining MoCo's momentum encoder and queue, which lets it outperform SimCLR with a smaller batch size.

### 13. [T/F] `[L05]`
DeepCluster uses k-means clusters to build positive/negative pairs and trains the network with a contrastive loss on those pairs.

### 14. [MCQ] `[L05]`
According to the comparison of End-to-End, Memory Bank, and MoCo, which row correctly describes MoCo?
(A) Key: current encoder / Negative source: mini-batch / Dictionary: small / Freshness: fresh
(B) Key: momentum encoder / Negative source: FIFO queue / Dictionary: large / Freshness: consistent
(C) Key: memory bank / Negative source: memory bank / Dictionary: large / Freshness: stale
(D) Key: momentum encoder / Negative source: mini-batch / Dictionary: small / Freshness: consistent

### 15. [T/F] `[L06]`
In Mean Teacher, the teacher's parameters are updated only by EMA of the student, and no backpropagation is applied to the teacher.

### 16. [T/F] `[L06]`
The low-density separation assumption states that decision boundaries should lie in high-density regions.

### 17. [MCQ] `[L06]`
What is the main motivation of Temporal Ensembling over the Π-model?
(A) The Π-model cannot use unlabeled data.
(B) The Π-model stores a prediction for every training sample, which is memory intensive.
(C) The Π-model requires two forward passes for every input, which is computationally expensive.
(D) The Π-model uses ground-truth labels to find adversarial perturbations.

### 18. [T/F] `[L07]`
In Noisy Student, noise such as RandAugment, Dropout, and Stochastic Depth is added when training the student, which prevents the student from simply copying the teacher.

### 19. [T/F] `[L07]`
Compared with pseudo labeling, Label Propagation is confidence-based and predicts each sample independently.

### 20. [MCQ] `[L07]`
How does MixMatch produce the guessed label for an unlabeled sample?
(A) It averages the model's predictions over K augmentations and then sharpens the result with a temperature T
(B) It takes the argmax of a single weakly augmented prediction and keeps it only if it exceeds a confidence threshold
(C) It diffuses labels from labeled samples over a kNN similarity graph in feature space
(D) It uses a teacher network that is trained with feedback from the student's labeled loss

---

<details>
<summary><b>정답 & 해설 (클릭)</b></summary>

**1.** `[L01]` False — background, lighting, noise, camera angle은 irrelevant information으로 discard (p.11)

**2.** `[L01]` (A) — From rules to world knowledge (p.7)

**3.** `[L02]` True — 평가 절차 그대로 (p.18)

**4.** `[L02]` False — 반대. L2만 쓰면 blurry, adversarial이 sharp/realistic하게 만들고, 둘을 함께 쓰는 게 최고 (p.25–27)

**5.** `[L02]` (D) — Relative Position = Local Spatial Context, Inpainting = Context & Object Parts, Rotation = Object-Centric Understanding (p.45)

**6.** `[L03]` True — "1 positive <-> many negatives", "Cross entropy loss for a N-way softmax classifier". (p.17–20)

**7.** `[L03]` False — hard negative는 anchor와 가깝지만 다른 class라서 **stronger** learning signal을 줌. easy negative가 loss·gradient 기여가 작음. (p.14)

**8.** `[L03]` (D) — "not smooth, so relaxed"; LSE = dense & focused gradient (softmax weight). (A)는 max, (B)는 sum의 특징. (p.15–16)

**9.** `[L04]` True — "return f, throw away g"; $h$가 더 많은 정보를 보존. (p.24–25, p.28)

**10.** `[L04]` False — PBA는 PBT로 학습 중 online으로 진화하는 **schedule**을 학습 ("In contrast to AutoAugment, PBA learns a schedule instead of a fixed policy"). (p.17)

**11.** `[L04]` (C) — (A) AutoAugment, (B) PBA, (D) UDA. (p.16)

**12.** `[L05]` True — negative 수와 batch size를 decouple해서 256 batch로 SimCLR 8192를 이김 (p.16, p.18)

**13.** `[L05]` False — DeepCluster는 pos/neg pair를 만들지 않고 cluster 결과를 pseudo label로 써서 classification 학습 (p.23–24)

**14.** `[L05]` (B) — (A)는 End-to-End, (C)는 Memory Bank. MoCo = 두 방식의 hybrid (p.14)

**15.** `[L06]` True — Teacher: "Parameters updated only by EMA / No backpropagation". (p.24)

**16.** `[L06]` False — boundary는 **low-density** region에 있어야 하고, cluster를 관통하지 말아야 함. (p.13)

**17.** `[L06]` (C) — "Π-model requires two forward passes for every input → high computational cost". (B)는 오히려 TE의 한계(→ Mean Teacher의 동기). (p.19, p.22)

**18.** `[L07]` True — student에만 noise → 더 robust한 decision boundary (p.17)

**19.** `[L07]` False — 그건 Pseudo Labeling. LP는 **similarity-based**, neighborhood relationships 활용 (p.13)

**20.** `[L07]` (A) — label guessing(평균) → sharpening. (B) FixMatch, (C) Label Propagation, (D) Meta PL (p.32)

</details>
