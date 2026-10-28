# Mock Quiz 4 — Advanced Representation Learning (Lectures 01–07 + Pseudo code)

> 20 questions · **새 형식: T/F 6 + MCQ 7 + 빈칸(Blank) 7** · pseudo code 문항 4개 (`·PC` 태그). 빈칸은 **영어 단어/식을 직접 써서** 답할 것 (철자까지). 실전처럼 답 먼저 다 적고 맨 아래 정답 확인.
> 출처: `exam/_parts/` pool (C·D 섹션 + `pc_1~4`), mock 1~3과 겹치지 않는 문항만. 모두 Opus 독립 검수 완료 (2026-10-22).

### 1. [Blank] `[L01]`
"Compression is not about losing information. It is about removing ______ information."

### 2. [MCQ] `[L02]`
In the Relative Position of Patches pretext task (Doersch et al.), what does the network predict?
(A) The rotation angle applied to the whole image
(B) Which of the 8 neighboring positions a patch came from, relative to the center patch
(C) The original permutation of 9 shuffled tiles
(D) The pixel values of a masked center region

### 3. [T/F] `[L02]`
Equivariance means that the representation changes predictably according to the input transformation, i.e., $f(t(x)) = \phi(t, f(x))$.

### 4. [MCQ] `[L03]`
How does Multistage Contrastive Learning (MCL) differ from hard negative mining?
(A) MCL selects negatives that are closest to the anchor in the current embedding to find difficult samples.
(B) MCL generates new negatives with a diffusion model at every stage.
(C) MCL replaces the binary targets with a soft similarity graph from a text encoder.
(D) MCL selects negatives only from the same cluster of the previous stage, to prevent reusing previously learned features.

### 5. [Blank] `[L03]`
The InfoNCE loss uses (1) ______ cross-entropy loss to identify the positive sample amongst a set of unrelated noise samples. Learning a good $f(\mathbf{x},\mathbf{c})$ means estimating a good density ratio, which increases the (2) ______ information between $\mathbf{x}$ and $\mathbf{c}$.

### 6. [T/F] `[L04]`
Image augmentations for contrastive learning should significantly change the visual appearance of an image while preserving its semantic meaning, so that the resulting views can form positive pairs.

### 7. [MCQ] `[L04]`
According to the lecture, what is the key difference between BYOL and DINO?
(A) BYOL uses an EMA teacher, while DINO does not.
(B) BYOL matches feature vectors with an MSE loss, while DINO matches probability distributions with a cross-entropy loss.
(C) BYOL requires labels, while DINO does not.
(D) BYOL uses a cross-correlation loss, while DINO uses InfoNCE with negatives.

### 8. [Blank] `[L04]`
Barlow Twins computes the (1) ______ matrix between the two views' embeddings along the batch dimension and pushes it toward the identity matrix; the off-diagonal part of this objective is called the (2) ______ term.

### 9. [T/F] `[L04·PC]`
In DINO's pseudo code, centering and sharpening are applied to the student output, while the teacher output goes through a plain softmax with temperature `tps`.

### 10. [MCQ] `[L05]`
What is the key idea of SwAV?
(A) Use k-means cluster IDs from the whole dataset as pseudo labels for classification
(B) Predict the cluster assignment (code) of one augmented view from another view
(C) Keep a FIFO queue of keys encoded by a momentum encoder
(D) Align image and text embeddings in a shared space

### 11. [T/F] `[L05]`
Compared with cross-entropy, SupCon is more sensitive to hyperparameter changes.

### 12. [Blank] `[L05·PC]`
In MoCo's pseudo code, the line `k = k.______()  # no gradient to keys` ensures that the contrastive loss does not backpropagate into the key encoder.

### 13. [Blank] `[L06]`
H1, the ______ assumption: nearby samples should have similar predictions, and small perturbations should not change the label.

### 14. [MCQ] `[L06]`
Which statement best describes the core idea of consistency regularization?
(A) Predictions should remain consistent under small perturbations, which may come from data augmentations or model stochasticity (e.g., Dropout).
(B) High-confidence predictions are converted into hard labels and used as targets with cross-entropy.
(C) Labels are propagated from labeled samples to unlabeled samples over a similarity graph.
(D) Decision boundaries should be placed in high-density regions of the data.

### 15. [T/F] `[L06]`
In ICT, random MixUp samples are likely to lie near the decision boundary, so enforcing consistency on them aligns with the low-density separation assumption.

### 16. [MCQ] `[L06·PC]`
Going from Algorithm 1 (Π-model) to Algorithm 2 (Temporal Ensembling), what is the key change in how the target $\tilde z_i$ is obtained?
(A) $\tilde z_i$ is no longer computed by a second forward pass; it is read from a per-sample memory of EMA predictions that is updated after each epoch.
(B) $\tilde z_i$ is computed by a second network whose weights are an EMA of $\theta$.
(C) $\tilde z_i$ is replaced by the ground-truth one-hot label $y_i$ for every sample.
(D) $\tilde z_i$ is computed by a second forward pass on the original input without augmentation.

### 17. [Blank] `[L07]`
(1) ______ bias: pseudo labels from an imperfect teacher can be incorrect, the student is trained to imitate them, and the student reinforces the teacher's mistakes. It can be reduced with MixUp with soft labels, (2) ______ labeled data, strong data augmentation, and dropout.

### 18. [T/F] `[L07]`
In ReMixMatch's Augmentation Anchoring, pseudo labels are generated from strongly augmented images and consistency is enforced on weakly augmented versions.

### 19. [MCQ] `[L07]`
Which statement correctly describes FixMatch?
(A) Pseudo labels are made from strongly augmented images and the model is trained on weakly augmented images
(B) It relies on sharpening, MixUp, and distribution alignment to obtain stable targets
(C) A pseudo label is generated from a weakly augmented image, kept only if its confidence is high, and the model is trained to predict that label on a strongly augmented image
(D) It uses a separate EMA teacher network to generate soft targets with temperature scaling

### 20. [Blank] `[L07·PC]`
In DivideMix's co-divide step, samples with clean probability $w_i \ge \tau$ form the (1) ______ training set $\mathcal X_e^{(k)}$, while samples with $w_i < \tau$ form the (2) ______ training set $\mathcal U_e^{(k)}$.


---

<details>
<summary><b>정답 & 해설 (클릭)</b></summary>

**1.** `[L01]` irrelevant (accept: task-irrelevant, nuisance, redundant — p.21 "removing redundancy and nuisances") — 압축 = 정보 손실이 아니라 **무관한 정보(nuisance) 제거**. 이점 3개: Less Memory / Invariance / Occam's Razor (p.19)  ← pool: `01_Introduction` F5

**2.** `[L02]` (B) — center patch + 이웃 patch 한 쌍 → 8개 위치 중 어디인지 분류 (X = (center, neighbor), Y = 3). (A) RotNet, (C) Jigsaw, (D) Inpainting (p.21)  ← pool: `02_SSL` Q15

**3.** `[L02]` True — invariance는 $f(t(x)) = f(x)$ (representation 불변) (p.75)  ← pool: `02_SSL` Q17

**4.** `[L03]` (D) — "Rather than finding difficult samples, the goal is to prevent the reuse of previously learned features." (A)는 hard negative mining 정의. (p.56)  ← pool: `03_Contrastive_Learning_I` Q16

**5.** `[L03]` (1) categorical (2) mutual — NCE = binary(sigmoid) vs InfoNCE = N-way **categorical**(softmax). 최적 $f(\mathbf{x},\mathbf{c}) \propto p(\mathbf{x}|\mathbf{c})/p(\mathbf{x})$. "not simply learned to match positives" (p.23–24, p.28)  ← pool: `03_Contrastive_Learning_I` F7

**6.** `[L04]` True — Key Points 그대로: create different views / **significantly change visual appearance** / **preserve the semantic meaning** / form positive pairs. 겉모습은 크게, 의미는 그대로. (p.13)  ← pool: `04_Contrastive_Learning_II` Q21

**7.** `[L04]` (B) — feature matching (MSE, predictor) vs distribution matching (CE, softmax). 둘 다 EMA teacher, no labels. (p.41)  ← pool: `04_Contrastive_Learning_II` Q15

**8.** `[L04]` (1) cross-correlation (2) redundancy reduction — $C$는 $D\times D$ (D = feature dimension). diagonal $C_{ii}\to1$ = **invariance** (같은 feature가 augmentation 후에도 유지), off-diagonal $C_{ij}\to0$ = **redundancy reduction** (feature끼리 decorrelated). shared encoder, no negatives, batch size에 robust. (p.35–38)  ← pool: `04_Contrastive_Learning_II` F8

**9.** `[L04·PC]` False — 반대. **teacher**: `t = softmax((t - C) / tpt, dim=1) # center + sharpen`, **student**: `s = softmax(s / tps, dim=1)`만. `tps` = student temperature, `tpt` = teacher temperature (헤더 주석). 그림에서도 centering 박스는 teacher 쪽에만 있음. (L04 p.40)  ← pool: `pc_2_distill` Q1

**10.** `[L05]` (B) — "Learn by swapping cluster assignments between multiple views." (A) DeepCluster, (C) MoCo, (D) CLIP (p.26)  ← pool: `05_Contrastive_Learning_III` Q8

**11.** `[L05]` False — SupCon은 hyperparameter 변화에 **less** sensitive. CE보다 정확도는 small amount만 높고 ImageNet-C robustness는 더 좋음 (p.42)  ← pool: `05_Contrastive_Learning_III` Q13

**12.** `[L05·PC]` detach (accept: —) — callout "No gradient through the key". 그래서 `loss.backward()` 다음 `update(f_q.params)` → **query encoder만 SGD**, key encoder는 momentum update로만 바뀜. SimCLR와의 key difference: "Compute gradients and update the encoder only through the queries" (L05 p.11). (L05 p.12)  ← pool: `pc_1_infonce` Q4

**13.** `[L06]` smoothness (accept: smooth) — consistency regularization의 직접적 근거 ("perturbation 줘도 예측은 같아야"). H2 cluster (같은 cluster = 같은 label, boundary가 cluster를 쪼개면 안 됨), H3 low-density separation (boundary는 low-density region에), H4 manifold (고차원 데이터는 low-dimensional manifold 위)와 구분. (p.11–14)  ← pool: `06_Semi_Supervised_Learning_I` F3

**14.** `[L06]` (A) — Core idea 원문 그대로; perturbation 출처 = data augmentations / model stochasticity (Dropout). H1 smoothness를 loss로 만든 것. (B)는 Pseudo Label (다음 강의, $\mathcal L_u$ 설계의 다른 축), (C)는 Label Propagation (07강), (D)는 low-density separation을 뒤집은 오답. (p.9, p.16)  ← pool: `06_Semi_Supervised_Learning_I` Q20

**15.** `[L06]` True — 원문 그대로. (p.37)  ← pool: `06_Semi_Supervised_Learning_I` Q8

**16.** `[L06·PC]` (A) — "Instead of generating two predictions every iteration → use the **historical predictions of the same sample**" → forward 1회. (B)는 Mean Teacher. (C) unlabeled엔 $y_i$가 없음; 실제로 supervised 항은 $B\cap L$에만 걸리고 consistency 항이 $B$ 전체에 걸림. (D)는 어느 알고리즘에도 없음 (원본 vs strong aug를 맞추는 건 "noisy samples as learning targets" 쪽, p.30). (p.19–20)  ← pool: `pc_3_consistency` Q9

**17.** `[L07]` (1) Confirmation (2) oversampling (accept: oversample) — 틀린 pseudo label이 누적되며 성능이 점점 하락. Arazo는 regularization으로 틀린 label을 외우기 어렵게 만듦. oversampling labeled data = mini-batch마다 labeled sample 최소 개수 보장 ("A minimum number of labeled samples per mini-batch"). (p.20, p.21, p.62)  ← pool: `07_Semi_Supervised_Learning_II` F4

**18.** `[L07]` False — weak ↔ strong 뒤바뀜. **weak** anchor로 label 만들고 **strong** view에 consistency (p.38)  ← pool: `07_Semi_Supervised_Learning_II` Q7

**19.** `[L07]` (C) — three simple ideas: weak augmentation / confidence threshold / strong augmentation (p.43, p.45)  ← pool: `07_Semi_Supervised_Learning_II` Q17

**20.** `[L07·PC]` (1) labeled (2) unlabeled (clean / noisy는 샘플 성격 묘사로는 맞지만, 슬라이드가 set 이름으로 쓰는 말은 labeled / unlabeled. 시험 답은 이걸로) — 슬라이드 주석 "labeled training set for $\theta^{(k)}$" / "unlabeled training set for $\theta^{(k)}$". 이게 *DivideMix reformulates noisy-label learning as a semi-supervised learning problem* 의 실체. noisy 샘플은 **버리지 않음**, GT label만 버림 (p.53 Q2) (L07 p.50, p.53–54)  ← pool: `pc_4_pseudolabel` Q13

</details>
