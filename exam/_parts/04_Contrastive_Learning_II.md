# 04. Contrastive Learning II — Core Ideas & Question Pool

## A. Core ideas (one block per method/paper/concept, in lecture order)

### Key ingredients of contrastive learning (p.4–6)
- **Problem**: What makes contrastive learning work well without labels?
- **Core idea**: Three ingredients — **heavy data augmentation** (noise versions of data used as positives, without modifying semantic meaning), **large batch size** (many diverse in-batch negatives, essential for InfoNCE methods like SimCLR and CLIP), and **hard negative mining** (negatives close to the anchor give more informative supervision).
- **vs. previous / key difference**: Hard negatives are easy to find in supervised learning (class labels) but challenging in self-supervised learning; larger batches or memory banks raise the chance of finding them, at a higher memory cost.
- **Keywords**: noise version, in-batch negatives, diversity, hard negative, memory bank, trade-off
- 한 줄: augmentation은 positive를, 큰 batch는 negative 양·다양성을, hard negative는 negative의 질을 책임짐.

### Sampling bias & Debiased Contrastive Loss (Chuang et al., 2020, p.7–11)
- **Problem**: Random negative sampling may include **false negatives** (same-class samples), which leads to a big performance drop.
- **Core idea**: Estimate the **true negative distribution** without labels: $p_x^- = \frac{p - \eta^+ p_x^+}{\eta^-}$, i.e., *True Negative = (Random Sampling − Positive Contribution) / $\eta^-$*, and plug this estimate into the InfoNCE denominator.
- **vs. previous / key difference**: Standard InfoNCE treats every random sample as a negative (biased); debiased loss corrects it using only random samples and positives.
- **Keywords**: false negative, sampling bias, true negative distribution, $\eta^+$ (class prior), positive contribution, Biased vs Unbiased
- 한 줄: random 분포에서 positive 성분을 빼서 진짜 negative 분포를 추정 → label 없이 false negative 보정.

### Image augmentation for contrastive learning (p.13–14)
- **Problem**: How should positive views be built?
- **Core idea**: Create different views of the same image that **significantly change visual appearance** while **preserving semantic meaning**, forming positive pairs.
- **Keywords**: views, visual appearance, semantic meaning, random crop, color jitter, multi-crop (SwAV)
- 한 줄: 겉모습은 크게 바꾸고 의미는 유지 = 좋은 positive.

### AutoAugment (Cubuk et al., CVPR 2019, p.15)
- **Problem**: Hand-designing augmentation policies is hard.
- **Core idea**: Formulate augmentation search as a **reinforcement learning** problem (RNN controller) that maximizes **validation accuracy**. Key Idea: *Learn which augmentations to apply.*
- **Keywords**: reinforcement learning, controller, child network, validation accuracy, fixed policy
- 한 줄: RL로 best augmentation policy 탐색 (비싸다).

### RandAugment (Cubuk et al., 2019, p.16)
- **Problem**: AutoAugment's search is expensive (huge search space: which operation, what probability, what magnitude).
- **Core idea**: **Randomly** select augmentations and control everything with only **N** (number of operations) and **M** (one global magnitude).
- **vs. previous / key difference**: Eliminates the expensive search → very small search space; simpler and more efficient than AutoAugment.
- **Keywords**: random selection, N, M, single global magnitude, small search space
- 한 줄: 탐색 대신 랜덤 + hyperparameter 2개 (N, M).

### PBA: Population Based Augmentation (Ho et al., ICML 2019, p.17)
- **Problem**: A fixed augmentation policy may not be best throughout training.
- **Core idea**: Evolve augmentation policies **during training** with **population-based training (PBT)**. Key Idea: *Learn augmentations online via evolutionary optimization.*
- **vs. previous / key difference**: Unlike AutoAugment, PBA learns a **schedule** instead of a **fixed policy**.
- **Keywords**: PBT, evolutionary, online, schedule, continuously updates
- 한 줄: PBT로 학습 중 augmentation schedule을 진화시킴.

### UDA: Unsupervised Data Augmentation (Xie et al., NeurIPS 2020, p.18)
- **Problem**: How to learn from **unlabeled** data using augmentation?
- **Core idea**: Apply augmentations to unlabeled data and enforce **prediction consistency** between original and augmented inputs by minimizing **KL divergence**. Key Idea: *consistency regularization.*
- **Keywords**: unlabeled data, consistency regularization, KL divergence, RandAugment, back translation
- 한 줄: unlabeled에 강한 aug → 원본과 예측이 같도록 KL.

### Mixup (Zhang et al., ICLR 2018, p.19)
- **Core idea**: **Global-level** mixture — a weighted **pixel-wise** combination of two images (and their labels): $\tilde x = \lambda x_i + (1-\lambda)x_j$.
- **Keywords**: global-level, pixel-wise, weighted combination, $\lambda$
- 한 줄: 두 이미지를 통째로 반투명 섞기.

### CutMix (Yun et al., ICCV 2019, p.20)
- **Core idea**: **Region-level** mixture — paste a local region of one image onto the rest of another; labels mixed by area ratio.
- **vs. previous / key difference**: Unlike Mixup (global) and Cutout (removes region), CutMix fills the region with real pixels → improves classification, localization and detection.
- **Keywords**: region-level, binary mask $M_b$, area ratio label, localizable features
- 한 줄: 패치 잘라 붙이기, label은 면적비.

### MoCHi: Mixing of Contrastive Hard Negatives (Kalantidis et al., NeurIPS 2020, p.21)
- **Problem**: Most negatives are easy and give little signal.
- **Core idea**: Identify the hardest negatives, then synthesize harder negatives by **interpolating hard-negative embeddings** and **mixing the query with hard negatives**. Key Idea: *mix embeddings in the feature space.*
- **vs. previous / key difference**: Mixup/CutMix mix in pixel space; MoCHi mixes in **feature (embedding) space** to create negatives.
- **Keywords**: hardest negatives, synthetic hard negatives, feature space, query mixing
- 한 줄: feature 공간에서 hard negative를 합성.

### SimCLR (Chen et al., ICML 2020, p.23–29)
- **Problem**: A simple framework for contrastive learning of visual representations.
- **Core idea**: Maximize agreement between two differently augmented views of the same sample with a contrastive (InfoNCE / NT-Xent) loss on a **projection head** output $z = g(h)$, using all other $2(N-1)$ in-batch samples as negatives.
- **vs. previous / key difference**: No memory bank — in-batch negatives only; hence needs a **large batch size** (TPU training). After training, **throw away $g$** and use $h$.
- **Keywords**: 2N augmented samples, cosine similarity, temperature $\tau$, projection head, large batch, linear evaluation
- 한 줄: 같은 이미지 두 view는 당기고 batch 내 나머지는 밀기, 큰 batch 필수.

### BYOL: Bootstrap Your Own Latent (Grill et al., NeurIPS 2020, p.30–34)
- **Problem**: SimCLR depends on many negatives (large batch).
- **Core idea**: **Online** network ($\theta$, with a **predictor**) predicts the representation of another view produced by a **target** network ($\xi$), which is updated by **EMA** with **no backpropagation** (stop-gradient); loss = **MSE** between normalized vectors.
- **vs. previous / key difference**: **No negative samples** — learning by prediction instead of contrast.
- **Keywords**: online/target, predictor, EMA, stop-gradient, MSE, bootstrap
- 한 줄: negative 없이 EMA target의 출력을 predictor로 예측.

### Barlow Twins (Zbontar et al., ICML 2021, p.35–38)
- **Problem**: Avoid collapse without negatives (and without predictor/EMA tricks).
- **Core idea**: Compute the **cross-correlation matrix** between the two views' embeddings **along the batch dimension** ($D\times D$) and push it to the **identity**: diagonal → 1 (**invariance**), off-diagonal → 0 (**redundancy reduction**).
- **vs. previous / key difference**: Uses a **shared encoder**, no negatives; naturally avoids collapsed (trivial constant) representations and is **robust to batch size**.
- **Keywords**: cross-correlation, identity matrix, invariance term, redundancy reduction, decorrelated, batch dimension
- 한 줄: feature 차원 간 상관행렬을 identity로 → 불변 + 비중복.

### DINO: Self-Distillation with No Labels (Caron et al., ICCV 2021, p.39–44)
- **Problem**: Learn semantic representations without labels via distillation.
- **Core idea**: **Student** and **teacher** with identical architectures; the student is trained to match the teacher's **output (softmax) distribution** with **cross-entropy**; the teacher is an **EMA of the student** (centering + sharpening on teacher outputs).
- **vs. previous / key difference**: BYOL = feature matching with MSE + predictor; DINO = **distribution matching** with **cross-entropy** on softmax probabilities.
- **Keywords**: self-distillation, student/teacher, EMA teacher, cross-entropy, centering, emergent self-attention (object segmentation)
- 한 줄: 학생이 EMA 선생의 확률분포를 CE로 따라감 → ViT attention이 객체를 찾음.

### SynCLR (Tian et al., CVPR 2024, "Learning Vision from Models Rivals Learning Vision from Data", p.45–54)
- **Problem**: Can positive views be **generated** rather than augmented?
- **Core idea**: Llama-2 generates captions; a text-to-image model generates **4 images per caption** from different random noise (low CFG for diversity) → images from the same caption form a **positive set**, trained with a **multi-positive contrastive loss**.
- **vs. previous / key difference**: SimCLR preserves **image identity** (positive pair from augmentation); SynCLR preserves **caption-level semantics** (positive set from generation) — positives need not come from the same image.
- **Keywords**: synthetic captions, Llama-2, text-to-image, positive set, caption granularity, multi-positive, no real images
- 한 줄: 같은 caption으로 생성한 이미지들 = positive, real image 없이 학습.

### DINOv2 (Oquab et al., TMLR 2024, p.55–67)
- **Problem**: How can DINO become a **general-purpose visual foundation model**?
- **Core idea**: **DINOv2 = DINO + iBOT + KoLeo + Curated Data + Scaling** — image-level (CLS) + patch-level (masked iBOT) self-distillation, KoLeo + Sinkhorn-Knopp to prevent crowding/collapse, curated LVD-142M data, and distilling a large ViT-g teacher into smaller models.
- **vs. previous / key difference**: Not a single new loss but an **integrated system**; learns from **images alone (no labels, no captions)**, giving frozen features for both global and dense tasks.
- **Keywords**: iBOT, KoLeo, Sinkhorn-Knopp, LVD-142M, deduplication, retrieval, ViT-g distillation, frozen features
- 한 줄: DINO를 데이터·patch loss·안정화·distill로 키운 범용 backbone.

### DINOv3: Gram Anchoring (p.68–70)
- **Problem**: Longer training improves global features but **dense features gradually degrade** (noisy patch similarity, segmentation drops).
- **Core idea**: Select an **earlier model with clean patch features** and preserve its **pairwise patch-similarity structure (Gram matrix)** during longer training: $\|X_SX_S^\top - X_GX_G^\top\|_F^2$.
- **vs. previous / key difference**: Matches the similarity structure among patches, not the features themselves.
- **Keywords**: Gram matrix, earlier model, patch similarity, dense feature degradation, spatial consistency
- 한 줄: 깨끗했던 옛 체크포인트의 patch 유사도 구조를 붙잡아 dense 열화 방지.

## B. Must-know definitions & distinctions

- **Hard negative**: a negative that is close to the anchor in embedding space (more informative than easy negatives).
- **False negative (sampling bias)**: a "negative" that actually has the same class as the anchor.
- **True negative distribution**: $p_x^- = (p - \eta^+ p_x^+)/\eta^-$.
- **Projection head $g$**: loss is computed on $z=g(h)$; $z$ becomes invariant to transformations, so $h$ keeps more information and is used downstream.
- **EMA (BYOL/DINO)**: target/teacher weights = moving average of online/student weights; no backpropagation into target/teacher (stop-gradient).
- **Invariance term vs redundancy reduction term** (Barlow Twins): diagonal $C_{ii}\to1$ vs off-diagonal $C_{ij}\to0$.
- **Positive set** (SynCLR): images generated from the same caption.
- **KoLeo**: spreads image-level features over the embedding space (prevents crowding). **Sinkhorn-Knopp centering**: re-normalizes teacher outputs so prototype assignments are evenly distributed.
- **Gram matrix**: pairwise similarities between patch features ($XX^\top$).
- **AutoAugment vs RandAugment vs PBA**: RL search (fixed policy) vs random with N, M vs PBT (schedule).
- **Mixup vs CutMix vs MoCHi**: global pixel-wise vs region-level pixel vs feature-space hard negatives.
- **SimCLR vs BYOL**: negatives (InfoNCE) vs no negatives (prediction, MSE).
- **BYOL vs DINO**: feature matching / MSE / predictor vs distribution matching / cross-entropy / softmax (both: EMA teacher, no labels).
- **SimCLR vs Barlow Twins**: sample-to-sample similarity vs feature-to-feature cross-correlation ($D\times D$, along batch).
- **SimCLR vs Supervised vs SynCLR positives**: each image = own class vs all same-label images = one class (ignores actions/backgrounds) vs same caption = one class.
- **DINO vs DINOv2**: ImageNet-1K, image-level, ViT-S/B vs curated 142M, image + patch-level, up to ViT-g (~1B).
- **DINOv2 distillation vs DINOv3 Gram anchoring**: transfer a frozen large teacher's representation vs preserve an earlier model's patch similarity structure.

## C. Question pool (21 questions)

### Q1 [T/F]
Hard negatives are negatives that are close to the anchor in the embedding space, but they provide less informative supervision than easy negatives.
**Answer**: False — hard negative는 anchor에 가까운 negative라서 easy negative보다 **더** informative ("Provide more informative supervision than easy negatives"). (p.6)

### Q2 [T/F]
Debiased contrastive loss estimates the true negative distribution as (random sampling distribution − positive contribution) divided by the probability of observing a different class.
**Answer**: True — $p_x^- = (p - \eta^+p_x^+)/\eta^-$. (p.8–9)

### Q3 [T/F]
Like AutoAugment, Population Based Augmentation (PBA) learns a single fixed augmentation policy before training.
**Answer**: False — PBA는 PBT로 학습 중 online으로 진화하는 **schedule**을 학습 ("In contrast to AutoAugment, PBA learns a schedule instead of a fixed policy"). (p.17)

### Q4 [T/F]
In SimCLR, the loss is computed on the projection head output, and after training the projection head g is thrown away and the representation h is used.
**Answer**: True — "return f, throw away g"; $h$가 더 많은 정보를 보존. (p.24–25, p.28)

### Q5 [T/F]
In BYOL, both the online and target networks are updated by gradient descent through backpropagation.
**Answer**: False — target은 online의 **EMA**로 갱신, **No backpropagation**. (p.30–31)

### Q6 [T/F]
In Barlow Twins, the diagonal elements of the cross-correlation matrix are pushed toward 1 (invariance), and the off-diagonal elements are pushed toward 0 (redundancy reduction).
**Answer**: True — target이 identity matrix. (p.35–36)

### Q7 [T/F]
In DINO, the teacher is trained by backpropagation to match the student's output distribution, and the student is updated as an EMA of the teacher.
**Answer**: False — 반대. **student**가 teacher 분포를 따라가도록 학습, **teacher**가 student의 EMA. (p.39)

### Q8 [T/F]
In SynCLR, positive pairs do not need to originate from the same image, and representation learning can be performed without real images.
**Answer**: True — 같은 synthetic caption에서 생성된 이미지들이 positive set. "Positive pairs do not need to originate from the same image", "without real images". (p.50, p.54)

### Q9 [T/F]
The core of DINOv2 is a single, entirely new self-supervised loss function.
**Answer**: False — 새 loss가 아니라 기존 SSL 요소를 결합하고 학습을 안정화한 **integrated system**. (p.57)

### Q10 [T/F]
In DINOv3, longer training keeps improving global features (image classification), but dense features gradually degrade.
**Answer**: True — patch similarity가 noisy해지고 segmentation 하락. (p.68)

### Q11 [MCQ]
Why does a large batch size help contrastive methods based on InfoNCE such as SimCLR?
(A) It removes the need for data augmentation.
(B) It provides a large number of diverse in-batch negative samples.
(C) It allows the target network to be updated by EMA.
(D) It guarantees that no false negatives are sampled.
**Answer**: (B) — large batch = 많고 다양한 in-batch negatives. 오히려 random negative가 많아도 false negative는 생길 수 있음. (p.5, p.24)

### Q12 [MCQ]
Which statement correctly describes RandAugment?
(A) It uses an RNN controller trained with reinforcement learning to maximize validation accuracy.
(B) It evolves augmentation schedules using population-based training.
(C) It randomly selects augmentations and uses only the number of operations (N) and one global magnitude (M).
(D) It enforces KL-divergence consistency between original and augmented unlabeled data.
**Answer**: (C) — (A) AutoAugment, (B) PBA, (D) UDA. (p.16)

### Q13 [MCQ]
Which method generates harder negatives by mixing embeddings in the feature space?
(A) Mixup
(B) CutMix
(C) MoCHi
(D) UDA
**Answer**: (C) — Mixup은 global pixel-wise, CutMix는 region-level pixel 혼합, UDA는 consistency regularization. (p.19–21)

### Q14 [MCQ]
Which of the following is NOT used in BYOL?
(A) A predictor on the online network
(B) A target network updated by an exponential moving average
(C) Negative samples from the mini-batch
(D) A mean squared error loss between normalized outputs
**Answer**: (C) — "BYOL does not use negative pairs." predictor·EMA target·L2-normalized MSE는 모두 사용. (p.30, p.32, p.34)

### Q15 [MCQ]
According to the lecture, what is the key difference between BYOL and DINO?
(A) BYOL uses an EMA teacher, while DINO does not.
(B) BYOL matches feature vectors with an MSE loss, while DINO matches probability distributions with a cross-entropy loss.
(C) BYOL requires labels, while DINO does not.
(D) BYOL uses a cross-correlation loss, while DINO uses InfoNCE with negatives.
**Answer**: (B) — feature matching (MSE, predictor) vs distribution matching (CE, softmax). 둘 다 EMA teacher, no labels. (p.41)

### Q16 [MCQ]
In SynCLR, what defines the granularity of a visual class (i.e., which images are treated as positives)?
(A) Each individual image is its own class, as in SimCLR.
(B) A human-annotated category label such as "golden retriever".
(C) The caption used to generate the images.
(D) The k-means cluster of the image's embedding.
**Answer**: (C) — "A caption defines the granularity of a visual class"; 같은 caption에서 생성된 이미지 = 한 class. (A)는 SimCLR, (B)는 supervised. (p.48)

### Q17 [MCQ]
Which of the following is NOT a component of "DINOv2 = DINO + ... " as presented in the lecture?
(A) iBOT (masked patch-level objective)
(B) KoLeo regularizer
(C) Curated data (LVD-142M)
(D) Gram anchoring
**Answer**: (D) — Gram anchoring은 DINOv3. DINOv2 = DINO + iBOT + KoLeo + Curated Data + Scaling. (p.57, p.68–69)

### Q18 [MCQ]
What does Gram anchoring in DINOv3 preserve during longer training?
(A) The CLS-token output distribution of a larger frozen ViT-g teacher
(B) The pairwise similarity structure between patch features of an earlier model with clean patch features
(C) An identity cross-correlation matrix between two augmented views
(D) An even assignment of teacher outputs across prototypes
**Answer**: (B) — earlier model의 patch feature 간 pairwise similarity 보존. (A) DINOv2 distillation, (C) Barlow Twins, (D) Sinkhorn-Knopp. (p.69)

### Q19 [T/F]
Unsupervised Data Augmentation (UDA) learns from unlabeled data by maximizing the KL divergence between the predictions on an original image and on its augmented version.
**Answer**: False — UDA는 KL divergence를 **minimize**해서 원본·augmented 입력의 예측을 **일치**시킴 ("Minimizes KL divergence between predictions"). Key Idea: *Learn from unlabeled data through **consistency regularization**.* (p.18)

### Q20 [T/F]
Mixup performs a region-level mixture by pasting a local region of one image onto another image, whereas CutMix creates a global-level, weighted pixel-wise combination of two images.
**Answer**: False — 둘이 뒤바뀜. **Mixup = global-level** weighted pixel-wise combination ($\tilde x = \lambda x_i + (1-\lambda)x_j$), **CutMix = region-level** — local region of one image + rest of the other image ($I_{\text{cutmix}} = M_b \odot I_1 + (1-M_b)\odot I_2$). (p.19–20)

### Q21 [T/F]
Image augmentations for contrastive learning should significantly change the visual appearance of an image while preserving its semantic meaning, so that the resulting views can form positive pairs.
**Answer**: True — Key Points 그대로: create different views / **significantly change visual appearance** / **preserve the semantic meaning** / form positive pairs. 겉모습은 크게, 의미는 그대로. (p.13)

## D. Fill-in-the-blank pool (13 questions)

### F1 [Blank]
A large batch size is essential for contrastive methods relying on (1) ______ (e.g., SimCLR, CLIP) because it provides many diverse in-batch negatives, whereas hard negatives are negatives that lie (2) ______ to the anchor in the embedding space and therefore give more informative supervision than easy negatives.
**Answer**: (1) InfoNCE (accept: InfoNCE loss) (2) close (accept: near, similar) — large batch는 negative의 **양·다양성**, hard negative는 negative의 **질**을 책임짐. hard negative는 anchor랑 가까워서 구분이 어려움 → 더 informative. SSL에선 label이 없어서 찾기 어렵고, 큰 batch·memory bank가 찾을 확률을 높이지만 memory cost와 trade-off. (p.5–6)

### F2 [Blank]
Debiased contrastive loss addresses sampling bias, i.e., random negative sampling may include (1) ______, by estimating the true negative distribution: True Negative = (Random Sampling − (2) ______ Contribution) / $\eta^-$.
**Answer**: (1) false negatives (2) Positive — anchor와 같은 class인 샘플이 negative로 뽑히는 게 false negative → big performance drop. 식으로는 $p_x^-(x') = \big(p(x') - \eta^+ p_x^+(x')\big)/\eta^-$, 여기서 $\eta^- = 1-\eta^+$ = probability of observing a different class. 이 추정치를 InfoNCE 분모에 넣음. (p.7–10)

### F3 [Blank]
AutoAugment formulates augmentation search as a (1) ______ problem that maximizes validation accuracy, whereas RandAugment eliminates the expensive search and controls all transformations with the number of operations N and one global (2) ______ M.
**Answer**: (1) reinforcement learning (accept: RL) (2) magnitude — AutoAugment: controller(RNN)가 strategy를 sample → child network의 validation accuracy를 reward로 controller 갱신 (which operation / probability / magnitude → huge search space). RandAugment: random selection + N, M 두 개 → very small search space. (p.15–16)

### F4 [Blank]
UDA applies augmentations to unlabeled data and minimizes the KL divergence between predictions on the original and augmented inputs; its key idea is to learn from unlabeled data through ______ regularization.
**Answer**: consistency — 원본·augmented의 예측이 같아야 한다는 제약 = consistency regularization. labeled data엔 supervised cross-entropy, unlabeled data엔 consistency loss (RandAugment, back translation 등 augmentation). (p.18)

### F5 [Blank]
Mixup performs a (1) ______-level mixture by creating a weighted pixel-wise combination of two images, whereas MoCHi generates harder negatives by mixing embeddings in the (2) ______ space.
**Answer**: (1) global (2) feature (accept: embedding, latent) — Mixup = 이미지 통째로 섞기(global), CutMix = 패치 붙이기(region-level), 둘 다 **pixel** 공간. MoCHi는 hardest negative를 찾아 그 embedding끼리 interpolate하고 query와도 섞어서 **feature space**에서 synthetic hard negative를 만듦. (p.19–21)

### F6 [Blank]
In SimCLR, the contrastive loss operates on an extra ______ head $g(\cdot)$ applied to the representation $h$; after training, $g$ is thrown away and $h$ from the base encoder $f(\cdot)$ is used for downstream tasks.
**Answer**: projection (accept: projection layer) — $z = g(h)$ 공간은 data transformation에 invariant하게 학습돼서 downstream에 쓸모 있는 정보를 버릴 수 있음 → $g$를 두면 $h$에 더 많은 정보가 보존됨. "return encoder network $f(\cdot)$, and throw away $g(\cdot)$". (p.24–25, p.28)

### F7 [Blank]
In BYOL, only the online network has a (1) ______ $q_\theta$ that predicts the target network's projection of another augmented view, and the target network is updated as an (2) ______ of the online parameters with no backpropagation.
**Answer**: (1) predictor (2) exponential moving average (accept: EMA, moving average) — $\xi \leftarrow \tau\xi + (1-\tau)\theta$. loss = L2-normalized prediction $\bar q_\theta(z)$와 $\bar z'$ 사이 MSE, **negative pair 없음**. target 쪽은 sg(stop-gradient)라 $\theta$만 최적화됨. (p.30–32)

### F8 [Blank]
Barlow Twins computes the (1) ______ matrix between the two views' embeddings along the batch dimension and pushes it toward the identity matrix; the off-diagonal part of this objective is called the (2) ______ term.
**Answer**: (1) cross-correlation (2) redundancy reduction — $C$는 $D\times D$ (D = feature dimension). diagonal $C_{ii}\to1$ = **invariance** (같은 feature가 augmentation 후에도 유지), off-diagonal $C_{ij}\to0$ = **redundancy reduction** (feature끼리 decorrelated). shared encoder, no negatives, batch size에 robust. (p.35–38)

### F9 [Blank]
In DINO, the (1) ______ network is trained by backpropagation to match the output (softmax) distribution of a teacher that is an EMA of it; unlike BYOL, which matches feature vectors with an MSE loss, DINO matches probability distributions with a (2) ______ loss.
**Answer**: (1) student (2) cross-entropy — 방향 주의: **student가 teacher를 따라가고**, teacher = student의 EMA (momentum teacher). BYOL = feature matching / MSE / predictor, DINO = distribution matching / cross-entropy / softmax probability. 공통점: EMA teacher, no labels. (p.39, p.41)

### F10 [Blank]
In DINO's teacher branch, the teacher output goes through ______ before the softmax: a running mean $c \leftarrow mc + (1-m)\frac{1}{B}\sum_{i=1}^{B} g_{\theta_t}(x_i)$ is subtracted from it, and a stop-gradient (sg) blocks gradients into the teacher.
**Answer**: centering (accept: centering and sharpening) — 그림에서 **teacher 쪽에만** centering 블록이 있음 (student는 바로 softmax). teacher 출력은 center를 빼고 teacher temperature로 나눠 softmax → "center + sharpen". loss는 $-p_2\log p_1$. (p.40)
> 보충: centering은 한 dimension이 독점하는 걸 막고, sharpening은 uniform 분포로 퍼지는 걸 막아서 둘이 함께 collapse를 방지함 (DINO 논문 설명, 슬라이드엔 명시 X).

### F11 [Blank]
SimCLR preserves image identity, whereas SynCLR preserves ______-level semantics: images generated by a text-to-image model from the same synthetic text description form a positive set for multi-positive contrastive learning.
**Answer**: caption (accept: text) — "A caption defines the granularity of a visual class." SimCLR: 이미지 하나 = class 하나, Supervised: 모든 golden retriever = 한 class (action·background 무시), SynCLR: 같은 caption에서 생성한 이미지들 = 한 class (objects, actions, relationships 포착). positive가 같은 이미지에서 나올 필요 없음, real image 없이도 학습 가능. (p.47–48, p.54)

### F12 [Blank]
DINOv2 = DINO + (1) ______ + (2) ______ + Curated Data + Scaling, where (1) adds a masked patch-level self-distillation objective and (2) is a regularizer that spreads image-level features across the entire embedding space.
**Answer**: (1) iBOT (2) KoLeo — 새 loss 하나가 아니라 기존 SSL 요소를 합치고 학습을 안정화한 **integrated system**. iBOT: student patch 일부를 mask하고 teacher의 visible patch를 target으로 → local/spatial representation. KoLeo: $\mathcal L_{\text{koleo}} = -\frac1n\sum_i \log d_{n,i}$, 서로 다른 이미지 feature를 밀어내서 crowding 방지. (p.57, p.61–62)

### F13 [Blank]
In DINOv3, longer training improves global features but dense features gradually degrade; ______ anchoring fixes this by preserving the pairwise patch-similarity structure of an earlier model with clean patch features.
**Answer**: Gram — $\mathcal L_{\text{Gram}} = \|X_S X_S^\top - X_G X_G^\top\|_F^2$. feature 자체가 아니라 patch 간 **similarity 구조(Gram matrix)**를 맞춤 → patch similarity가 깨끗해지고 object boundary가 선명해짐 (spatially consistent). (p.68–70)

<!-- REVIEWED | fixes: 3 | reviewer: opus -->
<!-- AUGMENTED 2026-10-22 | C fixes: 1 | C added: 3 | D blanks: 13 | author: opus
Q17: page ref (p.57, p.69) -> (p.57, p.68–69) — the "DINOv3" label for Gram anchoring is on p.68; p.69 is titled only "Gram Anchoring".
-->


<!-- VERIFIED 2026-10-22 | checked: 34 | fixed: 1 | deleted: 0 | reviewer: opus
F11: stem gave away the blank ("SynCLR preserves ______-level semantics ... from the same synthetic caption"); changed "synthetic caption" -> "synthetic text description", added accept: text.
All other C/D items: key, p.N, and distractors confirmed against slides (image pages p.9, p.10, p.18, p.19, p.20, p.24, p.28, p.32, p.36, p.40, p.69 checked). Sections A/B: no slide contradictions found.
-->
