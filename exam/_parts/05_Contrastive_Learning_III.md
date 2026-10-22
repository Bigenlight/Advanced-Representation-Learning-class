# 05. Contrastive Learning III — Core Ideas & Question Pool

## A. Core ideas (one block per method/paper/concept, in lecture order)

### Memory Bank (concept, p.4)
- **Problem**: Large numbers of negatives improve contrastive learning, but computing embeddings for all negatives every step is too expensive, and mini-batch negatives are limited by batch size.
- **Core idea**: Store previously computed embeddings in memory and reuse them as negatives — "Trade stale representations for efficient large-scale contrastive learning."
- **vs. previous / key difference**: Mini-batch negatives are *limited but fresh*; memory bank negatives are *large but slightly stale*.
- **Keywords**: store & reuse, stale, fresh, large number of negatives, decouple from batch size
- 한 줄: 과거 embedding을 저장해 negative로 재사용 → 수는 많아지지만 stale해짐.

### Instance Discrimination with Memory Bank (Wu et al., CVPR 2018, p.5–9)
- **Problem**: Learn features without labels when treating every image as its own class makes an N-way softmax infeasible (too many heads).
- **Core idea**: Each image is a unique class (positive = same image, negatives = all other images); a non-parametric softmax uses the stored memory-bank features as class weights, and NCE with a Monte Carlo subset of M negatives approximates the full softmax.
- **vs. previous / key difference**: Replaces the learned classifier weights $\mathbf w_i$ with the stored features $\mathbf v_i$ (non-parametric), and adds a proximal regularization term that gradually vanishes as embeddings converge.
- **Keywords**: instance discrimination, non-parametric softmax, NCE, Monte Carlo approximation, proximal regularization, stale embeddings
- 한 줄: 이미지 하나 = class 하나, bank feature를 classifier weight로 쓰고 NCE로 근사. 단점은 stale feature.

### MoCo — Momentum Contrast (He et al., CVPR 2020, p.10–15)
- **Problem**: "Can we keep a large dictionary while reducing feature staleness?" (p.9)
- **Core idea**: Replace the memory bank with a FIFO queue of keys encoded by a momentum encoder, $\theta_k \leftarrow m\theta_k + (1-m)\theta_q$, giving a large and consistent dictionary trained with InfoNCE.
- **vs. previous / key difference**: Unlike SimCLR, gradients flow only through the query encoder (no gradient through the key), and the number of keys is decoupled from the mini-batch size; MoCo is a hybrid of end-to-end and memory bank.
- **Keywords**: FIFO queue, momentum encoder, dictionary look-up, consistent keys, no gradient through key, InfoNCE
- 한 줄: queue + momentum(EMA) key encoder → 크고(large) 일관된(consistent) dictionary.

### MoCo v2 (Chen et al., 2020, p.16–19)
- **Problem**: MoCo lagged behind SimCLR, which used better training tricks but needed huge batches.
- **Core idea**: Combine MoCo with SimCLR's effective training strategies — an MLP projection head and stronger data augmentation — while retaining the momentum encoder and queue.
- **vs. previous / key difference**: Not a new architecture; because negatives are decoupled from batch size, MoCo v2 outperforms SimCLR with a much smaller batch (256 vs. 8192).
- **Keywords**: MLP projection head, stronger augmentation, improved baseline, batch 256 vs 8192, less memory
- 한 줄: MoCo + SimCLR 레시피(MLP head, strong aug). 작은 batch로 SimCLR 이김.

### CURL (Srinivas et al., 2020, p.20–21)
- **Problem**: Reinforcement learning from raw pixel observations is sample-inefficient.
- **Core idea**: Apply MoCo-style contrastive learning (random-crop positive pairs, query encoder + momentum key encoder) to observations from the replay buffer, while simultaneously optimizing the RL policy/value network on the learned representation.
- **vs. previous / key difference**: Contrastive learning is used as a jointly trained auxiliary objective for RL rather than for image classification pre-training.
- **Keywords**: reinforcement learning, replay buffer, random crop, momentum encoder, sample efficiency, simultaneous optimization
- 한 줄: MoCo를 RL에 붙여서 pixel RL의 sample efficiency ↑. RL loss와 동시에 학습.

### DeepCluster (Caron et al., ECCV 2018, p.23–25)
- **Problem**: Instance-level contrastive learning needs positive/negative pairs and pushes away semantically similar images.
- **Core idea**: Cluster the current features (k-means), use the cluster assignments as pseudo labels, train the network to predict them, and repeat — "Alternate between clustering and classification."
- **vs. previous / key difference**: It does not create positive/negative pairs; it establishes pseudo-label learning as an alternative to contrastive learning.
- **Keywords**: k-means, pseudo labels, alternate clustering & classification, offline clustering, no pairs
- 한 줄: k-means cluster ID를 pseudo label로 분류 학습, clustering ↔ classification 반복.

### SwAV (Caron et al., NeurIPS 2020, p.26–30)
- **Problem**: Contrastive learning needs explicit pairwise comparisons (many negatives), and DeepCluster needs offline clustering over the whole dataset.
- **Core idea**: Compute online cluster assignments (codes) for augmented views using K trainable prototypes, and predict the code of one view from another view (swapped prediction).
- **vs. previous / key difference**: Feature ↔ Prototype comparison instead of Feature ↔ Feature; no explicit negative pairs and no offline clustering.
- **Keywords**: swapped prediction, codes, prototypes, online clustering, no explicit negatives, multiple views
- 한 줄: 한 view의 code를 다른 view로 예측 (swap). online clustering + prototype.

### CLIP (Radford et al., PMLR 2021, p.32–37)
- **Problem**: Fixed-label supervision limits transfer; can natural language serve as supervision?
- **Core idea**: Jointly train image and text encoders to align matched image–caption pairs in a shared space with a symmetric contrastive loss over an $N\times N$ similarity matrix; classify zero-shot by matching an image to text prompts.
- **vs. previous / key difference**: Positives are (image, caption) pairs instead of augmentations, and inference uses image–text matching instead of a learned classifier.
- **Keywords**: image–text pairs, shared embedding space, symmetric contrastive loss, zero-shot, text prompts, 400M pairs
- 한 줄: image–text를 contrastive로 정렬 → prompt로 zero-shot 분류. caption 예측보다 ~4× data-efficient.

### SupCon — Supervised Contrastive Learning (Khosla et al., NeurIPS 2020, p.38–42)
- **Problem**: Self-supervised contrastive learning treats same-class images as negatives, and cross-entropy gives less discriminative representations.
- **Core idea**: Use class labels — all samples of the same class are positives (pulled together), different classes are negatives (pushed apart); compute SimCLR's InfoNCE for each positive and average.
- **vs. previous / key difference**: Beats cross-entropy only by a small amount, but is more robust (ImageNet-C) and less sensitive to hyperparameters.
- **Keywords**: same-class positives, InfoNCE per positive and average, vs cross-entropy, ImageNet-C robustness, hyperparameter insensitivity
- 한 줄: 같은 class 전부 positive. CE보다 조금 낫고, robust하고 hyperparameter에 덜 민감.

### NNCLR → All4One (Estepa et al., ICCV 2023, p.43–53)
- **Problem**: "Can the memory bank provide more than negatives?" — and a single nearest neighbor can be a noisy positive.
- **Core idea**: NNCLR uses the nearest neighbor from a support set as the positive; All4One retrieves K neighbors, summarizes them with a small Transformer (self-attention) into a centroid, and combines three losses: Neighbor Contrast, Centroid Contrast, and Feature Contrast (Barlow Twins redundancy reduction).
- **vs. previous / key difference**: "NNCLR uses one neighbor. All4One summarizes multiple neighbors into one representation." The centroid is produced by self-attention, not an average or cluster center.
- **Keywords**: support set, nearest neighbor positive, centroid via self-attention, redundancy reduction, Barlow Twins, three losses
- 한 줄: memory bank에서 positive를 꺼냄. K-NN을 self-attention으로 centroid화 + Barlow Twins loss.

### PAPN (Tan et al., ICCV 2025, p.54–64)
- **Problem**: "Can cluster-based SSL preserve fine-grained visual details?" — multi-stage alignment helps on generic data but hurts on fine-grained data because strong augmentation changes shallow features much more.
- **Core idea**: (1) Stage-wise Progressive Augmentation (SPA): mild augmentation for shallow stages, strong for deep stages; (2) Prototype-based Holistic-Partial Learning (PHP): learnable prototypes over spatial descriptors extract part features, which are concatenated with whole features into InfoNCE.
- **vs. previous / key difference**: One augmentation strength does not fit every stage; and instead of choosing between global and local features, contrast both together.
- **Keywords**: fine-grained, stage-wise progressive augmentation, shallow vs deep stages, learnable prototypes, part representation, whole + part
- 한 줄: stage마다 aug 강도 다르게(SPA) + prototype으로 part feature 뽑아 whole과 함께 contrast(PHP).

### SCL-RHE (Long et al., ECCV 2024, p.65–75)
- **Problem**: In SupCon, imprecise (human) labels create wrong pair relationships, and human errors look like easy, high-similarity positives.
- **Core idea**: Since almost all incorrect signals are false positives, deprioritize easy positives — high similarity → lower sampling weight, $q^+(x^+) \propto e^{-\beta f(x)^\top f(x^+)} p_x^+(x^+)$ — emphasizing hard positives.
- **vs. previous / key difference**: Methods for heavy synthetic noise (Sel-CL, TCL) remove too many samples and can underperform SupCon; human errors overlap with true positives, unlike synthetic errors.
- **Keywords**: human labeling errors, false positives dominate, assigned vs latent label, easy vs hard positives, lower weight for high similarity
- 한 줄: human label error는 easy false positive가 됨 → similarity 높은 positive 가중 ↓, hard positive 강조.

## B. Must-know definitions & distinctions
- **Instance discrimination**: each image is considered a unique class; positive = same image, negatives = all other images (p.5).
- **Staleness**: memory-bank embeddings were computed by an older encoder, so current queries are compared with outdated features (p.9).
- **Mini-batch vs memory bank negatives**: limited but fresh vs large but slightly stale (p.4).
- **End-to-end vs Memory bank vs MoCo** (p.14): key from current encoder / memory bank / momentum encoder; negatives from mini-batch / memory bank / FIFO queue; dictionary small / large / large; freshness fresh / stale / **consistent**.
- **Momentum update**: $\theta_k \leftarrow m\theta_k + (1-m)\theta_q$ — the key encoder is updated by momentum, not by backpropagation (p.11–12).
- **MoCo vs SimCLR**: running queue of keys; gradients only through queries; batch size decoupled from number of keys (p.11).
- **DeepCluster vs SwAV**: offline clustering + pseudo labels vs online cluster assignments + swapped prediction (p.24, p.26).
- **Contrastive vs SwAV** (p.30): Feature↔Feature vs Feature↔Prototype; pos/neg pairs vs prototype assignments; similarity matching vs swapped assignment prediction; pairwise comparison vs online clustering.
- **CLIP training vs inference**: training maximizes similarity of matched pairs (symmetric loss); inference picks the text prompt with the highest image–text similarity (p.33).
- **Contrastive vs predictive (CLIP)**: contrastive ≈ 4× more data-efficient than caption prediction; BoW text encoder beats Transformer LM (p.36).
- **SupCon vs self-supervised contrastive**: same-class samples are positives instead of only the augmented view of the same image (p.38–39).
- **SupCon vs CE**: small accuracy gain, better robustness (ImageNet-C), less sensitive to hyperparameters (p.42).
- **NNCLR vs All4One**: one neighbor as positive vs K neighbors summarized into one representation (p.45).
- **Neighbor vs Feature contrast**: neighbor losses compare samples; feature contrast compares feature dimensions (p.49).
- **Shallow vs deep stages (PAPN)**: shallow stages are more affected by strong augmentation; deep stages are relatively augmentation-tolerant (p.58).
- **Assigned label (AL) vs latent label (LL)**; **False positive** = same AL, different LL (wrongly pulled); **False negative** = different AL, same LL (wrongly pushed) (p.67, p.69).
- **Human vs synthetic label errors**: humans confuse visually similar classes (errors stay close to the assigned class, overlap with true positives) vs random label changes (p.68).
- **Easy vs hard positive**: high similarity → low weight vs low similarity → high weight in SCL-RHE (p.72).

## C. Question pool (18 questions)

### Q1 [T/F]
In instance discrimination (Wu et al.), each image is considered a unique class: the positive pair is the same image and the negative pairs are all other images.
**Answer**: True — 슬라이드 정의 그대로 (p.5)

### Q2 [T/F]
Compared with mini-batch negatives, memory bank negatives are limited in number but always fresh.
**Answer**: False — 반대. Mini-batch = limited but fresh, memory bank = large but slightly stale (p.4)

### Q3 [MCQ]
How is the key encoder updated in MoCo?
(A) By backpropagating the contrastive loss through the keys
(B) $\theta_q \leftarrow m\theta_q + (1-m)\theta_k$
(C) $\theta_k \leftarrow m\theta_k + (1-m)\theta_q$
(D) By copying the query encoder weights at every iteration
**Answer**: (C) — key encoder는 query encoder의 momentum(EMA)으로만 갱신, key로는 gradient 안 흐름. (B)는 query/key 뒤바꾼 함정 (p.11–12)

### Q4 [MCQ]
According to the comparison of End-to-End, Memory Bank, and MoCo, which row correctly describes MoCo?
(A) Key: current encoder / Negative source: mini-batch / Dictionary: small / Freshness: fresh
(B) Key: memory bank / Negative source: memory bank / Dictionary: large / Freshness: stale
(C) Key: momentum encoder / Negative source: FIFO queue / Dictionary: large / Freshness: consistent
(D) Key: momentum encoder / Negative source: mini-batch / Dictionary: small / Freshness: consistent
**Answer**: (C) — (A)는 End-to-End, (B)는 Memory Bank. MoCo = 두 방식의 hybrid (p.14)

### Q5 [T/F]
MoCo v2 adopts SimCLR's MLP projection head and stronger data augmentation while retaining MoCo's momentum encoder and queue, which lets it outperform SimCLR with a smaller batch size.
**Answer**: True — negative 수와 batch size를 decouple해서 256 batch로 SimCLR 8192를 이김 (p.16, p.18)

### Q6 [MCQ]
Which of the following is NOT true about CURL?
(A) It applies contrastive learning to reinforcement learning.
(B) It uses random crop augmentation to create positive pairs.
(C) It employs a momentum encoder and contrastive loss, following MoCo.
(D) It first pre-trains the encoder with contrastive learning, then freezes it before training the RL policy.
**Answer**: (D) — CURL은 contrastive learning과 RL policy/value network를 **simultaneously** 최적화함 (p.20–21)

### Q7 [T/F]
DeepCluster uses k-means clusters to build positive/negative pairs and trains the network with a contrastive loss on those pairs.
**Answer**: False — DeepCluster는 pos/neg pair를 만들지 않고 cluster 결과를 pseudo label로 써서 classification 학습 (p.23–24)

### Q8 [MCQ]
What is the key idea of SwAV?
(A) Use k-means cluster IDs from the whole dataset as pseudo labels for classification
(B) Predict the cluster assignment (code) of one augmented view from another view
(C) Keep a FIFO queue of keys encoded by a momentum encoder
(D) Align image and text embeddings in a shared space
**Answer**: (B) — "Learn by swapping cluster assignments between multiple views." (A) DeepCluster, (C) MoCo, (D) CLIP (p.26)

### Q9 [T/F]
SwAV requires offline clustering of the entire dataset and explicit pairwise comparisons between features.
**Answer**: False — SwAV는 online으로 assignment 계산, explicit pairwise comparison도 offline clustering도 필요 없음 (p.26, p.30)

### Q10 [T/F]
At inference, CLIP performs zero-shot classification by encoding candidate text prompts and predicting the prompt with the highest image–text similarity, instead of using a learned classifier.
**Answer**: True — CLIP Inference의 Key Idea (p.33)

### Q11 [MCQ]
Which is one of the key findings about CLIP's training objective?
(A) Caption prediction with a Transformer language model is about 4× more data-efficient than contrastive learning.
(B) Contrastive learning is approximately 4× more data-efficient than caption prediction.
(C) A Transformer language model text encoder outperforms a simple Bag-of-Words encoder.
(D) Complex text modeling is more important than rich language supervision.
**Answer**: (B) — BoW 단순 encoder가 Transformer LM을 이기고, rich supervision > complex text modeling (p.36)

### Q12 [MCQ]
How does SupCon define positives and compute its loss?
(A) The nearest neighbor from a support set is the single positive.
(B) The matched caption of each image is the positive, with a symmetric loss.
(C) All samples of the same class are positives; SimCLR's InfoNCE is computed for each positive and then averaged.
(D) Samples in the same k-means cluster are positives, trained with cross-entropy on pseudo labels.
**Answer**: (C) — (A) NNCLR, (B) CLIP, (D) DeepCluster (p.38, p.40)

### Q13 [T/F]
Compared with cross-entropy, SupCon is more sensitive to hyperparameter changes.
**Answer**: False — SupCon은 hyperparameter 변화에 **less** sensitive. CE보다 정확도는 small amount만 높고 ImageNet-C robustness는 더 좋음 (p.42)

### Q14 [MCQ]
In All4One's Centroid Contrast, how is the centroid of the K retrieved neighbors obtained?
(A) The arithmetic average of the K neighbor embeddings
(B) The k-means cluster center that the query belongs to
(C) The first output representation of a small Transformer encoder that integrates the neighbors via self-attention
(D) The single nearest neighbor in the support set
**Answer**: (C) — "The centroid is produced by self-attention. It is not an average or a cluster center." (D)는 NNCLR (p.45, p.48)

### Q15 [T/F]
In All4One, the Feature Contrast (Barlow Twins style) compares feature dimensions rather than samples: diagonal terms keep the same dimension consistent across views and off-diagonal terms reduce redundancy.
**Answer**: True — "Neighbor losses compare samples. Feature contrast compares feature dimensions." (p.49)

### Q16 [T/F]
PAPN observes that strong augmentation affects shallow stages more severely than deep stages, so its Stage-wise Progressive Augmentation uses mild augmentation for shallow stages and strong augmentation for deep stages.
**Answer**: True — "One augmentation strength does not fit every stage." (p.58–59)

### Q17 [T/F]
In PAPN's Prototype-based Holistic-Partial Learning, the part representation replaces the whole (global) feature, so only part-level features enter the InfoNCE loss.
**Answer**: False — whole + part를 concat해서 InfoNCE에 넣음: "Do not choose between global and local features—contrast both together." (p.61)

### Q18 [MCQ]
Which of the following is NOT true about human labeling errors in supervised contrastive learning and SCL-RHE?
(A) Most incorrect contrastive signals come from false positives rather than false negatives.
(B) Human labeling errors overlap strongly with true positives, unlike synthetic label errors.
(C) SCL-RHE gives the highest sampling weight to positives most similar to the anchor, since they are the most reliable.
(D) Methods designed for heavy synthetic noise (Sel-CL, TCL) can underperform standard SupCon.
**Answer**: (C) — 반대. SCL-RHE는 high similarity → **lower** weight (easy positive에 human error가 섞여 있음), hard positive 강조. (A) p.69/p.73, (B) p.68, (D) p.74 (p.68–74)

<!-- REVIEWED | fixes: 1 | reviewer: opus -->
