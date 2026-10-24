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

## C. Question pool (19 questions)

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

### Q19 [T/F]
NNCLR retrieves the K nearest neighbors of a sample from a support set and summarizes them with a small Transformer into a single positive representation.
**Answer**: False — 그건 **All4One**의 Centroid Contrast. **NNCLR**은 support set에서 **nearest neighbor 하나**를 꺼내 positive로 씀 (두 augmented view를 직접 비교하지 않고, 첫 view의 nearest neighbor가 한쪽 view를 대체). "NNCLR uses one neighbor. All4One summarizes multiple neighbors into one representation." (p.45, p.47–48)

## D. Fill-in-the-blank pool (13 questions)

### F1 [Blank]
Mini-batch negatives are limited but (1) ______, whereas memory bank negatives are large in number but slightly (2) ______.
**Answer**: (1) fresh (2) stale (accept: outdated) — memory bank = 과거 embedding을 저장해 negative로 재사용 → 계산은 싸고 negative 수는 많지만, 옛 encoder로 계산된 feature라 현재 query와 안 맞음. Key Idea: "Trade stale representations for efficient large-scale contrastive learning." (p.4, p.9)

### F2 [Blank]
In instance discrimination (Wu et al.), each image is treated as a unique (1) ______; because training a softmax with that many heads is infeasible, the full softmax is approximated by (2) ______, which samples only some negatives.
**Answer**: (1) class (2) Noise Contrastive Estimation (accept: NCE) — positive = 같은 이미지, negatives = 나머지 전부. classifier weight $\mathbf w_i$ 대신 memory bank에 저장된 feature $\mathbf v_i$를 쓰는 **non-parametric** softmax $P(i|\mathbf v) = \frac{\exp(\mathbf v_i^\top \mathbf v/\tau)}{\sum_j \exp(\mathbf v_j^\top \mathbf v/\tau)}$, 분모는 random subset M으로 Monte Carlo 근사. (p.5–8)

### F3 [Blank]
MoCo's key idea is to replace the memory bank with a (1) ______-updated (2) ______ of keys, in which the newest keys are added and the oldest are removed (first-in, first-out).
**Answer**: (1) momentum (2) queue (accept: FIFO queue) — "Key Idea: Replace the memory bank with a momentum-updated queue." queue 덕분에 dictionary가 크고(large), momentum encoder 덕분에 key가 일관됨(consistent). (p.10)

### F4 [Blank]
Compared with a memory bank, MoCo's keys are more (1) ______ because they come from a slowly progressing momentum encoder $\theta_k \leftarrow m\theta_k + (1-m)\theta_q$, and gradients are computed and the encoder updated only through the (2) ______.
**Answer**: (1) consistent (2) queries (accept: query, query encoder) — 비교표 Feature Freshness: End-to-End = Fresh, Memory Bank = Stale, MoCo = **Consistent**. key로는 gradient 안 흐름 → key encoder는 backprop이 아니라 momentum update. MoCo = end-to-end와 memory bank의 hybrid. (p.11–12, p.14)

### F5 [Blank]
MoCo v2 adopts the (1) ______ projection head from SimCLR and uses (2) ______ data augmentation, while retaining MoCo's momentum encoder and queue.
**Answer**: (1) MLP (accept: non-linear) (2) stronger (accept: strong) — Key Idea: "Combine MoCo with SimCLR's effective training strategies." 새 architecture 아님. negative 수가 batch size와 decouple돼서 SimCLR보다 훨씬 작은 batch (256 vs. 8192)·적은 memory로 SimCLR를 이김. (p.16–19)

### F6 [Blank]
CURL applies MoCo-style contrastive learning (random-crop positive pairs, a query encoder and a momentum key encoder) to ______ from raw pixel observations, in order to improve sample efficiency.
**Answer**: reinforcement learning (accept: RL) — replay buffer에서 observation을 뽑아 두 view로 contrastive 학습하면서, **동시에(simultaneously)** RL policy/value network를 그 representation 위에서 최적화. pre-train 후 freeze하는 게 아님. (p.20–21)

### F7 [Blank]
DeepCluster clusters the current image features with (1) ______ and uses the cluster IDs as (2) ______ to train the network, alternating between clustering and classification.
**Answer**: (1) k-means (2) pseudo labels (accept: pseudo-labels) — contrastive learning과 달리 **positive/negative pair를 만들지 않음**. Key Idea: "Alternate between clustering and classification." pseudo-label learning을 contrastive의 대안으로 정립, SwAV·DINO에 영향. (p.23–25)

### F8 [Blank]
SwAV learns by (1) ______ cluster assignments between multiple views — it predicts the code of one view from another view — so it compares Feature ↔ (2) ______ instead of Feature ↔ Feature.
**Answer**: (1) swapping (accept: swapped) (2) Prototype (accept: prototypes, prototype vectors) — K개의 trainable prototype vector $C$로 **online** cluster assignment(code $q$)를 계산, $\mathcal L = \ell(z_t, q_s) + \ell(z_s, q_t)$. explicit negative pair도, DeepCluster식 offline clustering도 필요 없음. (p.26–27, p.30)

### F9 [Blank]
CLIP trains image and text encoders with a (1) ______ contrastive loss over an $N\times N$ image–text similarity matrix (image→text and text→image), and at inference performs (2) ______ classification by predicting the text prompt with the highest image–text similarity.
**Answer**: (1) symmetric (2) zero-shot — Training: matched pair 유사도 최대화, mismatched 최소화. Inference: "Classification is performed by image–text matching instead of a trained classifier." positive가 augmentation이 아니라 (image, caption) pair, 400M pairs, natural language가 supervisory signal. (p.32–33, p.37)

### F10 [Blank]
SupCon uses SimCLR's InfoNCE as is, but instead of a single positive it treats all samples of the same ______ as positives, computes InfoNCE for each positive, and takes the average.
**Answer**: class — self-supervised contrastive에선 같은 class 다른 이미지도 negative로 밀려나는데, SupCon은 label을 써서 같은 class는 당기고 다른 class는 밂. CE보다 정확도는 small amount만 높지만 ImageNet-C에 robust하고 hyperparameter에 덜 민감. (p.38–40, p.42)

### F11 [Blank]
NNCLR uses the single (1) ______ retrieved from a support set as the positive sample, whereas All4One retrieves K neighbors and summarizes them into a centroid via (2) ______ in a small Transformer encoder.
**Answer**: (1) nearest neighbor (2) self-attention (accept: a small Transformer, attention) — "The centroid is produced by self-attention. It is not an average or a cluster center." All4One = Neighbor Contrast + Centroid Contrast + Feature Contrast(Barlow Twins식 redundancy reduction). memory bank가 negative뿐 아니라 positive도 제공. (p.45–49)

### F12 [Blank]
PAPN's Stage-wise Progressive Augmentation applies (1) ______ augmentation to shallow stages and strong augmentation to deep stages, and its second component, holistic-partial learning, uses learnable (2) ______ over spatial descriptors to extract part-level features that are concatenated with the whole feature in InfoNCE.
**Answer**: (1) mild (accept: weak) (2) prototypes — strong augmentation은 **shallow stage**를 더 심하게 흔듦 (같은 새 두 장을 다른 개체로 인식), deep stage는 상대적으로 augmentation-tolerant → "One augmentation strength does not fit every stage." PHP: whole feature(전체 모양) + part feature(국소 판별 패턴) 같이 contrast. (p.58–61)

### F13 [Blank]
With human labeling errors in supervised contrastive learning, almost all incorrect signals come from false (1) ______, so SCL-RHE assigns a (2) ______ sampling weight to easy positives that are already highly similar to the anchor.
**Answer**: (1) positives (2) lower (accept: low, smaller) — false positive = assigned label 같음 + latent label 다름 → 잘못 당김. human error는 시각적으로 비슷한 class끼리 헷갈린 거라 anchor와 이미 가까운 **easy positive**로 숨어 있음. Core Idea: "High similarity → Lower sampling weight", $q^+(x^+) \propto e^{-\beta f(x)^\top f(x^+)}\, p_x^+(x^+)$ → hard positive 강조. (p.69, p.71–73)

<!-- REVIEWED | fixes: 1 | reviewer: opus -->
<!-- AUGMENTED 2026-10-22 | C fixes: 0 | C added: 1 | D blanks: 13 | author: opus
(no C fixes — all 18 answers and page refs verified against slides, incl. figure-only pages p.7–8, p.11, p.27, p.33, p.40, p.46, p.69, p.72)
-->


<!-- VERIFIED 2026-10-22 | checked: 32 | fixed: 2 | deleted: 0 | reviewer: opus
F3: "newest keys are enqueued and the oldest are dequeued" gave away blank (2) "queue"; changed to "added ... removed".
F12: the component name "Prototype-based Holistic-Partial Learning" gave away blank (2) "prototypes"; changed to "its second component, holistic-partial learning".
All other C/D items: key, p.N, and distractors confirmed against slides (image pages p.11, p.27, p.33, p.40, p.61 checked). Sections A/B: no slide contradictions found.
-->
