# 03. Contrastive Learning I — Core Ideas & Question Pool

## A. Core ideas (one block per method/paper/concept, in lecture order)

### From Pretext Tasks to Contrastive Learning (p.4–7)
- **Problem**: Pretext tasks (image completion, rotation, jigsaw, colorization) are all handcrafted objectives that someone has to design.
- **Core idea**: "Representation Learning = Learning Similarity": good representations place semantically similar samples nearby, so similarity itself becomes the supervision (pull similar samples together, push dissimilar samples apart).
- **vs. previous / key difference**: Instead of predicting a designed target (e.g., rotation angle), the goal is the distance/similarity relation between embeddings of a reference $x$, positive $x^+$, and negative $x^-$.
- **Keywords**: handcrafted objectives, similarity as supervision, metric learning, reference / positive / negative
- 한 줄: "task 설계하지 말고 similarity 자체를 supervision으로 쓰자" → metric learning.

### Siamese Network & Contrastive Representation Learning (p.8–9)
- **Problem**: Two inputs must be compared in a meaningful way (e.g., same person or not).
- **Core idea**: Same network + shared weights maps both inputs into the same embedding space; $d(z_1,z_2)$ decreases for the same person and increases otherwise. Goal: $\text{score}(f(x), f(x^+)) \gg \text{score}(f(x), f(x^-))$.
- **vs. previous / key difference**: Applies to both supervised and unsupervised settings; only the definition of positives changes.
- **Keywords**: shared weights, same embedding space, score function, supervised & unsupervised
- 한 줄: 가중치 공유해야 같은 공간에서 거리 비교가 의미 있음.

### Contrastive Loss (Chopra et al., 2005, p.11)
- **Problem**: Learn an encoder where same-class pairs are close and different-class pairs are far.
- **Core idea**: Pair-wise loss: minimize squared distance for positive pairs, push negative pairs apart beyond a margin $\varepsilon$ via $\max(0, \varepsilon - d)^2$.
- **vs. previous / key difference**: Limitations — considers only pairs, many negatives become inactive (already beyond margin → zero gradient), does not explicitly enforce relative ordering.
- **Keywords**: pair, margin ε, hinge, inactive negatives, absolute target
- 한 줄: pair 단위 절대 목표 (positive → 0, negative → ε 이상), 상대 순서는 강제 X.

### Triplet Loss (FaceNet, Schroff et al., 2015, p.12–13)
- **Problem**: Contrastive loss does not enforce relative similarity.
- **Core idea**: Anchor–Positive–Negative triplets; $\max(0, \|f(x)-f(x^+)\|^2 - \|f(x)-f(x^-)\|^2 + \varepsilon)$ enforces relative similarity / ranking relationships; margin = minimum offset between the two distances.
- **vs. previous / key difference**: Relative ordering instead of absolute targets; but Problem 1 Triplet Explosion ($N^3$) and Problem 2 Most Triplets are Useless (easy triplets dominate) → hard negatives are crucial.
- **Keywords**: anchor, relative similarity, ranking, triplet explosion $N^3$, easy triplets
- 한 줄: 상대 격차만 요구. 대신 조합 폭발 + easy triplet 문제.

### Hard Negative Mining (p.14)
- **Problem**: Most negatives are already far from the anchor; easy negatives contribute little to the loss and gradients.
- **Core idea**: Hard negatives are close to the anchor but belong to different classes → stronger learning signals; focus training on the most challenging samples.
- **Keywords**: easy negative, hard negative, informative negatives, stronger signal
- 한 줄: 가깝지만 다른 class인 놈이 진짜 공부거리.

### Lifted Structured Loss (Song et al., p.15–17)
- **Problem**: Contrastive/triplet use only a few pairs per batch.
- **Core idea**: Use all the pairwise edges within one training batch (computational efficiency); hardest negative is chosen by `max`, which is not smooth, so it is relaxed with log-sum-exp.
- **vs. previous / key difference**: Uses the whole batch, but still pair-based, $O(N^2)$, 1 positive ↔ 1 negative → motivates N-pair.
- **Keywords**: all pairwise edges, not smooth, relaxed, log-sum-exp, $O(N^2)$
- 한 줄: batch 전체 edge 활용 + max → LSE로 smooth화.

### Why Replace Max? — Max vs Sum vs LSE (p.16)
- **Core idea**: Max gives a sparse gradient (only the hardest), Sum/average gives uniform weights (not focused on hard ones), Log-Sum-Exp gives softmax weights → dense & focused gradient = soft hard negative mining; LSE is a smooth approximation of max.
- **Keywords**: sparse, uniform, dense & focused, softmax weight, smooth approximation
- 한 줄: LSE = 모든 negative에 gradient + 어려운 놈일수록 크게.

### N-pair Loss (Sohn et al., NeurIPS 2016, p.17–20)
- **Problem**: Triplet/lifted compare 1 positive with 1 negative at a time.
- **Core idea**: Multi-class N-pair loss generalizes triplet loss to compare 1 positive against N-1 negatives simultaneously; $\log(1+\sum e^{f^\top f^-_i - f^\top f^+})$ = cross entropy loss for an N-way softmax classifier.
- **vs. previous / key difference**: 1 positive ↔ many negatives → more information, (slide) no need of hard negative mining.
- **Keywords**: (N+1)-tuplet, multiple negatives, N-way softmax, cross entropy
- 한 줄: "N개 중 positive 찾기" 분류 문제로 재정의.

### NCE — Noise Contrastive Estimation (Gutmann et al., 2010, p.21–22)
- **Problem**: Can negatives be treated as noise and interpreted probabilistically?
- **Core idea**: Positive vs Negative → Data vs Noise; run logistic regression (sigmoid + binary cross-entropy) on the logit $\log p_\theta(u) - \log q(u)$ to tell target data from noise; a method for estimating parameters of a statistical model.
- **Keywords**: data vs noise, logistic regression, sigmoid, binary CE, parameter estimation
- 한 줄: NCE = 샘플별 data/noise 이진 분류.

### InfoNCE (Oord et al., 2018, CPC, p.23–28)
- **Problem**: Probabilistic version of the N-way positive identification.
- **Core idea**: Uses categorical cross-entropy to identify the positive sample amongst a set of unrelated noise samples; via Bayes rule the optimal $f(x,c) \propto p(x|c)/p(x)$ (density ratio). Learning a good $f$ → good density ratio → increasing mutual information.
- **vs. previous / key difference**: NCE = binary (sigmoid), InfoNCE = N-way categorical (softmax); same form as N-pair with anchor = context $c$.
- **Keywords**: categorical CE, density ratio, Bayes rule, mutual information, context
- 한 줄: InfoNCE는 단순 matching이 아니라 x와 c의 shared information을 늘림.

### Soft-Nearest Neighbors Loss (Salakhutdinov 2007, Frosst 2019, p.29)
- **Core idea**: Extends the loss to include multiple positive samples (all neighbors of the same class in the numerator, all neighbors in the denominator); temperature tunes how concentrated the features are.
- **Keywords**: multiple positives, same-class neighbors, temperature, concentration
- 한 줄: positive 여러 개 버전 (label 필요).

### Common Setup for Modern Contrastive Learning (p.31–32)
- **Core idea**: Anchor (query) / Positive = another augmented view of the same sample / Negatives = other samples in the batch / shared encoder (usually L2-normalized) / cosine similarity / InfoNCE (NT-Xent) with temperature $\tau$.
- **Positive construction**: with labels → same class is positive; without labels → augmented views of the same image are positive, other images in the batch are negative. $p_{pos}$ must satisfy Symmetry and Matching marginal.
- **Keywords**: anchor, augmented view, in-batch negatives, cosine, NT-Xent, symmetry, matching marginal
- 한 줄: label 없으면 augmentation이 positive를 정의.

### GenView (Li et al., ECCV 2024, p.33–45) — *views*
- **Problem**: Handcrafted views have two limitations: (a) False Positive Pair — a strong crop may remove the main object, (b) Limited Diversity — a weak crop may produce two nearly identical views.
- **Core idea**: (1) Adaptive View Generation: CLIP image embedding + adaptive noise (based on the foreground ratio) → frozen Stable unCLIP generates a diverse positive view; (2) Quality-Driven Contrastive Loss: a separate frozen encoder scores each pair by $q_i = s^f_i - s^b_i$ (similar foreground, different background) and $w_i = \text{softmax}(q_i)$ weights the SSL loss.
- **vs. previous / key difference**: Varies pose, viewpoint, background while preserving object identity; plug-in to various SSL losses (MoCo, SimSiam, BYOL, SwAV...). Small foreground → less noise, large foreground → more noise.
- **Keywords**: false positive pair, limited diversity, Stable unCLIP, adaptive noise, foreground ratio, quality-driven loss
- 한 줄: 생성 모델로 더 좋은 positive view 만들고, 품질 좋은 pair에 가중치.

### Feature Suppression & MCL (Zhang et al., ECCV 2024, p.46–61) — *features*
- **Problem**: Feature suppression = useful information present in the input is lost in the representation (SimCLR ignores shape, 0.935; CLIP ignores orientation, 0.933). It happens because the model learns the easiest discriminative feature (e.g., color) and the contrastive objective is already satisfied.
- **Core idea**: Multistage Contrastive Learning: ① Train → ② Cluster (K-means) → ③ Select negatives within clusters → ④ Repeat → ⑤ Concatenate; each stage removes the shortcut used in previous stages (Stage 0 color → Stage 1 texture → Stage 2 shape).
- **vs. previous / key difference**: Unlike hard negative mining (negatives close to the current embedding, to find difficult samples), MCL selects negatives from the same cluster of the previous stage to prevent the reuse of previously learned features.
- **Keywords**: feature suppression, easiest discriminative feature, K-means, same-cluster negatives, concatenation
- 한 줄: 이전 stage가 쓴 shortcut을 negative 선택으로 막아 억눌린 feature를 학습.

### X-CLR (Sobal et al., ICLR 2025, p.62–72) — *relationships*
- **Problem**: Standard targets are binary — another dog, a cat, and a piano are treated as equally negative.
- **Core idea**: Replace a binary graph with a soft similarity graph: a fixed text encoder embeds captions/class descriptions (e.g., "a photo of a Maltese dog"), their similarity (softmax with $\tau_s$) is the target graph, and the learned image-similarity graph is matched to it with cross-entropy $H(s_i, p_i)$.
- **vs. previous / key difference**: SSL graph connects only two views of the same image, SupCon connects same-class samples (binary), X-CLR allows different degrees of similarity. Low $\tau_s$ ≈ standard contrastive, high $\tau_s$ → softer graph.
- **Keywords**: soft similarity graph, target graph vs learned graph, fixed text encoder, $\tau_s$, low-data
- 한 줄: one-hot target → text 기반 soft target (InfoNCE의 soft label 버전).

## B. Must-know definitions & distinctions
- **Representation Learning = Learning Similarity**: semantically similar samples close, dissimilar far (p.5).
- **Siamese network**: same network + shared weights → both inputs in the same embedding space (p.8).
- **Margin ε (contrastive)**: minimum distance required between different classes (p.11).
- **Margin ε (triplet)**: minimum offset between anchor–positive and anchor–negative distances (p.13).
- **Hard negative**: close to the anchor but belongs to a different class (p.14).
- **Contrastive vs Triplet**: pairs + absolute targets vs triplets + relative ordering (p.11–13).
- **Max vs Sum vs LSE**: sparse vs uniform vs dense & focused (softmax-weighted) gradient (p.16).
- **Lifted vs N-pair**: all pairwise edges but still 1 pos ↔ 1 neg vs 1 pos ↔ many negatives at once (p.17).
- **NCE vs InfoNCE**: data vs noise logistic regression (binary) vs categorical CE over N candidates (p.21–23).
- **InfoNCE optimum**: $f(x,c) \propto p(x|c)/p(x)$ (density ratio) → increases mutual information (p.24, p.28).
- **Soft-NN**: multiple positives; temperature = how concentrated features are (p.29).
- **With labels vs without labels**: same class = positive vs augmented views = positive, other batch images = negative (p.32).
- **False positive pair vs limited diversity**: strong crop removes object vs weak crop gives nearly identical views (p.36).
- **GenView good pair**: similar foreground + different background (p.42).
- **Feature suppression**: useful input information lost in the representation (p.49).
- **MCL vs hard negative mining**: same-cluster (previous stage) negatives to prevent feature reuse vs close-to-current-embedding negatives to find difficult samples (p.56).
- **SSL vs SupCon vs X-CLR graph**: two views only vs same class (binary) vs soft degrees of similarity (p.65).
- **τ vs τs (X-CLR)**: image-similarity contrastive temperature vs softness of the target graph (p.68).
- **Three views**: GenView = views, MCL = features, X-CLR = relationships (p.73).

## C. Question pool (18 questions)

### Q1 [T/F]
In a Siamese network, the two inputs are processed by the same network with shared weights, so that both images are mapped into the same embedding space.
**Answer**: True — "Same Network + Shared Weights", "Both images are mapped into the same embedding space." (p.8)

### Q2 [T/F]
A key strength of the contrastive loss (Chopra et al., 2005) is that it explicitly enforces the relative ordering between positive and negative distances.
**Answer**: False — 반대. 슬라이드 한계로 "Does not explicitly enforce relative ordering"이 명시됨 → 이게 triplet loss의 동기. (p.11)

### Q3 [MCQ]
Which of the following is NOT listed as a problem of the triplet loss?
(A) The number of possible triplets grows as $N^3$ for $N$ samples.
(B) It requires a pretrained generative model to create positive samples.
(C) Easy triplets dominate, so most triplets are useless.
(D) Randomly chosen negatives often already satisfy the margin, so selecting hard negatives is crucial.
**Answer**: (B) — triplet loss의 문제는 Triplet Explosion($N^3$)과 easy triplet 지배. 생성 모델 의존은 GenView의 한계. (p.13)

### Q4 [T/F]
Hard negatives are samples that are close to the anchor but belong to a different class, and they provide weaker learning signals than easy negatives.
**Answer**: False — hard negative는 anchor와 가깝지만 다른 class라서 **stronger** learning signal을 줌. easy negative가 loss·gradient 기여가 작음. (p.14)

### Q5 [MCQ]
Why is the `max` over negatives in the Lifted Structured Loss replaced by log-sum-exp?
(A) Log-sum-exp ignores all negatives except the hardest one, making the gradient sparse.
(B) Log-sum-exp gives every negative the same uniform weight, like an average.
(C) Log-sum-exp removes the need for a positive pair in the loss.
(D) `max` is not smooth; log-sum-exp is a smooth approximation of max that gives dense gradients with larger weights on harder negatives.
**Answer**: (D) — "not smooth, so relaxed"; LSE = dense & focused gradient (softmax weight). (A)는 max, (B)는 sum의 특징. (p.15–16)

### Q6 [T/F]
The N-pair loss compares one positive against multiple negatives simultaneously, and its form is equivalent to the cross entropy loss of an N-way softmax classifier.
**Answer**: True — "1 positive <-> many negatives", "Cross entropy loss for a N-way softmax classifier". (p.17–20)

### Q7 [MCQ]
What is the core idea of Noise Contrastive Estimation (NCE)?
(A) Run logistic regression to tell apart target data from noise samples.
(B) Choose the positive among N candidates with a categorical cross-entropy loss.
(C) Generate positive views with a diffusion model and weight them by quality.
(D) Use all neighbors of the same class as positives with a temperature.
**Answer**: (A) — "Positive vs Negative → Data vs Noise", logistic regression으로 구분. (B) InfoNCE, (C) GenView, (D) Soft-NN. (p.21)

### Q8 [T/F]
The InfoNCE loss uses a binary cross-entropy loss to classify each sample independently as data or noise.
**Answer**: False — 그건 NCE. InfoNCE는 **categorical cross-entropy**로 unrelated noise samples 중 positive를 식별. (p.23)

### Q9 [MCQ]
According to the probabilistic derivation of InfoNCE, the optimal score function $f(\mathbf{x}, \mathbf{c})$ is proportional to:
(A) $p(\mathbf{x})$
(B) $p(\mathbf{x}, \mathbf{c})$
(C) $p(\mathbf{x}|\mathbf{c}) / p(\mathbf{x})$
(D) $p(\mathbf{x}) / p(\mathbf{x}|\mathbf{c})$
**Answer**: (C) — Bayes rule 식을 $\prod p(x_i)$로 나누면 density ratio $p(x|c)/p(x)$; 이게 mutual information 증가로 연결. (p.24, p.28)

### Q10 [T/F]
Pretext tasks such as rotation prediction, jigsaw puzzles, and colorization are handcrafted objectives, which motivates viewing representation learning more generally as learning similarity.
**Answer**: True — "They are all handcrafted Objectives" → "Representation Learning = Learning Similarity". (p.4–5)

### Q11 [MCQ]
In the common setup for modern contrastive learning without labels, how are the positive and negatives obtained for an anchor?
(A) Positive = a sample of the same class; negatives = samples of other classes, using ground-truth labels.
(B) Positive = another augmented view of the same sample; negatives = other samples in the batch.
(C) Positive = the nearest neighbor in the current embedding; negatives = the farthest samples.
(D) Positive = a sample from the same K-means cluster; negatives = samples from other clusters.
**Answer**: (B) — label 없을 땐 augmentation view가 positive, batch 내 다른 이미지가 negative. (A)는 label 있는 경우. (p.31–32)

### Q12 [T/F]
In GenView's adaptive view generation, images with a small foreground receive more noise so that more variation can be introduced.
**Answer**: False — 반대. Small foreground → **less** noise (object가 쉽게 사라짐), large foreground → more noise. (p.40)

### Q13 [MCQ]
In GenView's quality-driven contrastive loss, which positive pair receives a higher weight?
(A) A pair with similar foreground and different background.
(B) A pair with different foreground and similar background.
(C) A pair with similar foreground and similar background.
(D) A pair with different foreground and different background.
**Answer**: (A) — "Similar foreground, Different background"; $q_i = s^f_i - s^b_i$, $w_i=\text{softmax}(q_i)$. (p.42)

### Q14 [T/F]
According to the limitations of handcrafted views, a strong crop may produce two nearly identical views (limited diversity), while a weak crop may remove the main object (false positive pair).
**Answer**: False — 반대. **Strong** crop → main object 제거 (False Positive Pair), **weak** crop → 거의 같은 두 view (Limited Diversity). GenView의 motivation: good positive pair = preserve semantics + introduce diversity. (p.36)

### Q15 [T/F]
Feature suppression happens because, once the easiest discriminative feature (e.g., color) already satisfies the contrastive objective, the model has no need to additionally learn other useful features such as texture and shape.
**Answer**: True — "The model learns the easiest discriminative feature (Color), while other useful features are suppressed." (p.50)

### Q16 [MCQ]
How does Multistage Contrastive Learning (MCL) differ from hard negative mining?
(A) MCL selects negatives that are closest to the anchor in the current embedding to find difficult samples.
(B) MCL generates new negatives with a diffusion model at every stage.
(C) MCL replaces the binary targets with a soft similarity graph from a text encoder.
(D) MCL selects negatives only from the same cluster of the previous stage, to prevent reusing previously learned features.
**Answer**: (D) — "Rather than finding difficult samples, the goal is to prevent the reuse of previously learned features." (A)는 hard negative mining 정의. (p.56)

### Q17 [T/F]
In X-CLR, a low target temperature $\tau_s$ makes the target similarity graph softer by giving more weight to related samples.
**Answer**: False — 반대. Low $\tau_s$ → strongest positive에 집중 (standard CL과 유사), **High** $\tau_s$ → softer graph. (p.68)

### Q18 [MCQ]
Which matching of paper ↔ solution is correct?
(A) GenView — learn different features across stages
(B) MCL — learn from a soft similarity graph
(C) X-CLR — learn from a soft similarity graph built from text/metadata, instead of a binary graph
(D) X-CLR — generate and weight better positive views
**Answer**: (C) — GenView = views (generate & weight positive views), MCL = features (stages), X-CLR = relationships (soft similarity graph). (p.65–66, p.73)

<!-- REVIEWED | fixes: 2 | reviewer: opus -->
