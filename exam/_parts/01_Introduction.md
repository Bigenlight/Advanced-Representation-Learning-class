# 01. Introduction — Core Ideas & Question Pool

## A. Core ideas (one block per method/paper/concept, in lecture order)

### What AI has been learning (p.7)
- **Problem**: How has the target of what AI learns evolved over time?
- **Core idea**: Rules → Patterns → Features → Representations → World Knowledge; human-injected knowledge decreases while the abstraction and generality of what is learned from data increases.
- **vs. previous / key difference**: Features = edges, textures, shapes, parts; Representations = object/scene/semantic-level embeddings that generalize (transfer) across tasks.
- **Keywords**: hand-crafted rules, patterns from data, meaningful features, generalize across tasks, world knowledge
- 한 줄: 사람이 넣는 지식↓, 데이터에서 배우는 표현의 추상성·일반성↑. 이 강의의 중심은 4단계 Representations.

### What is a representation? (p.9–13)
- **Problem**: To a computer, an image is initially just a matrix of numbers — raw pixels are not understanding.
- **Core idea**: Representation learning maps data points to a simpler, organized space, $\mathbf{z}=f(\mathbf{x})$ (encoder $f$: data space $\mathcal{X}$ → representation space $\mathcal{Z}$), keeping useful information and discarding irrelevant information.
- **vs. previous / key difference**: The same data can have many different representations (Color Histogram = global color distribution, CNN Feature = hierarchical visual features, CLIP Embedding = aligns image with text semantics).
- **Keywords**: matrix of numbers, encoder, simpler organized space, downstream tasks, often lower dimension ($D \ll \dim(\mathbf{x})$), discard irrelevant information
- 한 줄: representation = downstream task를 쉽게 만드는 형태로 raw data를 변환한 것. 배경·조명·노이즈·camera angle은 버림.

### Feature engineering (hand-crafted features) (p.14–15)
- **Problem**: Before deep learning, who designs the features? → Humans.
- **Core idea**: Image → hand-crafted features (SIFT / HOG / SURF) → classifier (SVM, Random Forest, k-NN); different tasks require different features.
- **vs. previous / key difference**: Problems = Human-designed (domain knowledge, labor-intensive), Task-specific (new task = new feature design), Limited scalability (performance ceiling).
- **Keywords**: SIFT, HOG, SURF, SVM, task-specific, limited scalability, "Every new task requires new features"
- 한 줄: 사람이 feature 설계 + classifier 따로 → task마다 새 feature 필요, 확장성 한계.

### Feature learning (deep learning) & representation reuse (p.16–17)
- **Problem**: Hand-crafted features don't generalize or scale.
- **Core idea**: Deep learning learns representations automatically from data (Image → CNN → Representation → Classifier); a backbone pretrained on ImageNet is reused for detection, segmentation, tracking — "One representation, many applications. Learn once, use everywhere."
- **vs. previous / key difference**: Features are learned from data instead of designed by humans, and one general-purpose representation replaces task-specific models (Model A/B/C).
- **Keywords**: feature learning, AlexNet, Layer 1 edges → Layer 3 textures → Layer 5 object parts, increasingly semantic, pretraining, backbone, reuse
- 한 줄: CNN이 low-level → high-level 계층 feature를 스스로 학습, 한 번 학습한 backbone을 여러 task에 재사용.

### What makes a good representation: Compression (p.18–19)
- **Problem**: What properties should a good representation have? (Compress information / Support prediction / Generalize across tasks)
- **Core idea**: Compression captures just the essential characteristics of the data — benefits: Less Memory, Invariance (remove nuisance factors), Occam's Razor (prefer the simpler one).
- **vs. previous / key difference**: "Compression is not about losing information. It is about removing irrelevant information."
- **Keywords**: essential characteristics, less memory, invariance, nuisance factors, Occam's razor
- 한 줄: 압축 = 정보 손실이 아니라 무관한 정보(nuisance) 제거.

### What makes a good representation: Prediction (p.20)
- **Problem**: Why do we need representations?
- **Core idea**: To predict what we cannot directly observe — Future (video prediction, V-JEPA), Past (state estimation, temporal reasoning), Missing Data (MAE, image completion/inpainting).
- **Keywords**: unobserved properties, future, past, missing data, inpainting
- 한 줄: Prediction = 세계의 관측되지 않은 속성을 추론하는 것.

### Good Representations = Compression + Prediction (p.21–22)
- **Problem**: How do the two principles combine?
- **Core idea**: The intersection of Compression (Autoencoder, Contrastive, Clustering) and Prediction (Future prediction, Imputation, Pretext tasks) gives better representations that generalize better, transfer across tasks, support decision making, and are more sample efficient.
- **vs. previous / key difference**: Beyond these, good representations often exhibit Disentangled, Interpretable, Actionable, Transferable properties — often as consequences, not direct goals.
- **Keywords**: intersection, sample efficient, disentangled, interpretable, actionable, transferable
- 한 줄: 둘 다 만족하는 교집합이 좋은 representation, 추가 성질 4개는 그 결과로 흔히 나타남.

### Types of representation learners (p.23)
- **Problem**: How can existing learning methods be organized?
- **Core idea**: Compression: Autoencoding (remove redundant information), Contrastive (achieve invariance to viewing transformations), Clustering (quantize continuous data into discrete categories). Prediction: Future prediction (predict the future), Imputation (predict missing data), Pretext tasks (predict abstract properties of your data).
- **Keywords**: autoencoding, contrastive, clustering, future prediction, imputation, pretext tasks
- 한 줄: 6종 learner를 Compression 3 / Prediction 3으로 분류. Contrastive가 Compression 쪽인 게 함정 포인트.

## B. Must-know definitions & distinctions
- **Representation**: "a transformation of raw data into a form that makes downstream tasks easier" (p.12); $\mathbf{z}=f(\mathbf{x})$, $f$ = encoder.
- **Representation learning**: "aims to map data points to a simpler, organized space" (p.12).
- **Good representation**: "keeps useful information and discards irrelevant information" (p.11); "can be reused across many tasks" (p.17).
- **Compression**: "Captures just the essential characteristics of the data" — not losing information, but removing irrelevant information (p.19).
- **Prediction**: "inferring unobserved properties of the world" (p.20).
- **Invariance**: remove nuisance factors that do not affect semantics (p.19).
- **Occam's Razor**: among equally useful representations, prefer the simpler one (p.19).
- **Feature engineering vs Feature learning**: humans manually design features (SIFT/HOG/SURF) vs deep learning learns representations automatically from data (p.14, p.16).
- **Task-specific model vs General-purpose representation**: separate Model A/B/C per task vs learn once, reuse for many tasks (p.17).
- **Features vs Representations (AI evolution)**: edges/textures/shapes/parts vs semantic embeddings that generalize across tasks (p.7).
- **Color Histogram vs CNN Feature vs CLIP Embedding**: global color distribution vs hierarchical visual features vs image–text semantic alignment (p.13).
- **Compression learners vs Prediction learners**: Autoencoding / Contrastive / Clustering vs Future prediction / Imputation / Pretext tasks (p.23).
- **Disentangled / Interpretable / Actionable / Transferable**: additional properties, often consequences of learning compressive + predictive representations (p.22).

## C. Question pool (18 questions)

### Q1 [T/F]
To a computer, an image is initially just a matrix of numbers.
**Answer**: True — 슬라이드 원문 그대로. raw pixel ≠ understanding (p.10)

### Q2 [T/F]
A good representation keeps all information in the input image, including background, lighting, and camera angle, so that no information is lost.
**Answer**: False — background, lighting, noise, camera angle은 irrelevant information으로 discard (p.11)

### Q3 [MCQ]
Which of the following best describes a representation, according to the lecture?
(A) A set of hand-crafted rules that maps inputs to outputs
(B) A transformation of raw data into a form that makes downstream tasks easier
(C) A lossless copy of the raw pixel values stored in a smaller file format
(D) The final classifier output (e.g., the label "Dog")
**Answer**: (B) — p.12 정의 원문 (p.12)

### Q4 [T/F]
Representation learning aims to map data points to a simpler, organized space, and the representation is often of lower dimension than the input.
**Answer**: True — "simpler, organized space", "Often in lower dimension" (p.12)

### Q5 [MCQ]
In the order "What AI has been learning," which sequence is correct?
(A) Patterns → Rules → Features → Representations → World Knowledge
(B) Rules → Features → Patterns → World Knowledge → Representations
(C) Rules → Patterns → Features → Representations → World Knowledge
(D) Features → Rules → Patterns → Representations → World Knowledge
**Answer**: (C) — From rules to world knowledge (p.7)

### Q6 [T/F]
The same data can have only one correct representation; e.g., a color histogram and a CNN feature of the same image must encode the same information.
**Answer**: False — "The same data can have many different representations" (Color Histogram / CNN Feature / CLIP Embedding) (p.13)

### Q7 [MCQ]
Which representation of a dog image "aligns image with text semantics"?
(A) Color Histogram
(B) HOG descriptor
(C) CNN Feature
(D) CLIP Embedding
**Answer**: (D) — Color Histogram = global color distribution, CNN Feature = hierarchical visual features (p.13)

### Q8 [T/F]
In the feature engineering era, features such as SIFT, HOG, and SURF were learned automatically from data by deep networks.
**Answer**: False — humans manually designed features; deep learning 이전 시대 (p.14)

### Q9 [MCQ]
Which of the following is NOT one of the problems of hand-crafted features listed in the lecture?
(A) Human-designed (requires domain knowledge, labor-intensive)
(B) Task-specific (new task = new feature design)
(C) Limited scalability (performance ceiling)
(D) Requires large-scale labeled data for end-to-end training
**Answer**: (D) — 슬라이드의 세 문제는 Human-designed / Task-specific / Limited scalability. (D)는 hand-crafted가 아니라 deep learning pretraining 쪽 얘기 (p.15)

### Q10 [T/F]
Deep learning learns representations automatically from data, and in AlexNet the features become increasingly semantic from lower layers (edges) to higher layers (object parts).
**Answer**: True — Layer 1 Edges → Layer 3 Textures → Layer 5 Object Parts, "Representations become increasingly semantic" (p.16)

### Q11 [T/F]
"One representation, many applications. Learn once, use everywhere." refers to training a separate task-specific model from scratch for detection, segmentation, and tracking.
**Answer**: False — 반대. ImageNet pretrained backbone을 여러 task에 reuse하는 것 (p.16–17)

### Q12 [MCQ]
Which is NOT one of the three things a good representation should do, according to "What Makes a Good Representation?"
(A) Compress information
(B) Support prediction
(C) Generalize across tasks
(D) Memorize every training example exactly
**Answer**: (D) — Compress / Support prediction / Generalize across tasks (p.18)

### Q13 [T/F]
According to the lecture, compression is about losing information to make the representation smaller.
**Answer**: False — "Compression is not about losing information. It is about removing irrelevant information." (p.19)

### Q14 [MCQ]
Which benefit of compression is described as "Among equally useful representations, prefer the simpler one"?
(A) Less Memory
(B) Invariance
(C) Occam's Razor
(D) Transferability
**Answer**: (C) — Less Memory = easier to store, Invariance = remove nuisance factors (p.19)

### Q15 [T/F]
In the lecture, prediction means inferring unobserved properties of the world, such as the future, the past, or missing data.
**Answer**: True — Future (V-JEPA) / Past (state estimation) / Missing Data (MAE, inpainting) (p.20)

### Q16 [MCQ]
According to the "Types of Representation Learners" table, which learning method belongs to the **Compression** principle?
(A) Imputation
(B) Contrastive
(C) Future prediction
(D) Pretext tasks
**Answer**: (B) — Compression = Autoencoding / Contrastive / Clustering. 나머지는 Prediction (p.21, p.23)

### Q17 [MCQ]
Which pairing of learning method and short summary is correct?
(A) Clustering — Predict missing data
(B) Autoencoding — Achieve invariance to viewing transformations
(C) Contrastive — Quantize continuous data into discrete categories
(D) Pretext tasks — Predict abstract properties of your data
**Answer**: (D) — Clustering = quantize, Autoencoding = remove redundant information, Contrastive = invariance, Imputation = missing data (p.23)

### Q18 [T/F]
Disentangled, Interpretable, Actionable, and Transferable are additional desirable properties that are often consequences of learning good representations, rather than the main principles themselves.
**Answer**: True — "Beyond Compression and Prediction", "often consequences of learning representations that are both..." (p.22)

<!-- REVIEWED | fixes: 0 | reviewer: opus -->

## D. Fill-in-the-blank pool (7 questions)

### F1 [Blank]
In "What AI has been learning," the target of learning moves from rules to world knowledge: Rules → Patterns → Features → ______ → World Knowledge. This stage means learning object-, scene-, and semantic-level embeddings that generalize across tasks.
**Answer**: Representations (accept: representation) — 3단계 Features(edges, textures, shapes, parts) 다음이 4단계 **Representations** (transfer to many tasks). 사람이 넣는 지식↓, 학습된 것의 추상성·일반성↑. 이 강의의 중심 단계 (p.7)

### F2 [Blank]
A representation is a transformation of raw data into a form that makes (1) ______ tasks easier. It is written $\mathbf{z}=f(\mathbf{x})$, where the mapping $f$ from data space $\mathcal{X}$ to representation space $\mathcal{Z}$ is called the (2) ______.
**Answer**: (1) downstream (2) encoder (accept: encoding function) — p.12의 ★ 정의 문장 + "Mapping (Encoding)" 그림 그대로. $\mathbf{z}$를 classification / detection / retrieval 같은 **downstream task**에 씀, 보통 $D \ll \dim(\mathbf{x})$ (p.12)

### F3 [Blank]
Before deep learning, features such as SIFT, HOG, and SURF were ______ features, and a separate classifier (e.g., SVM) was trained on top of them.
**Answer**: hand-crafted (accept: handcrafted, hand-designed, human-designed, manually designed, engineered) — "Feature Engineering Era: Humans manually designed features." 문제 3개 = Human-designed / Task-specific / Limited scalability → "Every new task requires new features" (p.14–15)

### F4 [Blank]
With deep learning, a backbone pretrained on ImageNet is reused for detection, segmentation, and tracking: "One representation, many applications. Learn once, use ______."
**Answer**: everywhere — **representation reuse**. task마다 Model A/B/C를 따로 학습하는 task-specific model과 대비되는 general-purpose representation (p.16–17)

### F5 [Blank]
"Compression is not about losing information. It is about removing ______ information."
**Answer**: irrelevant (accept: task-irrelevant, nuisance, redundant — p.21 "removing redundancy and nuisances") — 압축 = 정보 손실이 아니라 **무관한 정보(nuisance) 제거**. 이점 3개: Less Memory / Invariance / Occam's Razor (p.19)

### F6 [Blank]
"Prediction means inferring ______ properties of the world," such as the future (video prediction, V-JEPA), the past (state estimation), or missing data (MAE, inpainting).
**Answer**: unobserved (accept: unobservable, not directly observed, hidden) — "Why do we need representations? To predict what we cannot directly observe." (p.20)

### F7 [Blank]
In the "Types of Representation Learners" table, Autoencoding, Contrastive, and Clustering belong to the (1) ______ principle, while Future prediction, Imputation, and Pretext tasks belong to the (2) ______ principle.
**Answer**: (1) Compression (2) Prediction — Contrastive가 **Compression** 쪽("Achieve invariance to viewing transformations")인 게 함정 포인트. 두 원리의 교집합 = Better Representations (generalize better, transfer, sample efficient) (p.21, p.23)

<!-- AUGMENTED 2026-10-22 | C fixes: 0 | C added: 0 | D blanks: 7 | author: opus -->

<!-- VERIFIED 2026-10-22 | checked: 25 | fixed: 3 | deleted: 0 | reviewer: opus
F2: accept에 "encoding function" 추가 (p.12 "Mapping (Encoding)" 박스 표현)
F3: stem의 "humans manually designed them" 삭제 — accept 목록(manually designed / human-designed)을 그대로 알려주는 giveaway였음. "engineered"(feature engineering era) accept 추가
F5: accept에 "redundant" 추가 — p.21 Compression "removing redundancy and nuisances"로도 방어 가능
-->
