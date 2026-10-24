# 고급표현학습 중간고사 퀴즈 대비 — Core Ideas (01–07)

> 퀴즈: **2026-10-29 (목), 20문제, 영어 T/F + MCQ + 빈칸(fill-in-the-blank), 쉬움.** 교수님: *"please study basic concept & core idea of the papers"* + *"study the pseudo code"* (10/22 추가 공지)
> 각 방법마다 **Problem → Core idea → vs. previous** 순서. 문제 pool 원본은 `exam/_parts/NN_*.md` (C: T/F·MCQ, D: 빈칸), 모의고사는 `exam/mock_quiz_1~3.md` (T/F·MCQ) + `exam/mock_quiz_4~5.md` (빈칸·pseudo code 포함 새 형식).
> **Pseudo code 9개 (X-CLR · SimCLR · DINO · MoCo · CLIP · Π-model · Temporal Ensembling · Label Propagation · DivideMix) → [`exam/pseudocode.md`](pseudocode.md)** 에 따로 정리.

## 목차

- [01. Introduction](#01)
- [02. Self-Supervised Learning (SSL)](#02)
- [03. Contrastive Learning I](#03)
- [04. Contrastive Learning II](#04)
- [05. Contrastive Learning III](#05)
- [06. Semi-Supervised Learning I](#06)
- [07. Semi-Supervised Learning II (Pseudo Labeling)](#07)

---

<a id="01"></a>

## 01. Introduction

### A. Core ideas

#### What AI has been learning (p.7)
- **Problem**: How has the target of what AI learns evolved over time?
- **Core idea**: Rules → Patterns → Features → Representations → World Knowledge; human-injected knowledge decreases while the abstraction and generality of what is learned from data increases.
- **vs. previous / key difference**: Features = edges, textures, shapes, parts; Representations = object/scene/semantic-level embeddings that generalize (transfer) across tasks.
- **Keywords**: hand-crafted rules, patterns from data, meaningful features, generalize across tasks, world knowledge
- 한 줄: 사람이 넣는 지식↓, 데이터에서 배우는 표현의 추상성·일반성↑. 이 강의의 중심은 4단계 Representations.

#### What is a representation? (p.9–13)
- **Problem**: To a computer, an image is initially just a matrix of numbers — raw pixels are not understanding.
- **Core idea**: Representation learning maps data points to a simpler, organized space, $\mathbf{z}=f(\mathbf{x})$ (encoder $f$: data space $\mathcal{X}$ → representation space $\mathcal{Z}$), keeping useful information and discarding irrelevant information.
- **vs. previous / key difference**: The same data can have many different representations (Color Histogram = global color distribution, CNN Feature = hierarchical visual features, CLIP Embedding = aligns image with text semantics).
- **Keywords**: matrix of numbers, encoder, simpler organized space, downstream tasks, often lower dimension ($D \ll \dim(\mathbf{x})$), discard irrelevant information
- 한 줄: representation = downstream task를 쉽게 만드는 형태로 raw data를 변환한 것. 배경·조명·노이즈·camera angle은 버림.

#### Feature engineering (hand-crafted features) (p.14–15)
- **Problem**: Before deep learning, who designs the features? → Humans.
- **Core idea**: Image → hand-crafted features (SIFT / HOG / SURF) → classifier (SVM, Random Forest, k-NN); different tasks require different features.
- **vs. previous / key difference**: Problems = Human-designed (domain knowledge, labor-intensive), Task-specific (new task = new feature design), Limited scalability (performance ceiling).
- **Keywords**: SIFT, HOG, SURF, SVM, task-specific, limited scalability, "Every new task requires new features"
- 한 줄: 사람이 feature 설계 + classifier 따로 → task마다 새 feature 필요, 확장성 한계.

#### Feature learning (deep learning) & representation reuse (p.16–17)
- **Problem**: Hand-crafted features don't generalize or scale.
- **Core idea**: Deep learning learns representations automatically from data (Image → CNN → Representation → Classifier); a backbone pretrained on ImageNet is reused for detection, segmentation, tracking — "One representation, many applications. Learn once, use everywhere."
- **vs. previous / key difference**: Features are learned from data instead of designed by humans, and one general-purpose representation replaces task-specific models (Model A/B/C).
- **Keywords**: feature learning, AlexNet, Layer 1 edges → Layer 3 textures → Layer 5 object parts, increasingly semantic, pretraining, backbone, reuse
- 한 줄: CNN이 low-level → high-level 계층 feature를 스스로 학습, 한 번 학습한 backbone을 여러 task에 재사용.

#### What makes a good representation: Compression (p.18–19)
- **Problem**: What properties should a good representation have? (Compress information / Support prediction / Generalize across tasks)
- **Core idea**: Compression captures just the essential characteristics of the data — benefits: Less Memory, Invariance (remove nuisance factors), Occam's Razor (prefer the simpler one).
- **vs. previous / key difference**: "Compression is not about losing information. It is about removing irrelevant information."
- **Keywords**: essential characteristics, less memory, invariance, nuisance factors, Occam's razor
- 한 줄: 압축 = 정보 손실이 아니라 무관한 정보(nuisance) 제거.

#### What makes a good representation: Prediction (p.20)
- **Problem**: Why do we need representations?
- **Core idea**: To predict what we cannot directly observe — Future (video prediction, V-JEPA), Past (state estimation, temporal reasoning), Missing Data (MAE, image completion/inpainting).
- **Keywords**: unobserved properties, future, past, missing data, inpainting
- 한 줄: Prediction = 세계의 관측되지 않은 속성을 추론하는 것.

#### Good Representations = Compression + Prediction (p.21–22)
- **Problem**: How do the two principles combine?
- **Core idea**: The intersection of Compression (Autoencoder, Contrastive, Clustering) and Prediction (Future prediction, Imputation, Pretext tasks) gives better representations that generalize better, transfer across tasks, support decision making, and are more sample efficient.
- **vs. previous / key difference**: Beyond these, good representations often exhibit Disentangled, Interpretable, Actionable, Transferable properties — often as consequences, not direct goals.
- **Keywords**: intersection, sample efficient, disentangled, interpretable, actionable, transferable
- 한 줄: 둘 다 만족하는 교집합이 좋은 representation, 추가 성질 4개는 그 결과로 흔히 나타남.

#### Types of representation learners (p.23)
- **Problem**: How can existing learning methods be organized?
- **Core idea**: Compression: Autoencoding (remove redundant information), Contrastive (achieve invariance to viewing transformations), Clustering (quantize continuous data into discrete categories). Prediction: Future prediction (predict the future), Imputation (predict missing data), Pretext tasks (predict abstract properties of your data).
- **Keywords**: autoencoding, contrastive, clustering, future prediction, imputation, pretext tasks
- 한 줄: 6종 learner를 Compression 3 / Prediction 3으로 분류. Contrastive가 Compression 쪽인 게 함정 포인트.

### B. Must-know definitions & distinctions
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


---

<a id="02"></a>

## 02. Self-Supervised Learning (SSL)

### A. Core ideas

#### Why SSL? — Limitations of supervised learning (p.3–10)
- **Problem**: Supervised pre-training works well, but "real" labels are difficult and expensive and cannot keep up with the data we generate.
- **Core idea**: ImageNet (14M images) took 22 human years; semi-automatic labels (hashtags, GPS) are noisy. Complex concepts (video) and rare concepts (long tail) are hard to label, and new domains (e.g., medical) need fresh labels, so we need supervision that does not come from humans.
- **Keywords**: 22 human years, semi-automatic labels, hashtags / GPS, long tail (10% of classes = 93% of data), "Labelling cannot scale to the size of the data we generate"
- 한 줄: label 비용, long-tail, video, domain 차이 때문에 supervised는 scale이 안 됨 → SSL 등장

#### Self-Supervised Learning — definition & paradigm (p.11–17, p.20)
- **Problem**: How can we learn representations without manual annotation?
- **Core idea**: Obtain "labels" from the data itself by using a "semi-automatic" process and predict part of the data from other parts. A **pretext task** (no manual annotation, but trained with supervised objectives like classification or regression) learns the encoder, which is then transferred to a **downstream task** (the application you care about, with a small labeled dataset).
- **vs. previous / key difference**: Unlike unsupervised learning (clustering, PCA), SSL has an explicit prediction target generated automatically from data; unlike supervised learning, the supervision comes from the data itself, not from humans.
- **Keywords**: pseudo labels, pretext task, downstream task, "predict part of the data from other parts", Word2vec / BERT ("fill in the blanks"), MAE / SimCLR / CLIP
- 한 줄: data 일부로 다른 일부를 예측하는 pretext task로 encoder를 학습하고, 소량 label downstream task로 transfer

#### Evaluating SSL — linear probing / shallow network on frozen features (p.18, p.46, p.48)
- **Problem**: A pretext task is only a means, so how do we tell whether an SSL method learned good features?
- **Core idea**: Learn a feature extractor with the pretext task, then attach a **shallow network** (e.g., a linear classifier on "fixed" features) and train it on the target task with a small amount of labeled data. We judge the method by **downstream performance**, not by pretext-task performance.
- **Keywords**: feature extractor, shallow network, linear classifier on fixed features, downstream (classification / detection / segmentation), fine-tuning
- 한 줄: pretext 성능이 아니라 downstream 성능으로 평가, shallow head만 학습해야 representation 품질을 그대로 볼 수 있음

#### Relative Position of Patches (Doersch et al., ICCV 2015, p.21–23)
- **Problem**: Learn visual features without labels from the spatial context inside an image.
- **Core idea**: Given a center patch and one of its 8 neighbors, predict which of the 8 positions the neighbor came from (8-way classification). A two-tower (shared-weight) network has to understand object parts and their spatial layout.
- **Keywords**: 8-way classification, center patch, siamese / shared weights, local spatial context, gap + jitter (against shortcuts)
- 한 줄: 이웃 patch의 상대 위치를 8-way 분류 → part 간 local spatial 관계를 학습

#### Inpainting — Context Encoders (Pathak et al., CVPR 2016, p.24–28)
- **Problem**: Can we predict missing regions from surrounding context?
- **Core idea**: Mask a region and train an encoder–decoder (with a channel-wise FC layer) to generate it. The network learns from **reconstruction (masked L2) + adversarial loss**: L2 alone gives blurry results, adversarial alone gives sharp but out-of-context results, and L2+Adv gives the best result.
- **vs. previous / key difference**: It is a generative, pixel-level reconstruction task, while relative position is a discriminative classification task.
- **Keywords**: missing region, object parts / scene structure / semantic context, channel-wise fully connected, reconstruction loss, adversarial loss, blurry
- 한 줄: 가려진 영역을 주변 context로 복원, L2(전체 구조) + adversarial(선명함) 조합

#### Jigsaw Puzzles (Noroozi & Favaro, ECCV 2016, p.29–33)
- **Problem**: Learn object parts and their global arrangement without labels.
- **Core idea**: Shuffle 3×3 tiles and classify which permutation was used. Since 9! = 362,880 permutations are too many classes, use a subset of 64 permutations that **maximize pairwise Hamming distance**. Each tile goes through a shared-weight branch (Context-Free Network), and features are joined only at the end.
- **vs. previous / key difference**: It generalizes relative position (2 patches) to 9 patches, so it learns global instead of local spatial structure.
- **Keywords**: 9! = 362,880, Hamming distance, permutation classification, shortcut (edge continuity, color/intensity statistics), random gap / color jitter / normalization
- 한 줄: 9개 tile 순열 index를 분류, low-level shortcut을 막아야 의미 있는 feature가 나옴

#### Colorization (Zhang et al., ECCV 2016, p.34–39)
- **Problem**: Can a network recover colors from grayscale images? (Color prediction requires semantic understanding.)
- **Core idea**: In Lab space, predict the ab channels from the L (lightness) channel. Multiple colors can be plausible and regression averages them (desaturated), so the problem is solved by **classification** over **313 color classes** instead of regression.
- **Keywords**: Lab color space, L → ab, 313 color classes, regression averages (desaturation), multimodal, object semantics
- 한 줄: 흑백 L에서 ab를 예측, 색은 multimodal이라 regression 대신 313-bin classification

#### Rotation Prediction — RotNet (Gidaris et al., ICLR 2018, p.40–43)
- **Problem**: Find a simple pretext task that forces the model to recognize objects.
- **Core idea**: Rotate the image by 0°, 90°, 180° or 270° and predict which rotation was applied (4-way classification). No labels are needed: the applied rotation becomes the label. To recognize rotation, the network must recognize objects.
- **Keywords**: 4 rotations, 4-way classification, "the applied rotation becomes the label", object-centric understanding, canonical orientation
- 한 줄: 회전 각도 4-way 분류, 구현이 가장 단순하고 당시 pretext 중 최고 성능

#### Limitations of hand-crafted pretext tasks (p.44–49, p.55)
- **Problem**: Pretext tasks focus on "visual common sense", but is solving them enough for good, general features?
- **Core idea**: Problem 1: coming up with individual pretext tasks is **tedious**. Problem 2: the learned representations **may not be general**, because higher layers specialize to the pretext task (in a Jigsaw-pretrained network, linear-probe accuracy peaks at res4 and drops at the last layer, res5). We only "hope" the pre-training task and the transfer task are "aligned".
- **Keywords**: visual common sense, tedious, "The Hope of Generalization", aligned, higher layers do not generalize, tied to a specific pretext task
- 한 줄: hand-crafted pretext는 (1) 설계가 tedious, (2) 마지막 layer가 pretext에 특화돼 general하지 않음

#### Multiple Pretext Tasks — CDJP (Kim et al., WACV 2018, p.50–53)
- **Problem**: A single pretext task ties the representation to that task.
- **Core idea**: Completing Damaged Jigsaw Puzzles: one network solves **jigsaw + inpainting + colorization** at the same time on damaged patches (dropped patch, dropped color channels), with a shared backbone and task-specific heads ($L_{jig}$, $L_{inp}$, $L_{col}$). Combining tasks generally improves the results.
- **vs. previous / key difference**: Multi-task instead of single-task, but the representation can still be tied to the chosen pretext tasks ("even multi tasks", p.55).
- **Keywords**: damaged jigsaw puzzle, jigsaw + inpainting + colorization, shared backbone, multi-task loss
- 한 줄: 세 pretext를 한 네트워크로 동시에 → 단일 task보다 general, 그래도 사람이 고른 task 집합에 묶임

#### Toward a more general pretext task (p.54–56)
- **Problem**: Representations learned from image transformations may be tied to the specific pretext task.
- **Core idea**: Treat the transformed images as views of the **same object** (pull together) and other images as **different objects** (push apart). This leads to contrastive learning (next lecture).
- **Keywords**: same object vs different object, positive / negative pair, task-agnostic, contrastive learning
- 한 줄: "변환을 맞혀라" 대신 "같은 object는 가깝게, 다른 object는 멀게" → contrastive

#### Space-Time Cubic Puzzles (Kim, Cho, Kweon, AAAI 2019, p.57–66)
- **Problem**: A video additionally provides temporal order, motion and changes over time, and an action cannot be understood from two frames.
- **Core idea**: Divide a video into a 2×2×4 grid, sample 4 cubes either spatially (2×2×1) or temporally (1×1×4), shuffle them, and predict the original order (4! = 24 permutations × 2 orientations = 48 classes). Four shared 3D CNN towers process one cube each and combine features only at the end; each tower does not know whether the puzzle is spatial or temporal.
- **vs. previous / key difference**: It extends image Jigsaw to a joint space-time puzzle. Joint space-time learning beats learning each dimension separately (spatial and temporal context are complementary).
- **Keywords**: 2×2×4 grid, spatial / temporal puzzle, 3D CNN towers, 48 classes, channel replication, spatio-temporal jittering, upside-down classification
- 한 줄: video cube를 공간·시간 순서대로 맞히는 3D jigsaw, joint 학습이 따로 학습보다 좋음

#### Audio-Visual Temporal Synchronization — AVTS (Korbar et al., NeurIPS 2018, p.67–70)
- **Problem**: Use the natural synchronization of video and audio as free supervision.
- **Core idea**: Classify whether the video and audio are temporally aligned (in-sync = same time segment, out-of-sync = different time segments). A video stream (spatio-temporal CNN, motion + appearance) and an audio stream (filterbank features) are trained cooperatively with a contrastive loss. **Curriculum learning** (easy negatives first, then a mix with hard negatives) works best.
- **Keywords**: in-sync / out-of-sync pair, easy / hard / super-hard negative, cooperative learning, curriculum learning, "how positive and negative examples are constructed"
- 한 줄: audio-video sync 여부를 판별하는 multi-modal SSL, negative 구성과 curriculum이 핵심

#### Invariance vs Equivariance (p.71–75)
- **Problem**: Modern SSL trains the model to ignore transformations (random crop, color jitter, flip, blur), but is invariance always desirable?
- **Core idea**: A pretext task determines what information is preserved in the representation. **Invariance**: $f(t(x)) = f(x)$, the representation does not change. **Equivariance**: $f(t(x)) = \phi(t, f(x))$, the representation changes predictably. Classification may benefit from invariance, but detection needs spatial information and flower classification needs color information.
- **Keywords**: invariance, equivariance, $\phi(t, f(x))$, "What is the object?" vs "Where is the object?", spatial / color information
- 한 줄: 무조건 invariant하면 위치·색 정보가 사라짐 → 변환을 예측 가능하게 보존하는 equivariance 필요

#### Self-supervised Transformation Learning — STL (Yu et al., NeurIPS 2024, p.76–79)
- **Problem**: Learn equivariant representations without transformation labels (parameters).
- **Core idea**: Three losses. Brown (invariant): same image under transformation, preserve semantic information. Blue (transformation learning): **same transformation on different images**, learn transformation representations. Green (equivariant): apply the learned transformation to predict the transformed representation.
- **vs. previous / key difference**: The transformation representation is learned without transformation labels, and it discovers structure (crop is separated from color transformations, similar color transformations are close, intensity changes continuously). Preserving transformation information improves localization and generalization.
- **Keywords**: transformation representation, same transformation on different images, without transformation labels, type and intensity, EquiMod / AugSelf / SimCLR
- 한 줄: 다른 이미지·같은 변환 쌍으로 label 없이 변환 표현을 학습 → equivariance, detection 향상

### B. Must-know definitions & distinctions

- **Self-supervised learning**: obtain "labels" from the data itself by using a "semi-automatic" process; predict part of the data from other parts. (p.12)
- **Pretext task**: a self-supervised task used for learning representations; often not the "real" task we care about. No manual annotation, but trained with supervised objectives (classification or regression). (p.16, p.20)
- **Downstream task**: the application you care about; the dataset is small and labeled. (p.16)
- **SSL evaluation**: attach a shallow network on the learned feature extractor and train it on the target task with a small amount of labeled data; we care about downstream usefulness, not pretext performance. (p.18, p.46)
- **Shortcut**: solving the pretext task with low-level cues (edge continuity, color/intensity statistics, chromatic aberration) instead of semantics. (p.31, p.63)
- **Invariance**: $f(t(x)) = f(x)$, the representation does not change after the input transformation. (p.75)
- **Equivariance**: $f(t(x)) = \phi(t, f(x))$, the representation changes predictably according to the input transformation. (p.75)
- **Supervised vs Unsupervised vs Self-supervised**: human labels (expensive, task-specific) vs no labels (limited semantic meaning) vs labels generated from data (very scalable, rich and transferable). (p.13)
- **Discriminative vs generative pretext**: relative position / jigsaw / rotation (classify a transformation) vs inpainting / colorization (reconstruct missing content).
- **Local vs global spatial structure**: relative position learns local spatial context, jigsaw learns global spatial structure. (p.45)
- **L2 vs Adversarial (inpainting)**: L2 gives coarse structure but blurry output; adversarial gives sharp, realistic output. (p.25–27)
- **Regression vs classification (colorization)**: regression averages plausible colors (desaturated); classification over 313 classes keeps vivid colors. (p.37–38)
- **Easy vs hard vs super-hard negative (AVTS)**: audio from another video vs a far segment of the same video vs an adjacent (overlapping) segment of the same video. (p.68, p.70)
- **Spatial vs temporal puzzle**: "Where was each cube located?" vs "When did each cube occur?" (p.59)
- **STL losses**: Brown = same image under transformation (semantic), Blue = same transformation on different images (transformation representation), Green = apply learned transformation (equivariance). (p.76)


---

<a id="03"></a>

## 03. Contrastive Learning I

### A. Core ideas

#### From Pretext Tasks to Contrastive Learning (p.4–7)
- **Problem**: Pretext tasks (image completion, rotation, jigsaw, colorization) are all handcrafted objectives that someone has to design.
- **Core idea**: "Representation Learning = Learning Similarity": good representations place semantically similar samples nearby, so similarity itself becomes the supervision (pull similar samples together, push dissimilar samples apart).
- **vs. previous / key difference**: Instead of predicting a designed target (e.g., rotation angle), the goal is the distance/similarity relation between embeddings of a reference $x$, positive $x^+$, and negative $x^-$.
- **Keywords**: handcrafted objectives, similarity as supervision, metric learning, reference / positive / negative
- 한 줄: "task 설계하지 말고 similarity 자체를 supervision으로 쓰자" → metric learning.

#### Siamese Network & Contrastive Representation Learning (p.8–9)
- **Problem**: Two inputs must be compared in a meaningful way (e.g., same person or not).
- **Core idea**: Same network + shared weights maps both inputs into the same embedding space; $d(z_1,z_2)$ decreases for the same person and increases otherwise. Goal: $\text{score}(f(x), f(x^+)) \gg \text{score}(f(x), f(x^-))$.
- **vs. previous / key difference**: Applies to both supervised and unsupervised settings; only the definition of positives changes.
- **Keywords**: shared weights, same embedding space, score function, supervised & unsupervised
- 한 줄: 가중치 공유해야 같은 공간에서 거리 비교가 의미 있음.

#### Contrastive Loss (Chopra et al., 2005, p.11)
- **Problem**: Learn an encoder where same-class pairs are close and different-class pairs are far.
- **Core idea**: Pair-wise loss: minimize squared distance for positive pairs, push negative pairs apart beyond a margin $\varepsilon$ via $\max(0, \varepsilon - d)^2$.
- **vs. previous / key difference**: Limitations — considers only pairs, many negatives become inactive (already beyond margin → zero gradient), does not explicitly enforce relative ordering.
- **Keywords**: pair, margin ε, hinge, inactive negatives, absolute target
- 한 줄: pair 단위 절대 목표 (positive → 0, negative → ε 이상), 상대 순서는 강제 X.

#### Triplet Loss (FaceNet, Schroff et al., 2015, p.12–13)
- **Problem**: Contrastive loss does not enforce relative similarity.
- **Core idea**: Anchor–Positive–Negative triplets; $\max(0, \|f(x)-f(x^+)\|^2 - \|f(x)-f(x^-)\|^2 + \varepsilon)$ enforces relative similarity / ranking relationships; margin = minimum offset between the two distances.
- **vs. previous / key difference**: Relative ordering instead of absolute targets; but Problem 1 Triplet Explosion ($N^3$) and Problem 2 Most Triplets are Useless (easy triplets dominate) → hard negatives are crucial.
- **Keywords**: anchor, relative similarity, ranking, triplet explosion $N^3$, easy triplets
- 한 줄: 상대 격차만 요구. 대신 조합 폭발 + easy triplet 문제.

#### Hard Negative Mining (p.14)
- **Problem**: Most negatives are already far from the anchor; easy negatives contribute little to the loss and gradients.
- **Core idea**: Hard negatives are close to the anchor but belong to different classes → stronger learning signals; focus training on the most challenging samples.
- **Keywords**: easy negative, hard negative, informative negatives, stronger signal
- 한 줄: 가깝지만 다른 class인 놈이 진짜 공부거리.

#### Lifted Structured Loss (Song et al., p.15–17)
- **Problem**: Contrastive/triplet use only a few pairs per batch.
- **Core idea**: Use all the pairwise edges within one training batch (computational efficiency); hardest negative is chosen by `max`, which is not smooth, so it is relaxed with log-sum-exp.
- **vs. previous / key difference**: Uses the whole batch, but still pair-based, $O(N^2)$, 1 positive ↔ 1 negative → motivates N-pair.
- **Keywords**: all pairwise edges, not smooth, relaxed, log-sum-exp, $O(N^2)$
- 한 줄: batch 전체 edge 활용 + max → LSE로 smooth화.

#### Why Replace Max? — Max vs Sum vs LSE (p.16)
- **Core idea**: Max gives a sparse gradient (only the hardest), Sum/average gives uniform weights (not focused on hard ones), Log-Sum-Exp gives softmax weights → dense & focused gradient = soft hard negative mining; LSE is a smooth approximation of max.
- **Keywords**: sparse, uniform, dense & focused, softmax weight, smooth approximation
- 한 줄: LSE = 모든 negative에 gradient + 어려운 놈일수록 크게.

#### N-pair Loss (Sohn et al., NeurIPS 2016, p.17–20)
- **Problem**: Triplet/lifted compare 1 positive with 1 negative at a time.
- **Core idea**: Multi-class N-pair loss generalizes triplet loss to compare 1 positive against N-1 negatives simultaneously; $\log(1+\sum e^{f^\top f^-_i - f^\top f^+})$ = cross entropy loss for an N-way softmax classifier.
- **vs. previous / key difference**: 1 positive ↔ many negatives → more information, (slide) no need of hard negative mining.
- **Keywords**: (N+1)-tuplet, multiple negatives, N-way softmax, cross entropy
- 한 줄: "N개 중 positive 찾기" 분류 문제로 재정의.

#### NCE — Noise Contrastive Estimation (Gutmann et al., 2010, p.21–22)
- **Problem**: Can negatives be treated as noise and interpreted probabilistically?
- **Core idea**: Positive vs Negative → Data vs Noise; run logistic regression (sigmoid + binary cross-entropy) on the logit $\log p_\theta(u) - \log q(u)$ to tell target data from noise; a method for estimating parameters of a statistical model.
- **Keywords**: data vs noise, logistic regression, sigmoid, binary CE, parameter estimation
- 한 줄: NCE = 샘플별 data/noise 이진 분류.

#### InfoNCE (Oord et al., 2018, CPC, p.23–28)
- **Problem**: Probabilistic version of the N-way positive identification.
- **Core idea**: Uses categorical cross-entropy to identify the positive sample amongst a set of unrelated noise samples; via Bayes rule the optimal $f(x,c) \propto p(x|c)/p(x)$ (density ratio). Learning a good $f$ → good density ratio → increasing mutual information.
- **vs. previous / key difference**: NCE = binary (sigmoid), InfoNCE = N-way categorical (softmax); same form as N-pair with anchor = context $c$.
- **Keywords**: categorical CE, density ratio, Bayes rule, mutual information, context
- 한 줄: InfoNCE는 단순 matching이 아니라 x와 c의 shared information을 늘림.

#### Soft-Nearest Neighbors Loss (Salakhutdinov 2007, Frosst 2019, p.29)
- **Core idea**: Extends the loss to include multiple positive samples (all neighbors of the same class in the numerator, all neighbors in the denominator); temperature tunes how concentrated the features are.
- **Keywords**: multiple positives, same-class neighbors, temperature, concentration
- 한 줄: positive 여러 개 버전 (label 필요).

#### Common Setup for Modern Contrastive Learning (p.31–32)
- **Core idea**: Anchor (query) / Positive = another augmented view of the same sample / Negatives = other samples in the batch / shared encoder (usually L2-normalized) / cosine similarity / InfoNCE (NT-Xent) with temperature $\tau$.
- **Positive construction**: with labels → same class is positive; without labels → augmented views of the same image are positive, other images in the batch are negative. $p_{pos}$ must satisfy Symmetry and Matching marginal.
- **Keywords**: anchor, augmented view, in-batch negatives, cosine, NT-Xent, symmetry, matching marginal
- 한 줄: label 없으면 augmentation이 positive를 정의.

#### GenView (Li et al., ECCV 2024, p.33–45) — *views*
- **Problem**: Handcrafted views have two limitations: (a) False Positive Pair — a strong crop may remove the main object, (b) Limited Diversity — a weak crop may produce two nearly identical views.
- **Core idea**: (1) Adaptive View Generation: CLIP image embedding + adaptive noise (based on the foreground ratio) → frozen Stable unCLIP generates a diverse positive view; (2) Quality-Driven Contrastive Loss: a separate frozen encoder scores each pair by $q_i = s^f_i - s^b_i$ (similar foreground, different background) and $w_i = \text{softmax}(q_i)$ weights the SSL loss.
- **vs. previous / key difference**: Varies pose, viewpoint, background while preserving object identity; plug-in to various SSL losses (MoCo, SimSiam, BYOL, SwAV...). Small foreground → less noise, large foreground → more noise.
- **Keywords**: false positive pair, limited diversity, Stable unCLIP, adaptive noise, foreground ratio, quality-driven loss
- 한 줄: 생성 모델로 더 좋은 positive view 만들고, 품질 좋은 pair에 가중치.

#### Feature Suppression & MCL (Zhang et al., ECCV 2024, p.46–61) — *features*
- **Problem**: Feature suppression = useful information present in the input is lost in the representation (SimCLR ignores shape, 0.935; CLIP ignores orientation, 0.933). It happens because the model learns the easiest discriminative feature (e.g., color) and the contrastive objective is already satisfied.
- **Core idea**: Multistage Contrastive Learning: ① Train → ② Cluster (K-means) → ③ Select negatives within clusters → ④ Repeat → ⑤ Concatenate; each stage removes the shortcut used in previous stages (Stage 0 color → Stage 1 texture → Stage 2 shape).
- **vs. previous / key difference**: Unlike hard negative mining (negatives close to the current embedding, to find difficult samples), MCL selects negatives from the same cluster of the previous stage to prevent the reuse of previously learned features.
- **Keywords**: feature suppression, easiest discriminative feature, K-means, same-cluster negatives, concatenation
- 한 줄: 이전 stage가 쓴 shortcut을 negative 선택으로 막아 억눌린 feature를 학습.

#### X-CLR (Sobal et al., ICLR 2025, p.62–72) — *relationships*
- **Problem**: Standard targets are binary — another dog, a cat, and a piano are treated as equally negative.
- **Core idea**: Replace a binary graph with a soft similarity graph: a fixed text encoder embeds captions/class descriptions (e.g., "a photo of a Maltese dog"), their similarity (softmax with $\tau_s$) is the target graph, and the learned image-similarity graph is matched to it with cross-entropy $H(s_i, p_i)$.
- **vs. previous / key difference**: SSL graph connects only two views of the same image, SupCon connects same-class samples (binary), X-CLR allows different degrees of similarity. Low $\tau_s$ ≈ standard contrastive, high $\tau_s$ → softer graph.
- **Keywords**: soft similarity graph, target graph vs learned graph, fixed text encoder, $\tau_s$, low-data
- 한 줄: one-hot target → text 기반 soft target (InfoNCE의 soft label 버전).

### B. Must-know definitions & distinctions
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


---

<a id="04"></a>

## 04. Contrastive Learning II

### A. Core ideas

#### Key ingredients of contrastive learning (p.4–6)
- **Problem**: What makes contrastive learning work well without labels?
- **Core idea**: Three ingredients — **heavy data augmentation** (noise versions of data used as positives, without modifying semantic meaning), **large batch size** (many diverse in-batch negatives, essential for InfoNCE methods like SimCLR and CLIP), and **hard negative mining** (negatives close to the anchor give more informative supervision).
- **vs. previous / key difference**: Hard negatives are easy to find in supervised learning (class labels) but challenging in self-supervised learning; larger batches or memory banks raise the chance of finding them, at a higher memory cost.
- **Keywords**: noise version, in-batch negatives, diversity, hard negative, memory bank, trade-off
- 한 줄: augmentation은 positive를, 큰 batch는 negative 양·다양성을, hard negative는 negative의 질을 책임짐.

#### Sampling bias & Debiased Contrastive Loss (Chuang et al., 2020, p.7–11)
- **Problem**: Random negative sampling may include **false negatives** (same-class samples), which leads to a big performance drop.
- **Core idea**: Estimate the **true negative distribution** without labels: $p_x^- = \frac{p - \eta^+ p_x^+}{\eta^-}$, i.e., *True Negative = (Random Sampling − Positive Contribution) / $\eta^-$*, and plug this estimate into the InfoNCE denominator.
- **vs. previous / key difference**: Standard InfoNCE treats every random sample as a negative (biased); debiased loss corrects it using only random samples and positives.
- **Keywords**: false negative, sampling bias, true negative distribution, $\eta^+$ (class prior), positive contribution, Biased vs Unbiased
- 한 줄: random 분포에서 positive 성분을 빼서 진짜 negative 분포를 추정 → label 없이 false negative 보정.

#### Image augmentation for contrastive learning (p.13–14)
- **Problem**: How should positive views be built?
- **Core idea**: Create different views of the same image that **significantly change visual appearance** while **preserving semantic meaning**, forming positive pairs.
- **Keywords**: views, visual appearance, semantic meaning, random crop, color jitter, multi-crop (SwAV)
- 한 줄: 겉모습은 크게 바꾸고 의미는 유지 = 좋은 positive.

#### AutoAugment (Cubuk et al., CVPR 2019, p.15)
- **Problem**: Hand-designing augmentation policies is hard.
- **Core idea**: Formulate augmentation search as a **reinforcement learning** problem (RNN controller) that maximizes **validation accuracy**. Key Idea: *Learn which augmentations to apply.*
- **Keywords**: reinforcement learning, controller, child network, validation accuracy, fixed policy
- 한 줄: RL로 best augmentation policy 탐색 (비싸다).

#### RandAugment (Cubuk et al., 2019, p.16)
- **Problem**: AutoAugment's search is expensive (huge search space: which operation, what probability, what magnitude).
- **Core idea**: **Randomly** select augmentations and control everything with only **N** (number of operations) and **M** (one global magnitude).
- **vs. previous / key difference**: Eliminates the expensive search → very small search space; simpler and more efficient than AutoAugment.
- **Keywords**: random selection, N, M, single global magnitude, small search space
- 한 줄: 탐색 대신 랜덤 + hyperparameter 2개 (N, M).

#### PBA: Population Based Augmentation (Ho et al., ICML 2019, p.17)
- **Problem**: A fixed augmentation policy may not be best throughout training.
- **Core idea**: Evolve augmentation policies **during training** with **population-based training (PBT)**. Key Idea: *Learn augmentations online via evolutionary optimization.*
- **vs. previous / key difference**: Unlike AutoAugment, PBA learns a **schedule** instead of a **fixed policy**.
- **Keywords**: PBT, evolutionary, online, schedule, continuously updates
- 한 줄: PBT로 학습 중 augmentation schedule을 진화시킴.

#### UDA: Unsupervised Data Augmentation (Xie et al., NeurIPS 2020, p.18)
- **Problem**: How to learn from **unlabeled** data using augmentation?
- **Core idea**: Apply augmentations to unlabeled data and enforce **prediction consistency** between original and augmented inputs by minimizing **KL divergence**. Key Idea: *consistency regularization.*
- **Keywords**: unlabeled data, consistency regularization, KL divergence, RandAugment, back translation
- 한 줄: unlabeled에 강한 aug → 원본과 예측이 같도록 KL.

#### Mixup (Zhang et al., ICLR 2018, p.19)
- **Core idea**: **Global-level** mixture — a weighted **pixel-wise** combination of two images (and their labels): $\tilde x = \lambda x_i + (1-\lambda)x_j$.
- **Keywords**: global-level, pixel-wise, weighted combination, $\lambda$
- 한 줄: 두 이미지를 통째로 반투명 섞기.

#### CutMix (Yun et al., ICCV 2019, p.20)
- **Core idea**: **Region-level** mixture — paste a local region of one image onto the rest of another; labels mixed by area ratio.
- **vs. previous / key difference**: Unlike Mixup (global) and Cutout (removes region), CutMix fills the region with real pixels → improves classification, localization and detection.
- **Keywords**: region-level, binary mask $M_b$, area ratio label, localizable features
- 한 줄: 패치 잘라 붙이기, label은 면적비.

#### MoCHi: Mixing of Contrastive Hard Negatives (Kalantidis et al., NeurIPS 2020, p.21)
- **Problem**: Most negatives are easy and give little signal.
- **Core idea**: Identify the hardest negatives, then synthesize harder negatives by **interpolating hard-negative embeddings** and **mixing the query with hard negatives**. Key Idea: *mix embeddings in the feature space.*
- **vs. previous / key difference**: Mixup/CutMix mix in pixel space; MoCHi mixes in **feature (embedding) space** to create negatives.
- **Keywords**: hardest negatives, synthetic hard negatives, feature space, query mixing
- 한 줄: feature 공간에서 hard negative를 합성.

#### SimCLR (Chen et al., ICML 2020, p.23–29)
- **Problem**: A simple framework for contrastive learning of visual representations.
- **Core idea**: Maximize agreement between two differently augmented views of the same sample with a contrastive (InfoNCE / NT-Xent) loss on a **projection head** output $z = g(h)$, using all other $2(N-1)$ in-batch samples as negatives.
- **vs. previous / key difference**: No memory bank — in-batch negatives only; hence needs a **large batch size** (TPU training). After training, **throw away $g$** and use $h$.
- **Keywords**: 2N augmented samples, cosine similarity, temperature $\tau$, projection head, large batch, linear evaluation
- 한 줄: 같은 이미지 두 view는 당기고 batch 내 나머지는 밀기, 큰 batch 필수.

#### BYOL: Bootstrap Your Own Latent (Grill et al., NeurIPS 2020, p.30–34)
- **Problem**: SimCLR depends on many negatives (large batch).
- **Core idea**: **Online** network ($\theta$, with a **predictor**) predicts the representation of another view produced by a **target** network ($\xi$), which is updated by **EMA** with **no backpropagation** (stop-gradient); loss = **MSE** between normalized vectors.
- **vs. previous / key difference**: **No negative samples** — learning by prediction instead of contrast.
- **Keywords**: online/target, predictor, EMA, stop-gradient, MSE, bootstrap
- 한 줄: negative 없이 EMA target의 출력을 predictor로 예측.

#### Barlow Twins (Zbontar et al., ICML 2021, p.35–38)
- **Problem**: Avoid collapse without negatives (and without predictor/EMA tricks).
- **Core idea**: Compute the **cross-correlation matrix** between the two views' embeddings **along the batch dimension** ($D\times D$) and push it to the **identity**: diagonal → 1 (**invariance**), off-diagonal → 0 (**redundancy reduction**).
- **vs. previous / key difference**: Uses a **shared encoder**, no negatives; naturally avoids collapsed (trivial constant) representations and is **robust to batch size**.
- **Keywords**: cross-correlation, identity matrix, invariance term, redundancy reduction, decorrelated, batch dimension
- 한 줄: feature 차원 간 상관행렬을 identity로 → 불변 + 비중복.

#### DINO: Self-Distillation with No Labels (Caron et al., ICCV 2021, p.39–44)
- **Problem**: Learn semantic representations without labels via distillation.
- **Core idea**: **Student** and **teacher** with identical architectures; the student is trained to match the teacher's **output (softmax) distribution** with **cross-entropy**; the teacher is an **EMA of the student** (centering + sharpening on teacher outputs).
- **vs. previous / key difference**: BYOL = feature matching with MSE + predictor; DINO = **distribution matching** with **cross-entropy** on softmax probabilities.
- **Keywords**: self-distillation, student/teacher, EMA teacher, cross-entropy, centering, emergent self-attention (object segmentation)
- 한 줄: 학생이 EMA 선생의 확률분포를 CE로 따라감 → ViT attention이 객체를 찾음.

#### SynCLR (Tian et al., CVPR 2024, "Learning Vision from Models Rivals Learning Vision from Data", p.45–54)
- **Problem**: Can positive views be **generated** rather than augmented?
- **Core idea**: Llama-2 generates captions; a text-to-image model generates **4 images per caption** from different random noise (low CFG for diversity) → images from the same caption form a **positive set**, trained with a **multi-positive contrastive loss**.
- **vs. previous / key difference**: SimCLR preserves **image identity** (positive pair from augmentation); SynCLR preserves **caption-level semantics** (positive set from generation) — positives need not come from the same image.
- **Keywords**: synthetic captions, Llama-2, text-to-image, positive set, caption granularity, multi-positive, no real images
- 한 줄: 같은 caption으로 생성한 이미지들 = positive, real image 없이 학습.

#### DINOv2 (Oquab et al., TMLR 2024, p.55–67)
- **Problem**: How can DINO become a **general-purpose visual foundation model**?
- **Core idea**: **DINOv2 = DINO + iBOT + KoLeo + Curated Data + Scaling** — image-level (CLS) + patch-level (masked iBOT) self-distillation, KoLeo + Sinkhorn-Knopp to prevent crowding/collapse, curated LVD-142M data, and distilling a large ViT-g teacher into smaller models.
- **vs. previous / key difference**: Not a single new loss but an **integrated system**; learns from **images alone (no labels, no captions)**, giving frozen features for both global and dense tasks.
- **Keywords**: iBOT, KoLeo, Sinkhorn-Knopp, LVD-142M, deduplication, retrieval, ViT-g distillation, frozen features
- 한 줄: DINO를 데이터·patch loss·안정화·distill로 키운 범용 backbone.

#### DINOv3: Gram Anchoring (p.68–70)
- **Problem**: Longer training improves global features but **dense features gradually degrade** (noisy patch similarity, segmentation drops).
- **Core idea**: Select an **earlier model with clean patch features** and preserve its **pairwise patch-similarity structure (Gram matrix)** during longer training: $\|X_SX_S^\top - X_GX_G^\top\|_F^2$.
- **vs. previous / key difference**: Matches the similarity structure among patches, not the features themselves.
- **Keywords**: Gram matrix, earlier model, patch similarity, dense feature degradation, spatial consistency
- 한 줄: 깨끗했던 옛 체크포인트의 patch 유사도 구조를 붙잡아 dense 열화 방지.

### B. Must-know definitions & distinctions

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


---

<a id="05"></a>

## 05. Contrastive Learning III

### A. Core ideas

#### Memory Bank (concept, p.4)
- **Problem**: Large numbers of negatives improve contrastive learning, but computing embeddings for all negatives every step is too expensive, and mini-batch negatives are limited by batch size.
- **Core idea**: Store previously computed embeddings in memory and reuse them as negatives — "Trade stale representations for efficient large-scale contrastive learning."
- **vs. previous / key difference**: Mini-batch negatives are *limited but fresh*; memory bank negatives are *large but slightly stale*.
- **Keywords**: store & reuse, stale, fresh, large number of negatives, decouple from batch size
- 한 줄: 과거 embedding을 저장해 negative로 재사용 → 수는 많아지지만 stale해짐.

#### Instance Discrimination with Memory Bank (Wu et al., CVPR 2018, p.5–9)
- **Problem**: Learn features without labels when treating every image as its own class makes an N-way softmax infeasible (too many heads).
- **Core idea**: Each image is a unique class (positive = same image, negatives = all other images); a non-parametric softmax uses the stored memory-bank features as class weights, and NCE with a Monte Carlo subset of M negatives approximates the full softmax.
- **vs. previous / key difference**: Replaces the learned classifier weights $\mathbf w_i$ with the stored features $\mathbf v_i$ (non-parametric), and adds a proximal regularization term that gradually vanishes as embeddings converge.
- **Keywords**: instance discrimination, non-parametric softmax, NCE, Monte Carlo approximation, proximal regularization, stale embeddings
- 한 줄: 이미지 하나 = class 하나, bank feature를 classifier weight로 쓰고 NCE로 근사. 단점은 stale feature.

#### MoCo — Momentum Contrast (He et al., CVPR 2020, p.10–15)
- **Problem**: "Can we keep a large dictionary while reducing feature staleness?" (p.9)
- **Core idea**: Replace the memory bank with a FIFO queue of keys encoded by a momentum encoder, $\theta_k \leftarrow m\theta_k + (1-m)\theta_q$, giving a large and consistent dictionary trained with InfoNCE.
- **vs. previous / key difference**: Unlike SimCLR, gradients flow only through the query encoder (no gradient through the key), and the number of keys is decoupled from the mini-batch size; MoCo is a hybrid of end-to-end and memory bank.
- **Keywords**: FIFO queue, momentum encoder, dictionary look-up, consistent keys, no gradient through key, InfoNCE
- 한 줄: queue + momentum(EMA) key encoder → 크고(large) 일관된(consistent) dictionary.

#### MoCo v2 (Chen et al., 2020, p.16–19)
- **Problem**: MoCo lagged behind SimCLR, which used better training tricks but needed huge batches.
- **Core idea**: Combine MoCo with SimCLR's effective training strategies — an MLP projection head and stronger data augmentation — while retaining the momentum encoder and queue.
- **vs. previous / key difference**: Not a new architecture; because negatives are decoupled from batch size, MoCo v2 outperforms SimCLR with a much smaller batch (256 vs. 8192).
- **Keywords**: MLP projection head, stronger augmentation, improved baseline, batch 256 vs 8192, less memory
- 한 줄: MoCo + SimCLR 레시피(MLP head, strong aug). 작은 batch로 SimCLR 이김.

#### CURL (Srinivas et al., 2020, p.20–21)
- **Problem**: Reinforcement learning from raw pixel observations is sample-inefficient.
- **Core idea**: Apply MoCo-style contrastive learning (random-crop positive pairs, query encoder + momentum key encoder) to observations from the replay buffer, while simultaneously optimizing the RL policy/value network on the learned representation.
- **vs. previous / key difference**: Contrastive learning is used as a jointly trained auxiliary objective for RL rather than for image classification pre-training.
- **Keywords**: reinforcement learning, replay buffer, random crop, momentum encoder, sample efficiency, simultaneous optimization
- 한 줄: MoCo를 RL에 붙여서 pixel RL의 sample efficiency ↑. RL loss와 동시에 학습.

#### DeepCluster (Caron et al., ECCV 2018, p.23–25)
- **Problem**: Instance-level contrastive learning needs positive/negative pairs and pushes away semantically similar images.
- **Core idea**: Cluster the current features (k-means), use the cluster assignments as pseudo labels, train the network to predict them, and repeat — "Alternate between clustering and classification."
- **vs. previous / key difference**: It does not create positive/negative pairs; it establishes pseudo-label learning as an alternative to contrastive learning.
- **Keywords**: k-means, pseudo labels, alternate clustering & classification, offline clustering, no pairs
- 한 줄: k-means cluster ID를 pseudo label로 분류 학습, clustering ↔ classification 반복.

#### SwAV (Caron et al., NeurIPS 2020, p.26–30)
- **Problem**: Contrastive learning needs explicit pairwise comparisons (many negatives), and DeepCluster needs offline clustering over the whole dataset.
- **Core idea**: Compute online cluster assignments (codes) for augmented views using K trainable prototypes, and predict the code of one view from another view (swapped prediction).
- **vs. previous / key difference**: Feature ↔ Prototype comparison instead of Feature ↔ Feature; no explicit negative pairs and no offline clustering.
- **Keywords**: swapped prediction, codes, prototypes, online clustering, no explicit negatives, multiple views
- 한 줄: 한 view의 code를 다른 view로 예측 (swap). online clustering + prototype.

#### CLIP (Radford et al., PMLR 2021, p.32–37)
- **Problem**: Fixed-label supervision limits transfer; can natural language serve as supervision?
- **Core idea**: Jointly train image and text encoders to align matched image–caption pairs in a shared space with a symmetric contrastive loss over an $N\times N$ similarity matrix; classify zero-shot by matching an image to text prompts.
- **vs. previous / key difference**: Positives are (image, caption) pairs instead of augmentations, and inference uses image–text matching instead of a learned classifier.
- **Keywords**: image–text pairs, shared embedding space, symmetric contrastive loss, zero-shot, text prompts, 400M pairs
- 한 줄: image–text를 contrastive로 정렬 → prompt로 zero-shot 분류. caption 예측보다 ~4× data-efficient.

#### SupCon — Supervised Contrastive Learning (Khosla et al., NeurIPS 2020, p.38–42)
- **Problem**: Self-supervised contrastive learning treats same-class images as negatives, and cross-entropy gives less discriminative representations.
- **Core idea**: Use class labels — all samples of the same class are positives (pulled together), different classes are negatives (pushed apart); compute SimCLR's InfoNCE for each positive and average.
- **vs. previous / key difference**: Beats cross-entropy only by a small amount, but is more robust (ImageNet-C) and less sensitive to hyperparameters.
- **Keywords**: same-class positives, InfoNCE per positive and average, vs cross-entropy, ImageNet-C robustness, hyperparameter insensitivity
- 한 줄: 같은 class 전부 positive. CE보다 조금 낫고, robust하고 hyperparameter에 덜 민감.

#### NNCLR → All4One (Estepa et al., ICCV 2023, p.43–53)
- **Problem**: "Can the memory bank provide more than negatives?" — and a single nearest neighbor can be a noisy positive.
- **Core idea**: NNCLR uses the nearest neighbor from a support set as the positive; All4One retrieves K neighbors, summarizes them with a small Transformer (self-attention) into a centroid, and combines three losses: Neighbor Contrast, Centroid Contrast, and Feature Contrast (Barlow Twins redundancy reduction).
- **vs. previous / key difference**: "NNCLR uses one neighbor. All4One summarizes multiple neighbors into one representation." The centroid is produced by self-attention, not an average or cluster center.
- **Keywords**: support set, nearest neighbor positive, centroid via self-attention, redundancy reduction, Barlow Twins, three losses
- 한 줄: memory bank에서 positive를 꺼냄. K-NN을 self-attention으로 centroid화 + Barlow Twins loss.

#### PAPN (Tan et al., ICCV 2025, p.54–64)
- **Problem**: "Can cluster-based SSL preserve fine-grained visual details?" — multi-stage alignment helps on generic data but hurts on fine-grained data because strong augmentation changes shallow features much more.
- **Core idea**: (1) Stage-wise Progressive Augmentation (SPA): mild augmentation for shallow stages, strong for deep stages; (2) Prototype-based Holistic-Partial Learning (PHP): learnable prototypes over spatial descriptors extract part features, which are concatenated with whole features into InfoNCE.
- **vs. previous / key difference**: One augmentation strength does not fit every stage; and instead of choosing between global and local features, contrast both together.
- **Keywords**: fine-grained, stage-wise progressive augmentation, shallow vs deep stages, learnable prototypes, part representation, whole + part
- 한 줄: stage마다 aug 강도 다르게(SPA) + prototype으로 part feature 뽑아 whole과 함께 contrast(PHP).

#### SCL-RHE (Long et al., ECCV 2024, p.65–75)
- **Problem**: In SupCon, imprecise (human) labels create wrong pair relationships, and human errors look like easy, high-similarity positives.
- **Core idea**: Since almost all incorrect signals are false positives, deprioritize easy positives — high similarity → lower sampling weight, $q^+(x^+) \propto e^{-\beta f(x)^\top f(x^+)} p_x^+(x^+)$ — emphasizing hard positives.
- **vs. previous / key difference**: Methods for heavy synthetic noise (Sel-CL, TCL) remove too many samples and can underperform SupCon; human errors overlap with true positives, unlike synthetic errors.
- **Keywords**: human labeling errors, false positives dominate, assigned vs latent label, easy vs hard positives, lower weight for high similarity
- 한 줄: human label error는 easy false positive가 됨 → similarity 높은 positive 가중 ↓, hard positive 강조.

### B. Must-know definitions & distinctions
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


---

<a id="06"></a>

## 06. Semi-Supervised Learning I

### A. Core ideas

#### Four ways to overcome limited labeled data (p.4–7)
- **Problem**: Labeled data are scarce and expensive, so we need ways to exploit unlabeled data or limited labeling budgets.
- **Core idea**: (1) Pre-training + fine-tuning: pre-train a task-agnostic model on a large unsupervised corpus, then fine-tune with a small labeled set; (2) Active learning: select the most valuable unlabeled samples to be labeled next under a limited budget; (3) Pre-training + dataset auto-generation: use a capable pre-trained model to auto-generate many more labeled samples (popular in language); (4) Semi-supervised learning: learn from labeled and unlabeled samples together.
- **vs. previous / key difference**: SSL trains on both data types **jointly**, whereas pre-train/fine-tune is **sequential (2-stage)** and active learning still asks humans for labels.
- **Keywords**: task-agnostic, limited budget, auto-generate, few-shot learning, together
- 한 줄: label 부족 해법 4개 — SSL만 labeled+unlabeled를 "한 번에 같이" 학습.

#### SSL objective & key question (p.8–10)
- **Problem**: How to combine scarce labeled data with plentiful unlabeled data in one training objective.
- **Core idea**: $\mathcal L = \mathcal L_s + \mu(t)\,\mathcal L_u$ — supervised loss + unsupervised loss + weighting term $\mu(t)$, which is **increasing in time**. The key question is how to design $\mathcal L_u$: **Consistency Regularization** (this lecture) or **Pseudo Label** (next lecture).
- **Keywords**: supervised loss, unsupervised loss, weighting term, increasing in time, consistency regularization, pseudo label
- 한 줄: $\mathcal L_u$를 어떻게 만드냐가 SSL의 핵심 질문, $\mu(t)$는 점점 키움 (초반 예측은 믿을 수 없으니).

#### Hypotheses H1–H4 (p.11–14)
- **Problem**: Why should unlabeled data help at all?
- **Core idea**: H1 Smoothness — nearby samples should have similar predictions / small perturbations should not change the label. H2 Cluster — samples in the same cluster share the same label; boundaries should not split a cluster. H3 Low-density separation — decision boundaries should lie in **low-density** regions. H4 Manifold — high-dimensional data lie on a **low-dimensional manifold**.
- **Keywords**: smoothness, cluster, low-density separation, manifold, decision boundary
- 한 줄: unlabeled data가 $p(x)$의 구조(cluster·density·manifold)를 알려줘서 도움 됨. H1이 consistency의 근거.

#### Consistency Regularization — core idea (p.15–16)
- **Problem**: Unlabeled data have no labels, so we need a loss that doesn't use $y$.
- **Core idea**: Predictions should remain consistent under small perturbations; $\mathcal L_u = D(f(x), f(\mathcal A(x)))$. Perturbations come from **data augmentations** or **model stochasticity (e.g., Dropout)**.
- **Keywords**: perturbation, data augmentation, dropout, model stochasticity, consistent predictions
- 한 줄: "perturbation 줘도 예측은 같아야 한다" = H1을 loss로 만든 것.

#### Π-model (Sajjadi et al., ICLR 2016 as on slide, p.17–18)
- **Problem**: Need a simple consistency target for unlabeled data.
- **Core idea**: Pass two versions of the same input with **different stochastic augmentation and dropout masks** through the **same network** and make the outputs consistent (MSE).
- **Keywords**: two forward passes, dropout masks, stochastic augmentation, MSE, same network
- 한 줄: 같은 모델에 두 번 forward (다른 noise) → 출력 MSE. 단점은 forward 2회.

#### Temporal Ensembling (Laine et al., ICLR 2017, p.19–22)
- **Problem**: Π-model requires two forward passes per input → high computational cost.
- **Core idea**: Use the **historical predictions of the same sample** as the target — an EMA of the model **prediction** in time **per training sample** (with bias correction).
- **vs. previous / key difference**: One forward pass and a more stable target than Π-model, but must store predictions for every sample, targets are updated only once per epoch, and it is hard to scale.
- **Keywords**: EMA of predictions, per training sample, one forward pass, once per epoch, memory
- 한 줄: sample별 과거 예측의 EMA를 target으로 → 싸고 안정적이지만 메모리·epoch 단위 갱신이 약점.

#### Mean Teacher (Tarvainen et al., p.23–29)
- **Problem**: Temporal Ensembling is memory-intensive and its targets are updated only once per epoch.
- **Core idea**: Replace the EMA of predictions with an **EMA of model weights**: teacher $\theta' \leftarrow \beta\theta' + (1-\beta)\theta$. The student is trained by gradient descent (supervised + consistency); the teacher is updated **only by EMA, no backpropagation**.
- **vs. previous / key difference**: No per-sample prediction memory, teacher updated every iteration, scales to large datasets.
- **Keywords**: EMA of weights, student, teacher, no backpropagation, $\beta$ 0.99 → 0.999, MSE > KL
- 한 줄: teacher = student weight의 EMA. 실전 팁: input aug 필수, dropout은 student만, β는 초반 0.99 → 후반 0.999, MSE가 KL보다 좋음.

#### Noisy Samples as Learning Targets (p.30–31)
- **Problem**: Weak internal noise gives little learning signal on unlabeled data.
- **Core idea**: Apply **strong augmentations** to an unlabeled sample; the prediction should stay the same before and after augmentation, using **the same model** (prediction invariance).
- **vs. previous / key difference**: vs Π-model — input augmentations instead of stochastic perturbations inside the network; vs Mean Teacher — self-consistency between original and augmented inputs instead of an EMA teacher.
- **Keywords**: strong augmentation, RandAugment, back-translation, prediction invariance, self-consistency
- 한 줄: 원본 vs strong aug 버전의 예측을 같은 모델로 맞춤.

#### Adversarial Training (Goodfellow et al., ICLR 2015, p.32–33)
- **Problem**: Small, imperceptible perturbations can make a model confidently wrong (panda → gibbon).
- **Core idea**: Add **worst-case** (adversarial) perturbations to the input during training and train the model to still produce the **correct (ground-truth)** prediction; $r_{adv} \approx \epsilon\,\mathrm{sign}(g)$ (FGSM).
- **Keywords**: worst-case perturbation, ground-truth label, robustness, FGSM, $\|r\|\le\epsilon$
- 한 줄: 모델이 가장 헷갈리는 방향의 noise를 넣고도 GT를 맞히게 → robustness (supervised).

#### Virtual Adversarial Training — VAT (Miyato et al., TPAMI 2018, p.34–35)
- **Problem**: Adversarial training needs ground-truth labels, so it can't use unlabeled data.
- **Core idea**: Replace the GT label with the model's **own current prediction**, enabling adversarial consistency training on **both labeled and unlabeled** samples.
- **vs. previous / key difference**: AT = GT labels, supervised, robustness; VAT = model predictions, semi-supervised, **local smoothness**. vs RPT: RPT smooths in random directions, VAT only along the most adversarial direction (much more effective).
- **Keywords**: virtual, model prediction, local smoothness, most adversarial direction, RPT
- 한 줄: "virtual" = 진짜 label 대신 모델 예측. 가장 민감한 방향에서도 예측이 안 바뀌게.

#### Interpolation Consistency Training — ICT (Verma et al., p.36–38)
- **Problem**: Want consistency exactly where it matters — near the decision boundary.
- **Core idea**: Following **MixUp**, linearly interpolate two **unlabeled** samples; the student's prediction on the mixed input should match the **mix of the Mean Teacher's predictions** on the originals ("mixing inputs should produce mixed predictions").
- **vs. previous / key difference**: MixUp samples tend to lie near the decision boundary → smoother boundaries → aligns with the **Low-Density Separation** assumption.
- **Keywords**: MixUp, interpolation, Mean Teacher, $\lambda\sim\mathrm{Beta}(\alpha,\alpha)$, low-density separation
- 한 줄: 섞은 입력 → 섞은 예측. teacher는 섞기 전, student는 섞은 후.

#### Unsupervised Data Augmentation — UDA (Xie et al., NeurIPS 2020, p.39–42)
- **Problem**: Small adversarial/random perturbations regularize only a narrow neighborhood.
- **Core idea**: Apply **strong, label-preserving semantic augmentations (RandAugment)** to unlabeled samples and enforce $p(y|x)\approx p(y|\mathrm{Aug}(x))$.
- **vs. previous / key difference**: Simply replacing adversarial perturbations with strong semantic augmentations beats VAT and ICT — **augmentation quality** is the key factor.
- **Keywords**: RandAugment, label-preserving, diverse, realistic, task-relevant, augmentation quality
- 한 줄: "How we perturb the input matters more than simply adding perturbations."

#### PS-MT — Perturbed and Strict Mean Teachers (Liu et al., CVPR 2022, p.43–54)
- **Problem**: Strong perturbations help generalization, but if the teacher is wrong the student reproduces the same error → need more reliable teacher targets first.
- **Core idea**: **Two mean teachers** (only one updated per epoch) whose **ensemble** gives the target; a **strongly perturbed student** (input: strong aug/CutMix; feature: **T-VAT**; network: teachers vs student); and a **strict consistency loss (Conf-CE)** instead of MSE.
- **vs. previous / key difference**: MSE is robust but gives small gradients near convergence (weak supervision); more accurate teachers allow stronger confidence-weighted CE. T-VAT: the **teachers** find the adversarial direction, not the student. CutMix is applied **after** prediction so teachers see natural images.
- **Keywords**: two teachers, teacher ensemble, Conf-CE, T-VAT, CutMix after prediction, reliable targets
- 한 줄: teacher를 먼저 믿을 만하게(2-teacher ensemble) → 그래야 강한 perturbation + 강한 CE loss를 쓸 수 있음.

#### Dual Teacher — Switching Temporary Teachers (Na et al., NeurIPS 2023, p.55–65)
- **Problem**: With EMA, the teacher gradually follows the student and becomes too similar (**coupling**) → little new information and inherits the student's errors.
- **Core idea**: Two **temporary** EMA teachers **take turns** (switch every epoch); only the **active** teacher generates targets and is updated, and the student's strong augmentation (ClassMix/CutMix) also changes.
- **vs. previous / key difference**: Diversity without training a second student (vs Dual Student); **switching** beats **ensembling** because ensembled teachers gradually lose their distinct characteristics. The EMA rule stays the same; only the teacher schedule changes.
- **Keywords**: coupling, temporary teachers, active teacher, switching vs ensemble, diversity
- 한 줄: "A teacher should be stable, but it should not become identical to the student."

#### S⁴Former (Hu et al., CVPR 2024, p.66–74)
- **Problem**: Image-level perturbations alone may not be enough for a Vision Transformer, which processes patch tokens with self-attention.
- **Core idea**: Regularize the ViT at three levels — **Input: PatchShuffle** (shuffle $M\times M$ blocks before encoding, restore before decoding), **Feature: PASA** (attention modulation on low-confidence patches, guided by teacher confidence), **Output: NCR** (match the distribution over non-pseudo-label classes).
- **vs. previous / key difference**: Keeps the Mean Teacher framework (EMA teacher, weak-to-strong consistency) but adds model-structure-aware perturbations; PatchShuffle gives the largest single gain.
- **Keywords**: PatchShuffle, PASA, Negative Class Ranking, ViT, patch tokens, self-attention
- 한 줄: "Effective consistency regularization should reflect the structure of the model."

### B. Must-know definitions & distinctions
- **Semi-supervised learning**: learn from the labeled and unlabeled samples together.
- **Active learning**: select the most valuable unlabeled samples to be labeled next (limited budget).
- **$\mu(t)$**: unsupervised loss weight, **increasing** in time.
- **Consistency regularization vs Pseudo label**: make two perturbed predictions consistent vs turn a confident prediction ($\max p>\tau$) into a hard label for CE.
- **H1 smoothness / H2 cluster / H3 low-density separation / H4 manifold** — boundaries in **low**-density regions; data on a **low**-dimensional manifold.
- **Π-model vs Temporal Ensembling vs Mean Teacher**: same model twice (2 forward passes) vs EMA of **predictions** per sample (once per epoch, $N\times C$ memory) vs EMA of **weights** (every iteration, scalable).
- **Mean Teacher**: teacher updated only by EMA, **no backprop**; dropout only for the **student**; $\beta$: 0.99 early → 0.999 later; **MSE > KL**.
- **Noisy samples vs Π-model**: input augmentation vs stochastic perturbation inside the network.
- **Adversarial Training vs VAT**: GT labels / supervised / robustness vs model predictions / semi-supervised / local smoothness.
- **RPT vs VAT**: random directions vs most adversarial direction.
- **ICT**: mixed inputs → mixed (teacher) predictions; aligns with low-density separation.
- **UDA**: strong semantic augmentation > adversarial perturbation; good aug = label-preserving, diverse, realistic, task-relevant.
- **PS-MT**: only one of two teachers updated per epoch; teacher ensemble; MSE → Conf-CE (stricter); T-VAT = teachers find perturbation; CutMix **after** prediction.
- **Teacher ensemble (PS-MT) vs teacher switching (Dual Teacher)**: both teachers guide simultaneously vs one active teacher per epoch.
- **Single Teacher vs Dual Student vs Dual Teacher**: 1 permanent EMA teacher (coupling) vs 2 trained students (higher cost) vs 2 temporary teachers, one at a time.
- **S⁴Former**: Input–PatchShuffle, Feature–PASA, Output–NCR.


---

<a id="07"></a>

## 07. Semi-Supervised Learning II (Pseudo Labeling)

### A. Core ideas

#### Pseudo-Label & Entropy Minimization (Lee, ICML 2013; "Bengio et al., NeurIPS 2004", p.4–9)
- **Problem**: How can unlabeled data provide a training target when no label exists?
- **Core idea**: Predict labels for unlabeled samples with the current model, treat **high-confidence** predictions as (hard) pseudo labels, and train on labeled + pseudo-labeled data. Pseudo labeling is the **hard-label approximation of Entropy Regularization**.
- **vs. previous / key difference**: Entropy minimization is a soft penalty on the whole prediction; pseudo labeling converts a confident soft prediction into a hard label ($y^*=\arg\max p(y|x)$). Minimizing entropy moves decision boundaries toward **low-density regions**.
- **Keywords**: high-confidence predictions, hard label, entropy regularization, low-density separation, better-separated clusters, segregation (t-SNE)
- 한 줄: confident 예측을 one-hot target으로 → entropy↓ → boundary가 low-density로 이동.

#### Label Propagation (Iscen et al., CVPR 2019, p.10–14)
- **Problem**: Plain pseudo labeling predicts each sample independently and ignores the data manifold.
- **Core idea**: Build a (kNN) **similarity graph in the feature space** and **propagate** labels from labeled samples to nearby unlabeled samples; pseudo labels come from neighborhood relationships (diffusion), not from the network's softmax.
- **vs. previous / key difference**: Pseudo labeling = independent, **confidence-based**; Label Propagation = neighborhood, **similarity-based**, leverages manifold structure (cluster assumption).
- **Keywords**: similarity graph, feature space, neighborhood relationships, smoother pseudo labels, cluster assumption, expensive for large datasets, depends on feature quality
- 한 줄: feature-space graph 위에서 label을 퍼뜨림. 대신 graph 비용 크고 feature 질에 의존.

#### Self-Training & Noisy Student (Xie et al., CVPR 2020, p.15–19)
- **Problem**: How to iterate pseudo labeling so the model keeps improving?
- **Core idea**: Iterative **teacher–student** process: teacher predicts pseudo labels, student learns from labeled + pseudo-labeled data and becomes the next teacher. Noisy Student trains an **equal-or-larger student with added noise** (RandAugment, Dropout, Stochastic Depth) so it does not simply copy the teacher.
- **vs. previous / key difference**: Plain self-training just adds high-confidence predictions to the labeled set; Noisy Student makes the **student** learn under **stronger perturbations than the teacher** → student becomes more robust than the teacher.
- **Keywords**: teacher/student, iterative, RandAugment, Dropout, Stochastic Depth, larger student > teacher, soft pseudo labels > hard labels
- 한 줄: noise는 student에만. student ≥ teacher 크기, 반복하며 점점 좋아짐.

#### Reducing Confirmation Bias (Arazo et al., IJCNN 2020, p.20–23)
- **Problem**: **Confirmation bias** — an imperfect teacher produces incorrect pseudo labels, the student imitates them, errors accumulate and performance degrades.
- **Core idea**: Regularize so wrong pseudo labels are hard to memorize: **MixUp with soft labels**, **oversampling labeled data** (minimum number of labeled samples per mini-batch), **strong data augmentation**, **dropout**.
- **vs. previous / key difference**: Shows pseudo labeling alone can be competitive when confirmation bias is handled (two moons: mixup + minimum labeled per mini-batch both needed).
- **Keywords**: confirmation bias, imperfect teacher, MixUp, soft labels, oversampling labeled data, dropout
- 한 줄: regularization(MixUp 등)으로 틀린 pseudo label의 자기강화를 억제.

#### Meta Pseudo Labels (Pham et al., CVPR 2021, p.24–27)
- **Problem**: Conventional self-training assumes the teacher generates good pseudo labels, but an imperfect teacher gives incorrect supervision.
- **Core idea**: Instead of fixing the teacher, **learn the teacher**: bi-level optimization — (inner) student is trained on teacher's pseudo labels, (outer) teacher is updated so the resulting student achieves **lower supervised loss on labeled data**.
- **vs. previous / key difference**: Arazo reduces bias by regularizing the student; Meta PL uses **feedback from the student's performance** (evaluated with correct labels) to train the teacher.
- **Keywords**: learn the teacher, bi-level optimization, inner/outer optimization, student feedback, labeled data evaluation, "importance of feedback"
- 한 줄: teacher를 student의 labeled 성능 feedback으로 학습.

#### Pseudo Labeling + Consistency Regularization (p.28–29)
- **Problem**: Pseudo labeling and consistency training each capture only part of the SSL assumptions.
- **Core idea**: Combine both — make a target from one view and train another view to match it.
- **Keywords**: hybrid, pseudo labeling + consistency training
- 한 줄: 한 view로 label, 다른 view로 학습 → 둘을 한 loss로.

#### MixMatch (Berthelot et al., NeurIPS 2019, p.30–35)
- **Problem**: Many successful SSL ideas exist separately.
- **Core idea**: **Unifies** consistency regularization, entropy minimization, pseudo labeling, label sharpening, and MixUp into one **holistic** framework: guess labels by **averaging predictions over K augmentations**, then **sharpen** with temperature $T$, then MixUp labeled & unlabeled; CE for labeled, **L2** for unlabeled.
- **vs. previous / key difference**: Does not propose a new component; it is the first holistic framework combining multiple SSL ideas.
- **Keywords**: holistic, label guessing, K augmentations, sharpening (temperature T), MixUp, $\lambda'=\max(\lambda,1-\lambda)$, L2 loss
- 한 줄: 평균 → sharpen → MixUp → L2. 새 부품 없이 통합. ablation 최대 하락은 MixUp 제거.

#### ReMixMatch (Berthelot et al., 2020, p.36–41)
- **Problem**: MixMatch suffers from **class imbalance in pseudo labels** and **unstable pseudo labels under strong augmentations**.
- **Core idea**: **Distribution Alignment** (rescale prediction by ratio labeled prior / estimated pseudo-label distribution), **Augmentation Anchoring** (pseudo label from a **weakly augmented anchor**, consistency on multiple **strong** augmentations), plus **self-supervised rotation loss**.
- **vs. previous / key difference**: New vs MixMatch = DA, AA, Rotation loss (label guessing, sharpening, MixUp, consistency are shared).
- **Keywords**: distribution alignment, target prior, augmentation anchoring, weak anchor, strong augmentation, rotation prediction
- 한 줄: MixMatch + DA + weak→strong anchoring + rotation. FixMatch의 원형.

#### FixMatch (Sohn et al., NeurIPS 2020 — slide says 2021, p.42–48)
- **Problem**: MixMatch/ReMixMatch pipelines became increasingly complicated.
- **Core idea**: Only three simple ideas — **weak augmentation** to generate a reliable pseudo label, a **confidence threshold** to discard uncertain predictions, **strong augmentation** to train the model to predict the same (hard, one-hot) label with CE. "Predict on an easy example, learn from a difficult one."
- **vs. previous / key difference**: Much simpler (no sharpening, no MixUp, no DA); confidence thresholding is more effective than temperature sharpening.
- **Keywords**: weak→strong, confidence threshold $\tau$, one-hot pseudo label, cross entropy, flip-and-shift, RandAugment/CTAugment
- 한 줄: weak에서 τ 넘는 hard label → strong에 CE. 이것만으로 충분.

#### DivideMix (Li et al., ICLR 2020 — slide says ICML, p.49–55)
- **Problem**: SSL assumes labeled samples are clean, but labeled data may contain **noisy labels**.
- **Core idea**: Reformulate noisy-label learning as SSL — fit a **2-component GMM on per-sample training loss** (low-loss → clean, high-loss → noisy), discard the noisy samples' **labels** (keep them as unlabeled data), then train with **MixMatch**. Two networks **co-divide**: each uses the clean/noisy division made by **the other** network to reduce confirmation bias.
- **vs. previous / key difference**: Not a new learning algorithm; a way to generate labeled/unlabeled sets for MixMatch, plus label co-refinement (clean) and co-guessing (noisy, average of two networks).
- **Keywords**: noisy labels, two networks, co-divide, GMM, per-sample loss, co-refinement, co-guessing
- 한 줄: loss GMM으로 clean/noisy 분할, noisy는 label만 버리고 unlabeled로. 분할은 상대 네트워크가.

#### Pre-training + SSL: Zoph et al. & SimCLRv2 (NeurIPS 2020, p.56–63)
- **Problem**: How do self-supervised pre-training and semi-supervised learning relate?
- **Core idea**: They are **complementary**; SSL is not a replacement for pre-training. Zoph: pre-training's benefit **decreases** with stronger augmentation and more labeled data, while **self-training remains effective** even with strong augmentation and abundant labels. SimCLRv2: **(1) self-supervised pre-training (task-agnostic big model) → (2) supervised fine-tuning → (3) self-training / distillation** on unlabeled data.
- **vs. previous / key difference**: SimCLRv2 distillation uses the teacher's **soft** (temperature-scaled) predictions, not thresholded hard labels.
- **Keywords**: complementary, labeled data scarce, task mismatch, self-training more expensive, task-agnostic backbone, knowledge distillation
- 한 줄: pre-training은 label 적을 때 효과 큼, self-training은 label 많아도 효과. 둘 다 쓰는 게 최신 파이프라인.

#### FreeMatch (Wang et al., ICLR 2023 — slide says ICML, p.64–75)
- **Problem**: FixMatch's **fixed** threshold is the same throughout training and for every class — early on it rejects most unlabeled samples; it ignores the model's learning status.
- **Core idea**: **Self-Adaptive Thresholding (SAT)**: global threshold = **EMA of average confidence** on unlabeled data (starts at $1/C$, increases as the model gets confident), modulated per class by the MaxNorm of the EMA of class probabilities (easy classes higher, difficult classes lower). **Self-Adaptive Fairness (SAF)** prevents predictions from concentrating on a few classes.
- **vs. previous / key difference**: Fixed, manually selected $\tau$ → adaptive $\tau_t(c)$ estimated from unlabeled predictions; no additional model inference.
- **Keywords**: self-adaptive threshold, learning status, EMA, $\tau_0=1/C$, class-specific threshold, MaxNorm, fairness, quality and quantity
- 한 줄: threshold를 낮게 시작해 모델 confidence 따라 올리고, class별로 다르게.

#### AllMatch (Wu et al., IJCAI 2024, p.76–91)
- **Problem**: Threshold-based methods **discard low-confidence samples** (no training signal), yet discarded samples' top-5 accuracy is ~99–100%.
- **Core idea**: **CAT** — global threshold same as FreeMatch, but class-specific scaling from **classifier weight norm** $\|W_c\|$. **BCC** — split classes into **candidate (top-$k_i$ from the weak view) vs negative** groups and enforce the same binary division on the strong view, applied to **every** unlabeled sample.
- **vs. previous / key difference**: FreeMatch = prediction-based local threshold, discards low-confidence samples, top-1 pseudo label; AllMatch = classifier-weight-based, uses every sample, candidate vs negative classes.
- **Keywords**: low confidence ≠ no information, classifier weight norm, candidate/negative classes, binary classification consistency, adaptive $k_i$, upper bound $K$, 100% utilization
- 한 줄: 버리던 샘플도 "정답은 이 후보 안에 있다"는 coarse supervision으로 활용.

#### IFMatch (Wu & Cui, ECCV 2024, p.92–104)
- **Problem**: Every sample above the threshold gets the **same** strong augmentation, but samples differ in difficulty — many **naive samples** have ≈0 consistency loss.
- **Core idea**: Expand weak-to-strong consistency from image space to **feature space** with two student branches sharing one pseudo label: Branch I = weak image + **strong feature** perturbation (high fixed threshold), Branch II = strong image + **optional weak feature** perturbation. **Confidence-Based Identification (CBI)** adds weak feature perturbation only to samples that **stay confident after strong image augmentation** ("easy").
- **vs. previous / key difference**: Not a new pseudo-label generation method — a plug-in that improves **how** selected pseudo labels are used; incorporates both image and feature perturbation (does not replace one with the other).
- **Keywords**: naive vs challenging samples, feature perturbation, Position A (after residual block, strong) / Position B (inside residual branch, weak), CBI, target confidence, plug-in
- 한 줄: 샘플별로 perturbation 강도를 조절. 쉬운 샘플엔 더 어렵게, 어려운 샘플엔 그대로.

### B. Must-know definitions & distinctions
- **Pseudo label**: a model's own high-confidence prediction used as a (hard) training target for an unlabeled sample.
- **Entropy regularization vs Pseudo labeling**: soft penalty on prediction entropy vs its **hard-label** approximation (argmax). Both push boundaries to **low-density** regions.
- **Pseudo Labeling vs Label Propagation**: independent / confidence-based / ignores manifold vs neighborhood / **similarity-based** / leverages manifold.
- **Confirmation bias**: the student reinforces the teacher's mistakes because incorrect pseudo labels accumulate.
- **Noisy Student**: noise only on the **student**; student is **equal-or-larger**; soft pseudo labels > hard.
- **Arazo vs Meta PL**: regularize the student (MixUp, oversampling labeled, strong aug, dropout) vs **learn the teacher** from student feedback on labeled data.
- **Sharpening**: $\mathrm{Sharpen}(p,T)_i=p_i^{1/T}/\sum_j p_j^{1/T}$; lower $T$ → more confident (lower entropy). MixMatch: **average first, then sharpen**.
- **MixMatch loss**: CE on labeled, **L2 (squared error)** on unlabeled.
- **Distribution Alignment**: $\tilde q=\mathrm{Normalize}(p_\theta(y|u)\,p(y)/\tilde p(\hat y))$ — labeled-data class prior over estimated pseudo-label distribution; suppresses over-represented classes, boosts under-represented ones.
- **Augmentation Anchoring**: pseudo label generated **once** from a **weak** anchor, shared across multiple **strong** views.
- **FixMatch**: weak → pseudo label (≥τ) → strong, CE; no sharpening, no MixUp. Thresholding > sharpening.
- **DivideMix co-divide**: per-sample loss → 2-GMM → low-loss clean / high-loss noisy; each network uses **the other's** division. Noisy samples: only their **labels** are discarded.
- **Pre-training vs Self-training (Zoph)**: pre-training helps most when labels are **scarce**, benefit shrinks with stronger aug & more labels; self-training still helps with strong aug & abundant labels; complementary.
- **SimCLRv2 order**: self-supervised pre-training → supervised fine-tuning → self-training (distillation).
- **FreeMatch vs AllMatch local threshold**: EMA of class prediction probabilities (prediction-based) vs **classifier weight norm**.
- **Pseudo-label confidence vs Target confidence (IFMatch)**: decides **whether** to use a target vs **how strongly** to perturb the sample.
- **FreeMatch / AllMatch / IFMatch**: which pseudo labels to select / what to do with unselected samples / should every pseudo label face the same perturbation (p.105).
