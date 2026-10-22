# 02. Self-Supervised Learning (SSL) — Core Ideas & Question Pool

## A. Core ideas (one block per method/paper/concept, in lecture order)

### Why SSL? — Limitations of supervised learning (p.3–10)
- **Problem**: Supervised pre-training works well, but "real" labels are difficult and expensive and cannot keep up with the data we generate.
- **Core idea**: ImageNet (14M images) took 22 human years; semi-automatic labels (hashtags, GPS) are noisy. Complex concepts (video) and rare concepts (long tail) are hard to label, and new domains (e.g., medical) need fresh labels, so we need supervision that does not come from humans.
- **Keywords**: 22 human years, semi-automatic labels, hashtags / GPS, long tail (10% of classes = 93% of data), "Labelling cannot scale to the size of the data we generate"
- 한 줄: label 비용, long-tail, video, domain 차이 때문에 supervised는 scale이 안 됨 → SSL 등장

### Self-Supervised Learning — definition & paradigm (p.11–17, p.20)
- **Problem**: How can we learn representations without manual annotation?
- **Core idea**: Obtain "labels" from the data itself by using a "semi-automatic" process and predict part of the data from other parts. A **pretext task** (no manual annotation, but trained with supervised objectives like classification or regression) learns the encoder, which is then transferred to a **downstream task** (the application you care about, with a small labeled dataset).
- **vs. previous / key difference**: Unlike unsupervised learning (clustering, PCA), SSL has an explicit prediction target generated automatically from data; unlike supervised learning, the supervision comes from the data itself, not from humans.
- **Keywords**: pseudo labels, pretext task, downstream task, "predict part of the data from other parts", Word2vec / BERT ("fill in the blanks"), MAE / SimCLR / CLIP
- 한 줄: data 일부로 다른 일부를 예측하는 pretext task로 encoder를 학습하고, 소량 label downstream task로 transfer

### Evaluating SSL — linear probing / shallow network on frozen features (p.18, p.46, p.48)
- **Problem**: A pretext task is only a means, so how do we tell whether an SSL method learned good features?
- **Core idea**: Learn a feature extractor with the pretext task, then attach a **shallow network** (e.g., a linear classifier on "fixed" features) and train it on the target task with a small amount of labeled data. We judge the method by **downstream performance**, not by pretext-task performance.
- **Keywords**: feature extractor, shallow network, linear classifier on fixed features, downstream (classification / detection / segmentation), fine-tuning
- 한 줄: pretext 성능이 아니라 downstream 성능으로 평가, shallow head만 학습해야 representation 품질을 그대로 볼 수 있음

### Relative Position of Patches (Doersch et al., ICCV 2015, p.21–23)
- **Problem**: Learn visual features without labels from the spatial context inside an image.
- **Core idea**: Given a center patch and one of its 8 neighbors, predict which of the 8 positions the neighbor came from (8-way classification). A two-tower (shared-weight) network has to understand object parts and their spatial layout.
- **Keywords**: 8-way classification, center patch, siamese / shared weights, local spatial context, gap + jitter (against shortcuts)
- 한 줄: 이웃 patch의 상대 위치를 8-way 분류 → part 간 local spatial 관계를 학습

### Inpainting — Context Encoders (Pathak et al., CVPR 2016, p.24–28)
- **Problem**: Can we predict missing regions from surrounding context?
- **Core idea**: Mask a region and train an encoder–decoder (with a channel-wise FC layer) to generate it. The network learns from **reconstruction (masked L2) + adversarial loss**: L2 alone gives blurry results, adversarial alone gives sharp but out-of-context results, and L2+Adv gives the best result.
- **vs. previous / key difference**: It is a generative, pixel-level reconstruction task, while relative position is a discriminative classification task.
- **Keywords**: missing region, object parts / scene structure / semantic context, channel-wise fully connected, reconstruction loss, adversarial loss, blurry
- 한 줄: 가려진 영역을 주변 context로 복원, L2(전체 구조) + adversarial(선명함) 조합

### Jigsaw Puzzles (Noroozi & Favaro, ECCV 2016, p.29–33)
- **Problem**: Learn object parts and their global arrangement without labels.
- **Core idea**: Shuffle 3×3 tiles and classify which permutation was used. Since 9! = 362,880 permutations are too many classes, use a subset of 64 permutations that **maximize pairwise Hamming distance**. Each tile goes through a shared-weight branch (Context-Free Network), and features are joined only at the end.
- **vs. previous / key difference**: It generalizes relative position (2 patches) to 9 patches, so it learns global instead of local spatial structure.
- **Keywords**: 9! = 362,880, Hamming distance, permutation classification, shortcut (edge continuity, color/intensity statistics), random gap / color jitter / normalization
- 한 줄: 9개 tile 순열 index를 분류, low-level shortcut을 막아야 의미 있는 feature가 나옴

### Colorization (Zhang et al., ECCV 2016, p.34–39)
- **Problem**: Can a network recover colors from grayscale images? (Color prediction requires semantic understanding.)
- **Core idea**: In Lab space, predict the ab channels from the L (lightness) channel. Multiple colors can be plausible and regression averages them (desaturated), so the problem is solved by **classification** over **313 color classes** instead of regression.
- **Keywords**: Lab color space, L → ab, 313 color classes, regression averages (desaturation), multimodal, object semantics
- 한 줄: 흑백 L에서 ab를 예측, 색은 multimodal이라 regression 대신 313-bin classification

### Rotation Prediction — RotNet (Gidaris et al., ICLR 2018, p.40–43)
- **Problem**: Find a simple pretext task that forces the model to recognize objects.
- **Core idea**: Rotate the image by 0°, 90°, 180° or 270° and predict which rotation was applied (4-way classification). No labels are needed: the applied rotation becomes the label. To recognize rotation, the network must recognize objects.
- **Keywords**: 4 rotations, 4-way classification, "the applied rotation becomes the label", object-centric understanding, canonical orientation
- 한 줄: 회전 각도 4-way 분류, 구현이 가장 단순하고 당시 pretext 중 최고 성능

### Limitations of hand-crafted pretext tasks (p.44–49, p.55)
- **Problem**: Pretext tasks focus on "visual common sense", but is solving them enough for good, general features?
- **Core idea**: Problem 1: coming up with individual pretext tasks is **tedious**. Problem 2: the learned representations **may not be general**, because higher layers specialize to the pretext task (in a Jigsaw-pretrained network, linear-probe accuracy peaks at res4 and drops at the last layer, res5). We only "hope" the pre-training task and the transfer task are "aligned".
- **Keywords**: visual common sense, tedious, "The Hope of Generalization", aligned, higher layers do not generalize, tied to a specific pretext task
- 한 줄: hand-crafted pretext는 (1) 설계가 tedious, (2) 마지막 layer가 pretext에 특화돼 general하지 않음

### Multiple Pretext Tasks — CDJP (Kim et al., WACV 2018, p.50–53)
- **Problem**: A single pretext task ties the representation to that task.
- **Core idea**: Completing Damaged Jigsaw Puzzles: one network solves **jigsaw + inpainting + colorization** at the same time on damaged patches (dropped patch, dropped color channels), with a shared backbone and task-specific heads ($L_{jig}$, $L_{inp}$, $L_{col}$). Combining tasks generally improves the results.
- **vs. previous / key difference**: Multi-task instead of single-task, but the representation can still be tied to the chosen pretext tasks ("even multi tasks", p.55).
- **Keywords**: damaged jigsaw puzzle, jigsaw + inpainting + colorization, shared backbone, multi-task loss
- 한 줄: 세 pretext를 한 네트워크로 동시에 → 단일 task보다 general, 그래도 사람이 고른 task 집합에 묶임

### Toward a more general pretext task (p.54–56)
- **Problem**: Representations learned from image transformations may be tied to the specific pretext task.
- **Core idea**: Treat the transformed images as views of the **same object** (pull together) and other images as **different objects** (push apart). This leads to contrastive learning (next lecture).
- **Keywords**: same object vs different object, positive / negative pair, task-agnostic, contrastive learning
- 한 줄: "변환을 맞혀라" 대신 "같은 object는 가깝게, 다른 object는 멀게" → contrastive

### Space-Time Cubic Puzzles (Kim, Cho, Kweon, AAAI 2019, p.57–66)
- **Problem**: A video additionally provides temporal order, motion and changes over time, and an action cannot be understood from two frames.
- **Core idea**: Divide a video into a 2×2×4 grid, sample 4 cubes either spatially (2×2×1) or temporally (1×1×4), shuffle them, and predict the original order (4! = 24 permutations × 2 orientations = 48 classes). Four shared 3D CNN towers process one cube each and combine features only at the end; each tower does not know whether the puzzle is spatial or temporal.
- **vs. previous / key difference**: It extends image Jigsaw to a joint space-time puzzle. Joint space-time learning beats learning each dimension separately (spatial and temporal context are complementary).
- **Keywords**: 2×2×4 grid, spatial / temporal puzzle, 3D CNN towers, 48 classes, channel replication, spatio-temporal jittering, upside-down classification
- 한 줄: video cube를 공간·시간 순서대로 맞히는 3D jigsaw, joint 학습이 따로 학습보다 좋음

### Audio-Visual Temporal Synchronization — AVTS (Korbar et al., NeurIPS 2018, p.67–70)
- **Problem**: Use the natural synchronization of video and audio as free supervision.
- **Core idea**: Classify whether the video and audio are temporally aligned (in-sync = same time segment, out-of-sync = different time segments). A video stream (spatio-temporal CNN, motion + appearance) and an audio stream (filterbank features) are trained cooperatively with a contrastive loss. **Curriculum learning** (easy negatives first, then a mix with hard negatives) works best.
- **Keywords**: in-sync / out-of-sync pair, easy / hard / super-hard negative, cooperative learning, curriculum learning, "how positive and negative examples are constructed"
- 한 줄: audio-video sync 여부를 판별하는 multi-modal SSL, negative 구성과 curriculum이 핵심

### Invariance vs Equivariance (p.71–75)
- **Problem**: Modern SSL trains the model to ignore transformations (random crop, color jitter, flip, blur), but is invariance always desirable?
- **Core idea**: A pretext task determines what information is preserved in the representation. **Invariance**: $f(t(x)) = f(x)$, the representation does not change. **Equivariance**: $f(t(x)) = \phi(t, f(x))$, the representation changes predictably. Classification may benefit from invariance, but detection needs spatial information and flower classification needs color information.
- **Keywords**: invariance, equivariance, $\phi(t, f(x))$, "What is the object?" vs "Where is the object?", spatial / color information
- 한 줄: 무조건 invariant하면 위치·색 정보가 사라짐 → 변환을 예측 가능하게 보존하는 equivariance 필요

### Self-supervised Transformation Learning — STL (Yu et al., NeurIPS 2024, p.76–79)
- **Problem**: Learn equivariant representations without transformation labels (parameters).
- **Core idea**: Three losses. Brown (invariant): same image under transformation, preserve semantic information. Blue (transformation learning): **same transformation on different images**, learn transformation representations. Green (equivariant): apply the learned transformation to predict the transformed representation.
- **vs. previous / key difference**: The transformation representation is learned without transformation labels, and it discovers structure (crop is separated from color transformations, similar color transformations are close, intensity changes continuously). Preserving transformation information improves localization and generalization.
- **Keywords**: transformation representation, same transformation on different images, without transformation labels, type and intensity, EquiMod / AugSelf / SimCLR
- 한 줄: 다른 이미지·같은 변환 쌍으로 label 없이 변환 표현을 학습 → equivariance, detection 향상

## B. Must-know definitions & distinctions

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

## C. Question pool (18 questions)

### Q1 [T/F]
Self-supervised learning obtains "labels" from the data itself by using a "semi-automatic" process, for example by predicting part of the data from other parts.
**Answer**: True — 슬라이드 정의 그대로 (p.12)

### Q2 [T/F]
A self-supervised learning method is mainly judged by how accurately it solves its pretext task.
**Answer**: False — pretext 성능엔 관심 없고 downstream task에서 learned feature가 얼마나 유용한지로 평가 (p.18, p.46)

### Q3 [MCQ]
According to the SSL paradigm, which statement correctly describes the **downstream task**?
(A) It is defined from the data itself and needs no manual annotation.
(B) It is the application you care about, and its dataset is small and labeled.
(C) It is the task used to generate pseudo labels automatically.
(D) It is always trained on a larger dataset than the pretext task.
**Answer**: (B) — (A), (C)는 pretext task 설명, downstream은 "You do not have large datasets" (p.16)

### Q4 [T/F]
To evaluate an SSL method, a shallow network is attached on the learned feature extractor and trained on the target task with a small amount of labeled data.
**Answer**: True — 평가 절차 그대로 (p.18)

### Q5 [MCQ]
Which pretext task is matched with the learned concept **"Global Spatial Structure"**?
(A) Relative Position
(B) Inpainting
(C) Jigsaw
(D) Rotation
**Answer**: (C) — Relative Position = Local Spatial Context, Inpainting = Context & Object Parts, Rotation = Object-Centric Understanding (p.45)

### Q6 [T/F]
In Context Encoders, the reconstruction (L2) loss makes the inpainted region sharp, while the adversarial loss alone causes blurry results.
**Answer**: False — 반대. L2만 쓰면 blurry, adversarial이 sharp/realistic하게 만들고, 둘을 함께 쓰는 게 최고 (p.25–27)

### Q7 [MCQ]
Why does the Jigsaw pretext task use only a subset of the possible permutations?
(A) 9! = 362,880 permutations are too many classes, so a subset of 64 permutations that maximize pairwise Hamming distance is used.
(B) Most permutations make the image look natural, so they are removed as easy negatives.
(C) A subset of 64 permutations that minimize pairwise Hamming distance makes the puzzle harder.
(D) Only permutations that keep the center tile fixed are needed for 8-way classification.
**Answer**: (A) — Hamming distance를 maximize (minimize 아님). (D)는 relative position과 헷갈리게 한 것 (p.30)

### Q8 [T/F]
In colorization, multiple colors can be plausible and regression averages them, so the task is solved as classification over 313 color classes.
**Answer**: True — regression → 313 color classes classification (p.37)

### Q9 [MCQ]
Which statement about Rotation Prediction (RotNet) is correct?
(A) Rotation angles must be manually annotated by humans.
(B) The network predicts the order of 9 shuffled tiles.
(C) The applied rotation becomes the label, and to recognize rotation the network must recognize objects.
(D) The network regresses a continuous rotation angle between 0° and 360°.
**Answer**: (C) — label 불필요, 적용한 회전(0/90/180/270)이 곧 label인 4-way classification (p.40–41)

### Q10 [T/F]
In the Jigsaw puzzle task, the network may solve the puzzle using low-level cues such as edge continuity and color/intensity statistics instead of semantics.
**Answer**: True — shortcut(cheating) 문제, gap·jitter·normalization 등으로 막음 (p.31)

### Q11 [MCQ]
Which of the following is **NOT** a problem of hand-crafted pretext tasks mentioned in the lecture?
(A) Coming up with individual pretext tasks is tedious.
(B) The learned representations may not be general.
(C) Learned representations may be tied to a specific pretext task, even with multiple tasks.
(D) Pretext tasks require a large amount of manual annotation.
**Answer**: (D) — pretext는 label을 자동 생성하므로 manual annotation 불필요. (A) p.46, (B) p.49, (C) p.55 (p.16, p.46, p.49, p.55)

### Q12 [T/F]
In a Jigsaw-pretrained network, linear classifiers on the last (highest) layer give the best transfer performance, because higher layers generalize best.
**Answer**: False — "Higher layers do not generalize": res4에서 최고, 마지막 res5에서 하락 → representation이 pretext에 특화 (p.49)

### Q13 [MCQ]
CDJP (Completing Damaged Jigsaw Puzzles, WACV 2018) learns representations by combining which pretext tasks?
(A) Rotation + relative position + colorization
(B) Jigsaw puzzle + inpainting + colorization
(C) Jigsaw puzzle + rotation + audio-visual synchronization
(D) Inpainting + temporal order prediction + rotation
**Answer**: (B) — damaged jigsaw = 퍼즐 + 빠진 patch 채우기 + 색 복원 (p.50, p.52)

### Q14 [T/F]
In Space-Time Cubic Puzzles, each 3D CNN tower knows whether the current puzzle is spatial or temporal.
**Answer**: False — "Each tower does not know whether the puzzle is spatial or temporal." 그래서 범용 spatio-temporal feature를 배움 (p.62)

### Q15 [MCQ]
In the Relative Position of Patches pretext task (Doersch et al.), what does the network predict?
(A) The rotation angle applied to the whole image
(B) Which of the 8 neighboring positions a patch came from, relative to the center patch
(C) The original permutation of 9 shuffled tiles
(D) The pixel values of a masked center region
**Answer**: (B) — center patch + 이웃 patch 한 쌍 → 8개 위치 중 어디인지 분류 (X = (center, neighbor), Y = 3). (A) RotNet, (C) Jigsaw, (D) Inpainting (p.21)

### Q16 [T/F]
In audio-visual temporal synchronization, an out-of-sync pair consists of audio and video that come from the same time segment.
**Answer**: False — 같은 time segment = in-sync pair, 다른 time segment = out-of-sync pair (p.68)

### Q17 [T/F]
Equivariance means that the representation changes predictably according to the input transformation, i.e., $f(t(x)) = \phi(t, f(x))$.
**Answer**: True — invariance는 $f(t(x)) = f(x)$ (representation 불변) (p.75)

### Q18 [MCQ]
In STL (Self-supervised Transformation Learning), which component learns **transformation representations**, without transformation labels?
(A) Making the same image under a transformation have the same representation
(B) Applying the same transformation to different images and matching their transformation representations
(C) Predicting the rotation angle (0°, 90°, 180°, 270°) of each image
(D) Pushing representations of different images apart as negatives
**Answer**: (B) — Blue loss: "Same transformation on different images → Learn transformation representations". (A)는 Brown(semantic 보존, invariance), (C)는 RotNet (p.76–77)

<!-- REVIEWED | fixes: 1 | reviewer: opus -->
