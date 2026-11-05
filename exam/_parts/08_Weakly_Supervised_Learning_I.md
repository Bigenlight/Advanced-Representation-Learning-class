# 08. Weakly-Supervised Learning I (Weakly-Supervised Object Detection) — Core Ideas & Question Pool

> 기말 형식 기준이야 (`exam/PLAYBOOK.md` 7장, 형식은 공지 전까지 중간과 같다고 가정).
> T/F엔 전부 `Why (EN)`을 붙였고, False엔 `Corrected (EN)`도 붙였어.
> 코드 빈칸은 여기 없어. `exam/pseudocode_final.md`에 있어.
> 슬라이드 cite는 웹으로 확인했어. 실제 표기와 다르면 `← slide엔 …`으로 남겼어. 시험 지문은 슬라이드 표기를 따를 수 있어.
> "(paper)" 표시는 슬라이드엔 없고 논문 원문에서 확인한 내용이야.

## A. Core ideas (one block per method/paper/concept, in lecture order)

### Weak supervision (Bearman et al., ECCV 2016 ← slide엔 "Berman et al.", p.3–4, 8)
- **Problem**: Detailed annotation is slow and expensive. Annotation time grows from image-level class labels to points, boxes, and pixel labels.
- **Core idea**: Train with "Lower degree (or cheaper) annotation at train time than the required output at test time". It "Typically uses pretrained networks and fine-tunes them by using weak supervision".
- **vs. previous / key difference**: Regular supervision gives annotation at the level of the output (a box to predict boxes). Weak supervision asks the model to output more than the labels contain, such as boxes from image labels.
- **Keywords**: lower degree (or cheaper) annotation, train time vs test time, image label, point, bounding box, pixel labels, annotation time, pretrained network, fine-tune
- **Definition ↔ trap**: "Weak supervision uses cheaper annotation at train time than the output required at test time." ↔ "Weak supervision uses cheaper annotation at test time than the output required at train time."
- 한 줄: train 때 주는 label이 test 때 내야 할 출력보다 약하고 싸.

### Related settings: semi-supervised, transfer, self-supervised (Jeong et al., NeurIPS 2019; Hoffman NIPS 14, p.5–7)
- **Problem**: Several settings reduce annotation, and their names are easy to confuse. They differ in which labels each image has.
- **Core idea**: **Semi-supervised**: labeled (bounding box) + unlabeled images. **Weakly semi-supervised**: labeled (bounding box) + weakly labeled (image label) images. **Transfer learning**: learn detectors for auxiliary categories with bounding boxes, then transfer the knowledge to weakly supervised categories. **Self-supervised learning**: define a pretext task, learn representations in an unsupervised way, then transfer pre-trained weights to a downstream task.
- **vs. previous / key difference**: In plain weakly supervised learning every image has only an image label. Semi-supervised images may have no label at all, and weakly semi-supervised images have either a box or an image label. Self-supervised transfer still "requires regular annotations" downstream.
- **Keywords**: labeled, unlabeled, weakly labeled, auxiliary categories, pretext task, downstream task
- **Definition ↔ trap**: "Weakly semi-supervised learning uses box-labeled images and image-labeled images." ↔ "Weakly semi-supervised learning uses box-labeled images and unlabeled images."
- 한 줄: 나머지 이미지에 image label이 있으면 weakly semi, 아무 label도 없으면 semi야.

### Challenges: generic recognition and co-occurrence (p.9–13, 40)
- **Problem**: Object recognition already faces intra-class variations (appearance, viewpoint, scale, aspect ratio), background clutter, and occlusions. An image label adds a further ambiguity about what the object is.
- **Core idea**: Things that occur together cannot be separated by an image label alone. There are three cases: **parts vs object** (face vs person), **context vs object** (rail vs train), and **multiple vs single instance** (several sheep vs one sheep). So "WSOD is an ill-posed problem and ambiguous due to part/whole and co-occurence issues" (slide spelling).
- **vs. previous / key difference**: Box labels state the object's extent directly. An image label only says the class is present somewhere, so the model must infer the extent.
- **Keywords**: intra-class variations, background clutter, occlusions, cooccurrence, parts vs object, context vs object, multiple vs single instance, ill-posed
- **Definition ↔ trap**: "An image label cannot tell a train from the rail that always appears with it." ↔ "An image label is enough to separate a train from its rail."
- 한 줄: 늘 같이 나오는 건 image label만으론 못 갈라 (part, context, 여러 instance).

### Multiple-Instance Learning, MIL (p.15)
- **Problem**: We know whether a class is in an image, but not which window contains it.
- **Core idea**: "MIL reformulates weakly supervised object detection as a bag classification problem, where each image is a bag of candidate windows, and only the bag label (not the instance labels) is known." The goals are to find true positive instances and to train a window classifier.
- **vs. previous / key difference**: Supervised detection labels each window. MIL labels only the bag, so instance labels are unknown (interpretation: hidden variables).
- **Keywords**: bag, instance, bags = images, instances = windows, positive bags, negative bags, bag label, window classifier
- **Definition ↔ trap**: "In MIL, each image is a bag of windows and only the bag label is known." ↔ "In MIL, each image is a bag of windows and the instance labels are known."
- 한 줄: 사진 = bag, window = instance. bag label만 알아.

### Standard MIL pipeline (p.16, 19, 32)
- **Problem**: A detector is needed to find the windows, and the windows are needed to train the detector.
- **Core idea**: After initialization, use "Two step alternating optimization": **Re-train** the detector on the selected windows, then **Re-localize** the windows with that detector, and repeat.
- **vs. previous / key difference**: Instead of solving both unknowns at once, fix one and update the other. The cost is two challenges: "Sensitive to initialization (local minimum)" and "Overfitting (locking) to predicted windows".
- **Keywords**: initialize, re-train, re-localize, alternating optimization, local minimum, locking
- **Definition ↔ trap**: "Re-train fixes the windows and updates the detector; re-localize fixes the detector and updates the windows." ↔ "Re-train fixes the detector and updates the windows; re-localize fixes the windows and updates the detector."
- 한 줄: box 고정하고 detector 학습, detector 고정하고 box 다시 고르기를 반복.

### Bag generation (p.17)
- **Problem**: A bag needs candidate windows.
- **Core idea**: **Sliding windows** are dense over translations, scales, and aspect-ratios (a 4D space). **Object proposals** (Selective Search, Edge Boxes) give about 2k windows per image and are "Commonly used in WSOD".
- **vs. previous / key difference**: Sliding windows scan the image exhaustively. Proposals only keep regions that may hold an object.
- **Keywords**: sliding windows, 4D space, object proposals, Selective Search, Edge Boxes, ~2k per image
- **Definition ↔ trap**: "Object proposals are commonly used to build bags in WSOD." ↔ "Sliding windows are commonly used to build bags in WSOD."
- 한 줄: bag은 보통 object proposal 약 2k개로 만들어.

### Initialization (Song et al., ICML 2014, p.18)
- **Problem**: The pipeline is sensitive to initialization, so the starting boxes matter.
- **Core idea**: Simple strategies use the whole image or the whole image minus a margin. Song et al. build a graph across positive images and pick initial boxes that are (1) **relevant** (occur in many positive images), (2) **discriminative** (dissimilar to the boxes in the negative images), and (3) **complementary** (capture multiple modes).
- **vs. previous / key difference**: The whole-image start uses no data. The graph start uses agreement among positives and contrast with negatives.
- **Keywords**: whole image, margin, proposal graph, relevant, discriminative, complementary
- **Definition ↔ trap**: "Song's initial boxes are dissimilar to boxes in negative images." ↔ "Song's initial boxes are similar to boxes in negative images."
- 한 줄: 시작 box는 positive에 자주 나오고, negative와 다르고, 여러 모양을 덮어야 해.

### Re-localize and Re-train with hard max ([Nguyen ICCV 09, Bilen CVPR 14, Cinbis CVPR 14, Papadopoulos CVPR 16], p.20–21)
- **Problem**: How to turn a bag label into a window-level update.
- **Core idea**: **Re-localize**: $\arg\max_b A(x_b)$. "Pick the highest scoring window as positive", "Only one positive instance per image". **Re-train** ("Hinge and cross-entropy loss"): the hinge form requires $\max_b A(x_b) > \Delta$ for positive images and $\max_b A(x_b) < -\Delta$ for negative images. The cross-entropy form uses $\log(\mathrm{Softmax}(\max_b A(x_b)))$ for positive images and $\log(1-\mathrm{Softmax}(\max_b A(x_b)))$ for negative images.
- **vs. previous / key difference**: Unlike supervised detection (e.g. Faster-RCNN), "Only 1 positive instance is used in each positive image" and there are "No negative instances from positive image". All proposals in negative images are treated as negatives.
- **Keywords**: argmax, proposal, appearance model, features, margin Δ, hinge, cross-entropy
- **Definition ↔ trap**: "The standard max formulation keeps one positive instance per image." ↔ "The standard max formulation keeps every window above the margin as positive."
- 한 줄: 제일 높은 window 하나만 positive, negative 사진은 window 전부 negative.

### Relaxing the max operator (Bilen et al., BMVC 2014, p.22)
- **Problem**: A hard max commits to one proposal, so a wrong early pick is learned as truth.
- **Core idea**: "Relax hard max into soft-max (log-sum-exp) to learn from multiple high-scoring proposals instead of a single proposal": $\log\sum_b \exp A(x_b^+) > \Delta$ and $\log\sum_b \exp A(x_b^-) < -\Delta$. "Hedge your bets on multiple proposals."
- **vs. previous / key difference**: Hard max uses one window. Log-sum-exp is "Less sensitive to initialization" and "Leverages multiple positive samples during training".
- **Keywords**: relaxing max, log-sum-exp, soft-max, hedge your bets, multiple positive samples
- **Definition ↔ trap**: "Log-sum-exp lets multiple high-scoring proposals contribute to training." ↔ "Log-sum-exp keeps only the single highest-scoring proposal."
- 한 줄: hard max를 log-sum-exp로 풀어서 여러 proposal에 걸어.

### Self-paced learning (Sangineto et al., TPAMI 2019 ← slide엔 "PAMI 17", p.23)
- **Problem**: If an incorrect box is selected initially, the detector is trained on it, "leading to a vicious cycle (locking) where that same box is repeatedly selected".
- **Core idea**: Inspired by Curriculum Learning, train progressively, beginning with easy samples. Easiness measures: confidence of the max scoring window, allowing smaller windows, and intercategory competition.
- **vs. previous / key difference**: Conventional training uses all samples, including difficult ones, from the start. Self-paced training adds harder samples "as localization becomes more reliable".
- **Keywords**: curriculum learning, easy samples, locking, vicious cycle, easiness measures
- **Definition ↔ trap**: "Self-paced WSOD starts from easy images and adds harder ones later." ↔ "Self-paced WSOD starts from all images and removes the easy ones later."
- 한 줄: 쉬운 샘플부터 학습해서 locking을 줄여.

### Multifold MIL (Cinbis et al., TPAMI 2017 ← slide image credit은 "Sangineto PAMI 17", p.24)
- **Problem**: "Detector overfits into the given proposal", so re-localization returns the same window it was trained on.
- **Core idea**: "the detector performing localization for an image must be one that has not been trained on that image." "Train using positive examples in all folds but k, and all negative examples", then re-localize fold k with that detector.
- **vs. previous / key difference**: Standard MIL localizes an image with a detector that was trained on that image. Multifold MIL uses cross-validation style training, so the localizer has never seen the image.
- **Keywords**: folds, all folds but k, all negative examples, cross-validation style, overfitting
- **Definition ↔ trap**: "Each image is localized by a detector that has never seen that image." ↔ "Each image is localized by the detector that was trained on that image."
- 한 줄: 내 box는 나를 안 본 detector가 골라 줘.

### Pairwise similarity, sub-categories, exemplars ([Chum CVPR 07, Deselaers ECCV 10, Siva ICCV 11, Bilen CVPR 15], Wang et al., ECCV 2014, p.25–26)
- **Problem**: Each image is localized on its own, so selected windows may disagree across positive images.
- **Core idea**: **Pairwise similarity**: "Enforce consistency across positive images by jointly selecting similar object proposals". In the fully connected CRF, nodes are images, states are windows, the unary term is objectness scores, and the pairwise term is similarity between windows. **Sub-categories**: cluster via pLSA to model intra-class variation. **Exemplars**: pick representative samples.
- **vs. previous / key difference**: Pairwise similarity gives "Less overfitting" but is "Expensive to optimize" (N² similarities) and "Ignores intra-class variation". Sub-categories model that variation but are "Sensitive to number of clusters". Exemplars have "No need to set number of clusters".
- **Keywords**: fully connected CRF, N² similarities, intra-class variation, pLSA, sub-categories, exemplars, number of clusters
- **Definition ↔ trap**: "Exemplars do not need the number of clusters to be set." ↔ "Exemplars are sensitive to the number of clusters."
- 한 줄: 사진끼리 비슷한 box를 같이 고르되, 다양성은 sub-category나 exemplar로 살려.

### Additional cue: context (Russakovsky ECCV 12, Kantorov et al., ECCV 2016, p.27)
- **Problem**: Object appearance alone may pick parts or background.
- **Core idea**: "Background provides contextual cues for recognition". **Additive**: "select a ROI that is semantically compatible with its context". **Contrastive**: "select a ROI that is outstanding from its context".
- **vs. previous / key difference**: Earlier scores look only inside the window. Context compares the window with its surroundings, which gives "Better separation of foreground and background".
- **Keywords**: context, ROI, foreground, background, additive, contrastive
- **Definition ↔ trap**: "The contrastive model selects a ROI that stands out from its context." ↔ "The contrastive model selects a ROI that is semantically compatible with its context."
- 한 줄: additive는 주변과 어울리는 ROI, contrastive는 주변에서 튀는 ROI.

### Additional cue: objectness (Slide credit: Vittorio Ferrari, p.28)
- **Problem**: Re-localization can drift to background or to discriminative parts.
- **Core idea**: Objectness quantifies "how likely a window is to contain an object of any class". Selection becomes $\arg\max_b \lambda A(x_b) + (1-\lambda)\mathrm{Obj}(b)$. It "Steers re-localization towards objects and away from background" and "Pushes towards whole objects instead of sub-regions". Sources of Obj(b): EdgeBoxes, Selective Search, BING.
- **vs. previous / key difference**: The appearance model $A$ is class-specific. Objectness is "a class-independent prior".
- **Keywords**: objectness, any class, class-independent prior, λ, Obj(b), whole objects, sub-regions
- **Definition ↔ trap**: "Objectness scores a window for an object of any class." ↔ "Objectness scores a window for the target class only."
- 한 줄: class 상관없이 물체다운 window에 점수를 더해.

### Priors: equivariance (Bilen et al., BMVC 2014, p.29)
- **Problem**: Predictions for an image and its transformed copy may disagree.
- **Core idea**: "The predicted output should be equivariant to image transformations." If the image changes, the prediction changes in the same way. The figure shows score maps of an image and its mirrored copy before and after.
- **vs. previous / key difference**: (interpretation) Invariance keeps the output the same. Equivariance moves the output along with the input, which fits locations.
- **Keywords**: prior, equivariant, image transformations
- **Definition ↔ trap**: "The predicted output should be equivariant to image transformations." ↔ "The predicted output should stay unchanged when the object moves in the image."
- 한 줄: 사진을 뒤집으면 예측 위치도 같이 뒤집혀야 해.

### Priors: scale (Shi & Ferrari, ECCV 2016, p.30)
- **Problem**: Weak detectors make localization errors caused by object parts.
- **Core idea**: Add an object-size prior. "Curriculum learning (bigger objects down to smaller ones)" and "Weight object proposals according to estimated size".
- **vs. previous / key difference**: It "Requires training a size estimator from a small set", so (interpretation) it needs a little extra supervision.
- **Keywords**: curriculum learning, bigger to smaller, size estimator, size order, size weighting
- **Definition ↔ trap**: "The scale prior learns from bigger objects down to smaller ones." ↔ "The scale prior learns from smaller objects up to bigger ones."
- 한 줄: 큰 물체부터 배우고, proposal은 추정 크기로 가중해.

### Priors: motion (Prest et al., CVPR 2012, p.31)
- **Problem**: A still image gives no temporal cue about where the object boundary is.
- **Core idea**: (1) Extract candidate spatio-temporal tubes based on motion segmentation, (2) select one tube per video jointly over all videos, (3) use objectness to reduce object candidates, (4) apply a domain adaptation method to address the video/image gap.
- **vs. previous / key difference**: "(+) Motion cues for object boundaries", "(-) Noisy data". Tubes transferred to image detectors give "much more reliable pseudo bounding boxes than using images alone".
- **Keywords**: spatio-temporal tubes, motion segmentation, one tube per video, domain adaptation, video/image gap
- **Definition ↔ trap**: "Motion selects one tube per video jointly over all videos." ↔ "Motion selects one tube per video independently of other videos."
- 한 줄: 영상의 움직임으로 tube를 뽑아 pseudo box를 만들고 image detector로 옮겨.

### WSDDN: Weakly Supervised Deep Detection Network (Bilen & Vedaldi, CVPR 2016, p.34–35)
- **Problem**: "Previous work uses CNN as black-box feature extractor", so features are not trained for the weak task.
- **Core idea**: "First end-to-end WSOD method, finetunes a pre-trained deep network". It uses RoI-Pooling and a two stream architecture. **Classification**: "Associates each region with a class score". **Localization**: "Compares regions for each class". An element-wise product gives proposal scores, and a sum over proposals gives image scores (p.36 figure), so image labels can train it.
- **vs. previous / key difference**: (paper) It replaces the alternating re-train/re-localize loop with one network. Its limitations remain: it "Detects discriminative parts rather than whole object extent", "Groups multiple instances together", and compared with Faster RCNN it uses no multiple positive instances, no negative instances in positive images, and no bounding box regression.
- **Keywords**: end-to-end, RoI-Pooling, two stream, softmax over classes, softmax over proposals, element-wise product, sum over proposals, implicit instance scoring
- **Definition ↔ trap**: "The localization stream compares regions for each class." ↔ "The localization stream compares classes for each region."
- 한 줄: 두 softmax를 곱해 proposal 점수를 내고, 더해서 image label로 학습해.

### OICR: Online Instance Classifier Refinement (Tang et al., CVPR 2017, p.36)
- **Problem**: "WSDDN learns to classify images based on implicit instance scoring", so nothing supervises instances directly.
- **Core idea**: "OICR learns to classify instances". It "Initially uses WSDDN to set instance labels" and "Iteratively refines them by using an instance classifier": WSDDN → pseudo labels → instance classifier → better pseudo labels → better instance classifier. The top proposal and surrounding ones become positives (p.37).
- **vs. previous / key difference**: WSDDN gets only image-level supervision. OICR adds instance-level pseudo labels, and each refinement head is supervised by the previous one inside one network. This is pseudo labeling at the instance level.
- **Keywords**: instance classifier, instance labels, pseudo labels, refinement, highest scoring proposal + surrounding ones, single pseudo object
- **Definition ↔ trap**: "OICR learns to classify instances." ↔ "OICR learns to classify images, while WSDDN learns to classify instances."
- 한 줄: WSDDN 점수로 pseudo instance label을 만들고, 다음 head가 그걸로 다듬어.

### PCL: Proposal Cluster Learning (Tang et al., TPAMI 2020 ← slide엔 "PAMI 18", p.37)
- **Problem**: "There can be multiple positive instances in an image", but "OICR uses a single pseudo object per image".
- **Core idea**: PCL "builds an undirected graph of top- ranking proposals" and "iteratively selects vertices which have most connections to be the cluster centres". "It groups high-scoring proposals into clusters and uses each cluster as an independent pseudo object".
- **vs. previous / key difference**: OICR: one pseudo object from the top proposal and its neighbors. PCL: several pseudo objects, one per cluster, "leading to more accurate instance-level supervision". The p.37 figure also assigns label 0 to a cluster inside a positive image.
- **Keywords**: proposal cluster, undirected graph, top-ranking proposals, cluster centres, independent pseudo object, multiple positive instances
- **Definition ↔ trap**: "PCL uses each cluster as an independent pseudo object." ↔ "PCL uses a single pseudo object per image."
- 한 줄: 높은 proposal을 cluster로 묶어서 cluster마다 pseudo object 하나.

### Instance-Aware, Context-Focused, and Memory-Efficient WSOD (Ren et al., CVPR 2020, p.38–39)
- **Problem**: OICR and PCL "mainly improve pseudo-label quality". The detector still focuses on the most salient parts.
- **Core idea**: "Builds on WSDDN and OICR". It "Picks top p % of the highest scoring proposals and surrounding ones as positives" and "Introduces ConcreteDropBlock: a data-driven and parametric drop-out to drop the most relevant regions", trained by "Adversarial optimization": $w^*,\theta^* = \arg\min_w\max_\theta \sum_I \mathcal{L}_{\text{img}}(w,\theta)+\mathcal{L}_{\text{roi}}(w,\theta)$. (paper) $w$ = detector parameters, $\theta$ = Concrete DropBlock parameters.
- **vs. previous / key difference**: "this work explicitly addresses the discriminative-part problem". Hiding the most discriminative regions forces the detector "to attend to the entire object". The p.38 figure also has a regression head (paper: its target comes from pseudo boxes).
- **Keywords**: top p %, ConcreteDropBlock, adversarial optimization, min-max, discriminative parts, student, MIST (paper: Multiple Instance Self-Training), regression
- **Definition ↔ trap**: "ConcreteDropBlock drops the most relevant regions." ↔ "ConcreteDropBlock drops the least relevant regions."
- 한 줄: 제일 눈에 띄는 부분을 적대적으로 가려서 물체 전체를 보게 해.

## B. Must-know definitions & distinctions
- **Weak supervision**: "Lower degree (or cheaper) annotation at train time than the required output at test time" (p.4).
- **WSOD**: object detection trained with image-level labels only, with boxes as the output at test time (p.4–5, 14).
- **Semi-supervised vs weakly semi-supervised**: box-labeled + **unlabeled** images vs box-labeled + **image-labeled** images (p.5).
- **Bag vs instance**: bag = image, instance = window. Only the **bag** label is known (p.15).
- **Positive bag vs negative bag**: an image that contains the class vs one that does not. All proposals in negative images are treated as negatives (p.15, 21).
- **Re-train vs re-localize**: update the detector with the windows fixed vs update the windows with the detector fixed (p.16, 20–21).
- **Locking**: the detector overfits to predicted windows, so "that same box is repeatedly selected" (p.16, 23).
- **Sliding windows vs object proposals**: dense scan over a 4D space vs Selective Search or Edge Boxes, about 2k per image, "Commonly used in WSOD" (p.17).
- **Appearance model $A(x_b)$**: the score of proposal $b$ computed from its features $x_b$ (p.20).
- **Hard max vs log-sum-exp**: one window receives the update vs several high-scoring windows share it (p.20–22).
- **Margin Δ**: the score gap required, above Δ for positive images and below −Δ for negative images (p.21–22).
- **Self-paced learning**: train progressively, beginning with easy samples (p.23).
- **Multifold MIL**: positives in all folds but k plus **all** negatives; re-localize fold k with that detector (p.24).
- **Pairwise similarity**: jointly selects similar windows across positive images; costs N² similarities and ignores intra-class variation (p.25).
- **Sub-categories vs exemplars**: pLSA clusters (sensitive to the number of clusters) vs representative samples (no need to set it) (p.26).
- **Additive vs contrastive context**: ROI semantically **compatible** with its context vs ROI **outstanding** from its context (p.27).
- **Objectness**: how likely a window is to contain an object of **any** class; a class-independent prior (p.28).
- **Equivariance**: the output should change along with the image transformation (p.29).
- **Scale prior**: curriculum from **bigger** objects down to smaller ones; needs a size estimator trained on a small set (p.30).
- **Spatio-temporal tube**: a candidate object region followed through video frames, from motion segmentation (p.31).
- **Classification stream vs localization stream (WSDDN)**: gives each region a class score (softmax over classes) vs compares regions for each class (softmax over proposals) (p.34, 36).
- **Discriminative part**: the most salient part of an object, such as a face, which a weak detector picks instead of the whole object (p.35, 38).
- **OICR vs WSDDN**: learns to classify **instances** vs classifies images through implicit instance scoring (p.36).
- **OICR vs PCL**: a single pseudo object (top proposal + surrounding ones) vs one pseudo object per proposal cluster (p.37).
- **ConcreteDropBlock**: "a data-driven and parametric drop-out to drop the most relevant regions" (p.38).

## C. Question pool (26 questions)

### Q1 [T/F]
Weak supervision uses a lower degree of annotation at test time than the output required at train time.
**Answer**: False
- **Why (EN)**: The two times are swapped. The cheap annotation is what we give during training, and the richer output is what the model must produce at test time.
- **Corrected (EN)**: Weak supervision uses a lower degree of annotation at train time than the output required at test time.
- **해설**: "Lower degree (or cheaper) annotation at train time than the required output at test time" (p.4). 예: train엔 image label, test엔 box.

### Q2 [T/F]
A weakly semi-supervised detector learns from a set of images with bounding boxes and a set of images with only image-level labels.
**Answer**: True
- **Why (EN)**: This is the definition of the setting. Part of the data has boxes, and the rest has weak image labels instead of no labels.
- **해설**: "Weakly semi-supervised: learns from a set of labeled (bounding box) and weakly labeled (image label) images." (p.5) 나머지가 unlabeled면 그냥 semi-supervised야.

### Q3 [T/F]
In MIL for weakly supervised object detection, each image is a bag of candidate windows, and only the instance labels are known.
**Answer**: False
- **Why (EN)**: Only the bag label is known. We know that the class is in the image, but we do not know which window contains it.
- **Corrected (EN)**: In MIL, each image is a bag of candidate windows, and only the bag label (not the instance labels) is known.
- **해설**: "only the bag label (not the instance labels) is known." (p.15) instance label을 알면 그냥 supervised detection이야.

### Q4 [T/F]
The standard MIL pipeline alternates between re-training a detector on the selected windows and re-localizing the windows with that detector.
**Answer**: True
- **Why (EN)**: The detector needs windows and the windows need a detector, so MIL fixes one and updates the other in turn.
- **해설**: "Two step alternating optimization": Re-train, Re-localize (p.16). Re-train은 box 고정, Re-localize는 detector 고정이야.

### Q5 [T/F]
Standard MIL training is robust to initialization, because re-localization corrects wrong initial boxes.
**Answer**: False
- **Why (EN)**: A wrong initial box is used to train the detector, and the detector then selects the same box again. This vicious cycle is called locking.
- **Corrected (EN)**: Standard MIL training is sensitive to initialization, because a wrong initial box can lock the detector into a local minimum.
- **해설**: "Sensitive to initialization (local minimum)", "Overfitting (locking) to predicted windows" (p.16). 고리는 p.23: "leading to a vicious cycle (locking) where that same box is repeatedly selected."

### Q6 [T/F]
During re-training in standard MIL, all proposals in negative images are treated as negatives, and no negative proposals are sampled from positive images.
**Answer**: True
- **Why (EN)**: A negative image does not contain the class, so all of its windows are safe negatives. In a positive image, MIL does not know which other windows are truly negative, so it does not use them.
- **해설**: "all proposals in negative images are treated as negatives ... no negative proposals are sampled from positive images." (p.21) 이게 supervised detection과의 차이야. 이유 문장은 슬라이드 원문이 아니라 해석이야.

### Q7 [T/F]
Relaxing the hard max into log-sum-exp lets training use several high-scoring proposals and makes it less sensitive to initialization.
**Answer**: True
- **Why (EN)**: Log-sum-exp is a smooth version of max, so several high-scoring windows share the training signal. One wrong early pick therefore matters less.
- **해설**: "Relax hard max into soft-max (log-sum-exp) to learn from multiple high-scoring proposals instead of a single proposal." "Less sensitive to initialization" (p.22).

### Q8 [T/F]
Self-paced learning for WSOD trains on all samples from the start and removes the easy samples as localization becomes more reliable.
**Answer**: False
- **Why (EN)**: It works in the opposite direction. Training begins with easy samples, and harder samples are added once localization is more reliable.
- **Corrected (EN)**: Self-paced learning starts from easy samples and progressively incorporates harder samples as localization becomes more reliable.
- **해설**: "Learn from easy images first, then progressively incorporate harder samples as localization becomes more reliable." (p.23) Curriculum Learning에서 온 아이디어야.

### Q9 [T/F]
In Multifold MIL, the detector that localizes an image must be one that has not been trained on that image.
**Answer**: True
- **Why (EN)**: A detector trained on an image tends to return the window it was trained on. A held-out detector judges the image without that bias, which reduces locking.
- **해설**: "the detector performing localization for an image must be one that has not been trained on that image." (p.24) cross-validation style이야.

### Q10 [T/F]
In Multifold MIL, the detector for fold k is trained with positive examples in all folds but k and the negative examples of fold k only.
**Answer**: False
- **Why (EN)**: Only the positive examples are held out by fold. All negative examples are used, because negative images have no uncertain window to overfit to.
- **Corrected (EN)**: The detector for fold k is trained with positive examples in all folds but k and all negative examples.
- **해설**: "Train using positive examples in all folds but k, and all negative examples" (p.24). "uncertain window가 없어서"는 슬라이드 문장이 아니라 해석이야.

### Q11 [T/F]
Exemplars are sensitive to the number of clusters, whereas sub-categories do not need this number to be set.
**Answer**: False
- **Why (EN)**: The two methods are swapped. Sub-categories come from clustering with pLSA, so the number of clusters matters, while exemplars only pick representative samples.
- **Corrected (EN)**: Sub-categories are sensitive to the number of clusters, whereas exemplars do not need this number to be set.
- **해설**: Sub-categories "(-) Sensitive to number of clusters", Exemplars "(+) No need to set number of clusters" (p.26).

### Q12 [T/F]
Objectness is a class-independent prior that steers re-localization towards whole objects and away from background.
**Answer**: True
- **Why (EN)**: Objectness measures how likely a window is to contain an object of any class, so it does not depend on the target class. It favors complete objects over parts and background.
- **해설**: "Objectness provides a class-independent prior that guides WSOD toward complete objects rather than discriminative object parts or background regions." (p.28)

### Q13 [T/F]
The scale prior uses a curriculum that goes from smaller objects up to bigger ones.
**Answer**: False
- **Why (EN)**: The curriculum starts with large objects, because they are easier to localize, and moves down to smaller ones. This reduces errors caused by object parts.
- **Corrected (EN)**: The scale prior uses a curriculum that goes from bigger objects down to smaller ones.
- **해설**: "Curriculum learning (bigger objects down to smaller ones)" (p.30). "encouraging the detector to localize large objects first". "easier to localize"는 해석이고, 슬라이드엔 "reducing localization errors caused by object parts"까지 있어.

### Q14 [T/F]
In WSDDN, the classification stream compares regions for each class, while the localization stream associates each region with a class score.
**Answer**: False
- **Why (EN)**: The two roles are swapped. The classification stream scores classes for each region, and the localization stream compares regions for each class.
- **Corrected (EN)**: The classification stream associates each region with a class score, while the localization stream compares regions for each class.
- **해설**: "Classification: Associates each region with a class score", "Localization: Compares regions for each class" (p.34). p.36 그림의 softmax over classes / softmax over proposals와 같은 얘기야.

### Q15 [T/F]
WSDDN tends to detect discriminative parts rather than the whole object extent and may group multiple instances together.
**Answer**: True
- **Why (EN)**: WSDDN is trained only to classify the image, so the most class-revealing region gets the high score. Nothing teaches it where the object ends or how to separate instances.
- **해설**: "(-) Detects discriminative parts rather than whole object extent", "(-) Groups multiple instances together" (p.35). 그림: 사람 얼굴만, 고양이 머리만, 자전거 무리를 box 하나로. 이유 문장은 슬라이드 원문이 아니라 해석이야.

### Q16 [T/F]
OICR initially uses WSDDN to set instance labels and then iteratively refines them by using an instance classifier.
**Answer**: True
- **Why (EN)**: WSDDN scores give the first pseudo labels, and an instance classifier trained on them gives better pseudo labels for the next step. The instance-level target therefore comes from the model's own scores, not from human box labels.
- **해설**: "Initially uses WSDDN to set instance labels", "Iteratively refines them by using an instance classifier" (p.36). "WSDDN -> Pseudo labels -> Instance classifier -> Better pseudo labels -> Better instance classifier".

### Q17 [T/F]
PCL considers only the highest scoring proposal and its surrounding ones as positives, so it uses a single pseudo object per image.
**Answer**: False
- **Why (EN)**: That describes OICR. PCL observes that an image can hold several positive instances, so it groups high-scoring proposals into clusters and treats each cluster as an independent pseudo object.
- **Corrected (EN)**: OICR uses a single pseudo object per image, whereas PCL uses each proposal cluster as an independent pseudo object.
- **해설**: "OICR uses a single pseudo object per image, whereas PCL recognizes that an image may contain multiple object instances." (p.37)

### Q18 [T/F]
ConcreteDropBlock drops the most relevant regions so that the detector must attend to the entire object rather than only its most salient parts.
**Answer**: True
- **Why (EN)**: If the most discriminative region is hidden, the detector can no longer rely on it and must use the rest of the object. This targets the discriminative-part problem directly.
- **해설**: "Introduces ConcreteDropBlock: a data-driven and parametric drop-out to drop the most relevant regions", "the detector is forced to attend to the entire object rather than only its most salient parts" (p.38).

### Q19 [MCQ]
What is the purpose of replacing the hard max with log-sum-exp in re-training?
(A) To reduce the number of proposals per image before re-localization
(B) To learn from multiple high-scoring proposals, not a single one
(C) To remove the need for negative images during re-training
(D) To turn image-level labels into bounding-box labels before re-training
**Answer**: (B)
- **Why (EN)**: Log-sum-exp relaxes the hard max, so several high-scoring proposals contribute to training and one wrong pick matters less.
- **해설**: "Hedge your bets on multiple proposals" (p.22). 식의 negative image 줄을 보면 negative image는 그대로 써서 (C)는 틀려.

### Q20 [MCQ]
Which of the following is NOT listed as an easiness measure for self-paced WSOD?
(A) Confidence of the max scoring window
(B) Allowing smaller windows in the window space
(C) Intercategory competition
(D) The number of negative images in the batch
**Answer**: (D)
- **Why (EN)**: The slide lists selection by window confidence, by allowing smaller windows, and by intercategory competition. Batch composition is not one of them.
- **해설**: Easiness measures for WSOD 세 개 (p.23).

### Q21 [MCQ]
What is the main purpose of the folds in Multifold MIL?
(A) To split the classes so that each detector handles fewer categories
(B) To reduce memory use during training
(C) To localize each image with a detector that has never seen it
(D) To generate more object proposals per image
**Answer**: (C)
- **Why (EN)**: The problem is that a detector overfits to the given proposal. Holding out each fold means that an image is re-localized by a detector not trained on it.
- **해설**: "Problem: Detector overfits into the given proposal", "localize each image with a detector that has never seen that image during training." (p.24)

### Q22 [MCQ]
A detector selects a region because the region stands out from its surroundings. Which model does this describe?
(A) Additive context model
(B) Contrastive context model
(C) Equivariance prior
(D) Scale prior
**Answer**: (B)
- **Why (EN)**: The contrastive model selects a ROI that is outstanding from its context, while the additive model selects one that is semantically compatible with it.
- **해설**: "Additive: select a ROI that is semantically compatible with its context", "Contrastive: select a ROI that is outstanding from its context" (p.27).

### Q23 [MCQ]
In $\arg\max_b \lambda A(x_b) + (1-\lambda)\mathrm{Obj}(b)$, what is the term $\mathrm{Obj}(b)$ for?
(A) It scores how likely the window is to contain the target class only
(B) It counts how many positive images contain the window
(C) It measures the similarity between windows in different images
(D) It adds a class-independent preference for windows that contain a whole object
**Answer**: (D)
- **Why (EN)**: Obj(b) quantifies how likely a window is to contain an object of any class, so it pushes selection towards whole objects and away from background.
- **해설**: "Quantify how likely a window is to contain an object of any class" (p.28). target class 점수는 $A(x_b)$ 쪽이야. λ가 둘의 비중을 정해.

### Q24 [MCQ]
What is the softmax over proposals in WSDDN for?
(A) It compares regions for each class, so it acts as the localization stream
(B) It compares classes for each region, so it acts as the classification stream
(C) It normalizes image-level scores across classes
(D) It removes proposals with low objectness
**Answer**: (A)
- **Why (EN)**: Normalizing across proposals makes regions compete for each class, which shows where that class is in the image.
- **해설**: "Localization: Compares regions for each class" (p.34), p.36 그림 "Softmax over proposals". p.36 블록 이름은 "Multiple instance detection network"이고, OICR은 이 블록으로 WSDDN을 써 (paper). p.34엔 softmax 표기가 없어. (B)는 softmax over classes야.

### Q25 [MCQ]
Which of the following is NOT a limitation of WSDDN compared with the supervised case?
(A) It does not leverage multiple positive instances in positive images
(B) It requires bounding-box annotations for its training images
(C) It does not exploit negative instances in positive images
(D) It does not learn bounding box regression
**Answer**: (B)
- **Why (EN)**: WSDDN is weakly supervised, so it trains from image-level labels. The other three are the listed limitations.
- **해설**: "Limitations (compared to the supervised case, eg Faster RCNN)" 세 개 (p.35). p.21의 차이 두 개에 box regression이 더해진 목록이야.

### Q26 [MCQ]
How does the method with ConcreteDropBlock differ from OICR and PCL in what it mainly tackles?
(A) It mainly improves pseudo-label quality by grouping proposals into clusters
(B) It removes the need for image-level labels by generating its own labels
(C) It tackles discriminative parts by hiding the most relevant regions
(D) It replaces the deep network with sliding windows over the whole image
**Answer**: (C)
- **Why (EN)**: OICR and PCL mainly improve pseudo-label quality, while this work suppresses the most discriminative regions so that the detector attends to the entire object.
- **해설**: "Unlike OICR and PCL that mainly improve pseudo-label quality, this work explicitly addresses the discriminative-part problem." (p.38)

## D. Fill-in-the-blank pool (9 questions)

### F1 [Blank]
Weak supervision: Lower degree (or cheaper) annotation at (1) ______ time than the required output at (2) ______ time.
**Answer**: (1) train (accept: training) (2) test (accept: testing, inference) — 순서가 함정이야 (p.4).

### F2 [Blank]
MIL reformulates weakly supervised object detection as a (1) ______ classification problem, where each image is a bag of candidate (2) ______.
**Answer**: (1) bag (2) windows (accept: proposals, instances, boxes) — bags = images, instances = windows (p.15).

### F3 [Blank]
The two challenges of the standard MIL pipeline are that it is sensitive to (1) ______ (local minimum) and that it overfits, or (2) ______, to predicted windows.
**Answer**: (1) initialization (2) locks (accept: locking) — p.16.

### F4 [Blank]
A graph-based initialization picks initial boxes that are relevant, (1) ______ (dissimilar to the boxes in the negative images), and (2) ______ (capture multiple modes).
**Answer**: (1) discriminative (2) complementary — p.18.

### F5 [Blank]
To hedge bets on multiple proposals, the hard max is relaxed into ______.
**Answer**: log-sum-exp (accept: soft-max, softmax) — p.22. 식은 $\log\sum_b \exp A(x_b^+) > \Delta$.

### F6 [Blank]
Multifold MIL: Train using positive examples in all folds but (1) ______, and all (2) ______ examples.
**Answer**: (1) k (accept: the held-out fold, one fold) (2) negative — p.24.

### F7 [Blank]
Motion prior: first extract candidate spatio-temporal (1) ______ based on motion segmentation, and later apply a (2) ______ method to address the video/image gap.
**Answer**: (1) tubes (2) domain adaptation — p.31.

### F8 [Blank]
WSDDN combines a softmax over classes and a softmax over proposals with an (1) ______ product, then takes a sum over (2) ______ to obtain image scores.
**Answer**: (1) element-wise (accept: Hadamard) (2) proposals (accept: regions) — p.36 그림 ("Multiple instance detection network" 블록 = WSDDN, paper). 이미지 점수를 만들어야 image label로 학습할 수 있어.

### F9 [Blank]
PCL builds an undirected (1) ______ of top-ranking proposals and iteratively selects vertices which have most connections to be the (2) ______ centres.
**Answer**: (1) graph (2) cluster — p.37.

## E. Component roles

"Without it" 열은 슬라이드의 문제 제기에서 이어지는 해석이야. 슬라이드 문장 그대로는 아니야.

| Part | What is it for? | Without it | Page |
|---|---|---|---|
| max over proposals | It turns a bag label into a window decision by treating the highest-scoring window as the bag's evidence. | The image-level label cannot be connected to any single window. | p.20–21 |
| log-sum-exp | It relaxes the hard max so that several high-scoring proposals share the training signal. | One proposal takes all the signal, and training is more sensitive to initialization. | p.22 |
| Margin Δ | It requires the best score to be clearly above Δ in positive images and clearly below −Δ in negative images. | Scores only need to differ slightly, which gives a weak separation. | p.21–22 |
| λ | It balances the class-specific appearance score against the class-independent objectness score. | The two scores cannot be traded off against each other. | p.28 |
| Objectness Obj(b) | It steers re-localization towards whole objects and away from background, for any class. | Selection relies on appearance alone and can drift to parts or background. | p.28 |
| Folds | They make sure that an image is localized by a detector that was not trained on it. | The detector returns the window it already fit, which is locking. | p.24 |
| Softmax over classes | It gives each region a class score by comparing classes for that region. | Regions have no class evidence to combine with location. | p.34, 36 |
| Softmax over proposals | It compares regions for each class, which provides the localization signal. | The network can classify, but it does not say which region is responsible. | p.34, 36 |
| Instance classifier | It learns to classify proposals directly from pseudo instance labels, which refines WSDDN's implicit scoring. | Instances are scored only implicitly through the image-level loss. | p.36 |
| IoU-surrounding rule | It extends the label of the top proposal to surrounding proposals, so a neighborhood becomes positive and not just one box. | Only one proposal is positive, and the classifier may lock onto a part. | p.37 (IoU itself: paper) |
| Cluster centre | It marks one object candidate, so proposals around it form an independent pseudo object. | Multiple instances in an image collapse into a single pseudo object. | p.37 |
| ConcreteDropBlock | It hides the most relevant regions, so the detector must attend to the entire object. | The detector keeps relying on the most discriminative parts. | p.38 |

## F. Pipelines

"Target comes from"와 "Gradient goes to" 줄은 슬라이드 문장이 아니라 해석이야. OICR·PCL·Ren은 논문으로 확인했어. Multifold의 detector는 논문상 linear SVM이라 "gradient"는 "학습되는 쪽"이라는 뜻으로만 읽어.

**Standard MIL** (p.16, 20–21)
1. Generate proposals for each image (the bag).
2. Initialize one box per positive image.
3. Re-train the detector on those boxes, using all proposals of negative images as negatives.
4. Re-localize: pick the highest-scoring proposal in each positive image.
5. Repeat steps 3–4.
- **Target comes from**: the initial boxes in the first round, then the detector's own previous selection, constrained by the image-level labels.
- **Gradient goes to**: the detector only. The box choice is an argmax, so it is fixed during re-training.

**Multifold MIL** (p.24)
1. Split the positive images into folds.
2. For fold k, train a detector on positives in all folds but k and on all negatives.
3. Re-localize the images in fold k with that detector.
4. Repeat for every fold, then repeat the whole round.
- **Target comes from**: the initial boxes in the first round, then boxes picked by a detector that never saw the image.
- **Gradient goes to**: the detector of the current fold.

**Motion prior** (p.31)
1. Extract candidate spatio-temporal tubes based on motion segmentation.
2. Select one tube per video jointly over all videos.
3. Use objectness to reduce object candidates.
4. Sample frames from the tubes and train an object detector, with domain adaptation to address the video/image gap.
- **Target comes from**: motion-based tubes, which act as pseudo bounding boxes.
- **Gradient goes to**: the image object detector.

**WSDDN** (p.34, 36)
1. Pass the image through conv layers and pool each region proposal with RoI-Pooling.
2. Classification stream: softmax over classes for each region.
3. Localization stream: softmax over proposals for each class.
4. Multiply the two element-wise to get proposal scores.
5. Sum over proposals to get image scores and compare them with the image labels.
- **Target comes from**: human image-level labels.
- **Gradient goes to**: both streams and the pre-trained network, end to end.

**OICR** (p.36–37)
1. Run WSDDN to get proposal scores.
2. Take the highest-scoring proposal and surrounding ones as positives to form pseudo instance labels.
3. Train an instance classifier with those labels.
4. Use its scores to make better pseudo labels for the next refinement head.
- **Target comes from**: human image labels for the WSDDN head. Each refinement head's target is built from the previous stage's scores, only for classes in the image label (paper).
- **Gradient goes to**: each head and the shared network. Labels come from argmax and overlap, so no gradient passes through label values (paper: PCL).

**PCL** (p.37)
1. Take the top-ranking proposals.
2. Build an undirected graph over them.
3. Iteratively select the vertices with the most connections as cluster centres.
4. Treat each cluster as an independent pseudo object.
5. Train the instance classifier on these cluster labels.
- **Target comes from**: human image labels for the first stream. Each later stream's target is built from clusters of the previous stream's scores, only for classes in the image label (paper).
- **Gradient goes to**: the instance classifiers and the shared network.

**Instance-aware, context-focused WSOD** (p.38)
1. Compute WSDDN-style proposal scores and image scores.
2. Pick the top p % of the highest-scoring proposals and surrounding ones as positives.
3. Train the first student block on these pseudo labels, with classification and regression heads. Each student then produces pseudo labels, through MIST, for the next student.
4. ConcreteDropBlock drops the most relevant regions during training.
5. Optimize adversarially: the detector minimizes the loss and ConcreteDropBlock maximizes it.
- **Target comes from**: image labels for $\mathcal{L}_{\text{img}}$, and pseudo boxes for $\mathcal{L}_{\text{roi}}$ (paper).
- **Gradient goes to**: $w$ (detector) to minimize, and $\theta$ (ConcreteDropBlock) to maximize.
