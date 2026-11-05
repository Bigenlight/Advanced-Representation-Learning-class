# 고급표현학습 기말 대비 — Core Ideas (L08–)

> 기말 치트시트야. **L08부터** 강의가 끝날 때마다 블록을 이어 붙여.
> 기말 일정은 2026-12-10 (W15)이고, 범위는 W10–W14로 적어 뒀어. **누적 여부는 아직 몰라.** 누적이면 중간 치트시트 `exam/core_ideas.md`도 같이 봐.
> 이 파일은 `exam/_parts/NN_*.md`의 A·B 섹션을 그대로 모은 거야. 고칠 땐 pool 파일을 고치고 `python3 -I exam/_tools/make_core_final.py`로 다시 생성해.
> 각 방법은 **Problem → Core idea → vs. previous** 순서고, 뒤집기 함정은 `Definition ↔ trap` 줄에 있어. 문제는 pool의 C·D, 부품 역할은 E, pipeline은 F, 코드 카드는 `exam/pseudocode_final.md`에 있어.

## 목차

- [08. Weakly-Supervised Learning I](#08)

---

<a id="08"></a>

## 08. Weakly-Supervised Learning I

### Core ideas

#### Weak supervision (Bearman et al., ECCV 2016 ← slide엔 "Berman et al.", p.3–4, 8)
- **Problem**: Detailed annotation is slow and expensive. Annotation time grows from image-level class labels to points, boxes, and pixel labels.
- **Core idea**: Train with "Lower degree (or cheaper) annotation at train time than the required output at test time". It "Typically uses pretrained networks and fine-tunes them by using weak supervision".
- **vs. previous / key difference**: Regular supervision gives annotation at the level of the output (a box to predict boxes). Weak supervision asks the model to output more than the labels contain, such as boxes from image labels.
- **Keywords**: lower degree (or cheaper) annotation, train time vs test time, image label, point, bounding box, pixel labels, annotation time, pretrained network, fine-tune
- **Definition ↔ trap**: "Weak supervision uses cheaper annotation at train time than the output required at test time." ↔ "Weak supervision uses cheaper annotation at test time than the output required at train time."
- 한 줄: train 때 주는 label이 test 때 내야 할 출력보다 약하고 싸.

#### Related settings: semi-supervised, transfer, self-supervised (Jeong et al., NeurIPS 2019; Hoffman NIPS 14, p.5–7)
- **Problem**: Several settings reduce annotation, and their names are easy to confuse. They differ in which labels each image has.
- **Core idea**: **Semi-supervised**: labeled (bounding box) + unlabeled images. **Weakly semi-supervised**: labeled (bounding box) + weakly labeled (image label) images. **Transfer learning**: learn detectors for auxiliary categories with bounding boxes, then transfer the knowledge to weakly supervised categories. **Self-supervised learning**: define a pretext task, learn representations in an unsupervised way, then transfer pre-trained weights to a downstream task.
- **vs. previous / key difference**: In plain weakly supervised learning every image has only an image label. Semi-supervised images may have no label at all, and weakly semi-supervised images have either a box or an image label. Self-supervised transfer still "requires regular annotations" downstream.
- **Keywords**: labeled, unlabeled, weakly labeled, auxiliary categories, pretext task, downstream task
- **Definition ↔ trap**: "Weakly semi-supervised learning uses box-labeled images and image-labeled images." ↔ "Weakly semi-supervised learning uses box-labeled images and unlabeled images."
- 한 줄: 나머지 이미지에 image label이 있으면 weakly semi, 아무 label도 없으면 semi야.

#### Challenges: generic recognition and co-occurrence (p.9–13, 40)
- **Problem**: Object recognition already faces intra-class variations (appearance, viewpoint, scale, aspect ratio), background clutter, and occlusions. An image label adds a further ambiguity about what the object is.
- **Core idea**: Things that occur together cannot be separated by an image label alone. There are three cases: **parts vs object** (face vs person), **context vs object** (rail vs train), and **multiple vs single instance** (several sheep vs one sheep). So "WSOD is an ill-posed problem and ambiguous due to part/whole and co-occurence issues" (slide spelling).
- **vs. previous / key difference**: Box labels state the object's extent directly. An image label only says the class is present somewhere, so the model must infer the extent.
- **Keywords**: intra-class variations, background clutter, occlusions, cooccurrence, parts vs object, context vs object, multiple vs single instance, ill-posed
- **Definition ↔ trap**: "An image label cannot tell a train from the rail that always appears with it." ↔ "An image label is enough to separate a train from its rail."
- 한 줄: 늘 같이 나오는 건 image label만으론 못 갈라 (part, context, 여러 instance).

#### Multiple-Instance Learning, MIL (p.15)
- **Problem**: We know whether a class is in an image, but not which window contains it.
- **Core idea**: "MIL reformulates weakly supervised object detection as a bag classification problem, where each image is a bag of candidate windows, and only the bag label (not the instance labels) is known." The goals are to find true positive instances and to train a window classifier.
- **vs. previous / key difference**: Supervised detection labels each window. MIL labels only the bag, so instance labels are unknown (interpretation: hidden variables).
- **Keywords**: bag, instance, bags = images, instances = windows, positive bags, negative bags, bag label, window classifier
- **Definition ↔ trap**: "In MIL, each image is a bag of windows and only the bag label is known." ↔ "In MIL, each image is a bag of windows and the instance labels are known."
- 한 줄: 사진 = bag, window = instance. bag label만 알아.

#### Standard MIL pipeline (p.16, 19, 32)
- **Problem**: A detector is needed to find the windows, and the windows are needed to train the detector.
- **Core idea**: After initialization, use "Two step alternating optimization": **Re-train** the detector on the selected windows, then **Re-localize** the windows with that detector, and repeat.
- **vs. previous / key difference**: Instead of solving both unknowns at once, fix one and update the other. The cost is two challenges: "Sensitive to initialization (local minimum)" and "Overfitting (locking) to predicted windows".
- **Keywords**: initialize, re-train, re-localize, alternating optimization, local minimum, locking
- **Definition ↔ trap**: "Re-train fixes the windows and updates the detector; re-localize fixes the detector and updates the windows." ↔ "Re-train fixes the detector and updates the windows; re-localize fixes the windows and updates the detector."
- 한 줄: box 고정하고 detector 학습, detector 고정하고 box 다시 고르기를 반복.

#### Bag generation (p.17)
- **Problem**: A bag needs candidate windows.
- **Core idea**: **Sliding windows** are dense over translations, scales, and aspect-ratios (a 4D space). **Object proposals** (Selective Search, Edge Boxes) give about 2k windows per image and are "Commonly used in WSOD".
- **vs. previous / key difference**: Sliding windows scan the image exhaustively. Proposals only keep regions that may hold an object.
- **Keywords**: sliding windows, 4D space, object proposals, Selective Search, Edge Boxes, ~2k per image
- **Definition ↔ trap**: "Object proposals are commonly used to build bags in WSOD." ↔ "Sliding windows are commonly used to build bags in WSOD."
- 한 줄: bag은 보통 object proposal 약 2k개로 만들어.

#### Initialization (Song et al., ICML 2014, p.18)
- **Problem**: The pipeline is sensitive to initialization, so the starting boxes matter.
- **Core idea**: Simple strategies use the whole image or the whole image minus a margin. Song et al. build a graph across positive images and pick initial boxes that are (1) **relevant** (occur in many positive images), (2) **discriminative** (dissimilar to the boxes in the negative images), and (3) **complementary** (capture multiple modes).
- **vs. previous / key difference**: The whole-image start uses no data. The graph start uses agreement among positives and contrast with negatives.
- **Keywords**: whole image, margin, proposal graph, relevant, discriminative, complementary
- **Definition ↔ trap**: "Song's initial boxes are dissimilar to boxes in negative images." ↔ "Song's initial boxes are similar to boxes in negative images."
- 한 줄: 시작 box는 positive에 자주 나오고, negative와 다르고, 여러 모양을 덮어야 해.

#### Re-localize and Re-train with hard max ([Nguyen ICCV 09, Bilen CVPR 14, Cinbis CVPR 14, Papadopoulos CVPR 16], p.20–21)
- **Problem**: How to turn a bag label into a window-level update.
- **Core idea**: **Re-localize**: $\arg\max_b A(x_b)$. "Pick the highest scoring window as positive", "Only one positive instance per image". **Re-train** ("Hinge and cross-entropy loss"): the hinge form requires $\max_b A(x_b) > \Delta$ for positive images and $\max_b A(x_b) < -\Delta$ for negative images. The cross-entropy form uses $\log(\mathrm{Softmax}(\max_b A(x_b)))$ for positive images and $\log(1-\mathrm{Softmax}(\max_b A(x_b)))$ for negative images.
- **vs. previous / key difference**: Unlike supervised detection (e.g. Faster-RCNN), "Only 1 positive instance is used in each positive image" and there are "No negative instances from positive image". All proposals in negative images are treated as negatives.
- **Keywords**: argmax, proposal, appearance model, features, margin Δ, hinge, cross-entropy
- **Definition ↔ trap**: "The standard max formulation keeps one positive instance per image." ↔ "The standard max formulation keeps every window above the margin as positive."
- 한 줄: 제일 높은 window 하나만 positive, negative 사진은 window 전부 negative.

#### Relaxing the max operator (Bilen et al., BMVC 2014, p.22)
- **Problem**: A hard max commits to one proposal, so a wrong early pick is learned as truth.
- **Core idea**: "Relax hard max into soft-max (log-sum-exp) to learn from multiple high-scoring proposals instead of a single proposal": $\log\sum_b \exp A(x_b^+) > \Delta$ and $\log\sum_b \exp A(x_b^-) < -\Delta$. "Hedge your bets on multiple proposals."
- **vs. previous / key difference**: Hard max uses one window. Log-sum-exp is "Less sensitive to initialization" and "Leverages multiple positive samples during training".
- **Keywords**: relaxing max, log-sum-exp, soft-max, hedge your bets, multiple positive samples
- **Definition ↔ trap**: "Log-sum-exp lets multiple high-scoring proposals contribute to training." ↔ "Log-sum-exp keeps only the single highest-scoring proposal."
- 한 줄: hard max를 log-sum-exp로 풀어서 여러 proposal에 걸어.

#### Self-paced learning (Sangineto et al., TPAMI 2019 ← slide엔 "PAMI 17", p.23)
- **Problem**: If an incorrect box is selected initially, the detector is trained on it, "leading to a vicious cycle (locking) where that same box is repeatedly selected".
- **Core idea**: Inspired by Curriculum Learning, train progressively, beginning with easy samples. Easiness measures: confidence of the max scoring window, allowing smaller windows, and intercategory competition.
- **vs. previous / key difference**: Conventional training uses all samples, including difficult ones, from the start. Self-paced training adds harder samples "as localization becomes more reliable".
- **Keywords**: curriculum learning, easy samples, locking, vicious cycle, easiness measures
- **Definition ↔ trap**: "Self-paced WSOD starts from easy images and adds harder ones later." ↔ "Self-paced WSOD starts from all images and removes the easy ones later."
- 한 줄: 쉬운 샘플부터 학습해서 locking을 줄여.

#### Multifold MIL (Cinbis et al., TPAMI 2017 ← slide image credit은 "Sangineto PAMI 17", p.24)
- **Problem**: "Detector overfits into the given proposal", so re-localization returns the same window it was trained on.
- **Core idea**: "the detector performing localization for an image must be one that has not been trained on that image." "Train using positive examples in all folds but k, and all negative examples", then re-localize fold k with that detector.
- **vs. previous / key difference**: Standard MIL localizes an image with a detector that was trained on that image. Multifold MIL uses cross-validation style training, so the localizer has never seen the image.
- **Keywords**: folds, all folds but k, all negative examples, cross-validation style, overfitting
- **Definition ↔ trap**: "Each image is localized by a detector that has never seen that image." ↔ "Each image is localized by the detector that was trained on that image."
- 한 줄: 내 box는 나를 안 본 detector가 골라 줘.

#### Pairwise similarity, sub-categories, exemplars ([Chum CVPR 07, Deselaers ECCV 10, Siva ICCV 11, Bilen CVPR 15], Wang et al., ECCV 2014, p.25–26)
- **Problem**: Each image is localized on its own, so selected windows may disagree across positive images.
- **Core idea**: **Pairwise similarity**: "Enforce consistency across positive images by jointly selecting similar object proposals". In the fully connected CRF, nodes are images, states are windows, the unary term is objectness scores, and the pairwise term is similarity between windows. **Sub-categories**: cluster via pLSA to model intra-class variation. **Exemplars**: pick representative samples.
- **vs. previous / key difference**: Pairwise similarity gives "Less overfitting" but is "Expensive to optimize" (N² similarities) and "Ignores intra-class variation". Sub-categories model that variation but are "Sensitive to number of clusters". Exemplars have "No need to set number of clusters".
- **Keywords**: fully connected CRF, N² similarities, intra-class variation, pLSA, sub-categories, exemplars, number of clusters
- **Definition ↔ trap**: "Exemplars do not need the number of clusters to be set." ↔ "Exemplars are sensitive to the number of clusters."
- 한 줄: 사진끼리 비슷한 box를 같이 고르되, 다양성은 sub-category나 exemplar로 살려.

#### Additional cue: context (Russakovsky ECCV 12, Kantorov et al., ECCV 2016, p.27)
- **Problem**: Object appearance alone may pick parts or background.
- **Core idea**: "Background provides contextual cues for recognition". **Additive**: "select a ROI that is semantically compatible with its context". **Contrastive**: "select a ROI that is outstanding from its context".
- **vs. previous / key difference**: Earlier scores look only inside the window. Context compares the window with its surroundings, which gives "Better separation of foreground and background".
- **Keywords**: context, ROI, foreground, background, additive, contrastive
- **Definition ↔ trap**: "The contrastive model selects a ROI that stands out from its context." ↔ "The contrastive model selects a ROI that is semantically compatible with its context."
- 한 줄: additive는 주변과 어울리는 ROI, contrastive는 주변에서 튀는 ROI.

#### Additional cue: objectness (Slide credit: Vittorio Ferrari, p.28)
- **Problem**: Re-localization can drift to background or to discriminative parts.
- **Core idea**: Objectness quantifies "how likely a window is to contain an object of any class". Selection becomes $\arg\max_b \lambda A(x_b) + (1-\lambda)\mathrm{Obj}(b)$. It "Steers re-localization towards objects and away from background" and "Pushes towards whole objects instead of sub-regions". Sources of Obj(b): EdgeBoxes, Selective Search, BING.
- **vs. previous / key difference**: The appearance model $A$ is class-specific. Objectness is "a class-independent prior".
- **Keywords**: objectness, any class, class-independent prior, λ, Obj(b), whole objects, sub-regions
- **Definition ↔ trap**: "Objectness scores a window for an object of any class." ↔ "Objectness scores a window for the target class only."
- 한 줄: class 상관없이 물체다운 window에 점수를 더해.

#### Priors: equivariance (Bilen et al., BMVC 2014, p.29)
- **Problem**: Predictions for an image and its transformed copy may disagree.
- **Core idea**: "The predicted output should be equivariant to image transformations." If the image changes, the prediction changes in the same way. The figure shows score maps of an image and its mirrored copy before and after.
- **vs. previous / key difference**: (interpretation) Invariance keeps the output the same. Equivariance moves the output along with the input, which fits locations.
- **Keywords**: prior, equivariant, image transformations
- **Definition ↔ trap**: "The predicted output should be equivariant to image transformations." ↔ "The predicted output should stay unchanged when the object moves in the image."
- 한 줄: 사진을 뒤집으면 예측 위치도 같이 뒤집혀야 해.

#### Priors: scale (Shi & Ferrari, ECCV 2016, p.30)
- **Problem**: Weak detectors make localization errors caused by object parts.
- **Core idea**: Add an object-size prior. "Curriculum learning (bigger objects down to smaller ones)" and "Weight object proposals according to estimated size".
- **vs. previous / key difference**: It "Requires training a size estimator from a small set", so (interpretation) it needs a little extra supervision.
- **Keywords**: curriculum learning, bigger to smaller, size estimator, size order, size weighting
- **Definition ↔ trap**: "The scale prior learns from bigger objects down to smaller ones." ↔ "The scale prior learns from smaller objects up to bigger ones."
- 한 줄: 큰 물체부터 배우고, proposal은 추정 크기로 가중해.

#### Priors: motion (Prest et al., CVPR 2012, p.31)
- **Problem**: A still image gives no temporal cue about where the object boundary is.
- **Core idea**: (1) Extract candidate spatio-temporal tubes based on motion segmentation, (2) select one tube per video jointly over all videos, (3) use objectness to reduce object candidates, (4) apply a domain adaptation method to address the video/image gap.
- **vs. previous / key difference**: "(+) Motion cues for object boundaries", "(-) Noisy data". Tubes transferred to image detectors give "much more reliable pseudo bounding boxes than using images alone".
- **Keywords**: spatio-temporal tubes, motion segmentation, one tube per video, domain adaptation, video/image gap
- **Definition ↔ trap**: "Motion selects one tube per video jointly over all videos." ↔ "Motion selects one tube per video independently of other videos."
- 한 줄: 영상의 움직임으로 tube를 뽑아 pseudo box를 만들고 image detector로 옮겨.

#### WSDDN: Weakly Supervised Deep Detection Network (Bilen & Vedaldi, CVPR 2016, p.34–35)
- **Problem**: "Previous work uses CNN as black-box feature extractor", so features are not trained for the weak task.
- **Core idea**: "First end-to-end WSOD method, finetunes a pre-trained deep network". It uses RoI-Pooling and a two stream architecture. **Classification**: "Associates each region with a class score". **Localization**: "Compares regions for each class". An element-wise product gives proposal scores, and a sum over proposals gives image scores (p.36 figure), so image labels can train it.
- **vs. previous / key difference**: (paper) It replaces the alternating re-train/re-localize loop with one network. Its limitations remain: it "Detects discriminative parts rather than whole object extent", "Groups multiple instances together", and compared with Faster RCNN it uses no multiple positive instances, no negative instances in positive images, and no bounding box regression.
- **Keywords**: end-to-end, RoI-Pooling, two stream, softmax over classes, softmax over proposals, element-wise product, sum over proposals, implicit instance scoring
- **Definition ↔ trap**: "The localization stream compares regions for each class." ↔ "The localization stream compares classes for each region."
- 한 줄: 두 softmax를 곱해 proposal 점수를 내고, 더해서 image label로 학습해.

#### OICR: Online Instance Classifier Refinement (Tang et al., CVPR 2017, p.36)
- **Problem**: "WSDDN learns to classify images based on implicit instance scoring", so nothing supervises instances directly.
- **Core idea**: "OICR learns to classify instances". It "Initially uses WSDDN to set instance labels" and "Iteratively refines them by using an instance classifier": WSDDN → pseudo labels → instance classifier → better pseudo labels → better instance classifier. The top proposal and surrounding ones become positives (p.37).
- **vs. previous / key difference**: WSDDN gets only image-level supervision. OICR adds instance-level pseudo labels, and each refinement head is supervised by the previous one inside one network. This is pseudo labeling at the instance level.
- **Keywords**: instance classifier, instance labels, pseudo labels, refinement, highest scoring proposal + surrounding ones, single pseudo object
- **Definition ↔ trap**: "OICR learns to classify instances." ↔ "OICR learns to classify images, while WSDDN learns to classify instances."
- 한 줄: WSDDN 점수로 pseudo instance label을 만들고, 다음 head가 그걸로 다듬어.

#### PCL: Proposal Cluster Learning (Tang et al., TPAMI 2020 ← slide엔 "PAMI 18", p.37)
- **Problem**: "There can be multiple positive instances in an image", but "OICR uses a single pseudo object per image".
- **Core idea**: PCL "builds an undirected graph of top- ranking proposals" and "iteratively selects vertices which have most connections to be the cluster centres". "It groups high-scoring proposals into clusters and uses each cluster as an independent pseudo object".
- **vs. previous / key difference**: OICR: one pseudo object from the top proposal and its neighbors. PCL: several pseudo objects, one per cluster, "leading to more accurate instance-level supervision". The p.37 figure also assigns label 0 to a cluster inside a positive image.
- **Keywords**: proposal cluster, undirected graph, top-ranking proposals, cluster centres, independent pseudo object, multiple positive instances
- **Definition ↔ trap**: "PCL uses each cluster as an independent pseudo object." ↔ "PCL uses a single pseudo object per image."
- 한 줄: 높은 proposal을 cluster로 묶어서 cluster마다 pseudo object 하나.

#### Instance-Aware, Context-Focused, and Memory-Efficient WSOD (Ren et al., CVPR 2020, p.38–39)
- **Problem**: OICR and PCL "mainly improve pseudo-label quality". The detector still focuses on the most salient parts.
- **Core idea**: "Builds on WSDDN and OICR". It "Picks top p % of the highest scoring proposals and surrounding ones as positives" and "Introduces ConcreteDropBlock: a data-driven and parametric drop-out to drop the most relevant regions", trained by "Adversarial optimization": $w^*,\theta^* = \arg\min_w\max_\theta \sum_I \mathcal{L}_{\text{img}}(w,\theta)+\mathcal{L}_{\text{roi}}(w,\theta)$. (paper) $w$ = detector parameters, $\theta$ = Concrete DropBlock parameters.
- **vs. previous / key difference**: "this work explicitly addresses the discriminative-part problem". Hiding the most discriminative regions forces the detector "to attend to the entire object". The p.38 figure also has a regression head (paper: its target comes from pseudo boxes).
- **Keywords**: top p %, ConcreteDropBlock, adversarial optimization, min-max, discriminative parts, student, MIST (paper: Multiple Instance Self-Training), regression
- **Definition ↔ trap**: "ConcreteDropBlock drops the most relevant regions." ↔ "ConcreteDropBlock drops the least relevant regions."
- 한 줄: 제일 눈에 띄는 부분을 적대적으로 가려서 물체 전체를 보게 해.

### Must-know definitions & distinctions
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
