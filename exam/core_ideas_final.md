# 고급표현학습 기말 대비 — Core Ideas (L08–)

> 기말 치트시트야. **L08부터** 강의가 끝날 때마다 블록을 이어 붙여.
> 기말 일정은 2026-12-10 (W15)이고, 범위는 W10–W14로 적어 뒀어. **누적 여부는 아직 몰라.** 누적이면 중간 치트시트 `exam/core_ideas.md`도 같이 봐.
> 이 파일은 `exam/_parts/NN_*.md`의 A·B 섹션을 그대로 모은 거야. 고칠 땐 pool 파일을 고치고 `python3 -I exam/_tools/make_core_final.py`로 다시 생성해.
> 각 방법은 **Problem → Core idea → vs. previous** 순서고, 뒤집기 함정은 `Definition ↔ trap` 줄에 있어. 문제는 pool의 C·D, 부품 역할은 E, pipeline은 F, 코드 카드는 `exam/pseudocode_final.md`에 있어.

## 목차

- [08. Weakly-Supervised Learning I](#08)
- [09. Weakly-Supervised Learning II](#09)

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

---

<a id="09"></a>

## 09. Weakly-Supervised Learning II

### Core ideas

#### Fully vs weakly supervised semantic segmentation (Slide credit: Rodrigo Benenson, p.3–4, 19)
- **Problem**: Semantic segmentation needs a class for every pixel. Pixel masks for training are the most expensive kind of label.
- **Core idea**: "Fully supervised → **Directly** supervised": the training label is the pixel mask itself. "Weakly supervised → **Indirectly** supervised": the training label is weaker than the output, e.g. a few clicked points in the p.4 figure. The lecture's main setting is "Image-level labels at training time" → "Pixel-level labels at test time" (p.19).
- **vs. previous / key difference**: L08 asked for boxes from image labels (WSOD). L09 asks for pixel masks from weaker labels (WSSS), so the output is even denser than the label.
- **Keywords**: semantic segmentation, pixel-level labels, image-level labels, directly supervised, indirectly supervised, training time vs test time
- **Definition ↔ trap**: "Weakly supervised segmentation is indirectly supervised: image-level labels at training time, pixel-level labels at test time." ↔ "Weakly supervised segmentation uses pixel-level labels at training time and image-level labels at test time."
- 한 줄: train엔 image label, test엔 pixel mask. 정답 mask를 직접 안 주니까 indirectly supervised야.

#### Two sources of information: priors and hints (Slide credit: Rodrigo Benenson, p.5–17)
- **Problem**: With only weak labels, the model needs extra information to decide which pixels belong to the object.
- **Core idea**: "There are only two sources of information". **Priors**: "what do I believe to be true independent of any particular image sample" — size, location, shape (p.6), number of instances, contrast (p.7), boundaries, saliency (p.8), class distribution (p.9), similarity across images, similarity with external images (p.10). **Hints**: "indirect supervision received for each image" — image (video) label, image caption (p.11–12), click inside object (p.13), object bounding boxes (p.14), object extreme points (p.15), scribbles (p.16), eye gaze (p.17).
- **vs. previous / key difference**: A prior is shared by all images and comes from belief. A hint is given per image by an annotator. Hints range from cheap (image label) to rich (boxes, scribbles).
- **Keywords**: priors, hints, size, location, shape, number of instances, contrast, boundaries, saliency, class distribution, similarity across images, image caption, click, extreme points, scribbles, eye gaze
- **Definition ↔ trap**: "Hints are indirect supervision received for each image." ↔ "Priors are indirect supervision received for each image."
- 한 줄: prior는 사진과 상관없이 믿는 것, hint는 사진마다 받는 간접 label.

#### Gray box approach: gradient-based saliency (Simonyan et al., ICLR 2014 workshop ← slide엔 "Simonyan et. al 2014", p.20–21, 23)
- **Problem**: An image classifier says what is in the image, but not where.
- **Core idea**: "Utilize the internal information in the network": it "identifies which areas the model focused on for classification by leveraging the activations or gradients within the CNN". Step 1) Train an ImageNet classifier. Step 2) "Calculate how much the score changes when the input is slightly modified", i.e. the gradient of the class score with respect to the input pixels.
- **vs. previous / key difference**: It needs no change to the network, but "it is extremely noisy" because "the gradient measures sensitivity at the level of individual pixels, resulting in a 'salt-and-pepper' appearance". Key issue (p.23): "Asking apples from the oranges tree" — the "Classifier was not built nor trained to segment objects", so it "highlights only the specific regions necessary for classification rather than the entire object".
- **Keywords**: gray box, internal information, activations, gradients, score change, saliency map, salt-and-pepper, extremely noisy, asking apples from the oranges tree
- **Definition ↔ trap**: "Gradient saliency is noisy because the gradient measures sensitivity at the level of individual pixels." ↔ "Gradient saliency is smooth because the gradient measures sensitivity at the level of whole objects."
- 한 줄: 입력을 살짝 바꿀 때 class 점수가 얼마나 변하나(gradient)로 위치를 보는데, 픽셀 단위라 salt-and-pepper처럼 지저분해.

#### SmoothGrad (Smilkov et al., arXiv 2017, p.22)
- **Problem**: A single input gradient is noisy.
- **Core idea**: Step 1) Train a standard CNN classifier. Step 2) Add small random noise to the images. Step 3) Calculate the gradient for each. Step 4) Average. "calculating the gradient multiple times and averaging it removes noise and yields a more meaningful saliency map."
- **vs. previous / key difference**: Plain gradient saliency uses one gradient of the clean image. SmoothGrad averages gradients over several noisy copies. It is still a gray box method on an unchanged classifier, so the p.23 limitation remains.
- **Keywords**: SmoothGrad, random noise, gradient for each noisy copy, average, saliency map
- **Definition ↔ trap**: "SmoothGrad adds noise to the input several times and averages the resulting gradients." ↔ "SmoothGrad averages several noisy images first and then computes a single gradient."
- 한 줄: 입력에 noise를 섞은 사본마다 gradient를 구해서 평균 내면 깨끗해져.

#### White box approach: CAM, Class Activation Mapping (Zhou et al., CVPR 2016 ← slide엔 "Zhou et al. 2016", p.24–26)
- **Problem**: Gray box maps need separate gradient calculations and are noisy.
- **Core idea**: Change the classifier so that it keeps spatial information. Step 1) "eliminate the FC layer". Step 2) "add GAP (Global Average Pooling)". Step 3) "add a 1×1 convolution that performs a linear combination of the features". The class activation map of a class is the weighted sum of the last conv feature maps with that class's weights: $M_c(x,y)=\sum_k w_k^c f_k(x,y)$ (p.25 figure: $w_1\ast$map$_1 + w_2\ast$map$_2 + \dots + w_n\ast$map$_n$; formula from the paper). "since the network itself is designed to preserve spatial information, activation maps can be obtained directly without the need for separate gradient calculations." It is "Designed to Generate Per-Class Maps" (p.26: one CAM for each of the top 5 predicted classes).
- **vs. previous / key difference**: Gray box reads gradients of an unchanged classifier. White box (CAM) redesigns the network (GAP + linear layer), so the map comes from a forward pass with no gradient computation.
- **Keywords**: white box, eliminate FC layer, GAP, 1×1 convolution, linear combination, class weights $w_k^c$, class activation map, per-class maps, no separate gradient calculation
- **Definition ↔ trap**: "CAM obtains activation maps directly because the network is designed to preserve spatial information." ↔ "CAM needs a separate backward pass to compute input gradients for each class."
- 한 줄: FC를 빼고 GAP + 선형 결합을 넣으면, class weight로 feature map을 더하는 것만으로 class별 위치 지도가 나와.

#### Seeds: thresholding class maps into training data (p.27)
- **Problem**: A CAM is a continuous score map, not a pixel label.
- **Core idea**: "The Class Maps are Thresholded, to Generate Training Data". "High value scores ➝ Foreground areas", "Low value scores ➝ Background areas". "(This confident regions are commonly named 'seeds')" (slide grammar kept).
- **vs. previous / key difference**: The seeds become pseudo labels for training a segmentation network. Only confident regions are labeled (interpretation: the in-between pixels are left out).
- **Keywords**: threshold, foreground, background, confident regions, seeds, training data, pseudo label
- **Definition ↔ trap**: "High CAM scores become foreground seeds and low scores become background." ↔ "Low CAM scores become foreground seeds and high scores become background."
- 한 줄: CAM을 threshold로 잘라서 확실한 곳만 foreground/background seed로 써.

#### CAM's two problems: "Good, but not good enough" (p.28–29)
- **Problem**: CAM seeds are useful but limited: "1. Resulting masks are not sharp" and "2. Focused on discriminative area only".
- **Core idea**: The following slides (p.29–41) are a list of fixes. For problem 1: "Increase top layers resolution" (p.29), "Use boundary-aware propagation" (p.30, SEC), "Use transfer learning" with saliency cues (p.33, Oh), "Enforce equivariance" (p.35, SEAM). For problem 2: "Let us play peek-a-boo!" (Hide-and-Seek, p.37–38) and "Let us play adversarial peek-a-boo!" (adversarial erasing, p.39–41).
- **vs. previous / key difference**: Problem 1 is about **boundaries** (masks are coarse). Problem 2 is about **extent** (only the most discriminative part, such as a face, lights up).
- **Keywords**: not sharp, discriminative area only, top layers resolution, boundary-aware propagation, transfer learning, equivariance, peek-a-boo, adversarial peek-a-boo
- **Definition ↔ trap**: "CAM masks are not sharp and focus on the discriminative area only." ↔ "CAM masks have sharp boundaries but cover too much background."
- 한 줄: CAM의 두 문제는 경계가 흐린 것과 가장 눈에 띄는 부분만 잡는 것.

#### SEC: Seed, Expand, Constrain (Kolesnikov & Lampert, ECCV 2016, p.30–32)
- **Problem**: CAM limitations: "Activation maps are small and coarse", "Focus only on discriminative regions", "Poor object boundaries".
- **Core idea**: "Three complementary losses" train a segmentation CNN. **1. Seeding Loss**: "Use CAM as reliable foreground seeds", "Preserve highly confident object regions". **2. Expansion Loss**: "Encourage predictions to expand beyond the seeds", "Use image-level labels via Global Weighted Rank Pooling (GWRP)". **3. Constrain-to-boundary Loss**: "Refine predictions using CRF", "Encourage masks to align with image boundaries". Summary: "Seed: Preserve reliable CAM activations. Expand: Grow seeds to cover the entire object using image-level supervision. Constrain: Refine object boundaries with CRF-based regularization." (paper) Foreground seeds come from CAM (scores above 20% of the maximum) and background seeds from a saliency map; the seeding loss is a cross-entropy on seed pixels only, and the constrain loss is a KL divergence between the network output and its CRF-refined version. The CRF has no learned weights, but the paper back-propagates through it ("For computing gradients of the fully-connected CRF we employ the procedure from [34]", Sec. 3.2); the gray arrows on p.30 also pass through the CRF.
- **vs. previous / key difference**: Using thresholded CAM seeds directly as labels keeps both CAM problems. SEC keeps the seeds, grows them with image-level labels, and pulls the result to image boundaries in one training objective. The slide files it under "Use boundary-aware propagation" (p.30).
- **Keywords**: seeding loss, expansion loss, constrain-to-boundary loss, GWRP, CRF, weak localization, segmentation CNN, downscale
- **Definition ↔ trap**: "SEC's expansion loss uses image-level labels via GWRP to grow predictions beyond the seeds." ↔ "SEC's expansion loss uses a CRF to shrink predictions to the seeds."
- 한 줄: seed는 지키고(Seed), image label로 키우고(Expand), CRF로 경계에 맞춰(Constrain).

#### Saliency-guided weak supervision (Oh et al., CVPR 2017, p.33–34)
- **Problem**: CAM "Provides class-specific localization" but "Covers only discriminative regions".
- **Core idea**: Saliency "Provides complete object extent" but is "Class-agnostic foreground estimation". "Combine CAM and saliency to generate class-specific pseudo masks". **Seeder**: "Generate class-specific seeds from CAM." **Saliency**: "Estimate foreground regions." **Guide Labeller**: "Fuse both cues into pseudo masks." **Segmentation CNN**: "Learn dense pixel-wise predictions." The slide calls this "Use transfer learning" and "We could also use saliency cues" (p.33). (paper) The saliency model is trained on a separate dataset, so its knowledge is transferred.
- **vs. previous / key difference**: SEC grows seeds inside the same task. Oh borrows an external, class-agnostic cue (saliency) that already knows the full object extent, and lets CAM decide the class.
- **Keywords**: saliency, class-agnostic, class-specific, seeder, guide labeller, segmenter convnet, dense classifier loss, transfer learning, pseudo masks
- **Definition ↔ trap**: "CAM gives class-specific localization, and saliency gives a class-agnostic foreground with complete extent." ↔ "Saliency gives class-specific localization, and CAM gives a class-agnostic foreground with complete extent."
- 한 줄: class는 CAM이, 물체 범위는 saliency가 맡아서 합친 pseudo mask로 학습해.

#### SEAM: equivariance for CAMs (Wang et al., CVPR 2020, p.35–36)
- **Problem**: "CAMs are often coarse and inconsistent", and "Small input transformations can produce different activation maps".
- **Core idea**: "Activation maps should be equivariant to the input" (p.35). "Key idea: Enforce Equivariance": "Apply geometric transformations (e.g., affine transform)", "Predict CAMs for both original and transformed images", "Enforce consistency after compensating for the transformation". A Siamese network shares the CNN. **PCM** (Pixel Correlation Module) "propagates CAM activations to other pixels within the same object by utilizing pixel-wise affinity among CNN features". Training objective: **Classification Loss** ("Preserve image-level classification performance"), **Equivariant Regularization (ER) Loss** ("Encourage transformation-consistent CAMs"), **Cross-view Consistency (ECR) Loss** ("Align CAMs across different views"). (paper) SEAM = Self-supervised Equivariant Attention Mechanism, and the paper names ECR "Equivariant Cross Regularization" ← slide엔 "Cross-view Consistency (ECR)". The figure uses L1 for ER and ECR.
- **vs. previous / key difference**: Plain CAM is trained only by classification, so nothing ties the maps of a transformed image together. SEAM adds a self-supervised constraint: transform the CAM of the original and it should match the CAM of the transformed image. This is the L08 equivariance prior (L08 p.29) applied to CAMs.
- **Keywords**: equivariance, affine transform, Siamese network, PCM, pixel-wise affinity, classification loss, ER loss, ECR loss, L1, transformation-consistent
- **Definition ↔ trap**: "SEAM enforces that CAMs are consistent after compensating for the transformation (equivariance)." ↔ "SEAM enforces that CAMs stay unchanged when the image is transformed (invariance)."
- 한 줄: 사진을 변환하면 CAM도 똑같이 변환돼야 한다(equivariance)는 규칙으로 CAM을 일관되게 만들어.

#### Hide-and-Seek (Singh & Lee, ICCV 2017 ← slide엔 "Wang et al. CVPR 2020", p.37–38)
- **Problem**: "CAM activates only the most discriminative regions."
- **Core idea**: "Solution (Hide-and-Seek)": "Randomly hide image patches during training", "Encourage learning from the remaining visible parts", "Generate more complete localization maps". Effect: "The classifier is forced to discover other visible object parts" and "CAM gradually covers a larger portion of the object." "Random hiding encourages the network to discover complementary object parts instead of relying on a single discriminative region." The figure splits the training image into S×S patches and hides a different random set each epoch. p.38 "Let us play peek-a-boo!" compares AlexNet-GAP and HaS heatmaps. (paper) Hiding is used only in training; the full image is used at test time.
- **vs. previous / key difference**: The fixes for problem 1 refine boundaries. Hide-and-Seek attacks problem 2 by changing the training input, so the network cannot rely on one part.
- **Keywords**: Hide-and-Seek (HaS), randomly hide patches, peek-a-boo, complementary object parts, more complete localization maps, grid of patches, epoch
- **Definition ↔ trap**: "Hide-and-Seek randomly hides image patches during training." ↔ "Hide-and-Seek hides the most discriminative region found by the CAM."
- 한 줄: 학습 때 patch를 무작위로 가리면 얼굴이 가려진 날엔 몸통을 보고 맞혀야 해서 CAM이 넓어져.

#### Adversarial erasing, "adversarial peek-a-boo" (Wei et al., CVPR 2017 ← slide엔 "Wei et al. 2017", p.39–41)
- **Problem**: CAM covers only the discriminative area, and random hiding does not choose what to hide.
- **Core idea**: "adversarial instead of random erasures". A classification network produces a heatmap $H_t$. It is thresholded into the mined region $F_t$, which is erased from the image: $I_{t+1}$ = $I_t$ minus $F_t$. The network is trained again on the erased image, so it must find another region (steps $t = 1, 2, 3$, "Recurrent"). The mined regions together form "The Mined Object Region" (p.39 figure). p.40 shows AE-Step1–3 heatmaps and the erased regions in gray; p.41 shows segmentation results (AE-PSL). (paper) Each step trains a new classification network, and PSL = prohibitive segmentation learning.
- **vs. previous / key difference**: Hide-and-Seek hides **random** patches. Adversarial erasing erases **the region the classifier currently relies on most**, step by step.
- **Keywords**: adversarial erasing, AE, peek-a-boo, heatmap, threshold, mined region, erase, recurrent, AE-Step, mined object region, AE-PSL
- **Definition ↔ trap**: "Adversarial erasing removes the currently most discriminative region so the classifier must discover new object regions." ↔ "Adversarial erasing removes random patches so the classifier must discover new object regions."
- 한 줄: 지금 제일 센 곳을 지우고 다시 학습하길 반복해서, 지운 영역을 모으면 물체 전체가 돼.

#### Takeaways: from image-level to pixel-level labels (p.42)
- **Problem**: Which of these ideas actually work?
- **Core idea**: "Training a classifier, and hoping to get segmentations out of it ✗". "Using spatial continuity priors will help, but to a limit ~". "The training can be modified to promote covering the object extent ✔". "With careful changes it is possible to obtain significant improvements ~". "⇒ How far can we go from image-level to pixel-level labels ?"
- **vs. previous / key difference**: Reading a plain classifier (gray box) is marked ✗. Changing training to cover the object extent (e.g. hiding, erasing) is marked ✔.
- **Keywords**: classifier, spatial continuity priors, object extent, careful changes
- **Definition ↔ trap**: "Training a classifier and hoping to get segmentations out of it does not work well." ↔ "Training a plain classifier is enough to get good segmentations out of it."
- 한 줄: 분류기만 학습하고 segmentation을 기대하면 안 되고, 물체 전체를 덮도록 학습을 바꿔야 해.

#### Webly supervised learning (Shen et al., CVPR 2018, p.43–52)
- **Problem**: Manual labels are expensive, and CAMs from a target-domain classifier are incomplete.
- **Core idea**: Webly supervised learning is "A learning method that uses data collected from the web as supervision". It collects images "directly from web search engines", "Enables large-scale learning without manual annotation", but "Labels are noisy, ambiguous, and incomplete", so "Learning must be robust to noisy supervision" (p.44). Pipeline (p.45): 1. Train an image classifier (Model-T) on clean image-level labels. 2. Use Model-T to filter noisy web images. 3. Train Web-SEC (pixel labeler) using the filtered web images. 4. Apply Web-SEC to generate pixel labels for the target images. 5. Use Model-T to produce CAMs on the target images. 6. Fuse the two label sources and train the final segmentation model. Key point (p.47): "web images are much cleaner—typically containing a single centered object—so approximate segmentation masks can be generated automatically", and the network trained on them "learns generic object shapes and can later be transferred to the target dataset".
- **vs. previous / key difference**: "Unlike CAM-based methods, Web-SEC actually learns a pixel classifier." The figure shows two domains: Model-T (target) **filters** the web data, and Model-W (web) **enhances** the target side. (paper) The paper calls this bi-directional transfer between the target domain and the web domain.
- **Keywords**: webly supervised, web search engines, noisy labels, Model-T, Model-W, Web-SEC, pixel labeler, filter, enhance, single centered object, fuse
- **Definition ↔ trap**: "Webly supervised learning trades annotation cost for label noise." ↔ "Webly supervised learning trades label noise for higher annotation cost."
- 한 줄: 웹 사진은 물체 하나가 가운데 있어서 mask를 자동으로 얻기 쉬워. 거기서 pixel labeler를 배우고 target 사진의 CAM과 합쳐.

#### Boxly supervised learning: "Convnets are robust to noise" (Khoreva et al., CVPR 2017, p.53–61)
- **Problem**: "From box-level labels to pixel-level instance labels" (p.54). "What we want" is object masks; "What we have" is boxes (p.55).
- **Core idea**: "Convnets are robust to noise. Let us use that !" "Initially, training is performed using a noisy mask, but the CNN's predictions are then used as pseudo-labels for retraining." "Segmentation networks are robust to noisy supervision", "Approximate masks are sufficient for training", "Iteratively refine pseudo masks during training", "Better masks lead to better segmentation" (p.56). Starting from filled rectangles, the output improves after 1, 5, 10 training rounds (p.57). Better initial masks (20% Bbox, MCG, Grabcut+, MCG ∩ Grabcut+) start higher (p.58–59). "We reach ~95% of full supervision by using GrabCut++" (p.60) ← slide 그래프 범례는 "Grabcut+", "MCG ∩ Grabcut+"(표는 "M ∩ G+"). (paper) The ~95% is reached with MCG ∩ GrabCut+. (paper) Between rounds the predictions are cleaned with the boxes (outside-box pixels reset, CRF), and GrabCut+ uses HED boundaries instead of color differences.
- **vs. previous / key difference**: Image-level methods must find the object's location and extent. Box labels already give location, so the remaining job is the shape inside the box, done by recursive self-training.
- **Keywords**: boxly supervised, box-level labels, noisy mask, pseudo-labels, retraining, training rounds, rectangles, 20% Bbox, MCG, GrabCut+, ~95% of full supervision
- **Definition ↔ trap**: "Box-supervised training starts from noisy masks and uses the CNN's own predictions as pseudo labels for retraining." ↔ "Box-supervised training uses the boxes as fixed labels in every round and never updates them."
- 한 줄: 네모를 대충 mask로 써서 학습하고, 예측을 다음 round의 label로 쓰길 반복해. 시작 mask를 좋게 하면(슬라이드 제목은 GrabCut++, 그림은 MCG ∩ GrabCut+) full supervision의 ~95%.

#### Summary (p.61)
- **Problem**: What to remember from WSSS.
- **Core idea**: "Many priors and hints have been explored". "Exploit all available priors, explore new information sources". "Do not underestimate the power of transfer learning (and webly supervision)". "Current techniques are quite effective, and ride the wave of better network architectures".
- **vs. previous / key difference**: Not a method. It ties the lecture together: priors and hints (p.5–17), CAM fixes (p.28–41), web data (p.43–52), boxes (p.53–60).
- **Keywords**: priors, hints, new information sources, transfer learning, webly supervision, better network architectures
- **Definition ↔ trap**: "Do not underestimate the power of transfer learning (and webly supervision)." ↔ "Transfer learning and webly supervision add little to weakly supervised segmentation."
- 한 줄: prior와 hint를 다 쓰고, transfer learning과 web data를 얕보지 마.

#### CD-CLIP: class distributions for CAM completeness and boundaries (Yang et al., CVPR 2026, p.62–74)
- **Problem**: "Beyond CAM" (p.62): "CAM activates only the most discriminative regions" and "CAM boundaries are coarse and inaccurate". Problem 1 "Incomplete activation". Problem 2 "Inaccurate boundaries": "Affinity propagation may spread activation" and "Foreground activation can cross class boundaries" (p.64). Conventional affinity "Uses feature or attention similarity between patches", but "Similar appearance does not always imply the same class" (p.65).
- **Core idea**: "A patch is characterized not only by its target-class score, but by its distribution over all classes" (target classes + super-classes, p.65, 67). **CDA** (Class Distribution Aware, p.66 figure): compare two patch distributions with JS divergence ("Small JS divergence: similar class distributions"), "Correct the original affinity", "Propagate CAM activation" (head → body). $S_d = \frac{1 - D_{JS}(\mathrm{Softmax}(S_a)\,\|\,\mathrm{Softmax}(S_a)^\top)}{\tau}$, $A_d = \mathrm{Softmax}(S_d)\cdot A_{init}$, $M^c_{re} = M^c_{init}\cdot (A_d)^t$ (p.69). **SBE** (Super-class Boundary Exploration, p.66 figure): "Use the CLIP super-class CAM as a semantic seed and average the corresponding DINO features": $f_s^n = \mathrm{CAP}(M_s^n \odot F_d)$, then "Compare the prototype with every DINO patch": $M_p^n = \mathrm{ReLU}(\mathrm{CosSim}(f_s^n, F_d))$ (p.70). Loss $\mathcal{L} = \mathcal{L}_{seg} + \lambda_1\mathcal{L}_b$, with $\mathcal{L}_b = \mathrm{Dice}(P, Y_s)$ only "if $|C_p| \ge 2$" (p.72). ← 슬라이드는 $S_a$, $A_{init}$, $\tau$, $t$, $M^c_{init}$, $F_d$, $M_s^n$를 정의하지 않아. 해석(p.66 그림 기준): $A_{init}$ = MHSA에서 온 initial affinity, $\tau$ = temperature, $t$ = 전파 횟수, $F_d$ = DINO features, $M_s^n$ = super-class CAM. 시험엔 식 전체보다 "JS로 affinity를 보정한 뒤 곱해서 CAM을 전파"만.
- **vs. previous / key difference**: Conventional affinity compares appearance; CDA compares class distributions, so it connects same-class patches without leaking across classes. "CDA significantly improves the CAM mIoU from 70.1 to 80.3"; SBE helps the final segmentation more than the CAM "because SBE contributes to boundary learning within the decoder" (p.73). "Better WSSS requires both knowing which patches belong together and knowing where their relationship should stop." (p.74)
- **Keywords**: CLIP, DINO, class distribution, target classes, super-classes, JS divergence, affinity, MHSA, CDA, super-class prototype, prototype CAM, CAP (class average pooling), SBE, boundary-enhanced loss, Dice, frozen encoders
- **Definition ↔ trap**: "CD-CLIP compares the class distributions of two patches with JS divergence to decide their affinity." ↔ "CD-CLIP compares only the target-class scores of two patches to decide their affinity."
- 한 줄: patch를 target class 점수 하나가 아니라 class 분포 전체로 비교해서 CAM을 넓히고(CDA), DINO prototype으로 어디서 멈출지 정해(SBE).

#### Learning from the Web for weakly-supervised incremental segmentation (Liu et al., ECCV 2024, p.75–85)
- **Problem**: "Can a segmentation model learn new classes from the web without storing its original training images?" In the new scenario "Only the initial model is retained", "Original training images are no longer accessible", and "New and old classes must be learned from web images" (p.77). "Three challenges": web images have a different distribution, "Search queries provide noisy single-class labels", and "Learning new classes causes forgetting of old classes" (p.78).
- **Core idea**: "Language is used for querying, labeling, filtering, and rehearsal" (p.80). ① Retrieve web images (query with new class names; retrieve rehearsal images using stored captions). ② Filter noisy web images ("Domain filtering removes out-of-distribution images", "Caption filtering removes semantically irrelevant images"). ③ Generate supervision ("Captions are converted into multi-label supervision", "The previous model produces pixel-level pseudo-labels"). ④ Update the model (learn new classes, "Rehearse old classes without storing original images"). "The class name retrieves images; the caption determines their labels" (p.81). For old classes: "Store Captions, Not Images": Old image → stored caption q' → web search → rehearsal image (p.82), then verify by captioning the retrieved image (q'') and comparing nouns (p.83).
- **vs. previous / key difference**: Conventional WILSS keeps curated dataset images and gives only image-level labels for new classes, with a localizer making pseudo labels (p.79). Here "Web images replace curated incremental data, while captions replace reliable manual image labels" (p.79), and "The web is not only a source of new data. It can serve as an external semantic memory" (p.85). (paper) WILSS = Weakly Incremental Learning for Semantic Segmentation (task from Cermelli et al., CVPR 2022), and the domain filter is a Fourier-based domain discriminator.
- **Keywords**: incremental learning, WILSS, forgetting, rehearsal, web retrieval, domain filtering, caption filtering, caption labeling, multi-label supervision, stored captions, external memory, pseudo labels, 10–10 / 15–5 / 10–1
- **Definition ↔ trap**: "The method stores captions of old images, not the images, and uses them to retrieve rehearsal images from the web." ↔ "The method stores the original training images and replays them for rehearsal."
- 한 줄: 옛 사진 대신 caption만 저장해서 웹에서 비슷한 사진을 다시 찾아 복습하고, 새 class도 웹 사진 + caption label로 배워.

#### WISH: heterogeneous weak labels in SAM's prompt space (Kweon & Yoon, CVPR 2025 ← slide에 cite 없음, p.86–95)
- **Problem**: "Under a fixed annotation budget, should every image receive the same type of weak label?" (p.86). In homogeneous supervision "Every training image uses the same weak label type": "Tags are cheap but provide no localization", "Boxes are informative but expensive", "Points provide an intermediate level of supervision" (p.87).
- **Core idea**: Heterogeneous supervision: "Different images may use different weak label types" ($Y_i^{\mathrm{Hetero}} = Y_i^t$ or $Y_i^p$ or $Y_i^b$), and "A single model learns from all label types" (p.87). "SAM's prompt space serves as a common representation for heterogeneous supervision": point prompt → point latent, box prompt → box latent, tag → converted point prompt, and the model predicts $\hat{Z} = \mathcal{H}_{\mathrm{prompt}}(Q)$ from Mask2Former object queries (p.88). Matching: class ($\mathrm{Cost}_{\mathrm{cls}}^{i,j} = \hat{y}^i_{\mathrm{cls}}(c^j)$), prompt ($\mathrm{KLD}(\hat Z^i, Z^j)$), mask ("Select the SAM candidate mask closest to the predicted instance mask", $\min_{n\in\{1,2,3\}} d(\hat M^i, M^{j,n}_{\mathrm{SAM}})$), combined $\alpha\,\mathrm{Cost_{cls}} + \beta\,\mathrm{Cost_{prompt}} + \gamma\,\mathrm{Cost_{mask}}$; "Hungarian matching assigns predictions to weakly labeled instances" (p.89–90). Tags become prompts through CAM peaks: "Local maxima are extracted from each class activation map", "Each selected CAM peak becomes a positive point prompt", "SAM generates an instance mask from each point" (p.91). The CAM comes from a trainable head, $A = \mathcal{H}_{\mathrm{cam}}(F)$, and self-enhancement trains it with $\mathcal{L}_{\mathrm{self}} = \mathcal{L}_{\mathrm{bce}}(A, \hat M_c)$, using the model's predicted masks as the target (p.91, 94). ← slide p.91 step 3("Improve CAMs with Model Predictions") 본문은 step 2 bullet 3개를 그대로 복붙한 슬라이드 오류야. 실제 내용은 $\mathcal{L}_{\mathrm{self}}$ 식과 p.94 "Current instance masks refine the CAM".
- **vs. previous / key difference**: "Conventional WSIS" "Chooses one weak label type for the entire dataset" and "Treats tags, points, and boxes as separate settings". WISH "Combines different label types in one model" and "Achieves better cost–performance trade-offs" (p.95). "Cheap labels provide breadth, while strong labels provide spatial precision" (p.93). "The optimal supervision strategy may be a portfolio of weak labels, not a single annotation type." (p.95)
- **Keywords**: heterogeneous supervision, homogeneous supervision, tags, points, boxes, SAM, prompt space, prompt latent, Mask2Former object queries, class matching, prompt matching, mask matching, Hungarian matching, CAM peaks, self-enhancement, annotation budget, WSIS
- **Definition ↔ trap**: "WISH uses SAM's prompt space as a common representation so that tags, points, and boxes can supervise one model." ↔ "WISH trains a separate model for each weak label type and picks the best one."
- 한 줄: tag·point·box를 SAM의 prompt 공간으로 옮겨서 한 모델이 섞인 label을 다 배워. 같은 예산이면 섞는 게 더 나을 수 있어.

### Must-know definitions & distinctions
- **Semantic segmentation**: predict a class for every pixel. The output of WSSS at test time ("Pixel-level labels at test time", p.19).
- **Fully vs weakly supervised**: **directly** supervised with pixel masks vs **indirectly** supervised with weaker labels (p.4).
- **Prior vs hint**: "what do I believe to be true independent of any particular image sample" vs "indirect supervision received for each image" (p.5).
- **Hint types**: image (video) label, image caption, click inside object, object bounding boxes, object extreme points, scribbles, eye gaze (p.11–17).
- **Gray box vs white box**: read activations or gradients of a normal classifier vs redesign the network so maps come out directly (CAM) (p.20, 24–25).
- **Gradient saliency**: how much the class score changes when the input is slightly modified; "extremely noisy", "salt-and-pepper" (p.21).
- **SmoothGrad**: add small random noise, compute the gradient for each copy, average (p.22).
- **GAP (Global Average Pooling)**: averages each feature map over all positions into one number, replacing the FC layer in CAM (p.24).
- **CAM**: the class-weighted sum of the last conv feature maps; one map per class, no separate gradient calculation (p.25–26).
- **Seeds**: confident regions of a thresholded class map; high scores → foreground, low scores → background (p.27).
- **Not sharp vs discriminative only**: CAM's boundary problem vs its extent problem (p.28).
- **Seeding vs expansion vs constrain-to-boundary loss (SEC)**: keep confident CAM regions vs grow beyond the seeds with image labels (GWRP) vs align with image boundaries via CRF (p.31).
- **GWRP**: Global Weighted Rank Pooling, the pooling that connects pixel predictions to image-level labels in the expansion loss (p.31). (paper) It sorts scores and weights them with decaying weights, so it lies between max and average pooling.
- **CRF**: refines predictions so that masks follow image boundaries; used in SEC's constrain-to-boundary loss (p.31).
- **Saliency (Oh) vs CAM**: class-agnostic foreground with complete object extent vs class-specific localization of discriminative regions (p.34).
- **Gradient saliency map vs saliency (Oh)**: a map of input sensitivity of a classifier (p.21–22) vs a foreground estimate from a saliency model (p.34). Same word, different things.
- **Equivariance (SEAM)**: CAMs should change along with the input transformation; "Enforce consistency after compensating for the transformation" (p.35–36).
- **PCM**: "propagates CAM activations to other pixels within the same object by utilizing pixel-wise affinity among CNN features" (p.36).
- **ER loss vs ECR loss**: transformation-consistent CAMs vs CAMs aligned across different views (p.36).
- **Hide-and-Seek vs adversarial erasing**: **random** patch hiding vs erasing the **currently mined discriminative** region, step by step (p.37, 39).
- **Webly supervised learning**: uses data collected from the web as supervision; trades annotation cost for label noise (p.44).
- **Model-T vs Web-SEC**: image classifier trained on clean target image-level labels (filters web images, makes CAMs) vs pixel labeler trained on filtered web images (p.45).
- **Boxly supervised learning**: from box-level labels to pixel-level labels by recursive retraining on the network's own predictions (p.54–56).
- **Training round (Khoreva)**: one cycle of training on the current masks and replacing them with the network's predictions (p.56–57).
- **Class distribution (CD-CLIP)**: a patch's scores over all target classes and super-classes, not just the target-class score (p.65, 67).
- **JS divergence**: measures how different two distributions are; small = similar class distributions (p.69).
- **CDA vs SBE**: completes the CAM by distribution-based affinity vs stops it at boundaries with DINO super-class prototypes (p.74).
- **Super-class prototype**: the average of DINO features inside the CLIP super-class CAM, $f_s^n = \mathrm{CAP}(M_s^n \odot F_d)$ (p.70).
- **Boundary-enhanced loss**: $\mathcal{L}_b = \mathrm{Dice}(P, Y_s)$ only when $|C_p| \ge 2$, i.e. two or more distinct super-classes (p.72).
- **Conventional WILSS vs web setting (Liu)**: curated images for new classes and image labels vs web images, caption labels, and no stored original images (p.79).
- **Rehearsal by captions**: store captions of old images, search the web with them, keep results whose new caption has nouns of a similar semantic family (p.82–83).
- **Domain filtering vs caption filtering**: removes out-of-distribution images vs removes semantically irrelevant images (p.80).
- **Homogeneous vs heterogeneous supervision**: every image has the same weak label type vs different images may use different types (p.87).
- **SAM prompt space (WISH)**: the shared latent where points and boxes are mapped; tags are converted to point prompts first (p.88).
- **Class vs prompt vs mask matching**: what it is vs where it is vs which SAM mask fits it best (p.90).
