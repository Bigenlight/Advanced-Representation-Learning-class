# 09. Weakly-Supervised Learning II (Weakly-Supervised Semantic Segmentation) — Core Ideas & Question Pool

> 기말 형식 기준이야 (`exam/PLAYBOOK.md` 7장, 형식은 공지 전까지 중간과 같다고 가정).
> T/F엔 전부 `Why (EN)`을 붙였고, False엔 `Corrected (EN)`도 붙였어.
> 코드 빈칸은 여기 없어. `exam/pseudocode_final.md`의 L09 카드(WS-9–)에 있어.
> 슬라이드 cite는 웹으로 확인했어. 실제 표기와 다르면 `← slide엔 …`으로 남겼어. 시험 지문은 슬라이드 표기를 따를 수 있어. 특히 Hide-and-Seek(p.37–38)은 slide엔 "Wang et al. CVPR 2020"인데 실제론 Singh & Lee, ICCV 2017이고, WISH(p.86–95)는 슬라이드에 cite가 없어 (Kweon & Yoon, CVPR 2025).
> "(paper)" 표시는 슬라이드엔 없고 논문 원문에서 확인한 내용이야.
> C·D는 작성자와 다른 Sonnet 2명이 정답을 가리고 먼저 풀어 본 뒤 대조했어 (38문항 전부 일치). A·B·E·F도 별도 검수를 거쳤어 (2026-11-09).

## A. Core ideas (one block per method/paper/concept, in lecture order)

### Fully vs weakly supervised semantic segmentation (Slide credit: Rodrigo Benenson, p.3–4, 19)
- **Problem**: Semantic segmentation needs a class for every pixel. Pixel masks for training are the most expensive kind of label.
- **Core idea**: "Fully supervised → **Directly** supervised": the training label is the pixel mask itself. "Weakly supervised → **Indirectly** supervised": the training label is weaker than the output, e.g. a few clicked points in the p.4 figure. The lecture's main setting is "Image-level labels at training time" → "Pixel-level labels at test time" (p.19).
- **vs. previous / key difference**: L08 asked for boxes from image labels (WSOD). L09 asks for pixel masks from weaker labels (WSSS), so the output is even denser than the label.
- **Keywords**: semantic segmentation, pixel-level labels, image-level labels, directly supervised, indirectly supervised, training time vs test time
- **Definition ↔ trap**: "Weakly supervised segmentation is indirectly supervised: image-level labels at training time, pixel-level labels at test time." ↔ "Weakly supervised segmentation uses pixel-level labels at training time and image-level labels at test time."
- 한 줄: train엔 image label, test엔 pixel mask. 정답 mask를 직접 안 주니까 indirectly supervised야.

### Two sources of information: priors and hints (Slide credit: Rodrigo Benenson, p.5–17)
- **Problem**: With only weak labels, the model needs extra information to decide which pixels belong to the object.
- **Core idea**: "There are only two sources of information". **Priors**: "what do I believe to be true independent of any particular image sample" — size, location, shape (p.6), number of instances, contrast (p.7), boundaries, saliency (p.8), class distribution (p.9), similarity across images, similarity with external images (p.10). **Hints**: "indirect supervision received for each image" — image (video) label, image caption (p.11–12), click inside object (p.13), object bounding boxes (p.14), object extreme points (p.15), scribbles (p.16), eye gaze (p.17).
- **vs. previous / key difference**: A prior is shared by all images and comes from belief. A hint is given per image by an annotator. Hints range from cheap (image label) to rich (boxes, scribbles).
- **Keywords**: priors, hints, size, location, shape, number of instances, contrast, boundaries, saliency, class distribution, similarity across images, image caption, click, extreme points, scribbles, eye gaze
- **Definition ↔ trap**: "Hints are indirect supervision received for each image." ↔ "Priors are indirect supervision received for each image."
- 한 줄: prior는 사진과 상관없이 믿는 것, hint는 사진마다 받는 간접 label.

### Gray box approach: gradient-based saliency (Simonyan et al., ICLR 2014 workshop ← slide엔 "Simonyan et. al 2014", p.20–21, 23)
- **Problem**: An image classifier says what is in the image, but not where.
- **Core idea**: "Utilize the internal information in the network": it "identifies which areas the model focused on for classification by leveraging the activations or gradients within the CNN". Step 1) Train an ImageNet classifier. Step 2) "Calculate how much the score changes when the input is slightly modified", i.e. the gradient of the class score with respect to the input pixels.
- **vs. previous / key difference**: It needs no change to the network, but "it is extremely noisy" because "the gradient measures sensitivity at the level of individual pixels, resulting in a 'salt-and-pepper' appearance". Key issue (p.23): "Asking apples from the oranges tree" — the "Classifier was not built nor trained to segment objects", so it "highlights only the specific regions necessary for classification rather than the entire object".
- **Keywords**: gray box, internal information, activations, gradients, score change, saliency map, salt-and-pepper, extremely noisy, asking apples from the oranges tree
- **Definition ↔ trap**: "Gradient saliency is noisy because the gradient measures sensitivity at the level of individual pixels." ↔ "Gradient saliency is smooth because the gradient measures sensitivity at the level of whole objects."
- 한 줄: 입력을 살짝 바꿀 때 class 점수가 얼마나 변하나(gradient)로 위치를 보는데, 픽셀 단위라 salt-and-pepper처럼 지저분해.

### SmoothGrad (Smilkov et al., arXiv 2017, p.22)
- **Problem**: A single input gradient is noisy.
- **Core idea**: Step 1) Train a standard CNN classifier. Step 2) Add small random noise to the images. Step 3) Calculate the gradient for each. Step 4) Average. "calculating the gradient multiple times and averaging it removes noise and yields a more meaningful saliency map."
- **vs. previous / key difference**: Plain gradient saliency uses one gradient of the clean image. SmoothGrad averages gradients over several noisy copies. It is still a gray box method on an unchanged classifier, so the p.23 limitation remains.
- **Keywords**: SmoothGrad, random noise, gradient for each noisy copy, average, saliency map
- **Definition ↔ trap**: "SmoothGrad adds noise to the input several times and averages the resulting gradients." ↔ "SmoothGrad averages several noisy images first and then computes a single gradient."
- 한 줄: 입력에 noise를 섞은 사본마다 gradient를 구해서 평균 내면 깨끗해져.

### White box approach: CAM, Class Activation Mapping (Zhou et al., CVPR 2016 ← slide엔 "Zhou et al. 2016", p.24–26)
- **Problem**: Gray box maps need separate gradient calculations and are noisy.
- **Core idea**: Change the classifier so that it keeps spatial information. Step 1) "eliminate the FC layer". Step 2) "add GAP (Global Average Pooling)". Step 3) "add a 1×1 convolution that performs a linear combination of the features". The class activation map of a class is the weighted sum of the last conv feature maps with that class's weights: $M_c(x,y)=\sum_k w_k^c f_k(x,y)$ (p.25 figure: $w_1\ast$map$_1 + w_2\ast$map$_2 + \dots + w_n\ast$map$_n$; formula from the paper). "since the network itself is designed to preserve spatial information, activation maps can be obtained directly without the need for separate gradient calculations." It is "Designed to Generate Per-Class Maps" (p.26: one CAM for each of the top 5 predicted classes).
- **vs. previous / key difference**: Gray box reads gradients of an unchanged classifier. White box (CAM) redesigns the network (GAP + linear layer), so the map comes from a forward pass with no gradient computation.
- **Keywords**: white box, eliminate FC layer, GAP, 1×1 convolution, linear combination, class weights $w_k^c$, class activation map, per-class maps, no separate gradient calculation
- **Definition ↔ trap**: "CAM obtains activation maps directly because the network is designed to preserve spatial information." ↔ "CAM needs a separate backward pass to compute input gradients for each class."
- 한 줄: FC를 빼고 GAP + 선형 결합을 넣으면, class weight로 feature map을 더하는 것만으로 class별 위치 지도가 나와.

### Seeds: thresholding class maps into training data (p.27)
- **Problem**: A CAM is a continuous score map, not a pixel label.
- **Core idea**: "The Class Maps are Thresholded, to Generate Training Data". "High value scores ➝ Foreground areas", "Low value scores ➝ Background areas". "(This confident regions are commonly named 'seeds')" (slide grammar kept).
- **vs. previous / key difference**: The seeds become pseudo labels for training a segmentation network. Only confident regions are labeled (interpretation: the in-between pixels are left out).
- **Keywords**: threshold, foreground, background, confident regions, seeds, training data, pseudo label
- **Definition ↔ trap**: "High CAM scores become foreground seeds and low scores become background." ↔ "Low CAM scores become foreground seeds and high scores become background."
- 한 줄: CAM을 threshold로 잘라서 확실한 곳만 foreground/background seed로 써.

### CAM's two problems: "Good, but not good enough" (p.28–29)
- **Problem**: CAM seeds are useful but limited: "1. Resulting masks are not sharp" and "2. Focused on discriminative area only".
- **Core idea**: The following slides (p.29–41) are a list of fixes. For problem 1: "Increase top layers resolution" (p.29), "Use boundary-aware propagation" (p.30, SEC), "Use transfer learning" with saliency cues (p.33, Oh), "Enforce equivariance" (p.35, SEAM). For problem 2: "Let us play peek-a-boo!" (Hide-and-Seek, p.37–38) and "Let us play adversarial peek-a-boo!" (adversarial erasing, p.39–41).
- **vs. previous / key difference**: Problem 1 is about **boundaries** (masks are coarse). Problem 2 is about **extent** (only the most discriminative part, such as a face, lights up).
- **Keywords**: not sharp, discriminative area only, top layers resolution, boundary-aware propagation, transfer learning, equivariance, peek-a-boo, adversarial peek-a-boo
- **Definition ↔ trap**: "CAM masks are not sharp and focus on the discriminative area only." ↔ "CAM masks have sharp boundaries but cover too much background."
- 한 줄: CAM의 두 문제는 경계가 흐린 것과 가장 눈에 띄는 부분만 잡는 것.

### SEC: Seed, Expand, Constrain (Kolesnikov & Lampert, ECCV 2016, p.30–32)
- **Problem**: CAM limitations: "Activation maps are small and coarse", "Focus only on discriminative regions", "Poor object boundaries".
- **Core idea**: "Three complementary losses" train a segmentation CNN. **1. Seeding Loss**: "Use CAM as reliable foreground seeds", "Preserve highly confident object regions". **2. Expansion Loss**: "Encourage predictions to expand beyond the seeds", "Use image-level labels via Global Weighted Rank Pooling (GWRP)". **3. Constrain-to-boundary Loss**: "Refine predictions using CRF", "Encourage masks to align with image boundaries". Summary: "Seed: Preserve reliable CAM activations. Expand: Grow seeds to cover the entire object using image-level supervision. Constrain: Refine object boundaries with CRF-based regularization." (paper) Foreground seeds come from CAM (scores above 20% of the maximum) and background seeds from a saliency map; the seeding loss is a cross-entropy on seed pixels only, and the constrain loss is a KL divergence between the network output and its CRF-refined version. The CRF has no learned weights, but the paper back-propagates through it ("For computing gradients of the fully-connected CRF we employ the procedure from [34]", Sec. 3.2); the gray arrows on p.30 also pass through the CRF.
- **vs. previous / key difference**: Using thresholded CAM seeds directly as labels keeps both CAM problems. SEC keeps the seeds, grows them with image-level labels, and pulls the result to image boundaries in one training objective. The slide files it under "Use boundary-aware propagation" (p.30).
- **Keywords**: seeding loss, expansion loss, constrain-to-boundary loss, GWRP, CRF, weak localization, segmentation CNN, downscale
- **Definition ↔ trap**: "SEC's expansion loss uses image-level labels via GWRP to grow predictions beyond the seeds." ↔ "SEC's expansion loss uses a CRF to shrink predictions to the seeds."
- 한 줄: seed는 지키고(Seed), image label로 키우고(Expand), CRF로 경계에 맞춰(Constrain).

### Saliency-guided weak supervision (Oh et al., CVPR 2017, p.33–34)
- **Problem**: CAM "Provides class-specific localization" but "Covers only discriminative regions".
- **Core idea**: Saliency "Provides complete object extent" but is "Class-agnostic foreground estimation". "Combine CAM and saliency to generate class-specific pseudo masks". **Seeder**: "Generate class-specific seeds from CAM." **Saliency**: "Estimate foreground regions." **Guide Labeller**: "Fuse both cues into pseudo masks." **Segmentation CNN**: "Learn dense pixel-wise predictions." The slide calls this "Use transfer learning" and "We could also use saliency cues" (p.33). (paper) The saliency model is trained on a separate dataset, so its knowledge is transferred.
- **vs. previous / key difference**: SEC grows seeds inside the same task. Oh borrows an external, class-agnostic cue (saliency) that already knows the full object extent, and lets CAM decide the class.
- **Keywords**: saliency, class-agnostic, class-specific, seeder, guide labeller, segmenter convnet, dense classifier loss, transfer learning, pseudo masks
- **Definition ↔ trap**: "CAM gives class-specific localization, and saliency gives a class-agnostic foreground with complete extent." ↔ "Saliency gives class-specific localization, and CAM gives a class-agnostic foreground with complete extent."
- 한 줄: class는 CAM이, 물체 범위는 saliency가 맡아서 합친 pseudo mask로 학습해.

### SEAM: equivariance for CAMs (Wang et al., CVPR 2020, p.35–36)
- **Problem**: "CAMs are often coarse and inconsistent", and "Small input transformations can produce different activation maps".
- **Core idea**: "Activation maps should be equivariant to the input" (p.35). "Key idea: Enforce Equivariance": "Apply geometric transformations (e.g., affine transform)", "Predict CAMs for both original and transformed images", "Enforce consistency after compensating for the transformation". A Siamese network shares the CNN. **PCM** (Pixel Correlation Module) "propagates CAM activations to other pixels within the same object by utilizing pixel-wise affinity among CNN features". Training objective: **Classification Loss** ("Preserve image-level classification performance"), **Equivariant Regularization (ER) Loss** ("Encourage transformation-consistent CAMs"), **Cross-view Consistency (ECR) Loss** ("Align CAMs across different views"). (paper) SEAM = Self-supervised Equivariant Attention Mechanism, and the paper names ECR "Equivariant Cross Regularization" ← slide엔 "Cross-view Consistency (ECR)". The figure uses L1 for ER and ECR.
- **vs. previous / key difference**: Plain CAM is trained only by classification, so nothing ties the maps of a transformed image together. SEAM adds a self-supervised constraint: transform the CAM of the original and it should match the CAM of the transformed image. This is the L08 equivariance prior (L08 p.29) applied to CAMs.
- **Keywords**: equivariance, affine transform, Siamese network, PCM, pixel-wise affinity, classification loss, ER loss, ECR loss, L1, transformation-consistent
- **Definition ↔ trap**: "SEAM enforces that CAMs are consistent after compensating for the transformation (equivariance)." ↔ "SEAM enforces that CAMs stay unchanged when the image is transformed (invariance)."
- 한 줄: 사진을 변환하면 CAM도 똑같이 변환돼야 한다(equivariance)는 규칙으로 CAM을 일관되게 만들어.

### Hide-and-Seek (Singh & Lee, ICCV 2017 ← slide엔 "Wang et al. CVPR 2020", p.37–38)
- **Problem**: "CAM activates only the most discriminative regions."
- **Core idea**: "Solution (Hide-and-Seek)": "Randomly hide image patches during training", "Encourage learning from the remaining visible parts", "Generate more complete localization maps". Effect: "The classifier is forced to discover other visible object parts" and "CAM gradually covers a larger portion of the object." "Random hiding encourages the network to discover complementary object parts instead of relying on a single discriminative region." The figure splits the training image into S×S patches and hides a different random set each epoch. p.38 "Let us play peek-a-boo!" compares AlexNet-GAP and HaS heatmaps. (paper) Hiding is used only in training; the full image is used at test time.
- **vs. previous / key difference**: The fixes for problem 1 refine boundaries. Hide-and-Seek attacks problem 2 by changing the training input, so the network cannot rely on one part.
- **Keywords**: Hide-and-Seek (HaS), randomly hide patches, peek-a-boo, complementary object parts, more complete localization maps, grid of patches, epoch
- **Definition ↔ trap**: "Hide-and-Seek randomly hides image patches during training." ↔ "Hide-and-Seek hides the most discriminative region found by the CAM."
- 한 줄: 학습 때 patch를 무작위로 가리면 얼굴이 가려진 날엔 몸통을 보고 맞혀야 해서 CAM이 넓어져.

### Adversarial erasing, "adversarial peek-a-boo" (Wei et al., CVPR 2017 ← slide엔 "Wei et al. 2017", p.39–41)
- **Problem**: CAM covers only the discriminative area, and random hiding does not choose what to hide.
- **Core idea**: "adversarial instead of random erasures". A classification network produces a heatmap $H_t$. It is thresholded into the mined region $F_t$, which is erased from the image: $I_{t+1}$ = $I_t$ minus $F_t$. The network is trained again on the erased image, so it must find another region (steps $t = 1, 2, 3$, "Recurrent"). The mined regions together form "The Mined Object Region" (p.39 figure). p.40 shows AE-Step1–3 heatmaps and the erased regions in gray; p.41 shows segmentation results (AE-PSL). (paper) Each step trains a new classification network, and PSL = prohibitive segmentation learning.
- **vs. previous / key difference**: Hide-and-Seek hides **random** patches. Adversarial erasing erases **the region the classifier currently relies on most**, step by step.
- **Keywords**: adversarial erasing, AE, peek-a-boo, heatmap, threshold, mined region, erase, recurrent, AE-Step, mined object region, AE-PSL
- **Definition ↔ trap**: "Adversarial erasing removes the currently most discriminative region so the classifier must discover new object regions." ↔ "Adversarial erasing removes random patches so the classifier must discover new object regions."
- 한 줄: 지금 제일 센 곳을 지우고 다시 학습하길 반복해서, 지운 영역을 모으면 물체 전체가 돼.

### Takeaways: from image-level to pixel-level labels (p.42)
- **Problem**: Which of these ideas actually work?
- **Core idea**: "Training a classifier, and hoping to get segmentations out of it ✗". "Using spatial continuity priors will help, but to a limit ~". "The training can be modified to promote covering the object extent ✔". "With careful changes it is possible to obtain significant improvements ~". "⇒ How far can we go from image-level to pixel-level labels ?"
- **vs. previous / key difference**: Reading a plain classifier (gray box) is marked ✗. Changing training to cover the object extent (e.g. hiding, erasing) is marked ✔.
- **Keywords**: classifier, spatial continuity priors, object extent, careful changes
- **Definition ↔ trap**: "Training a classifier and hoping to get segmentations out of it does not work well." ↔ "Training a plain classifier is enough to get good segmentations out of it."
- 한 줄: 분류기만 학습하고 segmentation을 기대하면 안 되고, 물체 전체를 덮도록 학습을 바꿔야 해.

### Webly supervised learning (Shen et al., CVPR 2018, p.43–52)
- **Problem**: Manual labels are expensive, and CAMs from a target-domain classifier are incomplete.
- **Core idea**: Webly supervised learning is "A learning method that uses data collected from the web as supervision". It collects images "directly from web search engines", "Enables large-scale learning without manual annotation", but "Labels are noisy, ambiguous, and incomplete", so "Learning must be robust to noisy supervision" (p.44). Pipeline (p.45): 1. Train an image classifier (Model-T) on clean image-level labels. 2. Use Model-T to filter noisy web images. 3. Train Web-SEC (pixel labeler) using the filtered web images. 4. Apply Web-SEC to generate pixel labels for the target images. 5. Use Model-T to produce CAMs on the target images. 6. Fuse the two label sources and train the final segmentation model. Key point (p.47): "web images are much cleaner—typically containing a single centered object—so approximate segmentation masks can be generated automatically", and the network trained on them "learns generic object shapes and can later be transferred to the target dataset".
- **vs. previous / key difference**: "Unlike CAM-based methods, Web-SEC actually learns a pixel classifier." The figure shows two domains: Model-T (target) **filters** the web data, and Model-W (web) **enhances** the target side. (paper) The paper calls this bi-directional transfer between the target domain and the web domain.
- **Keywords**: webly supervised, web search engines, noisy labels, Model-T, Model-W, Web-SEC, pixel labeler, filter, enhance, single centered object, fuse
- **Definition ↔ trap**: "Webly supervised learning trades annotation cost for label noise." ↔ "Webly supervised learning trades label noise for higher annotation cost."
- 한 줄: 웹 사진은 물체 하나가 가운데 있어서 mask를 자동으로 얻기 쉬워. 거기서 pixel labeler를 배우고 target 사진의 CAM과 합쳐.

### Boxly supervised learning: "Convnets are robust to noise" (Khoreva et al., CVPR 2017, p.53–61)
- **Problem**: "From box-level labels to pixel-level instance labels" (p.54). "What we want" is object masks; "What we have" is boxes (p.55).
- **Core idea**: "Convnets are robust to noise. Let us use that !" "Initially, training is performed using a noisy mask, but the CNN's predictions are then used as pseudo-labels for retraining." "Segmentation networks are robust to noisy supervision", "Approximate masks are sufficient for training", "Iteratively refine pseudo masks during training", "Better masks lead to better segmentation" (p.56). Starting from filled rectangles, the output improves after 1, 5, 10 training rounds (p.57). Better initial masks (20% Bbox, MCG, Grabcut+, MCG ∩ Grabcut+) start higher (p.58–59). "We reach ~95% of full supervision by using GrabCut++" (p.60) ← slide 그래프 범례는 "Grabcut+", "MCG ∩ Grabcut+"(표는 "M ∩ G+"). (paper) The ~95% is reached with MCG ∩ GrabCut+. (paper) Between rounds the predictions are cleaned with the boxes (outside-box pixels reset, CRF), and GrabCut+ uses HED boundaries instead of color differences.
- **vs. previous / key difference**: Image-level methods must find the object's location and extent. Box labels already give location, so the remaining job is the shape inside the box, done by recursive self-training.
- **Keywords**: boxly supervised, box-level labels, noisy mask, pseudo-labels, retraining, training rounds, rectangles, 20% Bbox, MCG, GrabCut+, ~95% of full supervision
- **Definition ↔ trap**: "Box-supervised training starts from noisy masks and uses the CNN's own predictions as pseudo labels for retraining." ↔ "Box-supervised training uses the boxes as fixed labels in every round and never updates them."
- 한 줄: 네모를 대충 mask로 써서 학습하고, 예측을 다음 round의 label로 쓰길 반복해. 시작 mask를 좋게 하면(슬라이드 제목은 GrabCut++, 그림은 MCG ∩ GrabCut+) full supervision의 ~95%.

### Summary (p.61)
- **Problem**: What to remember from WSSS.
- **Core idea**: "Many priors and hints have been explored". "Exploit all available priors, explore new information sources". "Do not underestimate the power of transfer learning (and webly supervision)". "Current techniques are quite effective, and ride the wave of better network architectures".
- **vs. previous / key difference**: Not a method. It ties the lecture together: priors and hints (p.5–17), CAM fixes (p.28–41), web data (p.43–52), boxes (p.53–60).
- **Keywords**: priors, hints, new information sources, transfer learning, webly supervision, better network architectures
- **Definition ↔ trap**: "Do not underestimate the power of transfer learning (and webly supervision)." ↔ "Transfer learning and webly supervision add little to weakly supervised segmentation."
- 한 줄: prior와 hint를 다 쓰고, transfer learning과 web data를 얕보지 마.

### CD-CLIP: class distributions for CAM completeness and boundaries (Yang et al., CVPR 2026, p.62–74)
- **Problem**: "Beyond CAM" (p.62): "CAM activates only the most discriminative regions" and "CAM boundaries are coarse and inaccurate". Problem 1 "Incomplete activation". Problem 2 "Inaccurate boundaries": "Affinity propagation may spread activation" and "Foreground activation can cross class boundaries" (p.64). Conventional affinity "Uses feature or attention similarity between patches", but "Similar appearance does not always imply the same class" (p.65).
- **Core idea**: "A patch is characterized not only by its target-class score, but by its distribution over all classes" (target classes + super-classes, p.65, 67). **CDA** (Class Distribution Aware, p.66 figure): compare two patch distributions with JS divergence ("Small JS divergence: similar class distributions"), "Correct the original affinity", "Propagate CAM activation" (head → body). $S_d = \frac{1 - D_{JS}(\mathrm{Softmax}(S_a)\,\|\,\mathrm{Softmax}(S_a)^\top)}{\tau}$, $A_d = \mathrm{Softmax}(S_d)\cdot A_{init}$, $M^c_{re} = M^c_{init}\cdot (A_d)^t$ (p.69). **SBE** (Super-class Boundary Exploration, p.66 figure): "Use the CLIP super-class CAM as a semantic seed and average the corresponding DINO features": $f_s^n = \mathrm{CAP}(M_s^n \odot F_d)$, then "Compare the prototype with every DINO patch": $M_p^n = \mathrm{ReLU}(\mathrm{CosSim}(f_s^n, F_d))$ (p.70). Loss $\mathcal{L} = \mathcal{L}_{seg} + \lambda_1\mathcal{L}_b$, with $\mathcal{L}_b = \mathrm{Dice}(P, Y_s)$ only "if $|C_p| \ge 2$" (p.72). ← 슬라이드는 $S_a$, $A_{init}$, $\tau$, $t$, $M^c_{init}$, $F_d$, $M_s^n$를 정의하지 않아. 해석(p.66 그림 기준): $A_{init}$ = MHSA에서 온 initial affinity, $\tau$ = temperature, $t$ = 전파 횟수, $F_d$ = DINO features, $M_s^n$ = super-class CAM. 시험엔 식 전체보다 "JS로 affinity를 보정한 뒤 곱해서 CAM을 전파"만.
- **vs. previous / key difference**: Conventional affinity compares appearance; CDA compares class distributions, so it connects same-class patches without leaking across classes. "CDA significantly improves the CAM mIoU from 70.1 to 80.3"; SBE helps the final segmentation more than the CAM "because SBE contributes to boundary learning within the decoder" (p.73). "Better WSSS requires both knowing which patches belong together and knowing where their relationship should stop." (p.74)
- **Keywords**: CLIP, DINO, class distribution, target classes, super-classes, JS divergence, affinity, MHSA, CDA, super-class prototype, prototype CAM, CAP (class average pooling), SBE, boundary-enhanced loss, Dice, frozen encoders
- **Definition ↔ trap**: "CD-CLIP compares the class distributions of two patches with JS divergence to decide their affinity." ↔ "CD-CLIP compares only the target-class scores of two patches to decide their affinity."
- 한 줄: patch를 target class 점수 하나가 아니라 class 분포 전체로 비교해서 CAM을 넓히고(CDA), DINO prototype으로 어디서 멈출지 정해(SBE).

### Learning from the Web for weakly-supervised incremental segmentation (Liu et al., ECCV 2024, p.75–85)
- **Problem**: "Can a segmentation model learn new classes from the web without storing its original training images?" In the new scenario "Only the initial model is retained", "Original training images are no longer accessible", and "New and old classes must be learned from web images" (p.77). "Three challenges": web images have a different distribution, "Search queries provide noisy single-class labels", and "Learning new classes causes forgetting of old classes" (p.78).
- **Core idea**: "Language is used for querying, labeling, filtering, and rehearsal" (p.80). ① Retrieve web images (query with new class names; retrieve rehearsal images using stored captions). ② Filter noisy web images ("Domain filtering removes out-of-distribution images", "Caption filtering removes semantically irrelevant images"). ③ Generate supervision ("Captions are converted into multi-label supervision", "The previous model produces pixel-level pseudo-labels"). ④ Update the model (learn new classes, "Rehearse old classes without storing original images"). "The class name retrieves images; the caption determines their labels" (p.81). For old classes: "Store Captions, Not Images": Old image → stored caption q' → web search → rehearsal image (p.82), then verify by captioning the retrieved image (q'') and comparing nouns (p.83).
- **vs. previous / key difference**: Conventional WILSS keeps curated dataset images and gives only image-level labels for new classes, with a localizer making pseudo labels (p.79). Here "Web images replace curated incremental data, while captions replace reliable manual image labels" (p.79), and "The web is not only a source of new data. It can serve as an external semantic memory" (p.85). (paper) WILSS = Weakly Incremental Learning for Semantic Segmentation (task from Cermelli et al., CVPR 2022), and the domain filter is a Fourier-based domain discriminator.
- **Keywords**: incremental learning, WILSS, forgetting, rehearsal, web retrieval, domain filtering, caption filtering, caption labeling, multi-label supervision, stored captions, external memory, pseudo labels, 10–10 / 15–5 / 10–1
- **Definition ↔ trap**: "The method stores captions of old images, not the images, and uses them to retrieve rehearsal images from the web." ↔ "The method stores the original training images and replays them for rehearsal."
- 한 줄: 옛 사진 대신 caption만 저장해서 웹에서 비슷한 사진을 다시 찾아 복습하고, 새 class도 웹 사진 + caption label로 배워.

### WISH: heterogeneous weak labels in SAM's prompt space (Kweon & Yoon, CVPR 2025 ← slide에 cite 없음, p.86–95)
- **Problem**: "Under a fixed annotation budget, should every image receive the same type of weak label?" (p.86). In homogeneous supervision "Every training image uses the same weak label type": "Tags are cheap but provide no localization", "Boxes are informative but expensive", "Points provide an intermediate level of supervision" (p.87).
- **Core idea**: Heterogeneous supervision: "Different images may use different weak label types" ($Y_i^{\mathrm{Hetero}} = Y_i^t$ or $Y_i^p$ or $Y_i^b$), and "A single model learns from all label types" (p.87). "SAM's prompt space serves as a common representation for heterogeneous supervision": point prompt → point latent, box prompt → box latent, tag → converted point prompt, and the model predicts $\hat{Z} = \mathcal{H}_{\mathrm{prompt}}(Q)$ from Mask2Former object queries (p.88). Matching: class ($\mathrm{Cost}_{\mathrm{cls}}^{i,j} = \hat{y}^i_{\mathrm{cls}}(c^j)$), prompt ($\mathrm{KLD}(\hat Z^i, Z^j)$), mask ("Select the SAM candidate mask closest to the predicted instance mask", $\min_{n\in\{1,2,3\}} d(\hat M^i, M^{j,n}_{\mathrm{SAM}})$), combined $\alpha\,\mathrm{Cost_{cls}} + \beta\,\mathrm{Cost_{prompt}} + \gamma\,\mathrm{Cost_{mask}}$; "Hungarian matching assigns predictions to weakly labeled instances" (p.89–90). Tags become prompts through CAM peaks: "Local maxima are extracted from each class activation map", "Each selected CAM peak becomes a positive point prompt", "SAM generates an instance mask from each point" (p.91). The CAM comes from a trainable head, $A = \mathcal{H}_{\mathrm{cam}}(F)$, and self-enhancement trains it with $\mathcal{L}_{\mathrm{self}} = \mathcal{L}_{\mathrm{bce}}(A, \hat M_c)$, using the model's predicted masks as the target (p.91, 94). ← slide p.91 step 3("Improve CAMs with Model Predictions") 본문은 step 2 bullet 3개를 그대로 복붙한 슬라이드 오류야. 실제 내용은 $\mathcal{L}_{\mathrm{self}}$ 식과 p.94 "Current instance masks refine the CAM".
- **vs. previous / key difference**: "Conventional WSIS" "Chooses one weak label type for the entire dataset" and "Treats tags, points, and boxes as separate settings". WISH "Combines different label types in one model" and "Achieves better cost–performance trade-offs" (p.95). "Cheap labels provide breadth, while strong labels provide spatial precision" (p.93). "The optimal supervision strategy may be a portfolio of weak labels, not a single annotation type." (p.95)
- **Keywords**: heterogeneous supervision, homogeneous supervision, tags, points, boxes, SAM, prompt space, prompt latent, Mask2Former object queries, class matching, prompt matching, mask matching, Hungarian matching, CAM peaks, self-enhancement, annotation budget, WSIS
- **Definition ↔ trap**: "WISH uses SAM's prompt space as a common representation so that tags, points, and boxes can supervise one model." ↔ "WISH trains a separate model for each weak label type and picks the best one."
- 한 줄: tag·point·box를 SAM의 prompt 공간으로 옮겨서 한 모델이 섞인 label을 다 배워. 같은 예산이면 섞는 게 더 나을 수 있어.

## B. Must-know definitions & distinctions
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

## C. Question pool (27 questions)

### Q1 [T/F]
Priors are the indirect supervision received for each image, while hints are what we believe to be true independent of any particular image sample.
**Answer**: False
- **Why (EN)**: The two definitions are swapped. A prior is general knowledge that holds for any image, such as typical size or shape, while a hint is the weak label attached to one particular image, such as its class tag or a scribble.
- **Corrected (EN)**: Priors are what we believe to be true independent of any particular image sample, while hints are the indirect supervision received for each image.
- **해설**: "Priors: what do I believe to be true independent of any particular image sample", "Hints: indirect supervision received for each image" (p.5). prior 예: Size, Location, Shape, Number of instances, Contrast, Boundaries, saliency, Class Distribution, Similarity (p.6–10). hint 예: Image (video) label, Image caption, Click inside object, Object bound boxes, Object extreme points, Scribbles, Eye gaze (p.11–17).

### Q2 [T/F]
In weakly supervised semantic segmentation from image-level labels, the model is given only image-level labels at training time but must produce pixel-level labels at test time.
**Answer**: True
- **Why (EN)**: The label we pay for, an image-level class tag, is weaker than the output we need, a class for every pixel. The model is therefore supervised only indirectly.
- **해설**: p.19 그림: "Image-level labels at training time" → "Pixel-level labels at test time". p.4: "Fully supervised → Directly supervised", "Weakly supervised → Indirectly supervised". L08의 weak supervision 정의(train annotation < test 출력)와 같은 구도야.

### Q3 [T/F]
Like gradient-based saliency, CAM needs a separate gradient calculation of the class score with respect to the input to obtain its activation maps.
**Answer**: False
- **Why (EN)**: CAM is the white-box approach: the network is redesigned (the FC layer is removed, and GAP and a 1×1 convolution are added) so that it keeps spatial information. The class maps can therefore be read out directly from the feature maps and class weights, without a separate gradient calculation.
- **Corrected (EN)**: Unlike gradient-based saliency, CAM obtains its activation maps directly, without separate gradient calculations, because the network itself is designed to preserve spatial information.
- **해설**: "since the network itself is designed to preserve spatial information, activation maps can be obtained directly without the need for separate gradient calculations." (p.25) gradient를 쓰는 건 gray box야: $\partial S_c / \partial I$ (p.21).

### Q4 [T/F]
CAM generates one map per class: the map for a class is a weighted sum of the last convolutional feature maps, using the weights that connect the GAP outputs to that class.
**Answer**: True
- **Why (EN)**: Each GAP output is the average of one feature map, and the class score is a weighted sum of these averages. Applying the same class weights to the feature maps themselves, before averaging, shows where in the image that class score comes from.
- **해설**: p.25 그림: $w_1 \ast (\text{map}_1) + w_2 \ast (\text{map}_2) + \dots + w_n \ast (\text{map}_n) =$ Class Activation Map (Australian terrier). p.27 "M2: Weighted Averaging": $M_k = w_{1,k}F_1 + \dots + w_{C,k}F_C$. p.26 제목이 "Designed to Generate Per-Class Maps"이고, 그림은 top 5 predicted categories마다 CAM을 하나씩 보여 줘. 첫 문장의 "평균을 낸 뒤 가중합 = 가중합한 뒤 평균"은 해석이야.

### Q5 [T/F]
When class maps are thresholded to generate training data, low-score regions are used as foreground and high-score regions as background.
**Answer**: False
- **Why (EN)**: The direction is reversed. A high CAM score means the classifier is confident that the class is there, so high-score regions become foreground and low-score regions become background. These confident regions are called seeds.
- **Corrected (EN)**: When class maps are thresholded, high value scores give foreground areas and low value scores give background areas, and these confident regions are commonly named seeds.
- **해설**: "High value scores → Foreground areas", "Low value scores → Background areas", "(This confident regions are commonly named "seeds")" (p.27, 원문 문법 그대로). 그림 M3: resize한 $M'_k$를 $> \tau_{loc}$로 잘라 $B_k$를 만들어.

### Q6 [T/F]
In SEC, the expansion loss refines predictions with a CRF so that masks align with image boundaries, while the constrain-to-boundary loss grows the seeds using image-level labels through Global Weighted Rank Pooling (GWRP).
**Answer**: False
- **Why (EN)**: The two losses are swapped. The expansion loss uses image-level labels through GWRP to push predictions beyond the seeds, and the constrain-to-boundary loss uses a CRF to align masks with image boundaries.
- **Corrected (EN)**: In SEC, the expansion loss encourages predictions to expand beyond the seeds using image-level labels via GWRP, while the constrain-to-boundary loss refines predictions with a CRF so that masks align with image boundaries.
- **해설**: "2. Expansion Loss: Encourage predictions to expand beyond the seeds, Use image-level labels via Global Weighted Rank Pooling (GWRP)", "3. Constrain-to-boundary Loss: Refine predictions using CRF, Encourage masks to align with image boundaries" (p.31). 나머지 하나는 "1. Seeding Loss: Use CAM as reliable foreground seeds". 이름 순서 그대로 Seed → Expand → Constrain이야.

### Q7 [T/F]
In saliency-guided weak supervision, saliency provides class-specific localization but covers only discriminative regions, whereas CAM provides the complete object extent as a class-agnostic foreground estimate.
**Answer**: False
- **Why (EN)**: The two cues are swapped. CAM knows which class is where but covers only the discriminative part, and saliency covers the whole object but does not know its class. The method combines them to get class-specific pseudo masks that cover whole objects.
- **Corrected (EN)**: CAM provides class-specific localization but covers only discriminative regions, whereas saliency provides the complete object extent as a class-agnostic foreground estimate; combining the two gives class-specific pseudo masks.
- **해설**: "CAM: Provides class-specific localization, Covers only discriminative regions", "Saliency: Provides complete object extent, Class-agnostic foreground estimation", "Combine CAM and saliency to generate class-specific pseudo masks" (p.34, Oh et al. CVPR 2017). 구성: Seeder(CAM) + Saliency → Guide Labeller → Segmentation CNN.

### Q8 [T/F]
SEAM assumes that activation maps should be invariant to the input, so it forces the CAM of an affine-transformed image to be identical to the CAM of the original image without compensating for the transformation.
**Answer**: False
- **Why (EN)**: SEAM enforces equivariance, not invariance. If the input is rescaled or shifted, the activation map should be rescaled or shifted in the same way, so the CAMs are compared only after compensating for the transformation.
- **Corrected (EN)**: SEAM assumes that activation maps should be equivariant to the input, so it predicts CAMs for the original and transformed images and enforces consistency after compensating for the transformation.
- **해설**: "Activation maps should be equivariant to the input" (p.35). "Apply geometric transformations (e.g., affine transform)", "Predict CAMs for both original and transformed images", "Enforce consistency after compensating for the transformation" (p.36). 이 일을 하는 게 "Equivariant Regularization (ER) Loss: Encourage transformation-consistent CAMs"야. ← slide엔 ECR이 "Cross-view Consistency (ECR) Loss"로 적혀 있는데, 논문 이름은 Equivariant Cross Regularization이야 (paper).

### Q9 [T/F]
Hide-and-Seek randomly hides image patches during training, so the classifier is forced to discover other visible object parts and the CAM gradually covers a larger portion of the object.
**Answer**: True
- **Why (EN)**: If the most discriminative patch is hidden in some epochs, the classifier cannot rely on it alone and must also learn the remaining visible parts. The localization map therefore spreads over more of the object.
- **해설**: "Solution (Hide-and-Seek): Randomly hide image patches during training", "Effect: The classifier is forced to discover other visible object parts", "CAM gradually covers a larger portion of the object" (p.37). "Random hiding encourages the network to discover complementary object parts instead of relying on a single discriminative region." ← slide엔 [Wang et al. CVPR 2020]으로 cite돼 있는데, Hide-and-Seek(HaS)은 Singh & Lee, ICCV 2017이야 (paper). p.11의 hint cite 목록엔 "Singh ICCV 17"이 있어.

### Q10 [T/F]
Adversarial erasing does not erase random patches: at each step, it erases the discriminative region highlighted by the classification network's heatmap and feeds the erased image back, so that the network mines further object regions.
**Answer**: True
- **Why (EN)**: The region to erase is chosen by the current classifier itself, so exactly the part it relies on is removed. After erasing, the classifier must find another part of the object, and the regions found over the steps add up to the mined object region.
- **해설**: "Let us play adversarial peek-a-boo!", "(adversarial instead of random erasures)" (p.39). 그림: heatmap $H_t$ → threshold → $F_t$ → $I_t \ominus F_t = I_{t+1}$ → 다시 Input Image, 아래 줄 "The Mined Object Region". p.40 caption: "The second to fourth rows show the produced heatmaps, where the discriminative regions are highlighted. The images with erased regions are shown in the last row in gray." (Wei et al. 2017) HaS는 random hiding이라는 게 차이야.

### Q11 [T/F]
According to the takeaways, training a classifier and hoping to get segmentations out of it does not work, whereas modifying the training to promote covering the object extent does work.
**Answer**: True
- **Why (EN)**: A plain classifier only needs the discriminative part, so its maps miss the rest of the object. Changing the training, for example by hiding or erasing parts, pushes the maps to cover the whole object. The slide marks the plain classifier with a cross and the modified training with a check mark, and adds that significant improvements still need careful changes (~).
- **해설**: p.42 Takeaways: "Training a classifier, and hoping to get segmentations out of it ✗", "Using spatial continuity priors will help, but to a limit ~", "The training can be modified to promote covering the object extent ✔", "With careful changes it is possible to obtain significant improvements ~". 기호 ✗ / ~ / ✔를 뒤집은 지문이 함정이 될 수 있어.

### Q12 [T/F]
In the webly supervised pipeline of Shen et al., Web-SEC is the image classifier trained on clean image-level labels and used to filter noisy web images, while Model-T is the pixel labeler trained on the filtered web images.
**Answer**: False
- **Why (EN)**: The roles are swapped. Model-T is the image classifier: it filters the web images and produces CAMs on the target images. Web-SEC is the pixel labeler: it is trained on the filtered web images and generates pixel labels for the target images.
- **Corrected (EN)**: Model-T is the image classifier trained on clean image-level labels and used to filter noisy web images, while Web-SEC is the pixel labeler trained on the filtered web images.
- **해설**: Pipeline (p.45–50): "1. Train an image classifier (Model-T) on clean image-level labels. 2. Use Model-T to filter noisy web images. 3. Train Web-SEC (pixel labeler) using the filtered web images. 4. Apply Web-SEC to generate pixel labels for the target images. 5. Use Model-T to produce class activation maps (CAMs) on the target images. 6. Fuse the two label sources and train the final segmentation model."

### Q13 [T/F]
In boxly supervised segmentation, Khoreva et al. first train on noisy masks made from the boxes and then use the CNN's own predictions as pseudo-labels for retraining, because segmentation networks are robust to noisy supervision.
**Answer**: True
- **Why (EN)**: A box only roughly covers the object, but the network does not need perfect masks. Its predictions are already cleaner than the rectangles, so they are reused as the labels for the next round and the masks improve over the rounds.
- **해설**: "Convnets are robust to noise. Let us use that!", "Initially, training is performed using a noisy mask, but the CNN's predictions are then used as pseudo-labels for retraining.", "Segmentation networks are robust to noisy supervision.", "Approximate masks are sufficient for training.", "Iteratively refine pseudo masks during training." (p.56) 그림: "Example input rectangles" → "Output after 1 training round" → "After 5 rounds" → "After 10 rounds" → "Ground truth".

### Q14 [T/F]
In CD-CLIP, a large JS divergence between the class distributions of two patches means that they are likely to belong to the same semantic region.
**Answer**: False
- **Why (EN)**: JS divergence measures how different two distributions are, so a small value means similar class distributions. Patches of the same semantic class show similar class distributions, so a small divergence suggests the same region.
- **Corrected (EN)**: In CD-CLIP, a small JS divergence between the class distributions of two patches means that they are likely to belong to the same semantic region.
- **해설**: "Small JS divergence: similar class distributions", "Large JS divergence: different class distributions" (p.69). 식도 $S_d = \frac{1 - D_{JS}(\cdot \Vert \cdot)}{\tau}$라서 divergence가 작을수록 similarity가 커 (p.69). "Patches of the same semantic class show similar class distributions." (p.65)

### Q15 [T/F]
In CD-CLIP, CDA mainly improves boundary accuracy through boundary learning in the decoder, while SBE mainly expands the CAM to cover the missing object regions.
**Answer**: False
- **Why (EN)**: The roles are swapped. CDA uses class distributions to correct the patch affinity and propagate CAM activation to missing regions, so it addresses completeness. SBE uses DINO-based super-class prototypes to supervise boundaries, so it addresses boundary accuracy.
- **Corrected (EN)**: In CD-CLIP, CDA mainly expands the CAM to cover the missing object regions (CAM completeness), while SBE mainly improves boundary accuracy through boundary learning in the decoder.
- **해설**: "1. CDA: CAM completeness — Class distributions → better patch affinity → complete CAM", "2. SBE: Boundary accuracy — DINO structure → super-class prototypes → boundary supervision" (p.74). "This is because SBE contributes to boundary learning within the decoder, rather than to activation expansion." (p.73)

### Q16 [T/F]
In the language-driven web learning method for weakly-supervised incremental segmentation, the original training images are not stored after the initial step; instead, their captions are stored and used to retrieve rehearsal images from the web.
**Answer**: True
- **Why (EN)**: Keeping old images is not allowed in this setting, but forgetting old classes must still be prevented. A stored caption acts as a search key, so the web can return similar images for rehearsing the old classes.
- **해설**: "Store Captions, Not Images: Original training images are unavailable after the initial step. The method stores their captions instead. Stored captions are used to retrieve rehearsal images from the web." "Old image → Stored caption q' → Web search → Rehearsal image" (p.82). "Captions serve as retrieval keys, and the web serves as external memory." (p.83)

### Q17 [T/F]
WISH uses SAM's prompt space as a common representation, so that point, box, and tag labels can be compared in one space; a tag is first converted into a point prompt.
**Answer**: True
- **Why (EN)**: Different weak labels have different formats, so they cannot be compared directly. SAM already maps points and boxes into one prompt space, and a tag can join that space once its CAM peaks are turned into point prompts.
- **해설**: "SAM's prompt space serves as a common representation for heterogeneous supervision." "Point prompt → point latent", "Box prompt → box latent", "Tag → converted point prompt", "Weak labels with different formats become comparable in one space." (p.88) tag를 point로 바꾸는 법은 p.91 (CAM peak → positive point prompt).

### Q18 [MCQ]
Which of the following is NOT one of the hints (indirect supervision received for each image) listed in the lecture?
(A) Image caption
(B) Click inside object
(C) Scribbles
(D) Class distribution
**Answer**: (D)
- **Why (EN)**: Class distribution, such as "2/3 cats, 1/3 cars", is a prior: it is believed to hold over the whole dataset, not given for one image. Captions, clicks, and scribbles are attached to each image.
- **해설**: Hints: Image (video) label, Image caption, Click inside object, Object bound boxes, Object extreme points, Scribbles, Eye gaze (p.11–17). Class Distribution은 Priors 쪽이야 (p.9, "⅔ Cats, ⅓ Cars").

### Q19 [MCQ]
What is SmoothGrad for?
(A) To replace the image classifier with a segmentation network
(B) To reduce the "salt-and-pepper" noise of gradient saliency by averaging the gradients computed on several noisy copies of the image
(C) To remove the FC layer and add global average pooling
(D) To erase the most discriminative region and retrain the classifier
**Answer**: (B)
- **Why (EN)**: A single gradient measures sensitivity pixel by pixel, so the saliency map is extremely noisy. Adding small random noise, computing the gradient for each copy, and averaging removes much of this noise.
- **해설**: "The problem is that it is extremely noisy. This is because the gradient measures sensitivity at the level of individual pixels, resulting in a "salt-and-pepper" appearance." (p.21) SmoothGrad: "Step 2) Add small random noise to the images, Step 3) Calculate the gradient for each, Step 4) Average", "calculating the gradient multiple times and averaging it removes noise and yields a more meaningful saliency map." (p.22) (C)는 CAM(white box), (D)는 adversarial erasing이야.

### Q20 [MCQ]
Which statement correctly describes the two main approaches for getting pixel-level labels from a classifier?
(A) The gray box approach redesigns the architecture so that class maps come directly out of the forward pass.
(B) The gray box approach uses the internal information of a trained classifier, its activations or gradients, to find the areas the model focused on for classification.
(C) The white box approach computes the gradient of the class score with respect to the input image.
(D) Both approaches require pixel-level labels at training time.
**Answer**: (B)
- **Why (EN)**: The gray box approach looks inside an ordinary classifier, for example with the gradient $\partial S_c / \partial I$. The white box approach, CAM, changes the network (no FC layer, GAP, 1×1 convolution) so that the maps can be read out without separate gradient calculations.
- **해설**: "Gray box: Utilize the internal information in the network. In other words, it identifies which areas the model focused on for classification by leveraging the activations or gradients within the CNN." (p.20) White box = CAM: "eliminate the FC layer", "add GAP", "add a 1×1 convolution" (p.24), "without the need for separate gradient calculations" (p.25). (A)·(C)는 둘을 뒤바꾼 거야. 두 방법 다 학습 때는 image-level label만 써 (p.19: "Image-level labels at training time").

### Q21 [MCQ]
Which of the following is NOT listed as a CAM limitation on the SEC slide?
(A) Activation maps are small and coarse
(B) Focus only on discriminative regions
(C) Poor object boundaries
(D) Activation maps cannot be computed separately for each class
**Answer**: (D)
- **Why (EN)**: CAM is designed to generate a map for every class, so per-class maps are not a limitation. The listed limitations are coarse maps, focus on discriminative regions, and poor boundaries, and SEC's three losses target them.
- **해설**: "CAM limitations: Activation maps are small and coarse, Focus only on discriminative regions, Poor object boundaries" (p.31). "Designed to Generate Per-Class Maps" (p.26).

### Q22 [MCQ]
On the "Good, but not good enough" slides, which remedy is listed under problem 2, "Focused on discriminative area only", rather than under problem 1, "Resulting masks are not sharp"?
(A) Increase top layers resolution
(B) Use boundary-aware propagation
(C) Enforce equivariance
(D) Let us play peek-a-boo (Hide-and-Seek)
**Answer**: (D)
- **Why (EN)**: Hiding patches attacks the discriminative-part problem directly: the classifier must use other visible parts of the object. The other three options are listed as remedies for masks that are not sharp. The question follows the slide layout: p.35 files "Enforce equivariance" under problem 1.
- **해설**: problem 1 아래 화살표: "Increase top layers resolution" (p.29), "Use boundary-aware propagation" (p.30, SEC), "Use transfer learning" (p.33), "Enforce equivariance" (p.35, SEAM). problem 2 아래 화살표: Hide-and-Seek (p.37), "Let us play peek-a-boo!" (p.38), "Let us play adversarial peek-a-boo!" (p.39–41). 주의: p.33의 saliency 방법("We could also use saliency cues", Oh et al.)은 화살표는 problem 1 쪽이지만, p.34에서 saliency가 "Provides complete object extent"라고 해서 problem 2(discriminative only)도 같이 메워. 그래서 saliency는 보기에서 뺐어.

### Q23 [MCQ]
In the webly supervised pipeline, why can Web-SEC learn a pixel classifier from web images?
(A) Web search engines return pixel-level annotations with each image
(B) Web images are much cleaner, typically containing a single centered object, so approximate segmentation masks can be generated automatically
(C) Annotators filter and label the web images by hand before training
(D) Web-SEC copies the CAMs of Model-T without any training
**Answer**: (B)
- **Why (EN)**: Because a typical web image is much cleaner and typically contains a single centered object, rough masks can be produced automatically and used as training targets. A network trained on them learns generic object shapes that can be transferred to the target dataset.
- **해설**: "Unlike CAM-based methods, Web-SEC actually learns a pixel classifier.", "The key observation is that web images are much cleaner—typically containing a single centered object—so approximate segmentation masks can be generated automatically.", "These masks are then used to train a pixel-level segmentation network, which learns generic object shapes and can later be transferred to the target dataset." (p.47) web 이미지 filtering은 사람이 아니라 Model-T가 해 (p.45).

### Q24 [MCQ]
What problem of conventional affinity motivates CD-CLIP to compare the class distributions of patches?
(A) Conventional affinity requires pixel-level labels
(B) Similar appearance does not always imply the same class, so feature or attention similarity can cause under-activation or boundary leakage
(C) Conventional affinity cannot be computed between image patches
(D) Conventional affinity ignores the CLIP text encoder, which makes it too slow
**Answer**: (B)
- **Why (EN)**: Two patches can look alike but belong to different classes, so appearance-based affinity may connect the wrong patches. Patches of the same semantic class have similar distributions over all classes, so comparing the full distributions gives a more reliable affinity.
- **해설**: "Conventional affinity: Uses feature or attention similarity between patches. Similar appearance does not always imply the same class. Incorrect affinity causes under-activation or boundary leakage." "CD-CLIP observation: Every patch has scores over multiple classes. Patches of the same semantic class show similar class distributions. The full distribution can be more informative than one target-class score." (p.65)

### Q25 [MCQ]
In CD-CLIP's super-class boundary exploration (SBE), what is DINO for?
(A) It provides the class names for the CLIP text encoder
(B) Its features are averaged inside the CLIP super-class CAM to form a prototype, and comparing the prototype with every DINO patch recovers a complete, structurally coherent super-class region
(C) It computes the JS divergence between patch class distributions
(D) It replaces the segmentation decoder at test time
**Answer**: (B)
- **Why (EN)**: CLIP tells which super-class a region belongs to, and DINO features carry object structure. Averaging DINO features under the CLIP super-class CAM gives a prototype, and its similarity to every patch outlines the region for boundary supervision.
- **해설**: "1. Construct a super-class prototype: Use the CLIP super-class CAM as a semantic seed and average the corresponding DINO features." $f^n_s = \mathrm{CAP}(M^n_s \odot F_d)$. "2. Generate a prototype CAM: Compare the prototype with every DINO patch to recover a complete and structurally coherent super-class region." $M^n_p = \mathrm{ReLU}(\mathrm{CosSim}(f^n_s, F_d))$ (p.70). "CLIP gives the prototype a semantic identity, and DINO expands it into a coherent region" (p.71). (C)는 CDA 쪽이야 (p.69).

### Q26 [MCQ]
In the language-driven web learning method for weakly-supervised incremental segmentation, which of the following is NOT done with language (class names or captions)?
(A) Querying the web for images of new classes
(B) Converting captions into multi-label image-level supervision
(C) Filtering out semantically irrelevant web images
(D) Producing the pixel-level pseudo-labels
**Answer**: (D)
- **Why (EN)**: The pixel-level pseudo-labels come from the previous segmentation model, not from text. Language is used for querying, labeling, filtering, and rehearsal.
- **해설**: "③ Generate supervision: Captions are converted into multi-label supervision. The previous model produces pixel-level pseudo-labels." "Language is used for querying, labeling, filtering, and rehearsal." (p.80) "The class name retrieves images; the caption determines their labels." (p.81)

### Q27 [MCQ]
How does WISH turn a class tag into instance-level supervision?
(A) An annotator clicks on every instance of the tagged class
(B) Local maxima of the class activation map become positive point prompts, and SAM generates an instance mask from each point
(C) A box around the whole image is given to SAM as a box prompt
(D) Web images retrieved with the tag provide copied masks
**Answer**: (B)
- **Why (EN)**: A tag says what is in the image but not where. The CAM gives rough class-specific locations, its peaks become point prompts, and SAM turns each point into an instance mask; overlapping masks are grouped into individual instances.
- **해설**: "1. Generate Class Activation Maps: CAM provides rough class-specific localization. Local maxima are extracted from each class activation map." "2. Convert CAM Peaks into Point Prompts: Each selected CAM peak becomes a positive point prompt. SAM generates an instance mask from each point. Overlapping masks are grouped to identify individual instances." "CAM supplies missing spatial cues, and SAM turns those cues into instance-level masks." (p.91)

## D. Fill-in-the-blank pool (11 questions)

### F1 [Blank]
There are only two sources of information. Priors: what do I believe to be true (1) ______ of any particular image sample. Hints: (2) ______ supervision received for each image.
**Answer**: (1) independent (2) indirect — p.5. 두 정의를 뒤바꾼 T/F가 나오기 좋아.

### F2 [Blank]
White box approach (CAM): Step 1) eliminate the (1) ______ layer. Step 2) add (2) ______. Step 3) add a 1×1 convolution that performs a linear (3) ______ of the features.
**Answer**: (1) FC (accept: fully connected) (2) GAP (accept: Global Average Pooling, global average pooling) (3) combination (accept: linear combination) — p.24. 그래서 gradient 계산 없이 map이 나와 (p.25).

### F3 [Blank]
SmoothGrad: Step 1) Train a standard CNN classifier. Step 2) Add small random (1) ______ to the images. Step 3) Calculate the (2) ______ for each. Step 4) (3) ______.
**Answer**: (1) noise (accept: random noise, Gaussian noise) (2) gradient (accept: gradients, $\partial S_c/\partial I$) (3) Average (accept: average the results, mean) — p.22. 그림 쪽 표기는 "Repeat 50 times step 2&3, average result".

### F4 [Blank]
The class maps are thresholded to generate training data: high value scores → (1) ______ areas, low value scores → (2) ______ areas. These confident regions are commonly named "(3) ______".
**Answer**: (1) foreground (2) background (3) seeds (accept: seed) — p.27.

### F5 [Blank]
SEC: the (1) ______ loss encourages predictions to expand beyond the seeds using image-level labels via Global Weighted Rank Pooling (GWRP), and the constrain-to-boundary loss refines predictions using a (2) ______.
**Answer**: (1) expansion (accept: expand) (2) CRF (accept: conditional random field) — p.31. 남은 하나가 seeding loss야 ("Use CAM as reliable foreground seeds").

### F6 [Blank]
SEAM: "Activation maps should be ______ to the input."
**Answer**: equivariant (accept: equivariance; NOT invariant) — p.35. invariant가 아니야. 변환을 보정(compensate)한 뒤 일관성을 걸어 (p.36).

### F7 [Blank]
Webly supervised learning leverages freely available web images as supervision, trading (1) ______ cost for label (2) ______.
**Answer**: (1) annotation (accept: labeling, manual annotation) (2) noise — p.44.

### F8 [Blank]
Convnets are robust to (1) ______. Initially, training is performed using a noisy mask, but the CNN's predictions are then used as (2) ______ for retraining.
**Answer**: (1) noise (accept: noisy supervision, noisy labels, noisy masks, label noise) (2) pseudo-labels (accept: pseudo labels, pseudo masks) — p.56 (Khoreva et al.).

### F9 [Blank]
CD-CLIP: "A patch is characterized not only by its target-class score, but by its (1) ______ over all classes." Two patch distributions are compared with the (2) ______ divergence, and a (3) ______ divergence means similar class distributions.
**Answer**: (1) distribution (accept: class distribution) (2) JS (accept: Jensen–Shannon, J-S) (3) small (accept: low) — p.65, p.69.

### F10 [Blank]
The web as external memory: Old image → Stored (1) ______ q' → Web search → (2) ______ image.
**Answer**: (1) caption (2) rehearsal — p.82. "Captions serve as retrieval keys, and the web serves as external memory." (p.83)

### F11 [Blank]
WISH: Class matching identifies the semantic (1) ______. Prompt matching identifies the approximate instance (2) ______. Mask matching transfers (3) ______'s segmentation knowledge.
**Answer**: (1) category (accept: class) (2) location (accept: position) (3) SAM (accept: Segment Anything Model) — p.90. 마지막에 Hungarian matching이 prediction을 weakly labeled instance에 배정해.

## E. Component roles

"Without it" 열은 슬라이드의 문제 제기에서 이어지는 해석이야. 슬라이드 문장 그대로는 아니야.

| Part | What is it for? | Without it | Page |
|---|---|---|---|
| Input gradient (gray box) | It measures how much the class score changes when each pixel is slightly modified, which marks the pixels the classifier used. | A plain classifier gives no location at all. | p.21 |
| Noise + averaging (SmoothGrad) | It averages gradients over noisy copies so that pixel-level noise cancels out. | The saliency map keeps its salt-and-pepper appearance. | p.22 |
| GAP | It keeps one value per feature map while the maps themselves stay spatial, so class weights can be applied back to the maps. | The FC layer would mix all positions, so the class weights could not be applied back onto the feature maps. | p.24 |
| Class weights $w_k^c$ (CAM) | They combine the last conv feature maps linearly into a class activation map for class c. | There is no rule for turning feature maps into a map of one class. | p.24–25 |
| Threshold (seeds) | It turns a continuous CAM into confident foreground and background seeds that can serve as training data. | The score map cannot be used as a pixel label. | p.27 |
| Seeding loss (SEC) | It keeps the network's predictions consistent with the reliable CAM seeds. | The network has no anchor for where the object is. | p.31 |
| Expansion loss + GWRP (SEC) | It uses image-level labels to push predictions to grow beyond the small seeds. | Predictions stay as small as the discriminative seeds. | p.31 |
| Constrain-to-boundary loss + CRF (SEC) | It pulls the predicted masks toward the CRF-refined masks that align with image boundaries (paper: gradients are also back-propagated through the CRF, Sec. 3.2). | Masks stay coarse and blurry at object boundaries. | p.31 |
| Saliency model (Oh) | It provides a class-agnostic foreground with the complete object extent. | Pseudo masks cover only the discriminative regions found by CAM. | p.34 |
| Guide labeller (Oh) | It fuses CAM seeds (class) and saliency (extent) into class-specific pseudo masks. | The class cue and the extent cue cannot be combined. | p.34 |
| Affine transform (SEAM) | It creates a second view whose correct CAM is known up to the same transform. | There is no self-supervised signal to check CAM consistency. | p.36 |
| ER loss (SEAM) | It encourages transformation-consistent CAMs between the original and transformed images. | Small input transformations can produce different activation maps. | p.36 |
| PCM (SEAM) | It propagates CAM activations to other pixels in the same object using pixel-wise feature affinity. | Activation is not spread to other pixels of the same object by feature affinity. | p.36 |
| Random hiding (HaS) | It hides random patches so the classifier must use other visible object parts. | The classifier keeps relying on a single discriminative region. | p.37 |
| Adversarial erasing | It erases the region the classifier currently relies on, so the next step must find new object regions. | Mining stops at the most discriminative region. | p.39 |
| Model-T (webly) | It filters noisy web images and produces CAMs on the target images. | Noisy web images enter training, and the target side has no CAM labels. | p.45 |
| Web-SEC (webly) | It learns a pixel classifier from clean web images, so it can generate pixel labels for target images. | Only CAM-based labels are available, which miss object shape. | p.45, 47 |
| Fusion of two label sources (webly) | It combines Web-SEC pixel labels with Model-T CAMs into the final training labels. | One incomplete label source alone trains the final model. | p.45 |
| Recursive retraining (boxly) | It replaces noisy box masks with the network's own predictions, round after round. | The network learns the rectangle shapes as they are. | p.56–57 |
| GrabCut+ / MCG initial masks (boxly) | They give better starting masks inside each box. | Training starts from rectangles and needs more rounds to recover shape. | p.59–60 |
| Class distribution (CD-CLIP) | It describes each patch by its scores over all classes, which separates patches that look alike but belong to different classes. | Affinity relies on appearance and can leak across classes. | p.65, 67 |
| JS divergence (CD-CLIP) | It measures how similar two patches' class distributions are, which corrects the original affinity. | Unreliable relationships in the original affinity stay. | p.69 |
| Super-class prototype + prototype CAM (SBE) | It uses DINO features averaged inside the CLIP super-class CAM to recover a coherent super-class region with clear structure. | The expanded CAM can cross class boundaries. | p.70–71 |
| $\lvert C_p\rvert \ge 2$ condition (boundary loss) | It applies the boundary loss only when two or more distinct super-classes exist in the image. | The boundary loss would also apply when there is no boundary between super-classes to learn. | p.72–73 |
| Domain filtering (Liu) | It removes web images with a large domain gap and keeps dataset-like images. | Out-of-distribution web images enter training. | p.80–81 |
| Captions as labels (Liu) | They turn the single-class search query into multi-label supervision covering all visible objects. | Each web image keeps a noisy single-class label. | p.81 |
| Stored captions (Liu) | They act as retrieval keys for rehearsal images, so old classes are rehearsed without storing images. | Learning new classes causes forgetting of old classes. | p.82 |
| Caption filtering (Liu) | It removes retrieved images whose new caption does not share a semantic family with the stored caption. | Irrelevant retrievals are used for rehearsal. | p.83 |
| SAM prompt space (WISH) | It gives tags, points, and boxes one common representation that a single model can learn from. | Each label type needs its own setting or model. | p.88 |
| Prompt head $\mathcal{H}_{\mathrm{prompt}}$ (WISH) | It predicts a prompt latent from each object query so that the query can be compared with weak labels. | Queries cannot be matched to point or box labels. | p.88 |
| Hungarian matching (WISH) | It assigns predictions to weakly labeled instances using the combined class, prompt, and mask cost. | Each query has no instance target to learn from. | p.89–90 |
| Adaptive SAM mask matching (WISH) | It picks, among SAM's three candidate masks, the one closest to the current prediction. | A fixed mask level is used, which performs worse. | p.89, 94 |
| CAM peaks → point prompts (WISH) | They convert class tags into point prompts, so SAM can produce instance masks. | Tags give no spatial cue for instances. | p.91 |
| Self-enhancement (WISH) | Current instance masks refine the CAM, and better CAMs give better point prompts for tags. | Tag-only images keep the initial rough CAM peaks. | p.94 |

## F. Pipelines

"Target comes from"와 "Gradient goes to" 줄은 슬라이드 문장이 아니라 해석이야. "(paper)"는 논문에서 확인한 내용이야.

**CAM → seeds → segmentation** (p.24–27)
1. Replace the FC layer of a classifier with GAP followed by a linear layer (1×1 convolution).
2. Train it as an image classifier with image-level labels.
3. For each class, compute the CAM as the class-weighted sum of the last feature maps.
4. Threshold the CAM: high scores become foreground seeds, low scores become background.
5. Train a segmentation network on these seeds.
- **Target comes from**: human image labels for the classifier; the classifier's own CAM (thresholded) for the segmentation network.
- **Gradient goes to**: the classifier in step 2, then the segmentation network in step 5. The CAM is read by a forward pass, so no gradient is computed to make it.

**SEC** (p.30–32)
1. A weak localization network (CAM) gives class seeds for each training image. (paper) Foreground seeds come from CAM, background seeds from a saliency map.
2. The segmentation CNN predicts class scores for every pixel.
3. Seeding loss: match the predictions on the seed pixels.
4. Expansion loss: pool the pixel predictions with GWRP and compare them with the image-level labels.
5. Constrain-to-boundary loss: compare the predictions with their CRF-refined version on the downscaled image.
6. Sum the three losses and train.
- **Target comes from**: CAM seeds (step 3), human image labels (step 4), and the CRF output of the network's own prediction (step 5).
- **Gradient goes to**: the segmentation CNN. The seeds are fixed. The CRF has no learned weights, but (paper) gradients are back-propagated through it, so the CRF output is not a stop-gradient target (p.30 gray arrows pass through the CRF).

**Saliency-guided (Oh)** (p.33–34)
1. Seeder: generate class-specific seeds from CAM.
2. Saliency: estimate foreground regions with a saliency model.
3. Guide labeller: fuse the seeds and the saliency into class-specific pseudo masks.
4. Train the segmenter convnet on the pseudo masks with a dense classifier loss.
- **Target comes from**: the guide labeller's fused pseudo masks. (paper) The saliency model was trained on another dataset.
- **Gradient goes to**: the segmenter convnet.

**SEAM** (p.35–36)
1. Apply an affine transform to the image.
2. Pass the original and the transformed image through the shared CNN (Siamese network) to get CAMs.
3. PCM refines each CAM with pixel-wise feature affinity.
4. Losses: classification loss, ER loss between the transformed original CAM and the CAM of the transformed image, ECR loss across the two views (L1).
- **Target comes from**: human image labels (classification), and each branch's own CAM after compensating for the transform (ER, ECR).
- **Gradient goes to**: the shared CNN of both branches.

**Hide-and-Seek** (p.37–38)
1. Split each training image into a grid of patches.
2. At each epoch, hide a random set of patches.
3. Train the classifier on the partly hidden images with image labels.
4. At test time, compute the CAM on the full image (paper).
- **Target comes from**: human image labels.
- **Gradient goes to**: the classifier.

**Adversarial erasing** (p.39–41)
1. Train a classification network on the image ($t = 1$) and compute its heatmap $H_t$.
2. Threshold $H_t$ to get the mined region $F_t$.
3. Erase $F_t$ from the image to get $I_{t+1}$.
4. Train again on the erased images and repeat (steps 1–3, $t = 1, 2, 3$).
5. Merge the mined regions into the object region, used as segmentation supervision.
- **Target comes from**: human image labels for each classification step; the merged erased regions for segmentation.
- **Gradient goes to**: the classification network at each step (paper: a new one per step), then the segmentation network.

**Webly supervised (Shen)** (p.45–50)
1. Train Model-T on clean target image-level labels.
2. Use Model-T to filter noisy web images.
3. Train Web-SEC (pixel labeler) on the filtered web images, with masks generated automatically from their clean single-object images.
4. Apply Web-SEC to the target images to generate pixel labels.
5. Use Model-T to produce CAMs on the target images.
6. Fuse the two label sources and train the final segmentation model.
- **Target comes from**: human image labels (Model-T), automatically generated web masks (Web-SEC), and the fused labels (final model).
- **Gradient goes to**: Model-T, Web-SEC, and the final segmentation model, each in its own stage.

**Boxly supervised (Khoreva)** (p.55–60)
1. Make an initial mask from each box: a filled rectangle, 20% Bbox (only the central part of the box is labeled, the rest is ignored, p.58), or a better mask inside the box from MCG, GrabCut+, or MCG ∩ GrabCut+.
2. Train the segmentation convnet on the current masks.
3. Predict masks on the training images and use them as pseudo labels for the next round (paper: cleaned with the boxes and a CRF).
4. Repeat steps 2–3 for several training rounds.
- **Target comes from**: the human boxes at first, then the network's own predictions from the previous round.
- **Gradient goes to**: the segmentation convnet.

**CD-CLIP** (p.66–72)
1. Frozen CLIP text and visual encoders give patch–class scores for target classes and super-classes, and initial CAMs (target and super-class).
2. CDA: compute the JS divergence between patch class distributions, correct the initial (MHSA) affinity, and propagate the CAM ($M^c_{re} = M^c_{init}\cdot(A_d)^t$).
3. SBE: average frozen DINO features inside each super-class CAM into a prototype, and compare it with every DINO patch to get the prototype CAM.
4. Turn the refined CAM and the prototype CAM into pseudo labels.
5. Train the decoder with $\mathcal{L}_{seg}$ (CAM pseudo label) $+\ \lambda_1\mathcal{L}_b$ (Dice with the super-class pseudo label, only if $|C_p| \ge 2$).
- **Target comes from**: human image labels (which classes are present) and the frozen CLIP and DINO features (where they are).
- **Gradient goes to**: the decoder. The CLIP and DINO encoders are frozen (p.66 snowflakes).

**Web incremental learning (Liu)** (p.79–83)
1. Retrieve web images with the new class names, and rehearsal images with stored captions of old images.
2. Filter them: domain filtering for out-of-distribution images, caption filtering for irrelevant rehearsal images.
3. Caption the web images and convert caption words into multi-label image labels.
4. Use the previous model (step $t-1$) to produce pixel-level pseudo labels.
5. Train the step-$t$ model on new classes and rehearse old classes, with $\ell_{cls}$ and $\ell_{seg}$.
- **Target comes from**: captions (image-level labels) and the frozen step-$t-1$ model (pixel pseudo labels); (p.80 figure) a localizer in the step-$t$ model also gives pseudo masks.
- **Gradient goes to**: the step-$t$ model, including the localizer. The step-$t-1$ model is frozen (p.80 figure).

**WISH** (p.87–94)
1. Convert each weak label to a prompt latent with SAM's prompt encoder: point → point latent, box → box latent, tag → CAM peaks → point prompts.
2. The Mask2Former-style model predicts, for each object query, a class, a prompt latent $\hat{Z} = \mathcal{H}_{\mathrm{prompt}}(Q)$, and a mask.
3. Compute class, prompt (KLD), and mask costs (closest of SAM's three masks) for every query–label pair.
4. Hungarian matching assigns queries to weakly labeled instances.
5. Train with class, prompt latent, and SAM mask supervision. The CAM head $\mathcal{H}_{\mathrm{cam}}$ is trained with the tag loss $\mathcal{L}_{\mathrm{cam}}$ and with $\mathcal{L}_{\mathrm{self}}=\mathcal{L}_{\mathrm{bce}}(A,\hat M_c)$, where the model's predicted instance masks $\hat M_c$ are the target (self-enhancement, p.88, 91).
- **Target comes from**: human weak labels (tags, points, boxes) translated by the frozen SAM prompt encoder and mask decoder.
- **Gradient goes to**: the segmentation model (encoder, decoder, heads incl. the CAM head). SAM is frozen (p.88 snowflakes).
