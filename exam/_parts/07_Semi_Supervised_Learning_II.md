# 07. Semi-Supervised Learning II (Pseudo Labeling) — Core Ideas & Question Pool

## A. Core ideas (one block per method/paper/concept, in lecture order)

### Pseudo-Label & Entropy Minimization (Lee, ICML 2013; "Bengio et al., NeurIPS 2004", p.4–9)
- **Problem**: How can unlabeled data provide a training target when no label exists?
- **Core idea**: Predict labels for unlabeled samples with the current model, treat **high-confidence** predictions as (hard) pseudo labels, and train on labeled + pseudo-labeled data. Pseudo labeling is the **hard-label approximation of Entropy Regularization**.
- **vs. previous / key difference**: Entropy minimization is a soft penalty on the whole prediction; pseudo labeling converts a confident soft prediction into a hard label ($y^*=\arg\max p(y|x)$). Minimizing entropy moves decision boundaries toward **low-density regions**.
- **Keywords**: high-confidence predictions, hard label, entropy regularization, low-density separation, better-separated clusters, segregation (t-SNE)
- 한 줄: confident 예측을 one-hot target으로 → entropy↓ → boundary가 low-density로 이동.

### Label Propagation (Iscen et al., CVPR 2019, p.10–14)
- **Problem**: Plain pseudo labeling predicts each sample independently and ignores the data manifold.
- **Core idea**: Build a (kNN) **similarity graph in the feature space** and **propagate** labels from labeled samples to nearby unlabeled samples; pseudo labels come from neighborhood relationships (diffusion), not from the network's softmax.
- **vs. previous / key difference**: Pseudo labeling = independent, **confidence-based**; Label Propagation = neighborhood, **similarity-based**, leverages manifold structure (cluster assumption).
- **Keywords**: similarity graph, feature space, neighborhood relationships, smoother pseudo labels, cluster assumption, expensive for large datasets, depends on feature quality
- 한 줄: feature-space graph 위에서 label을 퍼뜨림. 대신 graph 비용 크고 feature 질에 의존.

### Self-Training & Noisy Student (Xie et al., CVPR 2020, p.15–19)
- **Problem**: How to iterate pseudo labeling so the model keeps improving?
- **Core idea**: Iterative **teacher–student** process: teacher predicts pseudo labels, student learns from labeled + pseudo-labeled data and becomes the next teacher. Noisy Student trains an **equal-or-larger student with added noise** (RandAugment, Dropout, Stochastic Depth) so it does not simply copy the teacher.
- **vs. previous / key difference**: Plain self-training just adds high-confidence predictions to the labeled set; Noisy Student makes the **student** learn under **stronger perturbations than the teacher** → student becomes more robust than the teacher.
- **Keywords**: teacher/student, iterative, RandAugment, Dropout, Stochastic Depth, larger student > teacher, soft pseudo labels > hard labels
- 한 줄: noise는 student에만. student ≥ teacher 크기, 반복하며 점점 좋아짐.

### Reducing Confirmation Bias (Arazo et al., IJCNN 2020, p.20–23)
- **Problem**: **Confirmation bias** — an imperfect teacher produces incorrect pseudo labels, the student imitates them, errors accumulate and performance degrades.
- **Core idea**: Regularize so wrong pseudo labels are hard to memorize: **MixUp with soft labels**, **oversampling labeled data** (minimum number of labeled samples per mini-batch), **strong data augmentation**, **dropout**.
- **vs. previous / key difference**: Shows pseudo labeling alone can be competitive when confirmation bias is handled (two moons: mixup + minimum labeled per mini-batch both needed).
- **Keywords**: confirmation bias, imperfect teacher, MixUp, soft labels, oversampling labeled data, dropout
- 한 줄: regularization(MixUp 등)으로 틀린 pseudo label의 자기강화를 억제.

### Meta Pseudo Labels (Pham et al., CVPR 2021, p.24–27)
- **Problem**: Conventional self-training assumes the teacher generates good pseudo labels, but an imperfect teacher gives incorrect supervision.
- **Core idea**: Instead of fixing the teacher, **learn the teacher**: bi-level optimization — (inner) student is trained on teacher's pseudo labels, (outer) teacher is updated so the resulting student achieves **lower supervised loss on labeled data**.
- **vs. previous / key difference**: Arazo reduces bias by regularizing the student; Meta PL uses **feedback from the student's performance** (evaluated with correct labels) to train the teacher.
- **Keywords**: learn the teacher, bi-level optimization, inner/outer optimization, student feedback, labeled data evaluation, "importance of feedback"
- 한 줄: teacher를 student의 labeled 성능 feedback으로 학습.

### Pseudo Labeling + Consistency Regularization (p.28–29)
- **Problem**: Pseudo labeling and consistency training each capture only part of the SSL assumptions.
- **Core idea**: Combine both — make a target from one view and train another view to match it.
- **Keywords**: hybrid, pseudo labeling + consistency training
- 한 줄: 한 view로 label, 다른 view로 학습 → 둘을 한 loss로.

### MixMatch (Berthelot et al., NeurIPS 2019, p.30–35)
- **Problem**: Many successful SSL ideas exist separately.
- **Core idea**: **Unifies** consistency regularization, entropy minimization, pseudo labeling, label sharpening, and MixUp into one **holistic** framework: guess labels by **averaging predictions over K augmentations**, then **sharpen** with temperature $T$, then MixUp labeled & unlabeled; CE for labeled, **L2** for unlabeled.
- **vs. previous / key difference**: Does not propose a new component; it is the first holistic framework combining multiple SSL ideas.
- **Keywords**: holistic, label guessing, K augmentations, sharpening (temperature T), MixUp, $\lambda'=\max(\lambda,1-\lambda)$, L2 loss
- 한 줄: 평균 → sharpen → MixUp → L2. 새 부품 없이 통합. ablation 최대 하락은 MixUp 제거.

### ReMixMatch (Berthelot et al., 2020, p.36–41)
- **Problem**: MixMatch suffers from **class imbalance in pseudo labels** and **unstable pseudo labels under strong augmentations**.
- **Core idea**: **Distribution Alignment** (rescale prediction by ratio labeled prior / estimated pseudo-label distribution), **Augmentation Anchoring** (pseudo label from a **weakly augmented anchor**, consistency on multiple **strong** augmentations), plus **self-supervised rotation loss**.
- **vs. previous / key difference**: New vs MixMatch = DA, AA, Rotation loss (label guessing, sharpening, MixUp, consistency are shared).
- **Keywords**: distribution alignment, target prior, augmentation anchoring, weak anchor, strong augmentation, rotation prediction
- 한 줄: MixMatch + DA + weak→strong anchoring + rotation. FixMatch의 원형.

### FixMatch (Sohn et al., NeurIPS 2020 — slide says 2021, p.42–48)
- **Problem**: MixMatch/ReMixMatch pipelines became increasingly complicated.
- **Core idea**: Only three simple ideas — **weak augmentation** to generate a reliable pseudo label, a **confidence threshold** to discard uncertain predictions, **strong augmentation** to train the model to predict the same (hard, one-hot) label with CE. "Predict on an easy example, learn from a difficult one."
- **vs. previous / key difference**: Much simpler (no sharpening, no MixUp, no DA); confidence thresholding is more effective than temperature sharpening.
- **Keywords**: weak→strong, confidence threshold $\tau$, one-hot pseudo label, cross entropy, flip-and-shift, RandAugment/CTAugment
- 한 줄: weak에서 τ 넘는 hard label → strong에 CE. 이것만으로 충분.

### DivideMix (Li et al., ICLR 2020 — slide says ICML, p.49–55)
- **Problem**: SSL assumes labeled samples are clean, but labeled data may contain **noisy labels**.
- **Core idea**: Reformulate noisy-label learning as SSL — fit a **2-component GMM on per-sample training loss** (low-loss → clean, high-loss → noisy), discard the noisy samples' **labels** (keep them as unlabeled data), then train with **MixMatch**. Two networks **co-divide**: each uses the clean/noisy division made by **the other** network to reduce confirmation bias.
- **vs. previous / key difference**: Not a new learning algorithm; a way to generate labeled/unlabeled sets for MixMatch, plus label co-refinement (clean) and co-guessing (noisy, average of two networks).
- **Keywords**: noisy labels, two networks, co-divide, GMM, per-sample loss, co-refinement, co-guessing
- 한 줄: loss GMM으로 clean/noisy 분할, noisy는 label만 버리고 unlabeled로. 분할은 상대 네트워크가.

### Pre-training + SSL: Zoph et al. & SimCLRv2 (NeurIPS 2020, p.56–63)
- **Problem**: How do self-supervised pre-training and semi-supervised learning relate?
- **Core idea**: They are **complementary**; SSL is not a replacement for pre-training. Zoph: pre-training's benefit **decreases** with stronger augmentation and more labeled data, while **self-training remains effective** even with strong augmentation and abundant labels. SimCLRv2: **(1) self-supervised pre-training (task-agnostic big model) → (2) supervised fine-tuning → (3) self-training / distillation** on unlabeled data.
- **vs. previous / key difference**: SimCLRv2 distillation uses the teacher's **soft** (temperature-scaled) predictions, not thresholded hard labels.
- **Keywords**: complementary, labeled data scarce, task mismatch, self-training more expensive, task-agnostic backbone, knowledge distillation
- 한 줄: pre-training은 label 적을 때 효과 큼, self-training은 label 많아도 효과. 둘 다 쓰는 게 최신 파이프라인.

### FreeMatch (Wang et al., ICLR 2023 — slide says ICML, p.64–75)
- **Problem**: FixMatch's **fixed** threshold is the same throughout training and for every class — early on it rejects most unlabeled samples; it ignores the model's learning status.
- **Core idea**: **Self-Adaptive Thresholding (SAT)**: global threshold = **EMA of average confidence** on unlabeled data (starts at $1/C$, increases as the model gets confident), modulated per class by the MaxNorm of the EMA of class probabilities (easy classes higher, difficult classes lower). **Self-Adaptive Fairness (SAF)** prevents predictions from concentrating on a few classes.
- **vs. previous / key difference**: Fixed, manually selected $\tau$ → adaptive $\tau_t(c)$ estimated from unlabeled predictions; no additional model inference.
- **Keywords**: self-adaptive threshold, learning status, EMA, $\tau_0=1/C$, class-specific threshold, MaxNorm, fairness, quality and quantity
- 한 줄: threshold를 낮게 시작해 모델 confidence 따라 올리고, class별로 다르게.

### AllMatch (Wu et al., IJCAI 2024, p.76–91)
- **Problem**: Threshold-based methods **discard low-confidence samples** (no training signal), yet discarded samples' top-5 accuracy is ~99–100%.
- **Core idea**: **CAT** — global threshold same as FreeMatch, but class-specific scaling from **classifier weight norm** $\|W_c\|$. **BCC** — split classes into **candidate (top-$k_i$ from the weak view) vs negative** groups and enforce the same binary division on the strong view, applied to **every** unlabeled sample.
- **vs. previous / key difference**: FreeMatch = prediction-based local threshold, discards low-confidence samples, top-1 pseudo label; AllMatch = classifier-weight-based, uses every sample, candidate vs negative classes.
- **Keywords**: low confidence ≠ no information, classifier weight norm, candidate/negative classes, binary classification consistency, adaptive $k_i$, upper bound $K$, 100% utilization
- 한 줄: 버리던 샘플도 "정답은 이 후보 안에 있다"는 coarse supervision으로 활용.

### IFMatch (Wu & Cui, ECCV 2024, p.92–104)
- **Problem**: Every sample above the threshold gets the **same** strong augmentation, but samples differ in difficulty — many **naive samples** have ≈0 consistency loss.
- **Core idea**: Expand weak-to-strong consistency from image space to **feature space** with two student branches sharing one pseudo label: Branch I = weak image + **strong feature** perturbation (high fixed threshold), Branch II = strong image + **optional weak feature** perturbation. **Confidence-Based Identification (CBI)** adds weak feature perturbation only to samples that **stay confident after strong image augmentation** ("easy").
- **vs. previous / key difference**: Not a new pseudo-label generation method — a plug-in that improves **how** selected pseudo labels are used; incorporates both image and feature perturbation (does not replace one with the other).
- **Keywords**: naive vs challenging samples, feature perturbation, Position A (after residual block, strong) / Position B (inside residual branch, weak), CBI, target confidence, plug-in
- 한 줄: 샘플별로 perturbation 강도를 조절. 쉬운 샘플엔 더 어렵게, 어려운 샘플엔 그대로.

## B. Must-know definitions & distinctions
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

## C. Question pool (22 questions)

### Q1 [T/F]
Pseudo labeling can be viewed as the hard-label approximation of entropy regularization.
**Answer**: True — confident soft prediction을 argmax hard label로 바꾼 것 (p.7)

### Q2 [T/F]
Minimizing prediction entropy on unlabeled data moves the decision boundary toward high-density regions of the data.
**Answer**: False — **low**-density regions로 이동, better-separated clusters (p.6)

### Q3 [T/F]
Compared with pseudo labeling, Label Propagation is confidence-based and predicts each sample independently.
**Answer**: False — 그건 Pseudo Labeling. LP는 **similarity-based**, neighborhood relationships 활용 (p.13)

### Q4 [T/F]
In Noisy Student, noise such as RandAugment, Dropout, and Stochastic Depth is added when training the student, which prevents the student from simply copying the teacher.
**Answer**: True — student에만 noise → 더 robust한 decision boundary (p.17)

### Q5 [T/F]
In Meta Pseudo Labels, the teacher is updated so that the student trained on its pseudo labels achieves a lower supervised loss on labeled data.
**Answer**: True — outer optimization, evaluation is performed using correct labels (p.25)

### Q6 [T/F]
The key contribution of MixMatch is proposing a single new component, rather than combining existing SSL techniques.
**Answer**: False — "Rather than proposing a new component", 여러 SSL 기법을 **unify**한 holistic framework (p.30)

### Q7 [T/F]
In ReMixMatch's Augmentation Anchoring, pseudo labels are generated from strongly augmented images and consistency is enforced on weakly augmented versions.
**Answer**: False — weak ↔ strong 뒤바뀜. **weak** anchor로 label 만들고 **strong** view에 consistency (p.38)

### Q8 [T/F]
According to FixMatch's ablation, once low-confidence predictions are filtered out by thresholding, temperature sharpening provides little additional benefit.
**Answer**: True — confidence thresholding이 sharpening보다 효과적 (p.48)

### Q9 [T/F]
DivideMix completely discards the samples it identifies as noisy and never uses them for training.
**Answer**: False — **GT label만** 버리고 unlabeled data로 계속 사용 (p.53)

### Q10 [T/F]
Zoph et al. found that self-training remains effective even with strong augmentation and abundant labeled data, whereas the benefit of pre-training diminishes as more labeled data become available.
**Answer**: True — Key Findings 그대로 (p.58)

### Q11 [T/F]
In FreeMatch's class-specific thresholding, difficult classes receive relatively high thresholds and easy classes receive relatively low thresholds.
**Answer**: False — 반대. Easy → high, **difficult → low** (p.70)

### Q12 [T/F]
In AllMatch, the candidate classes for Binary Classification Consistency are determined from the weak view, and BCC is applied to every unlabeled sample.
**Answer**: True — same candidate–negative division을 strong view에 적용, 모든 unlabeled sample에 적용 (p.86)

### Q13 [T/F]
IFMatch expands weak-to-strong consistency from the image space to the feature space by replacing image perturbation with feature perturbation.
**Answer**: False — replace가 아니라 **incorporate both**: "Not to replace image perturbation with feature perturbation, but rather to incorporate both types of perturbation" (Branch I: weak image + strong feature, Branch II: strong image + optional weak feature) (p.97)

### Q14 [MCQ]
Which of the following is NOT one of the methods proposed by Arazo et al. to reduce confirmation bias?
(A) MixUp with soft labels
(B) Oversampling labeled data
(C) Dropout regularization
(D) Updating the teacher based on how much the student improves
**Answer**: (D) — 그건 Meta Pseudo Labels의 key idea (p.21, p.24)

### Q15 [MCQ]
How does MixMatch produce the guessed label for an unlabeled sample?
(A) It takes the argmax of a single weakly augmented prediction and keeps it only if it exceeds a confidence threshold
(B) It averages the model's predictions over K augmentations and then sharpens the result with a temperature T
(C) It diffuses labels from labeled samples over a kNN similarity graph in feature space
(D) It uses a teacher network that is trained with feedback from the student's labeled loss
**Answer**: (B) — label guessing(평균) → sharpening. (A) FixMatch, (C) Label Propagation, (D) Meta PL (p.32)

### Q16 [MCQ]
In ReMixMatch's Distribution Alignment, how is the prediction for an unlabeled sample adjusted?
(A) It is multiplied by the ratio of the labeled dataset's class distribution to the estimated class distribution of all pseudo labels, then normalized
(B) It is multiplied by the ratio of the estimated pseudo-label distribution to the labeled class distribution, then normalized
(C) It is replaced by a one-hot label if its maximum probability exceeds a fixed threshold
(D) It is averaged with the prediction of a second network trained in parallel
**Answer**: (A) — yellow(labeled prior) / green(pseudo-label 분포) 비율로 rescale → 과대 class 억제, 과소 class 증폭 (p.37)

### Q17 [MCQ]
Which statement correctly describes FixMatch?
(A) Pseudo labels are made from strongly augmented images and the model is trained on weakly augmented images
(B) It relies on sharpening, MixUp, and distribution alignment to obtain stable targets
(C) A pseudo label is generated from a weakly augmented image, kept only if its confidence is high, and the model is trained to predict that label on a strongly augmented image
(D) It uses a separate EMA teacher network to generate soft targets with temperature scaling
**Answer**: (C) — three simple ideas: weak augmentation / confidence threshold / strong augmentation (p.43, p.45)

### Q18 [MCQ]
In DivideMix, how is the labeled dataset divided into clean and noisy samples?
(A) Each network fits a 2-component GMM to its per-sample loss; low-loss samples are clean, and each network uses the division made by the other network
(B) Each network fits a 2-component GMM to its per-sample loss; high-loss samples are clean, and each network uses its own division
(C) Samples whose softmax confidence exceeds a fixed threshold are treated as clean
(D) A kNN graph is built, and samples far from their neighbors are treated as noisy
**Answer**: (A) — Low-loss → Clean, 상대 network의 판단을 사용해 confirmation bias 감소 (p.51)

### Q19 [MCQ]
What is the correct order of the three-step procedure in SimCLRv2-style modern pipelines?
(A) Supervised fine-tuning → Self-supervised pre-training → Self-training
(B) Self-training → Supervised fine-tuning → Self-supervised pre-training
(C) Self-supervised pre-training → Supervised fine-tuning → Self-training (distillation)
(D) Self-supervised pre-training → Self-training → Supervised fine-tuning
**Answer**: (C) — task-agnostic backbone pre-training → small labeled data로 fine-tuning → pseudo label/distillation (p.59)

### Q20 [MCQ]
How does FreeMatch set its self-adaptive global threshold?
(A) It is fixed and manually selected throughout training, as in FixMatch
(B) It is the EMA of the average confidence on unlabeled data, starting from 1/C and increasing as the model becomes more confident
(C) It is the classifier weight norm of the most confident class
(D) It starts high and decreases over training to admit more samples
**Answer**: (B) — $\tau_0=1/C$, average confidence의 EMA, confidence 오를수록 threshold 상승. (C)는 AllMatch의 local scaling 소재 (p.69, p.82)

### Q21 [MCQ]
What is the main difference between AllMatch and FreeMatch?
(A) AllMatch uses a fixed global threshold, while FreeMatch uses an adaptive one
(B) AllMatch uses a classifier-weight-based local threshold and also learns from low-confidence samples via candidate-versus-negative consistency
(C) AllMatch discards low-confidence samples, while FreeMatch uses every unlabeled sample
(D) AllMatch replaces pseudo labeling with graph-based label propagation
**Answer**: (B) — global threshold는 동일, local은 prediction-based → classifier-weight-based, low-confidence 샘플은 BCC로 활용 (p.80)

### Q22 [MCQ]
In IFMatch's Confidence-Based Identification (CBI), which samples receive the additional weak feature perturbation?
(A) Samples whose confidence becomes low after strong image augmentation
(B) All samples below the confidence threshold
(C) Samples whose confidence remains high after strong image augmentation ("easy" samples)
(D) Every unlabeled sample, regardless of confidence
**Answer**: (C) — strong aug 후에도 confidence 높으면 아직 easy → weak feature perturbation 추가. difficult 샘플엔 안 줌 (p.102)

## D. Fill-in-the-blank pool (14 questions)

### F1 [Blank]
Pseudo labeling treats (1) ______-confidence predictions of the current model as labels for unlabeled samples. It can be viewed as the hard-label approximation of (2) ______ regularization.
**Answer**: (1) high (2) entropy (accept: entropy minimization) — low entropy = confident prediction. 충분히 confident한 soft 예측 $[0.96, 0.03, 0.01]$을 $y^*=\arg\max_y p(y|x)$ one-hot으로 바꾸면 entropy minimization의 hard 근사. entropy를 줄이면 decision boundary가 **low-density** region으로 이동 → better-separated clusters. (p.4–7)

### F2 [Blank]
Label Propagation constructs a similarity graph in the (1) ______ space and propagates labels from labeled samples to nearby unlabeled samples. Compared with pseudo labeling, which is confidence-based, Label Propagation is (2) ______-based.
**Answer**: (1) feature (accept: embedding) (2) similarity — PL: 각 sample 독립 예측 / confidence-based / data manifold 무시 vs LP: neighborhood relationships / similarity-based / manifold structure 활용 (cluster assumption). 한계: graph 구축 필요 → large dataset에 expensive, feature quality에 의존. (p.10, p.13, p.14)

### F3 [Blank]
Self-training is an iterative teacher–student process: the student learns from labeled data and pseudo labels and becomes the (1) ______ in the next iteration. Noisy Student adds noise (RandAugment, Dropout, Stochastic Depth) when training the (2) ______, so that it does not simply copy the teacher.
**Answer**: (1) teacher (2) student — noise는 student 학습 때만. student가 teacher보다 강한 perturbation 하에서 배우니 더 robust한 decision boundary → student가 teacher보다 좋아지고, 반복할수록 개선. Key findings: larger student > teacher, soft pseudo labels > hard labels. **student↔teacher 뒤바꾼 함정** 주의. (p.15, p.17, p.19)

### F4 [Blank]
(1) ______ bias: pseudo labels from an imperfect teacher can be incorrect, the student is trained to imitate them, and the student reinforces the teacher's mistakes. Arazo et al. reduce it with MixUp with soft labels, (2) ______ labeled data, strong data augmentation, and dropout.
**Answer**: (1) Confirmation (2) oversampling (accept: oversample) — 틀린 pseudo label이 누적되며 성능이 점점 하락. Arazo는 regularization으로 틀린 label을 외우기 어렵게 만듦. oversampling labeled data = mini-batch마다 labeled sample 최소 개수 보장 ("A minimum number of labeled samples per mini-batch"). (p.20, p.21, p.62)

### F5 [Blank]
Meta Pseudo Labels does not keep the teacher fixed: the teacher is updated so that the student trained on its pseudo labels achieves a lower supervised loss on (1) ______ data. In this bi-level scheme, the student update is the (2) ______ optimization, and the teacher update is the outer optimization.
**Answer**: (1) labeled (2) inner — "Instead of fixing the teacher, learn the teacher to generate better pseudo labels." teacher를 직접 평가하지 않고, 그 pseudo label로 배운 student가 correct label로 평가받은 성능을 feedback으로 teacher를 갱신. Arazo(student를 regularize)와 달리 MPL은 **teacher 자체를 학습**. (p.24, p.25)

### F6 [Blank]
MixMatch guesses a label for an unlabeled sample by (1) ______ the model's predictions over $K$ augmentations, and then applies (2) ______ with temperature $T$ to make the guessed label more confident.
**Answer**: (1) averaging (accept: average) (2) sharpening (accept: sharpen, label sharpening) — $\bar q_b=\frac1K\sum_{k}p_{\text{model}}(y|\hat u_{b,k};\theta)$ → $\mathrm{Sharpen}(p,T)_i=p_i^{1/T}/\sum_j p_j^{1/T}$ ($T$↓ → entropy↓). 순서 **average → sharpen**. 그 뒤 labeled+unlabeled MixUp, loss는 labeled CE + unlabeled L2. 새 component 없이 consistency·entropy min·pseudo label·sharpening·MixUp을 묶은 first **holistic** framework. (p.30, p.32, p.33)

### F7 [Blank]
ReMixMatch addresses two limitations of MixMatch: class imbalance in pseudo labels is handled by (1) ______ Alignment, and unstable pseudo labels under strong augmentations are handled by Augmentation (2) ______, which generates the pseudo label from a weakly augmented image.
**Answer**: (1) Distribution (2) Anchoring — DA: $\mathrm{Normalize}\big(p_\theta(y|u)\,p(y)/\tilde p(\hat y)\big)$, labeled class prior ÷ pseudo-label 분포 비율로 rescale → 과대 class 억제, 과소 class 증폭. AA: weak anchor에서 pseudo label을 한 번 만들고 여러 strong view가 그걸 맞춤 (target generation과 consistency training 분리). 추가로 self-supervised rotation loss. (p.36–40)

### F8 [Blank]
Instead of combining many techniques, FixMatch relies on only three simple ideas: (1) ______ augmentation to generate a reliable pseudo label, a confidence (2) ______ to discard uncertain predictions, and strong augmentation to train the model to predict the same label.
**Answer**: (1) weak (accept: weakly) (2) threshold (accept: thresholding, $\tau$) — "Predict on an easy example, learn from a difficult one." weak = flip-and-shift, strong = RandAugment/CTAugment 등. sharpening·MixUp·DA 없이 단순. ablation: confidence thresholding > temperature sharpening, pseudo label은 반드시 weak에서 (strong에서 만들면 unreliable). **weak↔strong 뒤집은 함정** 주의. (p.43–45, p.48)

### F9 [Blank]
DivideMix fits a two-component (1) ______ to the per-sample training loss (low-loss → clean, high-loss → noisy). In (2) ______, each of the two networks uses the clean/noisy division made by the other network instead of its own judgment.
**Answer**: (1) GMM (accept: Gaussian Mixture Model) (2) co-divide (accept: Co-Divide, co-division) — 문제의식: labeled data에도 noisy label이 섞일 수 있음 → noisy-label learning을 SSL 문제로 재정의. 학습 초반 clean은 loss가 빨리 줄고 noisy는 높게 남아 loss histogram이 두 cluster → GMM. noisy 샘플은 **GT label만** 버리고 unlabeled로 MixMatch 학습. 상대 network 판단을 쓰는 이유 = confirmation bias 감소. (p.49–53)

### F10 [Blank]
Self-supervised pre-training and semi-supervised learning are highly (1) ______; SSL is not a replacement for pre-training. In the SimCLRv2-style three-step procedure, self-supervised pre-training and supervised fine-tuning are followed by (2) ______, which exploits unlabeled data through knowledge distillation.
**Answer**: (1) complementary (2) self-training (accept: self training) — Zoph: pre-training은 label이 scarce할 때 가장 효과적이고, label이 많아지거나 aug가 강해지면 이득 감소. self-training은 strong aug + abundant label에서도 효과 (대신 conventional fine-tuning보다 계산 비쌈). 순서: self-supervised pre-training → supervised fine-tuning → self-training. (p.56, p.58, p.59)

### F11 [Blank]
FixMatch's confidence threshold (0.95) is (1) ______: it does not adapt to the model's learning status and is the same for every iteration and every class. FreeMatch's Self-Adaptive Thresholding instead computes the average confidence of unlabeled samples and accumulates it with an (2) ______ to obtain the global threshold, which increases as the model becomes more confident.
**Answer**: (1) fixed (accept: constant, static) (2) exponential moving average (accept: EMA) — 초반엔 model confidence가 낮아 고정 0.95가 unlabeled sample 대부분을 reject → unsupervised signal 거의 없음. FreeMatch: $\tau_0=1/C$에서 시작해 learning status 따라 상승 + class별 local scaling (easy class → 높게, difficult class → 낮게). "The desirable threshold depends on the model's learning status." (p.66, p.67, p.69, p.70)

### F12 [Blank]
In FreeMatch, Self-Adaptive Thresholding (SAT) selects samples, while Self-Adaptive ______ (SAF) prevents the selection process from collapsing toward a few classes.
**Answer**: Fairness — label이 극소일 때 초반 예측이 소수 class에 쏠림 → dominant class가 pseudo label을 더 받음 → 더 강해짐 (initial bias가 self-reinforce). SAF는 diverse class prediction을 유도. Total loss $\mathcal L=\mathcal L_s+w_u\mathcal L_u+w_f\mathcal L_f$. (p.68, p.72, p.73)

### F13 [Blank]
AllMatch's Class-Specific Adaptive Threshold (CAT) uses the (1) ______ of each classifier weight to estimate how well each class has been learned. Its Binary Classification Consistency (BCC) divides classes into candidate and (2) ______ groups, so that even low-confidence samples provide a training signal.
**Answer**: (1) norm (accept: weight norm, magnitude) (2) negative — 문제의식: threshold 아래라 버려지는 sample도 top-5 안에는 정답이 거의 다 있음 → "Low confidence does not mean no semantic information." global threshold는 FreeMatch와 같고, local adjustment만 prediction-based → **classifier-weight-based**. BCC: weak view에서 정한 candidate/negative 분할을 strong view에서도 유지, 모든 unlabeled sample에 적용 → 100% utilization. (p.78–83, p.86, p.88)

### F14 [Blank]
IFMatch expands weak-to-strong consistency from the image space to the (1) ______ space. Its Confidence-Based Identification (CBI) adds an extra weak perturbation only to samples whose confidence remains (2) ______ after strong image augmentation.
**Answer**: (1) feature (2) high — 문제의식: threshold를 넘은 sample도 난이도가 제각각이고, naive sample은 strong aug 후에도 consistency loss ≈ 0. confidence가 높게 유지 = 아직 easy → weak feature perturbation 추가 / 낮아짐 = 이미 difficult → 추가 안 함. image perturbation을 **replace가 아니라 둘 다 incorporate**. pseudo-label confidence = target을 쓸지 여부, target confidence = 얼마나 세게 perturb할지. (p.96, p.97, p.102, p.104)

<!-- REVIEWED | fixes: 2 | reviewer: opus -->

<!-- AUGMENTED 2026-10-22 | C fixes: 0 | C added: 0 | D blanks: 14 | author: opus
(no C fixes: all 22 answers and p.N citations re-checked against slides p.4-105, incl. image-only pages p.32, p.33, p.37, p.69, p.70, p.84, p.87; every Section A method already has >=1 question)
-->

<!-- VERIFIED 2026-10-22 | checked: 36 | fixed: 2 | deleted: 0 | reviewer: opus
F11: blank (1) "FixMatch uses a ______ confidence threshold (0.95)" also admitted "high" (true and grammatical); reworded to "FixMatch's confidence threshold (0.95) is ______: it does not adapt to the model's learning status ...", added accept: constant, static.
F14: second sentence ("adds weak feature perturbation") gave away blank (1) "feature"; changed to "adds an extra weak perturbation".
All other C/D items: key, p.N, and distractors confirmed against slides (image pages p.21, p.22, p.32, p.33, p.35, p.69, p.70, p.73, p.78 checked). Sections A/B: no slide contradictions found.
-->
