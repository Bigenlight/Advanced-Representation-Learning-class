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

<!-- REVIEWED | fixes: 2 | reviewer: opus -->
