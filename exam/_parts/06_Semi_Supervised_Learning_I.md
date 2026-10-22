# 06. Semi-Supervised Learning I — Core Ideas & Question Pool

## A. Core ideas (one block per method/paper/concept, in lecture order)

### Four ways to overcome limited labeled data (p.4–7)
- **Problem**: Labeled data are scarce and expensive, so we need ways to exploit unlabeled data or limited labeling budgets.
- **Core idea**: (1) Pre-training + fine-tuning: pre-train a task-agnostic model on a large unsupervised corpus, then fine-tune with a small labeled set; (2) Active learning: select the most valuable unlabeled samples to be labeled next under a limited budget; (3) Pre-training + dataset auto-generation: use a capable pre-trained model to auto-generate many more labeled samples (popular in language); (4) Semi-supervised learning: learn from labeled and unlabeled samples together.
- **vs. previous / key difference**: SSL trains on both data types **jointly**, whereas pre-train/fine-tune is **sequential (2-stage)** and active learning still asks humans for labels.
- **Keywords**: task-agnostic, limited budget, auto-generate, few-shot learning, together
- 한 줄: label 부족 해법 4개 — SSL만 labeled+unlabeled를 "한 번에 같이" 학습.

### SSL objective & key question (p.8–10)
- **Problem**: How to combine scarce labeled data with plentiful unlabeled data in one training objective.
- **Core idea**: $\mathcal L = \mathcal L_s + \mu(t)\,\mathcal L_u$ — supervised loss + unsupervised loss + weighting term $\mu(t)$, which is **increasing in time**. The key question is how to design $\mathcal L_u$: **Consistency Regularization** (this lecture) or **Pseudo Label** (next lecture).
- **Keywords**: supervised loss, unsupervised loss, weighting term, increasing in time, consistency regularization, pseudo label
- 한 줄: $\mathcal L_u$를 어떻게 만드냐가 SSL의 핵심 질문, $\mu(t)$는 점점 키움 (초반 예측은 믿을 수 없으니).

### Hypotheses H1–H4 (p.11–14)
- **Problem**: Why should unlabeled data help at all?
- **Core idea**: H1 Smoothness — nearby samples should have similar predictions / small perturbations should not change the label. H2 Cluster — samples in the same cluster share the same label; boundaries should not split a cluster. H3 Low-density separation — decision boundaries should lie in **low-density** regions. H4 Manifold — high-dimensional data lie on a **low-dimensional manifold**.
- **Keywords**: smoothness, cluster, low-density separation, manifold, decision boundary
- 한 줄: unlabeled data가 $p(x)$의 구조(cluster·density·manifold)를 알려줘서 도움 됨. H1이 consistency의 근거.

### Consistency Regularization — core idea (p.15–16)
- **Problem**: Unlabeled data have no labels, so we need a loss that doesn't use $y$.
- **Core idea**: Predictions should remain consistent under small perturbations; $\mathcal L_u = D(f(x), f(\mathcal A(x)))$. Perturbations come from **data augmentations** or **model stochasticity (e.g., Dropout)**.
- **Keywords**: perturbation, data augmentation, dropout, model stochasticity, consistent predictions
- 한 줄: "perturbation 줘도 예측은 같아야 한다" = H1을 loss로 만든 것.

### Π-model (Sajjadi et al., ICLR 2016 as on slide, p.17–18)
- **Problem**: Need a simple consistency target for unlabeled data.
- **Core idea**: Pass two versions of the same input with **different stochastic augmentation and dropout masks** through the **same network** and make the outputs consistent (MSE).
- **Keywords**: two forward passes, dropout masks, stochastic augmentation, MSE, same network
- 한 줄: 같은 모델에 두 번 forward (다른 noise) → 출력 MSE. 단점은 forward 2회.

### Temporal Ensembling (Laine et al., ICLR 2017, p.19–22)
- **Problem**: Π-model requires two forward passes per input → high computational cost.
- **Core idea**: Use the **historical predictions of the same sample** as the target — an EMA of the model **prediction** in time **per training sample** (with bias correction).
- **vs. previous / key difference**: One forward pass and a more stable target than Π-model, but must store predictions for every sample, targets are updated only once per epoch, and it is hard to scale.
- **Keywords**: EMA of predictions, per training sample, one forward pass, once per epoch, memory
- 한 줄: sample별 과거 예측의 EMA를 target으로 → 싸고 안정적이지만 메모리·epoch 단위 갱신이 약점.

### Mean Teacher (Tarvainen et al., p.23–29)
- **Problem**: Temporal Ensembling is memory-intensive and its targets are updated only once per epoch.
- **Core idea**: Replace the EMA of predictions with an **EMA of model weights**: teacher $\theta' \leftarrow \beta\theta' + (1-\beta)\theta$. The student is trained by gradient descent (supervised + consistency); the teacher is updated **only by EMA, no backpropagation**.
- **vs. previous / key difference**: No per-sample prediction memory, teacher updated every iteration, scales to large datasets.
- **Keywords**: EMA of weights, student, teacher, no backpropagation, $\beta$ 0.99 → 0.999, MSE > KL
- 한 줄: teacher = student weight의 EMA. 실전 팁: input aug 필수, dropout은 student만, β는 초반 0.99 → 후반 0.999, MSE가 KL보다 좋음.

### Noisy Samples as Learning Targets (p.30–31)
- **Problem**: Weak internal noise gives little learning signal on unlabeled data.
- **Core idea**: Apply **strong augmentations** to an unlabeled sample; the prediction should stay the same before and after augmentation, using **the same model** (prediction invariance).
- **vs. previous / key difference**: vs Π-model — input augmentations instead of stochastic perturbations inside the network; vs Mean Teacher — self-consistency between original and augmented inputs instead of an EMA teacher.
- **Keywords**: strong augmentation, RandAugment, back-translation, prediction invariance, self-consistency
- 한 줄: 원본 vs strong aug 버전의 예측을 같은 모델로 맞춤.

### Adversarial Training (Goodfellow et al., ICLR 2015, p.32–33)
- **Problem**: Small, imperceptible perturbations can make a model confidently wrong (panda → gibbon).
- **Core idea**: Add **worst-case** (adversarial) perturbations to the input during training and train the model to still produce the **correct (ground-truth)** prediction; $r_{adv} \approx \epsilon\,\mathrm{sign}(g)$ (FGSM).
- **Keywords**: worst-case perturbation, ground-truth label, robustness, FGSM, $\|r\|\le\epsilon$
- 한 줄: 모델이 가장 헷갈리는 방향의 noise를 넣고도 GT를 맞히게 → robustness (supervised).

### Virtual Adversarial Training — VAT (Miyato et al., TPAMI 2018, p.34–35)
- **Problem**: Adversarial training needs ground-truth labels, so it can't use unlabeled data.
- **Core idea**: Replace the GT label with the model's **own current prediction**, enabling adversarial consistency training on **both labeled and unlabeled** samples.
- **vs. previous / key difference**: AT = GT labels, supervised, robustness; VAT = model predictions, semi-supervised, **local smoothness**. vs RPT: RPT smooths in random directions, VAT only along the most adversarial direction (much more effective).
- **Keywords**: virtual, model prediction, local smoothness, most adversarial direction, RPT
- 한 줄: "virtual" = 진짜 label 대신 모델 예측. 가장 민감한 방향에서도 예측이 안 바뀌게.

### Interpolation Consistency Training — ICT (Verma et al., p.36–38)
- **Problem**: Want consistency exactly where it matters — near the decision boundary.
- **Core idea**: Following **MixUp**, linearly interpolate two **unlabeled** samples; the student's prediction on the mixed input should match the **mix of the Mean Teacher's predictions** on the originals ("mixing inputs should produce mixed predictions").
- **vs. previous / key difference**: MixUp samples tend to lie near the decision boundary → smoother boundaries → aligns with the **Low-Density Separation** assumption.
- **Keywords**: MixUp, interpolation, Mean Teacher, $\lambda\sim\mathrm{Beta}(\alpha,\alpha)$, low-density separation
- 한 줄: 섞은 입력 → 섞은 예측. teacher는 섞기 전, student는 섞은 후.

### Unsupervised Data Augmentation — UDA (Xie et al., NeurIPS 2020, p.39–42)
- **Problem**: Small adversarial/random perturbations regularize only a narrow neighborhood.
- **Core idea**: Apply **strong, label-preserving semantic augmentations (RandAugment)** to unlabeled samples and enforce $p(y|x)\approx p(y|\mathrm{Aug}(x))$.
- **vs. previous / key difference**: Simply replacing adversarial perturbations with strong semantic augmentations beats VAT and ICT — **augmentation quality** is the key factor.
- **Keywords**: RandAugment, label-preserving, diverse, realistic, task-relevant, augmentation quality
- 한 줄: "How we perturb the input matters more than simply adding perturbations."

### PS-MT — Perturbed and Strict Mean Teachers (Liu et al., CVPR 2022, p.43–54)
- **Problem**: Strong perturbations help generalization, but if the teacher is wrong the student reproduces the same error → need more reliable teacher targets first.
- **Core idea**: **Two mean teachers** (only one updated per epoch) whose **ensemble** gives the target; a **strongly perturbed student** (input: strong aug/CutMix; feature: **T-VAT**; network: teachers vs student); and a **strict consistency loss (Conf-CE)** instead of MSE.
- **vs. previous / key difference**: MSE is robust but gives small gradients near convergence (weak supervision); more accurate teachers allow stronger confidence-weighted CE. T-VAT: the **teachers** find the adversarial direction, not the student. CutMix is applied **after** prediction so teachers see natural images.
- **Keywords**: two teachers, teacher ensemble, Conf-CE, T-VAT, CutMix after prediction, reliable targets
- 한 줄: teacher를 먼저 믿을 만하게(2-teacher ensemble) → 그래야 강한 perturbation + 강한 CE loss를 쓸 수 있음.

### Dual Teacher — Switching Temporary Teachers (Na et al., NeurIPS 2023, p.55–65)
- **Problem**: With EMA, the teacher gradually follows the student and becomes too similar (**coupling**) → little new information and inherits the student's errors.
- **Core idea**: Two **temporary** EMA teachers **take turns** (switch every epoch); only the **active** teacher generates targets and is updated, and the student's strong augmentation (ClassMix/CutMix) also changes.
- **vs. previous / key difference**: Diversity without training a second student (vs Dual Student); **switching** beats **ensembling** because ensembled teachers gradually lose their distinct characteristics. The EMA rule stays the same; only the teacher schedule changes.
- **Keywords**: coupling, temporary teachers, active teacher, switching vs ensemble, diversity
- 한 줄: "A teacher should be stable, but it should not become identical to the student."

### S⁴Former (Hu et al., CVPR 2024, p.66–74)
- **Problem**: Image-level perturbations alone may not be enough for a Vision Transformer, which processes patch tokens with self-attention.
- **Core idea**: Regularize the ViT at three levels — **Input: PatchShuffle** (shuffle $M\times M$ blocks before encoding, restore before decoding), **Feature: PASA** (attention modulation on low-confidence patches, guided by teacher confidence), **Output: NCR** (match the distribution over non-pseudo-label classes).
- **vs. previous / key difference**: Keeps the Mean Teacher framework (EMA teacher, weak-to-strong consistency) but adds model-structure-aware perturbations; PatchShuffle gives the largest single gain.
- **Keywords**: PatchShuffle, PASA, Negative Class Ranking, ViT, patch tokens, self-attention
- 한 줄: "Effective consistency regularization should reflect the structure of the model."

## B. Must-know definitions & distinctions
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

## C. Question pool (18 questions)

### Q1 [T/F]
Active learning selects the most valuable unlabeled samples to be labeled next, helping us act smartly with a limited budget.
**Answer**: True — 슬라이드 원문 그대로. (p.5)

### Q2 [T/F]
In the SSL objective $\mathcal L = \mathcal L_s + \mu(t)\mathcal L_u$, the unsupervised loss weight $\mu(t)$ is decreasing in time.
**Answer**: False — $\mu(t)$는 **increasing** in time (초반 예측이 부정확하니 점점 키움). (p.10)

### Q3 [T/F]
The low-density separation assumption states that decision boundaries should lie in high-density regions.
**Answer**: False — boundary는 **low-density** region에 있어야 하고, cluster를 관통하지 말아야 함. (p.13)

### Q4 [T/F]
Temporal Ensembling uses an exponential moving average of the model weights as the consistency target.
**Answer**: False — TE는 sample별 **prediction**의 EMA. weight EMA는 Mean Teacher. (p.19, p.23)

### Q5 [T/F]
In Mean Teacher, the teacher's parameters are updated only by EMA of the student, and no backpropagation is applied to the teacher.
**Answer**: True — Teacher: "Parameters updated only by EMA / No backpropagation". (p.24)

### Q6 [T/F]
For Mean Teacher, KL divergence performs better than MSE as the consistency loss.
**Answer**: False — 반대로 **MSE performs better than KL divergence**. (p.28)

### Q7 [T/F]
VAT replaces the ground-truth label used in adversarial training with the model's own current prediction, so the VAT loss can be applied to unlabeled samples as well.
**Answer**: True — "If GT is known" → use current model prediction; VAT loss applies to both labeled and unlabeled samples. (p.34)

### Q8 [T/F]
In ICT, random MixUp samples are likely to lie near the decision boundary, so enforcing consistency on them aligns with the low-density separation assumption.
**Answer**: True — 원문 그대로. (p.37)

### Q9 [T/F]
In PS-MT, both teachers are updated from the student in every epoch so that they stay identical.
**Answer**: False — **only one teacher is updated in each epoch** → 두 teacher가 서로 다른 temporal history를 유지. (p.47)

### Q10 [T/F]
In Dual Teacher, only the active teacher generates the target for the current epoch and is updated from the current student.
**Answer**: True — 원문 그대로; inactive teacher는 그 epoch 동안 멈춰 있음. (p.59)

### Q11 [MCQ]
The statement "Small perturbations should not change the label" belongs to which SSL hypothesis?
(A) Cluster assumption
(B) Smoothness assumption
(C) Low-density separation
(D) Manifold assumption
**Answer**: (B) — H1 Smoothness: nearby samples → similar predictions, small perturbations → same label. Consistency regularization의 직접 근거. (p.11)

### Q12 [MCQ]
What is the main motivation of Temporal Ensembling over the Π-model?
(A) The Π-model cannot use unlabeled data.
(B) The Π-model stores a prediction for every training sample, which is memory intensive.
(C) The Π-model requires two forward passes for every input, which is computationally expensive.
(D) The Π-model uses ground-truth labels to find adversarial perturbations.
**Answer**: (C) — "Π-model requires two forward passes for every input → high computational cost". (B)는 오히려 TE의 한계(→ Mean Teacher의 동기). (p.19, p.22)

### Q13 [MCQ]
Compared with Mean Teacher, how does the "noisy samples as learning targets" approach form its consistency target?
(A) It uses self-consistency between the original and strongly augmented inputs with the same model.
(B) It uses an EMA of the student weights as the teacher.
(C) It averages the predictions of two temporal teachers.
(D) It uses the EMA of each sample's past predictions.
**Answer**: (A) — "Uses self-consistency between original and augmented inputs instead of an EMA teacher." (B)=MT, (C)=PS-MT, (D)=TE. (p.31)

### Q14 [MCQ]
Which is NOT listed as a property of a good augmentation in UDA?
(A) Label-preserving
(B) Diverse
(C) Adversarial
(D) Task-relevant
**Answer**: (C) — 좋은 aug = label-preserving, diverse, realistic, task-relevant. UDA는 오히려 adversarial perturbation을 strong semantic aug로 바꾼 게 핵심. (p.40, p.41)

### Q15 [MCQ]
What is the key message of UDA?
(A) Perturbing along the most adversarial direction is more effective than random directions.
(B) Mixing inputs should produce mixed predictions.
(C) Replacing adversarial perturbations with strong, label-preserving semantic augmentations (e.g., RandAugment) substantially improves performance.
(D) A teacher should not become identical to the student.
**Answer**: (C) — augmentation quality가 핵심. (A)=VAT vs RPT, (B)=ICT, (D)=Dual Teacher. (p.41, p.42)

### Q16 [MCQ]
Why does PS-MT replace the MSE consistency loss with Conf-CE?
(A) MSE is too sensitive to noisy teacher predictions.
(B) MSE gives small gradients near convergence and relatively weak supervision; more accurate teachers allow stronger cross-entropy supervision.
(C) CE does not require a teacher model.
(D) MSE cannot be computed for pixel-level predictions.
**Answer**: (B) — MSE는 noisy target에 robust하지만 supervision이 약함; teacher가 정확해지면 CE로 더 강하게. (A)는 반대(MSE는 robust). ablation에서도 Conf-CE가 largest single gain. (p.48, p.52)

### Q17 [MCQ]
Which problem does Dual Teacher mainly address?
(A) The teacher target is not reliable enough to support strong perturbations.
(B) The teacher becomes too similar to the student (coupling), providing little new information and inheriting the student's errors.
(C) Image-level perturbations are not enough for a Vision Transformer.
(D) Two forward passes per input are too expensive.
**Answer**: (B) — (A)=PS-MT, (C)=S⁴Former, (D)=Temporal Ensembling의 동기. Dual Teacher는 ensemble 대신 **switching**으로 해결. (p.57, p.63, p.75)

### Q18 [MCQ]
Which correctly matches S⁴Former's regularization components to their levels?
(A) Input – PASA, Feature – NCR, Output – PatchShuffle
(B) Input – PatchShuffle, Feature – PASA, Output – NCR
(C) Input – NCR, Feature – PatchShuffle, Output – PASA
(D) Input – PatchShuffle, Feature – NCR, Output – PASA
**Answer**: (B) — PatchShuffle(블록 셔플 후 복원) = input, PASA(teacher confidence 기반 attention modulation) = feature, NCR(pseudo-label class 뺀 softmax 분포 매칭) = output. (p.69–72)

<!-- REVIEWED | fixes: 0 | reviewer: opus -->
