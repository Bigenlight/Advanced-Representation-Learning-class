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

## C. Question pool (20 questions)

### Q1 [T/F]
Active learning selects the most valuable unlabeled samples to be collected next, helping us act smartly with a limited budget.
**Answer**: True — 슬라이드 원문 그대로 ("Select most valuable unlabeled samples to be collected next / Helps us act smartly with a limited budget"). "collected" = 골라서 사람(oracle)한테 label 받아 labeled set에 추가한다는 뜻. (p.5)

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
**Answer**: True — AT 식의 GT $q(y|\mathbf x^l)$ 자리에 current model prediction $p_{\hat\theta}(y|\mathbf x)$를 넣음 → "The VAT loss applies to both labeled and unlabeled samples." (슬라이드 화살표 라벨 "If GT is known," ← 원문 유지, 문맥상 "GT를 모를 때(unknown) model prediction 사용"으로 읽어야 자연스러움) (p.34)

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

### Q19 [T/F]
In the Π-model, two versions of the same input with different stochastic augmentation and dropout masks are passed through two separately trained networks, and the two outputs are encouraged to be consistent.
**Answer**: False — 네트워크는 **하나**(같은 network with dropout)임. 같은 입력을 다른 noise(stochastic aug + dropout mask)로 **두 번 forward** 해서 두 출력을 MSE로 맞춤: p.18 pseudocode에서 $z_i \leftarrow f_\theta(g(x_i))$, $\tilde z_i \leftarrow f_\theta(g(x_i))$ ("again, with different dropout and augmentation") — 둘 다 같은 $f_\theta$, unsupervised term $w(t)\frac{1}{C|B|}\sum_i\|z_i-\tilde z_i\|^2$. 이 forward 2회가 단점 → Temporal Ensembling의 동기. (p.17, p.18, p.19)

### Q20 [MCQ]
Which statement best describes the core idea of consistency regularization?
(A) Predictions should remain consistent under small perturbations, which may come from data augmentations or model stochasticity (e.g., Dropout).
(B) High-confidence predictions are converted into hard labels and used as targets with cross-entropy.
(C) Labels are propagated from labeled samples to unlabeled samples over a similarity graph.
(D) Decision boundaries should be placed in high-density regions of the data.
**Answer**: (A) — Core idea 원문 그대로; perturbation 출처 = data augmentations / model stochasticity (Dropout). H1 smoothness를 loss로 만든 것. (B)는 Pseudo Label (다음 강의, $\mathcal L_u$ 설계의 다른 축), (C)는 Label Propagation (07강), (D)는 low-density separation을 뒤집은 오답. (p.9, p.16)

## D. Fill-in-the-blank pool (12 questions)

### F1 [Blank]
The SSL learning objective $\mathcal L = \mathcal L_s + \mu(t)\,\mathcal L_u$ consists of a supervised loss, an unsupervised loss, and a (1) ______ term $\mu(t)$, which is (2) ______ in time.
**Answer**: (1) weighting (accept: weight) (2) increasing (accept: increased, growing) — 초반 모델 예측은 부정확해서 예측 기반 target인 $\mathcal L_u$를 처음부터 세게 걸면 noise를 학습함 → $\mu(t)$를 작게 시작해 점점 키움 (ramp-up). "decreasing"으로 바꾼 T/F 함정 주의. (p.8, p.10)

### F2 [Blank]
The key question of semi-supervised learning is how to design the (1) ______ loss. The two main answers are consistency regularization (this lecture) and (2) ______ (next lecture).
**Answer**: (1) unsupervised (accept: $\mathcal L_u$) (2) pseudo label (accept: pseudo labeling, pseudo-labeling) — supervised loss는 그냥 CE라 설계할 게 없고, label 없는 데이터에서 loss를 어떻게 만드냐가 SSL의 본질. consistency regularization = perturbed 예측끼리 맞춤 / pseudo label = confident 예측을 hard label로 바꿔 CE. (p.9)

### F3 [Blank]
H1, the ______ assumption: nearby samples should have similar predictions, and small perturbations should not change the label.
**Answer**: smoothness (accept: smooth) — consistency regularization의 직접적 근거 ("perturbation 줘도 예측은 같아야"). H2 cluster (같은 cluster = 같은 label, boundary가 cluster를 쪼개면 안 됨), H3 low-density separation (boundary는 low-density region에), H4 manifold (고차원 데이터는 low-dimensional manifold 위)와 구분. (p.11–14)

### F4 [Blank]
In the Π-model, two versions of the same input with different stochastic augmentation and (1) ______ masks pass through the network, and the outputs are expected to be consistent. Its main drawback is that it requires (2) ______ forward passes for every input.
**Answer**: (1) dropout (2) two (accept: 2) — 같은 network에 다른 noise로 두 번 forward → 두 출력 MSE. forward 2회 = high computational cost → Temporal Ensembling이 이걸 해결하려고 나옴. (p.17, p.19)

### F5 [Blank]
Temporal Ensembling uses, as the consistency target, an EMA of the model (1) ______ in time, maintained per training sample. Its limitation is that these targets are updated only once per (2) ______.
**Answer**: (1) prediction (accept: predictions, outputs) (2) epoch — forward 1회라 싸고 Π-model보다 stable한 target. 대신 sample마다 EMA 예측을 저장해야 해서 memory intensive + epoch당 한 번만 갱신 → large dataset에 scale 어려움 → Mean Teacher로. **prediction EMA(TE) vs weight EMA(MT)** 헷갈리지 말 것. (p.19, p.22)

### F6 [Blank]
Mean Teacher replaces the EMA of predictions with an EMA of model (1) ______: $\theta' \leftarrow \beta\theta' + (1-\beta)\theta$. The teacher's parameters are updated only by this EMA, with no (2) ______.
**Answer**: (1) weights (accept: parameters) (2) backpropagation (accept: backprop, gradient update, gradient descent) — student는 gradient descent (supervised + consistency loss), teacher는 student weight의 EMA로만 갱신. per-sample prediction memory 불필요, 매 iteration 갱신 → scalable. 실전 팁: dropout은 student만, $\beta$ 0.99 (early) → 0.999 (later), MSE > KL. (p.23, p.24, p.28)

### F7 [Blank]
VAT replaces the ground-truth label used in adversarial training with the model's own current (1) ______, so the loss can also be applied to unlabeled samples. Unlike RPT, which smooths predictions in random directions, VAT smooths only along the most (2) ______ direction.
**Answer**: (1) prediction (accept: model prediction, output) (2) adversarial — "virtual" = 진짜 GT 대신 모델 자기 예측을 label처럼 씀. AT = GT labels / supervised / robustness vs VAT = model predictions / semi-supervised / local smoothness. 가장 민감한(worst-case) 방향만 골라 smoothing하니 random 방향(RPT)보다 훨씬 효과적. (p.34, p.35)

### F8 [Blank]
ICT follows the idea of (1) ______: two unlabeled samples are linearly interpolated, and the consistency target (from a Mean Teacher) is the same interpolation of their predictions — "Mixing inputs should produce (2) ______ predictions."
**Answer**: (1) MixUp (accept: mixup) (2) mixed (accept: interpolated) — $p(y|\mathrm{mixup}_\lambda(\mathbf x_i,\mathbf x_j)) \approx \lambda p(y|\mathbf x_i) + (1-\lambda)p(y|\mathbf x_j)$, target은 Mean Teacher가 만듦. 섞인 sample은 decision boundary 근처에 놓이기 쉬움 → 거기서 consistency 걸면 boundary가 smooth해짐 → low-density separation 가정과 잘 맞음. (p.36, p.37)

### F9 [Blank]
UDA applies strong, label-preserving semantic augmentations (e.g., RandAugment) to unlabeled samples and emphasizes the importance of augmentation ______ rather than adversarial perturbations.
**Answer**: quality — "How we perturb the input matters more than simply adding perturbations." adversarial perturbation을 strong semantic aug로 바꾸기만 해도 VAT·ICT를 이김. 좋은 aug = label-preserving, diverse, realistic, task-relevant ("adversarial"은 목록에 없음). (p.39–42)

### F10 [Blank]
In PS-MT, the student matches the (1) ______ of two mean teachers (only one teacher is updated in each epoch), and the MSE consistency loss is replaced by a stricter cross-entropy-based loss called (2) ______.
**Answer**: (1) ensemble (accept: teacher ensemble, average) (2) Conf-CE (accept: confidence-weighted CE, confidence-weighted cross-entropy) — 출발 질문 "Can a more accurate teacher support stronger perturbations?": teacher가 틀리면 strong perturbation 하에서 student가 그 오류를 재생산 → 먼저 target을 reliable하게 (2-teacher ensemble). MSE는 noisy target에 robust하지만 convergence 근처 gradient가 작아 supervision이 약함 → teacher가 정확해졌으니 CE로 더 strict하게. (p.43, p.45–48)

### F11 [Blank]
Dual Teacher targets teacher–student (1) ______: with EMA, the teacher gradually follows the student and becomes too similar, providing little new information. Its solution is teacher (2) ______ — two temporary EMA teachers take turns, and only the active teacher guides the student in each epoch.
**Answer**: (1) coupling (2) switching (accept: switch, alternation) — PS-MT식 ensemble은 두 teacher가 동시에 guide해서 점점 distinct characteristics를 잃음 → switching이면 각 teacher가 다른 student 상태를 관찰해 다른 perspective 유지. EMA 규칙은 그대로, teacher schedule만 바뀜. student를 하나 더 학습하는 Dual Student보다 저렴. "A teacher should be stable, but it should not become identical to the student." (p.57, p.58, p.63, p.65)

### F12 [Blank]
S⁴Former's input-level regularization, ______, groups neighboring patches into $M\times M$ blocks, randomly permutes the blocks before Transformer encoding, and restores the feature order before decoding.
**Answer**: PatchShuffle (accept: patch shuffle) — ViT는 이미지를 patch token + self-attention으로 처리하니 image-level perturbation만으론 부족 → Input: PatchShuffle / Feature: PASA (teacher가 uncertain한 patch의 attention 교란) / Output: NCR (pseudo-label class 뺀 나머지 분포 매칭). PatchShuffle이 largest individual improvement. "Effective consistency regularization should reflect the structure of the model." (p.68, p.70, p.73, p.74)

<!-- REVIEWED | fixes: 0 | reviewer: opus -->

<!-- AUGMENTED 2026-10-22 | C fixes: 2 | C added: 2 | D blanks: 12 | author: opus
Q1: stem "to be labeled next" -> "to be collected next"; explanation claimed verbatim slide wording but stem was paraphrased (p.5 says "collected"). Answer unchanged (True).
Q7: explanation quoted p.34 arrow label "If GT is known -> use current model prediction" as if correct; rewrote to explain GT -> model-prediction substitution and flagged the label as a likely slide slip ("unknown"). Answer unchanged (True).
Q19 added: Pi-model had no direct question (T/F, same network vs two networks).
Q20 added: Consistency Regularization core idea (p.16) had no question (MCQ vs pseudo label / LP / flipped LDS).
-->

<!-- VERIFIED 2026-10-22 | checked: 32 | fixed: 7 | deleted: 0 | reviewer: opus
Q19: explanation formula was not from the slides (Lil'Log-style MSE(f, f')); replaced with p.18 pseudocode (same f_theta evaluated twice) and added p.18 to citation. Answer unchanged (False).
F1: stem gave away blank (1) ("Here mu(t) is the unsupervised loss weight" right after "a ______ term"); merged into "a (1) ______ term mu(t), which is (2) ______ in time".
F5: stem gave away blank (1) ("uses the historical predictions ... : an EMA of the model ______"); removed the phrase so the blank actually tests prediction-EMA (TE) vs weight-EMA (MT).
F6: added "gradient descent" to accept for (2) (p.24 contrasts student "updated by gradient descent" vs teacher "no backpropagation").
F8: added "interpolated" to accept for (2) (stem itself describes the target as an interpolation of predictions).
F10: stem gave away (2) ("confidence-weighted cross-entropy loss called ______" ~ Conf-CE); reworded to "a stricter cross-entropy-based loss called ______", added "confidence-weighted cross-entropy" to accept.
F12: "shuffles the blocks" handed over the name PatchShuffle; changed to "randomly permutes the blocks".
All other C/D items: key, p.N, and distractors confirmed against slides (image pages p.8, p.10, p.18, p.23, p.33, p.34, p.36, p.47, p.48 checked). Sections A/B: no slide contradictions found.
-->
