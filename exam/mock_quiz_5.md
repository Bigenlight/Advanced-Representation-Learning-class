# Mock Quiz 5 — Advanced Representation Learning (Lectures 01–07 + Pseudo code)

> 20 questions · **새 형식: T/F 6 + MCQ 7 + 빈칸(Blank) 7** · pseudo code 문항 4개 (`·PC` 태그). 빈칸은 **영어 단어/식을 직접 써서** 답할 것 (철자까지). 실전처럼 답 먼저 다 적고 맨 아래 정답 확인.
> 출처: `exam/_parts/` pool (C·D 섹션 + `pc_1~4`), mock 1~3과 겹치지 않는 문항만. 모두 Opus 독립 검수 완료 (2026-10-22).

### 1. [T/F] `[L01]`
In the lecture, prediction means inferring unobserved properties of the world, such as the future, the past, or missing data.

### 2. [T/F] `[L02]`
In the Jigsaw puzzle task, the network may solve the puzzle using low-level cues such as edge continuity and color/intensity statistics instead of semantics.

### 3. [Blank] `[L02]`
In Rotation Prediction (RotNet, Gidaris et al.), no labels are needed because the applied rotation becomes the label. To recognize rotation, the network must recognize ______.

### 4. [MCQ] `[L03]`
What is the core idea of Noise Contrastive Estimation (NCE)?
(A) Run logistic regression to tell apart target data from noise samples.
(B) Choose the positive among N candidates with a categorical cross-entropy loss.
(C) Generate positive views with a diffusion model and weight them by quality.
(D) Use all neighbors of the same class as positives with a temperature.

### 5. [Blank] `[L03]`
In the contrastive loss (Chopra et al., 2005), positive pairs are pulled together, while negative pairs are pushed apart beyond a ______ $\varepsilon$, which defines the minimum distance required between different classes.

### 6. [MCQ] `[L04]`
Which of the following is NOT used in BYOL?
(A) A predictor on the online network
(B) A target network updated by an exponential moving average
(C) Negative samples from the mini-batch
(D) A mean squared error loss between normalized outputs

### 7. [T/F] `[L04]`
Unsupervised Data Augmentation (UDA) learns from unlabeled data by maximizing the KL divergence between the predictions on an original image and on its augmented version.

### 8. [MCQ] `[L04]`
Which of the following is NOT a component of "DINOv2 = DINO + ... " as presented in the lecture?
(A) iBOT (masked patch-level objective)
(B) KoLeo regularizer
(C) Curated data (LVD-142M)
(D) Gram anchoring

### 9. [Blank] `[L04·PC]`
In DINO's loop, the center is updated as `C = m*C + (1-m)*cat([t1, t2]).mean(dim=0)`. So the center is an exponential moving average of the batch mean of the ______ network's outputs.

### 10. [Blank] `[L05]`
Mini-batch negatives are limited but (1) ______, whereas memory bank negatives are large in number but slightly (2) ______.

### 11. [MCQ] `[L05]`
Which is one of the key findings about CLIP's training objective?
(A) Caption prediction with a Transformer language model is about 4× more data-efficient than contrastive learning.
(B) Contrastive learning is approximately 4× more data-efficient than caption prediction.
(C) A Transformer language model text encoder outperforms a simple Bag-of-Words encoder.
(D) Complex text modeling is more important than rich language supervision.

### 12. [MCQ] `[L05·PC]`
The following snippet (variable names changed) comes from the lecture's pseudo code. Which method is it from, and what are the positives?
```python
logits = np.dot(A_e, B_e.T) * np.exp(t)      # [n, n]
labels = np.arange(n)
loss = (ce(logits, labels, axis=0) + ce(logits, labels, axis=1)) / 2
```
(A) SimCLR — the positives are two augmented views of the same image.
(B) MoCo — the positive is always placed in column 0 of the logits.
(C) CLIP — the positives are the diagonal entries (matched image–text pairs).
(D) CLIP — the positives are all off-diagonal entries, since the two inputs come from different modalities.

### 13. [MCQ] `[L06]`
Which problem does Dual Teacher mainly address?
(A) The teacher target is not reliable enough to support strong perturbations.
(B) The teacher becomes too similar to the student (coupling), providing little new information and inheriting the student's errors.
(C) Image-level perturbations are not enough for a Vision Transformer.
(D) Two forward passes per input are too expensive.

### 14. [T/F] `[L06]`
In the Π-model, two versions of the same input with different stochastic augmentation and dropout masks are passed through two separately trained networks, and the two outputs are encouraged to be consistent.

### 15. [Blank] `[L06]`
Mean Teacher replaces the EMA of predictions with an EMA of model (1) ______: $\theta' \leftarrow \beta\theta' + (1-\beta)\theta$. The teacher's parameters are updated only by this EMA, with no (2) ______.

### 16. [Blank] `[L06·PC]`
At the end of each epoch, Temporal Ensembling constructs the targets as $\tilde z \leftarrow Z / ($(1) ______$)$. The comment says this step constructs target vectors by (2) ______.

### 17. [T/F] `[L07]`
Zoph et al. found that self-training remains effective even with strong augmentation and abundant labeled data, whereas the benefit of pre-training diminishes as more labeled data become available.

### 18. [MCQ] `[L07]`
How does FreeMatch set its self-adaptive global threshold?
(A) It is fixed and manually selected throughout training, as in FixMatch
(B) It is the EMA of the average confidence on unlabeled data, starting from 1/C and increasing as the model becomes more confident
(C) It is the classifier weight norm of the most confident class
(D) It starts high and decreases over training to admit more samples

### 19. [Blank] `[L07]`
Instead of combining many techniques, FixMatch relies on only three simple ideas: (1) ______ augmentation to generate a reliable pseudo label, a confidence (2) ______ to discard uncertain predictions, and strong augmentation to train the model to predict the same label.

### 20. [T/F] `[L07·PC]`
In line 17 of DivideMix, $p_b$ is computed by averaging the predictions of both networks $\theta^{(1)}$ and $\theta^{(2)}$ over the $M$ augmentations of $x_b$.


---

<details>
<summary><b>정답 & 해설 (클릭)</b></summary>

**1.** `[L01]` True — Future (V-JEPA) / Past (state estimation) / Missing Data (MAE, inpainting) (p.20)  ← pool: `01_Introduction` Q15

**2.** `[L02]` True — shortcut(cheating) 문제 (Low-level cue #1 Edge Continuity, #2 Color/Intensity Statistics). 해결책: multiple puzzle configurations per image, random gaps between patches, random crop, spatial jitter, color jitter, grayscale images (p.31)  ← pool: `02_SSL` Q10

**3.** `[L02]` objects (accept: the object, object) — "To recognize rotation, the network must recognize objects." 0°, 90°, 180°, 270° 중 무엇을 적용했는지 맞히는 4-way classification. Learned concept = Object-Centric Understanding (p.40–41, p.45)  ← pool: `02_SSL` F6

**4.** `[L03]` (A) — "Positive vs Negative → Data vs Noise", logistic regression으로 구분. (B) InfoNCE, (C) GenView, (D) Soft-NN. (p.21)  ← pool: `03_Contrastive_Learning_I` Q7

**5.** `[L03]` margin — negative 항 $\max(0, \varepsilon - \|f_\theta(x_i)-f_\theta(x_j)\|_2)^2$. 한계: pair만 봄, 이미 margin 밖인 negative는 inactive, relative ordering 강제 X (p.11)  ← pool: `03_Contrastive_Learning_I` F2

**6.** `[L04]` (C) — "BYOL does not use negative pairs." predictor·EMA target·L2-normalized MSE는 모두 사용. (p.30, p.32, p.34)  ← pool: `04_Contrastive_Learning_II` Q14

**7.** `[L04]` False — UDA는 KL divergence를 **minimize**해서 원본·augmented 입력의 예측을 **일치**시킴 ("Minimizes KL divergence between predictions"). Key Idea: *Learn from unlabeled data through **consistency regularization**.* (p.18)  ← pool: `04_Contrastive_Learning_II` Q19

**8.** `[L04]` (D) — Gram anchoring은 DINOv3. DINOv2 = DINO + iBOT + KoLeo + Curated Data + Scaling. (p.57, p.68–69)  ← pool: `04_Contrastive_Learning_II` Q17

**9.** `[L04·PC]` teacher (accept: `gt`, $g_{\theta_t}$) — `t1, t2` = teacher output (`# teacher output n-by-K`), 수식도 $g_{\theta_t}$. student 출력 `s1, s2`는 center에 안 들어감. center는 teacher 출력에서 빼는 값이니 teacher 통계로 만드는 게 자연스러움. 함정 두 개: "student"라고 쓰기, 그리고 center를 parameter EMA(`l` 줄)와 헷갈리기 — center는 **output** 평균의 EMA. (L04 p.40)  ← pool: `pc_2_distill` Q11

**10.** `[L05]` (1) fresh (2) stale (accept: outdated) — memory bank = 과거 embedding을 저장해 negative로 재사용 → 계산은 싸고 negative 수는 많지만, 옛 encoder로 계산된 feature라 현재 query와 안 맞음. Key Idea: "Trade stale representations for efficient large-scale contrastive learning." (p.4, p.9)  ← pool: `05_Contrastive_Learning_III` F1

**11.** `[L05]` (B) — BoW 단순 encoder가 Transformer LM을 이기고, rich supervision > complex text modeling (p.36)  ← pool: `05_Contrastive_Learning_III` Q11

**12.** `[L05·PC]` (C) — `np.exp(t)`(learned temperature) + `arange(n)` + axis 0/1 평균(symmetric)은 CLIP의 시그니처. `arange(n)`이라 정답이 **diagonal** = matched (image, caption) pair, off-diagonal $n-1$개는 negative. (A) SimCLR는 수식형 Algorithm이고 $2N\times2N$ 행렬에서 $\mathbb 1_{[k\neq i]}$로 자기 자신을 뺌. (B) MoCo는 `zeros(N)` + $N\times(1+K)$ logits. (D) off-diagonal은 negative (L05 p.33, p.35)  ← pool: `pc_1_infonce` Q12

**13.** `[L06]` (B) — (A)=PS-MT, (C)=S⁴Former, (D)=Temporal Ensembling의 동기. Dual Teacher는 ensemble 대신 **switching**으로 해결. (p.57, p.63, p.75)  ← pool: `06_Semi_Supervised_Learning_I` Q17

**14.** `[L06]` False — 네트워크는 **하나**(같은 network with dropout)임. 같은 입력을 다른 noise(stochastic aug + dropout mask)로 **두 번 forward** 해서 두 출력을 MSE로 맞춤: p.18 pseudocode에서 $z_i \leftarrow f_\theta(g(x_i))$, $\tilde z_i \leftarrow f_\theta(g(x_i))$ ("again, with different dropout and augmentation") — 둘 다 같은 $f_\theta$, unsupervised term $w(t)\frac{1}{C|B|}\sum_i\|z_i-\tilde z_i\|^2$. 이 forward 2회가 단점 → Temporal Ensembling의 동기. (p.17, p.18, p.19)  ← pool: `06_Semi_Supervised_Learning_I` Q19

**15.** `[L06]` (1) weights (accept: parameters) (2) backpropagation (accept: backprop, gradient update, gradient descent) — student는 gradient descent (supervised + consistency loss), teacher는 student weight의 EMA로만 갱신. per-sample prediction memory 불필요, 매 iteration 갱신 → scalable. 실전 팁: dropout은 student만, $\beta$ 0.99 (early) → 0.999 (later), MSE > KL. (p.23, p.24, p.28)  ← pool: `06_Semi_Supervised_Learning_I` F6

**16.** `[L06·PC]` (1) $1-\alpha^t$ (accept: 1 − α^t) (2) bias correction — $Z$가 $\mathbf 0$에서 시작해 초반 EMA가 0쪽으로 편향됨. 예측이 일정한 $z$면 $Z_t=(1-\alpha^t)z$라서 $1-\alpha^t$로 나누면 정확히 $z$ 복원 (Adam bias correction과 같은 원리). $t$ = epoch index. (p.20)  ← pool: `pc_3_consistency` Q5

**17.** `[L07]` True — Key Findings 그대로 (p.58)  ← pool: `07_Semi_Supervised_Learning_II` Q10

**18.** `[L07]` (B) — $\tau_0=1/C$, average confidence의 EMA, confidence 오를수록 threshold 상승. (C)는 AllMatch의 local scaling 소재 (p.69, p.82)  ← pool: `07_Semi_Supervised_Learning_II` Q20

**19.** `[L07]` (1) weak (accept: weakly) (2) threshold (accept: thresholding, $\tau$) — "Predict on an easy example, learn from a difficult one." weak = flip-and-shift, strong = RandAugment/CTAugment 등. sharpening·MixUp·DA 없이 단순. ablation: confidence thresholding > temperature sharpening, pseudo label은 반드시 weak에서 (strong에서 만들면 unreliable). **weak↔strong 뒤집은 함정** 주의. (p.43–45, p.48)  ← pool: `07_Semi_Supervised_Learning_II` F8

**20.** `[L07·PC]` False — line 17 $p_b=\frac1M\sum_m p_{\text{model}}(\hat x_{b,m};\theta^{(k)})$ = **지금 학습 중인 network $\theta^{(k)}$ 하나**의 $M$-aug 평균. 두 net 평균은 **line 20 (co-guessing, $u_b$)** 에서만. co-refinement의 "co-"는 $w_b$가 **상대 net**에서 온다는 뜻 (line 18 주석 *"clean probability produced by the other network"*) (L07 p.54)  ← pool: `pc_4_pseudolabel` Q4

</details>
