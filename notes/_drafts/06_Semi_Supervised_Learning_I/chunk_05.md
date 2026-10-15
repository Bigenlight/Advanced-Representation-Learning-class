<!-- DRAFT | lecture: 06_Semi_Supervised_Learning_I | chunk: 05 | pages 30–35 -->
<!-- VERIFIED | fixes: 2 | verifier: sonnet -->

## Noisy Samples as Learning Targets: Core Idea (p.30)

슬라이드 텍스트:
- **Core Idea**
  - Apply strong augmentations to an unlabeled sample.
  - The prediction should remain consistent before and after augmentation.
- **Key Insight**: The model should make the same prediction for an unlabeled sample and its augmented version.
- **Noisy consistency training directly enforces prediction invariance between an input and its augmented version using the same model.**

그림: 왼쪽에서 오른쪽으로 흐르는 블록 다이어그램.
- 첫째 줄: $(\mathbf{x}, y) \to f_\theta \to$ **cross entropy** 박스.
- 둘째 줄: $\mathbf{u} \to f_\theta \to$ **MSE** 박스 (원본 unlabeled 입력의 예측).
- 셋째 줄: $\mathbf{u} \to$ **Augmentation** (초록 박스; e.g. Image: mixup, RandAugment / Language: back-translation, TF-IDF repl) $\to \bar{\mathbf{u}} \to f_\theta \to$ MSE 박스 (증강 입력의 예측).
- cross entropy 출력과 MSE 출력이 **weighted sum**으로 합쳐지고, 가중치 $\mu(t)$가 위에서 들어감. 최종 출력: $\mathcal{L}_s + \mu(t)\mathcal{L}_u$.

**설명**
- 구조는 **supervised loss + unsupervised consistency loss**의 합이야: $\mathcal{L} = \mathcal{L}_s + \mu(t)\,\mathcal{L}_u$.
  - $\mathcal{L}_s$: labeled $(\mathbf{x}, y)$에 대한 **cross entropy**.
  - $\mathcal{L}_u$: unlabeled $\mathbf{u}$와 증강본 $\bar{\mathbf{u}}$의 예측 차이 (슬라이드에선 **MSE**).
  - $\mu(t)$: 시간 $t$에 따라 변하는 **weight ramp-up** 함수 ← 슬라이드엔 정의 없음, 보통 초반 0에서 시작해 점점 키움 (아래 보충).
- **핵심 아이디어**: label이 없어도 "**증강해도 같은 class여야 한다**"는 **invariance prior**를 loss로 직접 강제. 같은 모델 $f_\theta$를 두 번 통과시켜(원본 / 증강본) 예측을 맞추는 **self-consistency**임.
- **"Noisy samples as learning targets"**라는 제목의 의미: 증강(noise)을 가한 샘플의 예측이 **학습 target 역할**을 함 (또는 반대로 원본 예측이 target). 별도 teacher 없이 입력 쪽 noise만으로 target이 생성됨.
- **strong augmentation**을 쓰는 이유: 약한 noise는 모델이 쉽게 invariant해서 loss가 거의 0 → 학습 신호가 없음. mixup/RandAugment 같은 강한 변환을 써야 **decision boundary를 low-density region으로 밀어내는** regularization 효과가 큼.
> 보충: $\mu(t)$는 Π-model/Mean Teacher에서 쓰는 **ramp-up** (예: $\mu(t)=w_{\max}\exp(-5(1-t/T)^2)$, Gaussian ramp-up). 학습 초반엔 모델 예측이 부정확해서 consistency target 자체가 쓰레기이므로 $\mathcal{L}_u$ 비중을 작게 시작함. 슬라이드엔 수식 없음.
> 보충: 이 계열의 대표 방법이 **UDA (Unsupervised Data Augmentation)**. 원본 예측(gradient 차단, sharpening 가능)을 target으로, 증강본 예측과의 KL/MSE를 최소화. 슬라이드의 그림과 같은 구조. 슬라이드 자체엔 방법명이 안 나옴.
> 보충: 그림에서 $\mathbf{u}$ 쪽 $f_\theta$ 출력에 stop-gradient를 걸지 여부는 슬라이드에 명시되지 않음.

---

## Noisy Samples as Learning Targets: Π-model / Mean Teacher와의 비교 (p.31)

슬라이드 텍스트:
- **Compared with Π-model**
  - Uses input augmentations on unlabeled data instead of stochastic perturbations inside the network.
- **Compared with Mean Teacher**
  - Uses self-consistency between original and augmented inputs instead of an EMA teacher as the learning target.

**설명**
- 세 방법 모두 "**같은 입력에 대한 예측은 일관되어야 한다**"는 consistency regularization 계열인데, **noise를 어디서 넣느냐 / target을 어디서 얻느냐**가 다름.

| | noise 위치 | consistency target |
|---|---|---|
| **Π-model** | **network 내부** stochastic perturbation (dropout, 약한 입력 noise 등) | 같은 모델의 다른 stochastic pass |
| **Mean Teacher** | 양쪽 pass에 stochastic noise | **EMA teacher** $\theta' \leftarrow \alpha\theta' + (1-\alpha)\theta$의 예측 |
| **Noisy consistency (이 슬라이드)** | **입력에 strong augmentation** (mixup, RandAugment 등) | 원본 입력 자신의 예측 (EMA teacher 없음) |

- 정리: Π-model과 비교하면 **noise의 종류**(network 내부 → input augmentation)가 바뀐 것, Mean Teacher와 비교하면 **target 생성 방식**(EMA teacher → 같은 모델 self-consistency)이 바뀐 것.
- 장점: 입력 공간의 **도메인 지식 기반 augmentation**(image/NLP별)을 활용할 수 있고, EMA 모델을 따로 유지할 필요가 없음.
> 보충: Π-model은 Laine & Aila, Mean Teacher는 Tarvainen & Valpola. 둘 다 앞 슬라이드(이 chunk 밖)에서 다룬 방법이라 여기선 비교 표만 정리함.

---

## Adversarial Training (Goodfellow et al., 2014) (p.32)

슬라이드 텍스트:
- **Adversarial Training (Goodfellow et al., 2014)**
  - Add adversarial **perturbations** to the input during training.
  - Train the model to **produce the correct prediction** even under worst-case perturbations.
  - Improves robustness against **adversarial attacks**.
- 출처: Goodfellow et al., ICLR 2015, *Explaining and Harnessing Adversarial Examples*

그림: 3개 패널이 "+ .007 × ... =" 로 연결.
- 왼쪽: 판다 이미지 $\boldsymbol{x}$, **"panda" 57.7% confidence**.
- 가운데: 노이즈 이미지 $\text{sign}(\nabla_{\boldsymbol{x}} J(\boldsymbol{\theta}, \boldsymbol{x}, y))$, **"nematode" 8.2% confidence**.
- 오른쪽: 판다처럼 보이는 이미지 $\boldsymbol{x} + \epsilon\,\text{sign}(\nabla_{\boldsymbol{x}} J(\boldsymbol{\theta}, \boldsymbol{x}, y))$, **"gibbon" 99.3% confidence**.

**설명**
- **Adversarial example**: 사람 눈엔 거의 같은 이미지인데, **입력에 아주 작은 perturbation**을 더해 모델이 **자신 있게 틀린 예측**을 내게 만든 샘플. 그림이 대표 예시: $\epsilon = 0.007$ 만큼의 노이즈로 panda(57.7%) → gibbon(99.3%).
- **Adversarial Training**은 학습 중에 이런 adversarial example을 만들어 넣고, **그래도 정답을 맞히도록** 학습함 → **worst-case perturbation에 robust**해짐. 즉 일반 augmentation이 "랜덤 noise"라면, 이건 "**모델이 가장 약한 방향**의 noise"임.
- 그림 가운데 노이즈의 "nematode 8.2%"는 노이즈 자체를 모델에 넣었을 때의 예측 (의미 없는 noise라 확신 낮음). 핵심은 **사람 눈엔 안 보이는 작은 노이즈가 모델에겐 큰 영향**이라는 점.
- 왜 neural net이 이렇게 취약한가: Goodfellow의 설명은 **linearity hypothesis** ← 고차원에서 입력 각 원소의 작은 변화 $\epsilon$이 선형 레이어에서 합쳐지면 $\epsilon\|w\|_1$ 크기의 큰 출력 변화가 됨.
> 보충: 슬라이드 본문은 "(Goodfellow et al., 2014)"이고 하단 출처는 "ICLR 2015" ← arXiv 2014 / ICLR 2015 게재 연도 차이이며 오타 아님.
> 보충: $J(\boldsymbol{\theta}, \boldsymbol{x}, y)$는 모델 파라미터 $\boldsymbol{\theta}$, 입력 $\boldsymbol{x}$, 정답 $y$에 대한 **학습 loss(cost function)**. FGSM은 이 loss를 **입력 $\boldsymbol{x}$에 대해** 미분함 (파라미터 미분이 아님).

---

## Adversarial Training: How it works / Adversarial perturbation (FGSM) (p.33)

슬라이드 텍스트:
- **How it works**
  1. Compute the adversarial perturbation that maximally changes the prediction.
  2. Add the perturbation to the input.
  3. Minimize the prediction difference between clean and perturbed inputs.
- **Adversarial perturbation**
  - Find the worst-case perturbation within a small norm constraint.
  - In practice, approximate it using the gradient direction (FGSM):

$$\mathcal{L}_{\text{adv}}(\mathbf{x}^l, \theta) = D\big[q(y \mid \mathbf{x}^l),\ p_\theta(y \mid \mathbf{x}^l + r_{\text{adv}})\big]$$

$$r_{\text{adv}} = \arg\max_{r;\ \|r\|\le\epsilon} D\big[q(y \mid \mathbf{x}^l),\ p_\theta(y \mid \mathbf{x}^l + r_{\text{adv}})\big]$$

$$r_{\text{adv}} \approx \epsilon \frac{g}{\|g\|_2} \approx \epsilon\,\text{sign}(g) \quad \text{where } g = \nabla_r D\big[y,\ p_\theta(y \mid \mathbf{x}^l + r)\big]$$

- 슬라이드 주석(빨간 박스): $q(y\mid\mathbf{x}^l)$ = **GT (label, one-hot)**, $p_\theta(y\mid\mathbf{x}^l + r_{\text{adv}})$ = **Model prediction for the attacked image**.
- 슬라이드 주석(파란 박스, $r_{\text{adv}}$ argmax 줄): **"Find the noise in the direction where the model is most confused."**
- ← 의심: 둘째 줄 argmax 안의 $p_\theta(y\mid\mathbf{x}^l + r_{\text{adv}})$는 정의상 $r_{\text{adv}}$가 자기 자신을 참조하는 순환 표기. 정확히는 argmax 변수 $r$이어야 함 (원 논문 Miyato 표기도 $\mathbf{x}+r$). 또 첫 줄은 $q(y\mid\mathbf{x}^l)$, 셋째 줄 $g$는 $y$로 표기가 섞여 있음 (둘 다 one-hot label).

**설명**
- **3단계 절차**: (1) 예측을 **가장 크게 바꾸는 perturbation** $r_{\text{adv}}$ 계산 → (2) 입력에 더함 → (3) clean/perturbed 입력에 대한 **예측 차이(= 정답과 attacked 예측의 divergence $D$)** 최소화.
- **notation**:
  - $q(y\mid\mathbf{x}^l)$: labeled sample $\mathbf{x}^l$의 **true label 분포** (one-hot).
  - $p_\theta(y\mid\cdot)$: 모델의 예측 분포.
  - $D[\cdot,\cdot]$: 두 분포 간 divergence (cross entropy나 KL). one-hot $q$일 때 $D$ = cross entropy와 동치 (아래 보충).
  - $\epsilon$: perturbation의 **norm 상한** (perturbation budget). $\|r\|\le\epsilon$ → "small norm constraint".
- **min-max 구조**: 안쪽 $\max_r$ (가장 나쁜 노이즈 찾기, $r_{\text{adv}}$), 바깥쪽 $\min_\theta$ (그 노이즈에도 맞히도록 $\mathcal{L}_{\text{adv}}$ 최소화). 
- **FGSM 근사 유도 (Fast Gradient Sign Method)**: 정확한 $\arg\max_{\|r\|\le\epsilon}$는 어려운 최적화 → $D$를 $r=0$ 근처에서 **1차 Taylor 전개**:
  $$D(\mathbf{x}+r) \approx D(\mathbf{x}) + g^\top r,\quad g = \nabla_r D\big|_{r=0}$$
  - 이 선형 근사를 $\|r\|_2\le\epsilon$ 아래서 최대화 → Cauchy–Schwarz로 $r = \epsilon\, g/\|g\|_2$ (**gradient 방향으로 $\epsilon$만큼**). ← 슬라이드 $\epsilon\frac{g}{\|g\|_2}$.
  - $L_\infty$ ball($\|r\|_\infty\le\epsilon$)을 쓰면 선형 목적 $g^\top r$의 최대는 $r=\epsilon\,\text{sign}(g)$ ← 슬라이드 $\epsilon\,\text{sign}(g)$. 즉 **norm 선택(L2 vs L∞)에 따라 두 식이 각각 정확한 해**. 슬라이드의 "$\approx$"는 두 norm을 같은 것처럼 쓴 느슨한 표기 ← 의심: 두 식은 서로 다른 norm에서의 해라 엄밀한 등호/근사가 아님.
- **왜 gradient 방향이 "모델이 가장 혼란스러워하는 방향"인가**: $g$는 loss를 **가장 가파르게 증가**시키는 입력 방향 (steepest ascent). 그 방향으로 가면 정답 확률이 가장 빨리 떨어짐.
- **$g$의 계산은 $r=0$에서** 평가하는 게 FGSM의 핵심 (one-step): clean 입력에서 backprop 한 번으로 얻음. 그래서 "Fast".
> 보충: one-hot $q$일 때 $D[q, p] = \text{KL}(q\|p) = -\log p_\theta(y\mid\mathbf{x}^l + r)$ = **cross entropy**. 그래서 FGSM은 보통 $\nabla_{\mathbf{x}} J(\theta, \mathbf{x}, y)$ = CE loss의 입력 gradient로 쓰임 (p.32의 식과 동일).
> 보충: 중요한 구분 — **정답 label이 있으면 $g$가 $r=0$에서도 0이 아님** (예측 $p\neq$ one-hot이므로 gradient 존재). 반면 VAT(다음 슬라이드)는 target이 모델 자신의 예측이라 $r=0$에서 $D=0$이 **최소점** → gradient 0 → FGSM을 그대로 못 씀 → **power iteration**이 필요해짐 (p.34 보충 참고).
> 보충: 학습 시 전체 loss는 보통 $\mathcal{L}_{\text{CE}} + \alpha\,\mathcal{L}_{\text{adv}}$ 형태(Goodfellow: $\alpha=0.5$). 슬라이드엔 없음.

---

## Virtual Adversarial Training: Adversarial Training에서 VAT로 (p.34)

슬라이드 텍스트:
- **VAT extends the idea to work in semi-supervised learning.**
  - The VAT loss applies to both labeled and unlabeled samples.
- 위쪽 식 (Adversarial Training):

$$\mathcal{L}_{\text{adv}}(\mathbf{x}^l, \theta) = D\big[q(y\mid\mathbf{x}^l),\ p_\theta(y\mid\mathbf{x}^l + r_{\text{adv}})\big]$$

  - 주석: $q(y\mid\mathbf{x}^l)$ = **GT (label, one-hot)**.
- 화살표: **If GT is known,** → **use current model prediction** (파란 박스가 $p_{\hat\theta}(y\mid\mathbf{x})$를 가리킴)
- 아래쪽 식 (VAT):

$$\mathcal{L}_u^{\text{VAT}}(\mathbf{x}, \theta) = D\big[p_{\hat\theta}(y\mid\mathbf{x}),\ p_\theta(y\mid\mathbf{x} + r_{\text{vadv}})\big]$$

$$r_{\text{vadv}} = \arg\max_{r;\ \|r\|\le\epsilon} D\big[p_{\hat\theta}(y\mid\mathbf{x}),\ p_\theta(y\mid\mathbf{x} + r)\big]$$

- 비교 표:

| Adversarial Training | VAT |
|---|---|
| Uses ground-truth labels | Uses model predictions |
| Supervised learning | Semi-supervised learning |
| Improves robustness | Enforces local smoothness of predictions |

- 결론 박스: **VAT replaces the ground-truth label with the model's own prediction, enabling adversarial consistency training on unlabeled data.**
- 출처: Miyato et al., TPAMI 2018, *Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning*

**설명**
- **한 줄 요약**: Adversarial Training의 target $q(y\mid\mathbf{x}^l)$ (정답 label)을 **현재 모델의 예측 $p_{\hat\theta}(y\mid\mathbf{x})$**로 바꾼 것이 VAT. label이 필요 없으니 **unlabeled data에도 적용** 가능 → semi-supervised.
- **"Virtual"인 이유**: 진짜 label(real label) 대신 **가상의(virtual) label = 모델 자신의 예측**을 쓰기 때문. adversarial direction도 "virtual label에 대한 adversarial".
- **$\hat\theta$의 의미**: **현재 파라미터를 상수로 고정(stop-gradient)**한 것. $p_{\hat\theta}(y\mid\mathbf{x})$ 쪽으로는 gradient를 흘리지 않고, $p_\theta(y\mid\mathbf{x}+r_{\text{vadv}})$ 쪽만 $\theta$로 미분함 → **perturbed 예측을 clean 예측에 맞추는** 방향. ← 슬라이드엔 "current model prediction"이라고만 표기, stop-grad 명시는 없음 (보충).
- **VAT의 의미 (local smoothness)**: 각 입력 $\mathbf{x}$ 주변 $\epsilon$-ball 안에서 **출력 분포가 가장 크게 변하는 방향**(adversarial direction)에서도 예측이 안 변하도록 강제 → 입력에 대한 **출력 분포의 local smoothness**. cluster assumption/smoothness assumption과 일치 (decision boundary가 data 밀집 영역을 가르지 않게 함).
- **supervised 대비 목적 차이**: AT는 "label을 지키며 attack을 견뎌라(**robustness**)", VAT는 "예측이 주변에서 **매끄러워라(smoothness regularization)**". AT는 정답이 있어야만 하고, VAT는 정답이 불필요.
- 전체 목적: 
$$\min_\theta\ \mathcal{L}_s(\mathcal{D}_l,\theta) + \alpha\,\mathbb{E}_{\mathbf{x}\in\mathcal{D}_l\cup\mathcal{D}_u}\mathcal{L}_u^{\text{VAT}}(\mathbf{x},\theta)$$
  슬라이드 문구 "applies to both labeled and unlabeled samples"가 이를 의미 (label 없이도 계산되므로 labeled 샘플에도 같은 식 적용 가능). ← 위 전체 목적식 자체는 슬라이드에 없음.
- $D$는 VAT 논문에서 **KL divergence**: $D[p,p'] = \sum_y p(y)\log\frac{p(y)}{p'(y)}$. 슬라이드는 $D$로만 표기 (아래 보충).

> 보충 (**KL 형태로 쓴 VAT loss**): 
> $$\mathcal{L}_u^{\text{VAT}}(\mathbf{x},\theta) = \text{KL}\big[p_{\hat\theta}(y\mid\mathbf{x})\ \|\ p_\theta(y\mid\mathbf{x}+r_{\text{vadv}})\big],\quad r_{\text{vadv}} = \arg\max_{\|r\|_2\le\epsilon}\text{KL}\big[p_{\hat\theta}(y\mid\mathbf{x})\ \|\ p_\theta(y\mid\mathbf{x}+r)\big]$$
>
> 보충 (**왜 FGSM을 못 쓰나 → power iteration 유도**): 
> 1. $r=0$에서 $D(r) = \text{KL}[p(y|\mathbf{x})\|p(y|\mathbf{x}+r)]$는 **최소값 0**이고 KL은 $\geq 0$이므로 $\nabla_r D|_{r=0}=0$. 1차 항이 사라져 FGSM의 선형 근사가 **무의미**.
> 2. 대신 **2차 Taylor 전개**: $D(r) \approx \tfrac12\, r^\top H\, r$, 여기서 $H = \nabla_r\nabla_r D|_{r=0}$ (Hessian; KL의 경우 **Fisher information matrix**와 같음).
> 3. $\|r\|_2\le\epsilon$ 아래서 $r^\top H r$을 최대화하는 $r$은 **$H$의 dominant eigenvector** $u_1$ 방향: $r_{\text{vadv}} \approx \epsilon\,\overline{u_1}$ ($\overline{u}$ = 단위 벡터 정규화).
> 4. $H$를 명시적으로 구하면 $O(d^2)$ 메모리라 불가능 → **power iteration**: 무작위 단위 벡터 $d$에서 시작해 $d \leftarrow \overline{Hd}$를 반복하면 $u_1$으로 수렴.
> 5. $Hd$를 **finite difference**로 근사: $Hd \approx \dfrac{\nabla_r D(r)\big|_{r=\xi d} - \nabla_r D(r)\big|_{r=0}}{\xi} = \dfrac{\nabla_r D(r)\big|_{r=\xi d}}{\xi}$ (두 번째 항은 0). $\xi$는 작은 상수(예: $10^{-6}$).
> 6. 따라서 갱신식 $d \leftarrow \overline{\nabla_r D\big(p(y|\mathbf{x}),\,p(y|\mathbf{x}+r)\big)\big|_{r=\xi d}}$, 즉 **backprop 한 번으로 한 iteration**. 논문은 **$K=1$ iteration**으로 충분하다고 보고.
> 7. 최종: $r_{\text{vadv}} \approx \epsilon\,\dfrac{g}{\|g\|_2},\ \ g=\nabla_r D\big|_{r=\xi d}$. (p.33의 FGSM 식과 형태가 같되 **gradient를 $r=0$이 아니라 $r=\xi d$에서 평가**한다는 점이 차이.) 
> 8. 계산 비용: **forward 2회 + backward 2회** 정도 (adversarial direction 계산용 1회 + loss의 일반 pass). 슬라이드에는 없음.
>
> 보충: 초기 방향 $d$는 **random Gaussian**에서 뽑으므로 VAT에는 확률성이 남음. 또 $d$가 우연히 $u_1$과 거의 직교하면 수렴이 느리지만, 고차원에서 그럴 확률은 낮음.

---

## Virtual Adversarial Training: 실험 결과 (RPT vs VAT) (p.35)

슬라이드 텍스트:
- 하단 결론: **RPT smooths predictions in random directions, whereas VAT smooths only along the most adversarial direction, making the regularization significantly more effective.**
- 출처: Miyato et al., TPAMI 2018

**TABLE 3** (MNIST, permutation invariant setting). Test error rate (%), $N_l$ = number of labeled examples. Top part: 원 논문 결과, bottom part: Miyato 구현. (PEA = Pseudo Ensembles Agreement, DGM = Deep Generative Models, FM = feature matching)

| Method | $N_l=100$ | $N_l=1000$ |
|---|---|---|
| TSVM [8] | 16.81 | 5.38 |
| PEA [5] | 5.21 | 2.87 |
| DGM (M1+M2) [22] | 3.33 (±0.14) | 2.40 (±0.02) |
| CatGAN [37] | 1.91 (±0.1) | 1.73 (±0.18) |
| Skip DGM [27] | 1.32 (±0.07) | |
| Ladder networks [33] | 1.06 (±0.37) | **0.84** (±0.08) |
| Auxiliary DGM [27] | 0.96 (±0.02) | |
| GAN with FM [36] | **0.93** (±0.07) | |
| **RPT** | 6.81 (±1.30) | 1.58 (±0.54) |
| **VAT** | 1.36 (±0.03) | 1.27 (±0.11) |

**TABLE 4** (SVHN, CIFAR-10, **without** image data augmentation). 별표(*)는 permutation invariant setting 결과. 위쪽은 원 논문 결과, 중간·아래는 Miyato 구현.

| Method | SVHN $N_l=1000$ | CIFAR-10 $N_l=4000$ |
|---|---|---|
| SWWAE [48] | 23.56 | |
| *Skip DGM [27] | 16.61 (±0.24) | |
| *Auxiliary DGM [27] | 22.86 | |
| Ladder networks, Γ model [33] | | 20.40 (±0.47) |
| CatGAN [37] | | 19.58 (±0.58) |
| GAN with FM [36] | 8.11 (±1.3) | 18.63 (±2.32) |
| Π model [24] | 5.43 (±0.25) | 16.55 (±0.29) |
| *(on Conv-Small used in [36])* RPT | 8.41 (±0.24) | 18.56 (±0.29) |
| VAT | **6.83** (±0.24) | **14.87** (±0.13) |
| *(on Conv-Large used in [24])* VAT | 5.77 (±0.32) | 14.18 (±0.38) |
| VAT+EntMin | **4.28** (±0.10) | **13.15** (±0.21) |

**설명**
- **RPT (Random Perturbation Training)**: VAT와 같은 smoothness regularizer인데 perturbation 방향 $r$을 **랜덤**으로 뽑음. VAT는 **가장 adversarial한 방향 하나**만 smoothing. 슬라이드 결론: "**VAT smooths only along the most adversarial direction → regularization이 훨씬 효과적**".
- **왜 adversarial 방향이 더 효과적인가**: 입력 차원이 매우 큰데 랜덤 방향은 대부분 모델 출력이 **이미 둔감한 방향**이라 loss가 작음 → 정규화 효과 미미. 반면 $\arg\max$ 방향은 출력이 **가장 민감한 방향**이라 거기만 눌러줘도 local Lipschitz가 크게 줄어듦 (worst-case 방어).
- **수치로 보는 RPT vs VAT** (같은 Conv-Small, 구현 비교):
  - MNIST $N_l=100$: RPT 6.81 vs **VAT 1.36**, $N_l=1000$: 1.58 vs **1.27**.
  - SVHN ($N_l=1000$): 8.41 vs **6.83**, CIFAR-10 ($N_l=4000$): 18.56 vs **14.87**.
  - 모든 설정에서 VAT가 RPT보다 낮은 error (그리고 **분산도 작음**: MNIST-100에서 ±1.30 → ±0.03).
- **다른 방법과의 비교**: SVHN/CIFAR-10 (augmentation 없음)에서 **VAT (Conv-Small)**가 GAN with FM, CatGAN, Ladder보다 우수. **Conv-Large**로 키우고 **VAT+EntMin** (entropy minimization 추가)하면 SVHN 4.28 / CIFAR-10 13.15로 최고 (Π model 5.43 / 16.55 대비 개선). MNIST에서는 VAT 1.36(100)이 GAN with FM 0.93, Ladder 1.06보다는 못하지만 비교적 상위권. 
- **EntMin**: 예측의 entropy를 최소화해 **더 확신 있는(low-entropy) 예측**을 유도하는 항 (Grandvalet & Bengio, 2005). VAT와 **보완적**이라 합치면 추가 이득.
> 보충: 표의 "permutation invariant setting"은 이미지의 공간 구조(conv)를 쓰지 않고 **픽셀을 flatten한 fully-connected 모델**로 푸는 MNIST 설정. Table 3의 RPT/VAT도 이 설정.
> 보충: "without image data augmentation" 조건이라, 이 표의 결과는 **VAT 자체의 regularization 효과**만 본 것. p.30의 strong augmentation 기반 consistency와는 별도의 축.

---

## FIGURES

- p.30: Noisy consistency training 구조도 ($(\mathbf{x},y)\to$ CE, $\mathbf{u}\to f_\theta$와 augmentation $\bar{\mathbf{u}}\to f_\theta$ 두 경로 → MSE, weighted sum $\mathcal{L}_s+\mu(t)\mathcal{L}_u$), 중요도 상
- p.32: 판다 → 노이즈 $\epsilon\,\text{sign}(\nabla_x J)$ → gibbon 99.3% adversarial example 그림, 중요도 상
- p.33: FGSM adversarial perturbation 수식 3줄 + 주석 박스 (GT one-hot, attacked 예측, "most confused" 방향), 중요도 상
- p.34: $\mathcal{L}_{\text{adv}} \to \mathcal{L}_u^{\text{VAT}}$ 수식 변환 + Adversarial Training vs VAT 비교 표, 중요도 상
- p.35: Table 3 (MNIST) / Table 4 (SVHN, CIFAR-10) test error rate, RPT vs VAT 비교, 중요도 중

## EXAM POINTS

1. **Noisy consistency training의 loss**: $\mathcal{L}=\mathcal{L}_s+\mu(t)\mathcal{L}_u$, $\mathcal{L}_s$ = labeled CE, $\mathcal{L}_u$ = 원본/증강본 예측 간 consistency (MSE), $\mu(t)$ = weight (ramp-up). 핵심 가정: **augmentation에 대한 prediction invariance**.
2. **Π-model / Mean Teacher / Noisy consistency 비교**: noise 위치 (network 내부 stochastic vs 입력 strong augmentation), target (자기 자신의 다른 pass / **EMA teacher** / 원본-증강본 self-consistency, teacher 없음).
3. **Adversarial Training 정의**: worst-case perturbation을 입력에 더하고 **정답을 맞히도록** 학습 → robustness. 판다 → gibbon 예시 ($\epsilon=0.007$, 57.7% → 99.3%).
4. **$r_{\text{adv}}$ 정의와 FGSM 근사**: $r_{\text{adv}}=\arg\max_{\|r\|\le\epsilon}D[q(y|\mathbf{x}^l),p_\theta(y|\mathbf{x}^l+r)]\approx\epsilon\,g/\|g\|_2\approx\epsilon\,\text{sign}(g)$, $g=\nabla_r D$. **1차 Taylor 전개 + norm 제약(L2 → 정규화 gradient, L∞ → sign)**으로 유도할 수 있어야 함.
5. **AT의 min-max 구조**: inner max (adversarial perturbation 탐색) / outer min (파라미터 학습). one-hot target일 때 $D$ = cross entropy.
6. **VAT loss**: $\mathcal{L}_u^{\text{VAT}}=D[p_{\hat\theta}(y|\mathbf{x}),p_\theta(y|\mathbf{x}+r_{\text{vadv}})]$, $r_{\text{vadv}}=\arg\max_{\|r\|\le\epsilon}D[p_{\hat\theta}(y|\mathbf{x}),p_\theta(y|\mathbf{x}+r)]$. $D$ = **KL divergence**, $\hat\theta$ = 현재 파라미터 상수 (no gradient).
7. **AT vs VAT 차이**: label 사용 여부 (ground-truth vs model prediction = "virtual" label), supervised vs semi-supervised, **robustness vs local smoothness of predictions**. 핵심 문장: "VAT replaces the ground-truth label with the model's own prediction → adversarial consistency training on unlabeled data."
8. **왜 VAT는 FGSM이 아니라 power iteration인가 (서술형 대비)**: $r=0$에서 KL이 최소 → gradient 0 → 2차 근사 $\tfrac12 r^\top H r$ → **$H$의 dominant eigenvector**가 $r_{\text{vadv}}$ → power iteration + finite difference ($Hd\approx\nabla_r D|_{r=\xi d}/\xi$), 1 iteration, backprop 1회.
9. **RPT vs VAT**: RPT = **random** direction smoothing, VAT = **most adversarial** direction smoothing → 같은 조건에서 VAT가 훨씬 낮은 error (CIFAR-10 18.56 → 14.87, MNIST-100 6.81 → 1.36). 이유: 고차원에서 random 방향은 대부분 모델이 이미 둔감.
10. **VAT + EntMin**: SVHN 4.28%, CIFAR-10 13.15%로 표 내 최고 (Conv-Large) — 두 정규화가 보완적.
