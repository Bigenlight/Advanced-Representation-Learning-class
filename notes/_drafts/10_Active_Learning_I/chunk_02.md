<!-- DRAFT | lecture: 10_Active_Learning_I | chunk: 02 | pages 20–36 -->
<!-- VERIFIED | fixes: 12 | verifier: sonnet ×2 (p.20–27, p.28–36) -->

## Diversity Sampling (p.20)

슬라이드 텍스트:
- Select representative samples that cover the data distribution.
  - Avoid selecting redundant samples
  - Cover different regions of the feature space
  - Improve generalization

**설명**
- 지금까지(p.9–19)는 "모델이 **헷갈리는** sample"을 골랐어. diversity sampling은 기준이 달라. "데이터 분포를 **골고루 대표하는** sample"을 골라.
- **representative sample** = 그 sample 하나가 주변의 비슷한 sample 여럿을 대신할 수 있는 것. 분포의 한 구역을 "대표"해.
- **redundant sample** = 이미 고른 것과 거의 같은 sample. label을 달아도 새로 배우는 게 거의 없어.
- **feature space** = 모델이 sample을 바꿔 놓은 embedding 벡터들이 사는 공간. 여기서 가까우면 모델 눈에 비슷한 sample이야.
- 왜 필요해? p.6 notation의 **b (batch size)** 를 떠올려. 한 round에 b개를 한꺼번에 고르면, uncertainty 점수만으로 top-b를 뽑을 때 decision boundary의 **같은 구역**에서 거의 같은 sample이 여러 개 뽑힐 수 있어. 예산 b개 중 대부분이 겹치는 정보야.
- 그래서 "feature space의 **다른 구역**을 덮자"는 게 핵심이야. 그러면 모델이 한 구역만이 아니라 전체 분포를 보고 배워서 **generalization**이 좋아져.
- 슬라이드엔 구체적인 식이나 알고리즘이 없어. 정의 문장 하나와 목적 세 개만 외우면 돼.

> 보충 (슬라이드 밖, 시험 범위 아님): 대표적인 diversity 방법으로 Core-set (Sener & Savarese, ICLR 2018)이 있어. 고른 집합이 나머지 전체를 feature space에서 가장 잘 덮도록(가장 먼 점까지의 거리가 최소가 되도록) 고르는 방식이야. p.23의 "Pairwise similarity computation" 비용이 바로 이런 거리 계산 얘기야.

---

## Expected Model Change (p.21)

슬라이드 텍스트:
- Select samples expected to change the model the most after labeling.
  - Large parameter update
  - Large loss reduction
  - High training impact

**설명**
- 기준: "label을 달고 학습에 넣었을 때 **모델을 가장 많이 바꿀** sample"을 골라.
- 바뀌는 정도를 재는 방법이 세 줄로 나와: parameter가 크게 움직이나(**Large parameter update**), loss가 크게 줄어드나(**Large loss reduction**), 학습에 주는 영향이 큰가(**High training impact**).
- 왜 "**expected**"(기댓값)야? (슬라이드엔 이유가 없어서 notes 해석이야, 아래 EGL 보충 참고) 고르는 시점엔 그 sample의 label을 **아직 몰라**. 그래서 가능한 label마다 "이 label이면 모델이 얼마나 바뀔까"를 계산하고, 모델이 예측한 확률로 가중 평균을 내.
- 비용 (notes 해석): 후보 sample마다, 가능한 label마다 gradient나 update를 계산해야 해. 그래서 p.23 Limitations의 "Expected model update estimation"이 비싼 항목으로 나와.

> 보충 (식은 notes 보충, 슬라이드엔 식 없음 → 시험 범위 아님): 대표적인 구현이 Expected Gradient Length (EGL, Settles, Craven & Ray, NIPS 2007)야.
> $$U_{\text{EGL}}(x) = \sum_{y} P_\theta(y \mid x)\, \big\lVert \nabla_\theta \ell(x, y) \big\rVert$$
> - $\ell(x,y)$: sample $x$에 label $y$를 달았을 때의 loss. $\nabla_\theta \ell$: 그 loss의 gradient = 학습 한 step에서 parameter가 움직일 방향과 크기.
> - $\lVert \cdot \rVert$: gradient의 길이. 길수록 "parameter를 크게 바꿀" sample이야.
> - $\sum_y P_\theta(y \mid x)\,\cdot$: label을 모르니까 예측 확률로 평균 낸 것. 이게 "expected"의 뜻이야.

---

## Hybrid Strategy (p.22)

슬라이드 텍스트:
- Combine multiple criteria for better sample selection.
  - Uncertain and representative
  - Informative and diverse
  - Most modern active learning methods

**설명**
- 기준 하나만 쓰면 각자 약점이 있어.
  - uncertainty만: 같은 구역의 비슷한 sample이 몰려 뽑혀 (p.20 설명).
  - diversity만: 골고루 뽑지만, 모델이 이미 잘 아는 쉬운 sample도 뽑혀서 예산이 새.
- 그래서 **둘 이상을 섞어**: "헷갈리면서 대표성도 있는" sample, "정보가 많으면서 서로 다른" sample.
- 슬라이드 마지막 줄 "**Most modern active learning methods**": 요즘 방법 대부분이 이 hybrid 쪽이라는 뜻이야.
- 섞는 방법의 예 (설명용, 슬라이드엔 방법 없음): uncertainty 상위 후보를 넉넉히 뽑은 뒤, 그 안에서 feature space의 서로 다른 구역에 있는 b개를 고르기.

> 보충 (슬라이드 밖, 시험 범위 아님): 대표 예로 BADGE (Ash et al., ICLR 2020)가 있어. last layer gradient embedding의 크기가 uncertainty를 반영하고, 그 공간에서 k-means++ 식으로 흩어지게 골라 diversity를 얻는 방법이야.

---

## Limitations (p.23)

슬라이드 텍스트:
- Hand-crafted acquisition functions
  - Entropy, Margin, Diversity, ...
  - No single strategy works best for all tasks
  - Difficult to adapt across datasets
- Computationally expensive
  - Committee models
  - Pairwise similarity computation
  - Expected model update estimation
  - Retraining after every acquisition round
- Separate objectives
  - Uncertainty
  - Diversity
  - Expected model change
- (굵게) Can we learn the acquisition function directly using deep neural networks?

**설명**
- p.9–22의 고전 전략(classical query strategies)의 한계를 세 묶음으로 정리해.
- **① Hand-crafted acquisition functions** (사람이 손으로 설계한 점수 함수):
  - Entropy, Margin, Diversity … 전부 사람이 정한 식이야.
  - 모든 task에서 제일 좋은 전략 하나는 없고, dataset이 바뀌면 다시 골라야 해.
- **② Computationally expensive**: (슬라이드엔 방법 이름이 없어. 아래 짝짓기는 notes 해석이야) 각 줄이 앞의 어느 방법 비용인지 짝지어 봐.
  - Committee models → QBC (p.14): model을 C개 학습해야 해.
  - Pairwise similarity computation → Diversity (p.20): sample 쌍마다 거리를 재야 해.
  - Expected model update estimation → Expected Model Change (p.21): 후보·label마다 update를 계산해야 해.
  - Retraining after every acquisition round → 모든 방법 공통. p.5 workflow의 "Retrain the model"을 round마다 반복해.
- **③ Separate objectives**: uncertainty, diversity, expected model change가 **따로 노는 목표**야. 하나의 기준으로 묶이지 않아서 hybrid(p.22)도 "어떻게 섞을지"를 또 사람이 정해야 해.
- 마지막 질문 "**Can we learn the acquisition function directly using deep neural networks?**"가 다음 섹션(Deep Active Learning, p.24–)으로 넘어가는 다리야.

> 보충 (해석): p.24–35는 "deep network로 uncertainty를 어떻게 잘 재나"(ensemble, MC Dropout, BALD)를 다뤄. acquisition function **자체를 학습**하는 방법은 이 강의 슬라이드엔 아직 안 나와. p.36 "Next Week.. Deep Active Learning .."으로 이어질 것 같아 (예고, 다음 강의에서 확인).

---

## Section: Deep Active Learning (p.24)

슬라이드 텍스트:
- Deep Active Learning

**설명**
- 섹션 구분 슬라이드야. p.2 목차의 "3. Deep Active Learning (Ensemble-based Estimation)"이 여기서 시작해 (p.24–35).
- 질문이 바뀌어: "어떤 점수로 고를까"에서 "**deep network에서 그 uncertainty를 어떻게 믿을 만하게 재나**"로.
- 괄호 속 "Ensemble-based Estimation"이 답의 방향이라는 건 목차(p.2) 제목에서 읽은 notes 해석이야: 여러 model(또는 그 흉내)의 예측이 **엇갈리는 정도**로 uncertainty를 재.

---

## Measuring Uncertainty (p.25)

슬라이드 텍스트:
- Aleatoric Uncertainty: Uncertainty caused by data noise
  - Sensor noise
  - Occlusion
  - Motion blur
  - Intrinsic ambiguity
  - **Cannot be reduced by collecting more data**
- Epistemic Uncertainty: Uncertainty caused by limited model knowledge
  - Model has not seen similar examples
  - Large uncertainty in model parameters
  - **Can be reduced with more labeled data**
- (굵게) Active Learning mainly targets Epistemic Uncertainty.

**설명**
- uncertainty(불확실성)에는 **원인이 다른 두 종류**가 있어. 이 구분이 p.29–31 BALD의 핵심이라 꼭 외워.
- **Aleatoric uncertainty** (aleatoric = "주사위의", 우연에서 오는) = **data 자체의 noise** 때문에 생기는 불확실성.
  - 예: sensor noise, occlusion(물체가 가려짐), motion blur(움직여서 번짐), intrinsic ambiguity(원래 애매함, 예: 3인지 8인지 사람도 모르는 손글씨).
  - **data를 더 모아도 안 줄어들어.** 번진 사진에 label을 100개 달아도 그 사진은 여전히 번져 있어.
- **Epistemic uncertainty** (epistemic = "지식의") = **모델이 아직 몰라서** 생기는 불확실성.
  - 예: 비슷한 sample을 본 적이 없음, model parameter가 아직 확실히 안 정해짐.
  - **labeled data를 더 주면 줄어들어.** 본 적 없는 종류를 보여 주면 배워.
- 결론 문장: "**Active Learning mainly targets Epistemic Uncertainty.**" label을 달아서 줄일 수 있는 건 epistemic뿐이니까, label 예산은 epistemic이 큰 sample에 써야 해.
- 앞 전략과의 연결: LC·Margin·Entropy(p.11–13)는 model 하나의 예측 확률만 봐서, 확률이 애매하면 그 이유가 data noise인지 모델의 무지인지 **구분을 못 해**. 그래서 번진 사진처럼 label을 달아도 소용없는 sample에 예산을 쓸 수 있어. 이 문제를 BALD(p.29–31)가 풀어.

| | Aleatoric | Epistemic |
|---|---|---|
| 원인 | data noise | limited model knowledge |
| 예 | sensor noise, occlusion, motion blur, intrinsic ambiguity | 비슷한 sample을 본 적 없음, parameter uncertainty 큼 |
| data를 더 모으면 | **안 줄어듦** (Cannot be reduced) | **줄어듦** (Can be reduced with more labeled data) |
| AL의 target? | 아님 | **맞음** (mainly targets) |

> RL 비유 (슬라이드 밖, 시험 범위 아님): aleatoric은 환경 자체의 stochasticity(같은 state·action에도 결과가 매번 달라짐), epistemic은 agent가 그 state를 아직 덜 탐험해서 모르는 것이야. exploration bonus는 epistemic 쪽만 보상해야 해. prediction error를 그대로 curiosity로 쓰면 무작위로 바뀌는 화면("noisy TV")처럼 aleatoric만 큰 곳에 agent가 붙잡혀. 고칠 수 없는 noise에 예산을 쓰는 것과 같은 실패야.

---

## Ensemble-based Estimation (p.26)

슬라이드 텍스트:
- Idea
  - Train multiple models independently.
  - If they disagree,
  - → the sample is likely informative.
  - EX) AdaBoost, Bootstrapping, Random forests
- Problems
  - Need multiple networks
  - High training cost
  - High inference cost

**설명**
- **ensemble** = model 여러 개를 따로 학습해서 예측을 합치는 방법이야.
- 아이디어: model 여러 개를 **독립적으로** 학습해. 어떤 sample에서 그 model들의 예측이 **엇갈리면**, 그 sample은 informative(정보가 많을) 가능성이 커.
- 왜 엇갈림이 epistemic의 신호야? 학습 data가 많은 구역에선 model들이 다 같은 답으로 수렴해. data가 없는 구역에선 각자 다르게 추측해. 그래서 **엇갈림 = 모델이 아직 모르는 구역**이야. (직관 설명, 슬라이드 p.26은 "likely informative"까지만; epistemic과의 연결은 p.27에 나와)
- QBC(p.14)와 같은 아이디어야. QBC의 committee가 여기선 deep network ensemble이야.
- 예시 세 개 (ensemble 학습의 고전 기법):
  - **Bootstrapping**: 학습 data에서 복원 추출로 data set을 여러 벌 만들어 model마다 다르게 학습 (bagging의 바탕).
  - **Random forests**: bootstrapping으로 decision tree를 여러 개 만들고, 갈라 볼 feature도 무작위로 골라 다양성을 키운 것.
  - **AdaBoost**: 앞 model이 틀린 sample에 가중치를 더 줘서 다음 model을 학습하는 boosting 방식. ← 다음 model이 앞 model의 오류를 보고 학습되니까 엄밀히는 "independently"가 아니야. 슬라이드엔 같은 예시 줄에 있어서 원문 유지.
- **Problems**: network가 여러 개 필요해 → 학습 비용도, 추론 비용도 그 배수로 커. deep network 하나도 학습이 오래 걸리는데 그걸 C번 해야 해. p.23의 "Committee models" 비용과 같은 얘기야.
- 이 비용 문제가 바로 다음 슬라이드 MC Dropout(p.27)의 동기야.

---

## Approximate Ensemble: MC Dropout (p.27)

슬라이드 텍스트:
- MC Dropout
  - Instead of training multiple networks, apply dropout during inference.
- Key Idea
  - Different dropout masks
  - ≈ Different models
- Advantages
  - Single network
  - Low memory
  - Good approximation to Bayesian inference
- (굵게) MC Dropout performs multiple stochastic forward passes with different dropout masks, approximating an ensemble of neural networks. The variance of their predictions is used as an estimate of epistemic uncertainty.
- Gal et al., ICML 2016, Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning

**설명**
- 먼저 **dropout**: 학습 중 forward마다 neuron을 확률 $p$로 무작위로 꺼(0으로 만들어). 어느 neuron을 끌지 정한 0/1 패턴을 **dropout mask**라고 해. 원래 목적은 overfitting 방지야.
- 보통은 **test(inference) 때 dropout을 끄고** network 전체를 써. 그러면 같은 입력엔 늘 같은 출력이 나와.
- **MC Dropout**은 반대로 **inference 때도 dropout을 켜 둬**. 같은 입력을 $T$번 넣으면 매번 mask가 달라서 출력이 매번 조금씩 달라.
  - **MC = Monte Carlo**: 무작위 sample을 여러 번 뽑아 평균으로 기댓값을 근사하는 방법.
- **Key Idea**: mask가 다르면 사실상 **다른 sub-network**야 → "Different dropout masks ≈ Different models". 그래서 network 하나로 ensemble을 흉내 내 (= **approximate ensemble**).
- **Advantages**:
  - Single network: 학습은 한 번만.
  - Low memory: weight 한 벌만 저장.
  - Good approximation to Bayesian inference: 아래 보충.
- 굵은 문장: $T$번의 stochastic forward pass(무작위성이 들어간 forward) 결과가 ensemble 예측 역할을 하고, 그 **예측들의 variance(퍼진 정도)** 를 epistemic uncertainty의 추정으로 써.
- **Bayesian inference** (기본 용어): model parameter $\theta$를 값 하나로 정하지 않고 **확률분포**로 다루는 방식이야.
  - **posterior** $p(\theta \mid \mathcal{D})$: data $\mathcal{D}$를 본 뒤 "parameter가 이 값일 가능성"의 분포.
  - 예측은 가능한 모든 $\theta$의 예측을 posterior로 가중 평균해: $p(y \mid x, \mathcal{D}) = \int p(y \mid x, \theta)\, p(\theta \mid \mathcal{D})\, d\theta$ (notes 보충, 슬라이드엔 p.29에 $p(y \mid x, \mathcal{D})$ 기호만).
  - deep network에선 이 적분을 정확히 못 해. MC Dropout은 dropout mask마다 $\theta_t$ 하나를 (dropout이 만드는 근사 posterior에서) 뽑은 걸로 보고 $\frac{1}{T}\sum_t p(y \mid x, \theta_t)$로 근사해. 이게 "approximation to Bayesian inference"의 뜻이야.
- 주의 (p.27 vs p.28–29): 이 슬라이드는 epistemic 추정으로 "**variance** of their predictions"를 말해. 다음 DBAL(p.28–29)은 같은 $T$번의 예측에서 **BALD(mutual information)** 를 계산해. 둘 다 "여러 예측이 얼마나 엇갈리나"를 재지만 **식이 달라**. 시험에서 "MC Dropout = variance"와 "DBAL = BALD"를 섞어 묻는지 조심해.

> 보충: 원 논문은 Gal & Ghahramani, "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning", ICML 2016이야 (슬라이드 cite "Gal et al., ICML 2016"과 일치, 저자 두 명). dropout으로 학습한 network가 Bayesian 모델의 근사(variational inference)로 해석된다는 걸 보였어.
>
> 구현 보충 (시험 범위 아님): PyTorch에선 `model.eval()`이 dropout을 끄니까, MC Dropout은 dropout layer만 train mode로 켜 둔 채 `torch.no_grad()`로 $T$번 forward해. BatchNorm까지 train mode로 켜면 통계가 batch마다 바뀌어서 다른 문제가 생겨.

> RL 비유 (슬라이드 밖, 시험 범위 아님): model-based RL 연구 일부(Gal et al. 2016의 PILCO 확장 등)도 dynamics model의 epistemic uncertainty를 dropout을 켠 채 여러 번 sampling해서 재. ensemble을 C개 학습하는 대신 network 하나로 근사한다는 동기가 같아. (실제로는 ensemble이 더 잘 되는 경우가 많아.)

---

## Deep Bayesian Active Learning (DBAL) (p.28)

슬라이드 텍스트:
- Goal: We want to find "samples that the model is truly unsure about."
- Key Idea
  - Approximate Bayesian inference using **MC Dropout**
  - Estimate epistemic uncertainty by running multiple forward passes with different dropout masks
  - Query samples with the highest uncertainty
- Pipeline
  - Train a classifier with dropout
  - Perform multiple stochastic forward passes
  - **Estimate predictive uncertainty (BALD)**
  - Select the most informative unlabeled samples
  - Query labels and retrain
- Gal et al., ICML 2017, Deep Bayesian Active Learning with Image Data

**설명**
- **DBAL = Deep Bayesian Active Learning**. MC Dropout(p.27)을 active learning loop에 넣은 방법이야.
- Goal의 "**truly** unsure": 그냥 확률이 애매한 게 아니라 **모델이 진짜 몰라서**(epistemic) 헷갈리는 sample을 찾자는 뜻이야. p.25 구분을 그대로 쓴 표현이야.
- **Pipeline** (순서 그대로 외워, 순서 바꾸기 T/F 후보):
  1. dropout이 들어간 classifier를 현재 labeled set으로 학습해.
  2. unlabeled sample마다 dropout을 켠 채 여러 번($T$번) forward해.
  3. 그 $T$개 예측으로 **BALD** 점수를 계산해 (p.29).
  4. BALD가 가장 큰 sample들을 골라.
  5. 사람에게 label을 받고(query) 다시 학습해 → 1로 돌아가.
- p.5 workflow(Select → Human annotation → Retrain → Repeat)와 같은 loop야. 달라진 건 "Select" 단계의 점수가 **MC Dropout + BALD**라는 것뿐이야.

> 보충: 원 논문은 Gal, Islam & Ghahramani, "Deep Bayesian Active Learning with Image Data", ICML 2017 (PMLR 70)이야. 슬라이드 cite와 일치해. MNIST와 피부암 병변 이미지(ISIC 2016)로 실험했어.

---

## BALD Acquisition Function (p.29)

슬라이드 텍스트:
- BALD measures the mutual information between the label y of sample x and the model parameters θ.
- $\mathrm{BALD}(x) = I(y, \theta \mid x, \mathcal{D})$
- $\mathrm{BALD}(x) = \mathcal{H}[p(y \mid x, \mathcal{D})] - \mathbb{E}_{p(\theta \mid \mathcal{D})}\big[\mathcal{H}[p(y \mid x, \theta)]\big]$
- (박스) BALD = Entropy of mean prediction − Mean entropy of predictions
- $\mathrm{BALD}(x) \approx \mathcal{H}[\bar{p}(y \mid x)] - \dfrac{1}{T}\sum_{t=1}^{T} \mathcal{H}[p_t(y \mid x)]$
- (빨간 박스, 첫 항 안) Mean prediction
- (첫 항 아래) How uncertain is the average of all models?
- (둘째 항 아래) How uncertain is each model individually?

**설명**
- **BALD**는 (약자 풀이는 아래 보충) 이름대로 "model들의 disagreement"를 재는 acquisition function이야.
- **mutual information** $I(y, \theta)$ (기본 용어): 두 변수가 서로에 대해 알려 주는 정보량이야. "$\theta$를 알면 $y$에 대한 불확실성이 얼마나 줄어드나" = "$y$를 알면 $\theta$에 대한 불확실성이 얼마나 줄어드나" (대칭이야).
  - active learning 쪽으로 읽으면: "**이 sample의 label $y$를 받으면 model parameter $\theta$에 대해 얼마나 많이 알게 되나**". 딱 label을 달 가치의 정의야.
- 기호:
  - $\mathcal{H}[\cdot]$: Shannon entropy, $\mathcal{H}[p] = -\sum_y p(y)\log p(y)$. 분포가 고를수록 커.
  - $p(y \mid x, \mathcal{D})$: posterior로 평균 낸 예측 (p.27 설명). = 모든 model의 **평균 예측**.
  - $p(y \mid x, \theta)$: parameter가 $\theta$인 model **하나**의 예측.
  - $\mathbb{E}_{p(\theta \mid \mathcal{D})}[\cdot]$: posterior에서 $\theta$를 뽑아 평균.
  - $\bar{p}(y \mid x) = \frac{1}{T}\sum_t p_t(y \mid x)$: MC Dropout $T$번 예측의 평균 (= Mean prediction).
  - $p_t(y \mid x)$: $t$번째 dropout mask로 얻은 예측.
- **두 항 풀기**:
  - 첫 항 $\mathcal{H}[\bar{p}]$ = **평균 예측의 entropy** = "How uncertain is the average of all models?" 모든 원인의 uncertainty가 다 들어 있어.
  - 둘째 항 $\frac{1}{T}\sum_t \mathcal{H}[p_t]$ = **각 예측 entropy의 평균** = "How uncertain is each model individually?" model 하나하나가 이미 헷갈리는 정도야. data noise 때문이면 모든 model이 다 헷갈리니까 이 항이 커.
  - **빼면** = 평균은 헷갈리는데 각자는 확신하는 정도 = **model끼리 엇갈리는 정도** = epistemic.
- 순서 함정: "Entropy **of mean**" − "Mean **of entropy**". 두 항의 순서를 바꾸면 안 돼.

> 보충 (슬라이드 밖, 시험 범위 아님): 슬라이드 식을 옮기면 "전체 = (근사적으로) aleatoric + epistemic" 분해가 돼.
> $$\underbrace{\mathcal{H}[\bar{p}]}_{\text{total}} = \underbrace{\tfrac{1}{T}\textstyle\sum_t \mathcal{H}[p_t]}_{\approx\ \text{aleatoric}} + \underbrace{\mathrm{BALD}(x)}_{\approx\ \text{epistemic}}$$
> entropy는 concave라서 평균의 entropy ≥ entropy의 평균 (Jensen 부등식) → BALD ≥ 0. 두 항 순서를 뒤집으면 값이 0 이하가 돼.

> 보충: BALD = **B**ayesian **A**ctive **L**earning by **D**isagreement (슬라이드엔 풀이 없음). BALD는 Houlsby, Huszár, Ghahramani & Lengyel, "Bayesian Active Learning for Classification and Preference Learning" (arXiv 2011)에서 나온 이름이야. 슬라이드엔 이 cite가 없고 DBAL(p.28)이 이걸 MC Dropout으로 계산해 썼어.

---

## BALD Acquisition Function — Case 1: Aleatoric (p.30)

슬라이드 텍스트:
- Case 1: Aleatoric uncertainty (Inherently ambiguous data)
  - All models make similar predictions and are equally uncertain.
  - Sample x (blurred image)
  - Predictions from sampled models
  - Average prediction
  - Entropy: High
  - BALD(x) ≈ High − High ≈ 0 (Small)
  - → Labeling this sample is unlikely to significantly improve the model.
- 그림: 번진 사진 하나 → Model 1, Model 2, …, Model T가 모두 cat·dog 막대 높이가 같음 (각각 "Entropy: High"; 막대는 0.5 눈금보다 살짝 위로 그려졌지만 두 class 합이 1이니 반반으로 읽어) → 평균 예측도 cat·dog 반반 ("Entropy: High") → 박스 "BALD(x) ≈ High − High ≈ 0 (Small)" → "Labeling this sample is unlikely to significantly improve the model."

**설명**
- 번진 사진은 **어느 model이 봐도** 반반이야. model끼리 **의견이 같아** (다 같이 모르겠다고 해).
- 첫 항(평균의 entropy) High, 둘째 항(각자 entropy의 평균)도 High → 빼면 ≈ 0.
- 해석: 이 sample의 불확실성은 data noise(aleatoric) 때문이야. label을 달아도 사진은 여전히 번져 있으니 모델이 크게 좋아지지 않아 → **고르지 마**.
- 예시로 보면 (설명용 숫자, 슬라이드 수치 아님, 자연로그 ln 기준):
  - $T = 4$, 네 예측 모두 $p_t = (0.5,\ 0.5)$ (cat, dog)
  - 평균 $\bar{p} = (0.5,\ 0.5)$ → $\mathcal{H}[\bar{p}] = \ln 2 \approx 0.693$ (2-class 최대값)
  - 각자 $\mathcal{H}[p_t] = 0.693$ → 평균 0.693
  - BALD $= 0.693 - 0.693 = 0$
- 주의: 이 sample은 **Consensus Entropy(p.17)나 평균 예측에 적용한 Entropy sampling(p.13)으로는 최고점**이야 (평균 예측 entropy가 최대). 이 case는 model 하나의 entropy로 봐도 최고점이야 (모든 model이 반반이니까). 그래서 그 방법들은 이런 sample에 예산을 써 버려. BALD는 둘째 항으로 이 몫을 빼서 피해.

---

## BALD Acquisition Function — Case 2: Epistemic (p.31)

슬라이드 텍스트:
- Case 2: Epistemic uncertainty (Model lacks knowledge)
  - Different models make different predictions, while each model is individually confident.
  - Sample x (ambiguous image)
  - Predictions from sampled models
  - Average prediction
  - Entropy: High
  - BALD(x) ≈ High − Low = Large
  - → Labeling this sample is expected to provide the greatest improvement to the model.
- 그림: 고양이 사진 하나 → Model 1은 cat ≈ 0.9, Model 2는 dog ≈ 0.9, Model T는 cat ≈ 0.95 (각각 "Entropy: Low") → 평균 예측은 cat·dog 비슷 ("Entropy: High") → 박스 "BALD(x) ≈ High − Low = Large" → "Labeling this sample is expected to provide the greatest improvement to the model."

**설명**
- model 하나하나는 **확신**해(entropy Low). 그런데 **서로 다른 답**을 확신해. 평균을 내면 반반이 돼서 평균의 entropy는 High야.
- High − Low = Large → **고를 sample**이야. model들이 엇갈린다는 건 이 구역의 data가 부족해서 parameter가 아직 안 정해졌다는 뜻이고(epistemic), label 하나로 그 엇갈림을 정리할 수 있어.
- 예시로 보면 (설명용 숫자, 슬라이드 수치 아님, ln 기준):
  - $T = 4$, $p_1 = (0.9, 0.1)$, $p_2 = (0.1, 0.9)$, $p_3 = (0.9, 0.1)$, $p_4 = (0.1, 0.9)$
  - 평균 $\bar{p} = (0.5,\ 0.5)$ → $\mathcal{H}[\bar{p}] = 0.693$
  - 각자 $\mathcal{H}[(0.9, 0.1)] = -(0.9\ln 0.9 + 0.1\ln 0.1) \approx 0.325$ → 평균 0.325
  - BALD $\approx 0.693 - 0.325 = 0.368$ (Case 1의 0보다 큼)
- 두 case 나란히 (설명용):

| | Case 1 (aleatoric) | Case 2 (epistemic) | (비교) 모두 같은 답을 확신 |
|---|---|---|---|
| 각 model 예측 | 모두 (0.5, 0.5) | (0.9, 0.1)과 (0.1, 0.9)가 섞임 | 모두 (0.9, 0.1) |
| $\mathcal{H}[\bar{p}]$ (첫 항) | 0.693 High | 0.693 High | 0.325 Low |
| 평균 $\mathcal{H}[p_t]$ (둘째 항) | 0.693 High | 0.325 Low | 0.325 Low |
| BALD | **0** | **0.368 Large** | 0 |
| 평균 예측의 Entropy sampling 점수 (= 첫 항, DBAL의 Max Entropy) | 최고 | 최고 | 낮음 |
| model 하나의 Entropy sampling 점수 (p.13) | 최고 | 낮음 (각 model이 확신) | 낮음 |
| 고를까? | 아니 | **응** | 아니 (이미 앎) |

- 핵심 관찰: Case 1과 2는 **첫 항(= Consensus Entropy 점수, 평균 예측에 적용한 Entropy sampling)이 똑같아**. 첫 항만 보면 두 sample을 구분할 수 없어. 둘째 항이 있어야 aleatoric과 epistemic이 갈려.
- 반대로 model **하나**의 entropy만 보면 (notes 해석) Case 2에서 각 model이 확신하니까 점수가 낮아. 그러면 label을 줄 가치가 큰 epistemic sample을 놓쳐. "평균 예측의 entropy"와 "model 하나의 entropy"를 구분해.
- ← 표기 주의: 이 epistemic case의 sample을 슬라이드는 "**ambiguous image**"라고 불러. 그런데 p.25에선 "**Intrinsic ambiguity**"가 **aleatoric**의 예시야. 원문 그대로 두되, 시험에선 p.25 정의대로 "intrinsic ambiguity = aleatoric"으로 답하는 게 안전해. 여기서 "ambiguous"는 "model들이 서로 다르게 보는" 정도의 뜻으로 읽어.

> 보충 (슬라이드 밖 유도, 시험 범위 아님): QBC의 KL disagreement(p.18)를 전개하면 $\frac{1}{C}\sum_c D_{KL}(P_{\theta_c} \,\|\, P_C) = \mathcal{H}[P_C] - \frac{1}{C}\sum_c \mathcal{H}[P_{\theta_c}]$가 돼. committee $C$개를 posterior에서 뽑은 model로 보면 **BALD와 같은 식**이야. 그리고 첫 항 $\mathcal{H}[P_C]$가 바로 Consensus Entropy(p.17)야. 즉 Consensus Entropy는 BALD의 첫 항만 쓴 것이라 Case 1을 못 걸러.

---

## Deep Bayesian Active Learning (DBAL) — Observation (p.32)

슬라이드 텍스트:
- Observation
  - MC Dropout consistently outperforms random sampling.
  - Bayesian uncertainty leads to better sample selection.
- 그림: x축 0–1000 (number of acquired images), y축 94–100 (MNIST test accuracy). 선 다섯 개: BALD, Var Ratios, Max Entropy, MBR, Random.
- 그림 캡션: "Figure 3. MNIST test accuracy (two digit classification) as a function of number acquired images, compared to a current technique for active learning of image data: MBR (Zhu et al., 2003)."
- (굵게) More informative samples → Higher accuracy with the same labeling budget
- (굵게) BALD measures how much the models disagree with each other. It selects samples with high epistemic uncertainty, where acquiring labels is expected to provide the greatest improvement in model performance.
- Gal et al., ICML 2017, Deep Bayesian Active Learning with Image Data

**설명**
- 그래프 읽는 법: x축은 지금까지 label을 단 이미지 수(= 쓴 예산), y축은 그때의 test accuracy야. **같은 x에서 위에 있을수록** 같은 예산으로 더 잘 고른 방법이야.
- 선 이름:
  - BALD, Var Ratios, Max Entropy: MC Dropout 예측으로 계산한 acquisition function들 (Max Entropy = 평균 예측의 entropy, p.13의 Entropy sampling을 MC Dropout 평균에 적용한 것).
  - Random: 무작위로 고르기 (baseline).
  - MBR: 비교 대상인 기존 active learning 방법 (Zhu et al., 2003).
- 볼 것:
  - MC Dropout 기반 세 선(BALD, Var Ratios, Max Entropy)이 Random보다 **꾸준히 위**에 있어 → "MC Dropout consistently outperforms random sampling."
  - 세 선끼리는 서로 가까워. 이 그래프에서 BALD가 나머지 둘을 크게 앞서진 않아. 이 슬라이드의 주장은 "Bayesian uncertainty(MC Dropout)로 고르면 random보다 낫다"야.
  - MBR은 초반엔 높지만 곧 평평해지고, 나중엔 MC Dropout 기반 방법과 Random보다 아래야.
- 결론 문장 두 개는 그대로 외워: "More informative samples → Higher accuracy with the same labeling budget", "BALD … selects samples with high **epistemic** uncertainty".
- 수치는 비중 낮음. 축 범위(94–100%)가 좁아서 선 사이 차이가 커 보이는 것도 감안해.

> 보충 (슬라이드 밖, 시험 범위 아님): Var Ratios(variation ratios)는 $T$번 예측 중 가장 많이 나온 class의 비율을 1에서 뺀 값이야. MC Dropout판 voter entropy(p.16)와 비슷한 "표가 얼마나 갈리나" 점수야.

---

## Beyond MC Dropout — Limitation (p.33)

슬라이드 텍스트:
- Limitation
  - **Naïve ensembles** provide the most reliable uncertainty estimates.
  - However, training many independent networks is **computationally expensive**.

**설명**
- **naïve ensemble** = 꾀를 안 부린 ensemble: network $N$개를 처음부터 **따로따로** 학습하는 것 (p.26 방식).
- 문장 두 개가 trade-off야:
  - uncertainty 추정은 naïve ensemble이 **가장 믿을 만해** (most reliable).
  - 하지만 network를 여러 개 독립 학습하는 건 **비싸**.
- 제목 "Beyond MC Dropout": MC Dropout 말고도 이 trade-off 사이에서 비용을 줄이는 방법들이 있어. 다음 슬라이드(p.34)가 그 목록이야.
- 함정: "MC Dropout gives the most reliable uncertainty estimates"는 **False**야. 가장 믿을 만한 건 naïve ensemble이고, MC Dropout은 "**one of the best trade-offs**"(p.35)야.

> 보충 (슬라이드 밖, 시험 범위 아님): Beluch et al., "The Power of Ensembles for Active Learning in Image Classification", CVPR 2018이 active learning에서 ensemble이 MC Dropout보다 낫고 calibration도 좋다고 보고했어. 슬라이드엔 cite가 없어.

---

## Beyond MC Dropout — Efficient Ensemble Approximations (p.34)

슬라이드 텍스트:
- Efficient Ensemble Approximations
  - **Snapshot** Ensemble
    - Save multiple checkpoints from one training run using a cyclic learning rate.
    - Different local minima act as an implicit ensemble.
  - **Diversity Encouraging** Ensemble (DEE)
    - Initialize multiple models from a shared backbone.
    - Encourage prediction diversity with dropout.
  - **Multi-head** / **Split-head** Network
    - Share one feature extractor.
    - Use multiple prediction heads as lightweight ensemble members.
- 그림 네 칸:
  1. Naïve Ensemble (Independent): "Train N models independently from scratch." Model 1…N이 각자 Data로 학습 → Ensemble Prediction (average / vote)
  2. Snapshot Ensemble: "One training run with cyclic LR. Save multiple checkpoints." Loss–Time 곡선이 오르내리고 골짜기마다 ckpt 1, ckpt 2, …, ckpt K → "One model, multiple snapshots (different local minima)" → Ensemble Prediction
  3. Diversity Encouraging Ensemble (DEE): "Pretrain a backbone, then initialize multiple models and fine-tune with dropout to encourage diversity." Pretrained Backbone (shared initialization) → Fine-tune Model 1…N + Dropout → Ensemble Prediction
  4. Multi-head / Split-head: "Share one feature extractor. Use multiple heads as ensemble members." Shared Feature Extractor (Backbone) → Head 1…N → Ensemble Prediction

**설명**
- 셋 다 "ensemble의 **다양성**은 살리고 **비용**은 줄이자"는 방법이야. 어디서 비용을 아끼는지가 서로 달라.
- **Snapshot Ensemble**: 학습을 **한 번만** 해.
  - **cyclic learning rate**: learning rate를 크게 → 작게를 주기적으로 반복하는 schedule이야. 작아질 때마다 model이 어떤 local minimum(loss 골짜기)에 내려앉고, 다시 커지면 튀어나가서 다른 골짜기로 가.
  - 골짜기마다 checkpoint(그 시점의 weight)를 저장하면 서로 다른 model이 $K$개 생겨 → "Different local minima act as an implicit ensemble."
  - 아끼는 것: **학습 횟수** (1번). 추론은 여전히 $K$번.
- **DEE (Diversity Encouraging Ensemble)**: 미리 학습한 backbone 하나로 model 여러 개를 **초기화**하고, dropout을 넣고 fine-tune해서 예측이 서로 달라지게 해.
  - 아끼는 것: 처음부터 학습하는 비용 (pretrain은 한 번, model마다 fine-tune만).
  - 문제: 같은 backbone에서 출발하면 model들이 비슷해지기 쉬워 → dropout으로 일부러 다양성을 키워 ("Encourage prediction diversity with dropout").
- **Multi-head / Split-head**: feature extractor(backbone)는 **하나를 공유**하고, 끝의 prediction head만 여러 개 둬. head 하나하나가 ensemble member야.
  - 아끼는 것: backbone이 하나라 memory·계산 둘 다 (head는 작으니까 "lightweight").
- 그림 첫 칸 Naïve Ensemble은 비교 기준이야. 네 칸 모두 마지막엔 예측을 평균하거나 투표해(average / vote).

| | 학습 | 무엇을 공유하나 | 다양성은 어디서 |
|---|---|---|---|
| Naïve Ensemble | N번, 처음부터 따로 | 없음 | 서로 다른 초기화·학습 |
| Snapshot Ensemble | **1번** (cyclic LR) | 학습 run 하나 | 서로 다른 local minima |
| DEE | pretrain 1번 + fine-tune N번 | pretrained backbone (초기값) | fine-tune 때 dropout |
| Multi-head / Split-head | 1번 | **feature extractor** | head마다 다른 weight |
| MC Dropout (p.27) | 1번 | network 전체 | inference 때 dropout mask |

*(표는 notes 정리야. 슬라이드 원문은 위 bullet과 p.34 그림.)*

> 보충: 슬라이드엔 세 방법 모두 cite가 없어. Snapshot Ensemble은 Huang et al., "Snapshot Ensembles: Train 1, Get M for Free", ICLR 2017로 확인돼. "Diversity Encouraging Ensemble (DEE)"라는 이름의 논문은 웹에서 찾지 못했어 → 슬라이드 설명(공유 backbone 초기화 + dropout fine-tune)대로만 외워. Multi-head도 특정 논문 cite 없이 일반 구조로 나와.

> RL 비유 (슬라이드 밖, 시험 범위 아님): Bootstrapped DQN이 정확히 multi-head 구조야. torso(feature extractor) 하나에 Q-head $K$개를 달아. head들이 엇갈리는 정도가 그 state의 epistemic uncertainty를 나타내고, episode마다 head 하나를 뽑아 행동하는 방식으로 exploration해 (head disagreement를 bonus로 직접 쓰는 변형도 있어). "공유 backbone + 여러 head = 싼 ensemble"이라는 구조가 p.34 multi-head와 같아.

---

## Beyond MC Dropout — Takeaway (p.35)

슬라이드 텍스트:
- Takeaway
  - Accuracy: Naïve Ensemble > Efficient Approximations
  - Efficiency: Efficient Approximations > Naïve Ensemble
  - MC Dropout remains one of the best trade-offs between uncertainty quality and computational cost.

**설명**
- 부등호 두 개는 **방향이 반대**야. 뒤집기 T/F 1순위 후보야.
  - Accuracy(uncertainty 추정의 질): Naïve Ensemble **>** Efficient Approximations
  - Efficiency(비용): Efficient Approximations **>** Naïve Ensemble
- 결론: **MC Dropout**은 uncertainty quality와 computational cost 사이에서 "**one of the best trade-offs**"야.
  - "가장 정확하다"가 아니야 (그건 naïve ensemble, p.33).
  - "가장 싸다"도 슬라이드가 말한 게 아니야. "질과 비용의 균형이 좋다"까지만 말해.
- 강의 §3 전체 흐름 한 줄: 정확한 ensemble은 비싸다(p.26) → network 하나로 흉내 낸다(MC Dropout, p.27) → 그 예측들로 epistemic만 골라 잰다(BALD, p.29–31) → active learning에 넣으니 random보다 낫다(DBAL, p.32) → 다른 싼 근사도 있지만 MC Dropout이 여전히 좋은 균형(p.33–35).

---

## Next Week.. (p.36)

슬라이드 텍스트:
- Next Week..
- Deep Active Learning ..

**설명**
- 다음 강의(W13 11/26, Active Learning II)도 **Deep Active Learning**을 이어서 다뤄.
- p.23의 질문 "Can we learn the acquisition function directly using deep neural networks?"가 아직 답을 안 받은 채 남아 있어. 다음 강의에서 확인할 것 (예고, 내용 추측 아님).
