<!-- DRAFT | lecture: 03_Contrastive_Learning_I | chunk: 03 | pages 17–24 -->
<!-- VERIFIED | fixes: 0 | verifier: sonnet -->

## Lifted Structured Loss (p.17)

슬라이드 내용 (제목은 Lifted Structured Loss지만 실제로는 한계점 정리 + N-pair로의 전환 슬라이드):
- Limitations of Lifted Structured Loss
  - Requires dense pairwise comparisons (O(N^2))
  - Still fundamentally pair-based
    - Optimize pair relations instead of directly comparing multiple classes
  - Does not fully exploit multi-class batch structure
    - 1 positive <-> 1 negative
  - Optimization becomes increasingly complex
- Can we compare one positive against all negatives simultaneously?
- → N-pair Loss (NIPS 2016)
  - 1 positive <-> many negatives
  - → More information
  - → No need hard negative mining

그림 없음 (텍스트 슬라이드).

**설명**
- p.16("Why Replace Max?")에서 Lifted Structured Loss의 hard-negative **max** 항을 smooth하게 바꾼 흐름의 연장으로, 이 방법의 **남은 한계**를 짚고 다음 방법으로 넘어가는 연결 슬라이드임.
- **O(N^2) dense pairwise**: batch 안의 모든 positive pair에 대해 모든 negative와의 거리를 계산해야 해서 비용이 batch size의 제곱으로 증가함.
- **Still pair-based**: loss가 여전히 "(anchor, 샘플 하나)" 쌍 단위 관계를 최적화함. anchor 하나가 **여러 class를 동시에 보고** 비교하는 구조가 아니어서 한 번의 update에서 얻는 정보량이 제한됨.
- **1 positive <-> 1 negative**: triplet/lifted 모두 negative를 하나씩(pair별 항) 대비시킴. batch가 multi-class 구조를 가지는데 이를 충분히 활용하지 못함.
- 해결 아이디어: **1 positive vs. 모든 negative를 한 번에** 비교 → N-pair Loss (Sohn et al., NeurIPS 2016). 여러 negative를 동시에 쓰니 gradient에 담기는 정보가 많아지고, 가장 어려운 negative를 따로 골라야 할 필요(hard negative mining)가 줄어든다는 주장임.
- ← 슬라이드는 "NIPS 2016"이라 쓰고 p.18은 "NeurIPS 2016"으로 표기함. 같은 학회(이름만 바뀜), 표기 불일치.
- ← "No need hard negative mining"은 단순화된 주장임.
> 보충: 원 논문(Sohn 2016)도 class 수가 매우 많을 때는 **negative class mining**을 별도로 제안함. 즉 "mining이 완전히 불필요"가 아니라 "여러 negative를 동시에 쓰므로 instance-level hard negative mining 의존도가 낮아짐" 정도로 이해해야 정확함.

## N-pair Loss (p.18)

슬라이드 내용:
- Multi-Class N-pair loss generalizes triplet loss to include comparison with multiple negative samples.
- Given a $(N+1)$-tuplet of training samples, $\{\mathbf{x}, \mathbf{x}^+, \mathbf{x}_1^-, \dots, \mathbf{x}_{N-1}^-\}$, including one positive and $N-1$ negative ones, N-pair loss is defined as:

$$
\mathcal{L}_{\text{N-pair}}(\mathbf{x}, \mathbf{x}^+, \{\mathbf{x}_i^-\}_{i=1}^{N-1}) = \log\Big(1 + \sum_{i=1}^{N-1} \exp\big(f(\mathbf{x})^\top f(\mathbf{x}_i^-) - f(\mathbf{x})^\top f(\mathbf{x}^+)\big)\Big)
$$

$$
= -\log \frac{\exp\big(f(\mathbf{x})^\top f(\mathbf{x}^+)\big)}{\exp\big(f(\mathbf{x})^\top f(\mathbf{x}^+)\big) + \sum_{i=1}^{N-1} \exp\big(f(\mathbf{x})^\top f(\mathbf{x}_i^-)\big)}
$$

- If we only sample one negative sample per class, it is equivalent to the softmax loss for multi-class classification.
- 출처: Sohn et al., NeurIPS 2016, Improved Deep Metric Learning with Multi-class N-pair Loss Objective (https://lilianweng.github.io/)

그림 없음 (수식 슬라이드, Lilian Weng 블로그 캡처).

**설명**
- **기호**: $f(\cdot)$는 embedding network, $f(\mathbf{x})^\top f(\cdot)$는 내적(similarity score). $\mathbf{x}$ = anchor, $\mathbf{x}^+$ = positive, $\mathbf{x}_i^-$ = negative. tuple 크기 $N+1$ = anchor 1 + positive 1 + negative $N-1$ (분류 후보는 positive+negatives = $N$개).
- **첫째 줄 → 둘째 줄 유도**: positive score를 $b$, negative scores를 $a_i$라 하면 $\log(1+\sum_i e^{a_i-b}) = \log\frac{e^{b}+\sum_i e^{a_i}}{e^{b}} = -\log\frac{e^{b}}{e^{b}+\sum_i e^{a_i}}$. 즉 **log(1+Σexp(neg−pos)) 형태는 softmax cross-entropy와 동일**함.
- **의미**: positive score가 모든 negative score보다 커질수록 합 항이 0으로 가서 loss → 0. triplet loss의 margin 비교(neg 하나 vs pos 하나)를 **N-1개 negative 전체에 대한 smooth한 동시 비교**로 일반화한 것.
- **Softmax와의 관계**: class마다 negative를 하나씩 뽑으면 후보 N개 중 정답 class를 맞추는 **N-way softmax classification**과 같음.
- **Lifted Structured와의 차이**: pair별 hinge/max 대신 하나의 anchor에 대한 **softmax 분모에 모든 negative를 합산** → score가 큰(어려운) negative가 exp 때문에 자동으로 더 큰 가중을 받음 (soft hard-mining 효과).
> 보충: 원 논문에서는 batch를 N개 서로 다른 class에서 pair $(\mathbf{x}_i, \mathbf{x}_i^+)$ 하나씩 뽑아 구성하고, 다른 pair의 positive들을 anchor의 negative로 재사용함. forward $2N$번으로 $N$개의 (N+1)-tuple을 얻어 효율적임 (naive하게는 $N(N+1)$번 필요). 원 논문은 embedding norm 폭주를 막기 위해 L2 regularization도 추가함.

## N-pair Loss (p.19)

슬라이드 내용:
- 수식 (기대값 형태):

$$
L = -\mathbb{E}_X\left[\log \frac{\exp\big(s(f(x), f(x^+))\big)}{\exp\big(s(f(x), f(x^+))\big) + \sum_{j=1}^{N-1}\exp\big(s(f(x), f(x_j^-))\big)}\right]
$$

- **초록색 밑줄**: 분자 $\exp(s(f(x),f(x^+)))$ 와 분모의 첫째 항(positive 항). **빨간색 밑줄**: 분모의 $\sum_{j=1}^{N-1}\exp(s(f(x),f(x_j^-)))$ (negative 항들).
- 그림 (수식 아래): 왼쪽 쌍 = 고양이 사진 $x$와 그것을 회전/크롭한 고양이 $x^+$ (초록 밑줄 아래, positive pair). 오른쪽 = 같은 고양이 $x$와 negative들: $x_1^-$ 강아지, $x_2^-$ 배(컨테이너선), $x_3^-$ 다른 고양이 사진, 그리고 "..." (빨간 밑줄 아래).

**설명**
- p.18의 내적 $f(x)^\top f(x')$를 일반적인 **similarity function $s(\cdot,\cdot)$** 로 추상화한 표기 (cosine similarity나 내적 등). $\mathbb{E}_X$는 anchor $x$를 data 분포에서 뽑아 평균낸다는 뜻.
- **초록 = positive pair 점수 / 빨강 = N-1개 negative pair 점수**의 구분이 핵심. loss를 줄이려면 분자(positive)는 키우고 분모의 빨간 항(negative)은 줄여야 함 → **positive는 가깝게, negative는 멀게**.
- 그림에서 positive $x^+$는 **같은 이미지의 augmentation**(회전·crop)임 — 이후 SimCLR 등 self-supervised contrastive의 positive 생성 방식과 연결되는 예시. negative에는 다른 class(강아지, 배)뿐 아니라 **같은 class인 다른 고양이($x_3^-$)** 도 섞여 있음 → 라벨 없이 뽑으면 false negative 가능성이 있음을 암시함.
> 보충: false negative(같은 의미인데 negative로 취급되는 샘플)는 이후 contrastive learning의 대표적 문제로, supervised contrastive나 debiased contrastive loss에서 다뤄짐.

## N-pair Loss (p.20)

슬라이드 내용:
- p.19와 동일한 수식 (이미지 그림은 제거):

$$
L = -\mathbb{E}_X\left[\log \frac{\exp\big(s(f(x), f(x^+))\big)}{\exp\big(s(f(x), f(x^+))\big) + \sum_{j=1}^{N-1}\exp\big(s(f(x), f(x_j^-))\big)}\right]
$$

- 초록 라벨: "score for the positive pair" / 빨강 라벨: "score for the N-1 negative pairs"
- 텍스트:
  - This seems familiar …
  - Cross entropy loss for a N-way softmax classifier!
  - I.e., learn to find the positive sample from the N samples

**설명**
- 이 loss는 후보 $N$개(positive 1 + negative $N-1$)에 대해 $s(f(x),\cdot)$를 **logit**으로 보고 softmax를 취한 뒤, **정답 class = positive**인 **categorical cross-entropy**와 정확히 같음. 즉 $p(\text{positive}\mid x)=\frac{e^{s^+}}{e^{s^+}+\sum_j e^{s_j^-}}$ 에 $-\log$.
- 해석 전환이 중요함: metric learning을 "거리 크기 맞추기"가 아니라 "**N개 중에서 positive를 찾는 분류 문제**"로 재정의. 그래서 label 없는 self-supervised 설정(positive = augmentation)에도 바로 적용 가능 → 이후 InfoNCE, SimCLR의 NT-Xent로 이어짐.
- 일반 supervised softmax와 다른 점: 분류 head의 class weight 벡터가 있는 것이 아니라, **다른 샘플의 embedding 자체가 "class prototype" 역할**을 함 (후보 수가 batch마다 동적으로 정해짐).

## Are Negative Samples Just Noise? (p.21)

슬라이드 내용:
- Can we treat negatives as noise?
- Can we interpret it probabilistically?
  - Yes.
  - Positive vs Negative → Data vs Noise
  - The idea is to run logistic regression to tell apart the target data from noise.
  - → Noise Contrastive Estimation (NCE), is a method for estimating parameters of a statistical model

그림 없음.

**설명**
- 질문의 핵심: negative를 "임의의 다른 샘플"이 아니라 **noise 분포에서 뽑은 샘플**로 보면, contrastive loss를 **확률 모델 추정** 관점에서 해석할 수 있는가? → 답은 Yes.
- 대응 관계: **positive = 진짜 data 분포 샘플 / negative = noise 분포 샘플**. 두 집단을 구분하는 **binary logistic regression**을 학습하면 data 분포의 파라미터를 간접적으로 추정할 수 있음 → **Noise Contrastive Estimation (NCE)**.
> 보충: NCE의 원래 동기는 **unnormalized model**의 partition function $Z(\theta)=\int \tilde p_\theta(x)dx$ 계산이 불가능할 때 이를 직접 계산하지 않고도 학습하려는 것임. 분류 문제로 바꾸면 $Z$를 별도 학습 파라미터로 취급하거나 생략할 수 있음.

## NCE (p.22)

슬라이드 내용:
- Noise Contrastive Estimation (NCE) is a method for estimating parameters of a statistical model.
- Let $\mathbf{x}$ be the target sample $\sim P(\mathbf{x}|C=1;\theta)=p_\theta(\mathbf{x})$ and $\tilde{\mathbf{x}}$ be the noise sample $\sim P(\tilde{\mathbf{x}}|C=0)=q(\tilde{\mathbf{x}})$. Note that the logistic regression models the logit (i.e. log-odds) and in this case we would like to model the logit of a sample $u$ from the target data distribution instead of the noise distribution:

$$
\ell_\theta(\mathbf{u}) = \log\frac{p_\theta(\mathbf{u})}{q(\mathbf{u})} = \log p_\theta(\mathbf{u}) - \log q(\mathbf{u})
$$

- After converting logits into probabilities with sigmoid $\sigma(\cdot)$, we can apply cross entropy loss:

$$
\mathcal{L}_{\text{NCE}} = -\frac{1}{N}\sum_{i=1}^{N}\Big[\log\sigma(\ell_\theta(\mathbf{x}_i)) + \log\big(1-\sigma(\ell_\theta(\tilde{\mathbf{x}}_i))\big)\Big]
$$

$$
\text{where } \sigma(\ell) = \frac{1}{1+\exp(-\ell)} = \frac{p_\theta}{p_\theta + q}
$$

- 출처: Gutmann et al., PMLR 2010, Noise-contrastive estimation: A new estimation principle for unnormalized statistical models

그림 없음.

**설명**
- **기호**: $C=1$은 "data", $C=0$은 "noise" class label. $p_\theta$는 학습하려는 모델 분포, $q$는 알고 있고 샘플링 가능한 noise 분포. $\ell_\theta(u)$는 "$u$가 data에서 왔을 log-odds".
- **logit이 곧 log ratio**: data와 noise가 1:1로 섞여 있다고 가정하면 Bayes rule에 의해 $P(C=1|u)=\frac{p_\theta(u)}{p_\theta(u)+q(u)}$. 이것이 슬라이드의 $\sigma(\ell)=\frac{p_\theta}{p_\theta+q}$이며, $\sigma(\log\frac{p}{q})=\frac{p/q}{1+p/q}=\frac{p}{p+q}$ 로 확인됨.
- **Loss**: 진짜 data $\mathbf{x}_i$는 $\sigma$가 1에 가깝도록($\log\sigma$ 항), noise $\tilde{\mathbf{x}}_i$는 $\sigma$가 0에 가깝도록($\log(1-\sigma)$ 항) → 표준 **binary cross-entropy**. 최소화하면 $p_\theta$가 data 분포에 접근함.
- **N-pair/InfoNCE와의 차이**: NCE는 샘플 하나하나를 독립적으로 **data vs noise 이진 분류**(sigmoid, 2-class)하는 반면, InfoNCE는 한 번에 후보 N개 중 positive를 고르는 **N-way 분류**(softmax)임.
- ← 슬라이드는 data와 noise를 같은 index $i$로 합산(1:1 비율)하므로, noise 비율이 1이 아니면 $\sigma$ 식에 비율 $k$ 보정이 필요함 (슬라이드는 생략).
> 보충: Gutmann & Hyvärinen (2010)은 이 objective의 최적점이 정규화 상수까지 포함해 올바른 $p_\theta$가 됨을 보였음. 확장판(JMLR 2012)이 있으며, word2vec의 negative sampling은 NCE의 단순화 버전임.

## InfoNCE (p.23)

슬라이드 내용:
- The InfoNCE loss uses **categorical cross-entropy loss** to identify the positive sample amongst a set of unrelated noise samples.
- Given a context vector $\mathbf{c}$, the positive sample should be drawn from the conditional distribution $p(\mathbf{x}|\mathbf{c})$, while $N-1$ negative samples are drawn from the proposal distribution $p(\mathbf{x})$, independent from the context $\mathbf{c}$. For brevity, let us label all the samples as $X=\{\mathbf{x}_i\}_{i=1}^{N}$ among which only one of them $\mathbf{x}_{\text{pos}}$ is a positive sample. The probability of we detecting the positive sample correctly is:

$$
P(C=\text{pos}\mid X,\mathbf{c}) = \frac{P(X\mid C=\text{pos},\mathbf{c})\,P(C=\text{pos})}{\sum_{k=1}^{N}P(X\mid C=k,\mathbf{c})\,P(C=k)}
$$

- 식 위 주석: "Bayes Rule" (곡선 화살표), 분자의 $P(X|C=\text{pos},\mathbf{c})$에 "Likelihood", $P(C=\text{pos})$에 "Prior", 분모 합 아래에 "Marginal".
- 오른쪽 상자 ("Likelihood" 라벨):

$$
P(X\mid C=\text{pos},\mathbf{c}) = p(\mathbf{x}_{\text{pos}}\mid\mathbf{c})\prod_{i\neq \text{pos}}p(\mathbf{x}_i)
$$

- 출처: Oord et al., 2018, Representation Learning with Contrastive Predictive Coding

**설명**
- **설정**: context $\mathbf{c}$가 주어졌을 때, positive는 **조건부 분포 $p(\mathbf{x}|\mathbf{c})$** 에서, negative $N-1$개는 $\mathbf{c}$와 **무관한 marginal(proposal) 분포 $p(\mathbf{x})$** 에서 뽑음 (negative = noise). $N$개 중 어느 위치가 positive인지를 $C$로 두고 "그 위치를 맞출 확률"을 구하는 문제로 정식화.
- **Bayes rule**: $P(C=\text{pos}|X,\mathbf{c})$ = likelihood × prior / marginal. 분모는 모든 후보 위치 $k$가 positive였을 경우의 합.
- **Likelihood**: positive 위치는 $p(\mathbf{x}_{\text{pos}}|\mathbf{c})$, 나머지 $N-1$개는 서로 독립으로 $p(\mathbf{x}_i)$이므로 곱. 분모의 $k$번째 항도 같은 구조에서 positive 위치만 $k$로 바뀜.
- **Prior**: 슬라이드엔 값이 없으나 모든 위치가 같은 확률이라 두면 $P(C=k)=1/N$이고 분자·분모에서 약분됨 (다음 슬라이드에서 사라짐).
- 제목의 **categorical cross-entropy**: 이 확률에 $-\log$를 취하면 N-way categorical CE → p.24에서 $f$로 정리.
> 보충: Oord et al. 2018 (CPC, Contrastive Predictive Coding)에서 제안. $\mathbf{c}$는 autoregressive 모델이 요약한 과거 context, $\mathbf{x}$는 미래의 latent. 이름 "InfoNCE"는 Info(mutual information 하한) + NCE에서 옴.

## InfoNCE (p.24)

슬라이드 내용:
- p.23과 같은 도입 문단 (categorical cross-entropy, context $\mathbf{c}$, $N-1$ negatives from $p(\mathbf{x})$ 등).
- 수식 유도 (우상단 $\prod_i p(\mathbf{x}_i)$에서 분수로 이어지는 곡선 화살표 "divide": 분자·분모를 $\prod_i p(\mathbf{x}_i)$로 나눈다는 표시):

$$
p(C=\text{pos}\mid X,\mathbf{c}) = \frac{p(\mathbf{x}_{\text{pos}}|\mathbf{c})\prod_{i=1,\dots,N;\,i\neq\text{pos}}p(\mathbf{x}_i)}{\sum_{j=1}^{N}\Big[p(\mathbf{x}_j|\mathbf{c})\prod_{i=1,\dots,N;\,i\neq j}p(\mathbf{x}_i)\Big]} = \frac{\dfrac{p(\mathbf{x}_{\text{pos}}|\mathbf{c})}{p(\mathbf{x}_{\text{pos}})}}{\sum_{j=1}^{N}\dfrac{p(\mathbf{x}_j|\mathbf{c})}{p(\mathbf{x}_j)}} = \frac{f(\mathbf{x}_{\text{pos}},\mathbf{c})}{\sum_{j=1}^{N}f(\mathbf{x}_j,\mathbf{c})}
$$

- where the scoring function is $f(\mathbf{x},\mathbf{c}) \propto \frac{p(\mathbf{x}|\mathbf{c})}{p(\mathbf{x})}$.
- 출처: Oord et al., 2018, Representation Learning with Contrastive Predictive Coding

**설명**
- **유도 과정**: ① 첫째 등식은 p.23의 Bayes rule에 likelihood를 대입하고 prior를 약분한 것. ② 분자·분모를 전체 곱 $\prod_{i=1}^N p(\mathbf{x}_i)$로 나누면 각 항에서 $\mathbf{x}_j$ 하나만 $p(\mathbf{x}_j|\mathbf{c})/p(\mathbf{x}_j)$ 형태로 남음 ("divide" 화살표). ③ 이 비를 **scoring function $f$** 로 정의하면 **softmax 형태의 정규화**가 됨.
- **핵심 의미**: 최적의 $f$는 **density ratio $p(\mathbf{x}|\mathbf{c})/p(\mathbf{x})$ 에 비례**함. 이 ratio의 log는 $\mathbf{x}$와 $\mathbf{c}$의 **pointwise mutual information**이므로, InfoNCE 최소화는 context와 샘플 사이의 상호정보를 포착하는 것과 연결됨. 실제로는 density를 직접 모델링하지 않고 $f$를 neural net(예: $\exp(z^\top W c)$ 또는 $\exp(\text{sim}/\tau)$)으로 근사해 학습함.
- 결과적으로 $\mathcal{L}_{\text{InfoNCE}}=-\mathbb{E}\Big[\log\frac{f(\mathbf{x}_{\text{pos}},\mathbf{c})}{\sum_j f(\mathbf{x}_j,\mathbf{c})}\Big]$ 이고 이는 **p.20의 N-pair loss와 형태가 동일** ($\exp(s(\cdot))$가 $f$ 역할). N-pair는 metric learning에서, InfoNCE는 확률/MI 관점에서 도달한 같은 식이라는 점이 포인트.
> 보충: CPC 논문은 $I(x;c)\ge \log N - \mathcal{L}_N^{\text{InfoNCE}}$ 라는 **mutual information 하한**을 보임. negative 수 $N$이 클수록 하한이 타이트해질 수 있으나 $\log N$에서 포화되므로 큰 MI는 추정이 어려움.

## FIGURES

- p.19: 고양이 anchor $x$, 증강된 positive $x^+$, 다양한 negative($x_1^-$ 강아지, $x_2^-$ 배, $x_3^-$ 고양이) + 초록/빨강 밑줄로 positive/negative 항을 표시한 N-pair 수식 도식 (상)
- p.20: positive pair score / N-1 negative pairs score 라벨링 + "N-way softmax cross-entropy" 해석 (상)
- p.24: InfoNCE에서 $\prod p(x_i)$로 나누어 density ratio $f$를 얻는 유도 (divide 화살표) (상)
- p.23: InfoNCE의 Bayes rule 분해(Likelihood/Prior/Marginal 주석) (중)
- p.22: NCE의 logit→sigmoid→binary CE 수식 (중)
- p.18: N-pair loss 두 형태 수식 (log(1+Σexp) ↔ −log softmax) (중)

## EXAM POINTS

1. **Lifted Structured Loss의 한계 3가지**: O(N^2) dense pairwise 비교, 여전히 pair-based(1 positive ↔ 1 negative), multi-class batch 구조 미활용 → N-pair가 "1 positive ↔ many negatives"로 해결.
2. **N-pair loss 식**: $\log(1+\sum_i\exp(f^\top f_i^- - f^\top f^+)) = -\log\frac{\exp(f^\top f^+)}{\exp(f^\top f^+)+\sum_i\exp(f^\top f_i^-)}$ 두 형태가 같음을 유도할 수 있어야 함.
3. **N-pair loss = N-way softmax cross-entropy** (positive를 N개 후보에서 찾는 분류). negative를 class당 하나씩 뽑으면 multi-class softmax loss와 동치.
4. **Triplet → N-pair**: negative 1개 → N-1개로 일반화, 정보량 증가, hard negative mining 의존도 감소 (exp 가중으로 어려운 negative가 자동 강조됨을 설명).
5. **NCE의 아이디어**: positive=data, negative=noise로 보고 logistic regression(binary classification)으로 $p_\theta$ 추정. logit $\ell_\theta(u)=\log p_\theta(u)-\log q(u)$, $\sigma(\ell)=p_\theta/(p_\theta+q)$ 도출.
6. **NCE vs InfoNCE**: NCE는 sigmoid 기반 2-class(data vs noise) 이진 CE, InfoNCE는 softmax 기반 N-way categorical CE.
7. **InfoNCE 유도**: Bayes rule로 $P(C=\text{pos}|X,c)$를 세우고 $\prod p(x_i)$로 나눠 $f(x,c)\propto p(x|c)/p(x)$ 얻기 (positive는 $p(x|c)$, negative는 $p(x)$에서 샘플된다는 가정이 핵심).
8. **Why 질문**: 왜 InfoNCE의 최적 score가 density ratio에 비례하며 mutual information과 어떻게 연결되는가? (PMI = log ratio, $I\ge\log N-\mathcal{L}$)

## VERIFIER REPORT
- 수정 없음
