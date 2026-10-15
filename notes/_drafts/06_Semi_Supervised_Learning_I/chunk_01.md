<!-- DRAFT | lecture: 06_Semi_Supervised_Learning_I | chunk: 01 | pages 1–9 -->
<!-- VERIFIED | fixes: 0 | verifier: sonnet -->

## Title (p.1)

- Advanced Representation Learning (고급표현학습)
- Instructor: Donghyeon Cho, The Department of Computer Science, Hanyang University

**설명**
- 표지 슬라이드. Lecture 06은 **Semi-Supervised Learning (SSL) I**.

---

## Contents of Table (p.2)

- 1. Semi-Supervised Learning
  - Notations
  - Hypotheses
- 2. Consistency Regularization
  - Π-model
  - Temporal Ensembling
  - Mean Teacher
  - VAT
  - ICT
  - UDA

**설명**
- 이번 강의 구성: (1) SSL 기본(notation, hypotheses) → (2) **Consistency Regularization** 계열 방법들 (Π-model → Temporal Ensembling → Mean Teacher → VAT → ICT → UDA).
- 이 순서는 대체로 **시간 순서 + 발전 순서**: 같은 입력의 두 번 예측을 일치시키는 단순한 형태(Π-model)에서 시작해, target을 안정화(Temporal Ensembling, Mean Teacher), perturbation을 똑똑하게 선택(VAT), interpolation 활용(ICT), 강한 augmentation 활용(UDA)으로 발전함.
> 보충: 슬라이드 p.9의 Key Question에 나오는 **Pseudo Label** 계열은 이 목차에 없음. 다음 강의(SSL II)에서 다룰 가능성이 큼.

---

## Semi-Supervised Learning (section divider) (p.3)

- 섹션 구분 슬라이드: "Semi-Supervised Learning" (제목만 있음)

**설명**
- 여기서부터 SSL 본론 시작.

---

## Overcome Limited Amount of Labeled Data: Pre-training + fine-tuning (p.4)

- Pre-training + fine-tuning
  - Pre-train a powerful task-agnostic model on a large unsupervised data corpus
  - Then fine-tune it on the downstream task with a small set of labeled samples
- 그림: 왼쪽 "Large Unlabeled Data"(DB 아이콘 + 이미지/문서 아이콘) → "Pre-train (Self-supervised or Unsupervised)" 화살표 → "Pre-trained Model" 박스 → "Fine-tune" 화살표 → 오른쪽 "Small Labeled Data" (cat, dog 이미지 + label).

**설명**
- Labeled data가 부족할 때의 **첫 번째 해법**: **2-stage** 접근. (1) 라벨 없는 대규모 데이터로 **task-agnostic**한 representation을 학습 (self-supervised: contrastive learning, DeepCluster 등 앞 강의 내용), (2) 적은 라벨로 downstream task에 **fine-tune**.
- **task-agnostic** = 특정 downstream task에 묶이지 않고 범용으로 쓸 수 있는 feature라는 뜻.
- SSL과의 핵심 차이: pre-training + fine-tuning은 unlabeled data를 **pre-train 단계에서만** 쓰고 두 단계가 **분리(sequential)**되어 있음. 반면 SSL(p.7~8)은 **labeled와 unlabeled를 동시에(jointly)** 학습함.
> 보충: pre-train 데이터 분포와 downstream 분포가 다르면(domain gap) 이득이 줄어들 수 있음. 표준적인 transfer learning 설정임.

---

## Overcome Limited Amount of Labeled Data: Active learning (p.5)

- Active learning
  - Select most valuable unlabeled samples to be collected next
  - Helps us act smartly with a limited budget
- 그림: "Unlabeled Pool (large)" (회색 원들) → "Select most informative samples" (노란 점선 박스, 선택된 샘플) → "Query oracle (human labeler)" (사람 아이콘) → "Labeled Data (grows)" (cat, dog). 오른쪽 Labeled Data에서 다시 Unlabeled Pool로 돌아가는 화살표 "Add to pool and repeat".

**설명**
- **두 번째 해법**: 라벨을 "더 효율적으로 수집"함. 모델이 **가장 informative한 unlabeled sample을 골라** oracle(사람 labeler)에게 라벨을 요청 → labeled set에 추가 → 반복 (**iterative loop**).
- 핵심은 **labeling budget이 제한적**일 때 어떤 샘플을 라벨링할지 **선택(selection)**하는 문제. 즉 라벨 "양"이 아니라 라벨링 "대상"을 최적화함.
- 여전히 **사람이 라벨링해야 함** (oracle 필요) ← SSL은 사람 라벨을 추가로 요구하지 않는다는 점이 다름.
> 보충: "informative"의 기준으로 흔히 **uncertainty sampling** (예측 entropy가 큰 샘플, margin이 작은 샘플), query-by-committee, diversity-based selection 등을 씀. 슬라이드에는 구체 기준이 없음.

---

## Overcome Limited Amount of Labeled Data: Pre-training + dataset auto-generation (p.6)

- Pre-training + dataset auto-generation
  - Given a capable pre-trained model, we can utilize it to auto-generate a lot more labeled samples.
  - This has been especially popular within the language domain driven by the success of few-shot learning.
- 그림: "Pre-trained Model" → "Generate labeled data" → "Auto-generated labeled dataset (large)" (예: "The cat sits." → cat, "A dog runs." → dog, "…" → …) → "Train on augmented dataset" → "Model".

**설명**
- **세 번째 해법**: 이미 강력한 pre-trained model(예: LLM)이 있으면, 그 모델로 **labeled sample 자체를 대량 생성**하고, 이 합성 데이터로 (보통 더 작은) 모델을 학습.
- **language domain에서 특히 인기**인 이유 (슬라이드): **few-shot learning**의 성공 → 몇 개의 예시만 주면 모델이 task에 맞는 (input, label) 쌍을 만들어낼 수 있기 때문.
- 한계: 생성된 라벨의 **품질/노이즈**는 생성 모델의 성능에 의존함 (생성 모델의 bias가 그대로 전파).
> 보충: 대표 예로 LLM으로 합성 instruction data를 만드는 Self-Instruct 류, 또는 큰 모델 → 작은 모델 knowledge distillation과 유사한 흐름. 슬라이드에 구체 논문 인용은 없음.

---

## Overcome Limited Amount of Labeled Data: Semi-supervised learning (p.7)

- Semi-supervised learning
  - Learn from the labelled and unlabeled samples together. ← 슬라이드 철자는 "labelled" (영국식), 나머지 슬라이드는 "labeled"
- 그림: 왼쪽 "Labeled Data (small)" (cat, dog), 오른쪽 "Unlabeled Data (large)" (풍경/도시 이미지, 라벨 없음) → 둘 다 화살표로 하나의 "Model"에 입력.

**설명**
- **네 번째 해법 = 이번 강의 주제**. 작은 labeled set + 큰 unlabeled set을 **하나의 모델이 동시에** 학습.
- p.4(pre-train → fine-tune)와의 차이: **분리된 2단계가 아니라 한 번의 학습에서 두 종류 데이터를 모두 사용**. p.5(active learning)와의 차이: **추가 라벨링 없이** 있는 unlabeled data만 활용.
- 4가지 방법 비교 정리:

| 방법 | unlabeled data 활용 방식 | 추가 human label 필요? |
|---|---|---|
| Pre-training + fine-tuning | pre-train 단계에서 self-supervised로 사용 (2-stage) | 없음 (fine-tune용 small labeled set만) |
| Active learning | 라벨링할 후보 pool로 사용, 선택해서 사람이 라벨링 | **필요** (oracle) |
| Pre-training + auto-generation | pre-trained model이 label을 합성 | 없음 |
| **Semi-supervised learning** | labeled와 **joint**로 학습 (unsupervised loss) | 없음 |

---

## Overcome Limited Amount of Labeled Data: Semi-supervised Learning objective (p.8)

- Semi-supervised Learning
  - Learn from **both labeled and unlabeled data.**
  - Improve generalization when labeled data are scarce.
  - Most SSL methods combine **supervised** and **unsupervised** objectives.
- Learning Objective consists of supervised loss, unsupervised loss, weighting term.

$$\mathcal{L} = \mathcal{L}_s + \mu(t)\,\mathcal{L}_u$$

**설명**
- **SSL의 일반적 objective**: $\mathcal{L}_s$ (labeled data에 대한 supervised loss, 보통 cross-entropy) + $\mu(t)\,\mathcal{L}_u$ (unlabeled data에 대한 unsupervised loss).
- **$\mu(t)$ = weighting term** (슬라이드 표기에서 $t$에 의존 → **시간(epoch/step)에 따라 변하는 weight**). 즉 $\mathcal{L}_u$의 비중을 학습 진행에 따라 조절.
- 왜 $\mu(t)$가 time-dependent인가 (직관):
  - 학습 초기엔 모델이 아직 부정확해서 unlabeled data에 대한 예측/target이 **신뢰할 수 없음** → $\mathcal{L}_u$를 크게 주면 **잘못된 target을 강화(confirmation bias)**할 위험.
  - 그래서 보통 $\mu(t)$를 **0에서 시작해 점차 증가(ramp-up)**시킴.
> 보충: ramp-up 형태는 Π-model / Temporal Ensembling (Laine & Aila, 2017)의 Gaussian ramp-up $w(t)=\exp(-5(1-t/T)^2)$이 대표적. 슬라이드 p.8에는 구체 형태가 없고, 이후 슬라이드에서 나올 수 있음 (다음 chunk에서 확인).
- $\mathcal{L}_s$는 $N_l$개 labeled sample에서, $\mathcal{L}_u$는 보통 $N_u$개 unlabeled sample에서 계산됨. 보통 $N_u \gg N_l$이라 **unlabeled data의 양이 압도적**.
- **Improve generalization when labeled data are scarce**: 목표는 labeled data만 쓴 supervised baseline보다 **일반화 성능을 높이는 것**. unlabeled data가 입력 분포 $p(x)$에 대한 정보를 주어 decision boundary를 정하는 데 도움을 줌 (이 직관은 이후 "Hypotheses" 부분에서 다룰 예정).

---

## Key Question (p.9)

- How should we design the unsupervised loss?
  - Consistency Regularization
  - Pseudo Label
- 그림 (왼쪽, Consistency Regularization): "Unlabeled image $x$" → 두 갈래: "Augmentation 1 (weak/strong)" → Model $f_\theta(\cdot)$ → "Prediction 1 $p_1$" / "Augmentation 2 (strong)" → Model $f_\theta(\cdot)$ → "Prediction 2 $p_2$". 두 prediction 사이 양방향 점선 화살표 "Make them consistent!".
- 그림 (오른쪽, Pseudo Label): "Unlabeled image $x$" → Model $f_\theta(\cdot)$ → "Prediction $p$" → "High confidence? $\max(p) > \tau$" (점선 박스) → "Pseudo label $\hat{y}$ (e.g., "mountain")" → "Train with cross-entropy $CE(\hat{y}, p)$".

**설명**
- p.8의 $\mathcal{L}_u$를 **어떻게 설계하느냐**가 SSL 방법의 핵심 차이 → 두 가지 큰 흐름:
- **(1) Consistency Regularization**
  - 아이디어: 같은 unlabeled input $x$에 서로 다른 perturbation/augmentation을 적용해도 **모델의 예측은 같아야 한다** (**perturbation invariance**).
  - 형태: $\mathcal{L}_u = d\big(f_\theta(\text{Aug}_1(x)),\, f_\theta(\text{Aug}_2(x))\big)$, 즉 $p_1$과 $p_2$ 사이 distance (MSE나 KL divergence 등)를 줄임. **라벨이 필요 없음**.
  - 그림에서 Augmentation 1은 "weak/strong", Augmentation 2는 "strong"으로 표기됨 → 방법에 따라 weak-strong 또는 strong-strong 조합 (예: UDA는 weak 예측을 target으로, strong 예측을 맞추는 구조).
  - 이번 강의(SSL I)의 목차(p.2)가 바로 이 계열: Π-model, Temporal Ensembling, Mean Teacher, VAT, ICT, UDA.
- **(2) Pseudo Label**
  - 아이디어: 모델의 예측 $p$ 자체를 **가짜 라벨 $\hat{y}$로 삼아** 다시 supervised처럼 cross-entropy로 학습 (**self-training**).
  - **confidence thresholding**: $\max(p) > \tau$ 인 **확신이 높은 샘플만** pseudo label로 사용 → 틀린 라벨이 학습에 섞이는 것을 줄임. $\tau$ = threshold (hyperparameter).
  - 형태: $\mathcal{L}_u = \mathbb{1}[\max(p) > \tau]\; CE(\hat{y}, p)$, 여기서 $\hat{y} = \arg\max p$ (hard label). ← 이 수식은 슬라이드엔 없고 그림("High confidence? max(p) > τ", "CE(ŷ, p)")을 풀어 쓴 것.
- 두 방법의 공통점: **모델 자신의 예측에서 supervision을 만들어내는** 것 (label 불필요). 차이: consistency는 **두 예측 간 일치**를 강제, pseudo label은 **예측을 hard target으로 변환**해 CE로 학습.
> 보충: 두 계열을 결합한 것이 FixMatch (weak augmentation으로 pseudo label 생성 + 그 label을 strong augmentation 예측에 대해 CE로 학습, confidence threshold 사용). 슬라이드엔 없으나 p.9 두 그림을 합친 형태라 이해하는 데 도움이 됨.
> 보충: pseudo-labeling의 대표 논문은 Lee, 2013, "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks". (슬라이드에 인용 없음)

---

## FIGURES

- p.4: Pre-training + fine-tuning 파이프라인 (Large Unlabeled → Pre-train → Pre-trained Model → Fine-tune → Small Labeled), 중요도 중
- p.5: Active learning loop (Unlabeled Pool → select → oracle → Labeled Data → add to pool), 중요도 중
- p.6: Pre-training + dataset auto-generation (Pre-trained Model → auto-generated labeled dataset → train), 중요도 하
- p.7: SSL 개념도 (Labeled small + Unlabeled large → 하나의 Model), 중요도 중
- p.8: SSL objective 수식 $\mathcal{L} = \mathcal{L}_s + \mu(t)\mathcal{L}_u$ (캡처 가능, 이미지보다 LaTeX로 대체), 중요도 상
- p.9: Consistency Regularization 그림 (두 augmentation → $p_1$, $p_2$ → consistent) + Pseudo Label 그림 (confidence $\max(p)>\tau$ → $\hat{y}$ → CE), 중요도 상

## EXAM POINTS

1. **라벨 부족 문제를 푸는 4가지 접근**: (a) pre-training + fine-tuning, (b) active learning, (c) pre-training + dataset auto-generation, (d) semi-supervised learning. 각각의 한 줄 정의와 차이점을 비교할 수 있어야 함 (특히 SSL vs pre-train/fine-tune: **joint vs sequential**, SSL vs active learning: **추가 human labeling 필요 여부**).
2. **Pre-training + fine-tuning 정의**: task-agnostic 모델을 large unsupervised corpus로 pre-train → small labeled set으로 downstream fine-tune.
3. **Active learning 정의**: limited budget에서 **가장 valuable/informative한 unlabeled sample을 골라 oracle(human)에게 라벨링** → pool에 추가 → 반복.
4. **Dataset auto-generation**: capable pre-trained model로 labeled sample을 대량 생성. language domain에서 few-shot learning 성공으로 특히 인기.
5. **SSL 정의**: labeled와 unlabeled data를 **함께** 학습하여 labeled data가 scarce할 때 generalization 향상. 대부분 supervised + unsupervised objective를 결합.
6. **SSL loss 수식**: $\mathcal{L} = \mathcal{L}_s + \mu(t)\mathcal{L}_u$. 각 항의 의미($\mathcal{L}_s$: supervised, $\mathcal{L}_u$: unsupervised, $\mu(t)$: weighting term)와 **$\mu(t)$가 시간에 따라 변하는 이유** (초기 unreliable target 방지, ramp-up).
7. **Key Question: unsupervised loss 설계 방식 2가지**: **Consistency Regularization** (같은 입력의 서로 다른 augmentation → 예측을 일치) vs **Pseudo Label** (confident한 예측을 hard label로 삼아 CE, $\max(p)>\tau$). 두 그림을 보고 설명할 수 있어야 함.
8. **이번 강의의 흐름**: Consistency Regularization 계열 순서 암기 (Π-model → Temporal Ensembling → Mean Teacher → VAT → ICT → UDA).
