<!-- DRAFT | lecture: 06_Semi_Supervised_Learning_I | chunk: 04 | pages 23–29 -->
<!-- VERIFIED | fixes: 2 | verifier: sonnet -->

## Mean Teachers: Motivation & Core Idea (p.23)

슬라이드 텍스트:
- **Motivation**
  - Temporal Ensembling stores an EMA prediction **for every training sample.**
    - Memory intensive
    - Targets are updated only **once per epoch**
- **Core Idea**
  - Replace the EMA of **predictions** with an EMA of **model weights.**
  - The teacher model is the exponential moving average of the student model:

$$\theta' \leftarrow \beta\theta' + (1-\beta)\theta$$

- 출처: Tarvainen et al., 2018, Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results

**설명**
- 이전 슬라이드(Temporal Ensembling)의 한계가 출발점. Temporal Ensembling은 **샘플마다 과거 prediction의 EMA**를 target으로 저장 → (1) 데이터셋 크기 $N$ × 클래스 수만큼 **메모리** 필요, (2) 각 샘플의 target은 그 샘플이 **epoch에 한 번 방문될 때만** 갱신 → target이 **stale**(낡음), 큰 데이터셋일수록 심각.
- Mean Teacher의 핵심: **EMA를 prediction 공간이 아니라 weight 공간에서** 취함. 즉 student의 weight $\theta$를 매 step EMA로 평균 낸 $\theta'$를 teacher로 쓰고, teacher의 출력을 consistency target으로 사용.
- 수식 $\theta' \leftarrow \beta\theta' + (1-\beta)\theta$: $\theta$ = student weight (gradient descent로 업데이트), $\theta'$ = teacher weight, $\beta$ = EMA decay. $\beta$가 1에 가까울수록 teacher가 천천히 변함(과거를 오래 기억).
- EMA 전개: $\theta'_t = (1-\beta)\sum_{i=0}^{t}\beta^{i}\theta_{t-i}$ (초기값 항 생략) ← 최근 student일수록 가중치 큼, 가중치 합은 ≈1. 즉 **여러 step의 student를 앙상블한 것과 비슷한 효과** (weight-space ensemble).
- 왜 좋은가: 한 번의 update마다 teacher가 바뀌므로 **target이 매 iteration 갱신**되고, 샘플별 저장이 필요 없음.
> 보충: 슬라이드에서 $\beta$로 표기하지만 논문은 $\alpha$로 씀 (논문 $\theta'_t = \alpha\theta'_{t-1} + (1-\alpha)\theta_t$). 같은 식.
> 보충: 시간적으로 앞선 weight를 평균하는 아이디어는 Polyak averaging과 같은 계열. 여기서는 평균 모델을 **evaluation용이 아니라 학습 target 생성용**으로 쓴다는 점이 차이.

---

## Mean Teachers: Training Pipeline (p.24)

슬라이드 텍스트:
- **Training Pipeline**
  - **Student**
    - Updated by gradient descent
    - Optimized with supervised and consistency losses
  - **Teacher**
    - Receives the same input with different perturbations
    - Parameters updated only by EMA
    - No backpropagation
- **The consistency regularization loss is the distance between predictions by the student and teacher**
- 하단 수식: $\theta' \leftarrow \beta\theta' + (1-\beta)\theta$
- 그림: 왼쪽에 label(3)과 input(SVHN 숫자 이미지). 초록 화살표가 input을 **student model**(주황 layer 막대들)과 **teacher model**(파랑 layer 막대들) 양쪽으로 전달. 각 모델 옆에 noise $\eta$ (student), $\eta'$ (teacher) 표시. 두 모델 사이에 $\theta \mapsto \theta'$ "exponential moving average" 화살표. Student 출력(주황 prediction 막대 + "3")은 왼쪽 label(초록 one-hot)과 **classification cost**로 연결, 오른쪽 teacher 출력(파랑 prediction 막대 + "3")과는 **consistency cost**로 연결.

**설명**
- 한 iteration의 흐름:
  1. 같은 input $x$에 서로 다른 noise/augmentation $\eta, \eta'$ 적용.
  2. Student $f(x,\eta;\theta)$, Teacher $f(x,\eta';\theta')$ 각각 forward.
  3. **Labeled data**에만 classification cost (예: cross-entropy, label vs student prediction).
  4. **모든 data**(labeled + unlabeled)에 consistency cost (student prediction vs teacher prediction의 distance).
  5. 총 loss를 student에 대해서만 backprop → $\theta$ gradient step.
  6. $\theta' \leftarrow \beta\theta' + (1-\beta)\theta$로 teacher 갱신.
- 정리하면 student loss:
$$\mathcal{L}(\theta) = \mathbb{E}_{(x,y)\in\mathcal{D}_l}\big[\mathrm{CE}(y, f(x,\eta;\theta))\big] + \lambda\,\mathbb{E}_{x\in\mathcal{D}_l\cup\mathcal{D}_u}\big[\,d\big(f(x,\eta;\theta),\, f(x,\eta';\theta')\big)\big]$$
  (슬라이드엔 위 수식이 없음, 논문 기반 정리)
- **Teacher는 target 생성기일 뿐**: gradient가 흐르지 않음(**stop-gradient / No backpropagation**), 파라미터는 오직 EMA로만 변함. 그래서 teacher 쪽은 backward 연산·optimizer state가 필요 없음.
- 직관: student의 prediction이 noise에 대해 흔들려도, teacher는 여러 step의 평균이라 더 **안정적**이고 (보통) 더 **정확**함 → 이 teacher를 target으로 삼아 student가 noise-invariant하게 학습 = consistency regularization. Student가 좋아지면 teacher도 좋아지는 **선순환(bootstrapping)**. 한편 student와 teacher가 서로 너무 가까워져 target이 쓸모없어지는 confirmation bias 위험은 EMA의 느린 갱신이 완화함.
> 보충: 논문에서 consistency cost $J(\theta) = \mathbb{E}_{x,\eta',\eta}\big[\lVert f(x,\theta',\eta') - f(x,\theta,\eta)\rVert^2\big]$ (MSE). 슬라이드 p.28에서 MSE > KL 언급.
> 보충: 논문 그림은 label이 있을 때만 classification cost가 계산됨을 보여줌 (unlabeled 샘플은 consistency cost만).
> 보충: **Π model**은 같은 모델을 두 번 forward(서로 다른 noise)해 둘 사이 consistency를 거는 방식. Mean Teacher는 그 target 쪽을 EMA 모델로 대체한 것으로 볼 수 있음.

---

## Mean Teachers: Advantages (p.25)

슬라이드 텍스트:
- **Advantages**
  - Stable teacher predictions
  - No per-sample prediction memory
  - Teacher updated every iteration
  - Scales well to large datasets
- 강조 문장: **Learn from a stable teacher whose parameters are the EMA of the student.**

**설명**
- **Stable teacher predictions**: weight 평균은 단일 step의 SGD noise를 평활화(smoothing) → 예측 변동이 작음. (p.26 그래프에서 Π model 곡선의 spike와 대비)
- **No per-sample prediction memory**: Temporal Ensembling은 $N \times C$ 크기의 prediction 저장소가 필요했지만, Mean Teacher는 **모델 weight 사본 1개**만 추가 → 메모리 $O(\text{#params})$, 데이터셋 크기와 무관.
- **Teacher updated every iteration**: Temporal Ensembling은 target이 epoch당 1회 갱신 → feedback loop가 느림. Mean Teacher는 **매 step 갱신** → 학습 신호가 빠르게 반영.
- **Scales well to large datasets**: 위 두 가지 덕분에 샘플 수가 커져도 저장·갱신 비용이 늘지 않음 (online/large-scale에도 적용 가능).
- 정리 표 (슬라이드 내용 기반 비교):

| | Π model | Temporal Ensembling | Mean Teacher |
|---|---|---|---|
| Target | 같은 모델의 다른 noise 출력 | per-sample prediction EMA | EMA weight teacher의 출력 |
| Target 갱신 | 매 step (같은 모델) | epoch당 1회 | 매 step |
| 추가 메모리 | 없음 (forward 2회) | per-sample prediction | weight 사본 1개 |
| Target 안정성 | 낮음 | 높음 | 높음 |

  ← 이 표는 슬라이드에 없고 p.23·p.25 내용을 종합한 보충 정리.
> 보충: Π model은 target이 student 자신의 noisy prediction이라 noisy함 → p.26 그래프 회색 곡선이 흔들리는 이유.

---

## Mean Teachers: Experiments — SVHN learning curves (p.26)

슬라이드 텍스트:
- **Experiments**
  - The mean teacher (in orange) has better performance than the student model. ← 의심: 그래프 범례상 orange = "Mean teacher (student)", blue = "Mean teacher (teacher)". 즉 "teacher(파랑)가 student보다 좋다"가 맞는 문장으로 보임
  - The benefit is especially significant in the **low-label regime**.
- 그림: SVHN에서 classification error(로그 스케일, 2%~100%) vs training step(0k~100k) 곡선 2개 패널. 범례: Π model(밝은 회색), Π model (EMA)(짙은 회색), Mean teacher (student)(주황), Mean teacher (teacher)(파랑).
  - **왼쪽 패널**: 곡선이 대체로 겹치며 ~2.5–3%로 수렴 (차이 작음).
  - **오른쪽 패널**: Π model은 ~8–10% 부근에서 큰 spike와 함께 불안정, Π (EMA)는 ~7%, Mean teacher(student) ~4.5%, Mean teacher(teacher) ~4%로 수렴.
- 오른쪽 옆 주석: **Performance gap becomes much larger.**
- 캡션: Classification error on SVHN of Mean Teacher and the Π Model.

**설명**
- 두 패널은 label 수가 다른 설정. **왼쪽 = label 많음(거의 전체), 오른쪽 = label 매우 적음**으로 보임 (슬라이드에 라벨 없음).
> 보충: 논문 Figure 3 기준 SVHN 왼쪽 = 73257 labels (전체), 오른쪽 = 250 labels. p.27 표의 수치(전체 라벨 시 Π 2.50, MT 2.50 / 250 labels 시 Π 9.69, MT 4.35)와 정합.
- 읽는 법: label이 충분하면 Π model이나 Mean Teacher나 비슷하지만, **label이 적을수록 target의 질이 중요**해져 Mean Teacher의 안정적 EMA target 이득이 커짐 → gap 확대.
- **Teacher(파랑) ≤ Student(주황) error**: EMA 평균 모델이 student 자체보다 일관되게 낮은 error. 이것이 "teacher가 더 좋은 모델이라 target으로 쓸 가치가 있다"는 근거.
- Π model에도 EMA를 적용한 (Π model (EMA))가 일반 Π보다 낫지만, 여전히 Mean Teacher보다 나쁨. ← EMA를 **평가에만** 쓰는 것으론 부족, **학습 target으로 써야** 효과가 큼 (Mean Teacher의 핵심 차이).
- Π model 곡선의 spike: 노이즈가 큰 student prediction을 target으로 삼아 학습이 불안정해진 흔적.

---

## Mean Teachers: Experiments — SVHN table (p.27)

슬라이드 텍스트:
- **Experiments**
- Table 1: Error rate percentage on SVHN over 10 runs (4 runs when using all labels). We use exponential moving average weights in the evaluation of all our models. All the methods use a similar 13-layer ConvNet architecture. See Table 5 in the Appendix for results without input augmentation.

| Method | 250 labels (73257 images) | 500 labels | 1000 labels | 73257 labels |
|---|---|---|---|---|
| GAN [25] | | 18.44 ± 4.8 | 8.11 ± 1.3 | |
| Π model [13] | | 6.65 ± 0.53 | 4.82 ± 0.17 | 2.54 ± 0.04 |
| Temporal Ensembling [13] | | 5.12 ± 0.13 | 4.42 ± 0.16 | 2.74 ± 0.06 |
| VAT+EntMin [16] | | | **3.86** | |
| Supervised-only | 27.77 ± 3.18 | 16.88 ± 1.30 | 12.32 ± 0.95 | 2.75 ± 0.10 |
| Π model | 9.69 ± 0.92 | 6.83 ± 0.66 | 4.95 ± 0.26 | 2.50 ± 0.07 |
| **Mean Teacher** | **4.35 ± 0.50** | **4.18 ± 0.27** | 3.95 ± 0.19 | **2.50 ± 0.05** |

(위 세 줄 [Supervised-only / Π model / Mean Teacher]은 구분선 아래 그룹 = 저자들이 같은 설정으로 재실행한 결과, 위 4줄은 다른 논문에서 가져온 수치)

**설명**
- 모든 열에서 **총 이미지는 73257장**이고 그 중 label이 있는 개수만 250/500/1000/73257로 바뀜 → semi-supervised 설정. 나머지는 unlabeled로 사용.
- **핵심 관찰**
  - **250 labels**: supervised-only 27.77% → Π model 9.69% → **Mean Teacher 4.35%**. Π 대비 error 절반 이하. label이 적을수록 이득이 큼 (p.26 서술과 일치).
  - **500 labels**: MT 4.18% (Temporal Ensembling 5.12%, Π 6.83%보다 낮음).
  - **1000 labels**: MT 3.95%, 그러나 **VAT+EntMin 3.86%가 더 낮음(굵게)** ← MT가 항상 1등은 아님.
  - **전체 label(73257)**: MT 2.50%, Π 2.50%로 동일, supervised-only 2.75%보다도 약간 좋음 → label이 충분하면 semi-supervised 이득은 작아짐.
- 같은 backbone(13-layer ConvNet)으로 비교 → 방법 간 차이가 **학습 방식**에서 오는 것임.
- 주의: "We use exponential moving average weights in the evaluation of all our models" → Π model, supervised-only도 평가는 EMA weight로 하므로 공정 비교 (Mean Teacher의 이득 = EMA 평가 효과만이 아니라 **EMA target 학습** 효과).
- Std ($\pm$)가 Mean Teacher가 작은 편 → 학습이 안정적.
> 보충: 표 윗줄 GAN [25]은 Salimans et al.(Improved GAN), VAT+EntMin [16]은 Miyato et al. (virtual adversarial training + entropy minimization).
> 보충: 논문은 CIFAR-10(4000 labels 등)과 ImageNet(10% label)에서도 평가했으나 이 슬라이드엔 없음.

---

## Mean Teachers: Effective Perturbations, EMA decay, Consistency loss (p.28)

슬라이드 텍스트:
- **Effective Perturbations**
  - Input augmentation is essential.
    - Random flip
    - Gaussian noise
  - Dropout is applied only to the **student** model. ← 의심: p.24 그림은 teacher에도 noise $\eta'$를 줌. 논문도 student/teacher 양쪽에 noise 적용 (dropout 포함). 아래 설명 참고
  - Dropout is **not necessary** for the teacher model. ← 의심: 위와 같음
- **EMA Decay ($\beta$)**
  - Performance is sensitive to the EMA decay rate.
  - A practical strategy:
    - **Early** training: $\beta = 0.99$
    - **Later** training: $\beta = 0.999$
  - Increase the teacher stability as the student converges.
- **Consistency Loss**
  - **MSE** performs better than KL divergence.
  - MSE became the default consistency objective in many later methods.

**설명**
- **Effective perturbations**
  - consistency regularization이 작동하려면 student/teacher 입력에 **서로 다른 perturbation**이 있어야 함. 같으면 둘 출력이 거의 같아져 loss가 trivial.
  - 입력 augmentation (random flip, Gaussian noise 등)이 가장 중요: 모델이 **입력 변형에 invariant**해지도록 강제.
  - Dropout: 슬라이드는 student에만 적용하고 teacher엔 불필요하다고 서술.
> 보충: 원 논문(Tarvainen & Valpola)은 student와 teacher 모두 noise($\eta,\eta'$)를 주는 설정이고, 가장 중요한 건 입력 augmentation/noise라고 함 (ablation에서 input augmentation 제거 시 성능 크게 하락, dropout은 보조). 슬라이드의 "dropout은 student에만"은 논문 서술과 정확히 일치하지 않을 수 있으니 **시험에서는 슬라이드 표현대로 답하되**, 핵심은 "입력 augmentation 필수 / student와 teacher가 서로 다른 perturbation을 받음"이라고 이해해둘 것.
- **EMA decay $\beta$**
  - $\beta$가 작으면(예: 0.9) teacher가 student를 빨리 따라감 → 안정성·ensemble 효과 약함. $\beta$가 크면(0.999) teacher가 천천히 변해 **안정적이지만 student의 개선을 늦게 반영**.
  - 학습 초기: student가 빠르게 개선되므로 teacher가 따라가야 함 → **$\beta=0.99$** (유효 평균 창 ≈ $1/(1-\beta)=100$ step).
  - 학습 후기: student가 수렴하며 변동이 작아지고, 안정적 target이 더 유리 → **$\beta=0.999$** (창 ≈ 1000 step).
  - 즉 **"student가 수렴할수록 teacher stability를 높인다"**. 초기에 $\beta$가 너무 크면 teacher가 형편없는 초기 weight에 머물러 오히려 해로움.
- **Consistency loss**: MSE(logit/softmax 출력 간 squared error) > KL divergence. MSE는 **bounded gradient**이고 확률이 0에 가까운 클래스에 대해 KL처럼 폭발하지 않아 더 안정적임 (논문 서술: MSE가 약간 더 좋음). 이후 FixMatch 계열 이전의 많은 consistency 방법이 MSE를 기본으로 채택.
$$d\big(f_\theta(x),f_{\theta'}(x)\big)=\lVert p_\theta(x)-p_{\theta'}(x)\rVert_2^2\quad(\text{MSE}) \qquad \text{vs.}\quad \mathrm{KL}\big(p_{\theta'}\,\Vert\,p_\theta\big)$$
  (수식은 슬라이드에 없음, 정리용)
> 보충: 논문에서 consistency weight $\lambda$는 초반에 0에서 sigmoid-shaped **ramp-up**으로 올림 (초기에 teacher가 쓸모없는 target을 줄 때 classification loss가 지배하도록).

---

## Mean Teachers: Main Contributions (p.29)

슬라이드 텍스트:
- **Main Contributions**
  - Replaced EMA of predictions with EMA of model parameters.
  - Provided more stable consistency targets.
  - Eliminated per-sample prediction storage.
  - Improved scalability to large datasets.

**설명**
- 한 줄 요약: **"prediction-space EMA(Temporal Ensembling) → weight-space EMA(Mean Teacher)"**. 이 한 번의 변경으로 안정성(target noise 감소), 메모리(per-sample 저장 제거), scalability(매 step 갱신, 데이터셋 크기 무관)를 동시에 얻음.
- 이후 영향: EMA teacher 구조는 **BYOL, DINO**(self-distillation), **MoCo**(momentum encoder), 이후 semi-supervised 방법(Noisy Student, FixMatch 변형 등)에서 계속 재사용 → "momentum/EMA teacher"라는 설계 패턴의 원형.
> 보충: 앞 챕터(MoCo p.17–19의 momentum encoder)와 같은 EMA 업데이트 식 $\theta_k \leftarrow m\theta_k + (1-m)\theta_q$. 차이는 목적: MoCo는 negative key 일관성, Mean Teacher는 consistency target 안정화.

---

## FIGURES

- p.24: Mean Teacher 학습 파이프라인 다이어그램 (student/teacher, noise $\eta,\eta'$, classification cost & consistency cost, EMA 화살표), 중요도 상
- p.26: SVHN classification error 학습 곡선 (Π model, Π (EMA), Mean teacher student/teacher; 오른쪽 = 적은 label 설정에서 gap 확대), 중요도 중
- p.27: Table 1 SVHN error rate (250/500/1000/73257 labels, 방법별 비교), 중요도 상
- p.23: EMA 업데이트 수식 $\theta' \leftarrow \beta\theta' + (1-\beta)\theta$ (그림은 아니지만 필수 암기), 중요도 상

## EXAM POINTS

1. **Mean Teacher의 motivation**: Temporal Ensembling은 (a) 샘플별 EMA prediction 저장으로 **memory intensive**, (b) target이 **epoch당 1회만 갱신**. 이를 해결하려고 **prediction EMA → weight EMA**로 교체.
2. **EMA 업데이트 식**: $\theta' \leftarrow \beta\theta' + (1-\beta)\theta$ ($\theta$ = student, $\theta'$ = teacher). 각 기호 의미, $\beta$가 크면 teacher가 느리게 변함을 설명할 수 있어야 함.
3. **Student vs Teacher 역할**: Student는 gradient descent로 **supervised loss + consistency loss**를 최적화. Teacher는 **같은 input에 다른 perturbation**을 받고, **EMA로만 업데이트, backprop 없음**.
4. **Consistency regularization loss**: student와 teacher **prediction 간 distance** (MSE). classification cost는 labeled data에만, consistency cost는 labeled + unlabeled 모두에 적용. 총 loss 식을 직접 쓸 수 있게.
5. **Advantages 4가지**: stable teacher predictions / no per-sample prediction memory / teacher updated every iteration / scales well to large datasets. 각각이 Temporal Ensembling 대비 어떻게 나은지 비교.
6. **Π model vs Temporal Ensembling vs Mean Teacher 비교** (target 출처, 갱신 주기, 메모리).
7. **실험 결과 해석**: label이 적을수록(SVHN 250 labels: Π 9.69 → MT 4.35, supervised-only 27.77) Mean Teacher의 이득이 큼. 전체 label에서는 Π와 MT가 2.50으로 동일. 1000 labels에서는 VAT+EntMin(3.86)이 MT(3.95)보다 낮음. Teacher(EMA) 곡선이 student 곡선보다 error가 낮고 Π model보다 안정적.
8. **Effective perturbations**: input augmentation(random flip, Gaussian noise) 필수, student와 teacher에 서로 다른 perturbation 필요. (슬라이드: dropout은 student에만, teacher엔 불필요)
9. **EMA decay 전략**: 초반 $\beta=0.99$, 후반 $\beta=0.999$. 이유: student가 수렴할수록 teacher 안정성을 높임 (초반엔 teacher가 student를 빨리 따라가야 함). $\beta$ 민감도.
10. **MSE vs KL**: Mean Teacher는 consistency loss로 MSE가 KL divergence보다 좋다고 보고했고, 이후 많은 방법의 default가 됨.
11. **Main contributions**: weight-space EMA, 더 안정적인 consistency target, per-sample storage 제거, large dataset scalability. 인용: Tarvainen et al., 2018.
