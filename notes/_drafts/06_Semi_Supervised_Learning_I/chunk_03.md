<!-- DRAFT | lecture: 06_Semi_Supervised_Learning_I | chunk: 03 | pages 17–22 -->
<!-- VERIFIED | fixes: 2 | verifier: sonnet -->

## Π-model: 개념과 구조 (p.17)

슬라이드 텍스트:
- • Two versions of the same input with different stochastic augmentation and dropout masks pass through the network and the outputs are expected to be consistent.
- 출처: Sajjadi et al., ICLR 2016, *Regularization With Stochastic Transformations and Perturbations for Deep Semi-Supervised Learning*
- 수식: $\mathcal{L}_u^{\Pi} = \sum_{\mathbf{x}\in\mathcal{D}} \mathrm{MSE}\big(f_\theta(\mathbf{x}),\, f'_\theta(\mathbf{x})\big)$

그림: 위→아래 두 갈래 pipeline. 입력 $x_i$가 두 갈래로 나뉘어 각각 **stochastic augmentation** → **network with dropout**을 통과해 $z_i$, $\tilde z_i$ 두 출력을 냄. 라벨 $y_i$(점선, labeled일 때만)와 $z_i$가 **cross-entropy** 박스로, $z_i$와 $\tilde z_i$가 **squared difference** 박스로 들어가고, 두 loss가 $w(t)$를 곱한 **weighted sum**으로 합쳐져 최종 loss가 됨.

**설명**
- **Consistency regularization**의 가장 기본 형태. 같은 입력 $x$에 서로 다른 noise(augmentation + dropout mask)를 주고 두 번 forward → 두 출력이 같아야 한다는 제약을 걸음. 슬라이드 수식의 $f_\theta(\mathbf{x})$와 $f'_\theta(\mathbf{x})$는 **같은 network, 다른 stochastic perturbation**을 의미 (파라미터가 다른 게 아님).
- **핵심: unsupervised term은 label이 필요 없음** ← $\mathbf{x}\in\mathcal{D}$는 labeled + unlabeled 전체. 그래서 unlabeled data로도 학습 신호가 생김. Supervised CE는 labeled 샘플($i\in L$)에만 걸림.
- 직관: **smoothness / cluster assumption** — 입력에 작은 변화를 줘도 출력이 안 변해야 하니까 decision boundary가 data가 몰린 곳을 피해 low-density region으로 감.
- 두 loss는 $w(t)$로 가중 합산 (아래 p.18에서 ramp-up 함수로 정의).
- 이름 유래: 그림에서 입력이 두 갈래로 갔다 합쳐지는 모양이 대문자 **Π** 같아서 (Laine et al. 논문 표기).
> 보충: 같은 방향의 동시대 연구로 Sajjadi et al.의 **Transformation/Stability (TS) loss**가 있음. Π-model은 Laine & Aila (2017) 논문에서 이름이 붙은 것이고, 슬라이드는 Sajjadi 논문을 출처로 달아놨음.
> 보충: MSE를 쓰는 이유 — softmax 출력 간 **squared difference**는 KL보다 outlier에 덜 민감하고 gradient가 안정적. (논문에서도 squared difference 사용.)

---

## Π-model: Algorithm 1 pseudocode (p.18)

슬라이드 텍스트 (Algorithm 1: Π-model pseudocode):
- **Require:** $x_i$ = training stimuli
- **Require:** $L$ = set of training input indices with known labels
- **Require:** $y_i$ = labels for labeled inputs $i\in L$
- **Require:** $w(t)$ = unsupervised weight ramp-up function
- **Require:** $f_\theta(x)$ = stochastic neural network with trainable parameters $\theta$
- **Require:** $g(x)$ = stochastic input augmentation function
- **for** $t$ in $[1, num\_epochs]$ **do**
  - **for** each minibatch $B$ **do**
    - $z_{i\in B} \leftarrow f_\theta(g(x_{i\in B}))$  ▷ evaluate network outputs for augmented inputs
    - $\tilde z_{i\in B} \leftarrow f_\theta(g(x_{i\in B}))$  ▷ again, with different dropout and augmentation
    - $loss \leftarrow -\frac{1}{|B|}\sum_{i\in(B\cap L)} \log z_i[y_i]$  ▷ supervised loss component
    - $\qquad\;\; + \, w(t)\,\frac{1}{C|B|}\sum_{i\in B}\|z_i-\tilde z_i\|^2$  ▷ unsupervised loss component
    - update $\theta$ using, e.g., ADAM  ▷ update network parameters
  - **end for**
- **end for**
- **return** $\theta$

**설명**
- 전체 loss를 식으로 정리하면
$$
\mathcal{L} = \underbrace{-\frac{1}{|B|}\sum_{i\in B\cap L}\log z_i[y_i]}_{\text{supervised CE (labeled만)}} \;+\; w(t)\,\underbrace{\frac{1}{C\,|B|}\sum_{i\in B}\|z_i-\tilde z_i\|^2}_{\text{unsupervised consistency (labeled + unlabeled 전부)}}
$$
  여기서 $C$는 **class 수**. $\|z_i-\tilde z_i\|^2$는 $C$차원 확률벡터 간 squared L2 → $C$로 나눠 **element별 평균(MSE)**으로 만듦. 즉 $\frac{1}{C|B|}\sum\|\cdot\|^2 = $ MSE.
- **$z_i[y_i]$** = softmax 출력 벡터에서 정답 class의 확률 → $-\log z_i[y_i]$는 그냥 cross-entropy. 주의: 분모가 $|B\cap L|$이 아니라 **$|B|$**라서 minibatch에 labeled가 적으면 supervised 항이 자동으로 작아짐 ← 슬라이드 원문 그대로 (labeled 비율 만큼 scaling됨).
- **두 forward pass**: 같은 $x$에 대해 $g(\cdot)$(random augmentation)과 dropout이 매번 다르게 샘플되므로 $z_i \ne \tilde z_i$. 이 두 pass 때문에 **연산량이 supervised 대비 2배** → 다음 슬라이드(Temporal Ensembling)의 동기.
- **$w(t)$ ramp-up**: 학습 초반에는 network 출력이 쓸모없으니 consistency loss를 키우면 degenerate solution(모든 입력에 같은 출력) 쪽으로 붕괴할 위험 → $w(t)$를 0에서 시작해 점진적으로 올림.
> 보충: Laine & Aila 논문의 ramp-up은 **Gaussian ramp-up curve** $w(t)=w_{\max}\exp\!\big[-5(1-t/T)^2\big]$ ($t\le T$, 처음 80 epoch 정도), 이후 $w_{\max}$ 유지. 슬라이드에는 함수 형태가 없음. 
> 보충: 이 loss에서 $\tilde z_i$에도 gradient를 흘리는 게 원 Π-model (양쪽 모두 학습). 반면 Temporal Ensembling의 target $\tilde z$는 상수 취급.
> 보충: pseudocode에서 $z$와 $\tilde z$를 구하는 식이 글자상 동일한 건 오타가 아니라 **stochastic**이라 값이 다르다는 것을 ▷ 주석("again, with different dropout and augmentation")으로 설명한 것.

---

## Temporal Ensembling: 동기와 구조 (p.19)

슬라이드 텍스트:
- • Π-model requires two forward passes for every input.
- • → High computational cost.
- • Instead of generating two predictions every iteration,
- • → use the historical predictions of the same sample as the consistency target.
- • (exponential moving average (EMA) of the model prediction in time per training sample)
- 출처: Laine et al., ICLR 2017, *Temporal Ensembling for Semi-Supervised Learning*

그림: Temporal ensembling 다이어그램. $x_i$ → stochastic augmentation → network with dropout → $z_i$. $y_i$(점선)와 $z_i$가 cross-entropy로, $z_i$와 **$\tilde z_i$**(아래 입력 선, 과거 ensemble prediction)가 squared difference로 들어감. 두 loss는 $w(t)$로 weighted sum → loss. 또한 $z_i$는 오른쪽 출력($z_i$)으로 빠져나가 ensemble에 누적됨.

**설명**
- **Π-model 문제**: 입력마다 forward 2번 → 비용 2배.
- **아이디어**: 두 번째 branch를 따로 계산하지 말고, **같은 sample의 과거 epoch 예측들의 EMA**를 target $\tilde z_i$로 쓰자. 한 iteration당 forward는 **한 번**만.
- 직관: 과거 여러 epoch(서로 다른 dropout/augmentation, 서로 다른 weight 상태)에서의 예측을 평균한 것 = **network ensemble의 예측**과 비슷 → 단일 noisy 예측 $\tilde z$보다 **분산이 작고 더 정확한 target**. 그래서 이름이 "temporal" ensembling (시간 축에 걸친 ensemble).
- Π-model과 그림 차이: $\tilde z_i$가 network 출력이 아니라 **외부(메모리)에서 들어오는 입력**이고, $z_i$가 다시 밖으로 나가 메모리를 갱신함.
> 보충: 이 target은 "teacher 역할"을 과거 자기 자신이 하는 **self-ensembling** 구조. 다음 방법(Mean Teacher)에서는 prediction 대신 weight를 평균냄 (p.22 마지막 줄 참고).

---

## Temporal Ensembling: Algorithm 2 pseudocode (p.20)

슬라이드 텍스트 (Algorithm 2: Temporal ensembling pseudocode. Note that the updates of $Z$ and $\tilde z$ could equally well be done inside the minibatch loop; in this pseudocode they occur between epochs for clarity.):
- **Require:** $x_i$ = training stimuli
- **Require:** $L$ = set of training input indices with known labels
- **Require:** $y_i$ = labels for labeled inputs $i\in L$
- **Require:** $\alpha$ = ensembling momentum, $0\le\alpha<1$
- **Require:** $w(t)$ = unsupervised weight ramp-up function
- **Require:** $f_\theta(x)$ = stochastic neural network with trainable parameters $\theta$
- **Require:** $g(x)$ = stochastic input augmentation function
- $Z \leftarrow \mathbf{0}_{[N\times C]}$  ▷ initialize ensemble predictions
- $\tilde z \leftarrow \mathbf{0}_{[N\times C]}$  ▷ initialize target vectors
- **for** $t$ in $[1, num\_epochs]$ **do**
  - **for** each minibatch $B$ **do**
    - $z_{i\in B} \leftarrow f_\theta(g(x_{i\in B}), t)$  ▷ evaluate network outputs for augmented inputs
    - $loss \leftarrow -\frac{1}{|B|}\sum_{i\in(B\cap L)}\log z_i[y_i]$  ▷ supervised loss component
    - $\qquad\;\; + \, w(t)\,\frac{1}{C|B|}\sum_{i\in B}\|z_i-\tilde z_i\|^2$  ▷ unsupervised loss component
    - update $\theta$ using, e.g., ADAM  ▷ update network parameters
  - **end for**
  - $Z \leftarrow \alpha Z + (1-\alpha)\,z$  ▷ accumulate ensemble predictions
  - $\tilde z \leftarrow Z/(1-\alpha^t)$  ▷ construct target vectors by bias correction
- **end for**
- **return** $\theta$

**설명**
- **상태 변수**: $Z\in\mathbb{R}^{N\times C}$ = **전체 학습 데이터 $N$개 각각의 EMA 예측** (class 수 $C$), $\tilde z$ = 이번 epoch에서 consistency target으로 쓸 값. 둘 다 0으로 초기화.
- **Π-model과의 loss 차이**: forward가 **한 번**($z$만), 두 번째 항이 $\|z_i-\tilde z_i\|^2$ with $\tilde z_i$ = **저장된 target** (현재 forward 결과 아님). target은 gradient가 안 흐르는 상수.
- **EMA 업데이트** (epoch 끝에서):
$$
Z \leftarrow \alpha Z + (1-\alpha)\,z
$$
  $\alpha$는 ensembling momentum. $\alpha$가 클수록 과거 예측을 오래 기억 (긴 horizon), 작으면 최근 epoch 위주. $z$는 해당 epoch에서 모은 각 sample의 현재 예측.
- **Bias correction** 왜 필요한가: $Z_0=\mathbf{0}$으로 시작하니 EMA가 0 쪽으로 편향됨. 예측이 $t$ epoch 동안 일정한 값 $z$라고 가정하면
$$
Z_t = (1-\alpha)\sum_{k=0}^{t-1}\alpha^k z = (1-\alpha^t)\,z
$$
  → 그래서 $\tilde z = Z/(1-\alpha^t)$로 나눠주면 $\tilde z = z$로 **불편(unbiased) 추정**이 됨. 첫 epoch($t=1$)에서는 $Z=(1-\alpha)z$이므로 $\tilde z=z$ (정확히 현재 예측). $t$가 커지면 $\alpha^t\to0$이라 보정 효과는 사라짐. (Adam의 bias correction과 같은 원리.)
- **시간 순서 주의**: $\tilde z$는 **epoch 사이에서만** 갱신 → epoch $t$ 동안 쓰이는 target은 epoch $t-1$까지의 예측으로 만들어진 것. 첫 epoch에는 $\tilde z=\mathbf{0}$이지만 $w(1)\approx0$이라 문제 없음 (ramp-up과 함께 설계됨).
- 슬라이드 캡션처럼 이 갱신을 minibatch 안에서 해도 동등 (clarity 위해 epoch 간에 둔 것).
> 보충: $f_\theta(g(x), t)$에 $t$가 들어간 건 시간(epoch)에 따라 달라지는 network 설정(예: dropout/noise)을 표현한 것으로 보임. 슬라이드에 별도 설명 없음. ← 의심: Algorithm 1의 $f_\theta(g(x))$와 표기가 달라 의도가 불분명, 원문 유지.
> 보충: 논문의 $\alpha=0.6$이 기본 설정으로 알려져 있음 (슬라이드에는 없음).

---

## Temporal Ensembling: CIFAR-10 결과 (p.21)

슬라이드 텍스트 (Table 1: CIFAR-10 results with 4000 labels, averages of 10 runs (4 runs for all labels)):

| method | Error rate (%) 4000 labels | All (50000) |
|---|---|---|
| Supervised-only | 35.56 ± 1.59 | 7.33 ± 0.04 |
| &nbsp;&nbsp;with augmentation | 34.85 ± 1.65 | 6.05 ± 0.15 |
| Conv-Large, Γ-model (Rasmus et al., 2015) | 20.40 ± 0.47 | |
| CatGAN (Springenberg, 2016) | 19.58 ± 0.58 | |
| GAN of Salimans et al. (2016) | 18.63 ± 2.32 | |
| Π-model | 16.55 ± 0.29 | 6.90 ± 0.07 |
| &nbsp;&nbsp;Π-model with augmentation | 12.36 ± 0.31 | **5.56 ± 0.10** |
| &nbsp;&nbsp;Temporal ensembling with augmentation | **12.16 ± 0.24** | 5.60 ± 0.10 |

- 출처: Laine et al., ICLR 2017

**설명**
- 지표는 **error rate(%)** → 낮을수록 좋음. 4000 labels = CIFAR-10 50000장 중 **4000장만 labeled, 나머지는 unlabeled**로 쓰는 semi-supervised setting. All (50000)은 전부 labeled.
- **4000 labels**: supervised-only 35.56% → Π-model 16.55% → augmentation 추가 12.36% → temporal ensembling 12.16%. **unlabeled data + consistency 하나로 error를 약 1/3 수준으로** 줄임.
- 이전 SSL SOTA (Γ-model 20.40, CatGAN 19.58, GAN Salimans 18.63) 대비도 큰 폭 개선.
- **Augmentation 효과가 큼**: Π-model 16.55 → 12.36 (−4.2%p), supervised는 34.85로 거의 변화 없음 → consistency loss와 augmentation이 **시너지**를 냄 (augmentation이 만든 다양한 view 간 일관성이 학습 신호가 되기 때문).
- **Temporal ensembling vs Π-model (with aug)**: 4000 labels에서 12.16 vs 12.36으로 **소폭 우세**, 전체 label(50000)에서는 오히려 Π 5.56이 TE 5.60보다 근소하게 낮음 (표준편차 0.10 범위 내라 사실상 동급). 즉 TE의 장점은 **정확도보다 효율 (forward 1회)**과 target 안정성. 슬라이드 bold 표시는 4000 → TE, All → Π-model with augmentation.
- All labels에서도 consistency regularizer가 도움이 됨 (supervised+aug 6.05 → 5.56): unlabeled 없이도 **regularization 효과**.
> 보충: 논문은 이 표 외에 SVHN, CIFAR-100, noisy labels 실험에서 temporal ensembling이 label noise에 더 강건함을 보였음 (슬라이드에는 없음).

---

## Temporal Ensembling: 장단점과 다음 방법 (p.22)

슬라이드 텍스트:
- • Advantages
  - Only **one forward pass** per iteration
  - **More stable target** than Π-model
  - Lower computational cost
- • Limitation
  - The EMA target is maintained **per training sample.**
  - Requires storing predictions for every sample
  - Updated only **once per epoch**
  - Difficult to scale to large datasets
- • Next: Mean Teacher (**EMA over model parameters** instead of predictions)

**설명**
- **장점**
  - iteration당 forward 1회 → Π-model 대비 약 **절반의 연산** (두 번째 branch 불필요).
  - target이 여러 epoch 예측의 평균이라 한 번의 noisy pass보다 **안정적** (variance ↓).
- **한계** (모두 "prediction을 sample별로 저장"하는 설계에서 나옴)
  1. 데이터셋 전체에 대해 $N\times C$ 크기의 $Z$를 **저장해야 함** → 메모리가 데이터 크기에 비례, ImageNet 규모/class 수 큰 경우 부담, 스트리밍/online 데이터에는 불가.
  2. 각 sample의 target은 **epoch당 1번만** 갱신 → target이 **stale** (현재 모델보다 한 epoch 이전 정보). 큰 데이터셋이면 한 epoch이 길어서 feedback 속도가 느림.
  3. 위 이유로 **large dataset으로 scale 어려움**.
- **다음: Mean Teacher** — prediction 대신 **model parameter의 EMA**($\theta' \leftarrow \alpha\theta' + (1-\alpha)\theta$)로 teacher를 만들어, 매 step 갱신 + sample별 저장 불필요하게 해결.
> 보충: 비교 요약 — Π-model: target = 같은 step의 다른 noise 예측 (forward 2회, 저장 X) / Temporal Ensembling: target = 과거 prediction EMA (forward 1회, 저장 O, epoch당 갱신) / Mean Teacher: target = EMA weight 모델의 예측 (forward 2회지만 한쪽은 no grad, 저장 X, step마다 갱신).

---

## FIGURES

- p.17: Π-model 구조 다이어그램 (두 stochastic augmentation + dropout branch → $z_i$, $\tilde z_i$ → CE + squared difference → weighted sum with $w(t)$) + $\mathcal{L}_u^\Pi$ 수식, 중요도 상
- p.18: Algorithm 1 Π-model pseudocode (loss 식: CE 항 + $w(t)\frac{1}{C|B|}\|z-\tilde z\|^2$), 중요도 상
- p.19: Temporal ensembling 다이어그램 ($\tilde z_i$가 외부 입력, $z_i$가 출력으로 되돌아가 ensemble 갱신), 중요도 상
- p.20: Algorithm 2 Temporal ensembling pseudocode ($Z\leftarrow\alpha Z+(1-\alpha)z$, $\tilde z\leftarrow Z/(1-\alpha^t)$), 중요도 상
- p.21: Table 1 CIFAR-10 (4000 labels) error rate 비교, 중요도 중
- p.22: 장단점 정리 텍스트 슬라이드 (그림 없음), 중요도 중

## EXAM POINTS

1. **Π-model 정의**: 같은 입력에 서로 다른 stochastic augmentation + dropout mask를 적용해 두 번 forward → 두 출력의 **consistency (squared difference/MSE)**를 unlabeled 포함 전체 data에 대해 최소화. $\mathcal{L}_u^\Pi=\sum_{x\in\mathcal D}\mathrm{MSE}(f_\theta(x),f'_\theta(x))$.
2. **Π-model total loss 식**: $-\frac{1}{|B|}\sum_{i\in B\cap L}\log z_i[y_i] + w(t)\frac{1}{C|B|}\sum_{i\in B}\|z_i-\tilde z_i\|^2$. supervised는 labeled만, consistency는 모든 sample. $C$ = class 수 (MSE 평균용).
3. **$w(t)$ ramp-up의 이유**: 초반엔 예측이 부정확해 consistency를 크게 주면 degenerate/trivial solution으로 붕괴 → 0에서 시작해 점차 증가.
4. **Π-model의 단점 → Temporal Ensembling의 동기**: 입력당 forward 2회 (computational cost 2배). TE는 **같은 sample의 과거 예측 EMA**를 target으로 써서 forward 1회로 줄임.
5. **Temporal Ensembling 업데이트 식**: $Z\leftarrow\alpha Z+(1-\alpha)z$, $\tilde z\leftarrow Z/(1-\alpha^t)$. $\alpha$ = ensembling momentum ($0\le\alpha<1$), $Z,\tilde z\in\mathbb{R}^{N\times C}$.
6. **Bias correction 이유 (유도)**: $Z_0=0$이라 EMA가 0쪽으로 편향. 상수 예측 가정 시 $Z_t=(1-\alpha^t)z$ → $(1-\alpha^t)$로 나눠 보정. $t=1$이면 $\tilde z=z$.
7. **TE vs Π-model 비교**: forward 횟수(1 vs 2), target 안정성(TE가 ensemble 평균이라 더 stable), 정확도는 CIFAR-10 4000 labels에서 12.16 vs 12.36으로 비슷. TE는 target이 **epoch당 1회 갱신**(stale).
8. **TE의 한계**: sample별 EMA를 **전 데이터 크기($N\times C$)만큼 저장**, epoch당 1번 갱신, 대규모 데이터에 scale 어려움 → **Mean Teacher** (EMA over **model parameters**)로 해결.
9. **실험 포인트**: CIFAR-10 4000 labels — supervised-only 35.56% → Π-model 16.55% → +augmentation 12.36% → TE+aug 12.16%. augmentation과 consistency loss의 시너지, unlabeled data의 효과.
10. **출처 암기**: Π-model (Sajjadi et al., 2016 / Laine & Aila, ICLR 2017), Temporal Ensembling (Laine et al., ICLR 2017).
