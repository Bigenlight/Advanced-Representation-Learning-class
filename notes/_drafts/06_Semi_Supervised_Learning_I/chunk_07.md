<!-- DRAFT | lecture: 06_Semi_Supervised_Learning_I | chunk: 07 | pages 43–54 -->
<!-- VERIFIED | fixes: 0 | verifier: sonnet -->

## Section: Can a more accurate teacher support stronger perturbations? (p.43)

슬라이드 텍스트:
- "Can a more accurate teacher support stronger perturbations?"

**설명**
- 섹션 구분 슬라이드. 앞 chunk의 Mean Teacher 계열(teacher = student의 EMA, consistency regularization)에서 이어지는 질문: **perturbation을 세게 주려면 target(teacher 예측)이 먼저 정확해야 하지 않나?**
- 이 질문의 답으로 다음 슬라이드부터 **PS-MT (Perturbed and Strict Mean Teachers)** 를 case study로 소개함.

---

## Paper: Perturbed and Strict Mean Teachers (p.44)

슬라이드 텍스트:
- 출처: Liu et al., CVPR 2022, *Perturbed and Strict Mean Teachers for Semi-supervised Semantic Segmentation*
- 저자: Yuyuan Liu, Yu Tian, Yuanhong Chen, Fengbei Liu (Australian Institute for Machine Learning, University of Adelaide), Vasileios Belagiannis (Universität Ulm), Gustavo Carneiro (Adelaide)
- 그림: CVF Open Access 논문 첫 페이지 캡처.

**설명**
- 과제는 **semi-supervised semantic segmentation** (pixel 단위 분류, label 비용이 매우 큼 → unlabeled 활용의 동기가 큼).
- 인용 정보 암기용: **PS-MT, Liu et al., CVPR 2022**.

---

## Strong Perturbations Need Reliable Targets (p.45)

슬라이드 텍스트:
- Mean Teacher uses teacher predictions as training targets for unlabeled images.
- **Strong perturbations** provide more challenging examples and can improve **generalization**.
- However, teacher predictions are **not always correct**.
- If the **teacher** is **wrong**, the student learns to reproduce the **same error** under strong perturbations.
- This can **reinforce incorrect predictions** during training.
- (강조 문장) Before applying stronger perturbations, we need more reliable teacher predictions.

**설명**
- 문제 설정의 **trade-off**: perturbation이 셀수록 student에게 어려운 문제 → generalization 이득. 그런데 target이 틀리면 **confirmation bias**(틀린 예측을 student가 강한 변형 아래에서도 그대로 재현·강화) 위험이 커짐.
- 따라서 순서가 중요: **(1) teacher 정확도부터 올리고 → (2) 그 위에서 perturbation을 강화**. PS-MT의 설계 논리 전체가 이 한 줄에서 나옴.
> 보충: "confirmation bias"는 슬라이드 용어가 아니고 pseudo-labeling 문헌의 일반 용어임. Mean Teacher의 student-teacher는 가중치가 서로 묶여 있어서 오류가 순환 강화되기 쉬움.

---

## Perturbed and Strict Mean Teachers: Overview (p.46)

슬라이드 텍스트:
- Labeled data
  - Train the student with ground-truth masks.
- Unlabeled data
  - Two teachers generate the target.
  - The student receives stronger perturbations.
  - The student matches the teacher ensemble.
- 구성요소 번호 리스트: 1. Two Mean Teachers  2. Teacher Ensemble  3. Strongly Perturbed Student  4. Strict Consistency Loss
- 그림: 프레임워크 다이어그램. 위쪽 unlabeled image $x^u$ → **Weak Aug.** → 두 teacher ($\theta_h^{t1}\to\theta_g^{t1}$, $\theta_h^{t2}\to\theta_g^{t2}$; $h$ = encoder, $g$ = decoder로 보임) → 출력을 $\oplus$로 합쳐 pseudo-label $\tilde y^u$ (stop gradient 표시). 같은 $x^u$는 **Strong Aug.** → student encoder $\theta_h^s$ → feature $z^s$ → student decoder $\theta_g^s$ → 예측 $\hat y^u$. $\ell_{con}(\mathcal D_U,\theta^s)$가 $\tilde y^u$와 $\hat y^u$ 사이. $z^s$에는 두 teacher decoder(해칭된 파란 박스)로부터 **T-VAT** 화살표가 연결됨. 아래쪽 labeled $x^l$ → student → $\hat y^l$, ground truth $y$와 $\ell_{sup}(\mathcal D_L,\theta^s)$.

**설명**
- **Labeled**: 일반 supervised segmentation loss $\ell_{sup}$ (student만 gradient로 업데이트).
- **Unlabeled**: weak aug 입력을 받은 **두 teacher의 ensemble**이 pseudo-target $\tilde y^u$를 만들고, **strong aug + feature perturbation(T-VAT)** 을 받은 student가 이를 맞추도록 $\ell_{con}$ 최소화. teacher 쪽은 **stop gradient**.
- 네 가지 핵심 요소와 대응되는 질문:
  1. **Two Mean Teachers** → 더 정확한 target
  2. **Teacher Ensemble** → 분산 감소
  3. **Strongly Perturbed Student** → input/feature/network 레벨 perturbation (p.49~51)
  4. **Strict Consistency Loss** → MSE 대신 confidence-weighted CE (p.48)
> 보충: 그림에서 $\theta_h$/$\theta_g$의 역할(encoder/decoder)은 슬라이드에 명시 없음. 논문에서 $f = g\circ h$로 feature extractor와 classifier head를 구분하는 표기.

---

## A More Reliable Teacher (p.47)

슬라이드 텍스트:
- Only one teacher is updated in each epoch.
- The two teachers maintain different temporal histories.
- Their ensemble reduces prediction variance.
- (강조) Each teacher averages the student over time. The method then averages the two teachers.
- 수식:
$$\hat{\mathbf y} = \sigma\big(0.5\times(f_{\theta^{t1}}(\mathbf x) + f_{\theta^{t2}}(\mathbf x))\big)$$
$$\theta^{k} = \gamma\times\theta^{k} + (1-\gamma)\times\theta^{s}$$
- 그림: p.46 다이어그램의 teacher 부분 확대.

**설명**
- 각 teacher는 **EMA(Mean Teacher) 업데이트**: $\theta^k \leftarrow \gamma\theta^k + (1-\gamma)\theta^s$, $k\in\{t1,t2\}$, $\gamma$는 EMA decay. 슬라이드에서 $\theta^s$는 student.
- 핵심 트릭: **epoch마다 둘 중 하나의 teacher만 갱신**. → 두 teacher는 student의 **서로 다른 시점 평균**(different temporal histories)이라 **서로 다른 예측 → 다양성(diversity)** 확보. 둘 다 매번 같이 갱신하면 거의 동일해져서 ensemble 이득이 없음.
- Ensemble: teacher logit을 **평균(0.5×(합))한 뒤 softmax $\sigma$** 를 적용. 서로 다른 모델의 평균 → **prediction variance 감소** → 더 믿을 만한 target. (ensemble이 variance를 줄인다는 일반 원리와 동일)
- 요약 문장 그대로: **시간 방향 평균(EMA) + 모델 방향 평균(2 teachers)**.
> 보충: logit 평균 후 softmax이므로 probability 평균과는 약간 다름. 논문에서 $\hat{\mathbf y}$ 는 이후 pseudo-label/confidence 계산의 기반이 됨.

---

## From MSE to Strict Consistency (p.48)

슬라이드 텍스트:
- MSE loss
  - Robust to noisy teacher predictions
  - Small gradients near convergence
  - Relatively weak supervision
- Conf-CE
$$\ell_{con}(\mathcal D_U,\theta^s)=\frac{1}{|\mathcal D_U||\Omega|}\sum_{\mathbf x\in\mathcal D_U}\sum_{\omega\in\Omega} c(\omega)\,\ell\big(\tilde{\mathbf y}(\omega),\,p_{\theta^s}(\mathbf x)(\omega)\big)$$
- 그림: 정렬된 per-sample(추정) loss 막대그래프. y축 0~0.04, 파랑 = MSE, 빨강 = Conf-CE. 대부분 구간에서 Conf-CE(빨강)가 MSE(파랑)보다 크고, 오른쪽 끝(어려운/큰 loss 구간)에서 약 0.035~0.04 vs 약 0.012로 차이가 큼.
- (강조) More accurate teachers allow stronger cross-entropy supervision.

**설명**
- **MSE consistency**(원래 Mean Teacher의 선택)는 확률 벡터 간 L2 거리 → **noisy teacher에 robust**하지만, 수렴 근처에서 **gradient가 작아** 학습 신호가 약함(weak supervision).
- **CE**는 target이 맞다면 훨씬 **강한 gradient**를 줌. 그러나 teacher가 틀리면 위험 → 그래서 p.47에서 teacher를 먼저 개선했고, 추가로 **confidence $c(\omega)$로 pixel별 가중**해서 틀리기 쉬운 pixel의 영향을 줄임 (= **Conf-CE, "strict" consistency**).
- 수식 읽기: $\Omega$ = pixel 집합(위치 $\omega$), $\tilde{\mathbf y}(\omega)$ = ensemble teacher의 pseudo-label, $p_{\theta^s}(\mathbf x)(\omega)$ = student 예측, $\ell$ = cross-entropy, $c(\omega)$ = teacher confidence. unlabeled set과 pixel에 대해 평균.
- 그래프 직관: 같은 예측 오차에 대해 CE가 MSE보다 **loss/gradient가 큼** → "more strict".
> 보충: 슬라이드에는 $c(\omega)$, $\ell$의 정의가 없음. 논문 기준 $c(\omega)$는 teacher ensemble 예측의 최대 확률 기반 confidence로 이해하면 됨 (시험에서는 "teacher confidence weighting"으로 설명하면 안전). 그래프 x축/y축 의미도 슬라이드에 명시 없음.

---

## Three Levels of Perturbation (p.49)

슬라이드 텍스트:
- 1. Input Perturbation
  - Weak augmentation for teachers
  - Strong augmentation for the student
  - CutMix and Zoom In/Out
- 2. Feature Perturbation
  - Teacher-guided adversarial perturbation
  - Applied to the student feature
- 3. Network Perturbation
  - Two temporal teachers
  - One gradient-updated student
  - Different model predictions
- (강조) Consistency is enforced across input, feature, and network variations.
- 그림: $x^u$에서 teacher encoders $t^{(1)}_{enc}, t^{(2)}_{enc}$ → decoders $t^{(1)}_{dec}, t^{(2)}_{dec}$ → $\oplus$ → Conf-CE Loss, student $s_{enc}\to s_{dec}$ → Conf-CE Loss로 연결. 입력 단에 "Input Perturb.", decoder 영역(파란 점선 박스)에 "Feature Perturb.", student 출력 쪽에 "Network Perturb." 라벨.

**설명**
- PS-MT는 consistency regularization의 perturbation을 **세 층위**에서 동시에 적용:

| level | 방법 | 비교 대상 |
|---|---|---|
| Input | teacher는 weak aug, student는 strong aug + CutMix, Zoom In/Out | 입력 공간의 변형 |
| Feature | **T-VAT** (teacher-guided adversarial) 로 student feature에 노이즈 | 중간 feature 변형 |
| Network | 두 temporal teacher vs gradient-updated student | 서로 다른 모델 |

- 아이디어: **perturbation을 여러 층위로 쌓아도 target이 정확하면(Conf-CE + ensemble) 버틸 수 있다** → 일반화 향상.
> 보충: Zoom In/Out은 crop 후 resize 계열 scale augmentation. 슬라이드에 자세한 설명 없음.

---

## Teacher-Guided VAT (p.50)

슬라이드 텍스트:
- Standard VAT.
  - The student finds its own adversarial direction.
  - Early student predictions can be inaccurate.
- T-VAT
  - The student generates a feature z_s.
  - Two teachers evaluate the feature.
  - Search for the direction that maximizes changes to the teacher predictions.
  - Apply the corresponding perturbation to the student feature.
  - Train the student to maintain the teacher targets.
- 수식:
$$\text{maximise}\quad d\Big(\sigma\big(0.5\times(g_{\theta_g^{t1}}(\mathbf z^s)+g_{\theta_g^{t2}}(\mathbf z^s))\big),\ \sigma\big(0.5\times(g_{\theta_g^{t1}}(\mathbf z^s+\mathbf r_{adv})+g_{\theta_g^{t2}}(\mathbf z^s+\mathbf r_{adv}))\big)\Big)$$
$$\text{subject to}\quad \|\mathbf r_{adv}\|_2 \le \epsilon$$
- 그림: mIoU 막대그래프 (y축 0.76~0.79): original ≈ 0.771, uniform ≈ 0.773, vat ≈ 0.777, t-vat ≈ 0.782 (값은 그래프에서 읽은 근사치). t-vat가 최고.
- (강조) Teachers find the perturbation. The student learns from it.

**설명**
- **VAT (Virtual Adversarial Training)**: 예측이 가장 크게 변하는 방향(adversarial direction)으로 작은 노이즈 $\mathbf r$ 를 입력(여기선 feature)에 더하고, 예측이 그대로 유지되도록 학습. 보통 **student 자신의 예측**으로 방향을 찾음 → 초기 student는 부정확해서 방향이 나쁨.
- **T-VAT**: 방향 탐색을 **teacher들이 수행**. student feature $\mathbf z^s$ 를 두 teacher decoder에 넣고, 노이즈 $\mathbf r_{adv}$ ($\|\mathbf r_{adv}\|_2\le\epsilon$) 를 더했을 때 **teacher ensemble 예측의 변화 $d(\cdot,\cdot)$ 를 최대화**하는 방향을 찾음. 그 perturbation을 **student feature에 적용**하고, student는 **원래 teacher target을 유지**하도록 학습.
- 왜 좋은가: teacher가 더 정확 → adversarial 방향이 더 의미 있음. "**Teachers find the perturbation, student learns from it.**"
- 그래프: random(uniform) noise < VAT < T-VAT → 방향성 있는 + teacher-guided perturbation이 가장 이득.
> 보충: $d(\cdot,\cdot)$ 는 두 예측 분포 간 거리(논문에서는 보통 KL 또는 유사 divergence)이며 슬라이드에 정의 없음. VAT는 power iteration으로 $\mathbf r_{adv}$ 를 근사하는 것이 일반적이며 슬라이드에는 없음. uniform = 균일 랜덤 노이즈 baseline으로 추정.

---

## CutMix After Prediction (p.51)

슬라이드 텍스트:
- CutMix before prediction
  - Artificial boundaries appear in the mixed image.
  - Teacher predictions can contain mixing artifacts.
- CutMix after prediction
  - First, predict each image, then blend the results.
  - Teacher sees natural images
- 수식:
$$\ell\big(\mu(\tilde{\mathbf y}_i,\tilde{\mathbf y}_j,\mathbf m),\ p_{\theta^s}(\mu(\mathbf x_i,\mathbf x_j,\mathbf m))\big)$$
$$\mu(\mathbf x_i,\mathbf x_j,\mathbf m)=(1-\mathbf m)\odot\mathbf x_i+\mathbf m\odot\mathbf x_j$$
- 그림: mIoU vs epochs(1/4, 2/4, 3/4, 4/4) 선그래프. CutMix after Pred. (빨강) ≈ 0.766 → 0.773 → 0.779 → 0.782, CutMix before Pred. (파랑) ≈ 0.728 → 0.738 → 0.740 → 0.752 (근사치).
- (강조) Mixing clean predictions gives a cleaner target than predicting a mixed image.

**설명**
- 기존 CutMix consistency: **섞은 이미지를 teacher에 넣어** target을 만듦 → 합성 경계(artificial boundary) 때문에 teacher가 **natural하지 않은 입력**에서 틀린 예측(mixing artifacts) → target 오염.
- PS-MT: **teacher는 원본(natural) 이미지 $\mathbf x_i,\mathbf x_j$ 를 각각 예측**($\tilde{\mathbf y}_i,\tilde{\mathbf y}_j$) → 같은 mask $\mathbf m$ 으로 **예측을 섞어** target 생성. student는 **섞인 이미지** $\mu(\mathbf x_i,\mathbf x_j,\mathbf m)$ 를 입력으로 받아 이 섞인 target을 맞춤.
- $\mu$: mask $\mathbf m$ (0/1, 영역별)로 두 입력을 합성. $\odot$ = element-wise product. 이미지와 label/prediction 양쪽에 같은 $\mu$ 를 적용하는 게 포인트.
- 그래프: after가 before보다 **모든 시점에서 높음** (최종 약 +3 mIoU 포인트 차이).
> 보충: 이것도 "teacher는 정확해야 한다"는 같은 철학: input-level perturbation은 student 쪽에만 주고 target은 깨끗한 입력에서 만듦.

---

## What Improves the Performance? (p.52)

슬라이드 텍스트:
- Conf-CE provides the largest single gain.
- T-VAT adds effective feature perturbation.
- The auxiliary teacher further improves the target.
- The components provide complementary improvements.
- (강조) Better targets make stronger consistency learning possible.
- 표 (ablation, mIoU %), MT = Mean Teacher, AT = auxiliary teacher:

| MT | conf-CE | T-VAT | AT | ResNet-50 | ResNet-101 |
|---|---|---|---|---|---|
| ✓ | | | | 71.49 | 73.50 |
| ✓ | ✓ | | | 73.79 | 76.39 |
| ✓ | ✓ | ✓ | | 74.87 | 77.36 |
| ✓ | ✓ | ✓ | ✓ | **75.70** | **78.20** |

**설명**
- 증분(ResNet-50 / ResNet-101):
  - + Conf-CE: **+2.30 / +2.89** (**가장 큰 단일 gain**, 슬라이드 bullet과 일치)
  - + T-VAT: +1.08 / +0.97
  - + AT(auxiliary teacher): +0.83 / +0.84
  - 총합: +4.21 / +4.70
- 세 요소가 **누적적으로(complementary) 기여**: strict loss → feature perturbation → 더 좋은 target(teacher ensemble).
- 해석: 학습 신호의 **강도(Conf-CE)** 가 가장 큰 영향 → 정확한 teacher 위에서 CE를 쓰는 것이 핵심 설계라는 주장과 일관.
> 보충: 증분은 표에서 계산한 값임(슬라이드에 직접 적혀 있지 않음). 표는 슬라이드 기준 1/8 label (1323) 설정으로 보이나 슬라이드에 명시 없음 (p.53의 1/8(1323) Ours 값 75.70 / 78.20 과 일치).

---

## Results and Takeaways: SOTA comparison (p.53)

슬라이드 텍스트:
- The improvement is consistent across label ratios.
- The gain is larger when fewer labels are available.
- Table 1. Comparison with SOTA on Pascal VOC 2012. All approaches based on DeepLabV3+, under different partition protocols [9]. * = re-implemented by [9]. Best in bold. (mIoU %)

| Method | R50 1/16(662) | 1/8(1323) | 1/4(2646) | 1/2(5291) | R101 1/16(662) | 1/8(1323) | 1/4(2646) | 1/2(5291) |
|---|---|---|---|---|---|---|---|---|
| MT* | 66.77 | 70.78 | 73.22 | 75.41 | 70.59 | 73.20 | 76.62 | 77.61 |
| French et al.* | 68.90 | 70.70 | 72.46 | 74.49 | 72.56 | 72.69 | 74.25 | 75.89 |
| CCT* | 65.22 | 70.87 | 73.43 | 74.75 | 67.94 | 73.00 | 76.17 | 77.56 |
| GCT* | 64.05 | 70.47 | 73.45 | 75.20 | 69.77 | 73.30 | 75.25 | 77.14 |
| ECS | - | 67.38 | 70.70 | 72.89 | - | - | - | - |
| CPS | 71.98 | 73.67 | 74.90 | 76.15 | 74.48 | 76.44 | 77.68 | 78.64 |
| CAC | 70.10 | 72.40 | 74.00 | - | 72.40 | 74.60 | 76.30 | - |
| **Ours (PS-MT)** | **72.83** | **75.70** | **76.43** | **77.88** | **75.50** | **78.20** | **78.72** | **79.76** |

**설명**
- 모든 label ratio(1/16 ~ 1/2), 두 backbone(ResNet-50/101)에서 PS-MT가 1위 → **consistent improvement**.
- **label이 적을수록 gain이 큼**: ResNet-50 기준 Mean Teacher(MT*) 대비 1/16에서 +6.06 (72.83−66.77), 1/2에서 +2.47 (77.88−75.41). 가장 강한 baseline CPS 대비로도 1/16 +0.85, 1/2 +1.73 이라 CPS 대비 차이는 일정하진 않음 — 슬라이드 주장은 주로 MT 같은 약한 baseline 대비 해석.
- 이유(해석): label이 적을수록 pseudo target의 품질이 성능을 좌우하고 → 정확한 teacher + strict loss의 효과가 커짐.
> 보충: 괄호 숫자(662, 1323, ...)는 labeled 이미지 수. Pascal VOC 2012 augmented train set 10582장 중 일부를 label로 쓰는 protocol([9] = CPS, Chen et al. CVPR 2021)이며 슬라이드에는 이 설명이 없음. 평가 지표는 mIoU로 추정(슬라이드에 명시 없음).

---

## Results and Takeaways: Qualitative (p.54)

슬라이드 텍스트:
- The teacher ensemble produces better boundaries.
- 그림 (Figure 6): Qualitative results from Pascal VOC 2012. (a) input images, (b) ground truth, (c) supervised baseline results, (d) student results and (e) results by our approach. 3행(버스 앞 사람, 개, 오토바이+사람) × 5열.

**설명**
- (c) supervised baseline은 노이즈 많고 경계가 뭉개짐, (d) student는 개선되지만 일부 잡음이 남음, (e) PS-MT 결과가 **GT(b)에 가장 가까운 깨끗한 경계**.
- 이 슬라이드의 주장: **teacher ensemble이 더 나은 boundary**를 만든다 (슬라이드 문구). 즉 정량(mIoU) 뿐 아니라 정성적으로도 target 품질 향상이 보임.
> 보충: 그림 캡션 상 (e) "our approach"는 PS-MT 최종 출력이며 teacher/student 중 어느 쪽 출력인지는 슬라이드에 명시 없음.

---

## FIGURES

- p.46: PS-MT 전체 프레임워크 다이어그램 (weak/strong aug, two teachers + ensemble, student + T-VAT, $\ell_{con}$, $\ell_{sup}$), 중요도 상
- p.47: teacher ensemble 및 EMA 수식 + 다이어그램 확대, 중요도 상
- p.48: MSE vs Conf-CE loss 크기 막대그래프 + Conf-CE 수식, 중요도 상
- p.49: Three levels of perturbation 다이어그램 (input/feature/network), 중요도 상
- p.50: T-VAT 수식 + mIoU 막대그래프 (original/uniform/vat/t-vat), 중요도 중
- p.51: CutMix after vs before prediction 곡선 + $\mu$ 수식, 중요도 중
- p.52: Ablation 표 (MT, conf-CE, T-VAT, AT), 중요도 상
- p.53: Pascal VOC SOTA 비교 표, 중요도 중
- p.54: Qualitative 결과 (Figure 6), 중요도 하

## EXAM POINTS

1. **PS-MT의 동기**: strong perturbation은 generalization에 좋지만 teacher 예측이 틀리면 student가 같은 오류를 강화 → **먼저 teacher를 정확하게, 그 위에서 perturbation 강화**.
2. **PS-MT 4요소**: Two Mean Teachers, Teacher Ensemble, Strongly Perturbed Student, Strict Consistency Loss (Conf-CE).
3. **Teacher 설계**: 각 teacher는 EMA $\theta^k=\gamma\theta^k+(1-\gamma)\theta^s$, **epoch마다 한 teacher만 갱신** → 서로 다른 temporal history → ensemble $\hat{\mathbf y}=\sigma(0.5(f_{t1}+f_{t2}))$ 로 **variance 감소**. ("시간 평균 + 모델 평균")
4. **MSE vs Conf-CE**: MSE는 noisy teacher에 robust하나 수렴 시 gradient 작고 supervision 약함 / Conf-CE는 teacher가 정확하다는 전제하에 강한 supervision. Conf-CE 수식 $\ell_{con}=\frac{1}{|\mathcal D_U||\Omega|}\sum_{\mathbf x}\sum_\omega c(\omega)\ell(\tilde{\mathbf y}(\omega),p_{\theta^s}(\mathbf x)(\omega))$ 의 각 항 설명 (pixel 평균, confidence weight).
5. **세 level perturbation**: Input (weak→teacher, strong+CutMix+Zoom→student), Feature (T-VAT), Network (두 teacher vs student).
6. **T-VAT vs VAT**: VAT는 student 자신의 adversarial 방향 (초기 부정확), T-VAT는 **teacher ensemble 예측 변화를 최대화하는 방향** $\mathbf r_{adv}$ ($\|\mathbf r_{adv}\|_2\le\epsilon$)을 찾아 student feature에 적용 후 teacher target 유지하도록 학습. 수식 maximise $d(\cdot,\cdot)$ 서술 가능해야 함.
7. **CutMix after prediction**: teacher는 natural image에서 예측 후 예측을 mask로 섞음 ($\mu(\mathbf x_i,\mathbf x_j,\mathbf m)=(1-\mathbf m)\odot\mathbf x_i+\mathbf m\odot\mathbf x_j$), student는 섞인 이미지를 입력 → artificial boundary로 인한 teacher 오류 회피. 그래프에서 after > before (약 0.78 vs 0.75 mIoU).
8. **Ablation 수치**: ResNet-50 71.49 → 73.79 (+Conf-CE, 최대 gain) → 74.87 (+T-VAT) → 75.70 (+AT); ResNet-101 73.50 → 76.39 → 77.36 → 78.20.
9. **결과 요약**: Pascal VOC 2012, DeepLabV3+, 모든 label ratio에서 SOTA (예: R50 1/8: 75.70, R101 1/8: 78.20); **label이 적을수록 gain이 큼** (R50 1/16: 72.83).
10. **한 줄 철학**: "Better targets make stronger consistency learning possible."
