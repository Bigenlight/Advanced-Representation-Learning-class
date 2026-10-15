<!-- DRAFT | lecture: 06_Semi_Supervised_Learning_I | chunk: 06 | pages 36–42 -->
<!-- VERIFIED | fixes: 2 | verifier: sonnet -->

## Interpolation Consistency Training: MixUp 아이디어와 consistency target (p.36)

슬라이드 텍스트:
- 제목: Interpolation Consistency Training
- ICT enhances the dataset
  - Following the idea of **MixUp**.
  - Add more interpolations of data points.
  - Expect the model prediction to be consistent with interpolations of the corresponding labels.
  - Uses a **Mean Teacher** to generate stable targets.
- 수식 (Two unlabeled samples are linearly interpolated):
$$\text{mixup}_\lambda(\mathbf{x}_i, \mathbf{x}_j) = \lambda \mathbf{x}_i + (1-\lambda)\mathbf{x}_j$$
- 수식 (Consistency Target):
$$p(\text{mixup}_\lambda(y \mid \mathbf{x}_i, \mathbf{x}_j)) \approx \lambda\, p(y \mid \mathbf{x}_i) + (1-\lambda)\, p(y \mid \mathbf{x}_j)$$ ← 의심: 좌변 표기가 $p(y\mid \text{mixup}_\lambda(\mathbf{x}_i,\mathbf{x}_j))$ 여야 자연스러움 (슬라이드는 mixup이 $y$ 앞에 붙어 있음, 표기 오류로 보임)
- 강조 문구: **Mixing inputs should produce mixed predictions.**
- 출처: Verma et al., Neural Networks 2022, Interpolation Consistency Training for Semi-Supervised Learning

**설명**
- **ICT**의 핵심: **unlabeled 데이터끼리 MixUp**한 입력 $\text{mixup}_\lambda(u_i, u_j)$에 대한 모델 예측이, 각 입력의 예측을 **같은 $\lambda$로 섞은 값**과 같아야 한다는 consistency 제약.
- 일반 MixUp은 label이 있어야 label을 섞을 수 있음. unlabeled에는 label이 없으니 **teacher의 예측(pseudo-target)을 섞어서** target으로 사용 → 이게 "consistency target".
- 즉 **model이 입력 사이에서 linear behavior**를 갖도록 유도하는 regularizer. 입력의 선형 보간 → 출력(확률)의 선형 보간.
- **Mean Teacher를 쓰는 이유**: target이 학습 중 요동치면 consistency가 불안정하고 confirmation bias가 생김. teacher $\theta'$는 student $\theta$의 **EMA**: $\theta' \leftarrow \alpha\theta' + (1-\alpha)\theta$ → 더 안정적이고 정확한 target.
> 보충: 원 논문에서 $\lambda \sim \text{Beta}(\alpha,\alpha)$. 학습 총 loss는 supervised loss + $w_t \cdot$ (consistency loss), $w_t$는 ramp-up 스케줄. Mixup의 원래 형태는 $\tilde x=\lambda x_i+(1-\lambda)x_j$, $\tilde y=\lambda y_i+(1-\lambda)y_j$ (Zhang et al., 2018).

---

## Interpolation Consistency Training: 구조와 loss (p.37)

슬라이드 텍스트:
- ICT enhances the dataset
  - Random MixUp samples are likely to lie near the decision boundary.
  - Consistency on these interpolated samples encourages smoother decision boundaries.
  - Aligns well with the **Low-Density Separation Assumption.**
- 그림: 왼쪽에 Labeled Sample $(x_i, y_i)\sim\mathcal{D}_L$ (숫자 2 이미지) → $f_\theta(x_i)$ → $\hat y_i$ → **Supervised loss $(\hat y_i, y_i)$**. 아래 Unlabeled Samples: $u_j \sim \mathcal{D}_{UL}$ (숫자 5), $u_k \sim \mathcal{D}_{UL}$ (숫자 1). 둘 다 $f_{\theta'}(\cdot)$ (teacher)를 거쳐 $\hat y_j, \hat y_k$. 동시에 $\text{Mix}_\lambda(u_j,u_k)$ (섞인 이미지) → $u_m$ → $f_\theta(u_m)$ (student) → $\hat y_m$. **Consistency loss $(\text{Mix}_\lambda(\hat y_j,\hat y_k), \hat y_m)$**. 최종 **Supervised Loss + $w_t \cdot$ Consistency Loss**.
- 강조 문구: Mixing inputs should produce mixed predictions.
- 수식:
$$\mathcal{L}_u^{\text{ICT}} = \mathbb{E}_{\mathbf{u}_i,\mathbf{u}_j\sim\mathcal{U}}\ \mathbb{E}_{\lambda\sim\text{Beta}(\alpha,\alpha)}\ D\Big[p_\theta\big(y \mid \text{mixup}_\lambda(\mathbf{u}_i,\mathbf{u}_j)\big),\ \text{mixup}_\lambda\big(p_{\theta'}(y\mid\mathbf{u}_i),\, p_{\theta'}(y\mid\mathbf{u}_j)\big)\Big]$$

**설명**
- 그림 읽는 법: **teacher($\theta'$)**는 섞기 전 원본 $u_j, u_k$를 보고 target을 만들고, **student($\theta$)**는 섞은 $u_m$을 보고 예측. 둘을 맞추는 게 consistency loss. gradient는 student쪽으로만 흐름 (teacher는 EMA).
- $D[\cdot,\cdot]$은 distance/divergence (원 논문은 MSE 사용, KL도 가능). 첫 인자는 student의 mixed-input 예측, 둘째 인자는 teacher의 예측을 mix한 target.
- 왜 decision boundary 근처? 서로 다른 클래스의 두 점을 섞으면 **두 class 사이 (low-density region)** 에 놓일 확률이 높음. 거기서 예측이 "섞인 확률"(= 불확실, 선형 변화)이 되도록 강제 → boundary가 data cluster 사이로 **부드럽게(smooth)** 지나가고 high-density region을 가르지 않음.
- **Low-Density Separation Assumption**: decision boundary는 data density가 낮은 곳을 지나야 한다. ICT는 interpolation 지점에서 예측이 급변하지 않게 하므로 boundary를 low-density 영역으로 밀어냄.
> 보충: 같은 batch에서 $\lambda$ 샘플링 후 batch를 permute해 $u_k$를 만드는 식으로 구현. ICT는 $u_j$를 따로 뽑아 mixing하므로 teacher는 mixing 전 입력만 처리해 계산이 단순.

---

## Interpolation Consistency Training: 실험 결과 (p.38)

슬라이드 텍스트: Table 1. Error rates (%) on CIFAR-10 using CNN-13 architecture. We ran three trials for ICT. (50 000 unlabeled)

| Model | 1000 labeled | 2000 labeled | 4000 labeled |
|---|---|---|---|
| Supervised | 39.95 ± 0.75 | 31.16 ± 0.66 | 21.75 ± 0.46 |
| Supervised (Mixup) | 36.48 ± 0.15 | 26.24 ± 0.46 | 19.67 ± 0.16 |
| Supervised (Manifold Mixup) | 34.58 ± 0.37 | 25.12 ± 0.52 | 18.59 ± 0.18 |
| Π model (Laine & Aila, 2017) | 31.65 ± 1.20 | 17.57 ± 0.44 | 12.36 ± 0.31 |
| TempEns (Laine & Aila, 2017) | 23.31 ± 1.01 | 15.64 ± 0.39 | 12.16 ± 0.24 |
| MT (Tarvainen & Valpola, 2017) | 21.55 ± 1.48 | 15.73 ± 0.31 | 12.31 ± 0.28 |
| VAT (Miyato et al., 2018) | – | – | 11.36 ± NA |
| VAT+Ent (Miyato et al., 2018) | – | – | 10.55 ± NA |
| VAdD (Park et al., 2018) | – | – | 11.32 ± 0.11 |
| SNTG (Luo et al., 2018) | 18.41 ± 0.52 | 13.64 ± 0.32 | 10.93 ± 0.14 |
| MT+ Fast SWA (Athiwaratkun et al., 2019) | 15.58 ± NA | 11.02 ± NA | 9.05 ± NA |
| **ICT** | **15.48 ± 0.78** | **9.26 ± 0.09** | **7.29 ± 0.02** |

**설명**
- 모든 label 수(1000/2000/4000)에서 **ICT가 최저 error**. 특히 4000 labels에서 7.29% (MT 12.31%, VAT+Ent 10.55%, MT+Fast SWA 9.05%).
- **Supervised + Mixup**만으로도 error가 줄지만 (21.75 → 19.67), unlabeled를 활용하는 consistency 방법들이 훨씬 큼. Manifold Mixup이 Mixup보다 약간 더 좋음 (18.59).
- label이 적을수록(1000) 방법 간 격차가 큼: supervised 39.95 vs ICT 15.48.
- 비교 대상 방법들(Π, TempEns, MT, VAT, SNTG 등)은 모두 **consistency regularization 계열** (p.31~ 앞 내용). ICT는 그중 perturbation으로 **MixUp(interpolation)**을 쓰는 방법.
> 보충: 슬라이드의 "Fast SWA"는 Stochastic Weight Averaging 기반. 표의 NA는 원 논문이 std를 보고하지 않은 것.

---

## Unsupervised Data Augmentation: 개요 (p.39)

슬라이드 텍스트:
- Unsupervised Data Augmentation (UDA)
  - Applies strong data augmentations to unlabeled samples.
  - Encourages consistent predictions between the original and augmented images.
  - Emphasizes the importance of **augmentation quality** rather than adversarial perturbations.
- 수식: $p(y\mid x) \approx p(y\mid \text{Aug}(x))$
- 출처: Xie et al., NeurIPS 2020, Unsupervised Data Augmentation for Consistency Training

**설명**
- UDA는 **consistency training**에서 입력 perturbation을 **VAT 같은 adversarial noise가 아니라 강한 data augmentation**으로 바꾼 방법. $x$와 $\text{Aug}(x)$에 대한 예측 분포가 같아야 함.
- 원본 $x$에 대한 예측은 **target(teacher 역할, gradient 차단)**, augmented 입력 예측이 student.
- 핵심 주장: 성능을 좌우하는 건 "어떤 noise를 넣느냐"의 **품질**. 
> 보충: 논문의 UDA loss:
> $$\mathcal{L} = \mathbb{E}_{x\in L}\big[-\log p_\theta(y^*\mid x)\big] + \lambda\, \mathbb{E}_{x\in U}\,\mathbb{E}_{\hat x\sim q(\hat x\mid x)}\Big[ \text{KL}\big(p_{\tilde\theta}(y\mid x)\,\|\,p_\theta(y\mid\hat x)\big)\Big]$$
> 여기서 $\tilde\theta$는 gradient를 흘리지 않는 고정 복사본 (슬라이드엔 없음). 추가 기법: **TSA (Training Signal Annealing)** (labeled 데이터 과적합 방지, 이미 확신 높은 labeled 샘플의 loss를 점진적으로 제외), **confidence-based masking**, **sharpening**, 큰 batch의 unlabeled 데이터 사용. 이 슬라이드 범위에는 없음.

---

## Unsupervised Data Augmentation: 좋은 augmentation의 조건 (p.40)

슬라이드 텍스트:
- Unsupervised Data Augmentation (UDA)
  - What makes a good augmentation?
  - A good augmentation should be
    - Label-preserving
    - Diverse
    - Realistic
    - Task-relevant
- 강조 문구: **Crucial to use advanced data augmentation methods**

**설명**
- **Label-preserving**: 변환 후에도 정답 class가 같아야 consistency 목표 ($p(y|x)\approx p(y|\text{Aug}(x))$)가 성립. 
- **Diverse**: 다양한 변형이어야 모델이 넓은 입력 영역에서 smooth해짐 (정보량 큼).
- **Realistic**: 실제 데이터 분포에서 벗어나지 않는 변형 (adversarial noise는 사람 눈엔 무의미한 pixel 변화).
- **Task-relevant**: task에 의미 있는 invariance (예: 분류에는 rotation/color 변화에 invariance, 단 숫자 6/9는 rotation 주의).
- 결론: 단순 Gaussian noise 같은 것보다 **advanced augmentation (RandAugment 등)** 이 필수.

---

## Unsupervised Data Augmentation: Key observation과 결과 (p.41)

슬라이드 텍스트:
- Key Observation
  - Simply replacing adversarial perturbations with **strong semantic augmentations** substantially improves performance.
  - UDA outperforms previous consistency-based methods, including **VAT** and **ICT**.
  - Demonstrates that **augmentation quality is a key factor** in semi-supervised learning.
- 표 (error rate %, 오른쪽에 **RandAugment** 목록: Rotate, Translate, Shear, Color, Brightness, Contrast, Posterize, Solarize, Equalize, ...)

| Method | Model | # Param | CIFAR-10 (4k) | SVHN (1k) |
|---|---|---|---|---|
| Π-Model [32] | Conv-Large | 3.1M | 12.36 ± 0.31 | 4.82 ± 0.17 |
| Mean Teacher [58] | Conv-Large | 3.1M | 12.31 ± 0.28 | 3.95 ± 0.19 |
| VAT + EntMin [41] | Conv-Large | 3.1M | 10.55 ± 0.05 | 3.86 ± 0.11 |
| SNTG [35] | Conv-Large | 3.1M | 10.93 ± 0.14 | 3.86 ± 0.27 |
| ICT [60] | Conv-Large | 3.1M | 7.29 ± 0.02 | 3.89 ± 0.04 |
| Pseudo-Label [33] | WRN-28-2 | 1.5M | 16.21 ± 0.11 | 7.62 ± 0.29 |
| LGA + VAT [25] | WRN-28-2 | 1.5M | 12.06 ± 0.19 | 6.58 ± 0.36 |
| ICT [60] | WRN-28-2 | 1.5M | 7.66 ± 0.17 | 3.53 ± 0.07 |
| MixMatch [3] | WRN-28-2 | 1.5M | 6.24 ± 0.06 | 2.89 ± 0.06 |
| Mean Teacher [58] | Shake-Shake | 26M | 6.28 ± 0.15 | - |
| Fast-SWA [1] | Shake-Shake | 26M | 5.0 | - |
| MixMatch [3] | WRN | 26M | 4.95 ± 0.08 | - |
| **UDA (RandAugment)** | WRN-28-2 | 1.5M | 4.32 ± 0.08 | **2.23 ± 0.07** |
| **UDA (RandAugment)** | Shake-Shake | 26M | 3.7 | - |
| **UDA (RandAugment)** | PyramidNet | 26M | **2.7** | - |

**설명**
- **같은 모델(WRN-28-2, 1.5M)** 에서 비교: ICT 7.66 → MixMatch 6.24 → **UDA 4.32** (CIFAR-10, 4k labels). SVHN(1k)도 UDA 2.23이 최고.
- 모델이 커질수록(Shake-Shake, PyramidNet) UDA는 3.7 → 2.7까지 감소. 참고로 slide 표에서 **ICT는 Conv-Large 7.29**가 p.38 표(CNN-13 7.29)와 일치.
- 핵심 메시지: **adversarial perturbation(VAT) → strong semantic augmentation(RandAugment)** 로 바꾸기만 해도 큰 향상.
- **RandAugment**: 미리 정한 변환 pool(Rotate, Translate, Shear, Color, Brightness, Contrast, Posterize, Solarize, Equalize, ...)에서 **무작위로 몇 개를 골라 일정 강도로 적용**하는 augmentation. AutoAugment처럼 policy를 search할 필요 없이 hyperparameter가 적음 (적용 개수 $N$, 강도 $M$).
> 보충: UDA 논문에서는 이미지의 경우 RandAugment + Cutout 사용, 텍스트에서는 back-translation, TF-IDF word replacement를 사용 (슬라이드에 없음).

---

## Unsupervised Data Augmentation: Why is UDA better? (p.42)

슬라이드 텍스트:
- Why is UDA Better?
  - Uses **strong, label-preserving augmentations** (RandAugment).
  - Produces more diverse training samples.
  - Encourages consistency under realistic semantic variations instead of small adversarial perturbations.
- 강조 문구: **How we perturb the input matters more than simply adding perturbations. High-quality semantic augmentations lead to much stronger consistency regularization.**

**설명**
- VAT류 adversarial perturbation은 **작은 norm-bounded noise**라 입력 공간에서 매우 좁은 neighborhood만 규제. 반면 RandAugment는 **큰 변화(semantic 수준)** 를 주어도 label이 유지되는 변환이라 **훨씬 넓은 invariance**를 학습.
- 다양성(diversity)이 크면 unlabeled 데이터 하나에서 **여러 개의 다른 view**를 얻는 효과 → consistency signal이 풍부.
- 이는 contrastive learning에서 "augmentation이 representation 품질을 좌우"한다는 관찰과도 연결됨.
> 보충: 이후 FixMatch가 UDA의 strong augmentation 아이디어와 pseudo-label thresholding을 단순화해 결합함.

---

## FIGURES

- p.36: ICT의 mixup 수식 2줄 (입력 보간 / consistency target), 중요도 상
- p.37: ICT 구조 다이어그램 (labeled → supervised loss, unlabeled $u_j,u_k$ → teacher, mixed $u_m$ → student → consistency loss) + $\mathcal{L}_u^{\text{ICT}}$ 수식, 중요도 상
- p.38: CIFAR-10 error rate 표 (CNN-13, 1000/2000/4000 labels, ICT 비교), 중요도 중
- p.39: $p(y|x)\approx p(y|\text{Aug}(x))$ 수식, 중요도 상
- p.41: UDA vs 기존 방법 error rate 표 + RandAugment 변환 목록, 중요도 중

## EXAM POINTS

1. **ICT 정의**: unlabeled 두 샘플의 MixUp 입력 예측이 teacher 예측들의 같은 $\lambda$ 보간과 일치하도록 하는 consistency regularization. "Mixing inputs should produce mixed predictions."
2. **ICT loss 수식**: $\mathcal{L}_u^{\text{ICT}} = \mathbb{E}_{u_i,u_j}\mathbb{E}_{\lambda\sim\text{Beta}(\alpha,\alpha)} D[p_\theta(y|\text{mixup}_\lambda(u_i,u_j)),\ \text{mixup}_\lambda(p_{\theta'}(y|u_i), p_{\theta'}(y|u_j))]$ — student $\theta$ vs. Mean Teacher $\theta'$ 구분.
3. **ICT에서 Mean Teacher를 쓰는 이유**: 안정적인 target 생성 (EMA).
4. **ICT와 Low-Density Separation Assumption**: mixup 샘플은 decision boundary 근처 → 거기서 consistency 강제 → boundary가 smooth해지고 low-density 영역으로 이동.
5. **ICT vs MixUp (supervised)**: label 대신 teacher 예측으로 target을 섞음 (unlabeled에 적용 가능).
6. **ICT 결과**: CIFAR-10 (CNN-13) 4000 labels에서 error 7.29%로 MT(12.31), VAT+Ent(10.55) 등보다 낮음.
7. **UDA 정의**: $p(y|x)\approx p(y|\text{Aug}(x))$, unlabeled에 strong augmentation (RandAugment)을 적용한 consistency training. adversarial perturbation 대체.
8. **좋은 augmentation의 4조건**: Label-preserving, Diverse, Realistic, Task-relevant.
9. **UDA가 VAT/ICT보다 좋은 이유**: strong label-preserving augmentation, 다양성, realistic semantic variation. "How we perturb the input matters more than simply adding perturbations."
10. **RandAugment 변환 예**: Rotate, Translate, Shear, Color, Brightness, Contrast, Posterize, Solarize, Equalize.
11. **수치 암기**: UDA (WRN-28-2) CIFAR-10(4k) 4.32 / SVHN(1k) 2.23, PyramidNet 2.7; ICT (WRN-28-2) 7.66 / 3.53; MixMatch 6.24 / 2.89.
12. (보충) UDA loss 구성: supervised CE + $\lambda$ · KL(원본 예측 || augmented 예측), TSA, confidence masking.
