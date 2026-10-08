<!-- DRAFT | lecture: 05_Contrastive_Learning_III | chunk: 10 | pages 73–76 -->
<!-- VERIFIED | fixes: 1 | verifier: sonnet -->

## Most Incorrect Signals Are False Positives (p.73)

- CIFAR-100
  - 99.04% of incorrect contrastive signals come from false positives.
  - 0.96% come from false negatives.
- ImageNet-1K
  - 99.91% of incorrect contrastive signals come from false positives.
  - 0.09% come from false negatives.
- Correcting false positives addresses almost all error signals.
- Negative correction provides only a small additional benefit.
- SCL-RHE focuses primarily on positive sampling.
- 하단 강조 문구: **Positive pairs are the main source of label-error propagation in SCL.**
- 그림/표 없음 (텍스트 슬라이드).

**설명**
- SupCon(SCL)은 label이 같으면 positive, 다르면 negative로 취급함. label에 오류가 있으면 잘못된 contrastive signal이 생기는데, 이 슬라이드는 그 **오류 신호의 출처를 분해**한 결과임.
- **false positive** = 실제로는 다른 클래스(또는 의미가 다른) 샘플인데 label이 같다고 되어 있어 positive로 묶인 pair → 서로 끌어당기는 잘못된 신호.
- **false negative** = 실제로는 같은 클래스인데 label이 달라 negative로 취급된 pair → 서로 밀어내는 잘못된 신호.
- 결과: CIFAR-100에서 99.04% vs 0.96%, ImageNet-1K에서 99.91% vs 0.09%로 **거의 전부가 false positive**.
- **SCL-RHE는 positive sampling(어떤 positive를 얼마나 신뢰/강조할지)에 집중**하는 설계임. negative 보정은 이득이 작아 부차적임.
> 보충: 이유는 구조적으로 설명 가능함. positive는 같은 label의 소수 샘플이고 negative는 대다수인데, 무작위 negative가 실제로 같은 클래스일 확률은 낮음. 클래스 수가 많을수록(ImageNet 1000 클래스) 이 확률이 더 낮아져 false negative 비율이 더 작아짐 (0.96% → 0.09%). 슬라이드에 명시된 설명은 아님.
> 보충: SCL-RHE는 human label error에 robust한 SupCon 변형으로 보임. 정확한 논문 인용은 슬라이드 범위(73–76)에 없음.

## Results from Training from Scratch (p.74)

- SCL-RHE consistently improves over SupCon.
- Improvement is larger for BEiT-3.
- Methods designed for heavy synthetic noise can underperform standard SupCon.
- 하단 주석: Sel-CL and TCL are designed to aggressively remove synthetic noise. However, since the actual human noise rate is relatively low, removing too many samples ends up reducing the training data.

**Table 1**: Model accuracy measured using the acc@1 metric when trained with different loss functions on 3 popular image classification benchmarks. All models here are trained from scratch using only the indicated dataset, without pre-training.

| Model | Loss | CIFAR-10 | CIFAR-100 | ImageNet-1K |
|---|---|---|---|---|
| BEiT-3 | CE [43] | 71.70 | 59.67 | 77.91 |
| BEiT-3 | SupCon [27] | 88.96 | 60.77 | 82.57 |
| BEiT-3 | Sel-CL [29] | 86.33 | 59.51 | 81.87 |
| BEiT-3 | TCL [24] | 85.16 | 59.22 | 81.74 |
| BEiT-3 | Ours | **90.16** | **64.47** | **84.21** |
| ResNet-50 | CE [43] | 95.00 | 75.30 | 78.20 |
| ResNet-50 | SupCon [27] | 96.00 | 76.50 | 78.70 |
| ResNet-50 | Sel-CL [29] | 93.10 | 74.29 | 77.85 |
| ResNet-50 | TCL [24] | 92.80 | 74.14 | 77.17 |
| ResNet-50 | Ours | **96.39** | **77.82** | **79.15** |

**설명**
- 설정: pre-training 없이 해당 데이터셋만으로 **from scratch** 학습, metric은 acc@1. "Ours" = SCL-RHE.
- **SCL-RHE > SupCon**: 모든 데이터셋·백본에서 일관되게 상승. BEiT-3: CIFAR-10 +1.20, CIFAR-100 +3.70, ImageNet +1.64. ResNet-50: +0.39, +1.32, +0.45. → **BEiT-3(ViT 계열)에서 이득이 더 큼**.
- **Sel-CL [29], TCL [24]** (noisy-label용 방법)는 오히려 SupCon보다 낮음 (예: BEiT-3 CIFAR-10 88.96 → 86.33 / 85.16). 이유(슬라이드 주석): 이들은 **synthetic noise(인위적으로 높은 비율로 주입한 noise)를 공격적으로 제거**하도록 설계됨. 실제 human labeling noise 비율은 낮아서, 과도하게 샘플을 걸러내면 **학습 데이터가 줄어** 손해임.
- CE [43]는 cross-entropy baseline.
> 보충: ViT 계열은 inductive bias가 약해 from scratch에서 data/label 품질에 더 민감하므로 label error의 영향과 보정 이득이 크다는 해석이 가능함 (슬라이드에 명시된 이유는 아님).
> 보충: Sel-CL = Selective-Supervised Contrastive Learning with noisy labels (Li et al., CVPR 2022), TCL = Twin Contrastive Learning with noisy labels (Huang et al., CVPR 2023)로 알려져 있음. SupCon = Khosla et al., NeurIPS 2020. 슬라이드에는 번호만 있음.

## Corrected Labels Reveal a Larger Gain (p.75)

좌측: **Table 2**: Accuracy of the BEiT-3 model using the metric acc@1 on different datasets and with various loss functions, when evaluating original and corrected test-set labels. (괄호 안은 Original 대비 변화, 초록=증가, 빨강=감소)

| Loss | Test set | CIFAR-10 | CIFAR-100 | ImageNet-1K |
|---|---|---|---|---|
| CE [43] | Original | 71.70 | 59.67 | 77.91 |
| CE [43] | Corrected | 71.79 (+0.09) | 59.82 (+0.15) | 78.24 (+0.33) |
| SupCon [27] | Original | 88.96 | 60.77 | 82.57 |
| SupCon [27] | Corrected | 89.11 (+0.15) | 61.49 (+0.72) | 83.74 (+1.17) |
| Sel-CL [29] | Original | 86.33 | 59.51 | 81.87 |
| Sel-CL [29] | Corrected | 86.21 (-0.12) | 58.62 (-0.89) | 81.35 (-0.52) |
| TCL [24] | Original | 85.16 | 59.22 | 81.74 |
| TCL [24] | Corrected | 84.97 (-0.19) | 58.14 (-1.08) | 81.28 (-0.46) |
| Ours | Original | 90.16 | 64.47 | 84.21 |
| Ours | Corrected | **90.41 (+0.25)** | **65.34 (+0.87)** | **86.02 (+1.81)** |

좌측 하단 bullet:
- Since the original test labels contain errors, the actual performance improvement may be underestimated.
- The extent of improvement for SCL-RHE appears greater on the corrected test set.

우측: **Table 4**: Classification accuracy after fine-tuning a pre-trained BEiT-3 with different loss functions, on several benchmarks.

| Model | Loss | CIFAR-100 | CUB-200 | Caltech-256 | Oxf-Flowers | Oxf-Pets | iNat2017 | Places365 | ImageNet-1K |
|---|---|---|---|---|---|---|---|---|---|
| ViT | CE [43] | 87.13 | 76.93 | 90.92 | 90.86 | 93.81 | 65.26 | 54.06 | 77.91 |
| BEiT-3 | CE [43] | 92.96 | 98.00 | 98.53 | 94.94 | 94.49 | 72.31 | 59.81 | 85.40 |
| BEiT-3 | SupCon [27] | 93.15 | 98.23 | 98.66 | 95.10 | 94.52 | 72.85 | 60.31 | 85.47 |
| BEiT-3 | Sel-CL [29] | 91.48 | 94.52 | 97.19 | 93.71 | 94.51 | 72.43 | 58.36 | 85.21 |
| BEiT-3 | TCL [24] | 90.92 | 93.89 | 97.26 | 93.89 | 94.68 | 72.47 | 59.22 | 85.18 |
| BEiT-3 | Ours | **93.81** | **98.95** | **99.41** | **95.89** | **96.41** | **76.25** | **62.53** | **86.51** |

- 우측 텍스트: In Table 4, SCL-RHE achieves the highest accuracy across all eight downstream datasets. Notably, compared to SupCon:
  - iNaturalist 2017: +3.4%p
  - Places365: +2.2%p
- 하단 강조 (굵게): Human labeling errors often become easy false positives. / SCL-RHE reduces their influence by emphasizing informative hard positives.
- ← Table 4의 ViT CE ImageNet-1K 값 77.91은 Table 1의 BEiT-3 from-scratch CE 값과 동일함. 원문 유지, 출처 혼동 가능성만 플래그. (슬라이드 이미지 캡처라 정확한 판독은 원본 논문 확인 권장)

**설명**
- **Table 2의 핵심**: test set의 label 자체에도 human error가 있음. 오류를 수정한 **corrected test label**로 평가하면 값이 바뀜. SupCon은 소폭 상승(+0.15/+0.72/+1.17)하지만 **Ours는 +0.25/+0.87/+1.81로 더 크게 상승**. 즉 original label 기준 성능은 실제 이득을 **과소평가(underestimate)** 하고 있음.
- 해석: noisy test label에서는 올바른 예측이 오답 처리될 수 있음. SCL-RHE는 noisy label에 덜 끌려가 올바른 예측이 더 많고, corrected test에서 이득이 드러남. 반대로 Sel-CL/TCL은 corrected에서 오히려 하락(-0.12 ~ -1.08)함.
- **Table 4 (fine-tuning)**: pre-trained BEiT-3를 각 loss로 fine-tune. SCL-RHE가 **8개 downstream 전부에서 최고**. SupCon 대비 iNaturalist 2017 +3.4%p (72.85 → 76.25), Places365 +2.2%p (60.31 → 62.53). (iNat2017·Places365에서 이득이 두드러진다는 점까지만 슬라이드에 있음. "클래스가 많고 label이 모호해서"라는 이유는 슬라이드에 없는 추측임.)
- **결론 문구**: human labeling error는 모델 입장에서 **쉬운 false positive**가 되기 쉬움. SCL-RHE는 쉬운 positive 대신 **informative hard positive**를 강조해 그 영향을 줄임.
> 보충: label error로 묶인 pair가 쉬운 positive로 나타나면 잘못된 군집을 강화하는 학습 신호가 되므로, hard positive 중심 가중이 이를 완화한다는 직관임. RHE의 정확한 가중 수식은 이 범위에 없음.

## Next Week.. (p.76)

- Semi-supervised learning ..

**설명**
- 다음 주 주제 예고: **Semi-supervised learning** (소량의 labeled + 다량의 unlabeled 데이터 활용). 내용 슬라이드 아님.

## FIGURES

- p.74: Table 1 (from scratch, BEiT-3/ResNet-50 × 5 loss × 3 dataset) — 중요도 상
- p.75: Table 2 (original vs corrected test label, 변화량 색 표시) — 중요도 상
- p.75: Table 4 (fine-tuning, 8 downstream dataset) — 중요도 중

## EXAM POINTS

- **false positive vs false negative (SCL 관점)** 정의: label error로 인한 잘못된 positive(끌어당김) vs 잘못된 negative(밀어냄).
- **왜 incorrect contrastive signal의 대부분(CIFAR-100 99.04%, ImageNet-1K 99.91%)이 false positive인가?** → positive는 소수·negative는 다수 + 클래스 수가 클수록 false negative 확률 감소. 그래서 SCL-RHE는 **positive sampling**에 집중.
- **Sel-CL, TCL이 SupCon보다 낮은 이유**: synthetic noise를 공격적으로 제거하도록 설계 → 실제 human noise 비율은 낮아 과잉 제거로 학습 데이터 감소.
- **BEiT-3에서 이득이 더 큰 점**: from scratch CIFAR-100 SupCon 60.77 → Ours 64.47 (+3.70), ResNet-50은 +1.32.
- **Corrected test label 평가의 의미**: original label의 오류 때문에 이득이 과소평가됨; Ours는 corrected에서 ImageNet-1K +1.81 (SupCon +1.17), Sel-CL/TCL은 오히려 하락.
- **Table 4**: SCL-RHE가 8개 downstream 모두 최고, SupCon 대비 iNat2017 +3.4%p, Places365 +2.2%p.
- **why 질문 요약**: human labeling error는 easy false positive가 되기 쉽고, SCL-RHE는 informative hard positive를 강조해 영향을 줄임.

## VERIFIER REPORT

- 표/수치(Table 1, 2, 4), 증감값, 인용 번호·논문 정보 모두 원본과 일치 확인.
- p.75 설명의 "클래스가 많고 label이 모호한 데이터에서 이득이 큼"은 슬라이드에 없는 과장된 이유 설명이라 추측임을 명시하도록 수정.
