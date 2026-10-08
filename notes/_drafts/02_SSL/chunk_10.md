<!-- DRAFT | lecture: 02_SSL | chunk: 10 | pages 73–80 -->
<!-- VERIFIED | fixes: 3 | verifier: sonnet -->

## Should Everything Be Invariant? (p.73)

- Modern SSL often uses image transformations:
  - Random crop
  - Color jitter
  - Horizontal flip
  - Gaussian blur
- The model is trained to ignore these transformations.
- But is invariance always desirable?

**설명**

- 앞 슬라이드(p.72)의 pretext task들(rotation, colorization, jigsaw)은 "표현이 무엇을 보존하는가"를 정하는 장치였음. 이 슬라이드는 같은 관점을 **augmentation 기반 SSL**(SimCLR 계열 등)로 옮김.
- 현대 SSL은 한 이미지에 random crop / color jitter / horizontal flip / Gaussian blur를 걸어 두 view를 만들고, 두 view의 표현이 **같아지도록** 학습함. 즉 이 transformation들에 대해 **invariant**하게 만드는 것이 목적임.
- 이 방식은 transformation이 바뀌어도 변하지 않는 **semantic 정보만 남기는 효과**가 있지만, 반대로 **transformation이 담고 있던 정보(위치, 색, 방향 등)는 표현에서 버려짐**. 슬라이드의 질문 "invariance가 항상 좋은가?"는 이 trade-off를 던지는 도입부임.

> 보충: contrastive/Siamese 계열 loss는 보통 $\mathcal{L}(f(t_1(x)), f(t_2(x)))$를 최소화하는 형태라 invariance를 직접 강제함. 어떤 augmentation을 쓰느냐가 곧 "어떤 정보를 버릴지"에 대한 사전 가정(prior)임.

## Should Everything Be Invariant? (p.74)

- Classification:
  - "What is the object?" → Invariance may be useful.
- Detection:
  - "Where is the object?" → Spatial information is necessary.
- Flower classification:
  - "What color is the flower?" → Color information is necessary.

**설명**

- **downstream task에 따라 필요한 정보가 다름**을 세 예로 보여줌.
  - **Classification**: 객체 category만 맞히면 되므로 crop/flip/위치에 둔감한 **invariance가 유리**함.
  - **Detection**: bounding box를 맞혀야 하므로 객체의 **위치(spatial information)**가 표현에 남아 있어야 함. crop/flip에 완전히 invariant한 표현은 위치 정보를 잃어 불리함.
  - **Flower classification**: 꽃 종류가 색으로 구분되는 경우가 많아 **color 정보가 필수**임. color jitter에 invariant하게 학습하면 색 정보가 지워짐.
- 결론: **고정된 invariance는 task-agnostic하지 않음**. 그래서 변환 정보를 구조적으로 보존하는 **equivariance**가 필요해짐 (다음 슬라이드).

> 보충: 이 논지는 AugSelf(Lee et al., 2021)와 EquiMod(Devillers & Lefort, 2023) 등 "augmentation-aware" SSL 연구의 공통 동기임.

## Invariance vs. Equivariance (p.75)

**그림**: 좌우 두 패널.
- 왼쪽 "Invariance": 위쪽에 파란 삼각형 이미지 → (변환 $t$ 화살표) → 회전/변형된 청록 삼각형 이미지. 각 이미지에서 아래로 $f$ 화살표가 내려가 "Feature space" 박스로 들어가며, **두 화살표가 같은 한 점(파란 점)으로 수렴**함. 수식: $f(t(x)) = f(x)$
- 오른쪽 "Equivariance": 같은 입력 쌍(파란 삼각형 → $t$ → 청록 삼각형)이 각각 $f$를 통과해 Feature space에서 **서로 다른 두 점**(진한 파랑, 청록)이 됨. 두 점 사이에 곡선 화살표 $\phi(t,\cdot)$가 그려져 있음. 수식: $f(t(x)) = \phi(t, f(x))$

- **Invariance:** representation does not change after the input transformation.
- **Equivariance:** representation changes predictably according to the input transformation.

**설명**

- **Invariance**: $f(t(x)) = f(x)$. 입력을 변환해도 표현은 **그대로**임. 변환 정보는 완전히 버려짐.
- **Equivariance**: $f(t(x)) = \phi(t, f(x))$. 입력을 변환하면 표현도 변하되, 그 변화가 $t$로부터 결정되는 **예측 가능한 map $\phi(t,\cdot)$**을 따름. $f$는 encoder, $t$는 input-space transformation, $\phi(t,\cdot)$는 **feature space에서 $t$에 대응하는 작용(action)**임.
- invariance는 $\phi(t,\cdot)=\mathrm{id}$인 **equivariance의 특수한 경우**임. equivariance는 변환 정보를 표현 안에 구조화해서 남기므로 위치/색이 필요한 task에서도 쓸 수 있음.
- 직관: 삼각형을 회전시키면 invariant 표현은 "삼각형" 그대로이고, equivariant 표현은 "회전된 삼각형"을 반영해 움직임.

> 보충: CNN의 convolution은 translation에 대해 equivariant, global pooling을 거치면 approximately invariant해지는 것이 고전적 예시임.

## Transformation Equivariant Learning (p.76)

**그림**: 왼쪽에 학습 파이프라인, 오른쪽에 개념 그림.
- 왼쪽 파이프라인: 입력 $x$와 $t(x)$가 각각 encoder $f$를 통과해 feature가 됨($x$의 feature는 갈색 막대, $t(x)$의 feature는 짙은 회색 막대). feature에서 transformation head $f_T$를 거쳐 transformation representation $y_t^x$(파란 박스)가 나옴. 하단에는 다른 이미지 $x'$와 $t(x')$(같은 $t$ 적용)가 같은 $f$, $f_T$를 거쳐 $y_t^{x'}$를 만듦. $\phi(y_t,\cdot)$ 박스(초록)는 변환 표현을 받아 $x$의 feature(갈색)에 작용해 초록 막대(예측된 변환 후 표현)를 출력함.
- 오른쪽 3개 loss 박스:
  - **Transformation Invariant Learning**: $\mathcal{L}_{\text{inv}}(x,t) = \mathcal{L}\big(f(x), f(t(x))\big)$
  - **Transformation Equivariant Learning**: $\mathcal{L}_{\text{equi}}(x,x',t) = \mathcal{L}\Big(\phi\big(y_t^{x'}, f(x)\big),\; f(t(x))\Big)$
  - **Self-supervised Transformation Learning**: $\mathcal{L}_{\text{trans}}(x,x',t) = \mathcal{L}\big(y_t^{x}, y_t^{x'}\big)$
- 맨 오른쪽 개념도: 하트 이미지 $x$ → $t$ → 변형된 하트, 웃는 얼굴 $x'$ → $t$ → 변형된 얼굴. 두 이미지 쌍이 feature space $Y$와 transformation 표현 공간 $Y_T$로 투영되고, $f_T$를 통해 **서로 다른 이미지에 같은 $t$를 걸면 $Y_T$에서 가까운 점**이 됨. 초록 화살표 $\phi(y_t,\cdot)$는 $Y$에서 $f(x)$를 $f(t(x))$ 쪽으로 옮김.

- **Brown:** Same image under transformation → Preserve semantic information
- **Blue:** Same transformation on different images → Learn transformation representations
- **Green:** Apply the learned transformation → Predict the transformed representation

**설명**

- 이 방법은 뒤 슬라이드에서 **STL (Self-supervised Transformation Learning)**로 불림. **transformation label(파라미터) 없이** transformation 자체를 표현으로 학습하는 것이 핵심임.
- 세 loss의 역할:
  1. **Brown, $\mathcal{L}_{\text{inv}}$**: 같은 이미지의 원본과 변환본의 표현을 가깝게 함 → **semantic 정보 보존**(기존 invariance 학습과 동일).
  2. **Blue, $\mathcal{L}_{\text{trans}}$**: **서로 다른 이미지 $x, x'$에 같은 변환 $t$**를 적용해 얻은 transformation representation $y_t^x$, $y_t^{x'}$가 같아지도록 함. 이미지 내용이 달라도 "변환 그 자체"가 공유 신호이므로 $f_T$가 content가 아닌 **변환 정보만** 뽑도록 유도됨. 라벨(crop 좌표, jitter 강도) 없이 "같은 $t$"라는 사실만 감독 신호로 씀.
  3. **Green, $\mathcal{L}_{\text{equi}}$**: 다른 이미지 $x'$에서 뽑은 변환 표현 $y_t^{x'}$를 $\phi$로 $x$의 표현 $f(x)$에 적용해 **$f(t(x))$를 예측**. 학습된 변환 표현이 실제로 변환을 일으킬 수 있어야 함 → **equivariance** 달성.
- 기존 equivariant 방법(EquiMod, AugSelf)이 augmentation 파라미터를 직접 입력/예측하는 것과 달리, STL은 **파라미터 없이 변환 표현을 학습**하므로 파라미터화하기 어려운 변환도 다룰 수 있음.
- 설계 이유(슬라이드 색 설명 기반 해석): invariance만 쓰면 변환 정보가 사라지고, equivariance 항만 쓰면 semantic 보존 신호가 없으므로 Brown(semantic 보존)과 Blue/Green(변환 학습·적용)을 함께 사용하는 것으로 이해함.

> 보충: 이 논문은 "Self-supervised Transformation Learning for Equivariant Representations" (NeurIPS 2024)로 알고 있음. 정확한 저자/연도는 원문 확인 필요. $\phi$는 변환 표현을 조건으로 받아 feature를 변환하는 작은 network로 구현되는 것으로 이해함.

## What Does STL Learn? (p.77)

**그림**: 2D 시각화 scatter plot 3개, 아래에 (a) Equivariant Transformation, (b) Transformation Representation 캡션.
- (a) 왼쪽 "EquiMod": 5개의 **분리된 균일한 둥근 cluster**가 십자 배치. 파랑(Crop) 왼쪽, 갈색(Contr.) 위, 회색(Satur.) 가운데, 하늘색(Hue) 오른쪽, 연두(Bright.) 아래. 모든 변환이 서로 대등하게 떨어져 있음.
- (a) 가운데 "STL (Ours)": **파랑(Crop)은 오른쪽 위에 따로 떨어짐**, 하늘색/갈색/회색 등 color 계열은 왼쪽 위에 겹쳐 있고, 갈색/연두는 아래쪽에 겹쳐 모임.
- (b) "STL (Ours)" Transformation Representation: 오른쪽에 **파랑 Crop 덩어리**가 크게 분리, 왼쪽에는 갈색·하늘색·회색·연두의 **color jitter 계열이 반달/동심 구조로 겹쳐서** 모임. 범례: Crop(파랑), Satur.(회색), Bright.(연두), Hue(하늘색), Contr.(갈색); 괄호로 Satur./Bright./Hue/Contr.를 "Color Jitter"로 묶음.

- Crop is separated from color transformations.
- Related color transformations are placed close together.
- This structure is learned without transformation labels.

**설명**

- **EquiMod**는 변환 종류마다 서로 분리된 둥근 cluster로 놓음(그림상 변환 간 구조 없이 따로 떨어져 보임. 거리가 모두 같다는 뜻은 아님). 반면 **STL**은 **의미적으로 관련된 변환(color jitter의 4가지 성분)은 가깝게, 성질이 다른 crop(spatial)은 멀리** 둠.
- 이는 STL이 라벨 없이도 **변환 간의 구조(어떤 변환끼리 비슷한가)**를 스스로 발견했다는 증거임. 변환 표현이 단순한 one-hot 식 분리가 아니라 **의미 있는 geometry**를 가짐.
- color 계열끼리 겹침(overlap)이 보이는데, 서로 비슷한 효과를 내서 구분이 덜 되기 때문으로 해석 가능.

## Transformation Representations by Intensity (p.78)

**그림**: UMAP 2D scatter plot. 점 색은 변환 종류별(파랑 Crop, 회색-검정 Satur., 초록 Bright., 청록 Hue, 갈색 Contr.)이고, **같은 계열 안에서 진한 색일수록 intensity가 높음**. 범례에 각 변환마다 흰색→진한색 그라데이션 막대(Crop은 단일 파랑 점), Color Jitter 괄호로 4종 묶음. 왼쪽에 회색/검정 덩어리, 가운데에 연두→진초록의 세로로 긴 띠, 오른쪽에 연한 갈색→진한 갈색의 세로 띠, 청록/하늘색 계열은 곡선 띠로 이어지고, 맨 아래 왼쪽에 파랑 Crop 덩어리가 있음.

- UMAP visualization
- darker colors representing higher intensities
- ResNet-18 model pretrained on STL10 with a transformation backbone
- Different types form different regions.
- Similar intensities are located nearby.
- Intensity changes continuously in the representation space.
- STL captures both transformation type and intensity without direct parameter supervision.

**설명**

- p.77이 변환 **종류**의 구조였다면, 여기서는 변환 **강도(intensity)**의 구조를 보여줌.
- 관찰 3가지: (1) 변환 종류별로 **영역이 나뉨**, (2) 같은 종류 안에서 **비슷한 강도는 가까이**, (3) 강도가 **띠를 따라 연속적으로 변함**. 즉 변환 표현 공간이 **type은 군집, intensity는 연속 축(manifold)**으로 조직됨.
- $\mathcal{L}_{\text{trans}}$가 같은 $t$(같은 파라미터)를 가진 쌍을 가깝게 만들기 때문에 파라미터 값을 라벨로 쓰지 않았는데도 연속 구조가 나온 것으로 추정됨(슬라이드에는 원인이 명시되어 있지 않고, 관찰 결과만 제시됨).
- 실험 설정: **ResNet-18**, **STL10**에서 pretrain, "transformation backbone" 사용. ← 데이터셋명 "STL10"은 방법 이름 STL과 철자가 겹쳐 혼동하기 쉬움 (STL-10은 이미지 데이터셋).

## Downstream Results (p.79)

**Table 4: Object Detection.** Evaluation of representation generalizability on a downstream object detection task. Average precision is reported for ImageNet100-pretrained ResNet-50 fine-tuned on VOC07+12.

| Method | AP_all | AP_50 | AP_75 |
|---|---|---|---|
| SimCLR | 45.67 | 72.50 | 47.83 |
| AugSelf | 45.99 | 72.46 | 49.23 |
| EquiMod | 51.55 | 78.03 | 56.17 |
| STL (Ours) | 51.95 | 78.34 | 56.96 |
| with AugMix (Ours) | **52.70** | **78.81** | **57.76** |

**Table 5: Transformation Prediction.** Evaluation of transformation representation from learned representation pairs. Regression tasks use MSE loss, and transformation type classification uses accuracy (%).

| Method | Regression (↓) Crop | Regression (↓) Color | Regression (↓) All | Classification (↑) Trans. Type |
|---|---|---|---|---|
| SimCLR | 0.02 | 0.13 | 0.08 | 68.54 |
| AugSelf | **0.01** | 0.04 | 0.03 | 88.49 |
| EquiMod | **0.01** | 0.07 | 0.04 | 82.20 |
| STL (Ours) | **0.01** | **0.03** | **0.02** | **93.67** |

- Preserving transformation information improves localization and generalization.

**설명**

- **Table 4 (Object Detection)**: ImageNet100으로 pretrain한 ResNet-50을 VOC07+12에서 fine-tune해 AP를 측정함. invariance 중심의 **SimCLR(45.67)**보다 equivariant 계열 **EquiMod(51.55), STL(51.95)**이 크게 높고, AugMix를 결합한 STL이 최고(52.70)임. 특히 **AP75**(엄격한 IoU 기준, localization 정확도)에서 차이가 커서(SimCLR 47.83 → STL+AugMix 57.76) **위치 정보 보존이 localization에 도움**됨을 뒷받침함. AP_all은 여러 IoU 임계값 평균, AP50/AP75는 IoU 0.5/0.75 기준임.
- **Table 5 (Transformation Prediction)**: 표현 쌍으로부터 변환 파라미터를 회귀(MSE, 낮을수록 좋음)하거나 변환 종류를 분류(accuracy, 높을수록 좋음)함. **STL이 color/all 회귀와 type 분류(93.67%)에서 최고**. SimCLR은 invariance 때문에 변환 정보가 가장 많이 사라져 color MSE 0.13, type 정확도 68.54%로 최저임. 즉 **STL 표현은 변환 정보를 실제로 담고 있음**.
- 해석: equivariance로 변환 정보를 보존하는 것이 detection처럼 spatial 정보가 필요한 downstream과 generalization에 이득임. 다만 모든 지표에서 단조롭게 좋아지는 것은 아님(예: AugSelf의 AP50 72.46은 SimCLR 72.50보다 근소하게 낮음).

> 보충: 비교 방법 — SimCLR(Chen et al., 2020, invariance 기반 contrastive), AugSelf(Lee et al., 2021, augmentation 파라미터 차이를 예측하는 auxiliary task), EquiMod(Devillers & Lefort, ICLR 2023, augmentation 파라미터를 받는 predictor로 equivariance 학습), AugMix(Hendrycks et al., 2020, 여러 augmentation chain을 섞는 기법).

## Next Week.. (p.80)

- Contrastive Learning

**설명**

- 다음 주 주제 예고 슬라이드. 이번 강의의 invariance 관점은 **contrastive learning**에서 본격적으로 형식화됨.

## VERIFIER REPORT

- p.77 설명: EquiMod를 "모두 같은 거리로 흩어 놓음"이라 한 과장을 "서로 분리된 cluster, 거리가 같다는 뜻은 아님"으로 수정.
- p.78 설명: L_trans 때문에 연속 구조가 나왔다는 단정적 인과를 "추정, 슬라이드에 원인 명시 없음"으로 완화.
- p.76 설명: "설계 이유"의 근거 없는 단정(equivariance만 쓰면 semantic 구조가 흐트러짐)을 슬라이드 색 설명 기반 해석으로 수정.
- 수식, 표 수치(Table 4/5), 그림 설명, 인용(SimCLR, AugSelf, EquiMod, AugMix, STL) 모두 원본과 일치 확인.

## FIGURES

- p.75: Invariance vs Equivariance 개념도 (삼각형 변환, $f(t(x))=f(x)$ vs $\phi(t,f(x))$), 상
- p.76: Transformation Equivariant Learning 파이프라인 전체 도식 + 세 loss ($\mathcal{L}_{inv}$, $\mathcal{L}_{equi}$, $\mathcal{L}_{trans}$), 상
- p.77: EquiMod vs STL의 transformation representation 시각화 (crop과 color 분리, color 계열 근접), 중
- p.78: UMAP으로 본 type + intensity 구조, 중
- p.79: Object detection / transformation prediction 결과 표, 중

## EXAM POINTS

1. **Invariance vs Equivariance 정의와 수식**: $f(t(x))=f(x)$ vs $f(t(x))=\phi(t,f(x))$. invariance는 $\phi=\mathrm{id}$인 특수한 경우임을 설명할 수 있어야 함.
2. **Why 질문**: 왜 모든 것에 invariant하면 안 되는가? detection(위치), flower classification(색) 예로 task마다 필요한 정보가 다름.
3. **Pretext task가 보존하는 정보 vs augmentation invariance가 버리는 정보** 비교 (p.72–73 연결).
4. **STL의 세 loss** ($\mathcal{L}_{inv}$, $\mathcal{L}_{trans}$, $\mathcal{L}_{equi}$)의 입력, 역할, 색(brown/blue/green) 대응.
5. **$\mathcal{L}_{trans}$의 핵심 아이디어**: "다른 이미지에 같은 변환"을 positive로 삼아 label 없이 transformation representation을 학습함.
6. **EquiMod vs STL 표현 공간 차이**: STL은 crop과 color가 분리되고 color 하위 변환끼리 가까움(구조 학습).
7. **intensity의 연속 구조**: 파라미터 supervision 없이도 type은 군집, intensity는 연속적으로 배치됨.
8. **Downstream 결과 해석**: SimCLR 대비 EquiMod/STL의 AP75 향상은 localization 정보 보존 때문. transformation prediction에서 SimCLR이 낮은 이유는 invariance로 정보가 소실되기 때문.
