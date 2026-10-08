<!-- DRAFT | lecture: 02_SSL | chunk: 05 | pages 33–40 -->
<!-- VERIFIED | fixes: 2 | verifier: sonnet -->

## Pretext Task3: Jigsaw Puzzles — Ablation (p.33)
출처: Noroozi et al., ECCV 2016, *Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles*

**Table 4: Ablation study on the impact of the permutation set.**

| Number of permutations | Average hamming distance | Minimum hamming distance | Jigsaw task accuracy | Detection performance |
|---|---|---|---|---|
| 1000 | 8.00 | 2 | 71 | **53.2** |
| 1000 | 6.35 | 2 | 62 | 51.3 |
| 1000 | 3.99 | 2 | 54 | 50.2 |
| 100 | 8.08 | 2 | 88 | 52.6 |
| 95 | 8.08 | 3 | 90 | 52.4 |
| 85 | 8.07 | 4 | 91 | 52.7 |
| 71 | 8.07 | 5 | 92 | 52.8 |
| 35 | 8.13 | 6 | 94 | 52.6 |
| 10 | 8.57 | 7 | 97 | 49.2 |
| 7 | 8.95 | 8 | 98 | 49.6 |
| 6 | 9 | 9 | 99 | 49.7 |

(위 3행과 아래 8행은 가로선으로 구분됨: 위 = permutation 수 1000 고정 + 평균 hamming distance 변화, 아래 = permutation 수 변화)

**Table 5: Ablation study on the impact of the shortcuts.**

| Gap | Normalization | Color jittering | Jigsaw task accuracy | Detection performance |
|---|---|---|---|---|
| ✗ | ✓ | ✓ | 98 | 47.7 |
| ✓ | ✗ | ✓ | 90 | 43.5 |
| ✓ | ✓ | ✗ | 89 | 51.1 |
| ✓ | ✓ | ✓ | 88 | 52.6 |

**설명**
- 두 표 모두 Jigsaw pretext task의 **설계 선택(permutation set, shortcut 방지 장치)이 downstream(detection) 성능에 어떻게 영향 주는지** 보는 ablation임.
- **Table 4 (permutation set)**
  - 3×3 = 9개 tile을 섞는 순열은 9! = 362,880개인데, 이 중 일부 집합(1000개, 100개 등)만 골라 분류 문제(class = permutation index)로 만듦.
  - **Hamming distance** = 두 순열에서 위치별로 값이 다른 tile 수. 순열 간 거리가 클수록 서로 구분하기 쉬움(= pretext task가 쉬워짐).
  - 윗 블록: permutation 수 1000 고정, 평균 hamming distance를 8.00 → 6.35 → 3.99로 줄이면 jigsaw accuracy가 71 → 62 → 54로 떨어지고 detection도 53.2 → 50.2로 감소 → **순열들이 서로 충분히 멀어야 좋은 표현이 학습됨**.
  - 아랫 블록: permutation 수를 줄일수록 minimum hamming distance가 커지고 jigsaw accuracy는 88 → 99로 올라가지만(task가 너무 쉬워짐), detection은 10개 이하에서 49 수준으로 **떨어짐**. 즉 **pretext task의 정확도가 높다고 representation이 좋은 게 아님**. 적당히 어려운 task가 downstream에 유리함.
  - 가장 좋은 detection(53.2, 굵게 표시)은 1000개 + 평균 hamming 8.00 설정.
- **Table 5 (shortcuts)**
  - **Shortcut(trivial solution)** = 네트워크가 semantic 이해 없이 low-level 단서로 퍼즐을 푸는 것. 예: tile 경계의 edge continuity, **chromatic aberration**(색수차)으로 tile 위치 추정.
  - 대응 장치: **Gap**(tile 사이에 간격, edge 연속성 차단), **Normalization**(tile별 평균/분산 정규화로 pixel 통계 단서 제거), **Color jittering**(색수차 단서 교란).
  - 장치를 하나라도 빼면 jigsaw accuracy는 오히려 올라감(98, 90, 89 vs 모두 사용 시 88) → shortcut으로 쉽게 풀리기 때문. 반면 detection은 떨어짐(47.7, 43.5, 51.1 vs 52.6) → **shortcut을 막아야 의미 있는 feature를 학습함**.
  - 특히 Normalization 제거 시 detection 하락폭이 가장 큼(43.5).
> 보충: Noroozi & Favaro(2016)는 AlexNet 기반 Context-Free Network(CFN)를 사용함. tile을 각각 독립적으로 siamese 구조에 통과시킨 후 fc layer에서 합쳐 permutation index를 분류함. 표의 "Detection performance"는 PASCAL VOC detection mAP(%)로 알려져 있음(슬라이드에는 명시 없음).

## Pretext Task4: Colorization (p.34)
출처: Zhang et al., ECCV 2016, *Colorful Image Colorization*

- Can a network recover colors from grayscale images?
  - Color prediction requires semantic understanding.

그림: 3행 × 3열 배치의 (grayscale 입력, 색 복원 결과) 쌍 이미지 모음. 크로켓 공/말렛(잔디밭), 카메라 든 사람, 설산+호수 풍경, 앵무새와 소년, 무당벌레(주황색), 노랑·파랑 물고기, 수탉(빨간 볏), 개, 엉겅퀴 위의 monarch 나비(보라 꽃, 주황 나비). 각 쌍은 왼쪽이 흑백 입력, 오른쪽이 자연스러운 컬러 복원.

**설명**
- **Colorization**은 흑백(L channel) 이미지로부터 색을 예측하는 pretext task임. 라벨은 원본 컬러 이미지 자체에서 공짜로 얻어짐(**self-supervised**).
- **왜 representation learning에 유효한가**: 잔디는 초록, 하늘/바다는 파랑, 무당벌레는 빨강/주황 등을 맞추려면 **물체가 무엇인지, 어떤 장면인지 인식(semantic understanding)** 해야 함 → 학습 과정에서 object/scene 수준의 feature가 생김.
- 앞선 pretext(context prediction, jigsaw)가 patch 간 공간 관계를 쓰는 것과 달리, 이건 **pixel-level dense prediction**이며 같은 이미지의 채널 간 관계(L → ab)를 이용함.

## Pretext Task4: Colorization — 문제 정의 (p.35)
출처: Zhang et al., ECCV 2016

그림: 왼쪽에 **grayscale 물고기 이미지**(검은 몸통, 흰 머리/꼬리지느러미) → 가운데 큰 화살표(안에 $\mathcal{F}$) → 오른쪽에 **ab 채널만 시각화한 이미지**(회색 배경 위에 보라/파랑 계열 몸통, 노란색 영역, 주황·초록 얼룩 배경). 아래 도식: 박스 $L$ → 화살표 → CNN 모양(세로 막대 여러 개, 가운데 $\mathcal{F}$) → 화살표 → 박스 $ab$.

- Grayscale image: $L$ channel, $\mathbf{X} \in \mathbb{R}^{H \times W \times 1}$
- Color information: $ab$ channels, $\widehat{\mathbf{Y}} \in \mathbb{R}^{H \times W \times 2}$

**설명**
- 이미지를 **CIE Lab color space**로 변환해 사용: $L$ = lightness(명도, 흑백 이미지에 해당), $a$ = green–red 축, $b$ = blue–yellow 축.
  - **RGB 대신 Lab을 쓰는 이유**: 밝기와 색도(chrominance)가 분리되어 있어 $L$을 입력으로 주고 $ab$ 두 채널만 예측하면 됨. 또 Lab은 지각적으로 균일(perceptually uniform)해서 거리가 지각 차이에 가까움.
- 네트워크 $\mathcal{F}$: $\widehat{\mathbf{Y}} = \mathcal{F}(\mathbf{X})$. 입력 $H \times W \times 1$ → 출력 $H \times W \times 2$. 예측된 $ab$를 입력 $L$과 합치면 컬러 이미지가 됨.
- 입력 이미지 자체가 supervision의 원천이므로 사람의 라벨이 필요 없음.

## Pretext Task4: Colorization — 네트워크와 L2 loss (p.36)
출처: Zhang et al., ECCV 2016

그림: 네트워크 구조도. 왼쪽 입력 **Lightness L**(256×256, 채널 1) → conv1(64 ch) → conv2(128) → conv3(256) → conv4(512) → conv5(512, **à trous / dilated**) → conv6(512, **à trous / dilated**) → conv7(512) → conv8(256) → **(a,b) probability distribution**(파란 블록, 313 × 64) → 오른쪽 **Color ab**(2 ch, 224) 출력, 이를 **+ L** 하여 **Lab Image** 생성(엉겅퀴와 나비 컬러 이미지). 블록 옆 숫자는 채널 수/해상도(128, 64, 32 등)로 conv가 진행되며 공간 해상도는 줄고 conv8에서 다시 키워짐.

수식:
$$\mathbf{X} \in \mathbb{R}^{H\times W\times 1},\quad \widehat{\mathbf{Y}} = \mathcal{F}(\mathbf{X})$$
$$L_2(\widehat{\mathbf{Y}}, \mathbf{Y}) = \frac{1}{2}\sum_{h,w} \left\| \mathbf{Y}_{h,w} - \widehat{\mathbf{Y}}_{h,w} \right\|_2^2$$

**설명**
- 네트워크는 VGG 스타일의 fully convolutional 구조로 conv1~conv8의 8개 블록. 앞쪽에서 해상도를 줄이고 conv5, conv6에는 **dilated(atrous) convolution**을 사용해 해상도를 더 줄이지 않으면서 receptive field를 넓힘(넓은 문맥으로 색을 판단). conv8 이후 upsampling으로 출력 해상도 복원.
- **L2 loss**: 각 pixel $(h,w)$에서 정답 $ab$ 값 $\mathbf{Y}_{h,w}\in\mathbb{R}^2$와 예측값의 제곱 오차를 합하고 $\tfrac12$ 곱함. 즉 **regression**(연속값 직접 예측).
- ← 그림은 313-bin 확률분포를 출력하는 classification 버전의 구조인데 수식은 L2(regression)만 적혀 있음. 다음 장(p.37)과 연결해서 봐야 함.
> 보충: L2 regression은 ambiguity가 있을 때 가능한 색들의 **평균(mean)** 으로 수렴해 채도가 낮고 탁한(desaturated) 결과가 나옴. 다음 슬라이드에서 이를 해결함.

## Pretext Task4: Colorization — Classification 문제로 전환 (p.37)
출처: Zhang et al., ECCV 2016

- Can it be solved by classification instead of regression.
  - Multiple colors can be plausible.
  - Regression averages them.
  - Continuous color prediction
  - → 313 color classes

수식:
$$L_{cl}(\widehat{\mathbf{Z}}, \mathbf{Z}) = -\sum_{h,w} v(\mathbf{Z}_{h,w}) \sum_{q} \mathbf{Z}_{h,w,q}\,\log\!\big(\widehat{\mathbf{Z}}_{h,w,q}\big)$$

그림: 우측에 $RGB(a,b\,|\,L=50)$ 제목의 2D 컬러 격자. 가로축 $b$(−110 ~ 110), 세로축 $a$(−110 ~ 110), 눈금 −55, 0, 55. $L=50$에서 RGB로 표현 가능한(in-gamut) $(a,b)$ 칸만 색으로 채워져 있고 모서리는 비어 있음. 좌하단 보라/파랑, 우하단 빨강, 우상단 초록, 중앙은 탁한 갈색~회색. (우측 끝에 잘린 다른 축 표시 일부가 보임.)

**설명**
- **문제의식**: 색은 본질적으로 **multimodal**함. 사과는 빨강이거나 초록일 수 있고 옷도 여러 색일 수 있음. L2 regression은 여러 가능한 값의 **평균**을 내기 때문에 채도 낮은 회색빛 결과가 나옴.
- **해결**: 연속적인 $ab$ 공간을 격자로 양자화해 **in-gamut인 313개 bin**을 만들고 pixel마다 **313-way classification** 수행. 출력은 각 pixel의 bin 확률분포 $\widehat{\mathbf{Z}}_{h,w}\in[0,1]^{Q}$, $Q=313$.
- 기호
  - $\mathbf{Z}_{h,w}$: 정답 $ab$를 soft-encoding한 target 분포(원 논문: 가까운 5개 bin에 Gaussian kernel로 분산), $\mathbf{Z}_{h,w,q}$는 bin $q$의 값.
  - $\widehat{\mathbf{Z}}$: 네트워크 예측 확률.
  - $\sum_q \mathbf{Z}_{h,w,q}\log\widehat{\mathbf{Z}}_{h,w,q}$: pixel별 **cross-entropy**.
  - $v(\cdot)$: **class rebalancing weight**. 자연 이미지는 배경(하늘, 벽 등) 저채도 색이 압도적이므로 드문(채도 높은) 색의 loss를 크게 가중함. 슬라이드엔 정의가 없음.
> 보충: 논문에서 $v(\mathbf{Z}_{h,w}) = w_{q^*}$, $q^*=\arg\max_q \mathbf{Z}_{h,w,q}$, $w \propto \big((1-\lambda)\tilde p + \lambda/Q\big)^{-1}$ ($\tilde p$ = Gaussian smoothing한 경험적 색 분포, $\lambda=1/2$)임. 추론 시에는 **annealed-mean**(temperature $T=0.38$)으로 분포를 $ab$ 값으로 변환함(mean과 mode의 절충).

## Pretext Task4: Colorization — Qualitative Results (p.38)
출처: Zhang et al., ECCV 2016

- Qualitative Results

그림: 2행 × 3열. 열 = (grayscale 입력, **Regression** 결과, **Classification** 결과). 1행: lionfish 이미지. regression은 전체가 누런 세피아/탁한 푸른빛, classification은 배경이 더 선명한 파란색. 2행: "Restaurant Hunzeberger" 열기구. regression은 풍선이 어두운 갈색/보라 톤으로 칙칙하고(하늘은 둘 다 파랑), classification은 풍선에 선명한 붉은색이 살아남.

**설명**
- 같은 구조에서 **L2 regression loss vs 313-class cross-entropy**만 바꾼 비교임. classification 결과가 **더 생생(vivid)하고 채도 높은** 색을 만듦 → regression은 **평균색 → desaturation** 문제가 있음을 시각적으로 확인.
- 정량 지표보다 시각적 결과로 설계 선택을 정당화하는 슬라이드.

## Pretext Task4: Colorization — Dataset/Task Generalization on PASCAL (p.39)
출처: Zhang et al., ECCV 2016

**Dataset and Task Generalization on PASCAL [37]**

| fine-tune layers | Class. [Ref] | Class. fc8 | Class. fc6-8 | Class. all | Det. [Ref] | Det. all | Seg. [Ref] | Seg. all |
|---|---|---|---|---|---|---|---|---|
| ImageNet [38] | - | 76.8 | 78.9 | 79.9 | [36] | 56.8 | [42] | 48.0 |
| Gaussian | [10] | – | – | 53.3 | [10] | 43.4 | [10] | 19.8 |
| Autoencoder | [16] | 24.8 | 16.0 | 53.8 | [10] | 41.9 | [10] | 25.2 |
| k-means [36] | [16] | 32.0 | 39.2 | 56.6 | [36] | 45.6 | [16] | 32.6 |
| Agrawal et al. [8] | [16] | 31.2 | 31.0 | 54.2 | [36] | 43.9 | – | – |
| Wang & Gupta [15] | – | 28.1 | 52.2 | 58.7 | [36] | 47.4 | – | – |
| *Doersch et al. [14] | [16] | 44.7 | 55.1 | **65.3** | [36] | **51.1** | – | – |
| *Pathak et al. [10] | [10] | – | – | 56.5 | [10] | 44.5 | [10] | 29.7 |
| *Donahue et al. [16] | – | 38.2 | 50.2 | 58.6 | [16] | 46.2 | [16] | 34.9 |
| Ours (gray) | – | **52.4** | **61.5** | **65.9** | – | 46.1 | – | 35.0 |
| Ours (color) | – | **52.4** | **61.5** | **65.6** | – | 46.9 | – | **35.6** |

(Class. = %mAP, Det. = %mAP, Seg. = %mIU. Ours (gray)/(color)는 colorization 모델에 gray / color 입력을 넣어 fine-tune한 비교임.)

**설명**
- self-supervised로 pre-train한 AlexNet을 **PASCAL VOC**의 세 task(classification, detection, segmentation)에 transfer해 평가. 열은 fine-tuning하는 layer 범위: **fc8**만(= linear classifier에 가까움), **fc6-8**, **all**.
- 첫 행 **ImageNet(supervised)** 이 상한선(79.9 / 56.8 / 48.0)이고 **Gaussian**(랜덤 초기화)이 하한 기준선임.
- Colorization(Ours)은 fc8/fc6-8처럼 **적은 layer만 fine-tune하는 설정에서 기존 방법보다 큰 폭으로 우수**(52.4, 61.5) → feature 자체가 이미 linear하게 분리되는 좋은 표현임. all fine-tune classification에서도 65.9로 Doersch(65.3)와 비슷하거나 약간 우세. detection은 Doersch(51.1)가 더 높고, segmentation은 Ours(color) 35.6이 최고.
- `*` 표시의 의미는 슬라이드에 설명 없음. ← 원 논문 표에서는 해당 방법들이 다른 설정/구조를 쓴다는 표기로 추정됨.
- 학습 이미지가 ImageNet이지만 라벨은 쓰지 않았고 평가는 다른 데이터셋/task이므로 "dataset and task generalization"이라 함.
> 보충: Doersch et al.(ICCV 2015) = Context Prediction(relative position), Pathak et al.(CVPR 2016) = Context Encoders(inpainting), Donahue et al.(ICLR 2017) = BiGAN, Agrawal et al.(ICCV 2015) = egomotion, Wang & Gupta(ICCV 2015) = video tracking 기반 학습임.

## Pretext Task5: Rotation Prediction (p.40)
출처: Gidaris et al., ICLR 2018, *Unsupervised Representation Learning by Predicting Image Rotations*

- Which orientation is correct?
  - To recognize rotation, the network must recognize objects.

그림: 5장의 이미지가 가로로 나열되고 각각 캡션으로 적용된 회전 각도가 표기됨. 순서: 나무줄기의 새(**90° rotation**), 눈 위의 검은 개(**270° rotation**), 졸업 가운 입은 사람(**180° rotation**), 개구리(**0° rotation**), 선반 위 물체(**270° rotation**). 회전된 이미지는 옆으로 누워 있거나 거꾸로 되어 있어 방향이 어색함.

**설명**
- **Rotation prediction**: 이미지를 {0°, 90°, 180°, 270°} 중 하나로 회전시키고 네트워크가 **어떤 회전이 적용됐는지 4-way classification**으로 맞히게 함.
- **왜 되는가**: 회전을 맞히려면 "새는 위가 머리", "개구리는 땅에 앉아 있음", "사람 얼굴은 위가 이마" 같은 **물체의 canonical orientation**을 알아야 함 → 객체의 종류·자세를 인식하는 semantic feature가 학습됨.
- 라벨은 자동 생성: 적용한 회전 각도가 곧 라벨(다음 p.41: "No labels are needed, The applied rotation becomes the label").
- 장점: 구현이 단순하고 전체 이미지를 회전시키므로 jigsaw의 tile 경계/색수차 같은 **low-level shortcut이 적음**.
> 보충: Gidaris et al.은 AlexNet 기준 당시 unsupervised SOTA를 보고함. 회전 대칭 물체나 방향이 모호한 이미지(aerial 등)는 약점임.

## FIGURES
- p.33: Jigsaw ablation 표 두 개(Table 4 permutation set / Table 5 shortcuts) — 중 (shortcut 개념과 "pretext 정확도 ≠ downstream 성능")
- p.34: Colorization 결과 예시 모음(흑백 → 컬러 쌍) — 중
- p.35: L → F → ab 문제 정의 도식 — 상
- p.36: Colorization 네트워크 구조도(conv1~8, 313 확률분포, +L → Lab) — 상
- p.37: $ab$ 격자 313-bin 그림 $RGB(a,b|L=50)$ — 상
- p.38: Regression vs Classification 정성 비교 — 상
- p.39: PASCAL transfer 표 — 중
- p.40: 회전 예시 5장(0/90/180/270) — 상

## EXAM POINTS
1. **Jigsaw의 shortcut(trivial solution)** 이란 무엇이며(edge continuity, chromatic aberration) 이를 막는 장치 3가지(gap, normalization, color jittering)와 각각의 효과.
2. **Pretext accuracy가 높다고 representation이 좋은 것은 아님**: permutation 수가 적을수록(min hamming distance 큼) jigsaw accuracy는 계속 오르지만(88→99) detection은 35개까지는 ~52.6 수준으로 유지되다가 10개 이하에서 ~49로 떨어지는 이유(Table 4).
3. **Colorization 설정**: Lab color space에서 $L$ 입력($H\times W\times1$) → $ab$ 출력($H\times W\times2$)인 이유, 왜 colorization이 semantic understanding을 요구하는가.
4. **L2 regression vs classification**: color의 multimodality 때문에 regression은 평균색(desaturated)이 나옴 → $ab$를 313 bin으로 양자화한 cross-entropy로 해결. $L_{cl}$ 수식의 각 항($\mathbf{Z}$, $\widehat{\mathbf{Z}}$, $v(\cdot)$)의 의미.
5. 왜 class rebalancing $v(\cdot)$이 필요한가(자연 이미지는 저채도 색이 압도적으로 많음).
6. Colorization 네트워크에서 **dilated convolution**을 쓰는 이유(해상도 유지 + receptive field 확대).
7. **Rotation prediction**: 4-way classification 정의, 라벨 자동 생성 방식, 회전을 맞히려면 객체를 인식해야 하는 이유.
8. **Transfer 평가 프로토콜**: PASCAL에서 fc8 / fc6-8 / all fine-tune의 의미, ImageNet supervised가 상한이고 Gaussian(random init)이 하한 기준선인 이유.

## VERIFIER REPORT
- p.38 그림 설명: regression 결과의 하늘도 파랑이므로 "classification만 하늘이 파랑"으로 읽히던 표현을 수정(풍선 색 차이만 강조).
- EXAM POINTS 2번: Table 4에서 permutation 35개까지 detection은 거의 유지(52.6~52.8)되고 10개 이하에서만 하락하므로 서술을 정확히 수정.
- 표(p.33, p.39) 수치, 수식(p.35-37), 인용 논문/연도는 원본과 일치 확인.
