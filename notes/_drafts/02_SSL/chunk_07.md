<!-- DRAFT | lecture: 02_SSL | chunk: 07 | pages 49–56 -->
<!-- VERIFIED | fixes: 6 | verifier: sonnet -->

## Higher layers do not generalize ... (p.49)

- 제목: "Higher layers do not generalize ..."
- 그래프: "Linear classifier on VOC07". x축 = 레이어(conv1, res2, res3, res4, res5), y축 = mAP (20~70 범위). 주석: mAP = mean Average Precision (Higher is better).
- 단일 곡선 "Jigsaw" (pretext task = Jigsaw로 학습한 네트워크). 값 대략 conv1 ≈ 27, res2 ≈ 46, res3 ≈ 57, **res4 ≈ 64.5 (최고)**, res5 ≈ 57 (다시 하락).
- 오른쪽: 입력 이미지(방 안 사진) → conv1 → … → res5 로 이어지는 CNN 블록 다이어그램 (깊어질수록 feature map이 작아지고 끝에 얇은 블록 2개 + 출력 화살표). conv1, res5 위치가 표시됨.
- 하단 굵은 글씨: "Problems: 2) the learned representations may not be general."

**설명**
- 각 레이어의 feature를 **freeze** 하고 그 위에 **linear classifier**만 학습해서 VOC07 분류 mAP를 측정하는 **linear probing** 실험임. 레이어별 representation 품질을 비교하는 표준적 방식임.
- 결과: res4까지는 깊어질수록 올라가지만 **마지막 층 res5에서 오히려 떨어짐**. supervised ImageNet pretraining에서는 최종 층 feature가 가장 좋은 것과 대비됨.
- 이유: 네트워크의 **마지막 층들은 pretext task 자체를 푸는 데 특화**(예: jigsaw permutation 맞히기)되어 down-stream task(object classification)에 필요한 semantic 정보와 어긋남. 즉 **representation이 pretext task에 overfit**되어 general하지 않음.
- "Problems: 2)" 는 pretext task 방식의 문제점 중 두 번째(일반성 부족)임. 1)은 이 chunk 앞쪽(p.48 이전)에서 다룬 내용임.
> 보충: layer-wise linear probe 벤치마크는 Goyal et al., ICCV 2019 ("Scaling and Benchmarking Self-Supervised Visual Representation Learning") 스타일과 같음. Jigsaw는 Noroozi & Favaro, ECCV 2016. 이후 SSL에서 마지막 층 대신 **중간 층 feature**를 쓰거나 projection head 앞 feature를 쓰는 관행(SimCLR 등)은 이와 같은 관찰(최종 층 특화)과 관련됨 (직접적 기원이라고 단정할 수는 없음).

## Multiple Pretext Tasks (p.50)

- 출처: Kim et al., WACV 2018, Learning Image Representations by Completing Damaged Jigsaw Puzzles
- 왼쪽 그림: 3×3 grayscale 패치 격자(소 + 철조망 기둥 + 하늘 + 풀밭 장면). 왼쪽 위 패치가 **점선 빈칸(missing patch)**이고 패치들이 섞여 있음. → (파란 삼각형 화살표) → 오른쪽: 3×3 **컬러**로 복원되고 올바른 위치로 재배열된 이미지.
- 오른쪽 텍스트 목록: Jigsaw Puzzle / Inpainting / Colorization / …

**설명**
- 하나의 입력으로 **세 가지 pretext task를 동시에** 풀게 하는 아이디어: (1) 섞인 패치의 **위치를 맞히는 jigsaw**, (2) 빠진 패치를 **채우는 inpainting**, (3) grayscale을 **컬러로 복원하는 colorization**.
- 동기: 한 가지 pretext task만 쓰면 표현이 그 task에 묶이므로(p.49) **여러 task를 합쳐 더 일반적인 표현**을 얻자는 시도임.
- 이 방법이 논문에서 **CDJP (Completing Damaged Jigsaw Puzzles)**로 불림 (p.53 표의 이름).
> 보충: Jigsaw = Noroozi & Favaro 2016, Inpainting = Pathak et al., CVPR 2016 (Context Encoders), Colorization = Zhang et al., ECCV 2016. 이 슬라이드는 세 가지를 합친 Kim et al. 2018의 개요임.

## Multiple Pretext Tasks (p.51)

- 출처: Kim et al., WACV 2018.
- 그림 (Figure 2) 7개 패널 (a)~(g), 물고기(노란 물고기 + 산호) 이미지 사용:
  - (a) Conventional 2×2 jigsaw puzzle: 4패치가 섞여 있음 → 올바른 배치로 복원.
  - (b) Complicated 2×2 jigsaw: 각 패치의 L 또는 ab 채널이 제거됨(일부는 흑백, 일부는 색만 있는 형태) → 복원.
  - (c) Complicated 2×2 jigsaw: 패치 하나가 완전히 제거됨(입력 쪽 한 칸이 점선 빈칸, 화살표 오른쪽 출력에도 점선 빈칸이 하나 표시됨) → 복원.
  - (d) Conventional inpainting: 가운데가 흰 사각형으로 가려진 컬러 이미지 → 빠진 영역 출력.
  - (e) Complicated inpainting: **L 채널만 입력**으로 받아 가려진 영역의 **ab 채널**을 출력.
  - (f) Conventional colorization: 전체 grayscale 이미지 → 컬러 출력.
  - (g) Complicated colorization: **이미지의 1/4만** 주어지고 colorization.
- 캡션 원문: "Figure 2. Illustrations of complicating self-supervised tasks. (a) Conventional 2×2 jigsaw puzzles. (b) Complicated 2×2 jigsaw puzzles; each patch's L or ab channel is dropped. (c) Complicated 2×2 jigsaw puzzles; one of the patches is completely dropped. (d) Conventional inpainting. (e) Complicated inpainting; it outputs in ab channels from an input in only L channel. (f) Conventional colorization. (g) Complicated colorization; only one-quarter of the entire image is given for colorization."

**설명**
- 핵심 설계: 각 task를 **"conventional"과 "complicated"** 버전으로 나누고 서로 **섞어서 하나로 통합**함. 즉 **damaged jigsaw** = 퍼즐 조각 중 일부는 **채널이 손상(L/ab drop)** 되거나 **통째로 사라짐** → 퍼즐을 풀면서 동시에 **누락 정보를 채우고 색을 복원**해야 함.
- **Lab 색공간**: L = 밝기(lightness), a/b = 색(chroma). L만 주고 ab를 예측하는 것이 colorization. 채널 drop은 이 분해를 이용한 것.
- 왜 "complicated"로 만드는가: task를 어렵게(damage 추가) 해야 단순한 **shortcut**(예: 패치 경계의 색/선 연속성)으로 풀 수 없고 더 **semantic한 이해**가 필요해짐. 동시에 한 네트워크가 세 task를 **공유 weight**로 풀도록 하여 표현을 일반화시킴.
> 보충: Jigsaw류 pretext는 경계 연속성, chromatic aberration 같은 low-level cue로 shortcut을 학습하는 문제가 알려져 있음 (Doersch et al. 2015, Noroozi & Favaro 2016이 gap/jitter, 채널 randomization으로 방지). 이 논문의 channel drop도 비슷한 효과가 있을 것으로 보임 (논문 의도 확인은 필요, 내 추론임).

## Multiple Pretext Tasks (p.52)

- 출처: Kim et al., WACV 2018.
- **전체 아키텍처 다이어그램** (왼→오):
  1. 입력: 3×3 패치 격자 이미지(소/철조망/하늘), 가운데 패치가 점선 빈칸.
  2. **Shuffle** 노드(모래시계 기호) → 패치들을 섞어 각각 따로 분기(점선 화살표).
  3. 각 패치(빈 패치 포함, "⋮"로 생략 표시)가 **conv1 ~ conv7** 블록(회색 = **Shared weights**)을 통과 → 출력 feature 블록 "*".
  4. 상단 경로: 모든 패치 feature를 **Concatenate(⊕)** → **fc9 → fc10** (파랑) → **L_jig.** (jigsaw loss).
  5. 중간 경로: **Arrange(◇)** 후 **⊕** → **conv9 → conv10** (빨강) → **L_inp.** (inpainting loss).
  6. 하단 경로: 주황 화살표 → **L_col.** (colorization loss).
  - 파랑/빨강/주황 점선 화살표가 "*" 블록들에서 각 head로 향함.
- 오른쪽 상세(* 블록 확대): 분기 입력 → 파랑 **conv8**(jigsaw용), 빨강 **conv8**(inpainting용), 주황 **conv8 → conv9**(colorization용) → 다시 합쳐짐.
- 범례: Shuffle(모래시계), Concatenate(⊕), Arrange(◇), Shared weights(회색 박스).

**설명**
- 구조: **공유 backbone(conv1~conv7)** 을 패치 단위로 siamese 방식(같은 weight)으로 돌리고, 그 위에 **task별 작은 head**를 둠 (conv8 이후가 task-specific). backbone은 세 task가 공유하므로 모든 task에 도움이 되는 표현을 학습함.
- **Jigsaw head**: 패치 feature들을 concat → FC → permutation 분류 (L_jig는 classification loss).
- **Inpainting head**: feature들을 **원래 위치로 arrange**(공간 배열 복원) 후 concat → conv → 빠진 패치 픽셀 복원 (reconstruction loss L_inp).
- **Colorization head**: 패치 feature에서 ab 채널 예측 (L_col).
- 전체 loss는 세 loss의 (가중)합일 것으로 보임. 슬라이드에는 합 식이 명시되지 않음.
> 보충: multi-task learning에서 backbone 공유 → 각 head가 서로 다른 gradient를 줘서 representation이 한 task에 편향되는 것을 완화. 단, p.55에서 지적하듯 task를 여러 개 합쳐도 **그 task들의 집합에 묶임**.

## Multiple Pretext Tasks (p.53)

- 출처: Kim et al., WACV 2018.
- **왼쪽 표** (Class. = 분류 %mAP, Det. = detection %mAP, Segm. = segmentation %mIU):

| Method | Class. | Det. | Segm. |
|---|---|---|---|
| ImageNet [20] | 79.9 | 56.8 | 48.0 |
| Random | 53.3 | 43.4 | 19.8 |
| RelativePosition [4] | 65.3 | 51.1 | - |
| Jigsaw [25] | 67.6 | **53.2** | 37.6 |
| Ego-motion [36] | 54.2 | 43.9 | - |
| Adversarial [6] | 58.6 | 46.2 | 34.9 |
| Inpainting [30] | 56.5 | 44.5 | 29.7 |
| Colorization [38] | 65.9 | 46.9 | 35.6 |
| Split-Brain [39] | 67.1 | 46.7 | 36.0 |
| ColorProxy [21] | 65.9 | - | _38.4_ |
| WatchingObjectMove [29] | 61.0 | 52.2 | - |
| Counting [26] | _67.7_ | 51.4 | 36.6 |
| CDJP | **69.2** | _52.4_ | **39.3** |

  (굵게 = 최고, 밑줄(여기선 기울임) = 2등: Jigsaw Det. 53.2 굵게; CDJP Class. 69.2 / Segm. 39.3 굵게; CDJP Det. 52.4, Counting Class. 67.7, ColorProxy Segm. 38.4 밑줄)
- **오른쪽 산점도**: x축 = Classification (%mAP, 55~70, 중간 축 생략 표시), y축 = Segmentation (%mIU, 29~40). 점: CDJP(우상단, 빨강, ≈69/39.3), Jig.+Col.(노랑, ≈68.5/38.6), Jig.+Inp.(노랑, ≈66.5/37.9), Jig.(노랑, ≈65/36.8), Jigsaw(초록, ≈67.6/37.6), Counting(파랑), Split-Brain(파랑, ≈67/36), ColorProxy(파랑, ≈65.9/38.4; Colorization과 거의 같은 x 위치), Colorization(초록, ≈65.9/35.6), Adversarial(파랑, ≈58.6/34.9), Inpainting(초록, 좌하단, ≈56.5/29.7). 색 구분: 초록 = 단일 pretext task, 노랑 = 저자 자체 ablation(Jig., Jig.+Inp., Jig.+Col.; 단일 Jig.와 조합 버전), 파랑 = 기타 방법, 빨강 = CDJP.

**설명**
- 비교 기준선: **ImageNet(supervised) 79.9**가 상한, **Random init 53.3**이 하한. SSL 방법은 이 사이에 위치하며 **supervised와의 격차가 아직 큼** (CDJP도 69.2 vs 79.9).
- **CDJP**는 분류(69.2)·세그멘테이션(39.3)에서 최고, detection은 Jigsaw(53.2) 다음(52.4). 즉 **task를 합치면 단일 task보다 대체로 향상**됨 (산점도에서 Jig. → Jig.+Inp. → Jig.+Col. → CDJP로 우상향).
- 단독으로 보면 **Segm.은 Inpainting(29.7)이 가장 낮고, Class.는 Ego-motion(54.2)이 가장 낮으며 Inpainting(56.5)이 그 다음**, Jigsaw/Counting이 강함 → pretext task별 **전이 성능 편차가 큼**. 이게 "pretext task 선택에 따라 representation 품질이 좌우된다"는 문제를 보여주고, 다음 절의 **더 일반적인 pretext(contrastive)** 동기가 됨.
- 표의 "[숫자]"는 논문 reference 번호. 산점도 좌표는 눈대중 근사임.

## Toward Contrastive Learning (p.54)

- 섹션 구분 슬라이드: "Toward Contrastive Learning" (진한 파랑 배경).

**설명**
- 사람이 손으로 설계한 pretext task 단계에서 **instance 간 비교(contrastive)** 로 넘어가는 전환부임.

## Pretext tasks from image transformations (p.55)

- bullet:
  - Learned representations may be tied to a specific pretext task (even multi tasks)!
  - Can we come up with a more general pretext task?
- 그림 (고양이 한 장에서 만든 4가지 변환, 각각 아래 캡션):
  - **image completion**: 윗부분이 회색 박스 "?"로 가려지고 아랫부분(고양이 몸통)만 보임.
  - **rotation prediction**: 기울어진 고양이 사진, 파란 수직 화살표(기준)와 초록 화살표(회전된 방향) 사이 각도 "θ=?".
  - **"jigsaw puzzle"**: 고양이 이미지를 2×2 패치로 잘라 섞어 놓은 그림.
  - **colorization**: 흑백 고양이 이미지.

**설명**
- 지금까지의 pretext task는 모두 **원본 이미지에 변환(가림, 회전, 셔플, 탈색)을 가하고 그걸 되돌리거나 변환 정보를 예측**하는 구조임.
- 문제의식: 학습된 표현이 **해당 변환 정보(예: 회전각, 퍼즐 순서)에 특화**되어 down-stream에 일반화가 안 됨. 여러 task를 합쳐도(p.50~53) 근본적으로 **사람이 고른 task 집합에 묶임**.
- 질문: 특정 변환의 inverse를 푸는 대신 **모든 변환에 공통된 것**(= 같은 물체라는 정체성)을 학습할 수 없을까? → p.56으로 이어짐.
> 보충: Rotation prediction = Gidaris et al., ICLR 2018 (RotNet), Image completion = Context Encoders (Pathak et al., 2016).

## A more general pretext task? (p.56)

- 이미지: 가운데 아래에 **원본 고양이 사진**. 위쪽에 p.55의 네 가지 변환 결과(가려진 고양이 "?", 회전된 고양이 θ=?, 섞인 jigsaw 패치, 흑백 고양이).
- **초록 양방향 화살표**: 원본 고양이 ↔ 각 변환 버전 4개, 라벨 **"same object"**.
- **빨강 양방향 화살표**: 원본 고양이 ↔ 강아지 사진, 라벨 **"different object"**.

**설명**
- 핵심 아이디어: 변환된 이미지들은 모두 **같은 객체(같은 instance)의 view**이므로 representation이 **서로 가까워야(similar)** 하고, **다른 객체(강아지)** 와는 **멀어져야(dissimilar)** 함. 이것이 **contrastive learning**의 직관임.
- 이전과의 차이: 기존 pretext는 "변환을 되돌려라/변환 종류를 맞혀라"라는 **task-specific 목표**였는데, 여기서는 "**같은 것끼리 가깝게, 다른 것끼리 멀게**"라는 **task-agnostic한 목표**라 변환 종류에 덜 묶임. 결과적으로 representation이 **변환에 invariant**하게 되고 객체 정체성 같은 semantic 정보를 담게 됨.
- positive pair = (원본, 변환본), negative pair = (원본, 다른 이미지).
> 보충: 이후 SimCLR(Chen et al., 2020), MoCo(He et al., 2020) 등에서 InfoNCE 계열 loss 사용 (아래 식은 SimCLR의 NT-Xent 형태이며, MoCo는 queue의 negative로 구성한 InfoNCE를 씀): $\mathcal{L}=-\log\frac{\exp(\mathrm{sim}(z_i,z_j)/\tau)}{\sum_{k\neq i}\exp(\mathrm{sim}(z_i,z_k)/\tau)}$. 이 슬라이드에는 수식이 없고 이후 슬라이드에서 다룰 것으로 보임.

## FIGURES

- p.49: Jigsaw 네트워크 레이어별 linear probe mAP 곡선 (res4 최고, res5 하락) — 중요도 상
- p.52: CDJP 전체 아키텍처 (shared conv1~7, jigsaw/inpainting/colorization head, L_jig/L_inp/L_col) — 중요도 상
- p.56: same object / different object 화살표 도식 (contrastive 동기) — 중요도 상
- p.53: 방법별 Class./Det./Segm. 비교 표 + 산점도 — 중요도 중
- p.51: conventional vs complicated 7개 task 예시(Figure 2) — 중요도 중
- p.55: 4가지 image transformation pretext task 예시 — 중요도 중

## EXAM POINTS

1. **Linear probing**이란? 레이어별로 feature를 freeze하고 linear classifier만 학습해 representation 품질을 평가하는 방식. Jigsaw에서 **res5가 res4보다 성능이 떨어지는 이유**(마지막 층이 pretext task에 특화/overfit).
2. SSL pretext task의 문제 "**학습된 representation이 general하지 않을 수 있다**"의 의미와 근거 (p.49).
3. Kim et al. 2018 (**CDJP**)의 아이디어: jigsaw + inpainting + colorization을 **공유 backbone + task별 head**로 통합, **damaged(채널 drop / 패치 drop)** 로 task를 어렵게 함. 왜 어렵게 만드는가?
4. **Lab 색공간**에서 L, ab 채널의 의미와 colorization/inpainting에서의 역할 (complicated inpainting: L → ab).
5. 단일 vs 다중 task 비교 결과: CDJP가 Class. 69.2, Segm. 39.3으로 최고. **Segm.은 Inpainting이 최저**(29.7), Class.는 Ego-motion(54.2) 최저·Inpainting(56.5) 다음. supervised ImageNet(79.9)과 격차 존재.
6. 여러 pretext task를 합쳐도 해결되지 않는 한계: **특정 task 집합에 representation이 묶임** → 더 general한 pretext 필요.
7. Contrastive learning의 직관: 같은 이미지의 변환본(**same object**)은 가깝게, 다른 이미지(**different object**)는 멀게 → positive/negative pair 정의.
8. Why 질문: 왜 "변환 예측/복원" pretext보다 "instance 구분(contrastive)" pretext가 더 일반적인가? (변환 종류에 비종속, 변환 invariance 학습)

## VERIFIER REPORT

- p.49 보충: SimCLR의 projection head 관행이 p.49 결과에서 "연유"했다는 과장 표현을 완화.
- p.51 (c): 점선 빈칸 개수 서술 수정 (캡션은 패치 하나 drop, 입력/출력 각 한 칸 점선).
- p.53 산점도: ColorProxy 좌표를 ≈64에서 ≈65.9로 수정.
- p.53 산점도: 노랑 점 설명을 저자 자체 ablation(Jig. 포함)으로 정정.
- p.53 설명/EXAM POINT 5: "Inpainting이 가장 낮음"을 정정 (Class.는 Ego-motion 54.2가 최저, Segm.은 Inpainting 최저).
- p.56 보충: 수식은 SimCLR NT-Xent 형태이고 MoCo는 다른 InfoNCE 구성임을 명시.
