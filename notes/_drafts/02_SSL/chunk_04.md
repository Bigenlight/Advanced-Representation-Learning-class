<!-- DRAFT | lecture: 02_SSL | chunk: 04 | pages 25–32 -->
<!-- VERIFIED | fixes: 5 | verifier: sonnet -->

## Pretext Task2: Inpainting - Reconstruction (p.25)

- Learn from reconstruction.
- 출처: Pathak et al., CVPR 2016, Context Encoders: Feature Learning by Inpainting
- 그림: 왼쪽에 중앙 영역이 흰색 사각형으로 마스킹된 입력 이미지(미식축구 장면) → **Encoder** → **Encoder Features**(세로 박스) → **Channel-wise Fully Connected**(채널별로 연결된 가로 화살표 다발) → **Decoder Features** → **Decoder** → 복원된 패치(마스킹된 영역 크기). 이 출력이 $\mathcal{L}(\cdot,\cdot)$ 의 첫 번째 인자로 들어가고, 두 번째 인자는 원본 이미지에서 잘라낸 정답 패치(왼쪽 아래 작은 이미지에서 점선 화살표로 연결).
- 수식:

$$\mathcal{L}_{rec}(x) = \left\| \hat{M} \odot \left( x - F\big((1-\hat{M}) \odot x\big) \right) \right\|_2^2$$

**설명**
- **Inpainting**은 이미지 일부를 가리고 주변 context로부터 가려진 영역을 예측하게 하는 pretext task임. 라벨 없이 원본 이미지 자체가 정답이 되므로 self-supervised임.
- 기호: $x$ = 원본 이미지, $\hat{M}$ = 이진 마스크(가려진 영역=1, 나머지=0), $(1-\hat{M})\odot x$ = 가려진 영역을 0으로 만든 입력, $F$ = Encoder-Decoder 네트워크(context encoder), $\odot$ = element-wise 곱.
- $\hat{M}\odot(\cdot)$ 때문에 **손실은 가려진 영역에서만 계산**됨(보이는 영역은 그냥 복사하면 되므로 의미 없음). $\|\cdot\|_2^2$ 이므로 마스크 영역 pixel들의 제곱오차 합임(평균을 내면 MSE와 상수배 차이일 뿐).
- **Channel-wise Fully Connected**: 일반 FC는 파라미터가 너무 많아서, 각 feature channel 내부에서만 공간 전체를 연결(채널 간은 연결 안 함)한 뒤 1x1 conv 등으로 섞는 구조임. 이로써 feature map 한쪽 끝의 context가 반대쪽까지 전달됨(conv의 receptive field 한계 극복).
- 왜 이런 설계인가: 구멍을 채우려면 객체의 부분, 장면 구조, semantic 정보를 이해해야 하므로 encoder가 의미 있는 표현을 학습하게 됨(p.24의 논리).
> 보충: $\ell_2$ 손실만 쓰면 여러 그럴듯한 정답의 평균을 예측하게 되어 결과가 흐릿(blurry)해지는 경향이 있음. 다음 슬라이드의 adversarial loss가 이를 보완함.

## Pretext Task2: Inpainting - Reconstruction + Adversarial Loss (p.26)

- Learn from reconstruction + adversarial loss.
- 그림: p.25와 동일한 Encoder → Channel-wise FC → Decoder 구조 및 정답 패치 점선 연결.
- 수식:

$$\mathcal{L}_{rec}(x) = \left\| \hat{M} \odot \left( x - F\big((1-\hat{M}) \odot x\big) \right) \right\|_2^2$$

$$\mathcal{L}_{adv} = \max_D \ \mathbb{E}_{x \in \mathcal{X}}\left[ \log(D(x)) + \log\big(1 - D(F((1-\hat{M}) \odot x))\big) \right]$$

- 출처: Pathak et al., CVPR 2016

**설명**
- $D$ = discriminator, 진짜 이미지 $x$ 는 1, Context Encoder $F$ 가 채운 이미지는 0으로 구분하도록 학습됨. $F$ 는 반대로 $D$ 를 속이도록 학습됨(GAN의 minimax).
- **Reconstruction loss**는 전체적 구조(coarse structure)를 맞추는 역할, **Adversarial loss**는 결과가 실제 이미지 분포처럼 sharp하고 현실적이게 만드는 역할임. 둘의 역할 분담이 핵심.
- 논문에서는 총 손실을 $\mathcal{L} = \lambda_{rec}\mathcal{L}_{rec} + \lambda_{adv}\mathcal{L}_{adv}$ 로 가중합함(슬라이드에는 없음, 보충).
> 보충: Goodfellow et al., 2014 GAN의 목적함수 형태. $\mathcal{L}_{adv}$ 는 $D$ 에 대해 max, $F$ 에 대해 min 하는 min-max 문제임. 논문에서 $\lambda_{rec}=0.999$, $\lambda_{adv}=0.001$ 정도로 reconstruction 비중이 매우 큼(기억 기준, 확실하지 않음).

## Pretext Task2: Inpainting - Results (p.27)

- 왼쪽 **Qualitative Results**: 3행 x 4열 이미지. 열 = Image(마스킹된 입력, 흰 사각형), Ours(L2), Ours(Adv), Ours(L2+Adv). 행 = 파리 거리 건물, 건물+차(흰 밴), 황금색 개(골든 리트리버).
  - L2만: 구멍이 흐릿한 평균색으로 채워짐(blur). Adv만: 선명하지만 맥락과 안 맞는 이질적 내용(예: 개 사진에서 이상한 색 덩어리). L2+Adv: 가장 자연스럽고 선명함(개의 앞발/얼굴 형태 복원).
- 오른쪽 **KNN Results**: 쿼리(페키니즈 강아지 얼굴)에 대해 feature 공간에서의 최근접 이웃 5개를 Ours / HOG / AlexNet(supervised) 세 행으로 비교.
  - Ours: 말 얼굴, 갈색 동물 얼굴, 나뭇가지(와 어두운 동물), 달마시안, 로트와일러 등 대체로 동물 얼굴 위주(나뭇가지는 예외). HOG: 가죽 장갑, 간판 글자, 어두운 강아지 얼굴, 어두운 텍스처, 회색 질감 등 색/윤곽 같은 low-level 유사성에 의존. AlexNet(supervised): 콜리 얼굴 2개, 개 얼굴/혀, 파란 천, 새(chickadee) 등 — 앞쪽은 개 계열이지만 뒤쪽은 개가 아님.
- 출처: Pathak et al., CVPR 2016

**설명**
- Qualitative 결과는 **L2 + Adv 조합이 필요함**을 보임: L2는 blur, Adv는 semantics 불일치.
- KNN 결과는 학습된 feature가 단순 pixel/gradient 유사성(HOG)이 아니라 **semantic 수준의 유사성**을 포착함을 보여줌. 다만 슬라이드의 예시만으로 supervised AlexNet보다 낫다/못하다를 단정하긴 어려움(AlexNet 이웃에도 개가 아닌 것이 섞여 있음).
> 보충: HOG(Histogram of Oriented Gradients)는 hand-crafted low-level descriptor임.

## Pretext Task2: Inpainting - Downstream 비교표 (p.28)

- 표 (PASCAL VOC 계열 transfer 성능):

| Pretraining Method | Supervision | Pretraining time | Classification | Detection | Segmentation |
|---|---|---|---|---|---|
| ImageNet [26] | 1000 class labels | 3 days | 78.2% | 56.8% | 48.0% |
| Random Gaussian | initialization | < 1 minute | 53.3% | 43.4% | 19.8% |
| Autoencoder | - | 14 hours | 53.8% | 41.9% | 25.2% |
| Agrawal et al. [1] | egomotion | 10 hours | 52.9% | 41.8% | - |
| Wang et al. [39] | motion | 1 week | 58.7% | 47.4% | - |
| Doersch et al. [7] | relative context | 4 weeks | 55.3% | 46.6% | - |
| Ours | context | 14 hours | 56.5% | 44.5% | 30.0% |

- Doersch et al. 행이 빨간 박스로 강조되고 "Relative Position of Patches" 라고 주석이 달림.
- 출처: Pathak et al., CVPR 2016

**설명**
- Random init 대비 Context Encoder는 모든 task에서 향상(특히 Segmentation 19.8 → 30.0)이며 pretraining에 14시간만 걸려서 Doersch(4주)보다 훨씬 효율적임.
- 그러나 ImageNet supervised(78.2 / 56.8 / 48.0)와는 여전히 큰 격차 → 초기 self-supervised의 한계.
- **Doersch et al.** (ICCV 2015, Unsupervised Visual Representation Learning by Context Prediction)은 두 패치의 상대 위치를 맞추는 pretext task이며, 다음 Jigsaw Puzzle의 전신임.
- Detection에서는 Wang et al.(motion, 47.4%)과 Doersch(46.6%)가 Ours(44.5%)보다 높고, Classification에서도 Wang et al.(58.7%)이 Ours(56.5%)보다 높음 ← 슬라이드 수치 기준, Ours가 모든 항목 1등은 아님(Segmentation은 비교 가능한 방법 중 최고).
> 보충: Agrawal et al. (ICCV 2015, Learning to See by Moving)은 egomotion 예측, Wang & Gupta (ICCV 2015)는 비디오 tracking 기반 학습.

## Pretext Task3: Jigsaw Puzzles - 개념 (p.29)

- 질문: "How can we solve this Jigsaw Puzzles?"
- 출처: Noroozi et al., ECCV 2016, Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles
- 그림: 왼쪽에 호랑이 사진 위에 3x3 격자로 초록색 사각형 패치들이 (서로 약간 간격 있게) 표시됨. 가운데는 이 9개 패치를 섞어 놓은 3x3 타일(얼굴, 줄무늬, 바위 등이 뒤섞임). 오른쪽은 올바르게 재배치한 결과(호랑이 얼굴이 오른쪽 위, 몸통 줄무늬가 가운데, 바위 배경이 왼쪽 위 등).

**설명**
- 이미지에서 9개 타일을 뽑아 섞고, 원래 배치(permutation)를 맞추게 하는 pretext task임. 퍼즐을 풀려면 **각 부분이 무엇인지(객체 부분)와 그 부분들의 공간적 배열(구조)** 을 알아야 하므로 semantic 표현이 학습됨.
- p.28의 Doersch 상대 위치 예측(패치 2개)을 9개 패치로 일반화한 형태임.

## Pretext Task3: Jigsaw Puzzles - 구조 (p.30)

- 그림 구성(왼쪽 → 오른쪽):
  - 자동차 이미지에 3x3 패치(초록 박스)와 더 큰 3x3 격자(빨간 점선)가 표시됨. 이미지 아래 **Permutation Set** 표: index / permutation, 예: index 64 → 9,4,6,8,3,2,5,1,7. 설명: "Reorder patches according to the selected permutation".
  - 번호 1~9의 패치 열이 선택된 permutation에 따라 섞여 파란 선으로 다음 단계 입력 위치로 연결됨.
  - 9개 병렬 CNN 브랜치(각각 11x11x96 → 5x5x256 → 3x3x384 → 3x3x384 → 3x3x256 → fc6): **Shared weights**(빨간 화살표로 표시) 가 모든 브랜치에 적용됨. 이후 concat → fc7 (4608) → fc8 (4096) → softmax (100) → 출력 64 클래스. ← 그림의 "fc7/fc8/100/64" 표기가 혼재되어 보이며 텍스트와 정합성 주의(원 논문은 fc7 4096, 출력 = permutation 수).
- 텍스트: **Given 9 tiles, there are 9! = 362,880 possible permutations**
  - Too many classes for efficient training
  - Select a subset of 64 permutations that maximize pairwise Hamming distance between permutations

**설명**
- 구조는 **Context-Free Network (CFN)**: 각 타일이 독립적으로 siamese (weight sharing) AlexNet 스타일 branch를 통과해 feature를 얻고, fc6 출력들을 concat해 permutation index를 분류함. weight sharing 덕에 파라미터가 늘지 않고, 각 branch가 타일 내용만 보고 판단해야 하므로(타일 간 정보는 마지막 FC에서 결합) 객체 수준 표현을 학습하게 됨.
- 이 task는 **permutation index를 맞추는 분류 문제**임. $9! = 362{,}880$ 클래스는 너무 많아 대신 **64개(원 논문은 실험적으로 100개, 1000개 등도 시도)의 permutation 부분집합**을 선택함.
- **max Hamming distance** 선택 이유: 서로 permutation이 크게 달라야(다수 위치가 다름) 비슷한 permutation 간 모호성이 사라지고 퍼즐이 명확히 정의됨. ← 그림의 softmax 100과 텍스트 64가 불일치 (슬라이드 원문 유지)
> 보충: Hamming distance = 두 permutation에서 서로 다른 위치의 수. 논문은 greedy하게 max-min Hamming distance가 되도록 부분집합을 만듦.

## Pretext Task3: Jigsaw Puzzles - Cheating 방지 (p.31)

- **Preventing Cheating Problem:**
  - The network may solve the puzzle using low-level cues instead of semantics.
  - Solutions:
    - Multiple puzzle configurations per image
    - Random gaps between patches
    - Random crop
    - Spatial jitter
    - Color jitter
    - Grayscale images
- 그림: 풍경(산, 호수, 바위) 이미지를 3x3로 번호 매긴 타일(2,8,1 / 4,5,6 / 7,3,9)로 섞은 예. 화살표 2개로 연결:
  - **Low-level cue #1: Edge Continuity** — 타일 경계에서 선/엣지가 이어지는지 빨간 원으로 표시, "Cheating by matching edges".
  - **Low-level cue #2: Color/Intensity Statistics** — 타일들을 블러 처리해 색/밝기 분포만 남긴 그림, "Cheating by matching color or brightness distribution".
- 출처: Noroozi et al., ECCV 2016

**설명**
- **Shortcut learning**: 네트워크는 쉬운 길을 찾으므로 semantic을 이해하지 않고도 경계 엣지의 연속성이나 색/밝기 통계(예: 하늘은 위, 물은 아래)만으로 퍼즐을 풀 수 있음. 그러면 pretext는 풀리지만 feature는 downstream에 쓸모없음.
- 대응책 매핑: **Random gaps / random crop / spatial jitter** → 인접 타일 경계의 엣지 연속성 제거. **Color jitter / Grayscale** → 색/밝기 통계 단서 제거(논문은 색 채널 간 chromatic aberration 단서도 막기 위해 grayscale, 채널 jitter 사용). **Multiple puzzle configurations per image** → 이미지당 여러 구성을 써 특정 단서의 암기를 방지.
> 보충: 논문은 이 외에 각 타일을 독립적으로 평균/표준편차 정규화하여 밝기 통계 단서를 줄임(기억 기준). Chromatic aberration은 렌즈 특성상 RGB 채널이 위치에 따라 어긋나는 현상으로, 패치의 이미지 내 절대 위치를 알려주는 단서가 됨.

## Pretext Task3: Jigsaw Puzzles - 결과 (p.32)

- 표 (Table 1: Results on PASCAL VOC 2007 Detection and Classification. The results of the other methods are taken from Pathak et al. [30].):

| Method | Pretraining time | Supervision | Classification | Detection | Segmentation |
|---|---|---|---|---|---|
| Krizhevsky et al. [25] | 3 days | 1000 class labels | **78.2%** | **56.8%** | **48.0%** |
| Wang and Gupta [39] | 1 week | motion | 58.4% | 44.0% | - |
| Doersch et al. [10] | 4 weeks | context | 55.3% | 46.6% | - |
| Pathak et al. [30] | 14 hours | context | 56.5% | 44.5% | 29.7% |
| Ours | 2.5 days | context | **67.6%** | **53.2%** | **37.6%** |

- 빨간 박스: Doersch et al. → "Relative Position of Patches", Pathak et al. → "Inpainting".
- 출처: Noroozi et al., ECCV 2016

**설명**
- Jigsaw는 이전 context 기반 방법(Doersch 55.3 / Pathak 56.5)을 Classification 67.6, Detection 53.2, Segmentation 37.6 으로 크게 앞섬 → 같은 supervision 유형(context)에서 **퍼즐 task가 더 강한 semantic 표현**을 만듦. supervised ImageNet(78.2)과의 격차도 크게 좁힘.
- 학습 시간은 2.5일로 Doersch(4주)보다 훨씬 짧지만 Pathak(14시간)보다는 김.
- p.28 표와 비교하면 같은 방법의 수치가 일부 다름: Wang and Gupta Classification 58.7 (p.28) vs 58.4 (p.32), Detection 47.4 vs 44.0, Pathak Segmentation 30.0 vs 29.7 ← 슬라이드 간 수치 불일치(원문 유지, 논문별 보고 수치 차이로 추정).

## FIGURES

- p.25: Context Encoder 구조(마스킹 입력 → Encoder → Channel-wise FC → Decoder → 정답 패치와 loss), 중요도 상
- p.27: Inpainting 정성 결과(L2 / Adv / L2+Adv 비교)와 KNN 비교(Ours vs HOG vs AlexNet), 중요도 상
- p.28: Pretext task 간 downstream 성능 비교표, 중요도 중
- p.29: Jigsaw 퍼즐 개념(섞은 타일 → 원래 배열), 중요도 중
- p.30: Jigsaw Context-Free Network 구조(9 shared-weight branch, permutation set), 중요도 상
- p.31: Jigsaw cheating 단서(edge continuity, color/intensity statistics), 중요도 상
- p.32: Jigsaw vs Doersch vs Pathak 성능 비교표, 중요도 중

## EXAM POINTS

1. **Context Encoder 손실**: $\mathcal{L}_{rec}=\|\hat M\odot(x-F((1-\hat M)\odot x))\|_2^2$ 의 각 기호와 왜 마스크 영역에서만 loss를 계산하는지 설명.
2. **L2만 쓸 때 blur 현상, adversarial loss를 추가하는 이유** (L2 = coarse structure, Adv = sharp/realistic) 및 $\mathcal{L}_{adv}$ 의 min-max 형태.
3. **Channel-wise Fully Connected 층의 목적**: feature map 전체 공간으로 context 전파, 일반 FC 대비 파라미터 절감.
4. Inpainting feature가 semantic을 포착한다는 증거(KNN: Ours vs HOG vs AlexNet) 해석.
5. **Jigsaw Puzzle 구조**: 9개 타일, shared-weight branch(Context-Free Network), permutation index 분류, $9!=362{,}880$ 인데 부분집합 사용하는 이유와 **max pairwise Hamming distance** 기준.
6. **Cheating(shortcut) 문제**: low-level cue 2가지(edge continuity, color/intensity statistics)와 각 대응책(random gap, crop, jitter, color jitter, grayscale) 매핑.
7. Pretext task 비교: Doersch(상대 위치) → Pathak(inpainting) → Noroozi(jigsaw) 성능 흐름(Classification 55.3 → 56.5 → 67.6) 및 supervised ImageNet과의 격차.
8. Pretext task가 "semantic 이해를 강제"하도록 설계되어야 하는 이유(shortcut 방지가 설계 핵심).

## VERIFIER REPORT

- p.25 설명: $\ell_2$ 손실을 "평균 제곱오차(MSE)"라 한 부분을 제곱오차 합(상수배 차이)으로 정정.
- p.27 KNN 그림 설명: HOG 이웃(어두운 강아지 얼굴 누락)과 AlexNet 이웃(뒤쪽은 파란 천, 새 등 개가 아님)을 실제 그림에 맞게 정정, Ours의 나뭇가지 예외 명시.
- p.27 설명: "AlexNet만큼 정확하지 않음" 단정(과장)을 완화.
- p.28 설명: Classification에서도 Wang et al.(58.7)이 Ours(56.5)보다 높다는 점 추가.
- p.31 보충: 정규화는 "타일 간"이 아니라 각 타일 독립 정규화로 정정.
- 그 외 표 수치(p.28, p.32), 수식(p.25–26), 퍼즐/네트워크 구조(p.29–30), 인용 논문명/연도는 원본과 일치.
