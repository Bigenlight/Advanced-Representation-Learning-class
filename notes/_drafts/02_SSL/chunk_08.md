<!-- DRAFT | lecture: 02_SSL | chunk: 08 | pages 57–64 -->
<!-- VERIFIED | fixes: 4 | verifier: sonnet -->

## Space-Time Cubic Puzzles 논문 소개 (p.57)
- 섹션 도입 슬라이드: 논문 표지 캡처 — The Thirty-Third AAAI Conference on Artificial Intelligence (AAAI-19)
- 제목: **Self-Supervised Video Representation Learning with Space-Time Cubic Puzzles**
- 저자: Dahun Kim, Donghyeon Cho, In So Kweon (Dept. of Electrical Engineering, KAIST, Daejeon, Korea)
- 하단 인용: Kim et al., AAAI 2019, Self-Supervised Video Representation Learning with Space-Time Cubic Puzzles
- 직전 슬라이드(p.56)의 질문 "A more general pretext task?"에 대한 답으로 제시되는 논문

**설명**
- 이전까지(jigsaw, rotation 등)의 pretext task는 **이미지** 대상이었음. 이 논문은 같은 "퍼즐 풀기" 아이디어를 **비디오(시공간 3D 데이터)** 로 확장한 것임.
- 요지: 비디오를 시공간 큐브로 잘라 섞은 뒤 원래 순서를 맞추게 하면, 라벨 없이도 **appearance + motion/temporal order** 를 함께 담은 representation을 학습할 수 있음.
> 보충: Noroozi & Favaro (2016) Jigsaw puzzle (이미지 3×3 patch 순서 예측), Misra et al. (2016) Shuffle & Learn (프레임 순서 검증), Lee et al. (2017) OPN (Order Prediction Network)이 비디오 SSL의 선행 연구. Cubic Puzzle은 이들을 시간+공간 동시 퍼즐로 통합한 것.

## From Image Representation to Video Representation (p.58)
- 좌측: 논문 Figure 1 — 공을 던지는(throwing) 사람의 프레임 사진 묶음. 위 행 (a) 와 (a')(세로선으로 구분; (a)는 1장, (a')는 3프레임), 아래 행 (b) 와 (b')((b) 1장, (b')는 3프레임). 세로선 왼쪽 두 장(a, b)은 한 쌍의 이미지이고 오른쪽 (a'), (b')는 그 사이를 이어 주는 비디오 시퀀스. 사람이 공을 손으로 던지는 장면이며 (b')에는 오른쪽 가장자리에 다른 사람의 손이 보임.
- 캡션: "Figure 1: **Ambiguity in time direction.** Given a pair of images on the left, it is ambiguous to determine the time direction in between. One can finally identify the direction (b' → a') and action (*throwing*) when given video sequences."
- 우측 텍스트:
  - An image provides spatial appearance.
  - A video additionally provides:
    - Temporal order
    - Motion
    - Changes over time
- 하단 질문: **Can we understand an action from two frames?**

**설명**
- 이미지는 한 시점의 **공간적 appearance** 만 줌. 비디오는 여기에 **시간 순서, 움직임, 시간에 따른 변화**가 추가됨.
- Figure 1의 요점: 두 프레임만 보면 공이 손으로 *오는* 건지(catch) *가는* 건지(throw) **시간 방향이 모호함(ambiguity in time direction)**. 전체 시퀀스를 보면 b' → a' 방향과 "throwing" action이 확정됨.
- 따라서 action을 이해하려면 2-frame 정적 정보로는 부족하고, **여러 프레임에 걸친 temporal context** 를 모델이 학습해야 함 → 비디오 전용 pretext task의 동기.
> 보충: 이 "time direction" 모호성은 Arrow of Time (Pickup et al., 2014; Wei et al., 2018) 연구와도 연결됨. 비디오 라벨링은 이미지보다 훨씬 비싸서 SSL의 이득이 큼.

## Space-Time Cubic Puzzle (p.59)
- **Pretext task:**
  - Arrange shuffled video cubes
  - into their original space-time order.
- **Spatial Puzzle**
  - Where was each cube located?
- **Temporal Puzzle**
  - When did each cube occur?

**설명**
- 과제 정의: 비디오에서 잘라낸 **video cube**(공간×시간 3D 조각)들을 섞어 놓고 **원래 시공간 배치로 되돌리는** 것이 pretext task임. 정답 라벨은 우리가 섞은 순열 자체이므로 사람의 annotation이 불필요함.
- 두 종류의 퍼즐이 결합됨:
  - **Spatial puzzle**: 같은 시간대의 서로 다른 공간 위치 cube → "각 cube가 **어디(where)** 에 있었나" (이미지 jigsaw의 비디오 버전)
  - **Temporal puzzle**: 같은 공간 위치의 서로 다른 시간대 cube → "각 cube가 **언제(when)** 일어났나" (frame order prediction의 cube 버전)
- 이 퍼즐을 풀려면 객체의 구조/배치(공간 맥락)와 동작의 진행 순서(시간 맥락)를 이해해야 하므로, 그 과정에서 유용한 feature가 학습됨.

## Generating Space-Time Cubic Puzzles (p.60)
- 좌측 그림: 3D 직육면체 비디오 볼륨. 축은 **Width(224 pixels)**, **Height**, **Time(32 frames)**. 파란색 반투명 슬랩이 **Spatial sampling** (공간을 2×2로 나눈 4개 영역, 같은 시간 구간), 빨간색 반투명 슬랩이 **Temporal sampling** (시간축을 따라 4개로 나눈 구간, 같은 공간 영역). 큐브 높이는 **112 pixels** 로 표기됨.
- 우측 절차:
  1. Divide a video into a 2 × 2 × 4 grid.
  2. Sample four video cubes.
     - Spatial: 2 × 2 × 1
     - Temporal: 1 × 1 × 4
  3. Randomly permute the cubes.
  4. Predict the original arrangement.
  - → 4 cubes produce 4! = 24 possible permutations.

**설명**
- 비디오를 (Height 2) × (Width 2) × (Time 4) 격자로 분할 → 총 16개 cell. 여기서 매번 **4개의 cube** 만 뽑아 퍼즐을 구성함.
- 퍼즐 종류에 따라 뽑는 방식이 달라짐:
  - **Spatial puzzle**: 한 시간 구간에서 2×2 공간 위치 4개 (2×2×1)
  - **Temporal puzzle**: 한 공간 위치에서 시간 구간 4개 (1×1×4)
- 뽑은 4개를 **랜덤 순열**로 섞고, 네트워크는 어떤 순열이 적용되었는지 **classification** 으로 맞춤. 4개 → 4! = **24 클래스**.
- 수식으로 정리하면 순열 집합 $S_4$, $|S_4| = 4! = 24$. 모델은 $f_\theta(c_{\pi(1)},\dots,c_{\pi(4)}) \to \pi \in S_4$ 를 예측함. (여기서 $c_i$는 cube, $\pi$는 적용된 순열)
> 보충: 순서 맞추기를 regression이 아니라 permutation classification으로 푸는 건 Jigsaw (Noroozi & Favaro, 2016)와 동일한 설계. 순열이 많을 때는 Hamming distance가 큰 부분집합만 쓰지만, 여기선 4개뿐이라 전체 24개 사용.
- ← 슬라이드 그림의 치수(224 pixels = 전체 Width, 112 pixels = 한 cube의 높이, 32 frames)는 p.63의 112×112×32 cube와 같은 단위로 보이며, 학습 입력은 그 안에서 crop한 80×80×16임(p.62, p.63). 정식 노트에서는 p.63 기준(crop 후 80×80×16)으로 정리할 것.

## Generating Space-Time Cubic Puzzles — 예시 (p.61)
- 좌측 **(Spatial)**: 같은 시점의 cube 4개 A, B, C, D가 2×2 형태로 배치된 장면. A는 관중석/사람들, B와 C는 모래(sand)가 큰 영역, D는 점프하는 선수(흰 상의, 검은 반바지)가 있는 영역. 각 cube 뒤에 겹쳐진 프레임들은 시간축 깊이를 나타냄.
- 우측 **(Temporal)**: 한 위치에서 시간순으로 뽑은 cube 4개 A, B, C, D를 가로로 나열 (멀리뛰기 선수의 동작: A 도약 자세, B 멀리 있는 선수와 공, C 착지 직전의 크게 접힌 자세, D 달려오는 자세).
- 상단 문구: "Q. Can you arrange these?"
- 정답 문구는 뒤집혀(upside-down) 인쇄되어 있음: "A. Spatial: A-D-B-C", "A. Temporal: B-D-A-C"
- 하단 캡션: "Figure 4: Example spatial and temporal tuples."

**설명**
- 퍼즐이 실제로 어떻게 생겼는지 보여주는 예시. 질문(Q)에 대한 답(A)을 **거꾸로 인쇄**해 독자가 먼저 풀어 보게 하는 장치임.
- 슬라이드에 표시된 A~D는 섞인 상태이고, 정답 "A-D-B-C"(Spatial)는 원래 2×2 배치를 (좌상, 우상, 좌하, 우하) 순으로 읽은 것으로 해석됨(관중석 A, 선수 D가 위, 모래 B, C가 아래). Temporal 정답 "B-D-A-C"는 시간순.
- **Spatial 예**: 관중석은 위쪽, 모래는 아래쪽 같은 장면 구성 상식이 있어야 풀 수 있음 → 객체 부분(part)과 장면 레이아웃 이해를 유도.
- **Temporal 예**: 점프 동작의 순서(달려오기 → 도약 → 공중 → 착지)를 알아야 풀 수 있음 → motion과 action 진행 순서의 이해를 유도.
- 그림 자체가 퍼즐이라 정식 노트에도 넣을 가치가 있음.

## Learning Space-Time Features with a 3D CNN (p.62)
- 좌측 그림: 네트워크 구조도. 입력(input) 80×80×16×(3)의 cube 4개가 세로로 쌓여 있고, 각 cube마다 같은 구조의 3D CNN tower가 가로로 이어짐:
  - input 80×80×16 (3) → conv1 40×40×16 (64) → res2 20×20×8 (64) → res3 10×10×4 (128) → res4 5×5×2 (256) → res5 3×3×1 (512) → conv2 (512)
  - 4개 tower의 conv2 출력(각 512)을 **concat → (2048)** → **fc (48)**
- 우측 bullet:
  - Four shared 3D CNN towers
  - One tower processes one video cube
  - Features are combined only at the end
  - The network predicts the permutation
- 그림 아래: 24 permutations × 2 orientations = 48 classes
- 하단 문장: Each tower does not know whether the puzzle is spatial or temporal.

**설명**
- 구조는 **Siamese(weight sharing) 방식의 4-tower 3D ResNet-18** 계열. 같은 가중치를 공유하는 tower가 각 cube를 독립적으로 처리해 512-d feature를 만들고, **마지막에서만** concat(4×512=2048)해 FC layer로 순열(48 클래스)을 예측함.
- 왜 늦게 합치는가(late fusion): 중간에 cube 사이 정보를 섞으면 cube 경계의 low-level 단서로 쉽게 풀어버릴 수 있음. 각 tower가 **cube 내부의 semantic feature만으로** 표현을 만들도록 강제하는 설계임 (jigsaw의 CFN과 같은 철학).
- **3D conv** 를 쓰는 이유: 2D conv와 달리 시간축까지 kernel이 걸쳐 motion/temporal 패턴을 직접 학습함. downstream에서는 tower 하나가 feature extractor로 쓰이고 pretext head는 버림.
- 48 클래스 = 24 순열 × 2 orientation(정방향/상하반전, p.63의 upside-down classification과 연결).
- "Each tower does not know whether the puzzle is spatial or temporal" → tower는 가중치를 공유하고 자기 cube만 보기 때문에 퍼즐 종류를 모름. 따라서 **하나의 표현이 공간+시간 양쪽 퍼즐에 모두 쓰이는 범용 feature**가 되도록 학습됨.
- ← 슬라이드 그림의 치수 표기가 비스듬히 기울어져 읽기 어려움. 최종 3×3×1은 stride/pooling 결과로 추정.

## Avoiding Shortcut Solutions (p.63)
- 좌측 그림: 큰 직육면체(112 pixels × 112 pixels × 32 frames) 안에 빨간 점선으로 작은 cube(80 pixels × 80 pixels × 16 frames)가 들어 있음 — 큰 cube 내부에서 작은 crop을 임의로 잘라냄을 보여줌.
- 우측:
  - Possible shortcuts:
  - Chromatic aberration
  - Matching cube boundaries
  - Low-level image statistics
- 좌하단 **Regularization:**
  - Channel replication
  - Spatio-temporal jittering
  - Upside-down classification

**설명**
- **Shortcut solution**: 모델이 의도한 semantic 이해(객체/동작) 없이 **저수준 단서**만으로 pretext task를 푸는 현상. 이러면 학습한 feature가 downstream에 쓸모없음.
- 가능한 shortcut:
  - **Chromatic aberration**: 렌즈 색수차 때문에 이미지 위치(중심으로부터 거리)에 따라 R/G/B 채널이 미세하게 어긋남 → 색 정보만으로 cube의 공간 위치를 알아맞힘.
  - **Matching cube boundaries**: 인접한 cube의 경계 픽셀/프레임이 이어지는 것을 매칭해서 풀 수 있음.
  - **Low-level image statistics**: 밝기, 색, 노이즈 분포, 압축 아티팩트 등으로 순서를 추정.
- 대응 regularization:
  - **Channel replication**: 한 채널을 3채널로 복제해 입력 → 색수차 단서 제거.
  - **Spatio-temporal jittering**: 큰 cube(112×112×32)에서 작은 cube(80×80×16)를 임의 위치로 잘라 cube 사이에 **gap**과 위치 흔들림을 만듦 → 경계 매칭 불가.
  - **Upside-down classification**: cube를 상하 반전해 정방향/반전을 추가로 예측하게 함 → 24×2 = 48 클래스(p.62). 저수준 통계로는 풀 수 없고 장면의 상하 구조(중력, 자세)를 이해해야 하는 보조 과제임.
> 보충: 이 shortcut 문제는 Doersch et al. (2015) context prediction, Noroozi & Favaro (2016) Jigsaw에서 이미 관찰되었고(chromatic aberration → 채널 드롭/그레이 변환, gap 삽입), 이 논문이 그 대응책을 비디오에 확장함.

## Spatial and Temporal Context Are Complementary (p.64)
- 표 (Table 2: **comparison with alternative methods.** Top-1 accuracies on UCF10. All methods use 3D ResNet-18, and the accuracies are averaged over three splits.):

| Method | UCF101 (%) |
|---|---|
| 3D AE | 48.7 |
| 3D AE + future | 50.1 |
| 3D inpainting | 50.9 |
| 3D S-puzzle | 58.5 |
| 3D T-puzzle | 59.3 |
| 3D ST score ensemble | 61.3 |
| **3D ST-puzzle (full)** | **65.8** |

- 결론 bullet: Joint space-time learning performs better than learning each dimension separately.
- ← 캡션의 "UCF10"은 "UCF101"의 오타로 보임(원문 유지). 표에서 위 3행(복원/생성 계열)과 아래 4행(puzzle 계열) 사이에 구분선이 있음.

**설명**
- 모든 방법이 동일한 backbone(**3D ResNet-18**)을 쓰고 UCF101 fine-tuning 정확도(3 split 평균)로 비교한 ablation임.
- 비교군: 위 3개는 **생성/복원형 pretext** (3D autoencoder, 미래 프레임 예측을 더한 AE, 3D inpainting) → 48.7~50.9%. 아래 4개는 **퍼즐형 pretext**.
- 핵심 결과:
  - 공간만(S-puzzle) 58.5, 시간만(T-puzzle) 59.3 → 둘 다 복원형(48.7~50.9)보다 **약 7.6~10.6점 높음**: 픽셀 복원보다 구조/순서를 맞추는 discriminative pretext가 더 좋은 representation을 줌.
  - **ST score ensemble** 61.3: S-puzzle과 T-puzzle을 따로 학습한 뒤 점수를 앙상블 → 개별보다 약간만 향상.
  - **ST-puzzle (full)** 65.8: 하나의 네트워크가 공간+시간 퍼즐을 **함께(jointly)** 학습 → 앙상블 대비 +4.5점. 즉 공간 맥락과 시간 맥락은 **상호 보완적(complementary)** 이며, 단순 합이 아니라 joint learning에서 시너지가 생김.
- 슬라이드 제목의 주장: 하나의 representation이 where와 when을 동시에 풀어야 더 풍부한 spatio-temporal feature가 학습됨.

## FIGURES
- p.58: Figure 1 — 던지기 장면의 시간 방향 모호성 (두 프레임 vs 전체 시퀀스), 중요도 상
- p.60: space-time 볼륨에서 spatial sampling(파랑)/temporal sampling(빨강) cube 추출 3D 다이어그램, 중요도 상
- p.61: Figure 4 — spatial/temporal 퍼즐 예시 (A~D cube와 정답), 중요도 중
- p.62: 4-tower 공유 3D CNN 구조도 (conv1~res5 → concat 2048 → fc 48), 중요도 상
- p.63: 112×112×32 큰 cube 안에서 80×80×16 crop (spatio-temporal jittering), 중요도 중
- p.64: Table 2 — 3D AE/inpainting vs S/T/ST-puzzle UCF101 정확도, 중요도 상

## EXAM POINTS
- **Space-Time Cubic Puzzle pretext task 정의**: 비디오를 2×2×4 격자로 나눠 4개의 cube(spatial 2×2×1 또는 temporal 1×1×4)를 뽑고, 랜덤 순열로 섞은 뒤 원래 배열을 예측하는 permutation classification. 클래스 수 4! = 24 (×2 orientation = 48).
- **Spatial puzzle vs Temporal puzzle**: where(각 cube의 위치) vs when(각 cube의 시점). 각각 무엇을 학습하게 하는지 설명할 수 있어야 함.
- **왜 이미지 representation으로 부족한가**: 이미지는 spatial appearance만, 비디오는 temporal order/motion/change 추가. 두 프레임만으로는 시간 방향이 모호(Figure 1, throwing 예시).
- **아키텍처 설계 이유**: weight-shared 4 tower 3D CNN, cube별 독립 처리 후 **마지막에만 concat(late fusion)** → cube 사이 low-level 단서 이용을 막고 tower가 퍼즐 종류를 모른 채 범용 feature를 학습. 3D conv를 쓰는 이유(시간축 motion 포착).
- **Shortcut solution과 대응책**: chromatic aberration / cube boundary matching / low-level statistics 세 가지와 channel replication / spatio-temporal jittering / upside-down classification의 대응.
- **Ablation 해석 (Table 2)**: S 58.5, T 59.3, 앙상블 61.3, joint ST 65.8. 공간·시간 맥락은 complementary이며 joint learning > separate/ensemble. puzzle류 > reconstruction류(3D AE 48.7, inpainting 50.9)인 이유(discriminative pretext가 semantic structure를 강제).
- **Why 질문**: 왜 regression이 아니라 permutation classification인가 / 왜 jittering으로 cube를 80×80×16으로 잘라 쓰는가 / 왜 upside-down 분류를 추가하는가.

## VERIFIER REPORT
- p.58: Figure 1의 프레임 수 오기 수정 ((a')/(b')는 4장이 아니라 3프레임, (a)/(b)는 각 1장).
- p.60: 224/112/32 치수가 p.63과 "달라 보인다"는 메모를 수정 (224=전체 Width, 112=cube 크기로 p.63과 일관; 학습 입력은 80×80×16 crop).
- p.61: 정답 "A-D-B-C"가 섞인 배치와 어떻게 대응하는지 설명 bullet 추가.
- p.64: 퍼즐 vs 복원형 격차 "7~9점"을 실제 표 값에 맞게 "약 7.6~10.6점"으로 수정 (58.5-50.9=7.6, 59.3-48.7=10.6).
- 그 외 수식, 아키텍처 치수(p.62), 표 수치, 인용 논문/연도는 슬라이드와 일치 확인.
