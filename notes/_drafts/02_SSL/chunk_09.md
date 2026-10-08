<!-- DRAFT | lecture: 02_SSL | chunk: 09 | pages 65–72 -->
<!-- VERIFIED | fixes: 7 | verifier: sonnet -->

## Transfer to Action Recognition (p.65)

- 그림: Table 1 (Comparison with random initialization / fully-supervised pretraining). 행 = Initialization, 열 = UCF101(%), HMDB51(%).

| Initialization | UCF101(%) | HMDB51(%) |
|---|---|---|
| Random init. | 42.4 | 17.1 |
| 3D ST-puzzle (ours) | 65.8 | 33.7 |
| Kinetics 1/8 | 64.2 | 33.2 |
| Kinetics 1/4 | 71.1 | 41.1 |
| Kinetics 1/2 | 78.0 | 48.6 |
| Kinetics full | 84.4 | 56.4 |
| ImageNet-inflated | 60.3 | 30.7 |

- 캡션: Top-1 accuracies on UCF101 and HMDB51. All methods use 3D ResNet-18, and the accuracies are averaged over three splits.
- +23.4 points on UCF101
- +16.6 points on HMDB51
- over random initialization

**설명**
- 앞 슬라이드들의 **3D Space-Time Cubic Puzzle**(video jigsaw형 pretext task)로 pretrain한 3D ResNet-18을 **action recognition(UCF101, HMDB51)에 fine-tuning**한 downstream 결과임.
- 비교 기준: (1) **Random init.** = SSL 이점 없는 하한, (2) **Kinetics supervised pretraining** = 라벨 풀로 쓴 상한 성격 (데이터 비율 1/8 ~ full), (3) **ImageNet-inflated** = 2D ImageNet 가중치를 시간축으로 복제(inflate)해 3D conv에 넣은 것.
- 수치: 65.8 − 42.4 = **+23.4** (UCF101), 33.7 − 17.1 = **+16.6** (HMDB51) → 슬라이드 bullet과 일치함.
- 해석: **라벨 없이** 만든 representation이 ImageNet-inflated(60.3/30.7)보다 높고, Kinetics 1/8(64.2/33.2)보다도 약간 높음. 단 Kinetics full(84.4/56.4)에는 크게 못 미침 → SSL은 supervised의 완전한 대체라기보다 **라벨 없이 상당 부분을 따라잡는** 수준.
- 이 표는 SSL 평가의 표준 프로토콜인 **pretrain → fine-tune → downstream accuracy** 형태임.

> 보충: inflation은 Carreira & Zisserman, 2017 (I3D, "Quo Vadis, Action Recognition?")의 아이디어로, 2D 필터를 시간축으로 복제하고 1/T로 스케일해 3D 필터를 초기화함. 이 표는 Kim et al., CVPR 2019 ("Self-Supervised Video Representation Learning with Space-Time Cubic Puzzles")의 Table 1로 보임 (슬라이드에는 출처 명시 없음).

## Learned Spatio-Temporal Filters (p.66)

- 그림: Figure 5. conv1 layer의 학습된 64개 필터 시각화. 3개 묶음으로 구성.
  - (a) **Self-supervised 3D filters (with RGB frames)**: 상단 가로 7개 타일, 각 타일이 64개 필터 격자(8×8)로 구성. 7개 타일은 7×7×7 필터의 **시간축 7개 슬라이스**(t=1..7)에 해당. 색상 RGB 필터가 다양한 색/edge 패턴으로 나타남.
  - (b) **Kinetics-pretrained 3D filters**: 하단 가로 7개 타일, 동일 구조 (supervised 3D 필터).
  - (c) **ImageNet-pretrained 2D filters**: 오른쪽 작은 단일 8×8 격자 (2D 7×7 필터, 시간축 없음).
- 캡션: Learned filters with self-supervision vs. fully supervised-pretraining. Visualization of the learned 64 filters in conv1 layer: (a) the resulting 3D 7×7×7 filters of our self-supervised learning, (b) 3D 7×7×7 filters from the Kinetics-pretrained network, and (c) 2D 7×7 filters from ImageNet-pretrained network. Note that our representation incorporate temporal dynamics and have rich temporal structure, without requiring massive human labels.

**설명**
- 첫 conv layer 필터를 직접 보는 **qualitative 분석**임. 필터가 어떤 low-level 패턴(edge, color blob, 방향성 gabor-like)을 잡는지 확인하는 용도.
- **시간축 슬라이스마다 필터 모양이 달라짐** → 필터가 단순 정지 edge만이 아니라 **움직임(temporal change)** 도 인코딩할 수 있음을 시사함. (c) 2D ImageNet 필터는 시간축이 없어 정적 패턴만 가짐.
- (a)와 (b)가 질적으로 비슷한 구조(방향성 edge, 시간에 따라 변하는 패턴)를 가짐 → **라벨 없이도 supervised와 유사한 temporal structure를 학습**했다는 주장.
- 한계: 시각적 유사성은 정성적 증거일 뿐이고, 정량 근거는 p.65 표의 성능임.

## (Section) Cooperative Learning of Audio and Video Models from Self-Supervised Synchronization (p.67)

- 논문 제목 슬라이드: "Cooperative Learning of Audio and Video Models from Self-Supervised Synchronization" — Bruno Korbar (Dartmouth College), Du Tran (Facebook Research), Lorenzo Torresani (Dartmouth College).
- 하단 출처: Korbar et al., NeurIPS 2018.

**설명**
- 이후 p.68–70은 이 논문(**AVTS**, Audio-Visual Temporal Synchronization)의 내용임. 비디오 SSL을 **multi-modal(audio+video)** 로 확장하는 단계.

## Audio-Visual Temporal Synchronization (p.68)

- Video and audio are naturally synchronized.
  - Pretext task:
  - Are the video and audio temporally aligned?
- 그림: 상단에 3종 pair 라벨 — **Positive pair**(Video 초록, Audio 초록), **Hard negative pair**(Video 초록, Audio 파랑), **Super hard negative pair**(Video 초록, Audio 빨강). 가운데는 기타 연주 영상의 프레임 열(13개 썸네일 정도), 그 아래 오디오 waveform. 초록 박스가 비디오 클립 구간(프레임 3~5)과 그와 정렬된 waveform 구간을 표시. 점선이 비디오 구간과 오디오 구간을 수직으로 연결. 빨강 박스는 초록 구간 바로 뒤에 이어 겹치는 인접 오디오 구간, 파랑 박스는 멀리 떨어진 오디오 구간.
- In-sync pair:
  - Audio and video come from the same time segment.
- Out-of-sync pair:
  - Audio and video come from different time segments.
- What must the model understand to detect synchronization?

**설명**
- **Pretext task** = 영상 clip과 오디오 clip이 시간적으로 맞는지(in-sync) 아닌지를 **이진 분류**. 라벨은 영상 자체에서 공짜로 얻음 (같은 구간 = positive, 다른 구간 = negative).
- Negative 종류 (슬라이드 그림 기준, 색 대응: Hard = 파랑 Audio, Super hard = 빨강 Audio):
  - **Hard negative** (파랑 박스): 같은 비디오에서 **멀리 떨어진(겹치지 않는)** 다른 시간 구간의 오디오. 그림에서 파랑 박스는 초록 구간과 멀리 떨어져 있음.
  - **Super hard negative** (빨강 박스): 같은 비디오에서 초록 구간 **바로 뒤에 이어지는 인접 구간**(시간 shift만 다름)의 오디오. 소리 내용이 거의 같아 가장 구분이 어려움.
  - (비교용) **Easy negative** = 다른 비디오의 오디오 (p.70에서 "easy"로 등장).
- 마지막 질문 "What must the model understand?"에 대한 답: 단순한 장면 종류가 아니라 **소리와 동작의 시간적 대응(예: 손이 현을 튕기는 순간과 소리 발생 시점)** 을 이해해야 함 → 의미적 + 시간적 수준의 audio-visual correspondence를 학습하게 됨.

> 보충: AVTS의 negative 정의 (Korbar et al., 2018): easy = 다른 비디오의 오디오, hard = 같은 비디오의 **겹치지 않는** 다른 구간, super-hard = 같은 비디오에서 **부분적으로 겹치는** (수 초 이내 shift) 구간. 위 해석은 이 정의와 부합함.

## Cooperative Audio-Video Learning (p.69)

- 그림 (Figure 2): 세 부분.
  - a) 전체 구조: Input video → **N video frames**, 그리고 **Audio filter banks**(스펙트로그램 형태 이미지). 프레임 입력은 **Video subnetwork**(왼쪽 파란 세로 박스), 오디오 입력은 **Audio subnetwork**(오른쪽 초록 세로 박스)로 들어가고, 두 출력이 함께 하단 **objective: contrastive loss**(분홍 박스)로 연결됨.
  - b) Video subnetwork 상세 (파란 블록, 위→아래): Input video → conv_1 (3x7x7, 3, 64) → conv_2_1 (3x3x3, 64, 64) → conv_2_2 (3x3x3, 64, 64) → conv_3_1 (1x3x3, 64, 128) → conv_3_2 (1x3x3, 128, 128) → conv_4_1 (1x3x3, 128, 256) → conv_4_2 (1x3x3, 256, 256) → conv_5_1 (1x3x3, 256, 512) → conv_5_2 (1x3x3, 512, 512) → Avg Pool → fc6 (512, final_dim).
  - c) Audio subnetwork 상세 (초록 블록, 위→아래): Input audio → conv_1 (3x3, 1, 96) → conv_2 (3x3, 96, 256) → conv_3 (3x3, 256, 512) → conv_4 (3x3, 512, 512) → conv_5 (3x3, 512, 512) → pool_5 (3x3, MaxPool) → fc6 (512*11*13, 4096) → fc7 (4096, final_dim).
- Video stream
  - Processes motion and appearance
  - Uses a spatio-temporal CNN
- Audio stream
  - Processes audio filterbank features
  - Learns sound representations
- 캡션: Figure 2: Our architecture design. The complete model for AVTS training can be viewed in (a). The video subnetwork (shown in (b)) is a MCx network [19] using 3D convolutions in the early layers, and 2D convolutions in the subsequent layers. The audio subnetwork (shown in (c)) is the VGG model used by Chung and Zisserman [26].

**설명**
- **Two-stream(dual-encoder) 구조**: 비디오와 오디오를 **각자의 encoder**로 임베딩해 같은 차원(final_dim)의 벡터로 만든 뒤, **contrastive loss**로 in-sync pair는 가깝게, out-of-sync pair는 멀게 학습함. 두 modality가 서로의 supervision이 되므로 "cooperative"라 부름.
- Video subnetwork: **MCx**(Mixed Convolution, Tran et al., 2018 "A Closer Look at Spatiotemporal Convolutions")로, 초기 layer(conv_1, conv_2)는 **3D conv**(시간축 커널 3)로 motion을 잡고, 이후 layer(conv_3~5)는 **2D conv(1x3x3)** 로 appearance를 처리함. 이유: 시간 정보는 초기 low-level에서 중요하고, 후반부는 2D로 해도 충분하면서 파라미터/연산이 절약됨.
- Audio subnetwork: 오디오를 **filterbank(스펙트로그램류 2D 표현)** 로 바꿔 이미지처럼 2D CNN(**VGG형**, Chung & Zisserman)으로 처리함. 최종 fc6 입력이 512*11*13인 것은 마지막 feature map 크기를 의미.
- 두 stream 모두 마지막에 같은 final_dim으로 projection → 임베딩 공간에서 거리 비교 가능.

> 보충: 논문에서는 contrastive loss의 형태로 $L = \frac{1}{2N}\sum_n \big[ y_n d_n^2 + (1-y_n)\max(m - d_n, 0)^2\big]$ ($d_n = \|f_v - f_a\|_2$, $y_n=1$이면 in-sync, $m$은 margin)를 쓰는 것으로 알려져 있음 (슬라이드에는 수식 없음).

## The Choice of Negative Pairs Matters (p.70)

- 부제: Curriculum Learning
- Table 1: AVTS accuracy achieved by our system on the Kinetics test set, which includes negatives of only "easy" type, as in [21]. The table shows that curriculum learning with a mix of easy and hard negatives in a second stage of training leads to a significant gain in accuracy.

| Method | Negative type | Epochs | Accuracy (%) |
|---|---|---|---|
| Single learning stage | easy | 1 - 90 | 69.0 |
| Single learning stage | 75% easy, 25% hard | 1 - 90 | 58.9 |
| Single learning stage | hard | 1 - 90 | 52.3 |
| Single learning stage | easy | 1 - 50 | 67.2 |
| Curriculum learning (i.e., second learning stage applied after a first stage of 1-50 epochs with easy negatives only) | 75% easy, 25% hard | 51 - 90 | **78.4** |
| Curriculum learning | hard | 51 - 90 | 65.7 |

- A good pretext task depends not only on the labels, but also on how positive and negative examples are constructed.

**설명**
- 핵심: **negative를 어떻게 뽑느냐**가 성능을 크게 좌우함. 평가는 easy negative만 있는 Kinetics test set 기준.
- Single stage에서 **처음부터 hard negative를 섞으면 오히려 성능 하락** (easy 69.0 → 75/25 mix 58.9 → hard only 52.3). 가능한 이유: 초기 학습에서는 표현이 아직 약해 hard negative가 너무 어려울 수 있고(슬라이드에 명시된 설명은 아님), 또 평가셋이 easy negative만 포함하므로 hard를 섞으면 train/test negative 분포가 어긋나는 영향도 있음.
- **Curriculum learning**: 먼저 1–50 epoch 동안 easy negative만으로 기본 correspondence를 학습(67.2) → 이후 51–90 epoch에 hard를 섞어(75% easy + 25% hard) fine-tune → **78.4%** 로 단일 stage 최고(69.0)보다 +9.4 향상. hard only로 second stage를 하면 65.7로 mix보다 낮음 → **easy를 일부 유지하는 mix가 중요**.
- 의의: 쉬운 것부터 어려운 것 순으로 학습하는 **curriculum** 아이디어 + hard negative mining의 효과를 보여줌. 마지막 문장이 요약: pretext task의 품질은 라벨(정의)뿐 아니라 **sample 구성 방식**에도 달려 있음.
- 시험용 주의: 표의 69.0(easy 전체 90 epoch)과 단일 stage 중 "easy 1–50" 67.2는 서로 다른 행이며, 후자는 curriculum의 first stage 결과에 해당함.

## (Section) Self-supervised Transformation Learning for Equivariant Representations (p.71)

- 논문 제목 슬라이드: "Self-supervised Transformation Learning for Equivariant Representations" — Jaemyung Yu, Jaehyun Choi, Dong-Jae Lee, HyeongGwon Hong, Junmo Kim (KAIST).
- 하단 출처: Yu et al., NeurIPS 2024.

**설명**
- 다음 슬라이드(p.72)의 "pretext task가 어떤 정보를 보존하는가" 관점과 연결되는 논문. **equivariant representation**(변환에 따라 표현이 예측 가능하게 변하는 표현)을 self-supervised로 학습하는 주제임.

> 보충: invariance는 변환 $t$에 대해 $f(t(x)) = f(x)$, equivariance는 $f(t(x)) = T_t(f(x))$ (표현 공간에서의 대응 변환). 이 논문은 변환 라벨 없이 변환 자체를 표현 공간에서 학습하는 방법으로 알려져 있음 (정확한 방법은 슬라이드에 없음).

## What Do Pretext Tasks Preserve? (p.72)

- Rotation Prediction
  - The model must preserve rotation information.
- Colorization
  - The model must preserve object and color information.
- Jigsaw Puzzle
  - The model must preserve spatial relationships.
- A pretext task determines what information is preserved in the representation.

**설명**
- 핵심 메시지: **pretext task = representation에 어떤 정보가 남을지를 결정하는 설계 선택**임.
- Rotation prediction (Gidaris et al., 2018): 회전 각도를 맞히려면 물체의 **방향/자세(orientation)** 를 알아야 하므로 회전 정보가 보존됨 (회전에 invariant하면 안 됨).
- Colorization (Zhang et al., 2016): gray → color를 예측하려면 **어떤 물체인지(semantic)와 그 색**을 알아야 하므로 object·color 정보가 보존됨.
- Jigsaw puzzle (Noroozi & Favaro, 2016): patch 순서를 맞히려면 **patch 간 공간 배치 관계**를 알아야 함.
- 대비: **contrastive/invariance 기반** 방법은 augmentation(회전, 색 변화 등)에 invariant해지도록 하면 해당 변환 정보가 **버려질 수 있음**. 반면 rotation prediction처럼 변환 자체를 맞히는 pretext task는 그 변환 정보를 **보존**하도록 강제함(엄밀한 equivariance와는 구분; colorization·jigsaw는 변환 예측이 아님) → p.71 논문(equivariant representations)과 연결되는 맥락(슬라이드가 명시한 연결은 아니고 해석임).
- 시사점: downstream task에 필요한 정보와 pretext task가 보존하는 정보가 맞아야 transfer가 잘 됨.

## FIGURES

- p.65: Table 1, 3D ST-puzzle vs random / Kinetics / ImageNet-inflated의 UCF101·HMDB51 비교 (중요도 상)
- p.66: Figure 5, self-supervised 3D vs Kinetics 3D vs ImageNet 2D conv1 필터 시각화 (중요도 중)
- p.68: audio-visual sync pretext, positive / hard / super-hard negative 구성 다이어그램 (중요도 상)
- p.69: AVTS two-stream 구조 (video MCx + audio VGG → contrastive loss) (중요도 상)
- p.70: negative 유형·curriculum 학습 결과 표 (중요도 상)

## EXAM POINTS

1. **Pretext task 정의 (AVTS)**: 비디오와 오디오가 시간적으로 정렬되었는지 이진 판별. in-sync = 같은 시간 구간, out-of-sync = 다른 구간. 라벨은 영상에서 공짜로 얻음.
2. **Negative 종류**: easy(다른 비디오) / hard(같은 비디오의 겹치지 않는 먼 구간) / super-hard(같은 비디오의 인접·부분 겹침 구간). 왜 hard negative가 필요한가 → 단순한 장면 인식이 아니라 **시간적 sync 자체를 학습**하게 만듦.
3. **Curriculum learning 결과**: single stage에서 hard를 처음부터 쓰면 성능 하락(69.0 → 58.9 → 52.3), easy 50 epoch 후 75% easy + 25% hard로 전환하면 **78.4%**. why: 초기 표현이 약할 때 hard negative는 너무 어려워 학습 방해.
4. **AVTS 구조**: Video = MCx(초기 3D conv + 후반 2D conv), Audio = VGG 형 2D CNN on filterbank, 두 임베딩을 contrastive loss로 학습. 왜 mixed 3D/2D인가 (motion은 초기, appearance는 후반 2D로 효율).
5. **SSL 평가 프로토콜**: 3D ST-puzzle pretrain 후 fine-tune 시 random init 대비 UCF101 +23.4, HMDB51 +16.6; ImageNet-inflated보다 높고 Kinetics full보다는 낮음 (supervised 상한 대비 위치).
6. **Learned filter 비교**: self-supervised 3D 필터도 Kinetics supervised 필터처럼 **temporal structure**를 가짐 (2D ImageNet 필터는 시간축 없음).
7. **"Pretext task가 보존하는 정보"**: rotation → 회전/방향, colorization → object+color, jigsaw → spatial relationship. pretext 설계가 representation에 담길 정보를 결정함 (invariance 방식과 대비되는 equivariance 관점).
8. **좋은 pretext의 조건**: 라벨 정의뿐 아니라 positive/negative **샘플 구성 방식**에 의존한다는 p.70의 결론.

## VERIFIER REPORT

- p.68 설명: Hard/Super hard negative 설명이 자기모순적이어서 그림 기준으로 재작성 (Hard=파랑, 멀리 떨어진 구간 / Super hard=빨강, 초록 바로 뒤 인접 구간).
- EXAM POINTS 2번: hard를 "겹치지 않는 먼 구간", super-hard를 "인접·부분 겹침"으로 명확화.
- p.70 설명: hard 혼합 시 성능 하락 이유를 단정하던 부분을 추측으로 완화하고, 평가셋이 easy-only라는 train/test 분포 불일치 요인 추가.
- p.72 설명: "무엇인 물체인지" 오타 수정; pretext = equivariant로 일반화한 과장 완화 (colorization/jigsaw는 변환 예측이 아님, p.71 연결은 해석임을 명시).
- p.69: "두 출력이 합쳐져"를 "함께 loss로 입력"으로 수정 (fusion이 아니라 contrastive loss 입력).
- p.66 설명: "움직임을 인코딩함" 단정을 "시사함"으로 완화.
- 표 수치(p.65, p.70), 아키텍처 layer 사양(p.69), 저자/출처/연도(p.67, p.71), 인용(I3D 2017, Kim 2019, MCx 2018, Gidaris 2018, Zhang 2016, Noroozi 2016)은 원본과 일치, 수정 없음.
