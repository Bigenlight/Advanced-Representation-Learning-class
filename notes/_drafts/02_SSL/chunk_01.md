<!-- DRAFT | lecture: 02_SSL | chunk: 01 | pages 1–8 -->
<!-- VERIFIED | fixes: 3 | verifier: sonnet -->

## 표지: Advanced Representation Learning (고급표현학습) (p.1)

- 제목: Advanced Representation Learning (고급표현학습)
- Instructor: Donghyeon Cho / The Department of Computer Science / Hanyang University
- 그림: 파란 가로선 3개로 구분된 표지. 좌하단 한양대학교 로고, 우하단 한양대 인장(1939)과 연구실 로고(카메라 모양 + V)

**설명**
- 강의 전체 표지 슬라이드이고 02_SSL 강의(Self-Supervised Learning)의 시작 페이지임
- 내용 없음, 강의 소개용

## Contents of Table (p.2)

- 1. Why Self-Supervised Learning?
- 2. What is Self-Supervised Learning?
- 3. Pretext Tasks
  - Relative Position of Patches
  - Inpainting
  - Jigsaw Puzzle
  - Colorization
  - Rotation Prediction
- 4. Summary of Pretext Tasks
- 5. Toward Contrastive Learning

**설명**
- 이 강의의 흐름: **동기(왜 SSL인가) → 정의 → 대표 pretext task 5종 → 요약 → contrastive learning으로의 연결**
- 핵심 논리 구조는 "supervised의 한계 → label 없이 representation을 배우는 SSL → 사람이 설계한 pretext task의 한계 → contrastive learning으로 발전"임
- Pretext task 5종은 모두 **이미지 자체에서 label을 자동 생성**하는 방식임 (patch 상대 위치, 가려진 영역 복원, 퍼즐 순서, 흑백→컬러, 회전 각도)

> 보충: 각 task의 대표 논문은 Doersch et al. 2015 (Relative Position), Pathak et al. 2016 (Context Encoders, inpainting), Noroozi & Favaro 2016 (Jigsaw), Zhang et al. 2016 (Colorful Image Colorization), Gidaris et al. 2018 (RotNet). 슬라이드 본문에 나오면 그때 다시 정리할 것

## 섹션 구분: Why Self-Supervised Learning? (p.3)

- 섹션 구분 슬라이드 (남색 배경, "Why Self-Supervised Learning?")

## Success Story of Supervision: Pre-training (p.4)

- Features from networks pre-trained on ImageNet can be used for a variety of different downstream tasks.
- 그림: 왼쪽부터 오른쪽으로 흐르는 파이프라인 다이어그램
  - 왼쪽: 격자(7행 x 6열)로 배열된 ImageNet 이미지 썸네일들 + "..." / 캡션 "Images from ImageNet (Pre-train)"
  - 화살표 → ConvNet: 점점 spatial 크기가 작아지고 channel(두께)이 늘어나는 직육면체 블록들(conv layer들)이 이어지고, 끝에 FC layer를 나타내는 세로 점 열 2개
  - 화살표 → 출력 박스: 클래스 label 막대 "husky", "terrier", "tench", "..."
  - 하단: ConvNet 전체 아래에 큰 중괄호 + "Learn a representation" 문구, 뇌 아이콘 (양옆에 "...")

**설명**
- **Supervised pre-training**의 성공 사례를 먼저 상기시키는 슬라이드. SSL의 동기를 만들기 위해 "supervised가 얼마나 잘 되었는가"를 먼저 보여줌
- ImageNet 1000-class classification으로 ConvNet을 학습하면 마지막 classifier뿐 아니라 **중간 feature(representation)** 도 범용적으로 쓸 수 있음. 즉 classification은 목적이 아니라 representation을 얻는 수단임
- 학습된 feature는 object detection, segmentation 등 **downstream task**에 transfer(fine-tuning 또는 linear probing)되어 성능을 크게 올림
- 중괄호는 이미지 입력부터 ConvNet(및 출력)까지 파이프라인 전체를 묶어 "Learn a representation"이라 표기함. 의도는 label 예측 자체가 아니라 **ConvNet이 학습하는 feature가 representation**이라는 점임 (중괄호가 backbone만 가리키는 것은 아님)

> 보충: ImageNet pre-training의 효과는 Donahue et al. 2014 (DeCAF), Girshick et al. 2014 (R-CNN)에서 대표적으로 확인됨. downstream task의 label이 적을 때 특히 효과가 큼

## Success Story of Supervision: Recipe for Good Solutions (p.5)

- Pre-train on a large supervised dataset.
- Collect a dataset of "supervised" images
- Train a ConvNet

**설명**
- Supervised 방식의 성공 **recipe**: (1) 대규모 labeled dataset 수집 (2) 그 위에서 ConvNet 학습 (3) 학습된 모델을 pre-train 모델로 사용
- 슬라이드의 bullet 순서가 "Pre-train → Collect → Train"으로 실제 절차 순서와 뒤섞여 있음 ← 순서가 논리적으로 어색함 (원문 유지, 실제 순서는 Collect → Train → 이후 Pre-train 모델로 사용)
- 이 recipe의 병목은 **"supervised" images를 모으는 단계(= labeling)** 이고, 다음 슬라이드(p.6)부터 이 한계를 다룸. 따옴표("supervised")는 슬라이드 표기이며, 의도는 슬라이드에 설명되어 있지 않음 (label이 달린 이미지라는 뜻으로 읽힘)

## Limitations of Supervised Learning: Challenges in Labeling (p.6)

- Getting "real" labels is difficult and expensive
  - ImageNet with 14M images took 22 human years.
- Obtain labels using a "semi-automatic" process
  - Hashtags
  - GPS locations

**설명**
- 사람이 직접 달아주는 정확한 label은 **비싸고 어려움**. 근거: ImageNet 14M장 구축에 **22 human years** 소요 (슬라이드 수치)
- 대안으로 **semi-automatic** label 확보: 인터넷 데이터에 이미 붙어 있는 **hashtag**(Instagram 등), **GPS 위치**를 label로 사용. 사람 노동은 줄지만 label이 **noisy**하고 개념이 제한적임
- 이 두 방식도 근본적 해결책은 아님: 결국 "누군가 의미를 부여한 신호"에 의존하기 때문. 이것이 label을 아예 쓰지 않는 SSL로 가는 동기가 됨

> 보충: hashtag 기반 weakly supervised pre-training은 Mahajan et al. 2018 (Exploring the Limits of Weakly Supervised Pretraining, Facebook, 3.5B Instagram images), GPS 기반은 Weyand et al. 2016 (PlaNet) 등이 있음. 슬라이드는 논문을 명시하지 않았음. ImageNet 22 human years 수치의 출처는 슬라이드에 없으며, 정확한 출처는 확인하지 못함 (ImageNet 자체는 Deng et al. 2009)

## Can We Get Labels for All Data? (p.7)

- 제목: Can We Get Labels for All Data?
- 그림: 로그 스케일 막대 그래프 + 4단계 화살표(① → ② → ③ → ④)
  - y축: "Number of Unique Labels (Possible Concepts)", 로그 스케일 1E+00 ~ 1E+12 (눈금 1E+02 간격)
  - x축 4개 category와 상단 설명:
    1. **Bounding Boxes** — Object detection. 예시 이미지(개, 의자, 피자, 도넛에 색깔 박스) / 캡션 "Dog, chair, pizza, donut". 막대(회색)는 약 1E+06 부근(그보다 약간 아래)
    2. **Image Level** — Image classification. 같은 이미지에 박스 없음 / 캡션 "Dog, chair, pizza, donut". 막대(청록)는 약 1E+06~1E+07 사이로 1번보다 약간 높음
    3. **Internet Photos** — Webly crawled images. "Billions of objects, scenes, events, activities, ..." 막대(파랑)는 약 1E+09 부근
    4. **Real World** — Videos, sensors, text, etc. 여러 사진 격자 + "Unlimited concepts in the real world". 막대(진한 남색)는 약 1E+10 부근
  - 하단 화살표: 왼쪽 "Easier (More labels)" → 오른쪽 "Harder (Fewer labels)" (그라데이션 청록→남색)

**설명**
- 질문: "모든 데이터에 label을 얻을 수 있는가?" 답은 **아니오**
- 왼쪽에서 오른쪽으로 갈수록 데이터는 **현실에 가까워지고, 표현해야 할 concept 수는 폭증**함: box annotation(~1e6 규모) → image-level label → 웹 crawling 이미지(수십억 개 concept) → 실세계(영상, 센서, 텍스트 등, 사실상 무한 concept)
- 반면 label을 확보하기는 오른쪽으로 갈수록 **어려워짐** (하단 화살표). Bounding box는 가장 비싼 label인데도 그나마 가장 "쉬운" 쪽으로 분류되어 있음. 즉 **annotation 비용 대비 커버 가능한 concept의 한계**를 보이는 그림
- 슬라이드의 y축은 "label 개수"가 아니라 **possible concept 수**이므로 헷갈리지 말 것: 오른쪽으로 갈수록 막대가 커지는 것은 label이 많다는 뜻이 아니라 **라벨링해야 할 개념이 많다는 뜻**임
- 하단 화살표 "Easier (More labels)"와 막대 높이 증가 방향이 얼핏 모순돼 보임 ← 막대는 "필요한 concept 수", 화살표는 "실제 확보 가능한 label 양"으로 서로 다른 축임. 슬라이드 표현이 혼동을 유발할 수 있음
- 막대 높이의 정확한 값은 눈대중 추정이며 슬라이드에 수치는 없음

## Can We Get Labels for All Data? (p.8)

- What about complex concepts?
  - Video?
- Labelling cannot scale to the size of the data we generate

**설명**
- p.7의 결론 정리: 사물 이름 수준을 넘는 **복잡한 concept**(행동, 사건, 시간적 맥락이 있는 **video** 등)은 label을 정의하는 것부터 어렵고 비용이 큼
- 핵심 주장: **labelling은 우리가 생성하는 데이터 규모에 비례해 확장되지 않음 (does not scale)**. 데이터 생성 속도 >> annotation 속도
- 따라서 **label 없는 대량의 데이터에서 직접 representation을 배우는 방법**이 필요하고, 이것이 SSL의 존재 이유가 됨 (다음 섹션 "What is Self-Supervised Learning?"으로 연결)
- 참고: 다음 페이지 p.9는 "Rare Concepts?" 제목의 슬라이드로 같은 논지의 연장임 (이 chunk 범위 밖)

## FIGURES

- p.4: ImageNet pre-training 파이프라인 (이미지 → ConvNet → class label, "Learn a representation" 중괄호), 중요도 상
- p.7: 로그 스케일 막대 그래프, label 확보 난이도 vs concept 수 (Bounding Box / Image-level / Internet Photos / Real World), 중요도 상
- p.2: 강의 목차 (구조 파악용), 중요도 중

## EXAM POINTS

1. **Supervised pre-training의 recipe**: 대규모 labeled dataset(ImageNet) 수집 → ConvNet 학습 → 학습된 feature를 downstream task에 transfer. 이때 학습되는 것은 classifier가 아니라 **representation**임
2. **Supervised learning의 한계 (why SSL?)**: label 확보가 어렵고 비쌈 (ImageNet 14M장 = 22 human years), semi-automatic label(hashtag, GPS)은 noisy하고 제한적임
3. **Labelling scalability 문제**: 복잡한 concept(video 등)은 label 정의 자체가 어렵고, label 양이 데이터 생성 속도를 따라가지 못함 → label-free representation learning 필요
4. p.7 그래프 해석: 오른쪽으로 갈수록 필요 concept 수는 증가하고 label 확보는 어려워짐 (Bounding Box < Image Level < Internet Photos < Real World)
5. Semi-automatic labeling의 예(hashtag, GPS)와 그 한계를 서술하는 문제 (왜 근본 해결책이 아닌가)
6. 강의 구성 흐름: Why SSL → What is SSL → Pretext tasks 5종 (Relative Position, Inpainting, Jigsaw, Colorization, Rotation) → Summary → Toward Contrastive Learning

## VERIFIER REPORT

- p.4 설명: 중괄호가 "conv backbone 전체"만 가리킨다는 서술을 수정 (실제로는 이미지 입력~ConvNet~출력 전체 파이프라인을 묶음)
- p.6 보충: ImageNet "22 human years" 출처를 Deng 2009 / Russakovsky 2015로 단정한 부분을 "출처 불명"으로 수정
- p.5 설명: 따옴표("supervised")의 의도를 추측해 서술한 부분을 슬라이드에 근거 없음으로 수정
- 누락 페이지/수식/그림 묘사(p.1, 2, 4, 7 수치·색·축)·인용 논문명/연도는 원본과 일치 확인
