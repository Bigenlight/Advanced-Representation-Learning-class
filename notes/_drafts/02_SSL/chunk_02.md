<!-- DRAFT | lecture: 02_SSL | chunk: 02 | pages 9–16 -->
<!-- VERIFIED | fixes: 2 | verifier: sonnet -->

## Rare Concepts? (p.9)

- 제목: "Rare Concepts?"
- 그림: "Objects in Vision Dataset (LabelMe)" 제목의 log-log plot.
  - x축: Rank (log scale), 10^0 ~ 10^4. y축: Frequency (log scale), 10^0 ~ 10^4.
  - 각 점 = 하나의 class, 빈도가 큰 순서로 정렬 (범례: "Each point is a class sorted by decreasing frequency").
  - 맨 왼쪽 위(rank 1, 빈도 약 10^4)에 원으로 강조된 점, 오른쪽 아래(rank 약 4×10^3, 빈도 10^0 = 1)에 원으로 강조된 점. 두 점을 붉은 점선(수평/수직)으로 연결해 box를 그림.
  - 곡선은 거의 직선처럼 우하향하다가 rank 약 10^3 이후 빈도 1에서 평평한 꼬리(plateau)를 이룸.
- 우측 callout 박스: **10% of the classes account for 93% of the data**

**설명**
- 앞 슬라이드(p.8, "Can We Get Labels for All Data?")의 연장선: label을 전부 달 수 없는 이유 중 하나가 **long-tail 분포**라는 점을 보여줌.
- 자연 데이터에서 class 빈도는 **power law (Zipf-like)** 를 따름. log-log plot에서 직선에 가까운 건 이 때문임: $f(r) \propto r^{-\alpha}$ → $\log f = -\alpha \log r + c$.
- 상위 10% class가 데이터의 93%를 차지함 → 나머지 90% class는 합쳐도 7%뿐인 **rare concepts**. 많은 class는 샘플이 1개 수준(빈도 10^0)임.
- 함의: supervised learning은 class당 충분한 labeled sample이 필요한데, rare concept은 sample 수집 자체가 어렵고 annotation 비용 대비 효율이 낮음. 그래서 label 없이 data 구조에서 표현을 배우는 SSL이 필요하다는 동기 부여.
- 데이터셋 이름 LabelMe는 객체 단위 annotation 웹 도구/데이터셋임.

> 보충: LabelMe (Russell et al., 2008)는 MIT CSAIL의 online annotation 데이터셋. long-tail 문제는 이후 long-tailed recognition (e.g., LVIS 데이터셋)에서 본격적으로 다뤄짐.

## Different Domains? (p.10)

- 제목: "Different Domains?"
- 그림: 어두운 배경의 라이트박스에 걸린 뇌 MRI 영상 필름 여러 장 (axial / sagittal 단면, 각 영상 옆에 "Sc 14", "FFE/M", "SI 11~16", "GD", "FH -26 feet" 같은 촬영 파라미터 텍스트). 슬라이드에 텍스트 bullet은 없음.

**설명**
- 의료영상 같은 **specialized domain**은 ImageNet식 natural image와 분포가 크게 다르고, 라벨링에 전문가(의사)가 필요해 비용이 매우 큼. 그래서 domain마다 대규모 labeled set을 만들 수 없음.
- 즉 p.8~10의 흐름: (1) 복잡한 개념/video는 label이 안 따라옴, (2) rare concept은 sample이 부족, (3) 도메인이 다르면 처음부터 다시 labeling해야 함 → **supervised labeling의 scalability 한계**.
- 그림이 MRI인 이유: 전문가 annotation이 필수이면서 unlabeled 데이터는 병원에 대량으로 쌓여 있는 대표 사례라서 SSL의 동기가 큼.

## What is Self-Supervised Learning? (p.11)

- 섹션 구분 슬라이드: "What is Self-Supervised Learning?"

**설명**
- 이후 p.12~16에서 SSL의 정의, supervised/unsupervised와의 비교, NLP 성공 사례, 동기, paradigm(pretext/downstream)을 다룸.

## What is "Self" Supervision? (p.12)

- Obtain "labels" from the data itself by using a "semi-automatic" process.
  - Predict part of the data from other parts.
- 그림: 4단계 흐름도 (왼→오 화살표).
  1. **Observed Data**: 3×3 격자 형태의 input data (e.g., image, audio, text).
  2. **Create Pseudo Labels (Semi-Automatic)**: 같은 3×3 격자에서 일부 칸이 비워진/점선 처리(masked parts). 캡션 "Pseudo labels (e.g., masked parts)", 옆 문구 "Automatically define a prediction target from the data itself".
  3. **Predict**: 입력 3 노드 – 은닉 2 노드 – 출력 3 노드 모양의 신경망 그림 (Model).
  4. **Learn Representation**: 2D 축 위에 초록/파랑/보라 점들이 색별로 군집된 scatter plot. 캡션 "Learn useful representations".

**설명**
- **SSL의 핵심 정의**: 사람이 label을 주는 대신, data 자체의 일부를 가리고(or 변형하고) 나머지로부터 그 일부를 예측하게 하여 **supervision signal을 data에서 자동 생성**함.
- "semi-automatic"인 이유: label 자체는 자동으로 얻지만, **어떤 부분을 가리고 어떤 변형을 줄지는 사람이 설계**하기 때문(pretext task 설계는 사람 몫).
- 흐름: 관측 데이터 → pseudo label 생성 → 모델이 예측 → 예측을 잘하려다 보니 **유용한 representation**이 학습됨. 목적은 예측 자체가 아니라 representation임 (p.16의 pretext vs downstream과 연결).
- 4단계 그림 중 2단계 pseudo label이 "masked parts"인 것은 MAE/BERT류 masked prediction을 암시함.

## What is "Self" Supervision? — Supervised vs Unsupervised vs Self-Supervised (p.13)

- 제목은 p.12와 동일 ("What is "Self" Supervision?").
- 그림: 세 열 비교.
  - **1) Supervised Learning**: Input(image, 소 사진) → Human label "COW" → Model → Representation(막대형 벡터).
  - **2) Unsupervised Learning**: Input(image) → Model → Representation. 하단 점선 박스 "Discover structure in data (clustering, density estimation, dimensionality reduction)".
  - **3) Self-Supervised Learning**: Input(image, 소 사진) ↔ "Another modality / augmented view" (소리 파형, "moo") 양방향 화살표 → "Self-generated supervision (e.g., predict other modality, match views, predict missing parts)" 박스 → Model → Representation. 오른쪽 "Examples": **MAE** (masked image modeling, 일부 패치가 검은 격자 그림), **SimCLR** (contrastive views, 두 이미지가 양방향 화살표로 연결), **CLIP** (image-text pairs, "A cow in a field" 텍스트 박스).
- 비교 표:

| | Supervised | Unsupervised | Self-Supervised |
|---|---|---|---|
| Labels | Human provided (expensive) | No labels | Automatically generated from data |
| Source of supervision | Human annotations | None | Data itself (pretext tasks) |
| Cost | High | Low | Low |
| Scalability | Limited | High | Very high |
| Learned representations | Task-specific | Limited semantic meaning | Rich, transferable representations |

- 하단 박스: **Self-supervised learning creates supervision signals directly from the data itself, enabling scalable learning of meaningful representations.**

**설명**
- **Supervised**: (x, y) 쌍에서 y는 사람 annotation. 비용이 크고 확장성이 낮으며, 표현이 해당 task에 특화(task-specific)됨.
- **Unsupervised**: label 없이 data의 구조(clustering, density estimation, dimensionality reduction)를 찾음. 싸고 확장성은 좋지만 얻는 표현의 semantic 수준이 제한적임 (예: PCA, k-means).
- **Self-supervised**: label은 없지만 **data 자체에서 supervision을 만들어** supervised 형태의 objective(예측/분류/회귀/contrastive)로 학습. unsupervised의 확장성 + supervised의 강한 학습 신호 둘 다 얻으려는 것.
- 그림 3번 열의 핵심: supervision이 **다른 modality(이미지↔소리/텍스트)**, **augmented view**, **가려진 부분** 중 하나에서 옴. 예시 3개가 각각 한 유형에 대응: MAE=missing parts 예측, SimCLR=view 간 매칭, CLIP=modality 간(image-text) 매칭.
- 구분 포인트: unsupervised와 SSL은 둘 다 사람 label이 없지만, SSL은 **pretext task라는 명시적 target**이 있다는 점이 다름.

> 보충: MAE = Masked Autoencoders (He et al., 2022), SimCLR (Chen et al., 2020), CLIP (Radford et al., 2021). 슬라이드에는 이름과 도식만 있고 세부 방법은 이후 강의에서 다룸.

## Success of Self-Supervised Learning in NLP (p.14)

- Ex. Word2vec
- Fill in the blanks is a powerful signal to learn representations
- Sentence/Word representations: BERT - Devlin et al., 2018
- 그림: Word2vec 구조도 (context 합산 → softmax 형태).
  - 맨 아래 **Projection layer**: "Sum (or average) of word embeddings in the context". 단어열 "the cat sits on the" (context / history $h$) 와 빨간 박스 "mat" (target $w_t$).
  - 가운데 **Hidden layer**: "Learned representation (embedding of context)" — 파란 점 5개짜리 작은 박스. 아래 projection 층에서 이 박스로 수렴하는 삼각형 위에 $\sum g(\text{embeddings})$ 표기.
  - 맨 위 **Softmax classifier**: "Predict probability of each word in the vocabulary", 노드 $w_1, w_2, w_t, \dots, w_V$. hidden에서 위로 퍼지는 역삼각형.
  - 빨간 화살표: target "mat"에서 올라가 softmax의 $w_t$로 연결, 라벨 "predict nearby word $w_t$". 범례: 파랑 = Forward computation, 빨강 = Prediction / target flow.
  - 하단 설명 박스: "Given a context (the history $h$: 'the cat sits on the'), predict the nearby word $w_t$ ('mat'). The model learns word embeddings so that useful words occur in similar contexts."

**설명**
- NLP에서 SSL이 성공한 대표 사례. **fill in the blanks (주변 문맥으로 빈 단어 예측)** 가 label 없이도 강력한 학습 신호가 됨. 문장 자체가 label(정답 단어)을 제공하므로 annotation이 필요 없음.
- Word2vec 계열(슬라이드 도식은 앞쪽 history만 쓰는 형태라 엄밀한 CBOW와는 다름): context 단어들의 embedding을 합/평균(= $\sum g(\cdot)$, 여기서 $g$는 embedding lookup/결합 함수)해서 hidden representation $c$를 만들고, vocabulary 크기 $V$의 softmax로 target 단어를 예측:
  $$P(w_t \mid h) = \frac{\exp(\mathbf{u}_{w_t}^\top \mathbf{c})}{\sum_{w'=1}^{V} \exp(\mathbf{u}_{w'}^\top \mathbf{c})}, \quad \mathbf{c}=\sum_{w\in h} g(\mathbf{e}_w)$$
  (위 식은 슬라이드 도식을 풀어 쓴 것이며 슬라이드에는 수식이 없음.)
- 결과적으로 **비슷한 context에 나타나는 단어는 비슷한 embedding**을 갖게 됨 (distributional hypothesis). 예측 task는 부산물이고, 학습된 embedding이 목적임.
- BERT (Devlin et al., 2018)는 이를 Transformer로 확장: 문장 내 일부 token을 [MASK]로 가리고 복원하는 masked language modeling → 문맥 의존적 word/sentence representation. 이후 vision의 MAE가 이 아이디어를 이미지로 옮긴 것.
- 슬라이드가 "Sentence/Word representations: BERT"라고만 적었고 pretraining 이후 downstream task에 fine-tuning한다는 점은 p.16 paradigm과 연결됨.

> 보충: Word2vec = Mikolov et al., 2013 (CBOW와 Skip-gram 두 변형). 슬라이드 그림은 "history h"("the cat sits on the") 다음 단어 "mat"을 예측하므로 CBOW(좌우 양쪽 context로 중심 단어 예측)보다는 next-word(language-model식) 예측에 가깝고, context 합산 → softmax 구조만 CBOW와 공통임. 실제 Word2vec은 full softmax의 비용을 피하려고 hierarchical softmax / negative sampling을 씀.

## Why Self Supervision? (p.15)

- Helps us learn using observations and interactions
- Does not require exhaustive annotation of concepts
- Leverage multiple modalities or structure in the domain

**설명**
- **observations and interactions**: 사람/에이전트는 label 없이 관찰과 상호작용(보고 만지고 듣는 것)으로 세상을 배움. SSL이 그런 학습 방식을 모사함.
- **exhaustive annotation 불필요**: p.8~10의 문제(복잡한 개념, video, rare concept, 도메인 차이)를 모든 개념에 label을 달지 않고 우회함.
- **multiple modalities or structure in the domain**: 이미지-소리, 이미지-텍스트, video의 시간 구조, 이미지의 공간 구조 등 data에 내재한 구조가 공짜 supervision이 됨 (p.13의 3번 열 참고).

## Self-Supervised Learning Paradigm (p.16)

- 그림: 위/아래 두 박스.
  - **1) Self-Supervised Learning** (파란 테두리): Dataset (no labels, 다양한 동물/사물 이미지 격자) → **Encoder** (사다리꼴, 좌에서 우로 좁아짐) → **Learned Representation** (세로 파란 박스) → **Decoder, Classifier, Regressor** (사다리꼴, 우로 넓어짐) → 점선 박스 "Labels/outputs automatically generated from data".
  - **2) Supervised Learning** (초록 테두리): Dataset (with labels, 이미지 격자) → **Encoder** → **Learned Representation** → **FC** (작은 사다리꼴) → **Labels**. 우측 흐릿한 점선 박스 "Labels/outputs automatically generated from data"(비활성).
  - 두 박스의 Learned Representation 사이를 굵은 파란 이중선 화살표(위→아래)로 연결: 1)에서 학습된 encoder/representation을 2)로 옮겨 씀 (transfer).
- 우측 텍스트:
  - **Pretext Task**
    - Define a task based on the data itself
    - No manual annotation
    - Could be considered an unsupervised task;
    - but we learn with supervised learning objectives, e.g., classification or regression.
  - **Downstream Task**
    - The application you care about
    - You do not have large datasets
    - The dataset is labeled

**설명**
- SSL은 **2단계 paradigm**임.
  1. **Pretext task** (pretraining): label 없는 대규모 data에서 자동 생성한 target으로 encoder를 학습. 형식은 supervised(classification/regression loss)지만 label이 공짜이므로 "unsupervised task로 볼 수도 있음"이라고 슬라이드가 설명함.
  2. **Downstream task**: 실제 관심 application. labeled data가 적으므로, 1단계의 **encoder를 가져와 (frozen 또는 fine-tune)** 위에 작은 head(FC)만 얹어 학습.
- 두 단계에서 **encoder + Learned Representation은 공유**되고, head만 다름: pretext에서는 decoder/classifier/regressor, downstream에서는 FC. pretext head는 버리는 것이 일반적임.
- 왜 이렇게 설계했나: 큰 unlabeled data로 **일반적인 feature**를 먼저 만들고, 적은 label로 task에 적응시키면 처음부터 supervised로 학습하는 것보다 data efficiency가 높음. 이것이 p.8~10 문제(라벨 부족)에 대한 해법임.
- 평가: 보통 representation의 품질은 downstream 성능(e.g., linear probing, fine-tuning)으로 측정함.

> 보충: "pretext task"는 이후 맥락(p.17)에서 image transformation 예측이나 corrupted image 완성 같은 예로 이어짐. 슬라이드의 "Could be considered an unsupervised task" 표현은 엄밀한 용어 구분이 모호하므로 "label은 없지만 supervised 형태의 loss를 쓴다"로 이해하면 됨.

## FIGURES

- p.9: LabelMe class 빈도의 log-log long-tail plot + "10% classes = 93% data" callout (중요도 상)
- p.12: SSL 4단계 흐름도 (observed data → pseudo labels → predict → learn representation) (중요도 중)
- p.13: Supervised / Unsupervised / Self-Supervised 3열 비교 도식 + 비교 표 (중요도 상)
- p.14: Word2vec 구조도 (projection → hidden → softmax, context→target) (중요도 상)
- p.16: Pretext task → Downstream task transfer paradigm 도식 (중요도 상)

## EXAM POINTS

1. **Self-supervised learning 정의**: label을 data 자체에서 semi-automatic하게 생성(data의 일부를 다른 부분으로 예측)해서 representation을 학습함. "semi-automatic"인 이유(pretext 설계는 사람이 함)를 설명할 수 있어야 함.
2. **Supervised vs Unsupervised vs Self-supervised 비교** (labels, supervision source, cost, scalability, learned representation): 표 5행을 외울 것. 특히 SSL = 자동 생성 label, 비용 low, scalability very high, rich/transferable representation.
3. **Why SSL? (label 확장 한계)**: 복잡한 개념/video, rare concept(long-tail: 10% class가 93% data), 도메인 다양성(의료영상 등)으로 annotation이 scale하지 않음 → SSL 필요.
4. **Long-tail 분포 해석**: LabelMe의 log-log plot에서 rank-frequency가 power law, 대부분 class는 빈도 1 근처. 이것이 supervised 학습에 주는 문제를 서술.
5. **Pretext task vs Downstream task**: 정의, 데이터(unlabeled 대규모 vs labeled 소규모), head 구성(decoder/classifier/regressor vs FC), encoder 재사용(transfer) 흐름.
6. **Word2vec/BERT와 fill-in-the-blank**: context로 target 단어를 예측하는 구조(projection → hidden → softmax over vocabulary)와 왜 비슷한 context의 단어가 비슷한 embedding을 갖는지.
7. **SSL 신호의 3가지 출처 → 대표 방법**: missing parts 예측(MAE), augmented view 매칭(SimCLR, contrastive), 다른 modality 매칭(CLIP, image-text).
8. **Why 질문**: SSL이 unsupervised와 다른 점(명시적 pretext target + supervised-style objective)과 pretext의 target이 아니라 representation이 목적인 이유.

## VERIFIER REPORT

- p.14: 슬라이드 도식은 history h("the cat sits on the") 다음 단어를 예측하는 형태인데 "CBOW에 해당"이라고 단정한 부분을 수정 (CBOW와 구조만 공통, next-word 예측에 가깝다고 명시). 그림 설명/FIGURES의 "CBOW" 표기도 중립화.
- p.14 설명의 "Word2vec (CBOW식)" 표현을 같은 취지로 정정.
- 그 외 p.9~16 누락/수식/그림 설명/인용(Devlin 2018, He 2022, Chen 2020, Radford 2021, Mikolov 2013, Russell 2008) 오류 없음.
