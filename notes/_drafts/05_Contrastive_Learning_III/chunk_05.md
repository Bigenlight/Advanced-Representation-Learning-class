<!-- DRAFT | lecture: 05_Contrastive_Learning_III | chunk: 05 | pages 33–40 -->
<!-- VERIFIED | fixes: 3 | verifier: sonnet -->

## CLIP – 전체 파이프라인 도식 (p.33)

- 제목: CLIP / bullet: CLIP (Contrastive Language-Image Pre-training)
- 출처: Radford et al., PMLR 2021, *Learning Transferable Visual Models From Natural Language Supervision*

**그림 (3단 구성 도식)**
- **(1) Contrastive pre-training** (좌측): 위쪽에 텍스트 입력 스택("Pepper the aussie pup" 등 캡션들) → **Text Encoder**(보라색 사다리꼴) → 텍스트 임베딩 $T_1, T_2, T_3, \dots, T_N$ (보라색 가로 막대). 아래쪽에 강아지 이미지 스택 → **Image Encoder**(초록색 사다리꼴) → 이미지 임베딩 $I_1, I_2, I_3, \dots, I_N$ (초록색 세로 막대). 두 임베딩 열/행이 만나 $N \times N$ 행렬을 이룸: 셀 $(i,j)$ = $I_i \cdot T_j$. **대각선 셀($I_1 \cdot T_1, I_2 \cdot T_2, \dots, I_N \cdot T_N$)이 파란색으로 강조**(= positive pair, 정답), 나머지 off-diagonal은 흰색(= negative).
- **(2) Create dataset classifier from label text** (우상단): 클래스 이름 리스트(plane, car, dog, ..., bird) → 템플릿 "A photo of a {object}." 에 삽입 → **Text Encoder** → $T_1, T_2, T_3, \dots, T_N$ (클래스별 텍스트 임베딩).
- **(3) Use for zero-shot prediction** (우하단): 테스트 이미지(강아지) → **Image Encoder** → $I_1$ → $T_1 \dots T_N$ 과의 내적 행 ($I_1 \cdot T_1, I_1 \cdot T_2, I_1 \cdot T_3, \dots, I_1 \cdot T_N$), 이 중 $I_1 \cdot T_3$ 셀이 파란색 → 화살표로 출력 "A photo of a dog."

**표 형태 요약 박스 (슬라이드 하단)**

| CLIP Training | CLIP Inference |
|---|---|
| Encode images and captions into a shared embedding space. | Encode an input image. |
| Compute an $N \times N$ image-text similarity matrix. | Encode candidate text prompts. |
| Maximize similarity for matched pairs. | Compute image-text similarities. |
| Minimize similarity for mismatched pairs using a **symmetric contrastive loss**. | Predict the text prompt with the highest similarity. |
| **Key Idea:** Learn aligned image and text representations. | **Key Idea:** Classification is performed by image-text matching instead of a trained classifier. |

**설명**
- CLIP은 **이미지 인코더와 텍스트 인코더를 동시에 학습**해서 두 modality를 **하나의 shared embedding space**에 정렬(align)하는 방법임. 학습 단위는 mini-batch의 $N$개 (image, text) pair.
- batch 안에서 가능한 $N^2$개 조합 중 **$N$개의 대각선만 정답(positive)**, 나머지 $N^2 - N$개는 전부 **in-batch negative**로 취급함. 즉 SimCLR의 "같은 이미지의 다른 augmentation = positive" 대신 **"같은 의미를 가진 (image, caption) 쌍 = positive"** 로 positive의 정의가 바뀐 것임. 이미 짝지어진 두 modality가 서로의 view 역할을 함.
- 행렬의 각 행을 보면 "이 이미지에 맞는 텍스트를 N개 중에서 고르는 N-way 분류", 각 열을 보면 "이 텍스트에 맞는 이미지를 고르는 분류" → **symmetric contrastive loss**(이미지→텍스트 + 텍스트→이미지)의 근거임. 구체 코드는 p.35.
- **Zero-shot 추론**: 고정된 classifier head가 없고, 클래스 이름을 "A photo of a {object}." 같은 **prompt template**에 넣어 text encoder로 임베딩하면 그것이 곧 **classifier weight** 역할을 함. 이미지 임베딩과 cosine similarity가 가장 큰 prompt를 예측으로 선택함 → 학습 때 보지 못한 클래스에도 text만 바꾸면 적용 가능함.
- 기존 supervised 분류와의 차이: 기존은 **고정된 label set**(예: ImageNet 1000 class)에 대한 one-hot 정답을 예측, CLIP은 **자연어 설명과의 매칭**이라 클래스 집합이 열려 있음(open-vocabulary).

> 보충: prompt template("A photo of a {object}.")은 paper에서 *prompt engineering / ensembling*으로 성능을 끌어올린 부분이며, 단어 하나만 넣는 것보다 문맥을 주면 정확도가 올라감(단어 하나는 polysemy 문제가 있음, 예: crane). 시험 범위 밖일 수 있으나 "왜 template을 쓰는가"에 대한 답이 됨.

---

## CLIP – 전체 파이프라인 도식 (재게시) (p.34)

- p.33과 **완전히 동일한 슬라이드**(제목, 도식, 하단 표, 출처 모두 동일, 페이지 번호만 34). ← 슬라이드 중복(의도적 반복 혹은 애니메이션 단계로 추정)
- 내용은 p.33 참조: (1) Contrastive pre-training의 $N \times N$ 행렬 / (2) 라벨 텍스트로 classifier 생성 / (3) zero-shot prediction.

**설명**
- 위 p.33 설명과 동일함. 강의 중 도식을 짚으며 한 번 더 설명하기 위한 반복 페이지로 보임. 정식 노트에서는 p.33 하나로 합치면 됨.

---

## CLIP – 학습 pseudo-code (p.35)

- 제목: CLIP / bullet: CLIP (Contrastive Language-Image Pre-training)
- 슬라이드 내용은 논문 Figure 3의 numpy 스타일 pseudo-code (코드 전문):

```python
# image_encoder - ResNet or Vision Transformer
# text_encoder  - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l]       - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t             - learned temperature parameter

# extract feature representations of each modality
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T)  #[n, d_t]

# joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)

# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)

# symmetric loss function
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss   = (loss_i + loss_t)/2
```

수식으로 쓰면 ($z_i^I$: 이미지 임베딩, $z_j^T$: 텍스트 임베딩, 둘 다 $\ell_2$-normalized, $\tau = e^{-t}$):

$$\mathcal{L} = \frac{1}{2N}\sum_{i=1}^{N}\left[-\log\frac{\exp(z^I_i\cdot z^T_i/\tau)}{\sum_{j=1}^{N}\exp(z^I_i\cdot z^T_j/\tau)} \;-\; \log\frac{\exp(z^I_i\cdot z^T_i/\tau)}{\sum_{j=1}^{N}\exp(z^I_j\cdot z^T_i/\tau)}\right]$$

**설명**
- **Encoder 선택**: image encoder는 ResNet 또는 ViT, text encoder는 CBOW 또는 Text Transformer. 각자 modality 고유의 feature 차원($d_i$, $d_t$)을 냄.
- **Projection $W_i, W_t$**: 서로 차원이 다른 feature를 **공통 차원 $d_e$** 로 보내는 learned linear projection. 두 modality를 같은 공간에서 비교하기 위해 필요함.
- **$\ell_2$ normalize**: 임베딩을 단위 벡터로 만들어서 내적 = **cosine similarity**가 되게 함. 크기(norm)가 아니라 방향만 비교하게 되어 학습이 안정적임.
- **learned temperature $t$**: logits에 $\exp(t)$ 를 곱함(= $1/\tau$). SimCLR 등에서 $\tau$를 hyperparameter로 고정한 것과 달리, CLIP은 **$\tau$를 학습 가능한 파라미터**로 둠(log-parameterization으로 $\exp(t)$를 써서 양수 보장). cosine similarity는 $[-1, 1]$ 범위라 그대로 softmax에 넣으면 분포가 너무 flat하므로 scale을 키워주는 역할임.
- **logits** $[n, n]$ : $(i, j)$ 원소 = 이미지 $i$와 텍스트 $j$의 scaled cosine similarity. 정답 label은 `np.arange(n)` — 즉 $i$번째 행의 정답은 $i$번째 열(대각선).
- **Symmetric loss**: `loss_i`(axis=0)와 `loss_t`(axis=1) 두 방향의 cross-entropy를 평균. 한 방향은 "텍스트를 고정하고 맞는 이미지 찾기", 다른 방향은 "이미지를 고정하고 맞는 텍스트 찾기". 슬라이드의 `axis` 표기상 어느 쪽이 image→text인지는 코드 주석만으로 모호함. ← 논문 원본 코드 그대로이며 axis 해석은 구현 관점에 따라 다름.
- **SimCLR(InfoNCE/NT-Xent)와의 관계**: 구조는 InfoNCE와 동일(positive 1개 vs in-batch negatives)하지만, (1) positive가 서로 다른 modality 쌍, (2) 양방향 loss, (3) learnable temperature, (4) 각 modality용 별도 encoder라는 점이 다름.

> 보충: 논문에서는 batch size 32,768 같은 매우 큰 batch로 학습해 in-batch negative 수를 크게 늘림. contrastive learning에서 negative가 많을수록 유리하다는 앞 강의 내용(SimCLR의 large batch)과 연결됨.

---

## CLIP – 효율성 비교 실험 (p.36)

- 제목: CLIP / bullet: CLIP (Contrastive Language-Image Pre-training)

**그림 (line chart)**
- x축: **# of images processed** (2M, 33M, 67M, 134M, 268M, 400M; 로그 스케일처럼 배치)
- y축: **Zero-Shot ImageNet Accuracy** (0~40)
- 3개 곡선:
  - 초록: **Bag of Words Contrastive (CLIP)** — 가장 위, 400M에서 약 41
  - 주황: **Bag of Words Prediction** — 중간, 400M에서 약 26.5
  - 파랑: **Transformer Language Model** — 가장 아래, 400M에서 약 16
- 화살표 주석: 같은 정확도(약 16%)에 도달하기 위해 필요한 이미지 수 비교 → 초록(CLIP) vs 주황: **4X efficiency**, 주황 vs 파랑: **3X efficiency**.

**우측 텍스트**
- The success of CLIP comes less from sophisticated language modeling and more from **contrastive alignment between images and natural language.**

**Key Findings**
- **Contrastive learning** is approximately **4× more data-efficient** than caption prediction.
- A simple **Bag-of-Words (BoW)** text encoder outperforms a Transformer language model.
- Rich language supervision is more important than complex text modeling.
- Contrastive alignment enables efficient large-scale vision-language pretraining.

**설명**
- 세 가지 방식의 비교: (a) **Transformer Language Model**: 이미지로부터 캡션을 단어 단위로 생성(autoregressive captioning) — 가장 어려운 예측 문제, (b) **BoW Prediction**: 캡션에 나온 단어 집합(bag of words)을 예측 — 순서를 버려 쉬워진 predictive objective, (c) **BoW Contrastive (CLIP)**: 정확한 단어를 예측하는 대신 **이미지-텍스트 매칭 여부만 구분**.
- 핵심 메시지: **정확히 어떤 단어를 쓸지 예측(generative/predictive)하는 것보다 어느 텍스트가 이 이미지와 맞는지 구분(contrastive)하는 편이 훨씬 data-efficient**. 같은 이미지에도 설명 방식은 무수히 많아서(comment, alt-text 등) 정확한 문장을 예측하려는 목표는 노이즈가 큼. contrastive는 "맞는 쌍 vs 틀린 쌍"만 구별하면 되어 **표현 학습 목적에 더 직접적**임.
- 단순한 BoW encoder가 Transformer LM보다 좋다 = 모델 복잡도가 아니라 **objective와 supervision의 종류**가 성능을 가름.

> 보충: 이 실험은 논문 Figure 2에 해당함. 슬라이드에 "4X"와 "3X" 화살표가 나오는데, 곡선 위에서 동일 accuracy 수준에 도달하는 데 필요한 샘플 수 비율을 의미함(논문에서 Transformer LM 대비 BoW prediction이 3배, BoW contrastive가 추가로 4배 효율적).

---

## CLIP – Why is CLIP Special? (p.37)

- 제목: CLIP / bullet: CLIP (Contrastive Language-Image Pre-training)
  - Why is CLIP Special?
    - Uses **natural language** as the supervisory signal.
    - Learns from **400 million image–text pairs** collected from the Internet.
    - Enables **zero-shot recognition** using text prompts.
    - Learns more transferable representations than fixed-label supervision.
- 출처: Radford et al., PMLR 2021

**설명**
- **Natural language supervision**: 사람이 정해놓은 고정 label(ImageNet의 1000 클래스) 대신 **인터넷의 caption/alt-text** 자체를 supervision으로 사용. label 수작업 비용 없이 규모를 키울 수 있음(WIT, WebImageText 데이터셋).
- **400M pairs**: 대규모 weakly-supervised(noisy) 데이터. 노이즈는 크지만 양이 압도적이라 상쇄됨.
- **Zero-shot recognition**: 새로운 task에 대해 fine-tuning 없이 text prompt만 바꿔서 분류 수행(p.33의 (2),(3)).
- **Transferable representation**: 고정 label supervision은 그 label set에 과적합되기 쉬운 반면, 자연어는 훨씬 풍부한 개념(속성, 문맥, 관계)을 담아서 다양한 downstream task에 전이가 잘 됨.

> 보충: 논문은 CLIP이 30개 이상의 vision dataset에서 zero-shot으로 완전 supervised ResNet-50과 경쟁할 수 있음을 보였고, distribution shift(ImageNet-R, Sketch 등)에서 robustness가 높다는 점도 강조함. 슬라이드에는 없음.

---

## Supervised Contrastive Learning – 개요 (p.38)

- 제목: Supervised Contrastive Learning / bullet: Supervised Contrastive Learning (SupCon)
  - Extends contrastive learning to **fully supervised settings**.
  - Treats all samples from **the same class as positive pairs.**
  - **Pulls** embeddings of the **same** class together.
  - **Pushes** embeddings from **different** classes apart.
- **Key Idea**: Use class labels to learn more discriminative representations than cross-entropy.
- 출처: Khosla et al., NeurIPS 2020, *Supervised Contrastive Learning*

**설명**
- 지금까지(SimCLR, MoCo 등)의 contrastive learning은 label이 없는 self-supervised 설정: positive = 같은 이미지의 두 augmentation, 나머지 전부 negative. 이 방식은 **같은 클래스 이미지도 negative로 취급**하는 문제(false negative)가 있음.
- **SupCon**은 label이 있을 때 **같은 class의 모든 샘플을 positive**로 묶어 같은 class는 가깝게(pull), 다른 class는 멀게(push) 만듦. 즉 contrastive loss의 정의를 label 정보로 확장한 것.
- **vs cross-entropy**: cross-entropy는 logit을 label에 맞추는 분류 손실이라 embedding 공간의 구조(class 내 compact, class 간 separation)를 간접적으로만 형성함. SupCon은 **embedding 간 거리 관계를 직접 최적화**해서 더 discriminative한 표현을 얻는다는 주장임.

> 보충: 논문은 cross-entropy가 label noise와 adversarial perturbation에 약하고 margin이 부족하다는 점을 동기로 들며, 학습 후 encoder를 freeze하고 linear classifier를 얹는 2-stage 방식으로 평가함(슬라이드 범위 밖).

---

## Supervised Contrastive Learning – Self-supervised vs Supervised 도식 (p.39)

- 제목: Supervised Contrastive Learning / bullet: Supervised Contrastive Learning (SupCon)

**그림 (좌우 2분할, 가운데 검은 세로선으로 구분)**
- **왼쪽: "Self Supervised Contrastive"**
  - 중앙에 회색 구(embedding space를 상징). 구 둘레의 점들에 선이 연결됨.
  - **Anchor**: 포메라니안 강아지(검은 테두리) — 회색 선으로 연결.
  - **Positive**: 같은 강아지의 augmented 이미지(작게) — 주황 선으로 연결.
  - **Negatives**: 코끼리, 새끼고양이 2장(고양이 두 마리), **그리고 빨간 테두리가 쳐진 스패니얼 강아지**(Cavalier King Charles Spaniel) — 모두 빨간 선으로 연결. → 같은 "강아지" class인 스패니얼도 negative로 처리됨(**빨간 테두리 = 잘못된 negative 강조**).
- **오른쪽: "Supervised Contrastive"**
  - 같은 구성인데, **Positives**(복수형)에 포메라니안 augmented 이미지 + **초록 테두리의 스패니얼 강아지**가 포함되어 주황 선으로 연결됨.
  - Negatives: 코끼리, 고양이 2마리(빨간 선).

**설명**
- 핵심 대비: 같은 class(dog) 이미지인 스패니얼이 self-supervised에서는 **negative(밀어냄)**, supervised에서는 **positive(끌어당김)**. 같은 의미 class인 샘플을 밀어내는 self-supervised의 **false negative 문제**를 label로 해소함.
- Positive가 **1개(augmentation)에서 여러 개(같은 class 전체)로 증가**하므로, anchor 하나당 학습 신호가 훨씬 풍부해짐.
- 결과적으로 embedding 공간에서 클래스별 **cluster**가 형성되고 cluster 사이는 멀어짐(p.38의 pull/push).

---

## Supervised Contrastive Learning – SupCon Loss 수식 (p.40)

- 제목: Supervised Contrastive Learning / bullet: Supervised Contrastive Learning (SupCon)
- 슬라이드 인용문(논문): Given a set of randomly sampled $n$ (image, label) pairs, $\{\mathbf{x}_i, y_i\}_{i=1}^{n}$, $2n$ training pairs can be created by applying two random augmentations of every sample, $\{\tilde{\mathbf{x}}_i, \tilde{y}_i\}_{i=1}^{2n}$.
- Supervised contrastive loss $\mathcal{L}_{\text{supcon}}$ utilizes multiple positive and negative samples, very similar to **soft nearest-neighbor loss**: (밑줄 처리됨)

$$\mathcal{L}_{\text{supcon}} = -\sum_{i=1}^{2n} \frac{1}{2|N_i| - 1} \sum_{j \in N(y_i),\, j \neq i} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_j / \tau)}{\sum_{k \in I,\, k \neq i} \exp(\mathbf{z}_i \cdot \mathbf{z}_k / \tau)}$$

- 빨간 박스 주석:
  - $\exp(\mathbf{z}_i \cdot \mathbf{z}_j/\tau)$ → "**Positive similarity**"
  - $|N_i|$ → "**A set of Positives of the same class as Anchor**"
  - $j \in N(y_i), j \neq i$ → "**For each positive**"
- 하단 굵은 문구: A method that uses SimCLR's InfoNCE as is, but instead of using only one positive, uses all samples of the same class as positive, calculates each InfoNCE, and then takes the average.

**설명**
- **기호**: $n$ = batch 내 원본 샘플 수, augmentation 2개씩 만들어 총 $2n$개 view. $I = \{1,\dots,2n\}$ = 모든 view의 index 집합. $\mathbf{z}_i$ = $i$번째 view의 (normalized) projection embedding, $\tau$ = temperature. 슬라이드 주석상 $N_i$ = anchor와 같은 class의 positive 집합(원본 샘플 기준), $j\in N(y_i), j\neq i$ = anchor와 같은 label을 가진 **다른 view들의 index**(anchor $i$ 자신 제외).
- **분자** $\exp(\mathbf{z}_i\cdot\mathbf{z}_j/\tau)$: anchor $i$와 positive $j$의 similarity. **분모** $\sum_{k\in I, k\neq i}$: anchor 자신을 제외한 **모든 view**(positive 포함, negative 포함)에 대한 합 → softmax 정규화.
- **안쪽 합** $\sum_{j\in N(y_i)}$: 각 positive $j$마다 하나의 InfoNCE 항을 만들고 합침 → **positive 여러 개 각각에 대해 InfoNCE**.
- **정규화 계수** $\frac{1}{2|N_i|-1}$ : positive 개수로 나눠 **평균**을 냄. 논문 원문의 $N_{\tilde y_i}$는 batch의 **원본 $n$개 샘플 중** anchor와 같은 label을 가진 샘플 수(augmentation 미포함)이며, 각 샘플이 2 view이므로 anchor 자신을 뺀 같은 class view 수 = $2N_{\tilde y_i}-1$ (논문 최종 식에서는 $\frac{-1}{|P(i)|}$로 표기). 슬라이드의 $|N_i|$도 이 의미로 읽으면 일관됨.
- **log가 sum 안쪽에 있는 형태**($\mathcal{L}_{out}$): positive마다 log를 취한 뒤 평균 (논문에서 $\mathcal{L}^{sup}_{out}$ 과 $\mathcal{L}^{sup}_{in}$ 두 변형 중 성능이 더 좋았던 쪽이 *out*). 슬라이드는 out 형태를 보여줌.
- **SimCLR와의 관계**: positive가 단 하나(다른 augmentation)이면 $|P|=1$ 이 되어 SimCLR의 NT-Xent/InfoNCE와 **정확히 일치**. 즉 SupCon은 InfoNCE의 일반화. 슬라이드 하단 문구가 이를 요약함: "InfoNCE를 그대로 쓰되 positive를 같은 class 전체로 늘리고 각각의 InfoNCE를 평균".
- 분모는 **같은 class의 다른 샘플까지 포함**하므로 positive 하나 입장에서는 다른 positive도 분모에 있는 셈 — 이것이 기존 InfoNCE 구조를 "as is" 유지한다는 의미.

> 보충: Soft nearest-neighbor loss(Salakhutdinov & Hinton 2007, Frosst et al. 2019; 뿌리는 Goldberger et al. 2004 NCA)는 같은 class 샘플이 서로 가까운 nearest neighbor가 되도록 하는 loss로, SupCon의 직접적 선조임. 또한 본 슬라이드 다음 페이지(p.41)에서 SupCon이 CIFAR-10/100, ImageNet에서 SimCLR와 cross-entropy를 능가한다고 소개됨(다음 chunk 범위).

---

## FIGURES

- p.33: CLIP 3단 파이프라인 도식 — (1) contrastive pre-training의 $N\times N$ 유사도 행렬(대각선=positive), (2) 라벨→prompt classifier, (3) zero-shot prediction (중요도 상)
- p.34: p.33과 동일 슬라이드 (중복, 정식 노트에는 p.33만 사용)
- p.35: CLIP pseudo-code (중요도 상, 코드 블록으로 옮겨 노트에 포함)
- p.36: 이미지 수 vs zero-shot ImageNet accuracy 곡선 (CLIP vs BoW prediction vs Transformer LM, 4X/3X efficiency) (중요도 상)
- p.39: Self-supervised vs Supervised contrastive 비교 도식 (anchor/positive/negatives, 스패니얼이 negative→positive로 바뀜) (중요도 상)

## EXAM POINTS

1. **CLIP 학습 방식**: $N$개 (image, text) pair에서 $N\times N$ similarity matrix를 만들고 **대각선 = positive, off-diagonal = negative**로 하는 **symmetric contrastive loss**(image→text, text→image 평균). 코드 구현(l2_normalize, logits $= I_e T_e^\top \exp(t)$, `labels = arange(n)`)을 설명할 수 있어야 함.
2. **CLIP의 learnable temperature**와 $\ell_2$ normalization의 역할 (cosine similarity, logit scale 조절). SimCLR의 고정 $\tau$와의 차이.
3. **Zero-shot 추론 과정**: 클래스명 → prompt template("A photo of a {object}.") → text encoder → 클래스별 text embedding이 classifier weight 역할 → image embedding과 cosine similarity 최대인 prompt 선택. 왜 별도 classifier 학습이 필요 없는가?
4. **Why contrastive over predictive (p.36)**: contrastive가 caption prediction보다 약 4× data-efficient, BoW encoder가 Transformer LM보다 우수 → 성능은 복잡한 language modeling이 아니라 **contrastive alignment + rich language supervision** 덕분.
5. **CLIP이 특별한 이유**: natural language supervision, 400M web pairs, zero-shot, fixed-label supervision보다 transferable한 표현.
6. **SupCon의 아이디어와 SimCLR와의 차이**: 같은 class 전체를 positive로 사용(false negative 문제 해소), positive 1개일 때 SimCLR InfoNCE와 일치하는 일반화 관계.
7. **SupCon loss 수식**: $\mathcal{L}_{\text{supcon}} = -\sum_{i=1}^{2n}\frac{1}{2|N_i|-1}\sum_{j\in N(y_i), j\ne i}\log\frac{\exp(z_i\cdot z_j/\tau)}{\sum_{k\ne i}\exp(z_i\cdot z_k/\tau)}$ — 각 기호(positive set, 분자/분모, 평균 정규화)의 의미.
8. **SupCon vs Cross-entropy**: 왜 label을 쓰는데도 contrastive objective가 더 discriminative한 embedding을 만드는가(embedding 공간의 거리 구조를 직접 최적화).

## VERIFIER REPORT
- p.40 설명: $N(y_i)$를 '$N_i$와 동일'로 쓴 정의 모순 수정 ($N_i$=positive 집합, $N(y_i)$=같은 label view index, i 제외).
- p.40 설명: $N_{\tilde y_i}$가 augmentation 포함 전체 수라는 오류 수정 (원본 샘플 수; $2N-1$이 positive view 수).
- 보충: soft nearest-neighbor loss 인용 정정 (Salakhutdinov & Hinton 2007 / Frosst 2019, NCA는 Goldberger 2004).
