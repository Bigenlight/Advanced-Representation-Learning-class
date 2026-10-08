<!-- DRAFT | lecture: 03_Contrastive_Learning_I | chunk: 09 | pages 65–72 -->
<!-- VERIFIED | fixes: 2 | verifier: sonnet -->

> 맥락: p.63 = Sobal et al., ICLR 2025, "X-Sample Contrastive Loss: Improving Contrastive Learning with Sample Similarity Graphs" (X-CLR). p.64 = "Are all negative samples equally unrelated? 다른 개 / 고양이 / 피아노를 똑같이 negative로 취급해야 하나?" 라는 문제의식. 이 chunk는 그 해답인 X-CLR의 본론.

## Contrastive Learning as a Similarity Graph (p.65)

**그림**: 가로 3개 heatmap (N×N 행렬, 축은 둘 다 `sample idx`). 축 옆 색 막대로 클래스 구분 (Cat=진파랑, Dog=하늘색, Piano=빨강, Guitar=분홍). 흑백 colorbar 0.0~1.0 (검정=1.0).
- (왼쪽) **Self-supervised**: 대각선(i=i)만 검은 점. 나머지 전부 흰색.
- (가운데) **Supervised**: 같은 클래스끼리 대각 방향 블록 4개(약 5×5)가 전부 검정. 블록 밖은 흰색.
- (오른쪽) **X-CLR**: 대각 블록 안에서도 회색 농도가 제각각이고, Cat–Dog, Piano–Guitar 블록 사이에도 회색 값이 있음 (Cat/Dog끼리, Piano/Guitar끼리 비슷). 동물 vs 악기 사이는 흰색.

텍스트:
1. Self-supervised learning — Only two views of the same image are connected.
2. Supervised contrastive learning — Samples from the same class are connected.
3. X-CLR — Samples can have different degrees of similarity.

**X-CLR replaces a binary graph with a soft similarity graph.**

**설명**
- contrastive learning을 **샘플 간 edge를 가진 graph**로 보는 관점. 행렬의 (i,j) 값 = 샘플 i, j가 "얼마나 positive인가"의 target.
- **Self-supervised (SimCLR 등)**: 같은 이미지의 두 augmentation view만 positive (graph = 단위행렬 꼴, binary). 같은 클래스 다른 이미지도 negative로 밀어냄 → false negative 문제.
- **Supervised contrastive (SupCon)**: label 같으면 1, 다르면 0 (binary 블록 대각). 클래스 내부가 전부 동일 1이라 세부 관계 표현 불가, 클래스 간 관계도 0.
- **X-CLR**: 값이 [0,1] 연속인 **soft graph** → "개와 고양이는 피아노보다 가깝다"를 반영. p.64의 "모든 negative가 똑같이 unrelated인가?"에 대한 직접적 대답.
- 이전 방법과 차이: binary(0/1) → continuous. positive/negative 이분법을 없앰.

## Where Does the Similarity Graph Come From? (p.66)

**그림**: 파이프라인 다이어그램. 왼쪽 위 노란 박스 "Source of graph" → (화살표) 대각선이 진한 회색 노이즈 행렬 → **X-CLR loss**(분홍 박스)로 입력. 왼쪽 아래 노란 박스 "Images" → 파란 $f_\text{image}$ (encoder) → 원 $Z_\text{image}$ → X-CLR loss로 입력. 오른쪽 확대 말풍선: 개 사진 2장이 "dogs"라는 굵은 파란 곡선 edge(similarity 0.8)로, 개–고양이가 "pets"라는 가는 edge(0.4)로 연결. 범례 Similarity: 굵은 선 0.8, 가는 선 0.4.

텍스트:
- A similarity graph can come from: Text captions / Class descriptions / Semantic hierarchies / Other metadata
- [EX] ImageNet: German shepherd → "a photo of a German shepherd", Maltese dog → "a photo of a Maltese dog"
  - The sentence is represented by the class name.
  - A fixed text encoder converts each sentence into an embedding.
  - Target similarity between the two samples
- **If there are captions in the dataset, we can just use it.**

**설명**
- target graph는 **이미지 밖의 정보 source**(caption, class 이름, 계층 구조, metadata)에서 만듦.
- ImageNet 예: class name을 "a photo of a ..." 문장으로 만들고 **frozen(fixed) text encoder**로 임베딩 → 두 문장 임베딩의 similarity가 두 샘플의 target similarity. 같은 class는 sim=1, 비슷한 품종(German shepherd vs Maltese dog)은 높은 값, 개 vs 피아노는 낮은 값.
- 이미지 encoder만 학습되고 text encoder는 고정 → text는 supervision signal 제공자 역할.
- caption이 있는 데이터셋이면 caption을 그대로 문장으로 사용.

> 보충: CLIP은 image-text 쌍을 맞추는 게 목표(text encoder도 학습)인 반면, X-CLR은 text를 target graph 생성에만 쓰고 최종 목표는 image representation의 질.

## X-CLR Objective (p.67)

**그림/코드**: 파이썬 의사코드 박스.
```python
def training_step(captions, images):
    # encode captions a frozen text encoder
    txt_emb = txt_enc(captions).detach() # N x D_txt
    # encode images with the model we are training
    img_emb = img_enc(images)  # N x D_img

    # calculate similarities
    tgt_sims = softmax(txt_emb @ txt_emb.T / tao_s, dim=1)
    img_sims = softmax(img_emb @ img_emb.T / tao, dim=1)
    # the loss is the cross-entropy between the two
    return CE(tgt_sims, img_sims, dim=1).mean()
```
우측 텍스트:
- Target graph — Similarity from text or metadata
- Learned graph — Similarity between image representations
- Training objective — Match the learned graph to the target graph

수식 (슬라이드 원문 그대로):
$$\mathcal{L}_{\text{X-CLR}} = \frac{1}{2N_b}\sum_{i=1}^{2N_b} H(s_i, p_i)$$
$$p_{i,j} = \frac{\exp(\text{sim}(z_i,z_j)/\tau)}{\sum_{k=1}^{2N}\mathbf{1}_{k\neq i}\exp(\text{sim}(z_i,z_k)/\tau)}$$  ← 분모 합의 상한이 $2N$ (다른 식은 $2N_b$), 표기 불일치로 보임
$$s_{i,j} = \frac{\exp(G^{(\text{soft})}_{i,j}/\tau_s)}{\sum_{k=1}^{2N_b}\exp(G^{(\text{soft})}_{i,k}/\tau_s)},\qquad G^{(\text{soft})}_{i,j}=\text{sim}(f_\text{text}(c_i), f_\text{text}(c_j))$$

**설명**
- **기호**: $N_b$ = batch size (view 2개씩이라 샘플 $2N_b$개), $z_i$ = i번째 이미지 representation, $\text{sim}$ = cosine similarity, $\tau$ = image 쪽 contrastive temperature, $c_i$ = i번째 샘플의 caption/text, $f_\text{text}$ = frozen text encoder, $G^{(soft)}$ = soft target graph (text 임베딩 간 similarity), $\tau_s$ = target 쪽 temperature, $H(s,p)=-\sum_j s_j\log p_j$ = cross-entropy.
- **$p_i$**: learned graph. 이미지 i가 batch 내 다른 샘플 j(자기 자신 제외, $\mathbf{1}_{k\ne i}$)와 얼마나 비슷한지를 softmax 분포로 만든 것 (InfoNCE의 softmax와 동일 형태).
- **$s_i$**: target graph. text similarity를 $\tau_s$로 softmax 정규화한 **soft label 분포**.
- **loss**: $s_i$(target)와 $p_i$(예측)의 cross-entropy → 각 행마다 learned graph를 target graph에 맞춤. 즉 "**InfoNCE의 one-hot label을 soft label $s_i$로 교체**"한 것. $s_i$가 one-hot이면 SimCLR/InfoNCE로 환원.
- `.detach()`: text embedding은 gradient 없음 (target은 상수).
- 직관: contrastive를 "어느 샘플이 내 positive인가" 분류 문제로 볼 때 정답 분포를 hard → soft로 바꾼 것 = label smoothing / knowledge distillation과 유사.
- 코드 vs 수식 차이 ← 코드의 `softmax(txt_emb @ txt_emb.T / tao_s)`에는 대각(자기 자신) 항이 포함되지만 $p_{i,j}$ 수식에는 $k\neq i$ 마스크가 있음. `CE(tgt_sims, img_sims, dim=1)`도 의사코드 수준 표기 (이미 softmax된 확률을 CE에 넣는 형태) ← 의심 표기.

## The Role of Similarity Temperature (p.68)

텍스트:
- **τ**: This is a common contrastive temperature used for image representation similarity.
- **τ_s**: It determines how "soft" to make the target similarity graph.
  - Low target temperature
    - Focus on the strongest positive.
    - Similar to standard contrastive learning.
  - High target temperature
    - Give more weight to related samples.
    - The target graph becomes softer.
- **The graph should preserve the true positive while including soft positives.**

**설명**
- temperature가 둘: $\tau$ (learned graph, 일반 InfoNCE temperature), $\tau_s$ (target graph를 얼마나 soft하게 만들지).
- **$\tau_s \to 0$**: softmax가 argmax처럼 → target이 거의 one-hot (가장 강한 positive만) → **standard contrastive learning**과 동일.
- **$\tau_s$ 큼**: 분포가 평평 → 관련 샘플들에 weight 분산 → soft graph.
- 너무 크면 모든 샘플이 비슷해져 true positive 신호가 묻힘 → 결론: **true positive는 유지하면서 soft positive를 포함**하도록 $\tau_s$를 맞춰야 함 (trade-off).
- 즉 $\tau_s$는 "SimCLR ↔ 완전 soft" 사이를 잇는 knob.

## What Does X-CLR Learn? (p.69)

**그림**: 12×12 heatmap 3개 (행/열 = 12개 class, 같은 순서). 클래스: cougar, lynx, tabby (Felines, 주황) / Maltese dog, German shepherd, Doberman (Dogs, 남색) / basketball, volleyball, tennis ball (Types of balls, 보라) / acoustic guitar, grand piano, saxophone (Musical instruments, 초록). 각 셀에 수치 표기.
- **SupCon target similarities**: 완전 단위행렬 (대각 1.00, 나머지 0.00).
- **X-CLR target similarities**: 같은 상위 그룹 내 값이 높음. 예: cougar–lynx 0.48, Maltese–German shepherd 0.63, German shepherd–Doberman 0.60, Maltese–Doberman 0.64, basketball–volleyball 0.62, basketball–tennis ball 0.63, volleyball–tennis ball 0.62, acoustic guitar–saxophone 0.36. 그룹 간 값은 낮음 (예: cougar–grand piano 0.07).
- **X-CLR Learned similarities**: 학습된 image representation의 class 간 similarity. 대각 0.80~1.00 (대부분 0.80~0.96, basketball만 1.00), 그룹 내 블록이 어둡고 구조가 target과 유사 (volleyball–basketball 0.70, grand piano–acoustic guitar 0.54, saxophone–grand piano 0.55). 수치가 target과 완전히 같진 않고 일부 그룹 간 값이 target보다 큼 (tabby–saxophone 0.49, tabby–grand piano 0.42).
- 범례: Felines / Dogs / Types of balls / Musical instruments.

**The embedding space reflects semantic structure beyond class labels.**

**설명**
- SupCon은 class label만 쓰므로 target이 identity → class 간 관계 정보 0. X-CLR target은 text encoder 덕에 계층적 의미 구조(고양잇과, 개, 공, 악기)를 포함.
- 핵심: **학습된 representation(오른쪽)이 이 구조를 거의 재현** → 명시적 상위 카테고리 label 없이도 embedding에 의미 구조가 반영됨.
- learned 값이 target과 다른 이유(보충 추측): image 쪽 $\tau$와 target $\tau_s$ scale 차이, 이미지의 실제 시각적 유사성도 반영.

## Does the Soft Graph Help? (p.70)

**그림(표)**: "Table 1: Accuracy of X-CLR and a range of baselines on a set of classification benchmarks. We freeze the encoder, and train a linear layer on top to probe the quality of representations. We test on ImageNet classification, as well as on a set of robustness benchmarks."

| Method | Soft loss | Uses labels | ImageNet | ImageNet Real | BG Decomp. Same Class | BG Decomp. Mixed | ObjectNet | MIT States Objects | MIT States Attributes |
|---|---|---|---|---|---|---|---|---|---|
| SCE | ✓ | ✗ | 71.3 | 78.7 | 61.7 | 58.4 | 20.2 | 44.5 | 31.0 |
| ReSSL (1 crop) | ✓ | ✗ | 69.4 | 76.9 | 56.3 | 53.2 | 18.3 | 44.5 | 31.2 |
| VICReg | ✗ | ✗ | 72.4 | 79.0 | 60.8 | 56.8 | 20.5 | 43.5 | 26.9 |
| Barlow Twins | ✗ | ✗ | 72.8 | 80.0 | 62.7 | 59.4 | 21.6 | **45.9** | **31.7** |
| SimCLR | ✗ | ✗ | 63.4 | 67.8 | 44.7 | 38.9 | 12.1 | 40.9 | 29.1 |
| SupCon | ✗ | ✓ | 74.3 | 79.7 | 64.0 | 59.1 | 24.4 | 45.6 | 30.8 |
| X-CLR | ✓ | ✓ | **75.6** | **81.5** | **66.6** | **62.7** | **27.5** | **45.9** | 31.1 |

**X-CLR improves classification and background robustness.**

**설명**
- 평가: encoder freeze + **linear probing**. ImageNet / ImageNet Real(재라벨링) / Background Decomposition(배경 변화 robustness) / ObjectNet(분포 이동) / MIT States.
- 비교 축 2개: **soft loss 사용 여부**, **label 사용 여부**. X-CLR만 둘 다 ✓.
- SimCLR(63.4) → SupCon(74.3) → X-CLR(75.6): label 추가 효과 + soft 추가 효과. SupCon 대비 ImageNet +1.3, ObjectNet +3.1, BG Same Class +2.6, Mixed +3.6 → robustness 향상이 상대적으로 큼.
- 전 지표 1등은 아님: MIT States Attributes는 Barlow Twins 31.7 > X-CLR 31.1. Objects는 Barlow Twins와 X-CLR 45.9 동률.
- soft loss만 있고 label 없는 SCE, ReSSL은 SimCLR보단 높지만 X-CLR보다 낮음 → soft + 외부 semantic 정보가 중요.

> 보충: SCE = Similarity Contrastive Estimation (Denize et al.), ReSSL = Relational Self-Supervised Learning (Zheng et al.). 둘 다 soft relation을 쓰는 SSL 방법 (슬라이드에는 설명 없음).

## X-CLR in Low-Data Settings (p.71)

**그림(표 + 그래프)**:
"Table 3: X-Sample Contrastive learning performance with CC3M and CC12M training. We train X-CLR on conceptual caption images with text caption similarities, and compare to CLIP and SimCLR. SimCLR does not use text data and is only trained on images."

| Data | Method | ImageNet | ImageNet Real | BG Same Class | BG Mixed | ObjectNet |
|---|---|---|---|---|---|---|
| CC3M | SimCLR | 57.0 | 64.0 | 24.4 | 18.9 | 10.8 |
| CC3M | CLIP | 41.0 | 47.6 | 12.5 | 10.6 | 7.8 |
| CC3M | X-CLR | **58.2** | **65.6** | **26.7** | **20.3** | **11.5** |
| CC12M | SimCLR | 58.9 | 66 | 24.6 | 19.8 | 12.7 |
| CC12M | CLIP | 58.8 | 66.1 | 20.5 | 17.1 | 11.9 |
| CC12M | X-CLR | **59.4** | **66.7** | **26.1** | **20.4** | **13.4** |

그래프(오른쪽): x축 `Samples per class` (0~1000), y축 `Imagenet Accuracy` (0~60+). 3개 선: X-CLR(파랑), SimCLR(주황), SupCon(초록). X-CLR과 SupCon은 거의 겹쳐 빠르게 상승 (320 샘플/class에서 약 58~61, 1000에서 ~75), SimCLR은 훨씬 낮음 (640에서 ~55, 1000에서 ~63). 샘플 수가 적은 구간에서 X-CLR/SupCon vs SimCLR 격차가 큼.

**Soft relationships are especially useful when training data is limited.**

**설명**
- (표) noisy한 web caption 데이터(CC3M/CC12M)에서도 X-CLR이 SimCLR/CLIP보다 우수. 특히 **CC3M(작은 데이터)에서 CLIP이 41.0으로 크게 낮은 반면** X-CLR 58.2. CC12M으로 데이터가 커지면 CLIP이 따라잡아 격차 감소 (59.4 vs 58.8).
- (그래프) class당 샘플이 적을 때(약 ~600 이하) SimCLR는 X-CLR/SupCon보다 크게 뒤처지고(예: 160/class에서 ~21 vs ~43~45) X-CLR은 SupCon 수준 유지 → soft graph가 샘플당 **supervision 신호 밀도**를 높여 data-efficient.
- 직관: binary graph에선 positive가 적지만 soft graph는 모든 샘플 쌍이 학습 신호가 됨.
- 주의: 그래프에서 X-CLR은 SupCon과 거의 같거나 약간 아래 ← 주장은 SimCLR 대비 우위이지 SupCon 대비 우위는 아님.

## Limitations (p.72)

텍스트:
- Captions, labels, or **metadata** are **required** to construct the similarity graph.
- Low **graph quality** leads to poor representation quality.
- **Noisy text**, such as CC captions, may fail to provide accurate semantic relations.
- **Storing** a graph of all samples requires significant **memory**; therefore, practical implementations necessitate batch-level similarity calculations or pre-computation.
- **Biases inherent** in the **text encoder** can also be propagated to the target graph.

**설명**
- **외부 정보 의존**: 순수 self-supervised가 아님. caption/label/metadata 없으면 target graph 못 만듦.
- **graph 품질 = 표현 품질의 상한**: target이 틀리면 틀린 관계를 그대로 학습. web caption(CC)처럼 noisy하면 부정확.
- **메모리**: 전체 데이터셋 N×N graph는 $O(N^2)$ → 불가능. 그래서 (1) **batch 안에서만** similarity 계산, (2) text embedding **pre-compute**.
- **편향 전파**: frozen text encoder의 bias가 target graph로 들어가 image encoder에 전이.

## FIGURES

- p.65: 세 가지 graph (Self-supervised 대각 / Supervised 블록 / X-CLR soft) 비교 heatmap — 중요도 상
- p.66: similarity graph source → X-CLR loss 파이프라인 + 개/고양이 edge 0.8/0.4 예시 — 중요도 중
- p.67: X-CLR 의사코드 + loss 수식 — 중요도 상
- p.69: SupCon target / X-CLR target / X-CLR learned similarity 3 heatmap (12 class) — 중요도 상
- p.70: baseline 비교 표 (soft loss / labels 컬럼 포함) — 중요도 중
- p.71: low-data 표 + samples-per-class 그래프 — 중요도 중

## EXAM POINTS

1. **Contrastive learning을 similarity graph로 보기**: self-supervised(두 view만 연결, 단위행렬형) / supervised(같은 클래스 binary 블록) / X-CLR(연속 soft graph) 비교.
2. **X-CLR loss 수식**: $\mathcal{L}=\frac{1}{2N_b}\sum_i H(s_i,p_i)$, $s_{i,j}=\text{softmax}(G^{(soft)}_{i,j}/\tau_s)$, $G^{(soft)}_{i,j}=\text{sim}(f_\text{text}(c_i),f_\text{text}(c_j))$, $p_{i,j}$는 이미지 representation의 InfoNCE형 softmax (자기 제외). 기호 의미 설명.
3. **Why**: InfoNCE의 one-hot target을 text similarity 기반 soft label로 대체 → 비슷한 negative(다른 개)를 피아노와 똑같이 밀어내는 문제 해결. $s_i$가 one-hot이면 SimCLR로 환원.
4. **두 temperature의 역할**: $\tau$(learned graph) vs $\tau_s$(target softness). 낮은 $\tau_s$ → 가장 강한 positive만 → standard contrastive, 높은 $\tau_s$ → 관련 샘플에 weight 분산. true positive 유지가 핵심.
5. **Target graph source와 text encoder**: caption, class description("a photo of a ..."), semantic hierarchy, metadata; text encoder는 **frozen**(detach), image encoder만 학습.
6. **What X-CLR learns**: SupCon target(단위행렬) vs X-CLR target/learned similarity에서 felines/dogs/balls/instruments 그룹 구조 반영 → class label 이상의 semantic structure.
7. **실험 결과**: ImageNet 75.6 (SupCon 74.3, SimCLR 63.4), robustness(BG Decomp., ObjectNet) 향상; low-data(CC3M, samples per class 적을 때)에서 특히 이득 (CLIP 41.0 vs X-CLR 58.2 @CC3M).
8. **Limitations**: 외부 metadata 필요, graph/caption 품질 의존, noisy text, $O(N^2)$ 메모리 → batch-level 계산 또는 pre-computation, text encoder bias 전파.

## VERIFIER REPORT
- p.69 설명: learned similarity 대각 범위 "0.80~0.96" -> basketball 대각이 1.00이므로 "0.80~1.00"으로 수정.
- p.71 설명: "SimCLR는 거의 학습 못 함"은 과장 (그래프상 160/class에서 ~21, 640에서 ~55) -> 수치 기반 표현으로 수정.
- 그 외 표/수식/인용(Sobal ICLR 2025, SCE, ReSSL) 수치 대조 결과 이상 없음.
