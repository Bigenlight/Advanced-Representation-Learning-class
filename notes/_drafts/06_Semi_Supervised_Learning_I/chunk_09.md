<!-- DRAFT | lecture: 06_Semi_Supervised_Learning_I | chunk: 09 | pages 66–76 -->
<!-- VERIFIED | fixes: 2 | verifier: sonnet -->

## Are image-level perturbations enough for a Vision Transformer? (섹션 제목, p.66)

슬라이드 텍스트:
- Are image-level perturbations enough for a Vision Transformer?

**설명**
- 여기서부터 **논문 case study** 섹션. 앞에서 본 PS-MT, Dual Teacher 같은 consistency regularization 계열이 전부 **image(input) level perturbation**에 의존한다는 점을 문제 삼으면서 시작함.
- 질문 자체가 이 섹션의 thesis: **ViT는 CNN과 구조가 다른데 perturbation도 구조에 맞게 설계해야 하지 않나?**

---

## Paper: Training Vision Transformers for Semi-Supervised Semantic Segmentation (p.67)

슬라이드 텍스트:
- 논문 첫 페이지 캡처: **Training Vision Transformers for Semi-Supervised Semantic Segmentation**
- 저자: Xinting Hu, Li Jiang, Bernt Schiele (Max Planck Institute for Informatics, Saarland Informatics Campus, Germany)
- 출처: Hu et al., CVPR 2024.

**설명**
- 이 논문의 방법 이름이 **S⁴Former** (Semi-Supervised Semantic Segmentation Transformer). 제목 그대로 **ViT 기반**의 semi-supervised semantic segmentation.
> 보충: 방법 이름 "S⁴Former"는 p.69 이후에 나오고, p.67 슬라이드에는 논문 제목/저자만 있음. 암기용: CVPR 2024, Hu et al.

---

## Are Image-Level Perturbations Enough for a Vision Transformer? (p.68)

슬라이드 텍스트:
- Most consistency methods perturb the input image.
- The teacher receives a weakly augmented image.
- The student receives a strongly augmented image.
- However, a Vision Transformer processes images as patch tokens.
- Its self-attention and output distribution provide additional places for regularization.
- 그림: grouped bar chart. y축 mIoU (%) (70~82), x축 "Number of Unlabeled Images (92 Labeled)" = 656 / 2623 / 10490. 범례 Weak / FD / CutMix / Ours. (FD: feature dropout, 슬라이드 우측 주석)

| unlabeled | Weak | FD | CutMix | Ours |
|---|---|---|---|---|
| 656 | 72.6 | 73.0 | 73.7 | 76.5 |
| 2623 | 72.9 | 74.3 | 74.6 | 78.5 |
| 10490 | 73.1 | 75.7 | 75.8 | 80.1 |

- **Can we regularize the input, feature, and output spaces together?**

**설명**
- 기존 consistency regularization (Mean Teacher / FixMatch 계열): teacher는 **weak aug**, student는 **strong aug** 이미지 → 두 예측을 일치시킴. 즉 perturbation이 **input space에만** 있음.
- 그런데 **ViT는 이미지를 patch token으로 처리**하고 **self-attention**으로 token끼리 섞음 → perturbation을 줄 수 있는 곳이 더 있음: (1) **patch token 배치(input)**, (2) **attention(feature)**, (3) **output distribution(class 간 관계)**.
- 그림 해석: 단순 weak만 쓰면 unlabeled 수가 늘어도 72.6 → 73.1로 거의 안 오름. FD(feature dropout), CutMix 같은 추가 perturbation은 조금 개선. **Ours(S⁴Former)는 모든 설정에서 확실히 우세**하고 unlabeled가 많아질수록(76.5 → 80.1) 이득이 커짐 → 좋은 perturbation이 있어야 unlabeled data를 **제대로 활용**할 수 있다는 메시지.
- 결론 질문: **input, feature, output 세 공간을 함께 regularize할 수 있나?** → S⁴Former의 동기.
> 보충: mIoU = mean Intersection-over-Union, semantic segmentation의 표준 지표. "92 labeled"는 슬라이드 축 라벨 그대로 (PASCAL VOC 계열 설정으로 추정, 슬라이드엔 데이터셋 명시 없음).

---

## S⁴Former: Three Levels of Regularization (p.69)

슬라이드 텍스트:
- Teacher
  - Receives a weakly augmented image
  - Generates pixel-level targets
  - Updated by EMA
- Student
  - Receives a strongly augmented image
  - Uses PatchShuffle and PASA
  - Matches both positive and negative class information
- 그림: 전체 프레임워크 다이어그램. 왼쪽 "Weak/Strong Augmentations": Unlabeled Image → Weak Augment → $x^T$ (teacher 입력), Unlabeled Image → Patch Shuffle → $x^S$ (student 입력). Labeled Image → Weak Augment → $x$ (student, 지도학습 경로). 중앙 위 "Teacher Model $T$": Transformer Encoder → (stop gradient) → Decoder. 중앙 아래 "Student Model $S$": Transformer Encoder → (feature restore) → Decoder. Teacher/Student 사이 **EMA** 화살표(encoder, decoder 각각). Teacher에서 "Patch-Adaptive Self-Attention" 박스가 student encoder로 연결. 오른쪽 "Outputs": Conf. Map $C^T$, Pseudo Label → CE Loss ⊕ NCR Loss (unlabeled 경로). Labeled 경로: student 출력 → CE Loss ← Ground Truth.
- **S⁴Former regularizes the Transformer at the input, feature, and output levels.**

**설명**
- **Teacher**: weak aug 이미지 → **pixel-level pseudo label + confidence map $C^T$** 생성. gradient는 안 흐름(stop gradient), 파라미터는 **student의 EMA**로 업데이트 (Mean Teacher 그대로).
- **Student**: strong aug 이미지, 여기에 **PatchShuffle(input)** 과 **PASA(feature)** 가 들어감. loss는 **CE(positive class) + NCR(negative classes)** 를 같이 맞춤 (output).
- 세 레벨 대응: **Input = PatchShuffle, Feature = PASA, Output = NCR**. 시험에서 이 3-way 매핑 가장 중요.
- PASA가 teacher의 confidence map을 사용한다는 점 주목: teacher가 **어디가 불확실한지** 알려주고, student attention을 거기서 교란함 (p.71).
- Labeled data는 평범하게 student에서 CE loss vs. Ground Truth.
> 보충: 전체 loss는 보통 $\mathcal{L} = \mathcal{L}_{sup} + \lambda\,\mathcal{L}_{unsup}$ 꼴(unsup에 CE+NCR)이지만 가중치 수식은 슬라이드에 없음.

---

## Input Regularization: PatchShuffle (p.70)

슬라이드 텍스트:
- Divide the image into an L x L patch grid.
- Group neighboring patches into M x M blocks.
- Shuffle the blocks before Transformer encoding.
- Restore the feature order before decoding.
- 그림: 6×6 patch grid (patch index 0~35)가 2×2 block 단위로 묶여 색(노랑/빨강/회색/파랑)으로 구분됨 → "Patch Shuffle" → 블록 단위로 순서가 섞인 Shuffled Patches(+Position Encoding 함께) → Transformer Encoder → Patch Embeddings → "Shuffle Back" → Decoder.
- **The student must preserve the same semantics even when the global patch layout changes.**

**설명**
- 절차: 이미지를 $L\times L$ patch grid로 나눔 → 인접 patch를 **$M\times M$ block**으로 묶음 → **block 단위**로 shuffle → Transformer encoder에 입력 → encoder 출력 feature를 **원래 순서로 복원(shuffle back)** → decoder.
- **왜 block 단위인가**: patch를 완전히 개별로 섞으면 local context(인접 patch의 연속성)가 다 깨져서 segmentation에 필요한 local 구조가 망가짐. block으로 묶으면 **블록 내부 local 구조는 유지**하면서 **global layout만 바꿈**. (그림에서 6×6 grid, 2×2 block이면 블록 9개.)
- **왜 효과적인가**: ViT는 self-attention이 모든 token 쌍 사이에서 작동하므로 patch 순서/위치 배치가 바뀌어도 token set 자체는 같음 → 같은 semantic을 예측해야 하는 student는 **global patch layout에 의존하지 않고 semantics를 보존**하도록 강제됨. 즉 위치 편향(positional bias)이나 context shortcut에 대한 regularization.
- **Restore 단계가 필수**: segmentation은 pixel-level dense prediction이라 decoder에서는 원래 공간 순서가 필요 → pseudo label과 pixel 단위로 대응시키려고 복원함.
- CNN에서는 이런 shuffle이 의미 없음(conv는 local grid 구조 고정)에 반해 ViT는 patch token + position encoding 구조라서 가능 → **구조 특화 perturbation**.
> 보충: 그림에서 Position Encoding이 shuffled patch와 같이 들어가는 것처럼 보이는데, 논문 구현 세부(position encoding을 섞는 지 여부)는 슬라이드만으로는 불확실.

---

## Feature Regularization: PASA (p.71)

슬라이드 텍스트:
- The teacher estimates confidence for each image patch.
- High-confidence patches keep their original attention.
- Low-confidence patches receive additional attention modulation.
- Uncertain regions interact more strongly with other uncertain regions.
- 그림: 표준 self-attention 블록 (Input $e$: $N\times d$ → $W^V$, $W^K$, $W^Q$ → $V,K,Q$; $QK^\top$ ($N\times N$)에 **Attention Mask $M$ ($N\times N$)** 를 ⊕(덧셈)한 뒤 softmax → $V$와 곱해 Output $\bar e$ ($N\times d'$)). 아래 점선 박스: Thresholded Image Conf. $\mathbb{I}(C^T>\beta)$ (pixel 단위) → "Patch Avg, Eq.(7)" → Patch Conf. $\bar{C}^T$ → "Eq.(6)" → mask entry $M_{ij}$. 예시로 patch conf 0.06(노란 박스, "$\bar C < \text{Median}(\bar C^T)$"), 0.88(분홍 박스) 표시.
- **PASA perturbs attention where the teacher is uncertain.**

**설명**
- PASA = **Patch-Adaptive Self-Attention**. 핵심: attention logit에 **mask $M$을 더해서**($\text{softmax}(QK^\top + M)$ 형태) attention을 교란하되, $M$을 **teacher의 patch별 confidence**로 만듦.
- 절차 (그림 기반):
  1. teacher confidence map $C^T$를 threshold $\beta$로 이진화: $\mathbb{I}(C^T>\beta)$ (pixel 단위).
  2. patch 단위로 평균 (Patch Avg) → patch confidence $\bar C^T_i$.
  3. 이 patch confidence로 $N\times N$ mask $M_{ij}$ 생성 (patch $i$, $j$ 쌍마다 값 0~1).
  4. **High-confidence patch는 원래 attention 유지**, **low-confidence patch(median 미만)는 추가 modulation**.
- 직관: **불확실한 patch끼리 서로 더 강하게 attend**하게 만듦(= "Uncertain regions interact more strongly with other uncertain regions"). teacher가 확신하는 영역은 건드리지 않아 target 신뢰도 높은 곳은 안정적으로, **어려운 영역에서만 feature perturbation**을 주어 student의 robust한 context 학습을 유도.
- 일반 feature dropout(FD)은 **모든 곳에 무작위**로 교란 → PASA는 **confidence-aware(guided)** 라는 점이 차이 (p.68 그림에서 FD보다 Ours가 높은 이유).
> 보충: Eq.(6), (7)의 정확한 수식은 슬라이드에 식 번호만 있고 본문이 없음. mask가 additive라 $M_{ij}$ 값이 클수록 해당 쌍의 attention logit이 커진다고 해석(슬라이드 그림상 ⊕).

---

## Output Regularization: Negative Class Ranking (p.72)

슬라이드 텍스트:
- CE matches the most likely class.
- NCR matches the relative distribution over the remaining classes.
- Similar classes receive similar relative probabilities.
- The student learns more than a hard pseudo-label.
- Recalculate the softmax excluding the pseudo-label class.
- 수식:

$$\mathcal{L}_{CE} = -\log\left(\frac{\exp(p^S_{\tilde y})}{\sum_{k=1}^{K}\exp(p^S_k)}\right)\ \Longrightarrow\ \mathcal{L}_{NCR} = \sum_{k\neq \tilde y,\ k\in K}\left(\hat p^T_k - \hat p^S_k\right)^2,$$

$$\text{where}\quad \hat p^{T\text{ or }S}_k = \frac{\exp(p^{T\text{ or }S}_k)}{\sum_{k\neq \tilde y,\ k\in K}\exp(p^{T\text{ or }S}_k)}$$

- **The teacher provides a class relationship, not only a single target class.**

**설명**
- **기존 CE**: pseudo label $\tilde y$ (teacher의 argmax class)의 확률만 키움. 나머지 $K-1$개 class 정보는 softmax 분모에 묻혀서 **사실상 버려짐** (hard pseudo-label이 주는 정보 = "정답 class 하나").
- **NCR**: pseudo-label class $\tilde y$를 **제외한 나머지 class들만으로 softmax를 다시 계산**($\hat p_k$) → 남은 negative class들 사이의 **상대 분포**를 teacher와 student가 맞추게 함 (L2 / squared difference).
  - 직관: 이미지가 "고양이"라면 "개"는 "자동차"보다 가능성이 높음 → teacher의 negative class ranking/relative probability에는 **class 간 유사도 정보**가 있음 ("dark knowledge", Hinton의 distillation과 같은 맥락). student는 이걸 같이 배움 → "hard pseudo-label 이상의 것을 학습".
- **왜 정답 class를 빼고 renormalize하나**: 안 빼면 $\tilde y$의 확률이 압도적으로 커서(예: 0.95) 나머지 class의 미세한 차이가 loss에 거의 반영되지 않음. 제외하면 **negative class 사이의 상대적 크기 관계**가 증폭되어 보임.
- 수식 읽기: $p_k^{T}$, $p_k^S$는 class $k$의 **logit**(softmax 전 값), $\hat p$는 $\tilde y$를 뺀 $K-1$ class에 대한 softmax. $\mathcal{L}_{NCR}$은 두 분포의 차를 제곱합으로 비교 (슬라이드의 수식 그대로 transcription).  ← 슬라이드에서 $p$를 "logit"이라 명시하진 않지만 exp()에 들어가므로 logit으로 해석.
- 전체 output regularization = **CE(positive) ⊕ NCR(negative)** (p.69 다이어그램). 이름의 "Ranking"은 negative class들의 순위/상대 크기 관계를 맞춘다는 의미.
> 보충: pixel-level segmentation이라 실제로는 pixel마다 이 loss를 계산해 평균. 슬라이드에는 per-pixel 표기가 생략됨.

---

## Do All Three Components Help? (p.73)

슬라이드 텍스트:
- Each component improves the Transformer baseline.
- PatchShuffle provides the largest individual improvement.
- PASA and NCR provide complementary gains.
- Combining all components gives the best result.
- 표 (mIoU, 괄호는 Base 대비 증가량):

| Methods | Split = 1/16 | 1/8 | 1/4 |
|---|---|---|---|
| SupOnly | 67.7 | 72.8 | 77.4 |
| Base | 75.8 | 77.4 | 80.0 |
| +PatchShuffle | 78.4 (+2.6) | 80.2 (+2.8) | 82.1 (+2.1) |
| +PASA | 77.9 (+2.1) | 79.4 (+2.0) | 81.3 (+1.3) |
| +NCR | 77.5 (+1.7) | 78.7 (+1.3) | 81.9 (+1.9) |
| +All | 80.1 (+4.3) | 81.3 (+3.9) | 82.4 (+2.4) |

- **The largest gain is observed when labeled data are most limited.**

**설명**
- **SupOnly**: labeled data만 쓴 supervised baseline. **Base**: Mean Teacher식 semi-supervised ViT baseline (세 regularization 없음). Base가 이미 SupOnly보다 크게 높음(1/16에서 +8.1) → unlabeled data 자체의 효과.
- 개별 컴포넌트: 모두 Base보다 개선. **PatchShuffle이 가장 큼** (세 split 모두 +2.6/+2.8/+2.1로 최대). PASA, NCR은 그다음 수준이며 split에 따라 순서가 바뀜(1/16, 1/8: PASA > NCR, 1/4: NCR > PASA).
- **+All**: 개별 gain의 단순 합(예: 1/16에서 2.6+2.1+1.7=6.4)보다는 작은 +4.3이지만 **개별보다 항상 큼** → 서로 **complementary**하지만 일부 겹침(overlap)이 있음. 세 레벨이 서로 다른 측면(input/feature/output)을 건드려 시너지.
- **labeled data가 적을수록 gain이 큼**: +4.3 (1/16) → +3.9 (1/8) → +2.4 (1/4). 이유: labeled가 적을수록 pseudo-label/consistency 신호에 대한 의존도가 크고, regularization의 이득도 큼.  ← 이 이유는 해석이며 슬라이드는 결과만 서술.
> 보충: split 1/16, 1/8, 1/4 = 전체 train set 중 labeled로 쓰는 비율. 지표는 mIoU로 추정(슬라이드에 명시 없음, p.68 그림 기준).

---

## What Does S⁴Former Add to Mean Teacher? (p.74)

슬라이드 텍스트:
- What remains unchanged
  - Teacher–student framework
  - EMA teacher update
  - Weak-to-strong consistency
- What changes
  - Patch-aware input perturbation
  - Confidence-aware attention perturbation
  - Class-distribution regularization
- 표:

| Level | Standard Mean Teacher | S⁴Former |
|---|---|---|
| Input | Weak and strong image augmentation | PatchShuffle |
| Feature | Standard feature extraction | PASA |
| Output | Positive pseudo-label consistency | Negative class ranking |
| Teacher update | EMA | EMA |

- **Effective consistency regularization should reflect the structure of the model.**

**설명**
- S⁴Former는 **Mean Teacher를 버린 게 아니라 확장**한 것: teacher-student, EMA update, weak-to-strong consistency는 그대로.
- 바뀐 것 3가지: **patch-aware** input perturbation (PatchShuffle), **confidence-aware** attention perturbation (PASA), **class-distribution** regularization (NCR).
- 표 해석 주의: Input 행에서 S⁴Former가 "PatchShuffle"이라고 쓰였지만 weak/strong augmentation을 없앤 게 아니라 **그 위에 추가**한 것 (p.69 다이어그램: student에 strong aug + PatchShuffle).  ← 슬라이드 표 표기는 단순화됨.
- 한 줄 메시지: **model의 구조(ViT = patch token + attention + class distribution)를 반영해서 consistency regularization을 설계해야 효과적**.

---

## Summary (p.75)

슬라이드 텍스트:
- PS-MT: How can the teacher target be made more reliable?
- Dual Teacher: How can we prevent the teacher from becoming too similar to the student?
- S⁴Former: To which part of the Transformer should consistency be applied?

**설명**
- 이 lecture의 세 가지 teacher-student 계열 방법을 **각각이 던지는 질문**으로 요약:
  - **PS-MT**: teacher target을 어떻게 더 **신뢰할 수 있게(reliable)** 만들까 → target quality 문제.
  - **Dual Teacher**: teacher가 student와 **너무 비슷해지는 문제**(EMA teacher가 student를 따라가며 coupling/confirmation bias 발생)를 어떻게 막을까 → teacher diversity 문제.
  - **S⁴Former**: consistency를 Transformer의 **어느 부분**(input / feature / output)에 적용할까 → perturbation location 문제.
- 시험 서술형에서 "세 방법의 motivation 비교"로 나오기 좋은 구조.
> 보충: PS-MT, Dual Teacher의 세부 메커니즘은 이 chunk 범위 밖(앞 chunk 참고).

---

## Next Week.. (p.76)

슬라이드 텍스트:
- Pseudo Labeling ..

**설명**
- 다음 주 주제: **Pseudo Labeling**. (이번 주 teacher-student consistency 방법들이 pseudo label을 teacher로 생성하는 것과 연결됨.)

---

## FIGURES

- p.68: mIoU bar chart (Weak / FD / CutMix / Ours vs. unlabeled 수 656/2623/10490, 92 labeled), 중요도 중
- p.69: S⁴Former 전체 프레임워크 다이어그램 (teacher/student, PatchShuffle, PASA, CE+NCR), 중요도 상
- p.70: PatchShuffle 도식 (6×6 patch → 2×2 block shuffle → encoder → shuffle back → decoder), 중요도 상
- p.71: PASA 도식 (attention mask $M$ + patch confidence), 중요도 상
- p.72: CE vs NCR 수식, 중요도 상
- p.73: ablation 표 (SupOnly/Base/+PatchShuffle/+PASA/+NCR/+All), 중요도 상
- p.74: Mean Teacher vs S⁴Former 비교 표, 중요도 상

## EXAM POINTS

1. **동기**: 기존 consistency regularization은 **input image perturbation**에 치중. ViT는 patch token + self-attention + output distribution이 있으므로 **input / feature / output 세 레벨을 함께 regularize** 해야 한다는 것이 S⁴Former (Hu et al., CVPR 2024)의 핵심.
2. **Three-level 매핑**: Input = **PatchShuffle**, Feature = **PASA** (Patch-Adaptive Self-Attention), Output = **NCR** (Negative Class Ranking).
3. **PatchShuffle 절차**: $L\times L$ patch grid → $M\times M$ block 묶기 → block 단위 shuffle (encoder 전) → feature order 복원 (decoder 전). 왜 block 단위(local 구조 유지, global layout만 교란)인지, 왜 복원하는지(dense prediction과 pixel 정렬).
4. **PASA 원리**: teacher의 patch confidence로 attention mask $M$ 생성 → $\text{softmax}(QK^\top + M)V$. high-confidence는 원래 attention 유지, **low-confidence(불확실) patch끼리 더 강한 interaction**. 일반 feature dropout과 달리 confidence-guided.
5. **NCR 수식 (암기)**: $\mathcal{L}_{NCR}=\sum_{k\neq\tilde y}(\hat p^T_k-\hat p^S_k)^2$, $\hat p_k=\exp(p_k)/\sum_{k\neq\tilde y}\exp(p_k)$. CE는 pseudo-label class만 맞추지만 NCR은 **나머지 class들의 상대 분포**(class 간 유사도, dark-knowledge 성격)를 맞춤. 왜 정답 class를 빼고 softmax를 재계산하는지 설명할 수 있어야 함.
6. **Ablation 결론**: 각 요소 모두 Base 개선, **PatchShuffle이 individual 최대**, PASA/NCR은 complementary, **+All 최고** (1/16: 75.8 → 80.1, +4.3). **labeled 적을수록 gain 큼** (+4.3 > +3.9 > +2.4).
7. **Mean Teacher와의 비교**: 유지 = teacher-student, EMA update, weak-to-strong consistency. 변경 = input(PatchShuffle) / feature(PASA) / output(NCR) perturbation. 메시지: **regularization은 model structure를 반영해야 한다**.
8. **Lecture summary 3줄**: PS-MT = teacher target 신뢰도, Dual Teacher = teacher가 student와 너무 닮는 것 방지, S⁴Former = Transformer의 **어느 부분**에 consistency를 적용할지.
9. 영어 답안 연습용: "S⁴Former extends Mean Teacher by perturbing not only the input image but also the attention (PASA) and the output class distribution (NCR), exploiting the patch-token structure of ViT."
10. 다음 주: Pseudo Labeling.
