<!-- DRAFT | lecture: 03_Contrastive_Learning_I | chunk: 07 | pages 49–56 -->
<!-- VERIFIED | fixes: 4 | verifier: sonnet -->

## Feature Suppression (p.49)

- Although the two representations exhibit high similarity, the information crucial for downstream tasks differs.
  - (previous slide) SimCLR focuses on color and texture while ignoring shape
  - (previous slide) CLIP recognizes the object 'rabbit' but struggles to distinguish its orientation
- This phenomenon, where useful information present in the input is lost in the representation, is known as feature suppression.

(그림 없음, 텍스트만 있는 정의 슬라이드)

**설명**
- 앞 슬라이드(p.48)의 예시 정리: SimCLR 임베딩은 색·텍스처가 같고 shape가 다른 두 이미지를 cosine similarity 0.935로 거의 같다고 봤고, CLIP은 같은 물체(rabbit)지만 방향이 다른 두 이미지를 0.933으로 거의 같다고 봄. 즉 **높은 유사도 != 같은 semantics**.
- **Feature suppression** = 입력에 downstream task에 유용한 정보(shape, orientation 등)가 있는데도 representation에서 **사라지거나 약하게만 인코딩되는 현상**.
- 핵심은 "유사도 점수"가 아니라 "표현에 어떤 정보가 남았는가"임. 예를 들어 shape이 중요한 task에서 SimCLR 표현으로 linear probing하면 성능이 낮을 것이라는 함의가 있음.
- 두 모델의 억제되는 feature가 다르다는 점에 주목: SimCLR는 shape, CLIP은 orientation. (원인으로 SimCLR의 증강 불변성, CLIP caption의 방향 정보 부족 등을 추측할 수 있으나 슬라이드에는 원인 설명이 없음, 보충 추론.)

> 보충: feature suppression은 Chen et al. (2021) "Intriguing Properties of Contrastive Losses"에서 "feature suppression / competing features"로 분석된 바 있음. 한 feature가 contrastive loss를 이미 줄여버리면 다른 feature 학습 압력이 줄어드는 경쟁(competition) 현상으로 설명됨. (슬라이드에는 인용 없음)

## Why Does Feature Suppression Happen? (p.50)

**그림**: 왼쪽에서 오른쪽으로 흐르는 다이어그램.
- 왼쪽 세 개의 박스: **Anchor**(빨간 줄무늬 원), **Positive**(빨간 줄무늬 원, 줄무늬 방향/굵기가 약간 다름), **Negative**(파란 단색 사각형).
- 가운데 위쪽: "Color learned" 라벨의 빨강/초록/파랑 색 사각형 3개에서 **진한 파란 굵은 화살표**가 Encoder로 들어감 (색 정보가 인코더를 통과).
- 그 아래: "Texture suppressed" (회색 줄무늬 사각형), "Shape suppressed" (회색 원/삼각형/사각형) 에서 **연한 회색 화살표**가 Encoder로 들어감 (통과가 약함/억제).
- 중앙: 사다리꼴 모양 **Encoder** 블록, 오른쪽으로 파란 화살표.
- 오른쪽: **2D embedding space** 축(x, y). 빨간 점 두 개(Anchor, Positive)가 점선 원 안에 붙어 있고, 파란 점(Negative)은 멀리 우상단에 위치, 초록 체크 표시(contrastive 목적 만족).

텍스트:
- Both Anchor and Positive are represented by red striped circles, while Negative is a blue square.
- The model can easily distinguish between positive and negative samples using color alone.
- The contrastive objective is already satisfied.
- Therefore, there is no need to additionally learn texture and shape.
- **The model learns the easiest discriminative feature (Color), while other useful features are suppressed.**

**설명**
- Contrastive loss는 "anchor-positive는 가깝게, anchor-negative는 멀게"만 요구함: 
  $$\mathcal{L} = -\log \frac{\exp(\mathrm{sim}(z_a, z_p)/\tau)}{\exp(\mathrm{sim}(z_a, z_p)/\tau) + \sum_{n}\exp(\mathrm{sim}(z_a, z_n)/\tau)}$$
  (수식은 슬라이드에 없음, 이전 슬라이드의 InfoNCE 형태를 가정한 보충)
- 이 예시에서 negative가 빨강 vs 파랑으로 **색만 달라도** positive/negative 구분이 가능 -> loss가 이미 낮음 -> gradient가 작아져 texture, shape를 추가로 학습할 **유인(incentive)이 없음**.
- 즉 원인은 모델이 멍청해서가 아니라 **목적함수가 "충분조건"만 만족하면 되는 shortcut 학습**을 허용하기 때문. 가장 쉬운(simplicity bias) 구분 feature가 먼저 학습되고 나머지는 억제됨.
- 그림에서 회색 화살표 = 정보가 있었지만 표현에는 거의 반영 안 된 것, 파란 굵은 화살표 = 지배적 feature.
- 이 분석이 다음 MCL의 설계 동기가 됨: **shortcut(색)을 쓸 수 없게 negative를 고르면 된다**.

> 보충: 이는 supervised learning의 shortcut learning / simplicity bias (Geirhos et al., 2020 "Shortcut learning in deep neural networks")와 같은 맥락. (슬라이드에 인용 없음)

## Multistage Contrastive Learning (p.51)

- MCL learns different features in multiple stages.

**그림**: 가로로 STAGE 0 / STAGE 1 / STAGE 2 세 패널(파랑/노랑/주황 점선 박스) + 오른쪽에 "N Stages ..." 와 최종 결과.
- 범례: 빨간 원 = **Anchor**, 빨간 사각형 = **Negative Sample**, 양방향 화살표 = **Maximizing Distance**.
- 각 패널 위쪽: 색(초록/주황)·텍스처(줄무늬/체크/빗금 등)·모양(삼각형/원/오각형)이 다른 물체들이 흩어져 있고, anchor(초록 삼각형)에서 negative들로 화살표가 뻗음.
  - Stage 0: negative가 색 구분 없이 전체에서 선택됨(초록, 주황 섞임).
  - Stage 1: negative가 anchor와 같은 색(초록)인 것만. 점선으로 Color 경계 표시.
  - Stage 2: 같은 색 + 같은 텍스처 클러스터 내에서만 negative 선택. Color, Texture 경계선 표시.
- 각 패널 가운데: 화살표 $f_0$, $f_1$, $f_2$ + "K-means" 라벨. 아래에 표현 공간 $Z_0$, $Z_1$, $Z_2$ (불규칙한 둥근 영역). Z_0는 "Color" 선으로, Z_1은 "Texture"로, Z_2는 "Shape"로 나뉨.
- 각 패널 아래: negative 라벨 벡터 $\hat{\mathbf{y}}^{-}_{(i,1)} = [\mathbf{y}^-_{(i,0)}]$, $\hat{\mathbf{y}}^{-}_{(i,2)} = [\mathbf{y}^-_{(i,0)}, \mathbf{y}^-_{(i,1)}]$, $\hat{\mathbf{y}}^{-}_{(i,3)} = [\mathbf{y}^-_{(i,0)}, \mathbf{y}^-_{(i,1)}, \mathbf{y}^-_{(i,2)}]$ (이전 stage들의 클러스터 할당을 누적).
- 오른쪽: N stages 후 여러 색 점선으로 쪼개진 공간, $\mathbf{Z} = [\mathbf{Z}_0, \mathbf{Z}_1, \mathbf{Z}_2, \dots, \mathbf{Z}_{N-1}]$, "Final representations depending on color, texture and shape".

우측 5단계 목록:
① Train / ② Cluster / ③ Select Negatives Within Clusters / ④ Repeat / ⑤ Concatenate Representations

하단: **Each stage removes the shortcut used in previous stages.**

**설명**
- **MCL(Multistage Contrastive Learning)** 은 한 번의 학습으로 모든 feature를 얻으려 하지 않고, **stage를 나눠 stage마다 다른 지배적 feature를 학습**하게 하는 방법.
- 알고리즘 루프: (1) 현재 stage의 encoder $f_k$를 contrastive로 학습 -> (2) 그 표현 $Z_k$에 K-means 클러스터링 -> (3) 다음 stage에서 negative를 **anchor와 같은 클러스터 안에서만** 선택 -> (4) 반복 -> (5) 각 stage 표현을 concat해 최종 $Z$.
- 설계 이유: 이전 stage가 학습한 feature(예: 색)는 클러스터링에 그대로 반영됨 -> 같은 클러스터 내 negative들은 그 feature 값이 비슷 -> 그 feature로는 구분이 불가 -> 모델은 **다른 feature를 학습해야 함**.
- 하단 라벨 $\hat{\mathbf{y}}^-$ 는 "어떤 클러스터 조합 안에서 negative를 뽑을지"를 나타내는 누적 cluster assignment로 해석됨 ← 슬라이드에서 기호 정의가 따로 없음, 해석은 추정.

> 보충: 이 방법은 Zhang et al. (ECCV 2024) "Learning the Unlearned: Mitigating Feature Suppression in Contrastive Learning"의 Multistage Contrastive Learning으로 보임 (슬라이드에 출처 명시 없음, 확인 필요).

## Multistage Contrastive Learning — Stage 0 (p.52)

- MCL learns different features in multiple stages.

**그림**: p.51의 STAGE 0 패널만 확대. anchor(초록 삼각형)에서 전체 영역의 여러 negative(빨간 사각형 4개, 색 섞임)로 화살표, $f_0$ 화살표 후 K-means, $Z_0$ 영역에 점선 "Color" 경계. 아래 $\hat{\mathbf{y}}^-_{(i,1)} = [\mathbf{y}^-_{(i,0)}]$.

텍스트:
- Stage 0
  - Standard contrastive learning is performed.
    - It is assumed that the model first learns color, which is the simplest feature.
    - As a result, the representation space is partitioned based on color.
  - Clustering
    - K-means clustering is applied to the representations from Stage 0.
    - Samples within the same cluster share similar dominant features learned in the previous stage.

**설명**
- Stage 0은 **일반 contrastive learning**(SimCLR류)이라 negative를 batch 내에서 무작위로 사용함.
- 가정: 모델은 **가장 쉬운 feature인 색**부터 학습함 (p.50의 simplicity bias). 그래서 $Z_0$ 공간이 색 기준으로 갈라짐.
- 이 표현에 **K-means**를 돌리면 클러스터 = "지배 feature(색)가 같은 샘플 묶음"이 됨. 즉 클러스터링이 "이 stage가 어떤 shortcut을 썼는지"를 라벨 없이 **추출**하는 역할.
- 중요: 색이라는 것은 예시이며, 실제로 어떤 feature가 먼저 학습되는지는 데이터/증강에 따라 다름 (슬라이드의 'assumed').

## Multistage Contrastive Learning — Stage 1 (p.53)

- MCL learns different features in multiple stages.

**그림**: 왼쪽 STAGE 0, 오른쪽 STAGE 1 패널 두 개. Stage 1에서는 anchor(초록 삼각형)와 세 negative(빨간 사각형)가 모두 초록색 물체이며 점선 "Color" 경계 한쪽(초록 쪽) 안에 있음. $f_1$ -> K-means -> $Z_1$ 영역이 "Texture" 점선으로 나뉨. 아래 $\hat{\mathbf{y}}^-_{(i,2)} = [\mathbf{y}^-_{(i,0)}, \mathbf{y}^-_{(i,1)}]$.

텍스트:
- Stage 1
  - Negative samples are selected only from the same cluster as the anchor.
    - Anchor: green textured triangle
    - Negatives: other green objects
  - Since all negatives share the same color, color can no longer be used to distinguish between them.
    - Consequently, the model must learn other features, such as texture.

**설명**
- Stage 1에서는 negative pool을 **anchor와 같은 Stage 0 클러스터(=같은 색)** 로 제한함.
- 이러면 positive(같은 이미지의 augmentation)와 negative가 모두 초록이라 색은 판별에 도움이 안 됨 -> loss를 줄이려면 **texture** 같은 다음으로 쉬운 feature를 써야 함.
- Stage 1의 encoder $f_1$은 (슬라이드상) Stage 0과 별도로 학습되어 $Z_1$을 생성하고, 다시 K-means -> 이번엔 texture 기준으로 분할됨.
- 직관: 한 번의 학습에서는 색이 gradient를 독점하지만, **색을 negative 쪽에서 무력화**하면 억눌려 있던 feature가 드러남.

## Multistage Contrastive Learning — Stage 2 (p.54)

- MCL learns different features in multiple stages.

**그림**: STAGE 0, 1, 2 패널 세 개. Stage 2에서는 anchor와 negative 2개(빨간 사각형)가 같은 색(초록)·같은 텍스처인 작은 영역(Color 점선과 Texture 점선이 교차하는 사분면)에 있음. $f_2$ -> K-means -> $Z_2$ 가 "Shape" 점선으로 나뉨. 아래 $\hat{\mathbf{y}}^-_{(i,3)} = [\mathbf{y}^-_{(i,0)}, \mathbf{y}^-_{(i,1)}, \mathbf{y}^-_{(i,2)}]$.

텍스트:
- Stage 2
  - Cluster assignments from both Stage 0 and Stage 1 are used together.
    - Same-color cluster
    - Same-texture cluster
  - Negative samples are selected from within the group that satisfies these conditions.
    - Since neither color nor texture can be used for discrimination in this case, the model learns to focus on shape.

**설명**
- Stage 2의 negative 조건 = **Stage 0 cluster 일치 AND Stage 1 cluster 일치** (교집합). 누적 cluster 라벨 $\hat{\mathbf{y}}^-_{(i,2)}=[\mathbf{y}^-_{(i,0)},\mathbf{y}^-_{(i,1)}]$ 가 이를 표현 (슬라이드에서는 Stage 1 패널 아래에 표기, Stage 2 패널 아래의 $\hat{\mathbf{y}}^-_{(i,3)}$ 은 Stage 3용).
- 색과 텍스처가 모두 같으니 남는 구분 단서는 **shape**. 따라서 $Z_2$는 shape 중심.
- stage가 늘수록 negative 후보 pool이 **점점 작아짐** (교집합이라서) -> 실무적으로는 클러스터 수 K와 batch/데이터 크기 trade-off가 있을 것임 ← 슬라이드엔 언급 없음, 보충 추론.

## Multistage Contrastive Learning — Concatenation (p.55)

- MCL learns different features in multiple stages.

**그림**: p.51과 동일한 전체 다이어그램(STAGE 0/1/2 + N Stages). 오른쪽 아래의 $\mathbf{Z} = [\mathbf{Z}_0, \mathbf{Z}_1, \mathbf{Z}_2 \dots, \mathbf{Z}_{N-1}]$ 와 "Final representations depending on color, texture and shape" 부분이 **빨간 타원**으로 강조됨. (N Stages 후 공간이 파랑/노랑/주황 점선으로 잘게 분할된 그림)

하단: **Concatenation preserves features learned at all stages.**

**설명**
- 각 stage의 표현은 **서로 다른 feature**를 주로 담고 있으므로, 최종 표현은 이를 **concatenate**하여 $Z \in \mathbb{R}^{N d}$ (stage당 차원 $d$ 가정) 로 구성함.
- 슬라이드 문구는 "Concatenation preserves features learned at all stages." 뿐임. 마지막 stage 표현만 쓰면 이전 stage의 feature(color/texture)가 빠질 수 있다는 점이 concat의 이유로 해석됨 (보충 추론; 실제로 완전히 사라진다고 단정할 수는 없음).
- 결과적으로 downstream에서 color, texture, shape 어느 것이 필요해도 linear classifier가 해당 부분 차원을 쓸 수 있음.
- 단점(보충): 표현 차원이 stage 수에 비례해 커지고 학습 비용이 N배 -> 효율성 trade-off.

## Difference from Hard Negative Mining (p.56)

- Hard negative mining: Selects negatives that are close to the current embedding.
- MCL: Selects only negatives that belong to the same cluster from the previous stage.
- Objective: Rather than finding difficult samples, the goal is to prevent the reuse of previously learned features.
- In other words, while MCL may appear similar to hard negative mining, its objective is different.

(그림 없음)

**설명**
- 둘 다 "anchor와 가까운 negative"를 고르는 듯 보임 (이전 stage 표현 기준 같은 클러스터 = 가까움). 그러나 **목적이 다름**.
  - **Hard negative mining**: 현재 임베딩에서 가까운 negative로 loss의 gradient 정보를 키워 **판별 경계를 세밀하게** 만듦 (학습 효율/정확도 향상).
  - **MCL**: 이전 stage에서 이미 학습한 feature 값이 같은 negative만 골라 그 feature를 **shortcut으로 못 쓰게 함**. 목적은 **새 feature 학습 유도(feature suppression 완화)**.
- 차이점 구체화: hard negative는 "현재 모델의 같은 embedding 공간"에서 거리를 기준으로, MCL은 "이전 stage 표현의 cluster membership"을 기준으로 하며 **stage마다 새 encoder 학습**과 결합됨.
- 부작용 관점: hard negative mining은 false negative(실제 같은 클래스)를 뽑을 위험이 있음; MCL도 같은 클러스터 = 비슷한 샘플이므로 유사한 위험이 있을 수 있음 ← 슬라이드 언급 없음, 보충 추론.

> 보충: hard negative mining의 대표 논문은 Robinson et al. (2021) "Contrastive Learning with Hard Negative Samples" (ICLR). (슬라이드에 인용 없음)

## FIGURES

- p.50: Anchor/Positive/Negative -> Encoder -> 2D embedding space 다이어그램 (색만 학습되고 texture/shape이 suppressed 되는 이유) — 중요도 상
- p.51: MCL 전체 파이프라인 (Stage 0/1/2, K-means, 누적 cluster 라벨, concat $Z$) + 5단계 목록 — 중요도 상
- p.52–54: 각 stage별 확대 패널 (Stage 0 -> Color, Stage 1 -> Texture, Stage 2 -> Shape) — 중요도 중 (p.51과 중복, 정식 노트에는 p.51/p.55 하나로 충분)
- p.55: 최종 concat $Z$ 강조 (빨간 타원) — 중요도 중

## EXAM POINTS

1. **Feature suppression의 정의**: 입력에 있는 유용한 정보가 representation에서 손실되는 현상. 예시: SimCLR(shape 무시, cos sim 0.935), CLIP(orientation 구분 못함, 0.933) -> "높은 유사도 != 같은 semantics".
2. **Why 질문**: feature suppression이 왜 발생하나? -> 가장 쉬운 discriminative feature(예: color)만으로 contrastive objective가 이미 만족되어 texture/shape를 학습할 유인이 없기 때문 (shortcut / simplicity bias).
3. **MCL의 5단계 절차**: Train -> Cluster(K-means) -> Select negatives within clusters -> Repeat -> Concatenate representations. 각 단계의 역할 서술.
4. **Stage별 학습 feature**: Stage 0(표준 CL) = color, Stage 1(같은 color cluster 내 negative) = texture, Stage 2(color+texture 같은 cluster 내 negative) = shape. 왜 그렇게 되는지(해당 feature로는 negative 구분 불가) 설명.
5. **Concatenation의 이유**: $Z=[Z_0,\dots,Z_{N-1}]$ — 모든 stage에서 학습한 feature 보존. 마지막 stage만 쓰면 이전 feature가 사라짐.
6. **MCL vs Hard Negative Mining 비교**: 둘 다 가까운 negative를 쓰는 것처럼 보이나, hard negative = 현재 임베딩 거리 기준 + 어려운 샘플 찾기 / MCL = 이전 stage cluster 기준 + 이전 feature 재사용 방지.
7. **누적 cluster 조건**: Stage k의 negative는 이전 모든 stage(0..k-1)의 cluster 할당이 anchor와 같은 샘플에서 선택됨 ($\hat{\mathbf{y}}^-_{(i,k)}=[\mathbf{y}^-_{(i,0)},\dots,\mathbf{y}^-_{(i,k-1)}]$). 슬라이드에서는 이 라벨이 각 stage 패널 아래, 즉 그 stage가 끝난 뒤(다음 stage용)에 $\hat{\mathbf{y}}^-_{(i,k+1)}=[\mathbf{y}^-_{(i,0)},\dots,\mathbf{y}^-_{(i,k)}]$ 로 표기됨.

## VERIFIER REPORT

- p.49 설명: SimCLR/CLIP이 각각 shape/orientation을 억제하는 원인(증강 불변성, caption 특성)을 사실처럼 쓴 부분을 슬라이드에 없는 보충 추론으로 표기.
- p.55 설명: "마지막 stage만 쓰면 shape만 남고 사라진다"는 단정을 슬라이드 근거(문구)와 추론으로 분리하여 완화.
- p.54 설명: 누적 라벨 $\hat{y}^-_{(i,2)}$의 위치(Stage 1 패널 아래, Stage 2 negative용)를 명확히 하여 오해 소지 수정.
- EXAM POINT 7: 인덱스 오류 수정. Stage k negative 조건은 $[y_0,\dots,y_{k-1}]$ (= $\hat{y}_{(i,k)}$)이며 슬라이드 표기는 stage 종료 후 $\hat{y}_{(i,k+1)}$.
- 그 외 페이지 49-56의 bullet/그림 설명/인용(Chen 2021, Geirhos 2020, Zhang ECCV 2024, Robinson ICLR 2021)은 이상 없음.
