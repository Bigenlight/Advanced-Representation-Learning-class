<!-- DRAFT | lecture: 05_Contrastive_Learning_III | chunk: 09 | pages 65–72 -->
<!-- VERIFIED | fixes: 6 | verifier: sonnet -->

## [섹션 구분] What if a mini-batch contains Imprecise Class Labels? (in supervised contrastive learning) (p.65)

섹션 구분 슬라이드: 네이비 배경에 "What if a mini-batch contains Imprecise Class Labels? (in supervised contrastive learning)" 한 문장만 있음. 이후 슬라이드는 supervised contrastive learning(SCL)에서 label noise를 다룸.

**설명**
- 앞 구간(self-supervised / SupCon)은 class label이 정확하다고 가정했음. 여기서는 그 가정이 깨질 때(**imprecise / noisy label**) SupCon이 어떻게 되는지 다룸.
- > 보충: SupCon(Khosla et al., NeurIPS 2020)은 같은 label이면 positive, 다르면 negative로 취급하므로 label 오류가 곧 pair 관계 오류로 이어질 수 있음.

## Understanding and Mitigating Human-Labelling Errors in Supervised Contrastive Learning (p.66)

논문 표지 캡처 슬라이드.
- 제목: "Understanding and Mitigating Human-Labelling Errors in Supervised Contrastive Learning"
- 저자: Zijun Long, Lipeng Zhuang, George Killick, Richard McCreadie, Gerardo Aragon-Camarasa, Paul Henderson (University of Glasgow, United Kingdom)
- 하단 출처 표기: Long et al., ECCV 2024, Understanding and Mitigating Human-Labelling Errors in Supervised Contrastive Learning

**설명**
- 이 구간 전체(p.66~)의 기반 논문 = **Long et al., ECCV 2024**. 제안 방법이 이후 슬라이드에 나오는 **SCL-RHE**임.
- 핵심 문제의식: 기존 noisy label 연구는 **synthetic(랜덤) noise** 위주였지만, 실제 **human labelling error**는 성질이 다르고 SCL에서는 특히 pair 단위로 영향을 줌.

## Human Label Errors in SupCon (p.67)

- It demonstrates that label errors affect Cross-Entropy (CE) and Supervised Contrastive Learning (SCL) differently.
  - In Cross-Entropy, an incorrect label results in an incorrect learning signal for that sample.
  - SCL learns based on the relationship between two samples.
  - Even if some labels are incorrect, the pairwise relationship can still be correct.
  - False positives or false negatives occur depending on the relationship between the assigned label and the latent label.
- 우측 정의 박스:
  - Assigned label (AL): Label provided by an annotator
  - Latent label (LL): True semantic class of the image
- 하단 굵은 문장 2개: "A mislabeled sample does not always produce an incorrect contrastive pair." / "The incorrect pair relationships resulting from label errors, rather than the label errors themselves, directly impact SCL."

**그림 서술**: 3개 영역으로 나뉜 큰 도식. 각 이미지 아래에 `AL: ... LL: ...` 표기, 빨간 테두리 = adverse impact.
- 왼쪽 "Cross-Entropy Loss": Positives / Negatives 열. 예) 여우 사진에 AL: Dog, LL: Fox (빨간 테두리, 오류), 다른 이미지 AL: Cat LL: Dog (빨간 테두리). 즉 CE는 mislabeled 샘플 자체가 오답 신호가 됨. 아래 줄은 정상 샘플(Dog/Dog, Cat/Cat).
- 가운데 "No Adverse Impacts on SCL": **True Positives**(Anchor–Paired Image 쌍; 예: AL Fox LL Dog 와 AL Fox LL Dog 처럼 둘 다 틀렸지만 같은 latent class라서 pair 관계는 맞음, 또는 AL Fox LL Dog / AL Fox LL Wolf 등)와 **True Negatives**(AL Fox/Cat 등 서로 다른 label이고 latent도 다름) 예시. label이 틀려도 pair 관계는 옳은 경우를 보여줌.
- 오른쪽 "Adverse Impacts on SCL" (빨간 제목): 6가지 케이스 I~VI. I(Dog anchor, AL:Dog LL:Fox paired), II(Wolf/Dog anchor, AL:Wolf LL:Wolf paired) = **False Positives**. III(AL: Cat LL: Dog + AL: Cat LL: Fox; 둘 다 틀리고 AL 같음, LL 다름)도 **False Positive**. IV~VI = **False Negatives**(예: V, VI에서 AL: Cat LL: Dog 와 AL: Dog LL: Dog 처럼 latent는 같은데 assigned가 달라 negative로 취급). I, II, III(FP)와 IV, V, VI(FN) 사이에 점선 경계.

**설명**
- **CE**는 각 샘플 (x, y) 하나가 독립 학습 신호이므로 y가 틀리면 그 샘플의 gradient가 곧바로 틀림 → 오류 단위 = sample.
- **SupCon**은 label이 같으면 positive pair, 다르면 negative pair로 쓰므로 오류 단위 = **pair relationship**. 두 샘플이 모두 mislabeled여도 같은 잘못된 label이면(또는 latent가 같으면) 관계는 옳을 수 있음.
- pair 관계 오류 2종:
  - **False positive**: AL 같음 + LL 다름 → 서로 다른 의미 class를 끌어당김.
  - **False negative**: AL 다름 + LL 같음 → 같은 의미 class를 밀어냄.
- 즉 "label error rate"와 "SCL에 해로운 pair error rate"는 다른 양이며, 후자를 분석해야 함.
- > 보충: 이 분석 덕분에 label noise 비율 τ가 같아도 CE와 SCL이 받는 피해 양상이 달라짐을 정량화할 수 있게 됨(다음 슬라이드들).

## Human Errors Differ from Synthetic Errors (p.68)

- Synthetic label errors
  - Labels are changed randomly.
  - Mislabeled samples often look different from the assigned class.
- Human label errors
  - Humans confuse visually similar classes.
  - Mislabeled samples often remain close to the assigned class.
- 하단 굵은 문장: "Human labeling errors overlap strongly with true positives."

**그림 서술**: 4개의 similarity density 그래프(x축 = Similarity(anchor와의 feature cosine similarity, -1.0 ~ 1.0 또는 -0.2 ~ 1.0), y축 = Density, log scale).
- 1·2번째: CIFAR-10 'frog'. 3·4번째: ImageNet 'Scottish terrier'.
- 범례: True Positive(파랑), True Negative(주황), Synthetic Label Errors(빨강), Human Mislabelled Errors(초록).
- 1번(CIFAR-10): 주황(true negative)이 넓게 -0.8~0.8 퍼지고, 파랑(true positive)이 오른쪽(0.4~1.0)에 있으며 초록(human error)이 파랑과 겹치는 높은 similarity(~0.6~0.9) 영역에 위치.
- 2번(CIFAR-10): synthetic error(빨강)가 주황 쪽 낮은 similarity(~-0.4~0.4)에 위치해 true negative와 겹침.
- 3번/4번(ImageNet)도 같은 양상: human mislabelled error(초록)는 true positive(파랑)와 겹치고, synthetic(빨강)은 낮은 similarity 쪽(~0.0~0.2)에 몰림.

**설명**
- **Synthetic noise** = label을 랜덤하게 뒤집음 → 틀린 샘플이 원래 class와 시각적으로 무관 → anchor와 similarity가 낮아 negative 분포에 섞임 → similarity로 쉽게 걸러짐.
- **Human noise** = 사람이 닮은 class(예: Dog vs Fox/Wolf)를 혼동 → 틀린 샘플이 이미 assigned class 근처에 있음 → similarity가 높아 **true positive 분포와 겹침** → similarity 기반 필터링이 실패.
- 결론: synthetic noise로 검증한 noisy-label 기법은 human noise에서 일반화되지 않을 수 있음. 이 점이 p.71의 "easy positive" 문제로 연결됨.
- > 보충: 이 그림은 pretrained/학습된 encoder의 embedding에서 anchor–sample cosine similarity 분포를 그린 것으로 추정(슬라이드에 encoder 명시 없음).

## False Positives Dominate the Errors (p.69)

- False positive
  - Assigned labels are the same
  - Latent labels are different
  - The model incorrectly pulls the two representations together.
- False negative
  - Assigned labels are different
  - Latent labels are the same
  - The model incorrectly pushes out both representations.
- If we let τ be the label error rate and C be the number of classes
- 수식(우측):

$$P_{FN} = \frac{\tau^2}{(C-2)^2} + \frac{2\tau - 2\tau^2}{C-1}$$

$$P_{FP} = \underbrace{2\tau - \tau^2}_{\text{at least one error}} - \underbrace{\frac{\tau^2}{(C-1)^2}}_{\text{both wrong but same latent class}}$$

$$P_{FN} \approx \frac{2\tau}{C-1}, \qquad P_{FP} \approx 2\tau$$

- 하단 굵은 문장: "With many classes, almost all incorrect SCL signals come from false positives."

**그림 서술**: p.67 우측 "Adverse Impacts on SCL" 도식 재사용. Anchor/Paired Image 쌍 I~VI. I, II, III = False Positives(점선 위·왼쪽), IV~VI = False Negatives(점선 아래·오른쪽). 각 이미지 아래 AL/LL 표기, 틀린 label은 빨간색 글씨, 오류 이미지는 빨간 테두리.
- I: Samoyed(Dog/Dog) + 여우(AL Dog, LL Fox)
- II: 늑대 같은 개(AL Wolf, LL Dog) + 늑대(AL Wolf, LL Wolf)
- III: 개(AL Cat, LL Dog) + 여우(AL Cat, LL Fox) — 둘 다 틀리고 같은 AL, LL 다름 → False positive
- IV: 개(AL Cat, LL Dog) + 개(AL Wolf, LL Dog) — AL 다르고 LL 같음 → False negative
- V: 개(AL Dog, LL Dog) + 개(AL Cat, LL Dog) → False negative
- VI: 개(AL Cat, LL Dog) + 개(AL Dog, LL Dog) → False negative

**설명**
- 정의: **FP** = AL 같음 & LL 다름 → positive로 묶지만 실제론 다른 class → **끌어당김(pull together)**. **FN** = AL 다름 & LL 같음 → negative로 취급해 **밀어냄(push apart)**.
- 확률 직관 (τ = label error rate, C = class 수, 오류는 균등하게 다른 class로 간다고 가정):
  - **P_FP**: pair 중 적어도 하나가 틀릴 확률 = $1-(1-\tau)^2 = 2\tau - \tau^2$. 여기서 "둘 다 틀렸지만 실제론 같은 latent class"인 경우($\tau^2/(C-1)^2$)는 False positive가 아니므로 뺌 (AL 같고 LL도 같으니 사실상 올바른 positive).
  - **P_FN**: AL이 다르면서 LL이 같은 경우. C가 크면 두 샘플의 AL이 우연히 같을 일이 거의 없고, 한쪽만 틀려서 AL이 달라지는 항 $\frac{2\tau-2\tau^2}{C-1}$ 이 주항.
  - C가 크면 $P_{FN} \approx \frac{2\tau}{C-1}$ (C에 반비례해 작아짐), $P_{FP} \approx 2\tau$ (C와 무관).
- 의미: **class 수가 많을수록(CIFAR-100, ImageNet-1K) 잘못된 SCL 신호는 거의 다 false positive** → 해결의 초점은 positive pair 쪽.
- 직관: FN은 두 샘플의 latent class가 같아야 하는데 그 확률이 약 1/C 로 작아 $rac{1}{C-1}$ 인자가 붙음(C가 클수록 감소). 반면 FP는 한쪽만 틀려도 발생하는 항($2	au-	au^2$)이 C와 무관하게 남음.
- ← 슬라이드 수식은 FP/FN 모두 "AL이 같다/다르다"를 조건화하지 않은 pair 확률로 보임. 정확한 분모(조건부 vs 결합확률) 정의는 슬라이드에 없음 → 논문 확인 필요.
- ← $P_{FN}$ 첫 항 $\tau^2/(C-2)^2$ 의 유도는 슬라이드에 설명 없음(원문 유지).

## False Positives Dominate the Errors — FP 확률 분해 (p.70)

슬라이드 제목은 p.69와 동일. 순수 그림 슬라이드.

**그림 서술**: 4개 패널(I, II, III, Excluded). 각 패널은 이미지 A, B 두 장(강아지/여우)이 붙어 있고, 아래에 "Assigned"/"Actual" label 표기, 그 아래 "Treated as positive"(두 카드를 잇는 괄호형 화살표) → 결론 박스.
- **I**: A = Assigned: Dog, Actual: Dog / B = Assigned: Dog (빨강), Actual: Fox. → Treated as positive. 박스: ✗ Different actual classes. "A correct, B wrong". 아래 확률 표기 (1-τ)τ.
- **II**: A = Assigned: Fox (빨강), Actual: Dog / B = Assigned: Fox, Actual: Fox. → positive 취급. ✗ Different actual classes. "A wrong, B correct". 확률 τ(1-τ).
- **III**: A = Assigned: Cat (빨강), Actual: Dog / B = Assigned: Cat (빨강), Actual: Fox. → positive 취급. ✗ Different actual classes. "Both wrong". 확률 ττ.
- **Excluded**: A, B 모두 Assigned: Dog (빨강), Actual: Fox. → positive 취급이지만 ✓ Same actual class, "Not a false positive".
- 하단 박스 수식: "At least one wrong: $2\tau - \tau^2$ − Both wrong but same actual class: $\frac{\tau^2}{(C-1)^2}$ = $P_{FP}$"

**설명**
- p.69 식의 **케이스 분해**: 같은 AL을 가지면서 LL이 다른 경우를 A만 틀림 / B만 틀림 / 둘 다 틀림으로 나눔.
  - A 맞고 B 틀림: $(1-\tau)\tau$
  - A 틀리고 B 맞음: $\tau(1-\tau)$
  - 둘 다 틀림: $\tau\cdot\tau$
  - 합: $2\tau(1-\tau)+\tau^2 = 2\tau - \tau^2$ = "at least one wrong".
- 단, 둘 다 틀렸는데 **원래 같은 class(Fox, Fox)** 이면 AL이 같고 LL도 같으니 FP가 아님 → 이 확률 $\tau^2/(C-1)^2$ 를 제외(Excluded).
- 따라서 $P_{FP} = 2\tau - \tau^2 - \frac{\tau^2}{(C-1)^2} \approx 2\tau$. 즉 **FP 비율은 label error rate의 약 2배**(pair에서 두 샘플 중 하나만 틀려도 발생).
- ← "At least one wrong" 안에서 "같은 AL을 가질 확률"에 대한 조건화는 슬라이드가 생략(시각화 단순화로 보임). III의 경우 예시가 Cat/Cat 인데 AL이 같을 확률은 별도 계산이 필요한 근사.

## The Problem with Easy Positives (p.71)

- Problem
  - Human labeling errors often satisfy the following two conditions simultaneously:
    - Same assigned label
    - High feature similarity
  - → Consequently, the model may classify them as high-confidence positives.
- Risks associated with existing noise mitigation methods:
  - Confident pairs are assigned higher weights.
  - Human labeling errors can appear highly confident.
  - The model reinforces false-positive relationships.
- Generally, "positives" located nearby are considered reliable. However, human labeling errors involve samples that humans confused due to visual similarity; consequently, they may already be close to the anchor in the representation space.
- 하단 굵은 문장: "High similarity does not always imply a correct positive pair."

**그림 서술**: 중앙 왼쪽에 **Anchor**(골든 리트리버 얼굴, 파란 테두리) 주위로 점선 원(anchor 주변 이웃 영역). 점선 선분으로 연결된 세 이미지:
- **Easy positive**(초록 테두리, 웃는 골든 리트리버 얼굴): anchor 가까이(원 바로 바깥 위쪽, 점선으로 연결).
- **False positive**(빨강 테두리, 여우 사진): anchor 가까이(원 바로 바깥 아래쪽, 점선으로 연결) → 사람이 Dog로 잘못 라벨링.
- **Hard positive**(초록 테두리, 물 위를 뛰는 골든 리트리버): anchor에서 멀리(원 바깥, 오른쪽), 긴 점선.

**설명**
- 기존 noisy-label 방법의 암묵적 가정: **similarity가 높거나 모델이 confident한 pair = 신뢰 가능한 positive** → 가중치를 높임(small-loss trick, confidence 기반 weighting 등).
- 하지만 human error는 **시각적으로 닮아서 틀린** 것이므로 처음부터 embedding에서 anchor와 가깝고, label도 같음 → 모델이 high-confidence positive로 간주 → **오히려 FP 관계를 강화**.
- 반대로 진짜 positive 중 **hard positive**(포즈/배경이 달라 멀리 있는 같은 class 샘플)가 정보량이 큼.
- 결론: **similarity ≠ correctness**. confidence-based 재가중은 human noise에서 역효과.
- > 보충: self-supervised에서도 "hard positive가 더 유용하다"는 논의가 있음(예: Tian et al., "What makes for good views" 등 view 난이도 논의). 슬라이드엔 이 인용 없음.

## SCL-RHE: Deprioritizing Easy Positives (p.72)

- Core Idea: High similarity → Lower sampling weight
  - 'Easy positives' that are already close to the anchor provide little new information for representation learning; moreover, they may include instances of human labeling error.
  - SCL-RHE reduces the weight assigned to these examples and places greater emphasis on 'hard positives' → samples belonging to the same class that are currently distant from the anchor.
- 수식:

$$q^+(x^+) := q\big(x^+ \mid z(x)=z(x^+)\big) \;\propto\; \frac{1}{e^{\beta f(x)^T f(x^+)}} \cdot p_x^+(x^+)$$

- 수식 주석(빨간 박스):
  - $q^+(x^+)$: The positive sampling distribution to be used by SCL-RHE
  - $z(x)=z(x^+)$: The condition that the anchor and the positive candidate share the same actual class.
  - $f(x)^T f(x^+)$: Similarity between anchor and positive
  - $p_x^+(x^+)$: true positive distribution
- 표:

| Positive type | Similarity | Weight |
|---|---|---|
| Easy positive | High | Low |
| Hard positive | Low | High |

**설명**
- **SCL-RHE**(Robust Hard-positive Emphasis로 추정 ← 슬라이드에 약자 풀이 없음, 논문 확인 필요) = positive를 **샘플링하는 분포**를 바꿔서 easy positive를 억제.
- 기호:
  - $x$: anchor, $x^+$: positive 후보, $z(\cdot)$: latent(true) class, $f(\cdot)$: encoder(보통 L2-normalized embedding), $\beta \ge 0$: 집중도(temperature 비슷한 hardness 하이퍼파라미터).
  - $f(x)^T f(x^+)$: cosine similarity (정규화 시).
  - $p_x^+(x^+)$: anchor와 같은 latent class에서 균등(또는 data-dist)으로 뽑는 **true positive distribution**.
- 식의 의미: $1/e^{\beta s} = e^{-\beta s}$ 이므로 **similarity $s$가 클수록 샘플링 확률이 지수적으로 감소**, 작을수록(멀수록) 커짐 → hard positive 강조. $\beta=0$이면 원래 SupCon(uniform positive)과 같아짐.
- 이 형태는 **Robinson et al. (ICLR 2021) "Contrastive Learning with Hard Negative Samples"** 의 negative 분포 $q^-(x^-) \propto e^{\beta f(x)^T f(x^-)}\cdot p(x^-)$ 와 **부호만 반대**(negative는 가까울수록, positive는 멀수록 강조)인 대칭 구조.
  - > 보충: Robinson et al.의 hard negative sampling은 label 없이 importance weighting으로 근사함. SCL-RHE의 positive 버전도 동일하게 sampling을 loss의 importance weight로 구현하는 방식으로 이해하면 됨(슬라이드에는 구현 설명 없음).
- 왜 이렇게 설계했나: (1) easy positive는 이미 가까워 gradient/정보가 작음, (2) human error 샘플이 여기에 몰려 있음 → 가중치를 낮추면 노이즈 영향도가 줄 것으로 기대됨. **noise robustness + hard positive mining**을 동시에 노리는 설계(효율 향상은 슬라이드에 없는 추측).
- 기존 방법과의 차이: 기존은 confident pair에 **높은 가중치**(p.71), SCL-RHE는 반대로 similar할수록 **낮은 가중치**.
- ← 주의(추측, 슬라이드에 없음): 멀리 있는 샘플의 가중치가 커지므로 synthetic noise처럼 멀리 있는 오라벨 샘플은 오히려 강조될 위험이 있음. 이 방법은 human noise(가까운 오류)를 겨냥한 것.
- ← 슬라이드 수식에서 조건 $z(x)=z(x^+)$ 는 latent class를 안다는 가정처럼 보이나 실제로는 알 수 없으므로 assigned label로 근사했을 것 (슬라이드에 명시 없음).

## FIGURES

- p.67: AL/LL 기반 CE vs SCL adverse impact 도식(FP/FN 케이스 I~VI) — 상
- p.68: human vs synthetic label error similarity density 4패널 — 상
- p.69: FP/FN 정의 + $P_{FP}$, $P_{FN}$ 수식과 케이스 그림 — 상
- p.70: $P_{FP}$ 케이스 분해 도식(I, II, III, Excluded) — 상
- p.71: anchor 주변 easy / hard / false positive 개념도 — 중
- p.72: SCL-RHE 샘플링 분포 수식 + easy/hard 표 — 상

## EXAM POINTS

1. **CE vs SCL의 label error 영향 차이**: CE는 sample 단위 오답 신호, SCL은 **pair relationship** 단위. mislabeled 샘플이 항상 잘못된 pair를 만들지는 않음 (Assigned label vs Latent label 구분).
2. **False positive / False negative 정의**: FP = AL 같음 & LL 다름 (잘못 pull), FN = AL 다름 & LL 같음 (잘못 push).
3. **왜 FP가 지배적인가**: $P_{FP} \approx 2\tau$, $P_{FN} \approx 2\tau/(C-1)$ → C가 클수록 FP 비중 증가 (CIFAR-100 99.04%, ImageNet-1K 99.91%; p.73 수치).
4. **$P_{FP}$ 유도**: at least one wrong $2\tau-\tau^2$ = $(1-\tau)\tau+\tau(1-\tau)+\tau\tau$, 둘 다 틀렸지만 같은 latent class인 경우 $\tau^2/(C-1)^2$ 제외.
5. **Human error vs synthetic error**: human error는 시각적으로 비슷한 class를 혼동 → anchor와 similarity가 높아 true positive 분포와 overlap, synthetic은 낮은 similarity.
6. **Easy positive 문제 (why)**: human error는 same AL + high similarity → high-confidence positive로 오인 → confidence 가중 기법이 FP를 강화. "High similarity ≠ correct positive pair".
7. **SCL-RHE 식 설명**: $q^+(x^+) \propto e^{-\beta f(x)^T f(x^+)} \cdot p_x^+(x^+)$, similarity가 높을수록 sampling weight 낮음 → hard positive 강조, $\beta$ 역할(0이면 uniform).
8. **Robinson et al. hard negative와의 비교**: negative는 가까울수록 가중(부호 +), positive는 멀수록 가중(부호 −).

## VERIFIER REPORT
- p.67 그림 서술: III을 False Negative로 분류한 오류 수정 (III은 FP, IV~VI이 FN)
- p.69 그림 서술: I,II,III=FP / IV~VI=FN으로 수정 (기존 서술이 자기 모순)
- p.69 설명: FN이 드문 이유 직관이 틀려 수정 (latent class 일치 확률 1/C)
- p.71 그림 서술: easy/false positive는 점선 원 바깥에 위치하므로 수정
- p.72 설명: 근거 없는 '학습 효율 증가' 과장 완화
- p.72 설명: '주의' 항목의 근거 없는 논리를 추측으로 표시하고 정정
