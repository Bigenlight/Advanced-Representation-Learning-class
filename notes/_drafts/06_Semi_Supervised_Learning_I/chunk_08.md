<!-- DRAFT | lecture: 06_Semi_Supervised_Learning_I | chunk: 08 | pages 55–65 -->
<!-- VERIFIED | fixes: 2 | verifier: sonnet -->

## Case Study 섹션 구분 (p.55)

슬라이드 텍스트:
- 제목: "What if the teacher becomes too similar to the student?" (페이지 번호 55, 질문형 섹션 구분 슬라이드)

**설명**
- 이후 p.56~65는 **Mean Teacher의 한계(teacher–student coupling)**를 푸는 논문 하나(Dual Teacher)를 case study로 다룸.
- 이 슬라이드의 질문이 chunk 전체의 문제의식: **EMA teacher가 student와 너무 비슷해지면 무슨 문제?**

---

## Paper: Switching Temporary Teachers for Semi-Supervised Semantic Segmentation (p.56)

슬라이드 텍스트:
- 논문 제목 페이지: **Switching Temporary Teachers for Semi-Supervised Semantic Segmentation**
- 저자: Jaemin Na, Jung-Woo Ha, Hyung Jin Chang, Dongyoon Han, Wonjun Hwang (Ajou Univ., NAVER AI Lab, Univ. of Birmingham)
- 출처: Na et al., NeurIPS 2023

**설명**
- 줄여서 **Dual Teacher**. **semi-supervised semantic segmentation**(labeled 적고 unlabeled 많음, pixel-level 예측)에서 teacher–student 구조를 개선한 논문.
- 인용 정보는 시험에서 이름 매칭용으로 외워둘 것: **Na et al., NeurIPS 2023**.

---

## What if the teacher becomes too similar to the student? (p.57)

슬라이드 텍스트:
- Mean Teacher updates the teacher with an exponential moving average of the student.
- The teacher gradually follows the student throughout training.
- Their parameters and predictions can become increasingly similar.
- A highly similar teacher provides little new information.
- The teacher may also inherit the student's errors.

**설명**
- **Mean Teacher**: teacher 파라미터 = student 파라미터의 EMA. 안정적인 target을 주는 게 장점이었음.
- 문제: 학습이 진행될수록 teacher가 student를 **계속 따라가서** 파라미터·예측이 **거의 같아짐 (coupling)**.
- 결과 1: **새로운 정보가 없음** ← teacher의 target이 student 자기 예측과 거의 같으면 consistency loss가 0에 가까워져 supervision 효과 소멸.
- 결과 2: **student의 오류를 그대로 상속** ← confirmation bias. student가 틀린 pseudo-label을 teacher가 따라 배우고, 다시 student에게 돌려줌 (오류 강화 루프).
> 보충: Mean Teacher의 EMA는 $\theta_t \leftarrow \alpha\theta_t + (1-\alpha)\theta_s$ (p.60). $\alpha$가 커도 결국 student의 시간 평균이므로 student와 같은 궤적을 따라가는 한 coupling은 피하기 어려움 (슬라이드엔 없음).

---

## Teacher–Student Coupling (p.58)

슬라이드 텍스트:
- **Single Teacher**
  - One student and one permanent EMA teacher
  - The teacher continuously follows the same student
- **Dual Student**
  - Two independently trained students
  - Higher training and memory cost
- **Dual Teacher**
  - One student and two temporary teachers
  - Only one teacher guides the student at a time
- 그림: (a) Teacher-Student: $x$ → strong aug $x_s$ → Student, weak aug $x_w$ → Teacher, Student→Teacher로 EMA, Teacher→Student로 Label. (b) Dual Student: Student #1, #2 두 개가 서로 Label 주고받음. (c) Dual Teacher (ours): Student 하나, Teacher #1 / Teacher #2 두 개, 각각 다른 strong aug $x_{s1}$, $x_{s2}$.
- 강조 문장: **Dual Teacher introduces diversity without training a second student.**

**설명**
- 세 가지 구조 비교 (시험 단골):

| 구조 | student | teacher | 특징 |
|---|---|---|---|
| Single Teacher (Mean Teacher) | 1 | 1 (permanent EMA) | teacher가 항상 같은 student를 따라감 → coupling |
| Dual Student | 2 (독립 학습) | 없음 (서로가 서로를 가르침) | diversity 있지만 **학습·메모리 비용 ↑** |
| Dual Teacher | 1 | 2 (temporary, 한 번에 하나만 active) | **student 하나만 학습**하면서 diversity 확보 |

- 핵심 설계 동기: diversity(서로 다른 예측)를 얻고 싶은데, Dual Student처럼 student를 2개 학습하면 비싸다. **teacher는 EMA라 gradient 학습이 필요 없으므로** teacher를 늘리는 건 거의 공짜.
- "temporary"의 뜻: teacher가 항상 student를 따라가는 게 아니라, **자기 차례(active)일 때만** student로부터 업데이트되고 target을 생성함.

---

## Switching Temporary Teachers (p.59)

슬라이드 텍스트:
- Two EMA teachers take turns guiding one student.
- Only the active teacher generates the target for the current epoch.
- The active teacher is updated from the current student.
- The student receives a different strong augmentation when the teacher changes.
- 그림: (c) Dual Teacher (ours) — 왼쪽 Student($x_{s1}$ 입력)와 Teacher #1(weak aug $x_w$ 입력)이 EMA/Label로 연결, 오른쪽 Student($x_{s2}$ 입력)와 Teacher #2가 EMA/Label로 연결.
- 표:

| Epoch | Active teacher | Student augmentation |
|---|---|---|
| 1 | Teacher 1 | ClassMix |
| 2 | Teacher 2 | CutMix |
| 3 | Teacher 1 | ClassMix or CutMix |
| 4 | Teacher 2 | The other augmentation |

- 강조 문장: **Each teacher preserves the student's characteristics from different training stages.**

**설명**
- 동작: epoch마다 teacher를 교대 (T1, T2, T1, T2, ...). **active teacher만** (i) pseudo-label target을 만들고 (ii) student로부터 EMA 업데이트됨. inactive teacher는 그 epoch 동안 **그대로 멈춰 있음**.
- 왜 diversity가 생기나? T1은 홀수 epoch의 student 상태들만, T2는 짝수 epoch의 student 상태들만 EMA로 흡수 → 서로 **다른 시점의 student 특성**을 보존. 또 쉬는 동안 student가 앞서 나가므로 teacher가 student를 쫓아가기만 하는 관계가 끊김.
- student의 strong augmentation도 teacher가 바뀔 때 함께 바뀜 (ClassMix ↔ CutMix). → teacher마다 서로 다른 형태의 학습 신호(다른 augmentation에서 생기는 오류 패턴)와 짝지어져 diversity ↑.
> 보충: **CutMix**는 이미지 한 장의 사각형 영역을 다른 이미지로 교체(label도 같은 영역 교체), **ClassMix**는 한 이미지의 특정 class pixel들(segmentation mask 기준)을 다른 이미지에 붙여넣는 augmentation. 둘 다 semi-supervised segmentation의 strong augmentation (슬라이드엔 정의 없음).
> 보충: 표의 epoch 3, 4는 "둘 중 아무거나 / 나머지 하나"로 되어 있어 이후 epoch은 두 augmentation이 번갈아 쓰이는 패턴으로 이해하면 됨.

---

## What Changes from Mean Teacher? (p.60)

슬라이드 텍스트:
- EMA of Dual Teacher
- Mean Teacher EMA:
$$\theta_t \leftarrow \alpha\,\theta_t + (1-\alpha)\,\theta_s$$
- ⇓ Dual Teacher:
$$k_e \in \{1, 2\}$$
$$\theta_{t_{k_e}} \leftarrow \alpha\,\theta_{t_{k_e}} + (1-\alpha)\,\theta_s$$
- 강조 문장: **The EMA rule remains the same. The teacher schedule changes.**

**설명**
- $\theta_s$: student 파라미터, $\theta_t$: teacher 파라미터, $\alpha$: EMA decay (momentum).
- $k_e \in \{1,2\}$: **epoch $e$에서 active인 teacher의 index**. 업데이트 수식은 Mean Teacher와 **완전히 동일**하되 대상이 $\theta_{t_{k_e}}$ 하나로 바뀐 것.
- 즉 바뀐 건 **수식(EMA rule)이 아니라 schedule**: 어느 teacher가, 언제 student를 따라가는가. 구현이 간단한 이유.
- 한 teacher의 입장에서 보면 $\alpha$-EMA가 **자기 epoch에서만** 일어나므로 student의 시간 평균을 서로 다른 구간에서 취함.
> 보충: 논문 기준 $k_e$는 epoch의 홀짝으로 정해짐 (예: $k_e = (e \bmod 2) + 1$ 꼴; 슬라이드엔 명시 없음).

---

## Training the Student (p.61)

슬라이드 텍스트:
- The active teacher receives a weakly augmented image.
- It generates a pixel-level target.
- The student receives a strongly augmented version.
- The student learns to match the teacher's target.
- 수식:
$$\mathcal{L} = \mathcal{L}_{\text{sup}} + \lambda_u \mathcal{L}_{\text{unsup}}$$
$$\mathcal{L}_{\text{unsup}} = \frac{1}{B_u HW}\sum_{i=1}^{B_u}\sum_{j=1}^{HW} \mathcal{L}_{\text{CE}}\left(p^u_{ij},\, \hat{y}^u_{ij}\right)$$
- 기호:
  - $B_u$: number of unlabeled images
  - $H \times W$: number of pixels per image
  - $p^u_{ij}$: student prediction for pixel $j$ (of image $i$)
  - $\hat{y}^u_{ij}$: target from the active teacher
  - $\lambda_u$: weight of the unlabeled loss
- 강조 문장: **The objective resembles Mean Teacher. The supervision source changes over time.**

**설명**
- 전체 loss = **supervised loss (labeled data) + $\lambda_u \cdot$ unsupervised loss (unlabeled data)**.
- $\mathcal{L}_{\text{unsup}}$: unlabeled batch의 모든 pixel에 대해 student 예측 $p^u_{ij}$와 teacher가 만든 **pseudo-label** $\hat{y}^u_{ij}$ 사이의 cross-entropy를 **평균** ($\frac{1}{B_u HW}$ 정규화).
- 흐름: unlabeled image → (weak aug) → active teacher → pixel-wise 예측 → $\hat{y}$ (hard pseudo-label) / (strong aug: ClassMix or CutMix) → student → $p$ → CE.
- **weak → teacher, strong → student**: FixMatch류의 consistency regularization 구도. teacher는 깨끗한 입력에서 신뢰할 만한 target을 주고, student는 어려운 입력에서도 같은 답을 내도록 강제.
- Mean Teacher와의 유일한 차이: $\hat{y}$를 만드는 teacher가 **epoch마다 바뀜** ("supervision source changes over time").
> 보충: 슬라이드의 $j=1..HW$ 인덱스는 pixel index. 표기상 $p^u_{ij}$의 $j$를 "pixel $j$"로 설명함. $\hat{y}$가 argmax hard label인지 soft인지는 슬라이드에 명시 안 됨 (논문은 pseudo-label로 CE 사용).

---

## Does Switching Actually Prevent Coupling? (p.62)

슬라이드 텍스트:
- **Figure 2(a): Prediction distance**
  - Single Teacher stays very close to the student.
  - Dual Teacher maintains a larger prediction distance.
  - Switching prevents one teacher from continuously following the student.
- **Figure 2(b): Class-wise IoU**
  - The two teachers have similar overall mIoU.
  - However, their strengths differ across classes.
  - They can provide complementary supervision.
- 그림 (a): x축 Training epochs (0~80), y축 Prediction distance (0~0.07). 파랑 = Dual Teacher: 초반 ~0.07에서 시작해 진동(zigzag)하며 서서히 감소해 80 epoch 근처에서 ~0까지 수렴. 주황 = Single Teacher: 거의 0 근처(~0.002)의 평평한 선.
- 그림 (b): class별 IoU(%) 막대그래프 (Teacher #1 파랑 / Teacher #2 주황). 대략 cow: T1 ~59, T2 ~78 / dog: T1 ~61, T2 ~53 / bottle: T1 ~62, T2 ~70 / cat: T1 ~63, T2 ~57 / sheep: T1 ~79, T2 ~76. (수치는 그래프 눈대중)
- 강조 문장: **Teacher diversity means different predictions, not simply different parameter values.**

**설명**
- (a): **prediction distance**는 teacher 예측과 student 예측이 얼마나 다른지의 척도. Single Teacher는 거의 0 → **teacher가 student와 사실상 같다 = coupled**. Dual Teacher는 더 큼 → 교대 때문에 gap이 유지됨. 파랑 선의 zigzag는 **teacher가 교대할 때마다(epoch 단위) 거리가 오르내리는** 패턴으로 해석 가능 (보충).
- 후반(80 epoch 근처)에는 Dual도 0에 가까워짐 ← student가 수렴하며 lr이 감소하고 teacher들도 student에 가까워지기 때문 (보충 해석).
- (b): 두 teacher의 **전체 mIoU는 비슷**한데 **class별 강점이 다름** (T2는 cow/bottle에서 강하고 T1은 dog/cat에서 강함). → 서로 **complementary**: 한 teacher의 약점을 다른 teacher가 보완. 그래서 같은 정확도의 teacher 둘이 student에 줄 수 있는 정보가 서로 다름.
- 마지막 문장의 의미: diversity의 척도는 **파라미터 거리가 아니라 prediction 차이**. 파라미터가 달라도 예측이 같으면 쓸모없음.
> 보충: 슬라이드엔 prediction distance의 정확한 정의(예: 확률 분포 간 거리)가 없음. 시험에서는 "teacher와 student 예측이 얼마나 다른가"로 설명하면 충분.

---

## Why Switch Instead of Ensemble? (p.63)

슬라이드 텍스트:
- **Teacher ensemble**
  - Both teachers guide the student simultaneously.
  - Their predictions are combined every epoch.
  - They can gradually lose their distinct characteristics.
- **Teacher switching**
  - One teacher guides the student at each epoch.
  - Each teacher observes a different student state.
  - Their temporal roles remain separated.
- 표 (Table 7: Performance analysis of different teaching strategies on PASCAL VOC 2012 across various partitions. Labeled images are sampled from the *blender* set and all methods are based on DeepLabv3+ with ResNet-50.)

| Method | 1/16 (186) | 1/8 (372) | 1/4 (744) | 1/2 (1,488) |
|---|---|---|---|---|
| Switching teachers | 77.00 | 79.23 | 78.94 | 80.18 |
| Ensembling teachers | 73.25 | 75.99 | 76.41 | 77.34 |

- 강조 문장: **Two teachers help only when they preserve different perspectives.**

**설명**
- Ensemble은 두 teacher를 **동시에** 쓰고 예측을 합치는데, 매 epoch 둘 다 student를 따라가므로 시간이 지나면 **둘이 비슷해져 개별 특성을 잃음** (결국 coupling으로 회귀).
- Switching은 한 번에 한 teacher만 student를 보므로 **각 teacher가 서로 다른 student 상태를 관찰**하고 **temporal role이 분리**됨 → diversity가 유지.
- 표: 모든 partition에서 switching이 ensembling보다 높음 (1/16: **77.00 vs 73.25**, 약 +3.8; 1/8: 79.23 vs 75.99; 1/4: 78.94 vs 76.41; 1/2: 80.18 vs 77.34). 즉 **teacher 수가 같아도 쓰는 방식(switch vs ensemble)이 성능을 좌우**.
- 수치 관찰: 1/8 (79.23) > 1/4 (78.94)로 labeled 비율이 늘어도 단조 증가가 아님 ← 의심: 슬라이드 원문 수치. 원 논문에서도 그렇게 보고되었는지는 슬라이드만으로 확인 불가 (원문 유지).
> 보충: 괄호 안 숫자는 labeled image 수 (186, 372, 744, 1,488). VOC blender set은 augmented/blender 데이터 기반 labeled 샘플링 설정. 지표는 mIoU(%)로 추정 (슬라이드에 단위 명시 없음).

---

## How Many Teachers Are Needed? (p.64)

슬라이드 텍스트:
- Two teachers outperform one teacher.
- A third teacher provides no additional gain.
- Two augmentation types perform better than one.
- More augmentation types do not guarantee better performance.
- 표 (Table 5: The performance relative to the number of teacher models and augmentations on 1/16 splits of the PASCAL VOC *blender* set.)

| Method | Single | Dual | Triple |
|---|---|---|---|
| Aug × 1 | 73.63 | 75.34 | 75.30 |
| Aug × 2 | 74.21 | **77.00** | 76.56 |
| Aug × 3 | 74.49 | 76.25 | 75.94 |

- 강조 문장: **The best result comes from two teachers and two augmentation types.**

**설명**
- 행 = augmentation 종류 수, 열 = teacher 수 (Single=1, Dual=2, Triple=3).
- 같은 Aug에서 Single → Dual이 크게 증가 (Aug×2: 74.21 → **77.00**), Dual → Triple은 오히려 소폭 하락 (77.00 → 76.56) → **세 번째 teacher는 이득 없음** (오히려 teacher당 student 관찰 빈도가 줄어 각자의 역할이 희석되는 것으로 해석 가능 — 보충).
- Aug도 1 → 2에서 이득 (Dual: 75.34 → 77.00), 3개로 늘리면 76.25로 **하락** → "more is not always better".
- 최적 = **Dual teacher + 2 augmentation (77.00)**. 이게 Dual Teacher의 기본 설정이고 p.59의 ClassMix/CutMix 두 개와 일치.
- 주의: Single Teacher에서는 Aug를 늘려도 73.63 → 74.21 → 74.49로 완만하게 증가 → augmentation 증가의 이득은 **teacher가 여러 개일 때 시너지**가 큼.

---

## Takeaway: Preserve a Useful Gap (p.65)

슬라이드 텍스트:
- **Mean Teacher**
  - Stabilizes targets through temporal averaging
  - May gradually couple the teacher and student
- **Dual Teacher**
  - Alternates two temporary EMA teachers
  - Preserves different temporal perspectives
  - Provides complementary supervision to one student
- 강조 문장: **A teacher should be stable, but it should not become identical to the student.**

**설명**
- 한 줄 요약: **stability와 diversity의 trade-off**. EMA는 target을 안정시키지만(장점) student와 너무 가까워지면 정보가 없어짐(단점).
- Dual Teacher는 EMA를 유지(안정성)하면서 **교대 스케줄**로 teacher–student 간 **useful gap**을 보존 (다양성). 추가 비용은 teacher 하나 분량의 파라미터뿐이고 student는 하나만 학습.
- 시험 답안 한 줄: "Mean Teacher couples teacher and student through EMA; Dual Teacher alternates two temporary EMA teachers so each preserves a different temporal perspective, giving complementary targets to a single student."

---

## FIGURES

- p.58: 세 구조 비교 그림 (a) Teacher-Student, (b) Dual Student, (c) Dual Teacher, 중요도 상
- p.59: Dual Teacher 구조 그림 + epoch별 active teacher / student augmentation 표, 중요도 상
- p.60: EMA 업데이트 수식 (Mean Teacher → Dual Teacher, $k_e \in \{1,2\}$), 중요도 상
- p.61: $\mathcal{L}=\mathcal{L}_{\text{sup}}+\lambda_u\mathcal{L}_{\text{unsup}}$ 및 pixel-wise CE 수식, 중요도 상
- p.62: Figure 2(a) prediction distance 곡선 (Single vs Dual), 중요도 상
- p.62: Figure 2(b) class-wise IoU 막대그래프 (Teacher #1 vs #2), 중요도 중
- p.63: Table 7 Switching vs Ensembling teachers (VOC 2012, 4 partitions), 중요도 중
- p.64: Table 5 teacher 수 × augmentation 수 성능표, 중요도 중

## EXAM POINTS

1. **Mean Teacher의 문제 = teacher–student coupling**: EMA 때문에 teacher가 student를 따라가 파라미터·예측이 비슷해짐 → 새 정보 부족 + student 오류(confirmation bias) 상속.
2. **세 구조 비교**: Single Teacher(1 student + 1 permanent EMA teacher), Dual Student(2 students 독립 학습, 비용↑), Dual Teacher(1 student + 2 temporary EMA teachers, 한 번에 1명만 guide). Dual Teacher의 장점: **second student 학습 없이 diversity** 확보.
3. **Switching 동작**: epoch마다 active teacher 교대, active teacher만 target 생성 + student로부터 EMA 업데이트, teacher 교체 시 student의 strong augmentation도 교체 (ClassMix ↔ CutMix).
4. **EMA 수식**: Mean Teacher $\theta_t \leftarrow \alpha\theta_t+(1-\alpha)\theta_s$ → Dual Teacher $\theta_{t_{k_e}} \leftarrow \alpha\theta_{t_{k_e}}+(1-\alpha)\theta_s$, $k_e\in\{1,2\}$. **EMA rule은 그대로, schedule만 변경**.
5. **Loss**: $\mathcal{L}=\mathcal{L}_{\text{sup}}+\lambda_u\mathcal{L}_{\text{unsup}}$, $\mathcal{L}_{\text{unsup}}=\frac{1}{B_uHW}\sum_i\sum_j \mathcal{L}_{\text{CE}}(p^u_{ij},\hat y^u_{ij})$. teacher는 weak aug 입력, student는 strong aug 입력, pixel-level pseudo-label.
6. **Figure 2 해석**: (a) Single Teacher의 prediction distance는 거의 0, Dual Teacher는 더 큼 (switching이 coupling을 방지). (b) 두 teacher의 mIoU는 비슷하지만 class별 강점이 달라 complementary.
7. **Diversity의 정의**: "different predictions, not simply different parameter values".
8. **Switching vs Ensemble**: ensemble은 둘을 동시에 써서 개별 특성이 사라짐. Table 7: 1/16 기준 **77.00 vs 73.25**, 모든 partition에서 switching 우세.
9. **Teacher/augmentation 수 ablation (Table 5)**: 2 teachers > 1 teacher, 3 teachers 이득 없음 (77.00 → 76.56), augmentation도 2종이 최적 (3종은 76.25로 하락). **최적 = Dual + Aug×2 = 77.00**.
10. **Takeaway**: teacher는 안정적이어야 하지만 student와 **동일해지면 안 됨** ("preserve a useful gap").
11. 예상 서술형: "Explain why Dual Teacher alleviates confirmation bias in Mean Teacher and why switching is preferred over ensembling." — 답안 키워드: temporary EMA teachers, different temporal perspectives, complementary supervision, prediction distance, no second student cost.
