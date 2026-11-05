<!-- DRAFT | lecture: 08_Weakly_Supervised_Learning_I | chunk: 04 | pages 33–41 -->
<!-- VERIFIED | fixes: 2 | verifier: opus -->

## Section: Deep and End-to-End WSOD (p.33)

슬라이드 텍스트:
- Deep and End-to-End WSOD

**설명**
- 섹션 구분 슬라이드야.
- **end-to-end** = 입력부터 출력까지 network 전체를 loss 하나의 gradient로 같이 학습하는 방식이야.
- 앞 파트는 feature를 뽑아 놓고 box 고르기와 학습을 번갈아 했어. 여기선 그 과정을 network 안으로 넣어.

---

## Weakly Supervised Deep Detection Network (WSDDN) (p.34)

슬라이드 텍스트:
- Previous work uses CNN as black-box feature extractor
- First end-to-end WSOD method, finetunes a pre-trained deep network
- Uses RoI-Pooling
- Two stream architecture
  - Classification: Associates each region with a class score
  - Localization: Compares regions for each class
- 그림: 사진 → Conv layers → feature map, Region Proposals가 feature map 위에 표시 → RoIPooling → 두 표. 위 표 "classification", 아래 표 "localization" (행 = horse, person / 열 = $R_1$–$R_4$) → 두 표를 ⊗로 결합 → 오른쪽 class별 칸 (horse, person)
- [Bilen CVPR 16]

**설명**
- **한 줄**: image label만으로 region 점수까지 같이 학습하는 첫 end-to-end WSOD야.
- **문제**: 이전 방법은 CNN을 **black-box feature extractor**로만 썼어. network 자체는 detection에 맞게 학습되지 않았어.
- **RoI pooling** = 크기가 제각각인 region을 feature map에서 잘라 같은 크기의 feature로 맞추는 연산이야.
- **stream** = network 안의 갈래야.
- **두 stream의 역할**
  - **Classification stream**: region마다 "**어느 class인가**"를 점수로 내. region 하나 안에서 class끼리 비교해.
  - **Localization stream**: class마다 "**어느 region인가**"를 비교해. class 하나 안에서 region끼리 비교해.
- 두 표를 곱해서 region별 점수를 만들고, 그걸 모아서 **image-level 점수**를 내. 그 점수에 image label로 학습해.

> 보충: p.36 그림의 "Multiple instance detection network"가 이 구조를 말로 적어 놨어. Softmax over classes, Softmax over proposals, Element-wise product → Proposal scores, Sum over proposals → Image scores야. 이걸 식으로 옮기면 region $r$, class $c$에 대해
> $$s_{c,r} = \text{softmax}_{c}(\cdot)\ \cdot\ \text{softmax}_{r}(\cdot), \qquad \phi_c = \sum_r s_{c,r}$$
> 야. 식 자체는 슬라이드에 없어. 슬라이드엔 학습 loss 식도 없어.

**왜 MIL과 이어지나**
- region이 instance고, 사진이 bag이야. region 점수를 합쳐서 bag 점수를 내면 bag label로 학습할 수 있어.
- p.21의 hard max와 달리 여러 region이 점수에 기여해.

---

## Weakly Supervised Deep Detection Network (WSDDN) (p.35)

슬라이드 텍스트:
- (+) State-of-the-art results with VGG-F (~AlexNet) (62% of supervised)
- (+) End-to-end learning + No custom deep learning layers
- (-) Detects discriminative parts rather than whole object extent
- (-) Groups multiple instances together
- Limitations (compared to the supervised case, eg Faster RCNN):
  - It does not leverage multiple positive instances in positive images
  - It does not exploit negative instances in positive images
  - It does not learn bounding box regression
- 그림: 예시 3장. 사람 얼굴만 잡은 box, 고양이 머리 쪽만 잡은 box, 여러 사람을 한 box로 묶은 사진
- [Bilen CVPR 16]

**설명**
- **discriminative part** = class를 구분하는 데 제일 도움 되는 부위야. 사람이면 얼굴이야.
- **bounding box regression** = box의 위치·크기를 조금씩 고쳐서 물체에 맞추는 학습이야.
- **단점 두 개**는 p.11, p.13의 ambiguity가 그대로 나타난 거야.
  - 얼굴만 잡기 ↔ Parts vs object
  - 여러 물체를 한 box로 ↔ Multiple vs single instance
- **왜 이렇게 돼?** loss가 image-level에만 걸려. 가장 class다운 부위만 맞혀도 image 점수가 잘 나와.
- **Faster RCNN 대비 한계 세 개**. 앞의 두 개는 p.21의 "Difference to supervised object detection"과 같은 말이야.
  1. positive image의 **여러 positive instance**를 쓰지 않아.
  2. positive image의 **negative instance**를 쓰지 않아.
  3. **bounding box regression**을 학습하지 않아.
- 뒤의 방법들은 이 한계를 고치는 쪽이야 (p.36–38).

**시험 관점**
- "+"와 "−"를 뒤집는 지문을 조심해. "Detects whole object extent"는 틀린 문장이야.
- VGG-F, 62% 같은 숫자는 비중이 낮아.

---

## Incorporating Instance Classification: OICR (p.36)

슬라이드 텍스트:
- WSDDN learns to classify images based on implicit instance scoring
- Online Instance Classifier Refinement (OICR)
- OICR learns to classify instances
  - Initially uses WSDDN to set instance labels
  - Iteratively refines them by using an instance classifier
- WSDDN -> Pseudo labels -> Instance classifier -> Better pseudo labels -> Better instance classifier --> ..
- ç ← 슬라이드 맨 아래 citation 옆에 깨진 글자처럼 보이는 "ç"가 있어 (원문 유지, 내용 없음)
- 그림: 사진 → Conv layers → Conv feature map, Proposals → SPP layer → Two fc layers → Proposal feature vector. 여기서 갈래가 나뉘어
  - Multiple instance detection network: Fc layer → Softmax over classes / Fc layer → Softmax over proposals → Element-wise product → Proposal scores → Sum over proposals → Image scores
  - Instance classifier refinement, 1-st time: Fc layer → Softmax over classes → Proposal scores. Supervision은 위 network의 proposal scores에서 와.
  - …
  - Instance classifier refinement, K-th time: 같은 구조. Supervision은 앞 단계에서 와.
- [Tang CVPR 17]

**설명**
- **한 줄**: WSDDN 점수로 pseudo label을 만들고, instance classifier가 이를 단계적으로 고쳐 가는 방법이야.
- **문제**: WSDDN은 **image**를 분류하도록 학습해. instance 점수는 그 과정의 부산물이야 (implicit).
- **핵심**: instance를 **직접** 분류하는 classifier를 붙여.
- **pseudo label** = model이 만들어서 정답처럼 쓰는 label이야. L07에서 봤어.
- **어떻게**
  1. WSDDN으로 proposal 점수를 내.
  2. 그 점수로 instance label을 정해서 1번째 instance classifier를 학습해.
  3. 1번째 classifier의 점수로 2번째의 target을 만들어.
  4. $K$번째까지 이어 가.
- **차이**: WSDDN = image classification + implicit instance scoring, OICR = explicit instance classification.
- 그림에서 refinement 갈래가 모두 같은 proposal feature vector에서 나와. 즉 한 network 안에서 같이 학습되는 구조로 읽혀. 이름의 Online이 이 뜻으로 보여 ← 슬라이드엔 Online의 정의가 없어.

> 보충 (RL 비유, 슬라이드 밖 = 시험 범위 아님): 앞 단계 출력이 뒷 단계의 target이 되는 건 bootstrapping과 닮았어. 다만 여기선 target을 만드는 쪽이 앞 branch고, 슬라이드에 EMA 같은 target network는 없어.

**시험 관점**
- target을 누가 만드나: 1번째 refinement는 WSDDN 쪽 proposal scores, 그 뒤는 앞 refinement야.
- 중간고사 질문 "target은 어디서 오나?"가 여기서도 통해.

---

## Incorporating Instance Classification: PCL (p.37)

슬라이드 텍스트:
- Proposal Cluster Learning (PCL) for WSOD
  - Observation: There can be multiple positive instances in an image
  - OICR considers only the highest scoring proposal + surrounding ones as positives
  - PCL
    - builds an undirected graph of top- ranking proposals
    - iteratively selects vertices which have most connections to be the cluster centres
- OICR uses a single pseudo object per image, whereas PCL recognizes that an image may contain multiple object instances. It groups high-scoring proposals into clusters and uses each cluster as an independent pseudo object, leading to more accurate instance-level supervision.
- 그림 (a)(b)(c): 같은 사진의 proposal을 세 방식으로 묶어서 Object Detector에 넣어
  - (a) 전체를 묶은 $X$ 하나: $\{(X, [1,2])\}$
  - (b) proposal마다 label: $\{(x_1,0),(x_2,0)\ldots(x_4,1),(x_5,1)\ldots(x_7,1),(x_8,1)\ldots(x_{10},2),(x_{11},2)\ldots\}$
  - (c) cluster마다 label: $\{(X_1,0),(X_2,1),(X_3,1),(X_4,2)\}$
- [Tang PAMI 18]

**설명**
- **문제**: 사진에 물체가 여러 개일 수 있어. 그런데 OICR은 최고 점수 proposal과 그 주변만 positive로 봐. 즉 사진당 pseudo object가 **하나**야.
- **핵심**: 점수 높은 proposal을 **cluster**로 묶고, cluster 하나를 **독립된 pseudo object**로 써.
- **어떻게**
  1. top-ranking proposal로 **undirected graph**(방향 없는 graph)를 만들어. ← 슬라이드엔 edge를 잇는 기준이 없어.
  2. 연결이 제일 많은 vertex(점)를 cluster 중심으로 뽑아. 이걸 반복해.
  3. cluster 단위로 supervision을 줘.
- 그림을 읽으면 (a)는 사진 전체가 bag, (b)는 proposal 단위, (c)는 cluster 단위야. 0은 배경, 1·2는 class로 보여 ← (a)(b)(c)의 이름과 0의 뜻은 슬라이드에 없어. 그림에서 읽은 거야.
- (c)에서 class 1 cluster가 $X_2$, $X_3$ 둘이야. 같은 class 물체가 둘이어도 따로 잡는다는 뜻이야.

**시험 관점**
- **차이**: OICR = single pseudo object per image, PCL = multiple pseudo objects (clusters).
- cluster centre는 "**most connections**"로 뽑아. "highest score"로 바꾼 지문을 조심해.

---

## Regularizing Discriminative Parts (p.38)

슬라이드 텍스트:
- Instance-Aware, Context-Focused, and Memory-Efficient WSOD
  - Builds on WSDDN and OICR
    - Picks top p % of the highest scoring proposals and surrounding ones as positives
    - Introduces ConcreteDropBlock: a data-driven and parametric drop-out to drop the most relevant regions
    - Adversarial optimization

$$w^*, \theta^* = \arg\min_{w}\max_{\theta}\ \sum_{I}\ \mathcal{L}_{\text{img}}(w,\theta) + \mathcal{L}_{\text{roi}}(w,\theta)$$

← 시그마 아래 첨자는 슬라이드에서 잘려서 $I$로 읽은 거야.

- Unlike OICR and PCL that mainly improve pseudo-label quality, this work explicitly addresses the discriminative-part problem. By suppressing the most discriminative regions using Concrete DropBlock, the detector is forced to attend to the entire object rather than only its most salient parts, leading to better localization.
- 그림: Base → Concrete DropBlock → Neck → 세 갈래. 위 두 갈래 $f_w$, $g_w$ → ⊗ → $s_w$ → ⊕ → $\phi_w$. 아래 갈래 Student. MIST가 $s_w$에서 $\hat{y}^1, \hat{t}^1$을 만들어 Student로 보내. Student 안에 $\hat{s}_w$ (cls. prob.)와 $\hat{\mu}_w$ (regress.) 두 head. 오른쪽 상자: Student #N → MIST → $\hat{y}^N, \hat{t}^N$ → Student #N+1
- [Ren CVPR 20]

← 슬라이드는 "drop-out"으로 적었고 (보통 dropout), 그림의 MIST·Student·regress. head는 bullet에서 설명하지 않아 (원문 유지).

**설명**
- **문제**: 앞 방법들은 pseudo label의 **질**을 올렸어. p.35의 "discriminative parts" 문제는 정면으로 안 다뤘어.
- **핵심**: 제일 눈에 띄는 부위를 **일부러 가려서**, detector가 물체의 나머지도 보게 만들어.
- **dropout** = 학습 중 일부 값을 무작위로 꺼서 특정 값에 의존하지 못하게 하는 regularization이야.
- **ConcreteDropBlock**은 균등 무작위가 아니야. "data-driven and parametric"이라서 **어디를 끌지 학습해**. 학습된 확률 map에서 샘플링해서 가장 relevant한 region을 더 자주 꺼.
- **positive 고르기**: OICR(최고 점수 하나 + 주변)과 달리 **top p %** proposal과 그 주변을 positive로 써. 여러 물체를 놓치지 않으려는 쪽이야.
- **Adversarial optimization**: 식이 $\min_w \max_\theta$야. 한쪽은 loss를 줄이고 다른 쪽은 키워.
  - 가리는 쪽은 detector가 틀리도록 제일 중요한 곳을 가려.
  - detector는 가려져도 맞히도록 학습해.
- $\mathcal{L}_{\text{img}}$는 image 단위 loss, $\mathcal{L}_{\text{roi}}$는 RoI 단위 loss로 읽혀 ← 슬라이드에 정의는 없어.

> 보충: 슬라이드엔 $w$와 $\theta$의 정의가 없어. 논문(Ren et al., Sec. 4.2)에서 $w$ = detector parameter, $\theta$ = Concrete DropBlock parameter로 확인했어.

> 보충: 그림의 Student에 "regress." head가 있어. p.35의 WSDDN 한계 "does not learn bounding box regression"과 대비되지만, 슬라이드 bullet엔 언급이 없어. 그림에서만 보이는 내용이야. MIST의 뜻도 슬라이드에 없어.

> 보충 (RL 비유, 슬라이드 밖 = 시험 범위 아님): adversary가 일부러 어렵게 만들고 policy가 버티도록 학습하는 robust RL과 동기가 닮았어.

**시험 관점**
- **차이**: OICR·PCL = pseudo-label quality, 이 방법 = **discriminative-part problem**.
- ConcreteDropBlock은 **most relevant** region을 꺼. "random"이나 "least relevant"로 바꾼 지문을 조심해.
- 식의 방향: $\min$은 $w$, $\max$는 $\theta$야.

---

## Regularizing Discriminative Parts (p.39)

슬라이드 텍스트:
- Successful cases / Failure cases
- 그림: 결과 사진 여러 장. box마다 class와 숫자(예: sheep-1.00, bus-0.99)가 붙어 있어. Successful 쪽은 양·버스·소·사람·개·사과 등이 물체별로 따로 잡혀. Failure 쪽은 사람의 몸 일부에 box가 따로 잡히거나, 단체 사진에 box가 많이 몰리거나, 넓은 영역을 한 box로 잡은 예가 있어.

**설명**
- 결과 예시 슬라이드야. 물체가 여럿이어도 따로 잡는 사례가 Successful에 많아.
- Failure를 보면 사람처럼 **부위가 뚜렷한** 물체에서 part 문제가 남아 있어. 그림에서 읽은 해석이고, 슬라이드엔 설명이 없어.
- 숫자는 표에 나온 실험 수치가 아니라 box마다 붙은 점수야. 외울 필요 없어.

---

## Summary on Weakly-Supervised Object Detection (p.40)

슬라이드 텍스트:
- WSOD is an ill-posed problem and ambiguous due to part/whole and co-occurence issues. ← "co-occurence"는 오타 (co-occurrence). 원문 유지
- WSOD training is sensitive to initialization and prone to overfitting.
- Remarkable progress due to
  - objectness and region proposals,
  - incorporating prior knowledge and additional cues,
  - end-to-end learning.
- Addressing the inherent ambiguity requires better objectness measures and stronger additional cues.

**설명**
- **ill-posed** = 주어진 정보만으론 답이 하나로 정해지지 않는 문제야. image label만으로는 얼굴과 사람 전체를 구분할 근거가 없으니까.
- 네 문장을 강의 구조에 맞추면 이래.
  1. 왜 어렵나: part/whole, co-occurrence (p.11–13)
  2. 학습이 왜 불안한가: initialization, overfitting (p.16)
  3. 뭐가 도왔나: objectness·proposals (p.17, p.28), priors·cues (p.27–31), end-to-end (p.34–38)
  4. 앞으로: 더 좋은 objectness와 더 강한 cue

**시험 관점**
- 이 슬라이드 문장은 T/F로 그대로 쓰기 좋아.
  - "WSOD training is **robust** to initialization"은 틀렸어.
  - progress 이유 세 개 중 하나를 다른 걸로 바꾼 지문을 조심해.

---

## Next Week.. (p.41)

슬라이드 텍스트:
- Weakly supervised ..

**설명**
- 다음 주도 weakly supervised가 이어져. `exam/PLAYBOOK.md` 0.2 일정엔 W11 11/12 Weakly-Supervised II로 적혀 있어.

---

## 이 강의 스캔 메모

- p.1–41 안에 Algorithm 박스나 코드 박스는 없어. 수식은 p.20, p.21, p.22, p.28, p.38에 있어.
