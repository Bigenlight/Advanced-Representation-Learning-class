<!-- DRAFT | lecture: 09_Weakly_Supervised_Learning_II | chunk: 02 | pages 18–30 -->
<!-- VERIFIED | fixes: 6 | verifier: sonnet -->

## Section: From Image-Level Labels to Pixel-Level Labels (p.18)

슬라이드 텍스트:
- From Image-Level Labels to Pixel-Level Labels

**설명**
- 섹션 구분 슬라이드야. 목차(p.2)의 2번 덩어리고, p.18–42가 이 섹션이야.
- 질문은 하나야: **"이 사진에 고양이가 있다"는 image-level label만 가지고, 어느 pixel이 고양이인지(pixel-level label)를 어떻게 얻나?**
- 이 섹션의 흐름: classifier 안을 들여다보는 두 방법(gray box / white box) → white box의 대표 **CAM** → CAM을 threshold해서 **seed** 만들기 → CAM의 두 약점 → 약점별 처방 (p.29–41).

---

## From Image-Level Labels to Pixel-Level Labels (p.19)

슬라이드 텍스트:
- 그림 왼쪽: 자전거 옆에 사람 두 명이 있는 사진, 왼쪽 아래에 글자 "person" (초록), "bicycle" (보라). 캡션 "Image-level labels at training time"
- 가운데: 파란 화살표
- 그림 오른쪽: 자전거 타는 사람들 사진 위에 사람 pixel은 초록, 자전거 pixel은 보라로 칠한 segmentation. 캡션 "Pixel-level labels at test time"

**설명**
- **image-level label** = 사진 한 장에 붙은 class 이름 목록이야. 위치 정보가 없어. 여기선 "person, bicycle".
- **pixel-level label** = 사진의 pixel 하나하나에 class를 매긴 것이야. 이게 semantic segmentation의 정답 형식이야.
- **semantic segmentation** = 모든 pixel을 class로 분류하는 task야. 같은 class의 물체 두 개(사람 두 명)는 구분하지 않고 둘 다 "person"으로 칠해.
- 핵심: **train 땐 image-level label만 주고, test 땐 pixel-level label을 내라**는 거야. L08의 weak supervision 정의 "Lower degree (or cheaper) annotation at train time than the required output at test time" (L08 p.4)를 segmentation에 적용한 거야.
- 색 대응: 왼쪽 글자색(person = 초록, bicycle = 보라)이 오른쪽 칠한 색과 같아. "글자로만 준 class가 test 땐 pixel 영역으로 나와야 한다"를 보여 주는 그림이야.

> 보충: 왼쪽과 오른쪽은 같은 사진이 아니야. train 사진과 test 사진을 하나씩 보여 준 거라서, 오른쪽 mask가 왼쪽 사진의 정답이라는 뜻은 아니야.

---

## Two Main Approaches (p.20)

슬라이드 텍스트:
- Gray box
  - Utilize the internal information in the network.
  - In other words, it identifies which areas the model focused on for classification by leveraging the activations or gradients within the CNN.
- White box

**설명**
- image label로 학습한 **classifier**에서 위치 정보를 꺼내는 두 갈래야.
- **Gray box** = 이미 학습한 classifier의 **구조는 그대로 두고**, 안쪽 정보(activation이나 gradient)를 들여다봐서 "분류할 때 어디를 봤나"를 알아내.
  - 슬라이드 정의에 "activations **or** gradients"라고 둘 다 적혀 있어. gray box = gradient만 쓴다고 외우면 안 돼.
  - 이 강의에서 보여 주는 gray box 예시(p.21–22)는 둘 다 **gradient** 기반이야.
- **White box** = p.20엔 이름만 있고 설명은 비어 있어. 설명은 p.24–25에 나와: network 구조를 **처음부터 위치 정보가 남게** 바꿔서 class map을 직접 읽어.
- 이름 직관 (설명용): 완전히 안 보이는 black box가 아니라 안을 조금 엿보는 게 gray box, 구조 자체를 투명하게 설계하는 게 white box야.

> 보충: "gray box / white box"는 이 슬라이드의 분류 이름이야. 논문마다 쓰는 표준 용어는 아니니까 슬라이드 뜻대로 외워.

---

## Gray Box Approach: gradient saliency (p.21) — Simonyan et al. 2014

슬라이드 텍스트:
- Step 1) Train an ImageNet classifier.
- Step 2) Calculate how much the score changes when the input is slightly modified.
- 그림 안 글: "Step 1: Train an image classifier / Step 2: given an image with known label, compute $\left.\frac{\partial S_c}{\partial I}\right|_{I_0}$ ("how much does changes in the input image affect the score?")" ← 원문 문법 그대로 ("does changes")
- 그림: 두 줄(독수리, 강아지). 각 줄이 원본 사진 → 검은 바탕에 흰 점이 흩뿌려진 saliency map → 빨강·하늘색·파랑 지도 → 물체만 남기고 배경을 검게 지운 결과
- ● The problem is that it is extremely noisy.
- ● This is because the gradient measures sensitivity at the level of individual pixels, resulting in a "salt-and-pepper" appearance.
- [Simonyan et. al 2014]

**설명**
- **어떻게**
  1. image label로 보통 classifier를 학습해.
  2. 사진 $I_0$와 그 label $c$가 주어지면, class 점수 $S_c$를 **입력 사진 $I$로 미분**해: $\left.\frac{\partial S_c}{\partial I}\right|_{I_0}$.
- 기호 풀이
  - $S_c$ = class $c$의 점수야 (softmax 전 값).
  - $I$ = 입력 사진, $I_0$ = 지금 보고 있는 그 사진이야. $|_{I_0}$는 "그 사진에서 계산한 값"이라는 뜻이야.
  - $\partial S_c / \partial I$ = 입력과 같은 크기의 지도야. pixel 하나를 살짝 바꾸면 점수가 얼마나 바뀌나(**sensitivity**)를 pixel마다 적은 거야.
- 이 지도가 **saliency map**(눈에 띄는 곳 지도)이야. 값이 큰 pixel = 분류에 영향이 큰 pixel이라서 물체 쪽일 거라고 봐.
- 문제: **"extremely noisy"**. gradient는 pixel 하나 단위의 민감도라서 이웃 pixel끼리 값이 들쭉날쭉해. 그래서 소금·후추를 뿌린 것 같은 **"salt-and-pepper" appearance**가 나와.
- 그림 읽는 법: 두 번째 칸이 saliency map이야. 아주 어두운 바탕에 희미한 점이 사진 전체에 흩어져 있어 (물체 윤곽이 또렷하지 않음). 세 번째·네 번째 칸은 이 지도로 segmentation까지 한 결과야.

> 보충 (논문): 원 논문은 Simonyan, Vedaldi, Zisserman, "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps", ICLR 2014 (workshop)야. 논문에선 color channel 3개 중 gradient 절댓값이 가장 큰 값을 그 pixel의 saliency로 써. 세 번째 칸은 saliency가 높은 곳(파랑)으로 foreground 색 모델을, 낮은 곳(하늘색)으로 background 색 모델을 만들고, 빨강은 안 쓰는 영역이야. 그 색 모델로 GraphCut segmentation을 돌려서 네 번째 칸을 얻어. 슬라이드엔 그림만 있고 이 설명은 없어. "확실히 높은 곳 = 물체, 확실히 낮은 곳 = 배경, 중간은 안 씀"이라는 점에서 p.27의 seed와 같은 생각이야.

> 보충 (RL 비유, 시험 범위 아님): $\partial S_c/\partial I$는 critic의 $\partial Q(s,a)/\partial a$나 $\partial Q/\partial s$처럼 "입력을 살짝 바꾸면 출력이 얼마나 바뀌나"를 보는 거야. DDPG가 $\nabla_a Q$로 action을 미는 것과 같은 종류의 미분이야. 한 점에서 잰 local 미분이라 noisy한 것도 같아.

---

## Gray Box Approach: SmoothGrad (p.22) — Smilkov et al. 2017

슬라이드 텍스트:
- Step 1) Training an Image Classifier
  - Train a standard CNN classifier.
- Step 2) Add small random noise to the images
- Step 3) Calculate the gradient for each
- Step 4) Average
- This slide explains SmoothGrad. The key point is that calculating the gradient multiple times and averaging it removes noise and yields a more meaningful saliency map.
- 그림 안 글: "Step 1: Train an image classifier / Step 2: **Add small random noise** to image / Step 3: given known label, compute $\left.\frac{\partial S_c}{\partial I}\right|_{I_0}$ ("how much does changes in the input image affect the score?") / Step 4: **Repeat 50 times** step 2&3, **average result**"
- 그림: 세 줄(drilling platform, great white shark, hognose snake) × 세 칸(원본 / Vanilla / SmoothGrad, SmoothGrad 칸은 빨간 테두리)
- ( no quantitative experiments either )
- [Smilkov et al. 2017]

**설명**
- p.21의 noisy한 gradient를 **여러 번 재서 평균**하는 방법이야.
- **어떻게**
  1. 보통 CNN classifier를 학습해.
  2. 사진에 작은 random noise를 더해.
  3. noise 섞인 사진마다 gradient $\partial S_c/\partial I$를 구해.
  4. 2–3을 **50번** 반복하고 평균해.
- 식으로 쓰면 (notes 보충, 슬라이드엔 단계만):
  $$\hat M_c(I) = \frac{1}{n}\sum_{i=1}^{n} M_c\big(I + \epsilon_i\big),\quad \epsilon_i \sim \mathcal N(0, \sigma^2)$$
  - $M_c(\cdot)$ = p.21의 gradient saliency map이야.
  - $n$ = 반복 횟수, 슬라이드 그림에선 50이야.
  - $\epsilon_i$ = 매번 새로 뽑는 작은 noise야.
- **왜 돼?** pixel 단위의 들쭉날쭉한 값은 noise마다 방향이 달라서 평균하면 서로 상쇄돼. 물체 쪽에서 일관되게 큰 값만 남아. 그래서 "removing noise by adding noise"(논문 제목)야.
- 그림 읽는 법: 가운데 Vanilla(= p.21 방식)는 사진 전체에 점이 흩어져 있고, 오른쪽 SmoothGrad는 시추선·상어·뱀의 윤곽이 보여.
- "( no quantitative experiments either )": 결과가 **눈으로 본 그림뿐이고 수치 비교 실험은 없다**는 꼬집음이야. 더 보기 좋은 map이 실제 segmentation에 얼마나 도움이 되는지는 안 쟀다는 뜻이야.
- **차이**: Simonyan은 gradient를 **한 번** 재고, SmoothGrad는 noise를 섞어 **여러 번** 재서 평균해. 둘 다 classifier 구조는 안 바꾸는 gray box야.

> 보충 (논문): 원 논문은 Smilkov, Thorat, Kim, Viégas, Wattenberg, "SmoothGrad: removing noise by adding noise", arXiv 2017이야.

> 보충 (RL 비유, 시험 범위 아님): 한 번 잰 gradient 대신 여러 sample의 gradient를 평균해서 분산을 줄이는 건 REINFORCE에서 sample을 많이 모아 평균할수록 gradient 추정이 덜 흔들리는 것과 같은 동기야. 단 SmoothGrad의 noise는 입력에 일부러 넣는 거고, REINFORCE의 noise는 policy sampling에서 와.

---

## Gray Box Approach: key issue (p.23)

슬라이드 텍스트:
- Key issue: Asking apples from the oranges tree **Classifier was not built nor trained to segment objects.**
- Gradient-based saliency involves forcing localization using a classifier. Since the classifier was not originally trained for localization, it suffers from a **fundamental limitation**: it highlights only the specific regions necessary for classification rather than the entire object.

**설명**
- **"Asking apples from the oranges tree"** = 오렌지 나무한테 사과를 달라는 격이야. classifier는 분류하라고 만든 거지 segmentation하라고 만든 게 아니야.
- 그래서 gray box에는 **근본 한계(fundamental limitation)**가 있어: map이 **분류에 필요한 부분만** 밝히고 **물체 전체**를 밝히진 않아.
  - 예: 새를 분류하는 데는 부리·머리만 봐도 충분하니까, 몸통은 map에서 빠져.
- 이건 noise 문제와 다른 문제야. SmoothGrad가 noise를 줄여도 이 한계는 그대로야.
- 이 "분류에 필요한 부분만" 문제는 white box(CAM)에도 그대로 남아. p.28의 "Focused on discriminative area only"가 같은 얘기야. L08의 discriminative part 문제(L08 p.35, 38)와도 같은 뿌리야.

---

## White Box Approach: architecture (p.24) — Zhou et al. 2016

슬라이드 텍스트:
- Step 1) eliminate the FC layer
- Step 2) add GAP (Global Average Pooling)
- Step 3) add a 1×1 convolution that performs a linear combination of the features
- 그림 왼쪽: Input image → Conv layer 1 → Conv layer …. → Conv layer N → FC (N+1, 노랑·보라 출력 두 개에 모든 연결)
- 가운데: 파란 화살표
- 그림 오른쪽: Input image → Conv layer 1 → Conv layer …. → Conv layer N → (초록·파랑·보라·주황 띠 네 개) "Spatial average pooling" → "1x1 Conv layer N+1" → 노랑·보라 출력 두 개
- [Zhou et al. 2016]

**설명**
- white box는 classifier의 **구조를 바꿔서** 처음부터 위치 정보가 남게 만들어.
- **어떻게**
  1. 마지막 conv 뒤의 **FC layer를 없애.** FC는 feature map을 한 줄로 펴서 섞기 때문에, 어느 위치에서 온 값인지가 사라져.
  2. **GAP(Global Average Pooling)**를 넣어. channel마다 feature map 전체를 평균해서 숫자 하나로 만들어.
  3. **1×1 convolution**을 넣어서 feature들을 **선형 결합(linear combination)**해. class마다 weight 한 벌이 있어.
- **GAP** = feature map 한 장($H\times W$)의 평균값 하나를 내는 연산이야. channel이 $K$개면 길이 $K$짜리 벡터가 나와.
- **1×1 convolution** = 위치마다 같은 weight로 channel끼리만 섞는 conv야. 위치는 안 섞어. 그래서 GAP로 줄인 벡터에 걸면 FC layer 하나와 똑같아.
- 그림의 "N+1": 왼쪽에선 FC가 N+1번째 layer고, 오른쪽에선 1×1 conv가 N+1번째 layer야. 출력 원 두 개(노랑, 보라)는 class 두 개의 점수야.
- **차이 (p.21–22와)**: gray box는 학습된 classifier를 그대로 두고 **gradient**를 계산해. white box는 **구조를 바꾸고** forward 계산만으로 map을 얻어.

> 보충: 슬라이드 Step 3은 "1×1 convolution", 원 논문(Zhou et al., "Learning Deep Features for Discriminative Localization", CVPR 2016)은 GAP 뒤에 "fully-connected layer"(softmax 출력)라고 적어. GAP로 공간이 1×1이 된 뒤엔 둘이 같은 연산이야. 오른쪽 그림이 pooling 위에 1×1 conv를 그린 이유야.

---

## White Box Approach: CAM, Class Activation Mapping (p.25) — Zhou et al. 2016

슬라이드 텍스트:
- CAM (Class Activation Mapping)
- 그림 위: 강아지를 안은 사람 사진 → CONV 여러 개 → 마지막 conv의 feature map 여러 장(파랑·빨강·초록 테두리) → GAP → 원 n개(파랑·빨강·…·초록) → weight $w_1, w_2, \dots, w_n$ → "Australian terrier" 출력
- 그림 아래 "Class Activation Mapping": $w_1 *$ (map 1) $+\; w_2 *$ (map 2) $+ \dots +\; w_n *$ (map n) $=$ Class Activation Map (Australian terrier)
- since the network itself is designed to preserve spatial information, activation maps can be obtained directly without the need for separate gradient calculations.
- [Zhou et al. 2016]

**설명**
- **CAM** = 마지막 conv의 feature map들을, 그 class의 weight로 **가중합**한 지도야. 값이 큰 곳 = 그 class 점수에 많이 기여한 곳이야.
- 그림 위 줄은 **분류**고, 아래 줄은 같은 weight를 **feature map에 바로** 곱해서 지도를 만드는 거야. 보라색 곡선 화살표는 마지막 conv feature map과 weight를 같이 아래 줄(Class Activation Mapping)로 가져가는 걸 뜻해. weight $w_1, \dots, w_n$을 feature map에 곱해서 합쳐.
- 식 (notes 보충, 슬라이드엔 그림 속 $w_1 * \cdot + w_2 * \cdot + \dots = $ CAM만):
  $$M_c(x,y) = \sum_{k} w_k^c\, f_k(x,y)$$
  - $f_k(x,y)$ = 마지막 conv의 $k$번째 feature map에서 위치 $(x,y)$의 값이야.
  - $w_k^c$ = GAP 뒤 layer에서 "$k$번째 feature가 class $c$에 얼마나 중요하냐"를 나타내는 weight야. 그림의 $w_1,\dots,w_n$이고, 여기선 $c$ = Australian terrier야.
  - $M_c(x,y)$ = 위치 $(x,y)$가 class $c$ 점수에 기여한 양이야.
- **왜 이게 맞는 지도야?** (notes 보충, 유도)
  - GAP: $F_k = \frac{1}{Z}\sum_{x,y} f_k(x,y)$ ($Z$ = 위치 개수)
  - class 점수: $S_c = \sum_k w_k^c F_k$
  - 대입하고 합의 순서를 바꾸면: $S_c = \frac{1}{Z}\sum_{x,y}\sum_k w_k^c f_k(x,y) = \frac{1}{Z}\sum_{x,y} M_c(x,y)$
  - 즉 **class 점수 = CAM의 평균**이야. 점수가 위치별 기여로 **정확히** 쪼개져. 평균과 선형 결합이 둘 다 선형이라 순서를 바꿔도 되기 때문이야.
  > 보충: 원 논문은 softmax의 bias를 무시해(0으로 둬). 슬라이드엔 없는 내용이야. bias $b_c$가 있으면 $S_c$ = CAM 평균 + $b_c$라서, "점수 = CAM 평균"은 bias 없는 head에서만 정확해.
- **왜 gradient가 필요 없어?** "since the network itself is designed to preserve spatial information, activation maps can be obtained directly without the need for separate gradient calculations." (p.25)
  - forward 때 이미 계산한 feature map $f_k$와, 학습된 weight $w_k^c$만 있으면 돼. backward를 따로 안 돌려.
  - gray box(p.21)는 $\partial S_c/\partial I$를 구하려고 backward를 돌려야 했어.
- 마지막 conv의 map은 입력보다 작아서(예: 14×14, p.29), 보통 입력 크기로 **upsample**해서 사진 위에 겹쳐 봐.

**예시로 보면** (설명용 숫자, 슬라이드 수치 아님)
- feature map 2장, 크기 2×2. $f_1 = \begin{bmatrix}4&0\\0&0\end{bmatrix}$ (머리에 반응), $f_2 = \begin{bmatrix}0&0\\2&2\end{bmatrix}$ (털·다리에 반응).
- dog weight $w^{dog} = (1.0,\ 0.5)$.
- $M_{dog} = 1.0\cdot f_1 + 0.5\cdot f_2 = \begin{bmatrix}4&0\\1&1\end{bmatrix}$
- 확인: GAP $F = (1, 1)$ → $S_{dog} = 1.0 + 0.5 = 1.5$. $M_{dog}$의 평균 $= (4+0+1+1)/4 = 1.5$. 같아.
- 왼쪽 위(머리)가 4로 압도적이야. threshold하면 머리만 남아. 이게 p.28 "discriminative area only"의 미니 버전이야.

> 보충 (RL 비유, 시험 범위 아님): CAM은 class 점수를 위치별 기여의 합으로 쪼개. multi-agent RL의 VDN이 $Q_{tot} = \sum_i Q_i$로 팀 value를 agent별 value의 합으로 쪼개는 것과 같은 구조야. head가 선형이라 쪼개기가 정확한 것도 같아. "어느 위치가 점수에 기여했나"는 credit assignment 질문이야.

---

## Designed to Generate Per-Class Maps (p.26) — Zhou et al. 2016

슬라이드 텍스트:
- 그림: 왼쪽 위 원본 사진 "dome"(밤의 돔 건물), 나머지 5칸은 class별 CAM: palace 0.459, dome 0.195, church 0.146, altar 0.091, monastery 0.051
- Figure 4. Examples of the CAMs generated from the top 5 predicted categories for the given image with ground-truth as dome.
- [Zhou et al. 2016]

**설명**
- CAM은 **class마다 하나씩** 나와. 같은 사진, 같은 feature map인데 weight $w^c$만 바꾸면 다른 지도가 돼.
- 그림 읽는 법: 칸 위 숫자는 그 class의 예측 확률이야. 정답은 dome인데 1등은 palace(0.459)야.
  - palace CAM은 **건물 아래 가로로 긴 부분**에 빨개.
  - dome CAM은 **위쪽 둥근 지붕**에 빨개.
  - 즉 class가 바뀌면 보는 곳도 바뀌어. 이게 "per-class map"의 뜻이야.
- 그래서 WSSS에 쓸 수 있어: image label에 있는 class마다 CAM을 하나씩 뽑으면 class별 위치 후보가 생겨.

---

## The Class Maps are Thresholded, to Generate Training Data (p.27) — Zhou et al. 2016

슬라이드 텍스트:
- 그림 M1: Global Average Pooling — 원숭이 사진 → CNN → feature map $\mathbf F$ (channel 1 2 3 … C) → GAP → $\mathbf p^{gap}$ → weight $w_{1,k}, w_{2,k}, \dots, w_{C,k}$ → "k: monkey"
- 그림 M2: Weighted Averaging — $w_{1,k} * \mathbf F_1 + w_{2,k} * \mathbf F_2 + w_{3,k} * \mathbf F_3 + \dots + w_{C,k} * \mathbf F_C = \mathbf M_k$
- 그림 M3: Thresholding — $\mathbf M_k$ → resize → $\mathbf M'_k$ → $> \tau_{loc}$ → $\mathbf B_k$ (검은 바탕에 흰 원숭이 모양 mask)
- High value scores → Foreground areas
- Low value scores → Background areas
- (This confident regions are commonly named "seeds") ← 원문 문법 그대로 ("This … regions")
- [Zhou et al. 2016]

**설명**
- CAM을 **threshold**해서 학습용 pixel label을 만들어. 이게 image label → pixel label의 첫 다리야.
- **어떻게** (그림 M1 → M2 → M3)
  1. M1: CNN + GAP + weight로 분류해 (p.24–25 구조).
  2. M2: class $k$의 weight로 feature map을 가중합해서 CAM $\mathbf M_k$를 만들어.
  3. M3: 입력 크기로 resize한 $\mathbf M'_k$에서 $\tau_{loc}$보다 큰 곳을 1로 둔 binary mask $\mathbf B_k$를 만들어.
- 기호 주의 (함정 후보):
  - 이 그림에선 **$C$가 feature channel 수**고 **$k$가 class**야 (k: monkey). p.25 식의 $k$(channel), $c$(class)와 글자가 뒤바뀌어 있어.
  - $\tau_{loc}$은 **threshold**야. temperature나 EMA 계수가 아니야.
- 슬라이드 문장: **High value scores → Foreground**, **Low value scores → Background**. 이렇게 **확신할 수 있는 영역**을 **"seeds"**라고 불러.
- **seed** = 씨앗이야. 물체 전체가 아니라 "여기는 확실히 물체 / 확실히 배경"인 작은 영역이고, 뒤 방법들(SEC 등)이 여기서 영역을 키워.

> 보충: 그림 M3은 threshold 하나로 binary mask를 만드는 localization 버전이야. WSSS에서 seed를 만들 땐 보통 높은 곳만 foreground, 낮은 곳만 background로 두고 **애매한 중간은 label을 비워 둬(ignore)**. 그래야 확신하는 pixel만 학습 target이 돼. SEC의 seeding loss가 seed pixel에만 걸리는 이유야 (p.31).

> 보충: 그림의 M1/M2/M3(Global Average Pooling · Weighted Averaging · Thresholding) 구성은 Bae et al., "Rethinking Class Activation Mapping for Weakly Supervised Object Localization" (ECCV 2020)이 CAM을 세 module로 나눈 방식과 같아. 다만 그림 출처는 확인 못 했어. 슬라이드 cite는 Zhou et al. 2016뿐이니까 시험엔 그걸 따라.

---

## Good, but not good enough (p.28)

슬라이드 텍스트:
- 1. Resulting masks are not sharp
- 2. Focused on discriminative area only
- 그림: 두 줄. 위 줄 ostrich 사진 12장과 그 CAM (점수 1.000, 0.973, 0.751 …), 아래 줄 spotted salamander 사진 12장과 그 CAM (0.938, 0.821 …)

**설명**
- CAM으로 seed까지 만들 수 있지만 두 가지가 모자라.
  1. **Resulting masks are not sharp**: CAM은 뭉개진 blob 모양이라 물체 경계를 못 따라가.
  2. **Focused on discriminative area only**: 분류에 결정적인 부분만 빨개.
- 그림 읽는 법: 타조 CAM은 사진마다 **한 덩어리 blob**만 빨개. 머리·목에 뜨는 경우가 많고, 몸통 한 곳에 뜨는 것도 있어. 어느 쪽이든 새 전체 윤곽을 따라가지 않아. 도롱뇽도 몸의 한 덩어리만 빨개. 즉 not sharp(뭉개진 blob) + discriminative part만(물체 전체가 아님)이야.
- **discriminative area** = 그 class를 다른 class와 구별하는 데 결정적인 부분이야. 예를 들면 타조의 긴 목과 머리 같은 곳이야 (설명용 예시). classifier는 거기만 봐도 맞히니까 나머지에 반응할 이유가 없어 (p.23과 같은 뿌리).
- 이 두 문제가 p.29–41의 목차야. 이후 슬라이드는 이 두 줄을 반복하면서 "⇢ 처방"을 하나씩 붙여.

---

## Good, but not good enough: increase resolution (p.29)

슬라이드 텍스트:
- 1. Resulting masks are not sharp
  - ⇢ Increase top layers resolution
- 2. Focused on discriminative area only
- 그림: p.25의 CAM network 그림에서 마지막 쪽 작은 CONV를 파란 동그라미로 표시하고 화살표로 "14x14 pixels"

**설명**
- 처방 1-a: **top layers의 해상도를 키워.**
- 왜 not sharp? CAM은 마지막 conv의 feature map에서 나와. 그림대로라면 그 크기가 **14×14**밖에 안 돼. ← 그림에서 파란 원은 색칠된 마지막 feature map 바로 앞의 작은 CONV block을 가리켜. 의도는 "top layers의 map이 14×14로 거칠다"야. 입력 사진(예: 224×224)보다 훨씬 거칠어서, upsample해도 경계가 뭉개져.
- 그래서 마지막 layer들의 해상도를 높이면(feature map을 덜 줄이면) CAM이 더 촘촘해져.
- 슬라이드엔 cite가 없고, 구체적으로 어떻게 해상도를 키우는지도 안 적혀 있어.

> 보충 (시험 범위 아님): 해상도를 키우는 흔한 방법은 마지막 몇 layer의 stride를 줄이거나 dilated(atrous) convolution을 쓰는 거야. 슬라이드엔 없는 내용이야.

---

## Good, but not good enough: boundary-aware propagation (p.30) — Kolesnikov & Lampert ECCV 2016

슬라이드 텍스트:
- 1. Resulting masks are not sharp
  - ⇢ Use boundary-aware propagation
- 2. Focused on discriminative area only
- 그림: 사진(소와 사람, image label "Person" "Cow")에서 세 갈래
  - 위: Weak Localization → seed 지도 (Person = 분홍, Cow = 초록, Backgr. = 남색, 나머지는 체크무늬) → **Seeding Loss**
  - 가운데: Segmentation CNN → 예측 mask (소 초록, 사람 분홍) → **Global Weighted Rank-Pooling** (label "Person" "Cow") → **Expansion Loss**
  - 아래: 사진을 Downscale → **CRF** (예측 mask와 함께) → 경계에 맞춘 mask → **Constrain-to-boundary Loss**
  - 세 loss가 ⊕로 합쳐지고, 회색 화살표는 gradient가 거꾸로 흐르는 방향
- [Kolesnikov & Lampert ECCV 2016]

**설명**
- 처방 1-b: **boundary-aware propagation**. seed를 경계(boundary)를 넘지 않게 퍼뜨려.
- 이 그림이 **SEC**(Seed, Expand, Constrain)야. 자세한 건 p.31–32에서 다뤄 (chunk 03).
- 그림 읽는 법 (세 loss = 세 원칙):
  - **Seeding Loss**: weak localization(CAM 같은 것)이 준 seed와 예측이 맞게 해. 체크무늬는 label이 없는 pixel이야.
  - **Expansion Loss**: image label을 써서 예측이 seed보다 넓게 퍼지게 해. Global Weighted Rank-Pooling으로 pixel 점수를 image 점수로 모아.
  - **Constrain-to-boundary Loss**: CRF로 다듬은 mask와 예측이 같게 해서 경계에 맞춰.
- 회색 화살표는 gradient가 거꾸로 흐르는 방향(backward)이야. 세 loss 모두에서 Segmentation CNN 쪽으로 돌아와. CRF 아래쪽 갈래에서도 회색 화살표가 CRF를 지나 예측 mask로 올라가.
- p.30은 "not sharp"의 처방으로 소개되지만, Expansion Loss는 seed를 키우니까 2번 문제(discriminative area only)에도 손을 대 (p.31 "Focus only on discriminative regions"를 CAM limitations에 같이 적어 둠).

> 보충: 원 논문은 Kolesnikov & Lampert, "Seed, Expand and Constrain: Three Principles for Weakly-Supervised Image Segmentation", ECCV 2016이야.

> 보충 (RL 비유, 시험 범위 아님): 세 loss를 더해서 한 network를 학습하는 건 main loss에 auxiliary loss 여러 개를 붙이는 것과 같아. 각 loss가 다른 원칙(seed 지키기, 넓히기, 경계 맞추기)을 맡아.
