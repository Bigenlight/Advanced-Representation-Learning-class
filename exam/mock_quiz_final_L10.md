# Mock Quiz — Final, L10 (Active Learning I)

> 단원 미니 모의고사. 10문제 = **T/F 5 + MCQ 3 + code 2** (중간 실제 비율 10 : 5 : 5의 절반).
> **답은 영어로 직접 써.** T/F는 고르고 이유까지, False면 고친 문장까지. MCQ도 왜 그 답인지 한 문장. code는 빈칸 하나 + 과정 설명 2–3문장 (입력 → target은 어디서 오나 → loss → 누가 update되나).
> 헷갈리면 `False?`처럼 `?`를 붙여. 맞아도 `?` 붙은 문제는 다시 봐. 답을 다 적기 전엔 맨 아래를 열지 마.
> 출처: `exam/_parts/10_Active_Learning_I.md` C 섹션, `exam/pseudocode_final.md`. 모두 작성자와 다른 Sonnet이 정답을 가리고 먼저 푼 뒤 대조함 (code card는 독립 테스트) (2026-11-19). 슬라이드 페이지 번호는 `lectures/10_Active_Learning_I.pdf` 기준.

### 1. [T/F] `[L10]`
In the active learning workflow, the model labels the selected samples with its own predictions and is then retrained on these pseudo labels.

> T/F:
>
> Why:
>
> Corrected (False일 때):

### 2. [T/F] `[L10]`
In margin sampling, $U(x)=P(\hat y_1|x)-P(\hat y_2|x)$, and samples with a large margin are selected first because they are the most uncertain.

> T/F:
>
> Why:
>
> Corrected (False일 때):

### 3. [T/F] `[L10]`
In query-by-committee, voter entropy averages the committee members' full probability distributions, while consensus entropy only counts the predicted label of each member.

> T/F:
>
> Why:
>
> Corrected (False일 때):

### 4. [T/F] `[L10]`
Aleatoric uncertainty is caused by limited model knowledge and can be reduced by collecting more labeled data.

> T/F:
>
> Why:
>
> Corrected (False일 때):

### 5. [T/F] `[L10]`
BALD equals the entropy of the mean prediction minus the mean entropy of the individual predictions.

> T/F:
>
> Why:
>
> Corrected (False일 때):

### 6. [MCQ] `[L10]`
A 3-class model predicts $P(y|x_a)=(0.50, 0.49, 0.01)$ and $P(y|x_b)=(0.50, 0.25, 0.25)$. Which statement is correct?  
(A) Least confidence prefers $x_a$, margin prefers $x_b$, and entropy prefers $x_b$.  
(B) Least confidence gives both the same score, margin prefers $x_a$, and entropy prefers $x_b$.  
(C) All three measures prefer $x_b$.  
(D) All three measures prefer $x_a$.

> Answer:
>
> Why:

### 7. [MCQ] `[L10]`
What is MC Dropout for in deep active learning?  
(A) To prevent overfitting during training only; it plays no role when scoring unlabeled samples.  
(B) To approximate an ensemble with a single network: multiple stochastic forward passes with different dropout masks give a spread of predictions that estimates epistemic uncertainty.  
(C) To make inference faster by dropping units at test time.  
(D) To remove the aleatoric noise from unlabeled images before they are queried.

> Answer:
>
> Why:

### 8. [MCQ] `[L10]`
Which pairing of query strategy and selection criterion is correct?  
(A) Diversity sampling — samples expected to change the model the most after labeling  
(B) Expected model change — representative samples that cover different regions of the feature space  
(C) Hybrid strategy — combine multiple criteria, e.g. samples that are uncertain and representative  
(D) Uncertainty sampling — avoid selecting redundant samples

> Answer:
>
> Why:

### 9. [Code] `[L10·PC]`
Fill in the blank, and explain what this code does in 2–3 sentences.

```python
p = model(x_u).softmax(dim=1)                          # P(y|x): [N, K] for unlabeled samples (p.10)
top2 = p.topk(2, dim=1).values                         # P(ŷ1|x), P(ŷ2|x)
U_lc = 1 - top2[:, 0]                                  # least confidence: 1 - P(ŷ|x) (p.11)
U_margin = top2[:, 0] - top2[:, 1]                     # margin: P(ŷ1|x) - P(ŷ2|x) (p.12)
U_ent = -(p * p.log()).sum(dim=1)                      # entropy: -Σ_y P(y|x) log P(y|x) (p.13)
pick_lc = U_lc.topk(b).indices                         # lower confidence -> higher priority (p.11)
pick_margin = U_margin.topk(b, largest=______).indices # small margin -> high uncertainty (p.12)
pick_ent = U_ent.topk(b).indices                       # higher entropy -> higher uncertainty (p.13)
```

> Blank:
>
> Process:

### 10. [Code] `[L10·PC]`
Fill in the blank, and explain what this code does in 2–3 sentences.

```python
model.______()                                           # dropout stays ON at inference (p.27)
with torch.no_grad():
    P = torch.stack([model(x_u).softmax(dim=1) for _ in range(T)])  # T stochastic passes, different masks: [T, N, K] (p.28)
p_bar = P.mean(dim=0)                                    # mean prediction p̄(y|x) (p.29)
bald = H(p_bar) - H(P).mean(dim=0)                       # entropy of mean - mean entropy (p.29)
pick = bald.topk(b).indices                              # highest epistemic uncertainty -> query labels (p.28)
```

> Blank:
>
> Process:

---

<details>
<summary><b>정답 · 모범 답안</b> (다 쓴 뒤에 열어)</summary>

**1.** `[L10]` **Answer**: False
- **Why (EN)**: The selected samples are labeled by a human annotator. Active learning saves annotation by choosing which samples to label, not by replacing human labels with the model's predictions.
- **Corrected (EN)**: In the active learning workflow, the selected informative samples are labeled by human annotation, the model is retrained, and the process repeats.
- **해설**: Workflow "Select informative samples / Human annotation / Retrain the model / Repeat" (p.5). 그림도 "Let's label!" → "Valuable labeled dataset" → "Better model"이야. 모델이 자기 예측으로 target을 만드는 건 L07 pseudo label(self-training) 쪽이야.

← pool: `10_Active_Learning_I` Q2

**2.** `[L10]` **Answer**: False
- **Why (EN)**: A large gap between the top two classes means the model clearly prefers one class. A small margin means the top two classes are almost tied, so that sample is the most uncertain.
- **Corrected (EN)**: In margin sampling, a small margin means high uncertainty, so samples with the smallest margin are selected first.
- **해설**: "Use the top two probabilities", "Small margin → High uncertainty" (p.12). LC·Entropy는 $U$가 **클수록** 먼저인데, margin만 $U$가 **작을수록** 먼저야. 방향이 반대인 게 함정이야.

← pool: `10_Active_Learning_I` Q6

**3.** `[L10]` **Answer**: False
- **Why (EN)**: The two are swapped. Voter entropy counts the votes, so it uses only the predicted labels; consensus entropy averages the members' probabilities, so it uses the full distributions.
- **Corrected (EN)**: Voter entropy only considers the predicted label of each model, while consensus entropy averages the prediction probabilities and uses the full probability distribution.
- **해설**: Voter Entropy "Count committee votes", $U(x)=H(V(y)/C)$, "Only considers the predicted label of each model" (p.16). Consensus Entropy "Average prediction probabilities", $U(x)=H(P_C)$, "Uses the full probability distribution" (p.17). 표 (p.19): Voter = "Final class labels only", Consensus = "Full probability distributions".

← pool: `10_Active_Learning_I` Q8

**4.** `[L10]` **Answer**: False
- **Why (EN)**: That describes epistemic uncertainty. Aleatoric uncertainty comes from data noise, such as sensor noise, occlusion, or motion blur, and cannot be reduced by collecting more data.
- **Corrected (EN)**: Epistemic uncertainty is caused by limited model knowledge and can be reduced with more labeled data, whereas aleatoric uncertainty is caused by data noise and cannot be reduced by collecting more data.
- **해설**: "Aleatoric Uncertainty: Uncertainty caused by data noise" (Sensor noise, Occlusion, Motion blur, Intrinsic ambiguity) "Cannot be reduced by collecting more data" / "Epistemic Uncertainty: Uncertainty caused by limited model knowledge" "Can be reduced with more labeled data" / "Active Learning mainly targets Epistemic Uncertainty." (p.25)

← pool: `10_Active_Learning_I` Q10

**5.** `[L10]` **Answer**: True
- **Why (EN)**: The first term is how uncertain the averaged prediction is, and the second is how uncertain each sampled model is on its own. Their difference is the mutual information between the label and the model parameters.
- **해설**: $\mathrm{BALD}(x)=I(y,\theta\mid x,\mathcal D)=\mathcal H[p(y|x,\mathcal D)]-\mathbb E_{p(\theta|\mathcal D)}[\mathcal H[p(y|x,\theta)]]$, "BALD = Entropy of mean prediction − Mean entropy of predictions", $\mathrm{BALD}(x)\approx\mathcal H[\bar p(y|x)]-\frac1T\sum_{t=1}^{T}\mathcal H[p_t(y|x)]$ (p.29). 순서를 뒤집은 지문이 나오기 좋아.

← pool: `10_Active_Learning_I` Q12

**6.** `[L10]` **Answer**: (B)
- **Why (EN)**: Both samples have a top-1 probability of 0.50, so least confidence gives 0.5 to both. The margin is 0.01 for $x_a$ and 0.25 for $x_b$, and the smaller margin is more uncertain, so margin picks $x_a$. Entropy uses all three probabilities: about 0.74 nats for $x_a$ and 1.04 nats for $x_b$, so entropy picks $x_b$.
- **해설**: 설명용 숫자야 (슬라이드 수치 아님). LC는 top-1만 (p.11), margin은 top-2만 (p.12), entropy는 분포 전체 (p.13)를 보니까, class가 3개 이상이면 셋이 서로 다른 sample을 고를 수 있어. $x_a$는 "두 class 사이에서 헷갈림", $x_b$는 "셋 다 조금씩 가능"이야. (보충) class가 2개면 셋 다 top-1 확률만의 함수라 순위가 같아.

← pool: `10_Active_Learning_I` Q16

**7.** `[L10]` **Answer**: (B)
- **Why (EN)**: Training many networks is expensive, so MC Dropout keeps dropout on at inference and treats each dropout mask as a different model. How much these passes disagree is used as an estimate of epistemic uncertainty.
- **해설**: "MC Dropout performs multiple stochastic forward passes with different dropout masks, approximating an ensemble of neural networks. The variance of their predictions is used as an estimate of epistemic uncertainty." (p.27) DBAL: "Estimate epistemic uncertainty by running multiple forward passes with different dropout masks" (p.28). (C): T번 forward하니까 오히려 느려져. (D): aleatoric은 data에서 지울 수 없고, BALD로 **점수에서** 빼는 거야 (p.29–30).

← pool: `10_Active_Learning_I` Q20

**8.** `[L10]` **Answer**: (C)
- **Why (EN)**: A hybrid strategy combines several criteria, such as uncertainty and representativeness. (A) and (B) are swapped, and avoiding redundant samples is part of diversity sampling.
- **해설**: Diversity "Select representative samples that cover the data distribution", "Avoid selecting redundant samples", "Cover different regions of the feature space" (p.20). Expected Model Change "Select samples expected to change the model the most after labeling" (p.21). Hybrid "Combine multiple criteria for better sample selection", "Uncertain and representative", "Informative and diverse", "Most modern active learning methods" (p.22).

← pool: `10_Active_Learning_I` Q22

**9.** `[L10·PC]` Uncertainty sampling — Least Confidence · Margin · Entropy. **Answer**: `False` (accept: `(-U_margin).topk(b)`, `U_margin.argsort()[:b]`. 핵심은 "작은 margin부터")

- **Process (EN)**: This code computes three uncertainty scores from the softmax output P(y|x) of one trained model on unlabeled samples. Least confidence uses only the top-1 probability, margin uses the gap between the top-2 probabilities, and entropy uses the whole distribution; least confidence and entropy select the samples with the highest score, but margin selects the samples with the smallest gap. The blank is `False`, because a small margin means high uncertainty; no model is updated here, and the selected samples are labeled by a human and used to retrain the model.

← `exam/pseudocode_final.md` AL-2

**10.** `[L10·PC]` MC Dropout + BALD — DBAL acquisition. **Answer**: `train` (accept: dropout layer만 켜기, `for m in model.modules(): if isinstance(m, nn.Dropout): m.train()`)

- **Process (EN)**: This code implements the acquisition step of DBAL: MC Dropout keeps dropout active at inference, so T stochastic forward passes of one network act like T sampled models. BALD is the entropy of the mean prediction minus the mean entropy of the individual predictions; it is near zero when every pass is equally unsure (aleatoric noise) and large when the passes are individually confident but disagree (epistemic uncertainty), and the top-b samples are queried and the model is retrained. The blank is `train`, because in eval mode dropout is off, all T passes give the same output, and BALD becomes zero.

← `exam/pseudocode_final.md` AL-4

</details>
