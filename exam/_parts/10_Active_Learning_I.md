# 10. Active Learning I (Query Strategies & Deep Bayesian Active Learning) — Core Ideas & Question Pool

> 기말 형식 기준이야 (`exam/PLAYBOOK.md` 7장, 형식은 공지 전까지 중간과 같다고 가정).
> T/F엔 전부 `Why (EN)`을 붙였고, False엔 `Corrected (EN)`도 붙였어.
> 코드 빈칸은 D의 F3·F4·F8·F9·F10(짧은 조각 + 과정 설명)과 `exam/pseudocode_final.md`의 L10 카드(AL-1–)에 있어. D의 코드 조각은 acl env torch로 같은 동작을 돌려 확인했어.
> 슬라이드 cite는 두 개뿐이야: MC Dropout = Gal & Ghahramani, ICML 2016 (p.27, slide엔 "Gal et al., ICML 2016", 일치), DBAL = Gal, Islam & Ghahramani, ICML 2017 (p.28·32, 일치). BALD(p.29–31)와 p.34의 Snapshot · DEE · Multi-head엔 cite가 없어.
> "(보충)" 표시는 슬라이드에 없는 해석·유도야. 슬라이드 문장으로 답해야 하는 시험에선 비중을 낮게 둬.
> 검수: (검수 후 갱신)

## A. Core ideas (one block per method/paper/concept, in lecture order)

### Active learning: motivation, goal, workflow (p.4–5)
- **Problem**: "We often have… Huge amounts of unlabeled data / Only a limited labeling budget / Human annotation is expensive and time-consuming". Key question: "If we can label only a small subset, which samples should we choose?" (p.4)
- **Core idea**: Let the model choose which unlabeled samples a human should label. "Goal of Active Learning: Select the most informative unlabeled samples / Maximize model improvement with minimum annotation cost" (p.4). Objective: "Achieve the highest accuracy / With the fewest labeled examples". Workflow: "Select informative samples → Human annotation → Retrain the model → Repeat" (p.5).
- **vs. previous / key difference**: (보충) Semi-supervised (L06–L07) and weakly supervised (L08–L09) learning keep the labels they are given and fill the gap with targets the model makes itself (pseudo labels, CAM seeds). Active learning keeps true human labels but spends the labeling budget only on samples the model selects.
- **Keywords**: unlabeled data, limited labeling budget, annotation cost, most informative samples, select → human annotation → retrain → repeat
- **Definition ↔ trap**: "Active learning selects the most informative unlabeled samples for human annotation and retrains the model." ↔ "Active learning labels the selected samples with the model's own predictions and retrains on them."
- 한 줄: label 예산이 적으니, 어느 sample에 사람 label을 달지 모델이 골라. 고르고 → 사람이 달고 → 다시 학습 → 반복.

### Notations and the acquisition function (p.6–7)
- **Problem**: "Which unlabeled sample should we label next?" (p.7) Unlabeled samples need a number that ranks them.
- **Core idea**: An acquisition function "Assigns an informativeness score to each unlabeled sample" (p.7). Notation (p.6): $K$ = number of classes; $(\mathbf x^l, y)\sim\mathcal X$, $y\in\{0,1\}^K$ = labeled set with one-hot labels; $\mathbf u\sim\mathcal U$ = unlabeled set; $\mathcal D=\mathcal X\cup\mathcal U$; $U(\mathbf x)$ = "Scoring function for active learning selection"; $P_\theta(y|\mathbf x)$ = softmax classifier; $\hat y=\arg\max_{y\in\mathcal Y}P_\theta(y|\mathbf x)$ = most confident prediction; $B$ = "Labeling budget (the maximum number of samples to label)"; $b$ = "Batch size". "The process of identifying the most valuable examples to label next is referred to as 'sampling strategy' or 'query strategy'." (p.7)
- **vs. previous / key difference**: (보충) In FixMatch-style pseudo labeling, a confidence score decides which unlabeled samples are **trusted** (high confidence). In uncertainty-based active learning, a score decides which samples are **sent to a human**, usually the low-confidence ones.
- **Keywords**: acquisition function, informativeness score, scoring function $U(\mathbf x)$, sampling strategy, query strategy, labeling budget $B$, batch size $b$, $\hat y$
- **Definition ↔ trap**: "An acquisition function assigns an informativeness score to each unlabeled sample." ↔ "An acquisition function assigns an informativeness score to each labeled sample."
- 한 줄: $U(\mathbf x)$ = unlabeled sample마다 "label 달 가치" 점수. 이 점수로 다음에 label 달 sample을 고르는 절차가 sampling(query) strategy야.

### Uncertainty sampling (p.9–10)
- **Problem**: "Not all unlabeled samples are equally informative." (p.9)
- **Core idea**: "Label the samples that the current model is least confident about." "Samples near the decision boundary often provide the greatest learning benefit." (p.9) The question is "How uncertain is this probability?" $P_\theta(y|\mathbf x)$, with "Three common measures: Least Confidence, Margin Sampling, Entropy" (p.10).
- **vs. previous / key difference**: It uses the predicted probabilities of **one** model. Query-by-committee (p.14) uses the disagreement of **several** models instead.
- **Keywords**: least confident, decision boundary, uncertainty region, greatest learning benefit, least confidence, margin, entropy
- **Definition ↔ trap**: "Uncertainty sampling labels the samples the current model is least confident about, near the decision boundary." ↔ "Uncertainty sampling labels the samples the current model is most confident about, far from the decision boundary."
- 한 줄: 지금 모델이 제일 헷갈려하는(decision boundary 근처) sample에 label을 달아.

### Least Confidence (p.11)
- **Problem**: We need one uncertainty number from a softmax output.
- **Core idea**: Predict $\hat y=\arg\max_y P(y|x)$, then $U(x)=1-P(\hat y|x)$. "Uses only the highest predicted probability". "Lower confidence → Higher priority".
- **vs. previous / key difference**: It looks at the top-1 probability only. Margin looks at the top two (p.12), entropy at the whole distribution (p.13).
- **Keywords**: $\hat y$, top-1 probability, $1-P(\hat y|x)$, lower confidence, higher priority
- **Definition ↔ trap**: "Least confidence uses only the highest predicted probability; lower confidence gives higher priority." ↔ "Least confidence uses the whole distribution; higher confidence gives higher priority."
- 한 줄: 1등 확률만 봐. 1등 확률이 낮을수록($U$가 클수록) 먼저 골라.

### Margin Sampling (p.12)
- **Problem**: Least confidence ignores how close the runner-up class is.
- **Core idea**: "Use the top two probabilities": $U(x)=P(\hat y_1|x)-P(\hat y_2|x)$. "Small margin → High uncertainty".
- **vs. previous / key difference**: The selection direction is reversed. For least confidence and entropy a **larger** $U$ is picked first; for margin the **smallest** $U$ is picked first. (보충) Some papers use the negative margin so that "pick the largest score" works for every measure.
- **Keywords**: top two probabilities, $\hat y_1$, $\hat y_2$, small margin, high uncertainty
- **Definition ↔ trap**: "In margin sampling, a small margin means high uncertainty." ↔ "In margin sampling, a large margin means high uncertainty."
- 한 줄: 1등과 2등 확률의 차이. 차이가 작을수록 헷갈리는 거라 먼저 골라 (이것만 작은 쪽을 골라).

### Entropy Sampling (p.13)
- **Problem**: Least confidence and margin ignore most of the classes.
- **Core idea**: $U(x)=-\sum_y P(y|x)\log P(y|x)$. "Uses the entire probability distribution". "Higher entropy → Higher uncertainty".
- **vs. previous / key difference**: It uses all $K$ probabilities. (보충) With $K=2$ the three measures rank samples the same way, because each one depends only on the top-1 probability. With $K\ge3$ they can disagree, e.g. $(0.50, 0.49, 0.01)$ vs $(0.50, 0.25, 0.25)$: LC ties, margin picks the first, entropy picks the second.
- **Keywords**: Shannon entropy, entire probability distribution, higher entropy, higher uncertainty, uniform distribution = maximum entropy $\log K$
- **Definition ↔ trap**: "Entropy sampling uses the entire probability distribution; higher entropy means higher uncertainty." ↔ "Entropy sampling uses only the top two probabilities; lower entropy means higher uncertainty."
- 한 줄: 확률 분포 전체가 얼마나 퍼졌나. 고르게 퍼질수록(uniform에 가까울수록) entropy가 커서 먼저 골라.

### Query-by-Committee, QBC (p.14–15)
- **Problem**: Uncertainty sampling trusts the probabilities of a single model.
- **Core idea**: "Instead of relying on a single model, measure uncertainty by disagreement among multiple models." "Train multiple models (committee) / Each model predicts independently / Select samples with the largest disagreement" (p.14). Figure: votes Cat 3, Dog 2 → "Vote distribution [3/5, 2/5]" → "Uncertainty (disagreement) High" → "Select this sample for labeling"; all members vote Dog → "Most models agree → low uncertainty". Disagreement is measured by "Voter Entropy / Consensus Entropy / KL Divergence" (p.15).
- **vs. previous / key difference**: Uncertainty sampling = how unsure **one** model is. QBC = how much **several** models disagree.
- **Keywords**: committee, multiple models, predict independently, votes, vote distribution, largest disagreement
- **Definition ↔ trap**: "QBC selects samples with the largest disagreement among committee members." ↔ "QBC selects samples on which the committee members agree the most."
- 한 줄: 모델 여러 개(committee)에게 투표시켜서, 의견이 제일 갈리는 sample에 label을 달아.

### Voter Entropy (p.16)
- **Problem**: Turn committee votes into one disagreement score.
- **Core idea**: "Count committee votes": $U(x)=H\!\left(\frac{V(y)}{C}\right)$, where $C$ = "Number of models in the committee", $V(y)$ = "Number of committee members voting for class $y$", $\frac{V(y)}{C}$ = "Fraction of votes for class $y$", $\mathcal H(\cdot)$ = "Shannon entropy". "Only considers the predicted label of each model". "High vote diversity → High uncertainty".
- **vs. previous / key difference**: It uses labels only. A member that votes Cat with probability 0.51 counts the same as one with 0.99 (보충). Consensus entropy uses the probabilities.
- **Keywords**: votes, $V(y)$, $C$, fraction of votes, predicted label only, vote diversity
- **Definition ↔ trap**: "Voter entropy only considers the predicted label of each model." ↔ "Voter entropy uses the full probability distribution of each model."
- 한 줄: 표만 세서 표 비율의 entropy를 내. 표가 고르게 갈릴수록 커 (5명 중 3:2면 크고 5:0이면 0).

### Consensus Entropy (p.17)
- **Problem**: Votes throw away how confident each member is.
- **Core idea**: "Average prediction probabilities": $P_C(y)=\frac1C\sum_{c=1}^{C}P_{\theta_c}(y|\mathbf x)$ ("Average probability over all committee members", $\theta_c$ = "Parameters of model $c$"), then $U(x)=H(P_C)$. "Uses the full probability distribution". "Measures uncertainty of the committee consensus".
- **vs. previous / key difference**: Voter entropy = entropy of the **vote fractions**; consensus entropy = entropy of the **averaged probabilities**. (보충) It is high both when members disagree and when all members agree on a flat distribution, so it cannot tell these two cases apart. The KL measure (p.18) and BALD (p.29) can.
- **Keywords**: average prediction probabilities, $P_C$, $\theta_c$, full probability distribution, committee consensus
- **Definition ↔ trap**: "Consensus entropy is the entropy of the members' averaged probabilities." ↔ "Consensus entropy is the average of the members' individual entropies."
- 한 줄: committee 확률을 먼저 평균 내고, 그 평균 분포의 entropy를 봐.

### KL Divergence for QBC (p.18–19)
- **Problem**: Measure directly how far the members are from each other.
- **Core idea**: $U(x)=\frac1C\sum_{c=1}^{C}D_{KL}(P_{\theta_c}\,\|\,P_C)$. "Measures disagreement between each model and the committee average". "Larger KL → Larger disagreement". "How far each model's prediction is from the committee average." Summary table (p.19): Voter Entropy — "Final class labels only"; Consensus Entropy — "Full probability distributions"; KL Divergence — "Distance between distributions". Voter: "How many people chose which class?" / Consensus: "How much uncertainty is there on average?" / KL: "How much does each model differ from the average opinion?"
- **vs. previous / key difference**: It is zero whenever all members predict the same distribution, even a flat one. (보충, 유도) $\frac1C\sum_c D_{KL}(P_{\theta_c}\|P_C)=H(P_C)-\frac1C\sum_c H(P_{\theta_c})$, i.e. consensus entropy minus the average member entropy. This is the same form as BALD (p.29) with committee members in place of sampled models.
- **Keywords**: $D_{KL}(P_{\theta_c}\|P_C)$, committee average, distance between distributions, larger KL, larger disagreement
- **Definition ↔ trap**: "The KL measure averages how far each member's prediction is from the committee average." ↔ "The KL measure computes how far the committee average is from the uniform distribution."
- 한 줄: 각 member가 평균 의견에서 얼마나 떨어졌나를 평균 내. 클수록 의견이 갈린 거야.

### Diversity Sampling (p.20)
- **Problem**: (보충) If we take the top-$b$ uncertain samples, many of them can be near-duplicates of each other.
- **Core idea**: "Select representative samples that cover the data distribution." "Avoid selecting redundant samples / Cover different regions of the feature space / Improve generalization".
- **vs. previous / key difference**: Uncertainty asks how unsure the model is about **one** sample. Diversity looks at the **selected set** and the data distribution.
- **Keywords**: representative samples, cover the data distribution, redundant samples, feature space, generalization
- **Definition ↔ trap**: "Diversity sampling selects representative samples that cover different regions of the feature space." ↔ "Diversity sampling selects the samples closest to the current decision boundary."
- 한 줄: data 분포를 고루 덮는 대표 sample을 골라. 비슷한 것끼리 중복해서 뽑지 마.

### Expected Model Change (p.21)
- **Problem**: A sample is worth labeling if its label would actually change the model.
- **Core idea**: "Select samples expected to change the model the most after labeling." "Large parameter update / Large loss reduction / High training impact".
- **vs. previous / key difference**: It scores the effect on **training**, not the uncertainty of the **prediction**. (보충) The label is unknown before annotation, so the change has to be estimated as an expectation over possible labels; this is why p.23 lists "Expected model update estimation" as a cost.
- **Keywords**: change the model the most, parameter update, loss reduction, training impact
- **Definition ↔ trap**: "Expected model change selects samples expected to change the model the most after labeling." ↔ "Expected model change selects samples expected to change the model the least, to keep training stable."
- 한 줄: label을 달면 모델이 제일 많이 바뀔 것 같은 sample을 골라.

### Hybrid Strategy (p.22)
- **Problem**: Each single criterion misses something.
- **Core idea**: "Combine multiple criteria for better sample selection." "Uncertain and representative / Informative and diverse / Most modern active learning methods".
- **vs. previous / key difference**: Uncertainty, diversity, and expected model change each optimize one criterion; hybrid methods combine them.
- **Keywords**: combine multiple criteria, uncertain and representative, informative and diverse, most modern methods
- **Definition ↔ trap**: "Most modern active learning methods combine multiple criteria, such as uncertainty and representativeness." ↔ "Most modern active learning methods rely on a single criterion."
- 한 줄: uncertainty + diversity처럼 기준 여러 개를 섞어. 요즘 방법 대부분이 이래.

### Limitations of classic query strategies (p.23)
- **Problem**: The strategies above are designed by hand.
- **Core idea**: "Hand-crafted acquisition functions: Entropy, Margin, Diversity, … / No single strategy works best for all tasks / Difficult to adapt across datasets". "Computationally expensive: Committee models / Pairwise similarity computation / Expected model update estimation / Retraining after every acquisition round". "Separate objectives: Uncertainty / Diversity / Expected model change". Question: "Can we learn the acquisition function directly using deep neural networks?"
- **vs. previous / key difference**: It motivates Part 3 (Deep Active Learning) and the next lecture. (보충) Each cost matches a strategy: committee models → QBC, pairwise similarity → diversity, expected model update → expected model change.
- **Keywords**: hand-crafted, no single best strategy, computationally expensive, retraining after every round, separate objectives, learn the acquisition function
- **Definition ↔ trap**: "No single hand-crafted strategy works best for all tasks." ↔ "Entropy sampling works best for all tasks, so other strategies are unnecessary."
- 한 줄: 손으로 만든 점수는 task마다 맞는 게 다르고, 비싸고, 목표가 따로 놀아 → acquisition을 DNN으로 직접 배울 수 없을까?

### Aleatoric vs epistemic uncertainty (p.25)
- **Problem**: Not every kind of uncertainty is worth paying for with a label.
- **Core idea**: "Aleatoric Uncertainty: Uncertainty caused by data noise" (Sensor noise, Occlusion, Motion blur, Intrinsic ambiguity) — "Cannot be reduced by collecting more data". "Epistemic Uncertainty: Uncertainty caused by limited model knowledge" ("Model has not seen similar examples", "Large uncertainty in model parameters") — "Can be reduced with more labeled data". "Active Learning mainly targets Epistemic Uncertainty."
- **vs. previous / key difference**: (보충) LC, margin, and entropy score one softmax output, so they mix both kinds. Ensembles, MC Dropout, and BALD try to isolate the epistemic part.
- **Keywords**: aleatoric, data noise, sensor noise, occlusion, motion blur, intrinsic ambiguity, epistemic, limited model knowledge, model parameters, reducible with more labeled data
- **Definition ↔ trap**: "Epistemic uncertainty is caused by limited model knowledge and can be reduced with more labeled data." ↔ "Aleatoric uncertainty is caused by limited model knowledge and can be reduced with more labeled data."
- 한 줄: aleatoric = data 자체 noise라 label을 더 모아도 안 줄어. epistemic = 모델이 몰라서라 label을 더 모으면 줄어. AL은 epistemic을 노려.

### Ensemble-based Estimation (p.26)
- **Problem**: We need a way to measure epistemic uncertainty of a model.
- **Core idea**: "Train multiple models independently. If they disagree, → the sample is likely informative." "EX) AdaBoost, Bootstrapping, Random forests". "Problems: Need multiple networks / High training cost / High inference cost".
- **vs. previous / key difference**: The same idea as QBC (disagreement of a committee), but with deep networks every member is a full network, so it is expensive.
- **Keywords**: multiple models, trained independently, disagreement, AdaBoost, bootstrapping, random forests, high training cost, high inference cost
- **Definition ↔ trap**: "Ensemble-based estimation needs multiple networks, with high training and inference cost." ↔ "Ensemble-based estimation needs only a single network, so it is cheap."
- 한 줄: 독립적으로 학습한 모델 여러 개가 엇갈리면 informative. 단 network가 여러 개라 학습·추론 둘 다 비싸.

### MC Dropout (Gal & Ghahramani, ICML 2016 ← slide엔 "Gal et al., ICML 2016", p.27)
- **Problem**: A real ensemble of networks is too expensive (p.26).
- **Core idea**: "Instead of training multiple networks, apply dropout during inference." "Different dropout masks ≈ Different models". "Advantages: Single network / Low memory / Good approximation to Bayesian inference". "MC Dropout performs multiple stochastic forward passes with different dropout masks, approximating an ensemble of neural networks. The variance of their predictions is used as an estimate of epistemic uncertainty."
- **vs. previous / key difference**: Normal dropout is used only in training and turned off at test time. MC Dropout keeps it on at inference and runs several passes. Compared with an ensemble, it needs one trained network instead of many.
- **Keywords**: dropout during inference, different dropout masks, stochastic forward passes, approximate ensemble, single network, low memory, Bayesian inference, variance of predictions, epistemic uncertainty
- **Definition ↔ trap**: "MC Dropout applies dropout during inference and runs multiple stochastic forward passes." ↔ "MC Dropout applies dropout only during training and trains multiple networks."
- 한 줄: test 때도 dropout을 켜 두고 여러 번 forward하면, mask마다 다른 모델처럼 행동해서 싼 ensemble이 돼.

### DBAL: Deep Bayesian Active Learning (Gal, Islam & Ghahramani, ICML 2017, p.28, 32)
- **Problem**: "Goal: We want to find 'samples that the model is truly unsure about.'" (p.28)
- **Core idea**: "Approximate Bayesian inference using MC Dropout / Estimate epistemic uncertainty by running multiple forward passes with different dropout masks / Query samples with the highest uncertainty". Pipeline: "Train a classifier with dropout → Perform multiple stochastic forward passes → Estimate predictive uncertainty (BALD) → Select the most informative unlabeled samples → Query labels and retrain" (p.28). Observation (p.32): "MC Dropout consistently outperforms random sampling. / Bayesian uncertainty leads to better sample selection." "More informative samples → Higher accuracy with the same labeling budget". Figure: MNIST test accuracy vs number of acquired images, curves BALD, Var Ratios, Max Entropy, MBR, Random.
- **vs. previous / key difference**: Classic strategies (Part 2) score one deterministic model. DBAL gets a Bayesian approximation of a deep image classifier from MC Dropout and picks samples by BALD, which targets epistemic uncertainty.
- **Keywords**: Bayesian inference, MC Dropout, epistemic uncertainty, multiple forward passes, BALD, query labels and retrain, outperforms random sampling, same labeling budget
- **Definition ↔ trap**: "DBAL approximates Bayesian inference with MC Dropout and selects samples by BALD." ↔ "DBAL trains an ensemble of independent networks and selects samples at random."
- 한 줄: MC Dropout으로 deep model의 Bayesian 불확실성을 근사하고, BALD로 "진짜 모르는" sample을 골라 사람에게 물어.

### BALD acquisition function (p.29–31)
- **Problem**: A sample can look uncertain because the data is noisy (aleatoric) or because the model lacks knowledge (epistemic). Only the second is worth a label (p.25).
- **Core idea**: "BALD measures the mutual information between the label y of sample x and the model parameters θ." $\mathrm{BALD}(x)=I(y,\theta\mid x,\mathcal D)=\mathcal H[p(y|x,\mathcal D)]-\mathbb E_{p(\theta|\mathcal D)}\big[\mathcal H[p(y|x,\theta)]\big]\approx\mathcal H[\bar p(y|x)]-\frac1T\sum_{t=1}^{T}\mathcal H[p_t(y|x)]$. "BALD = Entropy of mean prediction − Mean entropy of predictions". First term: "How uncertain is the average of all models?" Second term: "How uncertain is each model individually?" (p.29). Case 1, aleatoric (blurred image): "All models make similar predictions and are equally uncertain." → "BALD(x) ≈ High − High ≈ 0 (Small)" → "Labeling this sample is unlikely to significantly improve the model." (p.30). Case 2, epistemic: "Different models make different predictions, while each model is individually confident." → "BALD(x) ≈ High − Low = Large" → "Labeling this sample is expected to provide the greatest improvement to the model." (p.31). "BALD measures how much the models disagree with each other." (p.32)
- **vs. previous / key difference**: (보충) Only the entropy of the **averaged** prediction equals the first term: QBC's consensus entropy, or entropy on the MC-Dropout-averaged output ("Max Entropy" in the p.32 graph). It is "High" in both cases (p.30–31), so it cannot separate them. The entropy of a **single** sampled model is different: it is high in case 1 but **low** in case 2, because each model is individually confident. BALD subtracts the second term, the part each model is uncertain about by itself. (보충) BALD was originally proposed by Houlsby et al., "Bayesian Active Learning for Classification and Preference Learning", arXiv 2011; the slide gives no cite.
- **Keywords**: mutual information, $y$ and $\theta$, entropy of mean prediction, mean entropy of predictions, $T$ samples, case 1 aleatoric ≈ 0, case 2 epistemic large
- **Definition ↔ trap**: "BALD = entropy of the mean prediction − mean entropy of the predictions." ↔ "BALD = mean entropy of the predictions − entropy of the mean prediction."
- 한 줄: "평균 예측의 entropy − 각 예측 entropy의 평균". 모델들이 각자는 확신하는데 서로 다를 때만 커서, epistemic만 골라.

### Beyond MC Dropout: efficient ensemble approximations (p.33–35)
- **Problem**: "Naïve ensembles provide the most reliable uncertainty estimates. However, training many independent networks is computationally expensive." (p.33)
- **Core idea**: "Snapshot Ensemble: Save multiple checkpoints from one training run using a cyclic learning rate. Different local minima act as an implicit ensemble." "Diversity Encouraging Ensemble (DEE): Initialize multiple models from a shared backbone. Encourage prediction diversity with dropout." "Multi-head / Split-head Network: Share one feature extractor. Use multiple prediction heads as lightweight ensemble members." (p.34; figure: each gives an "Ensemble Prediction (average / vote)"). Takeaway (p.35): "Accuracy: Naïve Ensemble > Efficient Approximations", "Efficiency: Efficient Approximations > Naïve Ensemble", "MC Dropout remains one of the best trade-offs between uncertainty quality and computational cost."
- **vs. previous / key difference**: All three try to get ensemble disagreement without training N networks from scratch. Naive ensemble = N separate trainings; snapshot = one run, many checkpoints; DEE = one pretrained backbone, many fine-tuned models with dropout; multi-head = one backbone, many heads. (보충) Snapshot Ensembles = Huang et al., ICLR 2017; the slide gives no cite for any of the three.
- **Keywords**: naïve ensemble, snapshot ensemble, cyclic learning rate, checkpoints, local minima, implicit ensemble, DEE, shared backbone, dropout, multi-head, split-head, shared feature extractor, accuracy vs efficiency, trade-off
- **Definition ↔ trap**: "Naïve ensembles are more accurate, efficient approximations are more efficient, and MC Dropout is one of the best trade-offs." ↔ "Efficient approximations are both more accurate and more efficient than naïve ensembles."
- 한 줄: N개를 따로 학습하면 제일 정확하지만 비싸. snapshot(한 번 학습에서 checkpoint 여러 개), DEE(공유 backbone + dropout), multi-head(몸통 하나 + head 여러 개)로 싸게 흉내 내. MC Dropout이 여전히 좋은 절충이야.

## B. Must-know definitions & distinctions
- **Active learning**: select the most informative unlabeled samples, get human labels for them, retrain, and repeat, to "Maximize model improvement with minimum annotation cost" (p.4–5).
- **Labeling budget $B$ vs batch size $b$**: "the maximum number of samples to label" vs "Batch size" (p.6). (보충) In the loop, $b$ is read as the number of samples queried per round, so there are about $B/b$ rounds.
- **Acquisition function $U(\mathbf x)$**: "Assigns an informativeness score to each unlabeled sample" (p.7); "Scoring function for active learning selection" (p.6).
- **Sampling strategy = query strategy**: "The process of identifying the most valuable examples to label next" (p.7).
- **Uncertainty sampling**: "Label the samples that the current model is least confident about"; "Samples near the decision boundary often provide the greatest learning benefit" (p.9).
- **Least confidence vs margin vs entropy**: top-1 only, $1-P(\hat y|x)$, larger = first / top two, $P(\hat y_1|x)-P(\hat y_2|x)$, **smaller** = first / entire distribution, $-\sum_y P\log P$, larger = first (p.11–13).
- **Query-by-committee (QBC)**: "measure uncertainty by disagreement among multiple models"; "Select samples with the largest disagreement" (p.14).
- **Voter vs consensus entropy vs KL**: "Final class labels only" / "Full probability distributions" / "Distance between distributions" (p.19).
- **$V(y)$, $C$, $P_C$**: votes for class $y$, number of committee members, average probability over all members (p.16–17).
- **Diversity sampling**: "Select representative samples that cover the data distribution"; "Avoid selecting redundant samples" (p.20).
- **Expected model change**: "Select samples expected to change the model the most after labeling" — large parameter update, large loss reduction (p.21).
- **Hybrid strategy**: "Combine multiple criteria", e.g. "Uncertain and representative"; "Most modern active learning methods" (p.22).
- **Three limitations**: hand-crafted (no single best strategy), computationally expensive, separate objectives (p.23).
- **Aleatoric vs epistemic**: "caused by data noise", "Cannot be reduced by collecting more data" vs "caused by limited model knowledge", "Can be reduced with more labeled data"; "Active Learning mainly targets Epistemic Uncertainty" (p.25).
- **Ensemble-based estimation**: models trained independently; disagreement → informative; needs multiple networks, high training and inference cost (p.26).
- **MC Dropout**: dropout during **inference**, multiple stochastic forward passes, "Different dropout masks ≈ Different models", single network (p.27).
- **DBAL pipeline**: train with dropout → multiple stochastic forward passes → BALD → select → query labels and retrain (p.28).
- **BALD**: mutual information between $y$ and $\theta$ = "Entropy of mean prediction − Mean entropy of predictions" (p.29).
- **BALD case 1 vs case 2**: aleatoric, all models equally uncertain → High − High ≈ 0 / epistemic, models disagree but each confident → High − Low = Large (p.30–31).
- **Naïve ensemble vs efficient approximations**: more accurate vs more efficient; MC Dropout = "one of the best trade-offs" (p.33–35).
- **Snapshot vs DEE vs multi-head**: one run with cyclic LR, checkpoints at different local minima / models initialized from a shared backbone, diversity with dropout / one shared feature extractor, multiple heads (p.34).

## C. Question pool (22 questions)

### Q1 [T/F]
The goal of active learning is to select the most informative unlabeled samples and maximize model improvement with minimum annotation cost.
**Answer**: True
- **Why (EN)**: There is a lot of unlabeled data but only a limited labeling budget, so active learning spends human labels only on the samples that help the model the most.
- **해설**: "Goal of Active Learning: Select the most informative unlabeled samples / Maximize model improvement with minimum annotation cost" (p.4). p.5 Objective도 같은 말이야: "Achieve the highest accuracy / With the fewest labeled examples".

### Q2 [T/F]
In the active learning workflow, the model labels the selected samples with its own predictions and is then retrained on these pseudo labels.
**Answer**: False
- **Why (EN)**: The selected samples are labeled by a human annotator. Active learning saves annotation by choosing which samples to label, not by replacing human labels with the model's predictions.
- **Corrected (EN)**: In the active learning workflow, the selected informative samples are labeled by human annotation, the model is retrained, and the process repeats.
- **해설**: Workflow "Select informative samples / Human annotation / Retrain the model / Repeat" (p.5). 그림도 "Let's label!" → "Valuable labeled dataset" → "Better model"이야. 모델이 자기 예측으로 target을 만드는 건 L07 pseudo label(self-training) 쪽이야.

### Q3 [T/F]
An acquisition function assigns an informativeness score to each labeled sample, and the process of identifying the most valuable examples to label next is called the sampling strategy or query strategy.
**Answer**: False
- **Why (EN)**: Labeled samples already have labels. The score is computed for each unlabeled sample to decide which one to label next.
- **Corrected (EN)**: An acquisition function assigns an informativeness score to each unlabeled sample, and the process of identifying the most valuable examples to label next is called the sampling strategy or query strategy.
- **해설**: "Acquisition Function: Assigns an informativeness score to each unlabeled sample." / "The process of identifying the most valuable examples to label next is referred to as "sampling strategy" or "query strategy"." (p.7) 뒷부분은 맞아. 한 단어(unlabeled → labeled)만 비틀었어.

### Q4 [T/F]
Uncertainty sampling labels the samples that the current model is least confident about, because samples near the decision boundary often provide the greatest learning benefit.
**Answer**: True
- **Why (EN)**: Far from the boundary the model is already confident, so a new label adds little. Samples near the boundary are the most uncertain and give the greatest learning benefit when labeled.
- **해설**: "Label the samples that the current model is least confident about." / "Not all unlabeled samples are equally informative." / "Samples near the decision boundary often provide the greatest learning benefit." (p.9) 그림에서 보라색 원("Selected by uncertainty sampling")은 전부 "Uncertainty region (near the boundary)" 안에 있어.

### Q5 [T/F]
Least confidence uses only the highest predicted probability, and lower confidence gives higher priority.
**Answer**: True
- **Why (EN)**: $U(x)=1-P(\hat y|x)$ depends only on the top-1 probability. The lower that probability, the larger $U(x)$, so the sample is chosen earlier.
- **해설**: $\hat y=\arg\max_y P(y|x)$, $U(x)=1-P(\hat y|x)$, "Uses only the highest predicted probability", "Lower confidence → Higher priority" (p.11).

### Q6 [T/F]
In margin sampling, $U(x)=P(\hat y_1|x)-P(\hat y_2|x)$, and samples with a large margin are selected first because they are the most uncertain.
**Answer**: False
- **Why (EN)**: A large gap between the top two classes means the model clearly prefers one class. A small margin means the top two classes are almost tied, so that sample is the most uncertain.
- **Corrected (EN)**: In margin sampling, a small margin means high uncertainty, so samples with the smallest margin are selected first.
- **해설**: "Use the top two probabilities", "Small margin → High uncertainty" (p.12). LC·Entropy는 $U$가 **클수록** 먼저인데, margin만 $U$가 **작을수록** 먼저야. 방향이 반대인 게 함정이야.

### Q7 [T/F]
Entropy sampling uses only the top two predicted probabilities, and higher entropy means higher uncertainty.
**Answer**: False
- **Why (EN)**: Using the top two probabilities is margin sampling. Entropy sums $-P\log P$ over all classes, so it uses the entire probability distribution.
- **Corrected (EN)**: Entropy sampling uses the entire probability distribution, and higher entropy means higher uncertainty.
- **해설**: "Uses the entire probability distribution", "Higher entropy → Higher uncertainty", $U(x)=-\sum_y P(y|x)\log P(y|x)$ (p.13). "Use the top two probabilities"는 margin이야 (p.12). 뒷부분은 맞아.

### Q8 [T/F]
In query-by-committee, voter entropy averages the committee members' full probability distributions, while consensus entropy only counts the predicted label of each member.
**Answer**: False
- **Why (EN)**: The two are swapped. Voter entropy counts the votes, so it uses only the predicted labels; consensus entropy averages the members' probabilities, so it uses the full distributions.
- **Corrected (EN)**: Voter entropy only considers the predicted label of each model, while consensus entropy averages the prediction probabilities and uses the full probability distribution.
- **해설**: Voter Entropy "Count committee votes", $U(x)=H(V(y)/C)$, "Only considers the predicted label of each model" (p.16). Consensus Entropy "Average prediction probabilities", $U(x)=H(P_C)$, "Uses the full probability distribution" (p.17). 표 (p.19): Voter = "Final class labels only", Consensus = "Full probability distributions".

### Q9 [T/F]
The KL-divergence disagreement measure averages, over the committee members, how far each member's prediction is from the committee average, and a larger KL means a larger disagreement.
**Answer**: True
- **Why (EN)**: If every member predicts the same distribution, each KL to the average is zero. The more the members differ from the average opinion, the larger the score.
- **해설**: $U(x)=\frac1C\sum_{c=1}^{C}D_{KL}(P_{\theta_c}\|P_C)$, "Measures disagreement between each model and the committee average", "Larger KL → Larger disagreement", "How far each model's prediction is from the committee average." (p.18)

### Q10 [T/F]
Aleatoric uncertainty is caused by limited model knowledge and can be reduced by collecting more labeled data.
**Answer**: False
- **Why (EN)**: That describes epistemic uncertainty. Aleatoric uncertainty comes from data noise, such as sensor noise, occlusion, or motion blur, and cannot be reduced by collecting more data.
- **Corrected (EN)**: Epistemic uncertainty is caused by limited model knowledge and can be reduced with more labeled data, whereas aleatoric uncertainty is caused by data noise and cannot be reduced by collecting more data.
- **해설**: "Aleatoric Uncertainty: Uncertainty caused by data noise" (Sensor noise, Occlusion, Motion blur, Intrinsic ambiguity) "Cannot be reduced by collecting more data" / "Epistemic Uncertainty: Uncertainty caused by limited model knowledge" "Can be reduced with more labeled data" / "Active Learning mainly targets Epistemic Uncertainty." (p.25)

### Q11 [T/F]
MC Dropout turns dropout off at inference and instead trains several networks to approximate an ensemble.
**Answer**: False
- **Why (EN)**: MC Dropout keeps dropout on at inference and runs multiple stochastic forward passes through a single network. Each dropout mask acts like a different model, so no extra networks are trained.
- **Corrected (EN)**: MC Dropout applies dropout during inference and performs multiple stochastic forward passes with different dropout masks in a single network, approximating an ensemble.
- **해설**: "Instead of training multiple networks, apply dropout during inference." / "Different dropout masks ≈ Different models" / "Advantages: Single network, Low memory, Good approximation to Bayesian inference" (p.27).

### Q12 [T/F]
BALD equals the entropy of the mean prediction minus the mean entropy of the individual predictions.
**Answer**: True
- **Why (EN)**: The first term is how uncertain the averaged prediction is, and the second is how uncertain each sampled model is on its own. Their difference is the mutual information between the label and the model parameters.
- **해설**: $\mathrm{BALD}(x)=I(y,\theta\mid x,\mathcal D)=\mathcal H[p(y|x,\mathcal D)]-\mathbb E_{p(\theta|\mathcal D)}[\mathcal H[p(y|x,\theta)]]$, "BALD = Entropy of mean prediction − Mean entropy of predictions", $\mathrm{BALD}(x)\approx\mathcal H[\bar p(y|x)]-\frac1T\sum_{t=1}^{T}\mathcal H[p_t(y|x)]$ (p.29). 순서를 뒤집은 지문이 나오기 좋아.

### Q13 [T/F]
For a blurred image on which every sampled model predicts about 0.5 for cat and 0.5 for dog, BALD is large, so this sample should be labeled first.
**Answer**: False
- **Why (EN)**: Each model is already uncertain by itself, so the entropy of the mean prediction (high) minus the mean entropy (also high) is about zero. This is aleatoric uncertainty, and labeling the sample is unlikely to improve the model much.
- **Corrected (EN)**: For such a sample BALD ≈ High − High ≈ 0, so labeling it is unlikely to significantly improve the model.
- **해설**: Case 1 "Aleatoric uncertainty (Inherently ambiguous data)": "All models make similar predictions and are equally uncertain." / "Sample x (blurred image)" / "Entropy: High" / "BALD(x) ≈ High − High ≈ 0 (Small)" / "→ Labeling this sample is unlikely to significantly improve the model." (p.30) BALD가 큰 건 case 2(모델끼리 엇갈리는데 각자는 확신)야 (p.31).

### Q14 [T/F]
Efficient ensemble approximations such as snapshot ensembles give more accurate uncertainty estimates than a naive ensemble of independently trained networks, because their checkpoints come from different local minima.
**Answer**: False
- **Why (EN)**: Naive ensembles give the most reliable uncertainty estimates. Efficient approximations win on computational cost, not on accuracy.
- **Corrected (EN)**: A naive ensemble is more accurate, while efficient approximations such as snapshot ensembles are more efficient; MC Dropout remains one of the best trade-offs between uncertainty quality and computational cost.
- **해설**: "Naïve ensembles provide the most reliable uncertainty estimates. However, training many independent networks is computationally expensive." (p.33) Snapshot: "Save multiple checkpoints from one training run using a cyclic learning rate. Different local minima act as an implicit ensemble." (p.34) Takeaway: "Accuracy: Naïve Ensemble > Efficient Approximations / Efficiency: Efficient Approximations > Naïve Ensemble" (p.35).

### Q15 [MCQ]
What is an acquisition function for in active learning?
(A) To generate pseudo labels for unlabeled samples so that the model can be retrained without a human
(B) To assign an informativeness score to each unlabeled sample so that we can decide which sample to label next
(C) To measure the test accuracy of the model after each acquisition round
(D) To augment the labeled samples so that fewer labels are needed
**Answer**: (B)
- **Why (EN)**: The core question of active learning is which unlabeled sample to label next. The acquisition function $U(\mathbf x)$ answers it with a score that ranks the unlabeled samples.
- **해설**: "Core Question: Which unlabeled sample should we label next?" / "Acquisition Function: Assigns an informativeness score to each unlabeled sample." (p.7) / $U(\mathbf x)$ = "Scoring function for active learning selection" (p.6). (A)는 pseudo label(L07), label은 사람이 달아 (p.5).

### Q16 [MCQ]
A 3-class model predicts $P(y|x_a)=(0.50, 0.49, 0.01)$ and $P(y|x_b)=(0.50, 0.25, 0.25)$. Which statement is correct?
(A) Least confidence prefers $x_a$, margin prefers $x_b$, and entropy prefers $x_b$.
(B) Least confidence gives both the same score, margin prefers $x_a$, and entropy prefers $x_b$.
(C) All three measures prefer $x_b$.
(D) All three measures prefer $x_a$.
**Answer**: (B)
- **Why (EN)**: Both samples have a top-1 probability of 0.50, so least confidence gives 0.5 to both. The margin is 0.01 for $x_a$ and 0.25 for $x_b$, and the smaller margin is more uncertain, so margin picks $x_a$. Entropy uses all three probabilities: about 0.74 nats for $x_a$ and 1.04 nats for $x_b$, so entropy picks $x_b$.
- **해설**: 설명용 숫자야 (슬라이드 수치 아님). LC는 top-1만 (p.11), margin은 top-2만 (p.12), entropy는 분포 전체 (p.13)를 보니까, class가 3개 이상이면 셋이 서로 다른 sample을 고를 수 있어. $x_a$는 "두 class 사이에서 헷갈림", $x_b$는 "셋 다 조금씩 가능"이야. (보충) class가 2개면 셋 다 top-1 확률만의 함수라 순위가 같아.

### Q17 [MCQ]
Which disagreement measure in query-by-committee uses only the final class labels predicted by the committee members?
(A) Voter entropy
(B) Consensus entropy
(C) KL divergence to the committee average
(D) Least confidence
**Answer**: (A)
- **Why (EN)**: Voter entropy counts the votes $V(y)$ and takes the entropy of the vote fractions $V(y)/C$, so it needs only each member's predicted label. Consensus entropy and the KL measure need the full probability distributions.
- **해설**: Voter Entropy "Only considers the predicted label of each model" (p.16). 표 (p.19): Voter = "Final class labels only", Consensus = "Full probability distributions", KL = "Distance between distributions". (D)는 committee 측정이 아니라 한 모델의 uncertainty야 (p.11).

### Q18 [MCQ]
Which is NOT listed on the slides as a limitation of classic, hand-crafted query strategies?
(A) No single strategy works best for all tasks, and it is difficult to adapt across datasets.
(B) They are computationally expensive, e.g. committee models, pairwise similarity computation, and retraining after every acquisition round.
(C) Uncertainty, diversity, and expected model change are separate objectives.
(D) They need the true labels of all unlabeled samples to compute the acquisition score.
**Answer**: (D)
- **Why (EN)**: Acquisition scores are computed from the model's predictions on unlabeled samples. If the true labels were needed, there would be nothing left to query.
- **해설**: p.23 "Limitations": "Hand-crafted acquisition functions" (No single strategy works best for all tasks / Difficult to adapt across datasets), "Computationally expensive" (Committee models / Pairwise similarity computation / Expected model update estimation / Retraining after every acquisition round), "Separate objectives" (Uncertainty / Diversity / Expected model change). 결론 질문: "Can we learn the acquisition function directly using deep neural networks?"

### Q19 [MCQ]
Which is NOT a property of epistemic uncertainty?
(A) It arises when the model has not seen similar examples.
(B) It reflects large uncertainty in the model parameters.
(C) It can be reduced with more labeled data.
(D) It is caused by sensor noise, occlusion, or motion blur, and cannot be reduced by collecting more data.
**Answer**: (D)
- **Why (EN)**: (D) describes aleatoric uncertainty, which comes from noise in the data itself. Epistemic uncertainty comes from limited model knowledge, so more labeled data reduces it.
- **해설**: p.25 Aleatoric: "Sensor noise, Occlusion, Motion blur, Intrinsic ambiguity", "Cannot be reduced by collecting more data". Epistemic: "Model has not seen similar examples", "Large uncertainty in model parameters", "Can be reduced with more labeled data". "Active Learning mainly targets Epistemic Uncertainty."

### Q20 [MCQ]
What is MC Dropout for in deep active learning?
(A) To prevent overfitting during training only; it plays no role when scoring unlabeled samples.
(B) To approximate an ensemble with a single network: multiple stochastic forward passes with different dropout masks give a spread of predictions that estimates epistemic uncertainty.
(C) To make inference faster by dropping units at test time.
(D) To remove the aleatoric noise from unlabeled images before they are queried.
**Answer**: (B)
- **Why (EN)**: Training many networks is expensive, so MC Dropout keeps dropout on at inference and treats each dropout mask as a different model. How much these passes disagree is used as an estimate of epistemic uncertainty.
- **해설**: "MC Dropout performs multiple stochastic forward passes with different dropout masks, approximating an ensemble of neural networks. The variance of their predictions is used as an estimate of epistemic uncertainty." (p.27) DBAL: "Estimate epistemic uncertainty by running multiple forward passes with different dropout masks" (p.28). (C): T번 forward하니까 오히려 느려져. (D): aleatoric은 data에서 지울 수 없고, BALD로 **점수에서** 빼는 거야 (p.29–30).

### Q21 [MCQ]
On a BALD slide, different sampled models make different predictions on a sample, while each model is individually confident. Which describes this case?
(A) Entropy of the mean prediction: low; mean entropy: low; BALD small, so the sample is not worth labeling.
(B) Entropy of the mean prediction: high; mean entropy: low; BALD large, so labeling it is expected to give the greatest improvement.
(C) Entropy of the mean prediction: high; mean entropy: high; BALD ≈ 0, because this is aleatoric uncertainty.
(D) Entropy of the mean prediction: low; mean entropy: high; BALD large.
**Answer**: (B)
- **Why (EN)**: The models disagree, so their average is spread out and its entropy is high. Each model alone is confident, so the mean of the individual entropies is low, and BALD = high − low is large. This is epistemic uncertainty, which labels can reduce.
- **해설**: Case 2 "Epistemic uncertainty (Model lacks knowledge)": "Different models make different predictions, while each model is individually confident." / "Entropy: High" / "BALD(x) ≈ High − Low = Large" / "→ Labeling this sample is expected to provide the greatest improvement to the model." (p.31) (C)는 case 1 (p.30). 설명용 숫자: 모델들이 (0.9, 0.1)과 (0.1, 0.9)로 반반 갈리면 평균 (0.5, 0.5)의 entropy ≈ 0.69 nats, 각 entropy ≈ 0.33 → BALD ≈ 0.37. (D)는 불가능해 (보충: entropy가 concave라 BALD ≥ 0이고, 평균의 entropy가 entropy의 평균보다 작을 수 없어).

### Q22 [MCQ]
Which pairing of query strategy and selection criterion is correct?
(A) Diversity sampling — samples expected to change the model the most after labeling
(B) Expected model change — representative samples that cover different regions of the feature space
(C) Hybrid strategy — combine multiple criteria, e.g. samples that are uncertain and representative
(D) Uncertainty sampling — avoid selecting redundant samples
**Answer**: (C)
- **Why (EN)**: A hybrid strategy combines several criteria, such as uncertainty and representativeness. (A) and (B) are swapped, and avoiding redundant samples is part of diversity sampling.
- **해설**: Diversity "Select representative samples that cover the data distribution", "Avoid selecting redundant samples", "Cover different regions of the feature space" (p.20). Expected Model Change "Select samples expected to change the model the most after labeling" (p.21). Hybrid "Combine multiple criteria for better sample selection", "Uncertain and representative", "Informative and diverse", "Most modern active learning methods" (p.22).

## D. Fill-in-the-blank pool (10 questions)

### F1 [Blank]
Goal of Active Learning: Select the most (1) ______ unlabeled samples. Maximize model improvement with minimum (2) ______ cost.
**Answer**: (1) informative (2) annotation (accept: labeling) — p.4. p.5 Objective: "Achieve the highest accuracy / With the fewest labeled examples".

### F2 [Blank]
Least Confidence: $\hat y=\arg\max_y P(y|x)$, acquisition score $U(x)=1-P(\,\text{(1)}\,|x)$. It uses only the highest predicted probability. Lower confidence → Higher (2) ______.
**Answer**: (1) $\hat y$ (accept: y-hat, the most confident (predicted) class) (2) priority (accept: uncertainty) — p.11. $U$가 클수록 먼저 골라.

### F3 [Blank + process]
Fill in the blank, and explain what this code does in 2–3 sentences.
```python
top2 = probs.topk(2, dim=1).values            # probs: [N, K] softmax outputs on unlabeled samples
margin = top2[:, 0] - top2[:, 1]               # U(x) = P(y1|x) - P(y2|x)
idx = margin.topk(b, largest=______).indices   # b samples to send to the annotator
```
**Answer**: False
- **Process (EN)**: The current model predicts class probabilities for every unlabeled sample, and the margin is the gap between the two largest probabilities. A small margin means the top two classes are almost tied, i.e. high uncertainty, so the b samples with the smallest margin are sent for human annotation and the model is retrained.
- **해설**: "Use the top two probabilities", "Small margin → High uncertainty" (p.12). LC·entropy·BALD였으면 `largest=True`(기본값)야. 이 방향 차이가 빈칸 포인트야.

### F4 [Blank + process]
Fill in the blank, and explain what this code does in 2–3 sentences.
```python
U = -(probs * probs.______()).sum(dim=1)   # probs: [N, K]; entropy of each prediction
idx = U.topk(b).indices                    # most uncertain b samples
```
**Answer**: log (accept: log2 — bits 단위라 순위가 같아; `clamp_min(1e-12).log`처럼 0을 피한 log)
- **Process (EN)**: For each unlabeled sample, the entropy $-\sum_y P(y|x)\log P(y|x)$ is computed over all classes, so the whole distribution is used. Higher entropy means higher uncertainty, so the b samples with the largest entropy are queried for labels.
- **해설**: "Uses the entire probability distribution", "Higher entropy → Higher uncertainty" (p.13). 실제 코드에선 $P=0$에서 $\log 0$을 피하려고 작은 값을 더하거나 clamp해.

### F5 [Blank]
Voter Entropy: $U(x)=H\!\left(\dfrac{V(y)}{\text{(1)}}\right)$, where $V(y)$ is the number of committee members (2) ______ for class $y$. It only considers the predicted label of each model.
**Answer**: (1) $C$ (accept: the number of models in the committee) (2) voting — p.16. 예 (p.14 그림): 5명 중 Cat 3, Dog 2 → $H(3/5, 2/5)\approx 0.67$ nats (0.97 bits, 설명용 계산), 5명 모두 Dog → 0.

### F6 [Blank]
Consensus Entropy: $P_C(y)=\dfrac{1}{\text{(1)}}\sum_{c=1}^{C}P_{\theta_c}(y|\mathbf x)$, $U(x)=H(P_C)$. KL Divergence: $U(x)=\dfrac1C\sum_{c=1}^{C}D_{KL}\big(P_{\theta_c}\,\|\,\text{(2)}\big)$.
**Answer**: (1) $C$ (2) $P_C$ (accept: the committee average, the consensus distribution) — p.17–18. KL은 "each model ↔ committee average" 방향이야. (보충) 이 KL 평균은 $H(P_C)-\frac1C\sum_c H(P_{\theta_c})$와 같아서, BALD(p.29)와 같은 꼴이야.

### F7 [Blank]
Aleatoric Uncertainty: uncertainty caused by data (1) ______. It (2) ______ be reduced by collecting more data. Epistemic Uncertainty: uncertainty caused by limited model (3) ______. It can be reduced with more labeled data. Active Learning mainly targets (4) ______ Uncertainty.
**Answer**: (1) noise (2) cannot (3) knowledge (4) Epistemic — p.25.

### F8 [Blank + process]
Fill in the blank, and explain what this code does in 2–3 sentences.
```python
model.______()                        # keep dropout ON at inference
with torch.no_grad():
    P = torch.stack([model(x).softmax(dim=1) for _ in range(T)])   # [T, N, K]
```
**Answer**: train (accept: dropout layer만 train mode로 켜는 것, e.g. `m.train()` for each `nn.Dropout`)
- **Process (EN)**: The trained classifier is kept in a mode where dropout is still active, and T stochastic forward passes are run on the unlabeled samples. Each pass uses a different dropout mask, so it acts like a different model, and the spread of the T predictions estimates epistemic uncertainty. No gradient is needed because we only score samples.
- **해설**: "Instead of training multiple networks, apply dropout during inference." / "Different dropout masks ≈ Different models" (p.27). 보통 inference는 `model.eval()`로 dropout을 끄는데, MC Dropout은 그걸 일부러 안 해. (보충) `model.train()`은 BatchNorm도 batch 통계 쪽으로 바꾸니까, 실전에선 dropout layer만 켜는 경우가 많아.

### F9 [Blank + process]
Fill in the blank, and explain what this code does in 2–3 sentences.
```python
p_mean = P.mean(dim=0)                                            # mean prediction [N, K]
H_mean = -(p_mean * p_mean.clamp_min(1e-12).log()).sum(-1)        # entropy of mean prediction
mean_H = -(P * P.clamp_min(1e-12).log()).sum(-1).______(dim=0)    # mean entropy of predictions
bald = H_mean - mean_H
```
**Answer**: mean
- **Process (EN)**: From T MC Dropout passes, BALD subtracts the average entropy of the individual predictions from the entropy of the averaged prediction. It is about zero when every pass is equally uncertain (aleatoric) and large when the passes are individually confident but disagree (epistemic), so the samples with the largest BALD are queried for labels.
- **해설**: "BALD = Entropy of mean prediction − Mean entropy of predictions", $\mathrm{BALD}(x)\approx\mathcal H[\bar p(y|x)]-\frac1T\sum_{t=1}^{T}\mathcal H[p_t(y|x)]$ (p.29), case 1 ≈ 0 (p.30), case 2 Large (p.31). `.sum(dim=0)`이면 $T$배가 돼서 틀려.

### F10 [Blank + process]
Fill in the blank, and explain what this loop does in 2–3 sentences. (`acquisition`, `annotate`, `remove`, `train`은 자리표시 함수)
```python
for r in range(B // b):                        # B: labeling budget, b: batch size
    scores = acquisition(model, X_unlabeled)   # U(x) for every unlabeled sample
    idx = scores.topk(______).indices           # most informative samples this round
    X_labeled += annotate(X_unlabeled[idx])    # human annotation
    X_unlabeled = remove(X_unlabeled, idx)
    model = train(model, X_labeled)            # retrain
```
**Answer**: b
- **Process (EN)**: Starting from a model trained on a small labeled set, in each round the acquisition function scores every unlabeled sample and the b most informative ones are selected. A human labels them, they move from the unlabeled pool to the labeled set, and the model is retrained; this repeats until the labeling budget B is spent.
- **해설**: Workflow "Select informative samples / Human annotation / Retrain the model / Repeat" (p.5), $B$ = "Labeling budget (the maximum number of samples to label)", $b$ = "Batch size" (p.6). (보충) 한 round에 $b$개씩 고른다는 해석이야. margin처럼 작을수록 좋은 점수면 `largest=False`가 필요해 (F3).

## E. Component roles

"Without it" 열은 슬라이드의 설명에서 이어지는 해석이야. 슬라이드 문장 그대로는 아니야.

| Part | What is it for? | Without it | Page |
|---|---|---|---|
| Labeling budget $B$ | It caps how many samples a human will label in total, which is why we must choose carefully. | With unlimited labels we could simply label everything, and active learning would not be needed. | p.4, 6 |
| Batch size $b$ (per round) | It sets how many samples are queried before the model is retrained. (interpretation of "Batch size") | — (b = 1 would retrain after every single label, which is very costly; p.23 "Retraining after every acquisition round") | p.6, 23 |
| Human annotation | It gives true labels for the selected samples. | The model would train on its own guesses, which is pseudo labeling, not active learning. | p.5 |
| Retraining | It updates the model with the new labels so that the next round's scores reflect what it now knows. | Scores would never change, and the same kind of samples would keep being selected. | p.5 |
| Acquisition function $U(\mathbf x)$ | It assigns an informativeness score to each unlabeled sample so that we can rank them. | Samples would be chosen at random (the "Random" baseline in p.32). | p.6–7 |
| Top-1 probability (least confidence) | It measures how confident the model is in its best guess. | — | p.11 |
| Top-2 gap (margin) | It detects when the two best classes are almost tied. | Least confidence cannot tell "0.50 vs 0.49" from "0.50 vs 0.25". | p.12 |
| Whole distribution (entropy) | It measures how spread out the prediction is over all classes. | Spread over the remaining classes is ignored. | p.13 |
| Committee (QBC) | It provides several independent opinions so that uncertainty can be measured by disagreement. | We must trust the probabilities of a single model. | p.14 |
| Vote fraction $V(y)/C$ | It turns the members' labels into a distribution whose entropy measures vote diversity. | — | p.16 |
| Committee average $P_C$ | It is the consensus prediction; consensus entropy and the KL measure are both defined with it. | There is no common reference to measure each member against. | p.17–18 |
| KL to $P_C$ | It measures how far each member is from the average opinion, so it is zero when all agree. | Consensus entropy alone cannot tell "members disagree" from "all members agree on a flat distribution". (interpretation) | p.18 |
| Diversity criterion | It makes the selected set cover different regions of the feature space. | A batch can be filled with near-duplicate samples. | p.20 |
| Expected model change | It favors samples whose labels would change the model the most. | — | p.21 |
| Hybrid combination | It combines uncertainty with representativeness or diversity. | One criterion alone misses what the others capture. | p.22 |
| Dropout at inference (MC Dropout) | It makes each forward pass use a different mask, so one network behaves like many models. | Every pass is identical, so there is no spread to measure. | p.27 |
| $T$ stochastic forward passes | They give $T$ samples of the prediction, whose spread estimates epistemic uncertainty. | With one pass there is no ensemble to compare. | p.27–29 |
| First BALD term $\mathcal H[\bar p]$ | It measures how uncertain the averaged prediction is (total uncertainty, interpretation). | — | p.29 |
| Second BALD term $\frac1T\sum_t\mathcal H[p_t]$ | It measures how uncertain each model is by itself, which is the aleatoric part (interpretation), and is subtracted. | The score is high for noisy, inherently ambiguous samples that labels cannot fix (p.30 case 1). | p.29–30 |
| Cyclic learning rate (snapshot) | It lets one training run visit several local minima, and a checkpoint is saved at each. | One run gives only one model. | p.34 |
| Shared backbone + dropout (DEE) | It initializes several models from one pretrained backbone and uses dropout to make their predictions diverse. | Each member must be trained from scratch. | p.34 |
| Shared feature extractor + multiple heads | It makes the heads lightweight ensemble members on top of one network. | Each member needs its own full network. | p.34 |

## F. Pipelines

"Target comes from"와 "Gradient goes to" 줄은 슬라이드 문장이 아니라 해석이야.

**Active learning loop** (p.4–7)
1. Train the classifier $P_\theta(y|\mathbf x)$ on the current labeled set $\mathcal X$.
2. Score every unlabeled sample in $\mathcal U$ with the acquisition function $U(\mathbf x)$.
3. Select the $b$ most informative samples.
4. A human annotates them, and they move from $\mathcal U$ to $\mathcal X$.
5. Retrain the model on $\mathcal X$.
6. Repeat until the labeling budget $B$ is used up.
- **Target comes from**: human annotation (true labels), only for the selected samples.
- **Gradient goes to**: the classifier, during retraining on the labeled set. Scoring is only a forward pass.

**Uncertainty sampling** (p.9–13)
1. Run the current model on each unlabeled sample to get $P(y|x)$.
2. Compute least confidence $1-P(\hat y|x)$, the margin $P(\hat y_1|x)-P(\hat y_2|x)$, or the entropy $-\sum_y P\log P$.
3. Pick the most uncertain samples: largest LC or entropy, **smallest** margin.
4. Human annotation → retrain → repeat.
- **Target comes from**: human labels.
- **Gradient goes to**: the single classifier, during retraining.

**Query-by-committee** (p.14–19)
1. Train multiple models (the committee).
2. Each model predicts every unlabeled sample independently.
3. Measure disagreement: voter entropy (votes only), consensus entropy (entropy of the averaged probabilities), or the mean KL of each member to the committee average.
4. Select the samples with the largest disagreement, get human labels, retrain the committee, and repeat.
- **Target comes from**: human labels.
- **Gradient goes to**: every committee member, each trained on the labeled set. (The slide does not say how the members are made different; p.26 lists AdaBoost, bootstrapping, and random forests as ensemble examples.)

**DBAL with BALD** (p.28–32)
1. Train a classifier with dropout on the labeled set.
2. For each unlabeled sample, perform $T$ stochastic forward passes with dropout on (MC Dropout).
3. Compute BALD = entropy of the mean prediction − mean entropy of the predictions.
4. Select the unlabeled samples with the largest BALD (high epistemic uncertainty).
5. Query their labels from a human and retrain; repeat.
- **Target comes from**: human labels for the queried samples.
- **Gradient goes to**: the single dropout network, during training. The $T$ scoring passes need no gradient.

**Efficient ensemble approximations** (p.34)
1. Snapshot: one training run with a cyclic learning rate; save a checkpoint at each local minimum.
2. DEE: pretrain a backbone, initialize several models from it, and fine-tune them with dropout.
3. Multi-head: share one feature extractor and train several prediction heads.
4. In all three, the members' predictions are combined (average / vote), and their disagreement is used like an ensemble's.
- **Takeaway**: naive ensemble > approximations in accuracy, approximations > naive ensemble in efficiency, MC Dropout = one of the best trade-offs (p.35).
