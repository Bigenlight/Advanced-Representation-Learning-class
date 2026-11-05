# Mock Quiz — Final, L08 (Weakly-Supervised Learning I)

> 단원 미니 모의고사. 10문제 = **T/F 5 + MCQ 3 + code 2** (중간 실제 비율 10 : 5 : 5의 절반).
> **답은 영어로 직접 써.** T/F는 고르고 이유까지, False면 고친 문장까지. MCQ도 왜 그 답인지 한 문장. code는 빈칸 하나 + 과정 설명 2–3문장 (입력 → target은 어디서 오나 → loss → 누가 update되나).
> 헷갈리면 `False?`처럼 `?`를 붙여. 맞아도 `?` 붙은 문제는 다시 봐. 답을 다 적기 전엔 맨 아래를 열지 마.
> 출처: `exam/_parts/08_Weakly_Supervised_Learning_I.md` C 섹션, `exam/pseudocode_final.md`. 모두 작성자와 다른 Opus가 독립 검수함 (2026-11-05). 슬라이드 페이지 번호는 `lectures/08_Weakly_Supervised_Learning_I.pdf` 기준.

### 1. [T/F] `[L08]`
In MIL for weakly supervised object detection, each image is a bag of candidate windows, and only the instance labels are known.

> T/F:
>
> Why:
>
> Corrected (False일 때):

### 2. [T/F] `[L08]`
During re-training in standard MIL, all proposals in negative images are treated as negatives, and no negative proposals are sampled from positive images.

> T/F:
>
> Why:
>
> Corrected (False일 때):

### 3. [T/F] `[L08]`
In Multifold MIL, the detector for fold k is trained with positive examples in all folds but k and the negative examples of fold k only.

> T/F:
>
> Why:
>
> Corrected (False일 때):

### 4. [T/F] `[L08]`
Objectness is a class-independent prior that steers re-localization towards whole objects and away from background.

> T/F:
>
> Why:
>
> Corrected (False일 때):

### 5. [T/F] `[L08]`
In WSDDN, the classification stream compares regions for each class, while the localization stream associates each region with a class score.

> T/F:
>
> Why:
>
> Corrected (False일 때):

### 6. [MCQ] `[L08]`
What is the purpose of replacing the hard max with log-sum-exp in re-training?  
(A) To reduce the number of proposals per image before re-localization  
(B) To learn from multiple high-scoring proposals, not a single one  
(C) To remove the need for negative images during re-training  
(D) To turn image-level labels into bounding-box labels before re-training

> Answer:
>
> Why:

### 7. [MCQ] `[L08]`
What is the softmax over proposals in WSDDN for?  
(A) It compares regions for each class, so it acts as the localization stream  
(B) It compares classes for each region, so it acts as the classification stream  
(C) It normalizes image-level scores across classes  
(D) It removes proposals with low objectness

> Answer:
>
> Why:

### 8. [MCQ] `[L08]`
How does the method with ConcreteDropBlock differ from OICR and PCL in what it mainly tackles?  
(A) It mainly improves pseudo-label quality by grouping proposals into clusters  
(B) It removes the need for image-level labels by generating its own labels  
(C) It tackles discriminative parts by hiding the most relevant regions  
(D) It replaces the deep network with sliding windows over the whole image

> Answer:
>
> Why:

### 9. [Code] `[L08·PC]`
Fill in the blank, and explain what this code does in 2–3 sentences.

```python
scores = A(feats_pos).squeeze(1)        # A(x_b) for every proposal of a positive image
b_star = scores.______()                # re-localize
s_pos = A(feats_pos[b_star]).squeeze()  # only one positive instance per image
s_neg = A(feats_neg).squeeze(1).max()   # best window of a negative image
loss = F.relu(delta - s_pos) + F.relu(delta + s_neg)  # hinge with margin delta
loss.backward()                         # re-train
```

> Blank:
>
> Process:

### 10. [Code] `[L08·PC]`
Fill in the blank, and explain what this code does in 2–3 sentences.

```python
with torch.______():                    # build the pseudo label
    top = s_prev[:, c].argmax()         # highest scoring proposal of class c
    near = iou(boxes, boxes[top:top + 1]).squeeze(1) > t  # plus surrounding ones
target = torch.full((R,), C)            # C = background, the (C+1)-th class
target[near] = c
loss = F.cross_entropy(head_k(x), target)  # stage-k instance classifier
loss.backward()
```

> Blank:
>
> Process:

---

<details>
<summary><b>정답 · 모범 답안</b> (다 쓴 뒤에 열어)</summary>

**1.** `[L08]` **Answer**: False
- **Why (EN)**: Only the bag label is known. We know that the class is in the image, but we do not know which window contains it.
- **Corrected (EN)**: In MIL, each image is a bag of candidate windows, and only the bag label (not the instance labels) is known.
- **해설**: "only the bag label (not the instance labels) is known." (p.15) instance label을 알면 그냥 supervised detection이야.

← pool: `08_Weakly_Supervised_Learning_I` Q3

**2.** `[L08]` **Answer**: True
- **Why (EN)**: A negative image does not contain the class, so all of its windows are safe negatives. In a positive image, MIL does not know which other windows are truly negative, so it does not use them.
- **해설**: "all proposals in negative images are treated as negatives ... no negative proposals are sampled from positive images." (p.21) 이게 supervised detection과의 차이야. 이유 문장은 슬라이드 원문이 아니라 해석이야.

← pool: `08_Weakly_Supervised_Learning_I` Q6

**3.** `[L08]` **Answer**: False
- **Why (EN)**: Only the positive examples are held out by fold. All negative examples are used, because negative images have no uncertain window to overfit to.
- **Corrected (EN)**: The detector for fold k is trained with positive examples in all folds but k and all negative examples.
- **해설**: "Train using positive examples in all folds but k, and all negative examples" (p.24). "uncertain window가 없어서"는 슬라이드 문장이 아니라 해석이야.

← pool: `08_Weakly_Supervised_Learning_I` Q10

**4.** `[L08]` **Answer**: True
- **Why (EN)**: Objectness measures how likely a window is to contain an object of any class, so it does not depend on the target class. It favors complete objects over parts and background.
- **해설**: "Objectness provides a class-independent prior that guides WSOD toward complete objects rather than discriminative object parts or background regions." (p.28)

← pool: `08_Weakly_Supervised_Learning_I` Q12

**5.** `[L08]` **Answer**: False
- **Why (EN)**: The two roles are swapped. The classification stream scores classes for each region, and the localization stream compares regions for each class.
- **Corrected (EN)**: The classification stream associates each region with a class score, while the localization stream compares regions for each class.
- **해설**: "Classification: Associates each region with a class score", "Localization: Compares regions for each class" (p.34). p.36 그림의 softmax over classes / softmax over proposals와 같은 얘기야.

← pool: `08_Weakly_Supervised_Learning_I` Q14

**6.** `[L08]` **Answer**: (B)
- **Why (EN)**: Log-sum-exp relaxes the hard max, so several high-scoring proposals contribute to training and one wrong pick matters less.
- **해설**: "Hedge your bets on multiple proposals" (p.22). 식의 negative image 줄을 보면 negative image는 그대로 써서 (C)는 틀려.

← pool: `08_Weakly_Supervised_Learning_I` Q19

**7.** `[L08]` **Answer**: (A)
- **Why (EN)**: Normalizing across proposals makes regions compete for each class, which shows where that class is in the image.
- **해설**: "Localization: Compares regions for each class" (p.34), p.36 그림 "Softmax over proposals". p.36 블록 이름은 "Multiple instance detection network"이고, OICR은 이 블록으로 WSDDN을 써 (paper). p.34엔 softmax 표기가 없어. (B)는 softmax over classes야.

← pool: `08_Weakly_Supervised_Learning_I` Q24

**8.** `[L08]` **Answer**: (C)
- **Why (EN)**: OICR and PCL mainly improve pseudo-label quality, while this work suppresses the most discriminative regions so that the detector attends to the entire object.
- **해설**: "Unlike OICR and PCL that mainly improve pseudo-label quality, this work explicitly addresses the discriminative-part problem." (p.38)

← pool: `08_Weakly_Supervised_Learning_I` Q26

**9.** `[L08·PC]` Standard MIL — re-localize + re-train. **Answer**: `argmax` (same meaning: `scores.argmax(dim=0)`, `torch.argmax(scores)`)

- **Process (EN)**: This code implements one re-localize and re-train round of the standard MIL pipeline. It scores every proposal of a positive image with the appearance model A and keeps the top window as the only positive instance, then applies a hinge loss that pushes this window above Δ and the best window of a negative image below −Δ, and this loss updates A. The blank is `argmax`, which re-localizes the object by selecting the highest-scoring window.

← `exam/pseudocode_final.md` WS-1

**10.** `[L08·PC]` OICR — one refinement stage. **Answer**: `no_grad` (accept: `inference_mode`)

- **Process (EN)**: This code implements one refinement stage of OICR. For a class c in the image, it labels the previous stage's top-scoring proposal and the proposals overlapping it as class c and every other proposal as background, then applies cross-entropy to the stage-k classifier, whose gradient also reaches the shared proposal features. The blank is `no_grad`, which treats these pseudo labels as a fixed target, so no gradient flows back through them.

← `exam/pseudocode_final.md` WS-3

</details>
