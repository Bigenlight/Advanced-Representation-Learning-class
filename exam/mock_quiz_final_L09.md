# Mock Quiz — Final, L09 (Weakly-Supervised Learning II)

> 단원 미니 모의고사. 10문제 = **T/F 5 + MCQ 3 + code 2** (중간 실제 비율 10 : 5 : 5의 절반).
> **답은 영어로 직접 써.** T/F는 고르고 이유까지, False면 고친 문장까지. MCQ도 왜 그 답인지 한 문장. code는 빈칸 하나 + 과정 설명 2–3문장 (입력 → target은 어디서 오나 → loss → 누가 update되나).
> 헷갈리면 `False?`처럼 `?`를 붙여. 맞아도 `?` 붙은 문제는 다시 봐. 답을 다 적기 전엔 맨 아래를 열지 마.
> 출처: `exam/_parts/09_Weakly_Supervised_Learning_II.md` C 섹션, `exam/pseudocode_final.md`. 모두 작성자와 다른 Sonnet이 정답을 가리고 먼저 푼 뒤 대조함 (code card는 독립 테스트) (2026-11-09). 슬라이드 페이지 번호는 `lectures/09_Weakly_Supervised_Learning_II.pdf` 기준.

### 1. [T/F] `[L09]`
Like gradient-based saliency, CAM needs a separate gradient calculation of the class score with respect to the input to obtain its activation maps.

> T/F:
>
> Why:
>
> Corrected (False일 때):

### 2. [T/F] `[L09]`
When class maps are thresholded to generate training data, low-score regions are used as foreground and high-score regions as background.

> T/F:
>
> Why:
>
> Corrected (False일 때):

### 3. [T/F] `[L09]`
SEAM assumes that activation maps should be invariant to the input, so it forces the CAM of an affine-transformed image to be identical to the CAM of the original image without compensating for the transformation.

> T/F:
>
> Why:
>
> Corrected (False일 때):

### 4. [T/F] `[L09]`
Adversarial erasing does not erase random patches: at each step, it erases the discriminative region highlighted by the classification network's heatmap and feeds the erased image back, so that the network mines further object regions.

> T/F:
>
> Why:
>
> Corrected (False일 때):

### 5. [T/F] `[L09]`
In the webly supervised pipeline of Shen et al., Web-SEC is the image classifier trained on clean image-level labels and used to filter noisy web images, while Model-T is the pixel labeler trained on the filtered web images.

> T/F:
>
> Why:
>
> Corrected (False일 때):

### 6. [MCQ] `[L09]`
What is SmoothGrad for?  
(A) To replace the image classifier with a segmentation network  
(B) To reduce the "salt-and-pepper" noise of gradient saliency by averaging the gradients computed on several noisy copies of the image  
(C) To remove the FC layer and add global average pooling  
(D) To erase the most discriminative region and retrain the classifier

> Answer:
>
> Why:

### 7. [MCQ] `[L09]`
In CD-CLIP's super-class boundary exploration (SBE), what is DINO for?  
(A) It provides the class names for the CLIP text encoder  
(B) Its features are averaged inside the CLIP super-class CAM to form a prototype, and comparing the prototype with every DINO patch recovers a complete, structurally coherent super-class region  
(C) It computes the JS divergence between patch class distributions  
(D) It replaces the segmentation decoder at test time

> Answer:
>
> Why:

### 8. [MCQ] `[L09]`
How does WISH turn a class tag into instance-level supervision?  
(A) An annotator clicks on every instance of the tagged class  
(B) Local maxima of the class activation map become positive point prompts, and SAM generates an instance mask from each point  
(C) A box around the whole image is given to SAM as a box prompt  
(D) Web images retrieved with the tag provide copied masks

> Answer:
>
> Why:

### 9. [Code] `[L09·PC]`
Fill in the blank, and explain what this code does in 2–3 sentences.

```python
feat = backbone(img)                     # F: [K, h, w], last conv feature maps
logits = fc(feat.mean(dim=(1, 2)))       # GAP -> FC -> class scores [C]
loss = F.multilabel_soft_margin_loss(logits[None], labels[None])  # image labels only
loss.backward()                          # train backbone + fc as a classifier
cam = torch.einsum('k,khw->hw', fc.______[c], feat)  # M_c = sum_k w_{k,c} F_k
cam = F.relu(cam) / F.relu(cam).max()    # [h, w], scaled to 0..1
cam = F.interpolate(cam[None, None], size=img.shape[-2:], mode='bilinear')[0, 0]  # resize
fg = cam > tau                           # B_c: confident foreground
```

> Blank:
>
> Process:

### 10. [Code] `[L09·PC]`
Fill in the blank, and explain what this code does in 2–3 sentences.

```python
label = box_lbl.clone()                  # round 0: filled boxes, [B, H, W], 0 = background
for r in range(R):
    net = train_segnet(imgs, label)      # learn from the noisy masks
    with torch.no_grad():
        pred = net(imgs).argmax(dim=1)   # [B, H, W] predicted classes
    pred[box_lbl == 0] = 0               # outside every box -> background
    label = ______                       # next round's training label
```

> Blank:
>
> Process:

---

<details>
<summary><b>정답 · 모범 답안</b> (다 쓴 뒤에 열어)</summary>

**1.** `[L09]` **Answer**: False
- **Why (EN)**: CAM is the white-box approach: the network is redesigned (the FC layer is removed, and GAP and a 1×1 convolution are added) so that it keeps spatial information. The class maps can therefore be read out directly from the feature maps and class weights, without a separate gradient calculation.
- **Corrected (EN)**: Unlike gradient-based saliency, CAM obtains its activation maps directly, without separate gradient calculations, because the network itself is designed to preserve spatial information.
- **해설**: "since the network itself is designed to preserve spatial information, activation maps can be obtained directly without the need for separate gradient calculations." (p.25) gradient를 쓰는 건 gray box야: $\partial S_c / \partial I$ (p.21).

← pool: `09_Weakly_Supervised_Learning_II` Q3

**2.** `[L09]` **Answer**: False
- **Why (EN)**: The direction is reversed. A high CAM score means the classifier is confident that the class is there, so high-score regions become foreground and low-score regions become background. These confident regions are called seeds.
- **Corrected (EN)**: When class maps are thresholded, high value scores give foreground areas and low value scores give background areas, and these confident regions are commonly named seeds.
- **해설**: "High value scores → Foreground areas", "Low value scores → Background areas", "(This confident regions are commonly named "seeds")" (p.27, 원문 문법 그대로). 그림 M3: resize한 $M'_k$를 $> \tau_{loc}$로 잘라 $B_k$를 만들어.

← pool: `09_Weakly_Supervised_Learning_II` Q5

**3.** `[L09]` **Answer**: False
- **Why (EN)**: SEAM enforces equivariance, not invariance. If the input is rescaled or shifted, the activation map should be rescaled or shifted in the same way, so the CAMs are compared only after compensating for the transformation.
- **Corrected (EN)**: SEAM assumes that activation maps should be equivariant to the input, so it predicts CAMs for the original and transformed images and enforces consistency after compensating for the transformation.
- **해설**: "Activation maps should be equivariant to the input" (p.35). "Apply geometric transformations (e.g., affine transform)", "Predict CAMs for both original and transformed images", "Enforce consistency after compensating for the transformation" (p.36). 이 일을 하는 게 "Equivariant Regularization (ER) Loss: Encourage transformation-consistent CAMs"야. ← slide엔 ECR이 "Cross-view Consistency (ECR) Loss"로 적혀 있는데, 논문 이름은 Equivariant Cross Regularization이야 (paper).

← pool: `09_Weakly_Supervised_Learning_II` Q8

**4.** `[L09]` **Answer**: True
- **Why (EN)**: The region to erase is chosen by the current classifier itself, so exactly the part it relies on is removed. After erasing, the classifier must find another part of the object, and the regions found over the steps add up to the mined object region.
- **해설**: "Let us play adversarial peek-a-boo!", "(adversarial instead of random erasures)" (p.39). 그림: heatmap $H_t$ → threshold → $F_t$ → $I_t \ominus F_t = I_{t+1}$ → 다시 Input Image, 아래 줄 "The Mined Object Region". p.40 caption: "The second to fourth rows show the produced heatmaps, where the discriminative regions are highlighted. The images with erased regions are shown in the last row in gray." (Wei et al. 2017) HaS는 random hiding이라는 게 차이야.

← pool: `09_Weakly_Supervised_Learning_II` Q10

**5.** `[L09]` **Answer**: False
- **Why (EN)**: The roles are swapped. Model-T is the image classifier: it filters the web images and produces CAMs on the target images. Web-SEC is the pixel labeler: it is trained on the filtered web images and generates pixel labels for the target images.
- **Corrected (EN)**: Model-T is the image classifier trained on clean image-level labels and used to filter noisy web images, while Web-SEC is the pixel labeler trained on the filtered web images.
- **해설**: Pipeline (p.45–50): "1. Train an image classifier (Model-T) on clean image-level labels. 2. Use Model-T to filter noisy web images. 3. Train Web-SEC (pixel labeler) using the filtered web images. 4. Apply Web-SEC to generate pixel labels for the target images. 5. Use Model-T to produce class activation maps (CAMs) on the target images. 6. Fuse the two label sources and train the final segmentation model."

← pool: `09_Weakly_Supervised_Learning_II` Q12

**6.** `[L09]` **Answer**: (B)
- **Why (EN)**: A single gradient measures sensitivity pixel by pixel, so the saliency map is extremely noisy. Adding small random noise, computing the gradient for each copy, and averaging removes much of this noise.
- **해설**: "The problem is that it is extremely noisy. This is because the gradient measures sensitivity at the level of individual pixels, resulting in a "salt-and-pepper" appearance." (p.21) SmoothGrad: "Step 2) Add small random noise to the images, Step 3) Calculate the gradient for each, Step 4) Average", "calculating the gradient multiple times and averaging it removes noise and yields a more meaningful saliency map." (p.22) (C)는 CAM(white box), (D)는 adversarial erasing이야.

← pool: `09_Weakly_Supervised_Learning_II` Q19

**7.** `[L09]` **Answer**: (B)
- **Why (EN)**: CLIP tells which super-class a region belongs to, and DINO features carry object structure. Averaging DINO features under the CLIP super-class CAM gives a prototype, and its similarity to every patch outlines the region for boundary supervision.
- **해설**: "1. Construct a super-class prototype: Use the CLIP super-class CAM as a semantic seed and average the corresponding DINO features." $f^n_s = \mathrm{CAP}(M^n_s \odot F_d)$. "2. Generate a prototype CAM: Compare the prototype with every DINO patch to recover a complete and structurally coherent super-class region." $M^n_p = \mathrm{ReLU}(\mathrm{CosSim}(f^n_s, F_d))$ (p.70). "CLIP gives the prototype a semantic identity, and DINO expands it into a coherent region" (p.71). (C)는 CDA 쪽이야 (p.69).

← pool: `09_Weakly_Supervised_Learning_II` Q25

**8.** `[L09]` **Answer**: (B)
- **Why (EN)**: A tag says what is in the image but not where. The CAM gives rough class-specific locations, its peaks become point prompts, and SAM turns each point into an instance mask; overlapping masks are grouped into individual instances.
- **해설**: "1. Generate Class Activation Maps: CAM provides rough class-specific localization. Local maxima are extracted from each class activation map." "2. Convert CAM Peaks into Point Prompts: Each selected CAM peak becomes a positive point prompt. SAM generates an instance mask from each point. Overlapping masks are grouped to identify individual instances." "CAM supplies missing spatial cues, and SAM turns those cues into instance-level masks." (p.91)

← pool: `09_Weakly_Supervised_Learning_II` Q27

**9.** `[L09·PC]` CAM — GAP classifier + class activation map. **Answer**: `weight` (`fc.weight[c]` = class c의 FC weight 벡터 $w_{\cdot,c}$, shape `[K]`)

- **Process (EN)**: This code trains a classifier with global average pooling on image-level labels and then reads out the class activation map of class c. The feature maps are averaged by GAP and scored by the FC layer, and a multi-label loss against the image labels, the only target, updates the backbone and the FC layer; afterwards the FC weights of class c re-weight the K feature maps, and the map is resized and thresholded into a foreground mask. The blank is `weight`, because the CAM is the weighted sum of the last feature maps with the FC weights of class c, so no extra gradient computation is needed.

← `exam/pseudocode_final.md` WS-9

**10.** `[L09·PC]` Boxly — recursive training from boxes. **Answer**: `pred` (같은 뜻: `pred.clone()`, `pred.detach()`)

- **Process (EN)**: This code implements recursive training for box-supervised segmentation. The first labels are the boxes filled with their class, a noisy mask; in every round a segmentation network is trained with a per-pixel loss on the current labels, its predictions become the next labels, and pixels outside every box are reset to background. The blank is `pred`, the network's own prediction, which replaces the noisy label in the next round because convnets are robust to noisy supervision.

← `exam/pseudocode_final.md` WS-12

</details>
