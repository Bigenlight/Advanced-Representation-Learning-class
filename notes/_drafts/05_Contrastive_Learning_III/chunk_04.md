<!-- DRAFT | lecture: 05_Contrastive_Learning_III | chunk: 04 | pages 25–32 -->
<!-- VERIFIED | fixes: 3 | verifier: sonnet -->

## DeepCluster – 결과와 기여 (p.25)

**슬라이드 내용**
- 제목: DeepCluster / • DeepCluster (2018)
- 출처: Caron et al., 2018, Deep Clustering for Unsupervised Learning of Visual Features
- **Table 1** (왼쪽): AlexNet의 conv1~conv5 activation을 feature로 쓴 linear classification 정확도 (central crop). 열은 ImageNet(conv1~conv5), Places(conv1~conv5). 행: Places labels, ImageNet labels, Random, Pathak et al.[38], Doersch et al.[25], Zhang et al.[28], Donahue et al.[20], Noroozi and Favaro[26], Noroozi et al.[45], Zhang et al.[43], DeepCluster. 캡션: "Linear classification on ImageNet and Places using activations from the convolutional layers of an AlexNet as features. We report classification accuracy on the central crop. Numbers for other methods are from Zhang et al. [43]."
  - Places labels: Places 열 22.1 / 35.1 / 40.2 / 43.3 / 44.6
  - ImageNet labels(supervised): 19.3 / 36.3 / 44.2 / 48.3 / 50.5 (ImageNet), 22.7 / 34.8 / 38.4 / 39.4 / 38.7 (Places)
  - Random: 11.6 / 17.1 / 16.9 / 16.3 / 14.1 (ImageNet), 15.7 / 20.3 / 19.8 / 19.1 / 17.5 (Places)
  - Pathak 14.1/20.7/21.0/19.8/15.5 | 18.2/23.2/23.4/21.9/18.4
  - Doersch 16.2/23.3/30.2/31.7/29.6 | 19.7/26.7/31.9/32.7/30.9
  - Zhang[28] 12.5/24.5/30.4/31.5/30.3 | 16.0/25.7/29.6/30.3/29.7
  - Donahue 17.7/24.5/31.0/29.9/28.0 | 21.4/26.2/27.1/26.1/24.0
  - Noroozi and Favaro 18.2(굵게)/28.8/34.0/33.9/27.1 | 23.0/32.1/35.5/34.8/31.3
  - Noroozi et al.[45] 18.0/30.6(굵게)/34.3/32.5/25.7 | 23.3(굵게)/33.9(굵게)/36.3/34.7/29.6
  - Zhang[43] 17.7/29.3/35.4/35.2/32.8 | 21.3/30.7/34.0/34.1/32.5
  - **DeepCluster** 12.9/29.2/**38.2**/**39.8**/**36.1** | 18.6/30.8/**37.0**/**37.5**/**33.1**
- **Table 2** (오른쪽): Pascal VOC에서 classification / detection / segmentation 비교 (열: Classification fc6-8, ALL | Detection fc6-8, ALL | Segmentation fc6-8, ALL). 캡션: "Comparison of the proposed approach to state-of-the-art unsupervised feature learning on classification, detection and segmentation on Pascal VOC. * indicates the use of the data-dependent initialization of Krähenbühl et al.[68]. Numbers for other methods produced by us are marked with a †."
  - ImageNet labels 78.9/79.9 | –/56.8 | –/48.0
  - Random-rgb 33.2/57.0 | 22.2/44.5 | 15.2/30.1
  - Random-sobel 29.0/61.9 | 18.9/47.9 | 13.0/32.0
  - Pathak[38] 34.6/56.5 | –/44.5 | –/29.7
  - Donahue[20]* 52.3/60.1 | –/46.9 | –/35.2
  - Pathak[27] –/61.0 | –/52.2 | –/–
  - Owens[44]* 52.3/61.3
  - Wang and Gupta[29]* 55.6/63.1 | 32.8†/47.2 | 26.0†/35.4†
  - Doersch[25]* 55.1/65.3 | –/51.1
  - Bojanowski and Joulin[19]* 56.7/65.3 | 33.7†/49.4 | 26.7†/37.1†
  - Zhang[28]* 61.5/65.9 | 43.4†/46.9 | 35.8†/35.6
  - Zhang[43]* 63.0/67.1 | –/46.7 | –/36.0
  - Noroozi and Favaro[26] –/67.6 | –/53.2 | –/37.6
  - Noroozi et al.[45] –/67.7 | –/51.4 | –/36.6
  - **DeepCluster 70.4/73.7 | 51.4/55.4 | 43.2/45.1** (전부 굵게)
- **Contributions**
  - Introduced clustering-based self-supervised learning.
  - Established pseudo-label learning as an alternative to contrastive learning.
  - Achieved state-of-the-art performance among unsupervised methods at the time.
  - Inspired subsequent clustering-based methods such as SwAV and DINO.

**설명**
- Table 1은 **frozen AlexNet의 각 conv layer 출력 위에 linear classifier만 학습**해서 layer별 feature 품질을 보는 표임. DeepCluster는 conv3~conv5(깊은 층)에서 기존 pretext-task 방법(Doersch의 context prediction, Noroozi의 jigsaw 등)보다 높고, conv5에서 ImageNet 36.1로 기존 최고(Zhang[43] 32.8)를 넘음. 반면 **conv1은 12.9로 낮음** → DeepCluster는 상위층 semantic 구조에 맞춰 학습되므로 저수준 filter는 덜 최적화됨(Random 11.6과 비슷한 수준).
- 지도학습(ImageNet labels, conv5 50.5)과는 여전히 격차가 있음. Random(conv5 14.1) 대비로는 큰 향상.
- Table 2는 **transfer 성능**: fc6-8만 학습 vs ALL(전체 fine-tune). DeepCluster가 classification 73.7, detection 55.4, segmentation 45.1로 당시 unsupervised 최고. ImageNet label supervised(79.9/56.8/48.0)와 비교하면 detection(-1.4)·segmentation(-2.9)은 근접하나 classification은 6.2%p 격차가 남음.
- Contributions 해석: pretext task(회전 예측, jigsaw 등) 대신 **cluster ID를 pseudo label로 쓰는 분류 문제**로 self-supervised learning을 정의했고, positive/negative pair 비교 없이도 동작함을 보였음. 이후 SwAV(online clustering), DINO(self-distillation)로 이어짐.
> 보충: DeepCluster는 k-means(예: k=10,000)로 clustering → cluster ID를 label로 CNN 학습을 epoch마다 반복. 빈 cluster 재할당과 cluster에 대한 uniform sampling으로 trivial solution(모든 샘플이 한 cluster로) 방지. Input에 Sobel filter를 써서 색 정보 shortcut을 줄임("Random-sobel" 행이 이를 반영). 표의 † 와 * 표기는 슬라이드 캡션 그대로임.

## SwAV – 개요 (p.26)

**슬라이드 내용**
- • SwAV (Swapping Assignments between multiple Views)
  - Learn visual representations by predicting cluster assignments.
  - Compute **online** cluster assignments for augmented views.
  - Predict the assignment of one view from another view.
  - **Eliminate the need for explicit pairwise comparisons or offline clustering.**
- • **Key Idea**: Learn by swapping cluster assignments between multiple views.
- 출처: Caron et al., NeurIPS 2020, Unsupervised Learning of Visual Features by Contrasting Cluster Assignments

**설명**
- SwAV는 **DeepCluster의 clustering 아이디어 + contrastive learning의 multi-view augmentation**을 결합한 방법임.
- DeepCluster는 **offline**(전체 데이터셋 feature를 모아 epoch마다 k-means) → 큰 데이터에서 비효율적. SwAV는 **mini-batch 안에서 online**으로 code(cluster assignment)를 계산함.
- Contrastive(SimCLR/MoCo)는 모든 pair(positive/negative) 비교가 필요하고 큰 batch/memory bank가 필요. SwAV는 feature를 **K개 prototype과만 비교**하므로 pairwise 비교가 없음.
- 핵심은 "한 view의 feature로 **다른 view의 code를 예측**"하는 swapped prediction임. 같은 이미지의 두 augmentation은 같은 cluster에 속해야 한다는 consistency 제약임.
> 보충: 논문 제목의 "Contrasting Cluster Assignments"는 instance가 아니라 cluster assignment 수준에서 contrast한다는 의미임.

## SwAV – 구조와 Swapped Loss (p.27)

**슬라이드 내용**
- • SwAV (Swapping Assignments between multiple Views)
  - Given features of two augmented images z, computes codes q, measuring swapped loss.
- **그림** (좌우 2개 패널):
  - 왼쪽 "Contrastive instance learning": 이미지 $X$에서 변환 $t\sim\mathcal{T}$ 두 번 샘플(화살표 4개로 표시) → $X_1$, $X_2$ → 각각 encoder $f_\theta$(회색 사다리꼴) → feature $Z_1$, $Z_2$ → "Comparison" 박스에서 두 feature를 직접 비교.
  - 오른쪽 "Swapping Assignments between Views (Ours)": 같은 구조로 $X \to X_1, X_2 \to f_\theta \to Z_1, Z_2$. 추가로 회색 박스 **Prototypes $C$**가 중간에 있고, $Z_1$, $Z_2$와 $C$ 양쪽에서 화살표가 각각 $Q_1$, $Q_2$ (**Codes**)로 들어감. $Q_1$, $Q_2$가 "Swapped Prediction" 박스로 연결됨.
- 수식:
  - $\mathcal{L}_{\text{SwAV}}(\mathbf{z}_t,\mathbf{z}_s)=\ell(\mathbf{z}_t,\mathbf{q}_s)+\ell(\mathbf{z}_s,\mathbf{q}_t)$
  - $\ell(\mathbf{z}_t,\mathbf{q}_s)=-\sum_k \mathbf{q}_s^{(k)}\log \mathbf{p}_t^{(k)}$, where $\mathbf{p}_t^{(k)}=\dfrac{\exp(\mathbf{z}_t^\top \mathbf{c}_k/\tau)}{\sum_{k'}\exp(\mathbf{z}_t^\top \mathbf{c}_{k'}/\tau)}$
  - 주석: "cross entropy between the predicted code and a set of K trainable prototype vectors C"

**설명**
- 기호: $\mathbf{z}_t,\mathbf{z}_s$ = 같은 이미지의 두 augmented view의 (normalize된) feature. $\mathbf{C}=[\mathbf{c}_1,\dots,\mathbf{c}_K]$ = **K개의 trainable prototype**(cluster centroid 역할). $\tau$ = temperature. $\mathbf{p}_t$ = $\mathbf{z}_t$와 prototype의 내적 → softmax한 **예측 분포**. $\mathbf{q}_s$ = view $s$의 **code**(target assignment).
- 동작: view $s$의 feature $\mathbf{z}_s$와 $C$로 code $\mathbf{q}_s$를 계산 → view $t$의 예측 $\mathbf{p}_t$가 이 $\mathbf{q}_s$를 맞추도록 cross-entropy. 반대 방향도 더해 **swap** (loss 2항).
- 왜 swap인가: 자기 view로 자기 code를 예측하면 trivial. 다른 view의 code를 맞춰야 **view-invariant**한 feature가 학습됨.
- 왼쪽(기존 contrastive)은 feature 간 직접 비교(instance-level), 오른쪽은 prototype을 매개로 비교(cluster-level)함.
> 보충: code $\mathbf{q}$는 batch 내에서 **Sinkhorn-Knopp**로 계산하며, 모든 prototype에 샘플이 균등 분배되도록 equipartition 제약을 걸어 모든 샘플이 한 cluster로 붕괴하는 collapse를 방지함(슬라이드에는 없음). $\mathbf{q}$에는 gradient를 흘리지 않음. 논문은 2개 full-resolution crop 외에 여러 개의 small crop을 쓰는 **multi-crop**도 제안(제목의 "multiple views"와 연관).

## SwAV – ImageNet linear classification 결과 (p.28)

**슬라이드 내용**
- • SwAV (Swapping Assignments between multiple Views)
- 왼쪽 표 (Method / Arch. / Param. / Top1), 전부 R50, 24M:
  - Supervised 76.5
  - Colorization[65] 39.6, Jigsaw[46] 45.7, NPID[58] 54.0, BigBiGAN[15] 56.6, LA[68] 58.8, NPID++[44] 59.0, MoCo[24] 60.6, SeLa[2] 61.5, PIRL[44] 63.6, CPC v2[28] 63.8, PCL[37] 65.9, SimCLR[10] 70.0, MoCov2[11] 71.1
  - **SwAV 75.3** (굵게)
- 오른쪽 **그래프**: x축 number of parameters (24M, 94M, 375M, 586M), y축 ImageNet top-1 accuracy (68~80). 범례: 초록 원 Supervised, 주황 삼각형 SwAV. 초록 점선(Supervised): 76.5(24M) → 78(94M) → 78.9(375M) → 79.1(586M). 주황 점선(SwAV): 75.3(24M) → 77.3(94M) → 77.9(375M) → 78.5(586M). 보라 ×: MoCov2(~71, 24M), SimCLR-x1(70), SimCLR-x2(~74.4, 94M), SimCLR-x4(~76.8, 375M), CPCv2(~71.5), CMC(~68.5), AMDIM(~68, 586M).
- 캡션: "Figure 2: Linear classification on ImageNet. Top-1 accuracy for linear models trained on frozen features from different self-supervised methods. (left) Performance with a standard ResNet-50. (right) Performance as we multiply the width of a ResNet-50 by a factor ×2, ×4, and ×5."

**설명**
- 평가는 backbone을 freeze하고 linear classifier만 학습하는 **linear evaluation protocol**. 같은 ResNet-50에서 SwAV 75.3%로 SimCLR(70.0), MoCov2(71.1)를 크게 앞서고 supervised(76.5)와 **1.2%p 차이**.
- 오른쪽 그래프: backbone을 넓힐수록(×2, ×4, ×5) 모두 향상되며, SwAV는 supervised와 격차가 작게 유지됨(24M 1.2%p, 94M 0.7, 375M 1.0, ×5 586M에서 78.5 vs 79.1로 0.6%p; 단조 감소는 아님). 같은 파라미터 수에서 SimCLR보다 일관되게 높음.
- 의미: pairwise 비교/큰 negative 없이도 contrastive 계열 SOTA를 넘음 → **clustering 기반 접근의 유효성** 입증.
> 보충: 이 결과는 multi-crop 사용 설정에서의 값임(논문 기준). 그래프의 보라색 점 수치는 이미지에서 읽은 근사값임. 슬라이드 그래프에서 x5 폭은 586M 점으로 표시됨(캡션은 ×2, ×4, ×5).

## SwAV – Transfer learning 결과 (p.29)

**슬라이드 내용**
- • SwAV (Swapping Assignments between multiple Views)
- 캡션: "Table 2: **Transfer learning on downstream tasks.** Comparison between features from ResNet-50 trained on ImageNet with SwAV or supervised learning. We consider two settings. (1) Linear classification on top of frozen features. We report top-1 accuracy on all datasets except VOC07 where we report mAP. (2) Object detection with finetuned features on VOC07+12 trainval using Faster R-CNN [50] and on COCO [38] using Mask R-CNN [26] or DETR [6]. We report the most standard detection metrics for these datasets: AP50 on VOC07+12 and AP on COCO."
- 표:

| | Linear: Places205 | Linear: VOC07 | Linear: iNat18 | Det: VOC07+12 (Faster R-CNN R50-C4) | Det: COCO (Mask R-CNN R50-FPN) | Det: COCO (DETR) |
|---|---|---|---|---|---|---|
| Supervised | 53.2 | 87.5 | 46.7 | 81.3 | 39.7 | 40.8 |
| SwAV | **56.7** | **88.9** | **48.6** | **82.6** | **41.6** | **42.1** |

**설명**
- ImageNet에서 SwAV(self-supervised) vs supervised로 pretrain한 ResNet-50을 **다른 데이터셋/태스크에 전이**한 결과. 6개 전부에서 **SwAV가 supervised를 능가**.
- Linear classification(frozen)은 Places205·VOC07·iNat18에서 +3.5/+1.4/+1.9. Detection(fine-tune)은 VOC07+12 AP50 +1.3, COCO AP +1.9(Mask R-CNN), +1.3(DETR).
- 의미: label 없이 학습한 표현이 supervised 표현보다 **전이 능력이 높을 수 있음**. 지도학습은 ImageNet 1000 class에 특화된 feature를 만들 수 있음.
> 보충: VOC07 linear는 mAP, 나머지 linear는 top-1, detection은 AP50(VOC)/AP(COCO)임(캡션 기준). 슬라이드 캡션이 "Table 2"이나 p.25의 DeepCluster Table 2와는 서로 다른 논문의 표임.

## SwAV – Contrastive Learning과의 비교 (p.30)

**슬라이드 내용**
- • SwAV (Swapping Assignments between multiple Views)
  - SwAV replaces feature matching with prototype assignment prediction, learning representations by predicting the cluster assignment of one view from another without explicit negative pairs.
- 표:

| Contrastive Learning | SwAV |
|---|---|
| Feature ↔ Feature | Feature ↔ Prototype |
| Positive/Negative pairs | Prototype assignments |
| Similarity matching | Swapped assignment prediction |
| Pairwise comparison | Online clustering |

**설명**
- 비교 대상이 **feature↔feature에서 feature↔prototype**으로 바뀜. 비교 횟수가 batch 크기의 제곱(pairwise)에서 $B\times K$로 줄어 memory/compute 효율적.
- Positive/negative 정의 대신 **assignment 분포**를 target으로 사용. Negative가 명시적으로 없으므로 큰 batch/queue 불필요. 대신 collapse 방지를 위해 **equipartition(Sinkhorn)** 이 negative가 하던 "표현이 한 점으로 모이지 않게 하는" collapse 방지 기능을 대체함(negative와 수학적으로 동일한 것은 아님).
- DeepCluster와 비교: offline k-means(epoch 단위, hard assignment label) → **online, soft code**, 두 view 간 swapped prediction.
> 보충: 슬라이드의 "without explicit negative pairs"는 negative를 안 쓴다는 뜻이지 collapse 방지 장치가 불필요하다는 뜻이 아님.

## [섹션 구분] Working with Supervised Datasets (p.31)

섹션 구분 슬라이드: "Working with Supervised Datasets" (진한 파랑 배경, 흰 글씨 제목만 있음). 이후 supervised dataset(이미지-텍스트 pair)을 활용하는 방법(CLIP)으로 넘어감.

## CLIP – 개요 (p.32)

**슬라이드 내용**
- • CLIP (Contrastive Language-Image Pre-training)
  - Jointly learns **image and text** representations.
  - **Contrastively** matches images with their corresponding captions.
  - **Aligns** visual and language embeddings in a **shared space**.
  - Enables zero-shot image recognition using natural language prompts.
- • **Key Idea**: Learn visual representations by aligning images and text.
- 출처: Radford et al., PMLR 2021, Learning Transferable Visual Models From Natural Language Supervision

**설명**
- CLIP은 **image encoder와 text encoder를 동시에** 학습해 둘의 embedding을 같은 공간에 정렬하는 모델. batch 안의 (image, caption) pair에서 **맞는 pair는 가깝게, 틀린 pair는 멀게** InfoNCE 계열 contrastive loss를 사용함.
- 앞선 방법들(SimCLR, SwAV)이 이미지 augmentation으로 positive를 만든 것과 달리, CLIP의 positive는 **자연어 caption(supervision)** 임 → 섹션명 "Working with Supervised Datasets"의 이유.
- **Zero-shot recognition**: class 이름을 "a photo of a {class}" 같은 prompt 문장으로 text encoder에 넣어 class embedding을 만들고, 이미지 embedding과 cosine similarity가 가장 높은 class를 선택. 별도 fine-tuning 없이 새 class에 적용 가능.
> 보충: 학습 데이터는 인터넷에서 수집한 4억 개의 (image, text) pair(WIT). Loss는 image→text, text→image 양방향 symmetric cross-entropy이며 temperature는 learnable.

## FIGURES

- p.25: DeepCluster Table 1 (AlexNet conv layer별 linear classification) / Table 2 (VOC transfer) — 중
- p.27: SwAV 구조 다이어그램 (Contrastive instance learning vs Swapping Assignments between Views) + swapped loss 수식 — 상
- p.28: SwAV ImageNet linear classification 표 + 파라미터 수 대비 정확도 그래프 — 중
- p.29: SwAV transfer learning 표 (supervised 대비) — 중
- p.30: Contrastive vs SwAV 비교 표 — 상

## EXAM POINTS

1. **DeepCluster 절차**: feature 추출 → k-means → cluster ID를 pseudo label → classification으로 network 업데이트 → 반복. Key idea는 clustering과 classification의 **alternation**.
2. DeepCluster의 한계 vs SwAV: **offline clustering**(epoch마다 전체 데이터 필요) → SwAV는 **online** (mini-batch 내 code 계산).
3. **SwAV loss 수식**: $\mathcal{L}=\ell(\mathbf{z}_t,\mathbf{q}_s)+\ell(\mathbf{z}_s,\mathbf{q}_t)$, $\ell=-\sum_k \mathbf{q}_s^{(k)}\log\mathbf{p}_t^{(k)}$, $\mathbf{p}$는 prototype과의 내적 softmax(temperature $\tau$). 각 기호(z, q, p, C, K, τ) 설명 가능해야 함.
4. **왜 swap인가?** 다른 view의 code를 예측해야 augmentation-invariant 표현을 학습(자기 view code 예측은 trivial).
5. **Contrastive vs SwAV 비교표**: feature↔feature vs feature↔prototype, positive/negative pair vs prototype assignment, pairwise vs online clustering. 명시적 negative 없이도 가능한 이유(Sinkhorn equipartition으로 collapse 방지).
6. SwAV 성능: ResNet-50 linear eval **75.3%** (SimCLR 70.0, MoCov2 71.1, supervised 76.5), transfer에서는 supervised를 모두 능가.
7. **CLIP의 핵심 아이디어**: image/text encoder를 contrastive하게 joint 학습해 shared embedding space 정렬 → **zero-shot** classification(prompt로 class embedding 생성).
8. DeepCluster의 기여: pseudo-label learning을 contrastive의 대안으로 확립, SwAV·DINO에 영향.

## VERIFIER REPORT

- p.25 설명: DeepCluster가 supervised에 "근접"이라 한 과장 수정 (classification 6.2%p 격차, detection/segmentation만 근접).
- p.28 설명: "SwAV와 supervised 격차가 줄어듦"을 실제 수치(1.2/0.7/1.0/0.6%p, 단조 아님)로 정정.
- p.30 설명: equipartition이 negative 역할을 "대신"한다는 표현을 collapse 방지 기능 대체로 완화.
- 표 수치(p.25, 28, 29), 수식(p.27), 그림 설명, 인용(Caron 2018/NeurIPS 2020, Radford PMLR 2021)은 원본과 일치 확인.
