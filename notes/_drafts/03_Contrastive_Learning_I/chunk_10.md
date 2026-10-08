<!-- DRAFT | lecture: 03_Contrastive_Learning_I | chunk: 10 | pages 73–74 -->
<!-- VERIFIED | fixes: 1 | verifier: sonnet -->

## Three Views of Better Contrastive Learning (p.73)

표 (3열: Paper / Main question / Solution):

| Paper | Main question | Solution |
|---|---|---|
| GenView | Are positive views good enough? | Generate and weight better positive views |
| MCL | Are useful features suppressed? | Learn different features across stages |
| X-CLR | Are sample relations binary? | Learn from a soft similarity graph |

하단 강조 문장: **Contrastive learning depends on the views, the features, and the relationships used for training.**

그림: 없음. 파란 테두리의 3행 표 + 아래 굵은 한 줄 요약.

**설명**
- 이 슬라이드는 이번 강의에서 다룬 최신 논문 3편을 "contrastive learning이 무엇에 의존하는가"라는 세 축(**views / features / relationships**)으로 묶어 정리한 요약 슬라이드임.
- **GenView (views 축)**: 기존 contrastive learning은 augmentation으로 만든 positive view가 충분히 좋다고 가정함. 그러나 augmentation이 너무 약하면 정보가 거의 같아 쉬운 task가 되고, 너무 강하면 semantic이 깨진 noisy positive가 됨. 해법은 diffusion 등 생성 모델로 더 나은 positive view를 **generate**하고, 품질에 따라 **weight**를 줘서 학습에 반영하는 것임.
- **MCL (features 축)**: contrastive loss를 최적화하면 shortcut/쉬운 feature가 학습을 지배하고 다른 유용한 feature는 **suppressed**(feature suppression)될 수 있음. 해법은 학습 **stage**를 나눠 단계마다 다른 feature를 학습하게 하는 것임 (multistage contrastive learning).
- **X-CLR (relationships 축)**: 기존 InfoNCE는 positive(같은 이미지의 두 view) vs negative(나머지 전부)의 **binary** 관계만 사용함. 실제로는 같은 클래스·유사 caption 샘플끼리 "얼마나 비슷한지"가 연속적임. 해법은 caption 등으로 **soft similarity graph**를 만들어 그 soft target으로 학습하는 것임 (바로 앞 슬라이드 p.72 Limitations: graph 구성에 caption/label/metadata 필요, 노이즈 caption 문제, 메모리 비용 등이 이 방법의 한계로 언급됨).
- 정리하면, contrastive learning 개선은 (1) 어떤 pair를 positive로 만드느냐, (2) 어떤 feature를 학습하느냐, (3) 샘플 간 관계를 어떻게 정의하느냐의 세 방향으로 볼 수 있음.

> 보충: 정식 논문명/연도는 슬라이드에 없음. GenView는 "GenView: Enhancing View Quality with Pretrained Generative Model for Self-Supervised Learning" (Li et al., ECCV 2024), MCL은 "Multistage Contrastive Learning"(feature suppression 완화 계열), X-CLR은 "X-Sample Contrastive Loss: Improving Contrastive Learning with Sample Similarity Graphs" (Sobal et al., 2024; X-CLR는 이 논문의 손실/방법 이름)로 기억하는 것이며, 정확한 서지는 앞 chunk 슬라이드 기준으로 확인 필요.

## Next Week.. (p.74)

- Continue on Contrastive Learning ..

그림 없음. 제목 + bullet 한 줄만 있는 마지막 슬라이드.

**설명**
- 다음 주 강의도 **contrastive learning**을 이어서 다룬다는 예고임. 이번 주(Part I)는 기본 개념과 최신 개선 논문들을 다뤘고, 다음 주는 그 연속(Part II)임. 내용상 시험 포인트는 없음.

## FIGURES

- p.73: 세 논문(GenView, MCL, X-CLR)의 질문-해법 비교 표 — 중요도 상 (한 장 요약으로 정식 노트 말미에 표로 재구성할 가치 있음)

## EXAM POINTS

1. **Contrastive learning이 의존하는 세 요소**: views, features, relationships — 각 요소에 대응하는 논문(GenView / MCL / X-CLR) 매칭.
2. **GenView**의 문제의식(positive view의 품질이 충분한가?)과 해법(generate + weight better positive views) 서술.
3. **MCL**의 문제의식(feature suppression: 유용한 feature가 억제됨)과 해법(stage별로 서로 다른 feature 학습).
4. **X-CLR**: 기존 binary(positive/negative) 관계 대비 **soft similarity graph**를 쓰는 이유와 장점.
5. Why 질문: 왜 단순 augmentation positive만으로는 부족한가? (너무 약하면 trivial, 너무 강하면 semantic 손실)
6. X-CLR의 한계(p.72): caption/label 필요, 노이즈 caption(CC), 전체 graph 메모리 비용 → batch-level 계산 또는 pre-computation, text encoder bias 전파.

## VERIFIER REPORT

- 보충 인용의 X-CLR 논문 제목 오기 수정: "Cross-sample Relationships in Contrastive Learning" -> "X-Sample Contrastive Loss: Improving Contrastive Learning with Sample Similarity Graphs" (Sobal et al.)
- 표/문장/p.74 내용은 원본과 일치, 그 외 수정 없음
