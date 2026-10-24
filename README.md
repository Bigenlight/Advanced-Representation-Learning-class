# 고급표현학습 (Advanced Representation Learning) — 2026-2학기

한양대 인공지능학과 대학원 **고급표현학습** 강의 정리 · 공부 · 시험 대비 리포.

## 구조

| 폴더 | 내용 |
|---|---|
| `lectures/` | 강의 원본 PDF (외부 공유 금지) |
| `notes/` | 강의별 정식 노트 `NN_topic.md` |
| `notes/_drafts/` | 페이지 청크별 초안 (멀티에이전트 산출물, 검증 완료) |
| `assets/lectureNN/` | 노트용 슬라이드 그림 |
| `exam/` | 시험 대비 — **`flow.html` (7강 한 바퀴 학습 페이지: 처음 배우는 사람 눈높이로 용어·수식을 풀어 씀 + 슬라이드 그림 + 개념 그래프·인터랙티브, 브라우저로 열기 / 소스는 `_flow/`, 본문 원본은 `_flow/base.html`)**, `core_ideas.md` (01–07 core idea 치트시트), **`pseudocode.md` (pseudo code 9개 총정리)**, `mock_quiz_1~3.md` (T/F·MCQ 20문제), **`mock_quiz_4~5.md` (빈칸·pseudo code 포함 새 형식)**, `_parts/` (강의별 pool: T/F·MCQ 139 + 빈칸 84, `pc_1~4` pseudo code pool 59 — 전부 Opus 독립 검수) |

## 진도 표

| # | 강의 | PDF | 초안 | 정식 노트 | 복습 퀴즈 | 시험 정리 |
|---|---|---|---|---|---|---|
| 01 | Introduction (W1) | 24p | ☑ | ☑ | ☐ | ☑ |
| 02 | Self-Supervised Learning | 80p | ☑ | ☑ | ☐ | ☑ |
| 03 | Contrastive Learning I | 74p | ☑ | ☑ | ☐ | ☑ |
| 04 | Contrastive Learning II | 72p | ☑ | ☑ | ☐ | ☑ |
| 05 | Contrastive Learning III | 76p | ☑ | ☑ | ☐ | ☑ |
| 06 | Semi-Supervised Learning I | 76p | ☑ | ☑ | ☐ | ☑ |
| 07 | Semi-Supervised Learning II | 106p | ☑ | ☑ | ☐ | ☑ |

## 평가 구조

- **영어 수업** (시험도 영어 전제)

- **중간고사 퀴즈: 2026-10-29 (목)** — 총 **20문제**, 쉽게 출제. 교수님: *"please study basic concept & core idea of the papers"* → 논문별 핵심 아이디어 위주
- 시험 있음 (상세 TBD — 강의계획서 확인 후 기입)
- **퀴즈**: 영어 출제, 주로 **T/F + 객관식(MCQ)**, 교수님 말로는 **어렵지 않을 것** (2026-10-15 전달받음)
- **추가 공지 (2026-10-22, 시험 전 마지막 수업)**: **빈칸 채우기(fill-in-the-blank)**도 출제, **pseudo code도 공부할 것**. 완전 서술형은 없을 듯
  - 슬라이드 pseudo code 9개: X-CLR (L03 p.67) · SimCLR (L04 p.25) · DINO (L04 p.40) · MoCo (L05 p.12) · CLIP (L05 p.35) · Π-model (L06 p.18) · Temporal Ensembling (L06 p.20) · Label Propagation (L07 p.11) · DivideMix (L07 p.54)
