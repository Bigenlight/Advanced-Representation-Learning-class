# CLAUDE.md — 고급표현학습 (Advanced Representation Learning)

한양대 인공지능학과 **고급표현학습** (2026-2학기) 공부 리포. 강의 PDF → md 정리, 공부, **시험 대비**를 여기서 함.

## 사용자

- 한양대 인공지능학과 석박통합과정 (Physical AI 연구실). 상세 프로필·노트 스타일은 `~/Desktop/For-Neural-Network-Improvement-Private-/CLAUDE.md` 참고 — **말투·스타일 규칙은 그 파일 기준을 그대로 따를 것** (한국어 반말, 기술 용어 영어, `**굵게**` 핵심, `← ` 부연, `>` 참고).
- **배경 (2026-10-22 기록)**: 주 연구 분야는 **Physical AI · RL · flow matching · simulation · VLA** (VLA 중에서도 주로 **끝단 = action head / policy 쪽**) → representation learning은 주 분야가 아님. 수업은 **논문 세미나를 연달아 듣는 형식**이고 영어 전달도 잘 안 돼서, "기본 용어는 다 알겠지" 하고 넘어가는 부분을 **첫 수업부터 못 따라감**. 수업 중엔 키워드·모델 이름 정도만 잡았고, 논문 마감 때문에 수업 중 다른 일을 해야 한 적도 있음. 그래도 이쪽에서 발전한 이론이 physical AI로 넘어오는 경우가 많아서 제대로 공부할 생각.
  → 설명할 때: **기본 용어부터 깔고** 시작할 것 (embedding, projection head, positive/negative, collapse, stop-gradient, teacher/student, pseudo label …). 가능하면 **RL·flow matching·VLA 쪽 아는 개념에 빗대서** 설명할 것 (예: EMA teacher ↔ target network soft update, stop-gradient ↔ semi-gradient TD target, confirmation bias ↔ bootstrapping error 누적, DivideMix 두 network ↔ twin Q). physical AI로 이어지는 연결고리(VLA vision encoder = DINOv2·SigLIP, CURL 등)는 동기 부여용으로 `>` 보충에 넣되, **슬라이드 밖 = 시험 범위 아님**을 꼭 표시할 것. 단 CURL(L05 p.20–21)·DINOv2(L04 p.55–67) 방법 자체는 슬라이드에 있어 **시험 범위**임 (범위 밖은 "VLA가 이걸 쓴다"는 연결 부분만).
- 원리까지 파고드는 타입 → 정리할 때 정의·수식 유도·직관 흐름을 생략하지 말 것. 수식은 LaTeX(`$...$`, `$$...$$`)로.
- **영어 수업** (슬라이드·강의·시험 모두 영어 전제). 노트 본문은 한국어로 쓰되 용어·정의는 슬라이드 원문 영어 표현을 살릴 것. `exam/` 자료(예상문제·모범답안)는 **영어로 답안 작성 연습**할 수 있게 영어 모범답안 병기.
- **시험 있는 과목** → 정리할 때 "시험에 나올 만한 것"(정의, loss 수식, 방법 간 비교, 왜 그렇게 설계했는지)을 항상 의식할 것.
- **퀴즈는 영어 T/F + 객관식 + 빈칸 채우기(fill-in-the-blank)** (빈칸은 2026-10-22 마지막 수업에서 추가 공지, 완전 서술형은 없을 듯) → 예상문제는 T/F·MCQ·빈칸 형태로 만들 것. 슬라이드 원문 문장을 살짝 비튼 함정 지문(예: student↔teacher, prediction EMA↔weight EMA, MSE↔KL 뒤바꾸기), 수치·방법 이름 매칭, "which is NOT …" 유형을 의식할 것. 정답엔 근거 슬라이드 페이지(`p.N`)를 붙일 것. 교수님 말로는 난이도가 높지 않다고 함 → 핵심 개념·정의·방법 간 차이 위주로 내되, 교수 기준 "쉬움"이니 표 수치까지 완전히 버리지는 말 것 (비중만 낮게).
- **중간고사 퀴즈: 2026-10-29(목), 총 20문제, 쉽게 출제.** 교수님 말씀: *"please study basic concept & core idea of the papers"* → 시험 대비 자료는 **논문/방법별 basic concept + core idea**(무슨 문제를 풀려고 했고, 핵심 아이디어가 뭐고, 이전 방법과 뭐가 다른지)를 최우선으로. 표 수치는 최소화.
- **pseudo code도 공부하라고 하심** (2026-10-22). 슬라이드에 Algorithm/코드 박스로 나온 건 9개: X-CLR (L03 p.67), SimCLR Alg.1 (L04 p.25), DINO (L04 p.40), MoCo Alg.1 (L05 p.12), CLIP (L05 p.35), Π-model Alg.1 (L06 p.18), Temporal Ensembling Alg.2 (L06 p.20), Label Propagation Alg.1 (L07 p.11), DivideMix Alg.1 (L07 p.54). 준-알고리즘으로 DeepCluster process (L05 p.24)와 BYOL 단계 (L04 p.32)도 있음. 단, 교수님이 pseudo code를 깊게 가르치진 않았음 → 구현 디테일보다 **논문 core idea가 드러나는 핵심 줄**(stop-gradient/`detach`, momentum·EMA update, loss 식, `labels`, centering, bias correction 등) 위주로 나올 것. 그래도 **킬러 파트**일 가능성 높음 → 핵심 줄 빈칸·"이 코드는 어느 방법?"·"이 줄의 역할은?" 유형을 의식할 것.

## 폴더 규칙

```
lectures/            원본 PDF — 수정 금지, 외부 공유 금지. 파일명 NN_topic.pdf
lectures/_pages/     PDF 페이지별 렌더(png, 100dpi) + 텍스트(txt). gitignore, 작업용
notes/               정식 노트. 파일명: NN_topic.md (PDF 번호와 일치)
notes/_drafts/NN_topic/chunk_XX.md   PDF 페이지 범위별 초안 (멀티에이전트 산출물)
assets/lectureNN/    노트에 들어가는 이미지
exam/                시험 대비 (예상문제, 요약 치트시트, 기출)
```

## 작업 규칙

- **페이지 번호는 PDF 페이지 기준**으로 통일 (`p.12` 식으로 표기).
- **정식 노트**: 슬라이드 그대로 옮기지 말고 개념 → 정의/수식 → 직관 → 예시/실험 → 비교 순으로 재구성. 슬라이드에 없는 보충 설명은 `>` 인용 블록으로 구분.
- 슬라이드 오타·의심 내용은 원문 유지 + `← ` 인라인 플래그.
- 이미지: 필요한 슬라이드 그림은 PyMuPDF(`fitz`)로 `assets/lectureNN/`에 렌더/crop, md에서 `<img src="..." width="600">`로 삽입 (`md-img-resize` 스킬).
- README의 **진도 표(☐)** 는 정리·작업 끝날 때마다 갱신.
- 커밋/푸시는 `git-pull-push` 스킬 사용. 리포는 private.
- 복습 퀴즈는 `recall-quiz` 스킬 (노트 md 기반).
