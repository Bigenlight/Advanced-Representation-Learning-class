"""검수 끝난 pool과 code card에서 단원 미니 모의고사를 만든다 (문장을 복사하지 않고 스크립트로 가져옴).

  python3 -I exam/_tools/make_mock_final.py

새 강의가 나오면 맨 위 상수만 바꿔. 구성은 실제 비율의 절반: T/F 5 + MCQ 3 + code 2.
"""
import os, pathlib, re
os.chdir(pathlib.Path(__file__).resolve().parents[2])
NUM, TITLE = '08', 'Weakly-Supervised Learning I'
STEM = f'{NUM}_Weakly_Supervised_Learning_I'
TF = ['Q3', 'Q6', 'Q10', 'Q12', 'Q14']
MC = ['Q19', 'Q24', 'Q26']
PC = ['WS-1', 'WS-3']
DATE = '2026-11-05'
pool=open(f'exam/_parts/{STEM}.md',encoding='utf8').read()
cards=open('exam/pseudocode_final.md',encoding='utf8').read()
c=pool[pool.index('## C. Question pool'):pool.index('## D. Fill')]
Q={}
for b in re.split(r'\n(?=### Q)',c)[1:]:
    Q[re.match(r'### (Q\d+)',b).group(1)]=b.rstrip()
def parts(b):
    lines=b.split('\n')[1:]
    i=[n for n,l in enumerate(lines) if l.startswith('**Answer')][0]
    return '\n'.join(lines[:i]).strip(), '\n'.join(lines[i:]).strip()
def card(n):
    s=cards[cards.index('### %s.'%n):]
    s=s[:s.index('\n### ',5)] if '\n### ' in s[5:] else s
    title=s.split('\n')[0].replace('### ','')
    code=re.search(r'```python\n.*?```',s,re.S).group(0)
    ans=re.search(r'\*\*Answer\*\*: *(.*)',s).group(1)
    proc=re.search(r'\*\*Process \(EN\)\*\*: *(.*)',s).group(1)
    return title,code,ans,proc
o=[f'# Mock Quiz — Final, L{NUM} ({TITLE})','',
'> 단원 미니 모의고사. 10문제 = **T/F 5 + MCQ 3 + code 2** (중간 실제 비율 10 : 5 : 5의 절반).',
'> **답은 영어로 직접 써.** T/F는 고르고 이유까지, False면 고친 문장까지. MCQ도 왜 그 답인지 한 문장. code는 빈칸 하나 + 과정 설명 2–3문장 (입력 → target은 어디서 오나 → loss → 누가 update되나).',
'> 헷갈리면 `False?`처럼 `?`를 붙여. 맞아도 `?` 붙은 문제는 다시 봐. 답을 다 적기 전엔 맨 아래를 열지 마.',
f'> 출처: `exam/_parts/{STEM}.md` C 섹션, `exam/pseudocode_final.md`. 모두 작성자와 다른 Opus가 독립 검수함 ({DATE}). 슬라이드 페이지 번호는 `lectures/{STEM}.pdf` 기준.','']
n=0; key=[]
for q in TF:
    n+=1; st,an=parts(Q[q])
    o+=[f'### {n}. [T/F] `[L{NUM}]`',st,'','> T/F:','>','> Why:','>','> Corrected (False일 때):','']
    key.append(f'**{n}.** `[L{NUM}]` {an}\n\n← pool: `{STEM}` {q}\n')
for q in MC:
    n+=1; st,an=parts(Q[q])
    st=st.replace('\n(','  \n(')
    o+=[f'### {n}. [MCQ] `[L{NUM}]`',st,'','> Answer:','>','> Why:','']
    key.append(f'**{n}.** `[L{NUM}]` {an}\n\n← pool: `{STEM}` {q}\n')
for w in PC:
    n+=1; title,code,ans,proc=card(w)
    meth=title.split('. ',1)[1]
    o+=[f'### {n}. [Code] `[L{NUM}·PC]`','Fill in the blank, and explain what this code does in 2–3 sentences.','',code,'','> Blank:','>','> Process:','']
    key.append(f'**{n}.** `[L{NUM}·PC]` {meth}. **Answer**: {ans}\n\n- **Process (EN)**: {proc}\n\n← `exam/pseudocode_final.md` {w}\n')
o+=['---','','<details>','<summary><b>정답 · 모범 답안</b> (다 쓴 뒤에 열어)</summary>','']+key+['</details>','']
open(f'exam/mock_quiz_final_L{NUM}.md','w',encoding='utf8').write('\n'.join(o))
print('questions',n)
