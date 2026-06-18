# LLM Study Day 02 — LangChain 기초

OpenAI + LangChain을 사용한 LLM 학습 예제 모음입니다.

## 환경 설정

### 1. 저장소 클론

```bash
git clone https://github.com/woodyjeon/llm-study-02.git
cd llm-study-02
```

### 2. 가상환경 생성 및 패키지 설치

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. API 키 설정

`.env.example`을 복사해 `.env` 파일을 만듭니다.

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

`.env` 파일에 OpenAI API 키를 입력합니다.

```
OPENAI_API_KEY=your-api-key-here
```

> ⚠️ `.env` 파일은 Git에 올라가지 않습니다. API 키를 절대 커밋하지 마세요.

---

## 학습 순서

난이도 순으로 아래 파일을 차례대로 실행해 보세요.

| 순서 | 파일 | 학습 내용 |
|------|------|-----------|
| 1 | `langchain_role.py` | `SystemMessage` / `HumanMessage`로 역할 기반 대화 |
| 2 | `langchain_repeat.py` | `while` 루프로 반복 대화 (간단한 챗봇) |
| 3 | `langchain_prompt_template.py` | `PromptTemplate` + `format()` 수동 방식 |
| 4 | `langchain_chat_prompt_template.py` | `ChatPromptTemplate` + system/human 메시지 |
| 5 | `langchain_chain.py` | LCEL Chain (`prompt \| llm \| parser`) |
| 6 | `langchain_jsonparser.py` | `JsonOutputParser`로 JSON → dict 파싱 |

```bash
python langchain_role.py
python langchain_repeat.py
# ...
```

---

## 핵심 개념 정리

### 1. f-string vs 템플릿 변수

```python
# f-string: 코드 실행 시점에 Python 변수를 즉시 치환
name = "철수"
text = f"이름: {name}"          # → "이름: 철수"

# LangChain 템플릿: invoke() 호출 시 치환 (앞에 f 없음!)
prompt = "관심 분야: {interest}"  # f 없음!
chain.invoke({"interest": "AI"})  # → "관심 분야: AI"
```

### 2. format() vs invoke()

| 방식 | 사용 시점 | 예시 |
|------|-----------|------|
| `prompt.format()` | chain **없이** 수동 실행 | `llm.invoke(prompt.format(topic="AI"))` |
| `chain.invoke()` | chain **사용** | `chain.invoke({"topic": "AI"})` |

> ❌ `format()`과 `invoke()`를 섞으면 `{topic}`에 프롬프트 전체가 들어가는 버그 발생

### 3. LCEL Chain (`|` 파이프)

```python
chain = prompt | llm | parser
# 1) prompt  → 템플릿 변수 치환 → 프롬프트 문자열
# 2) llm     → AI 응답 생성 (AIMessage)
# 3) parser  → AIMessage → str 또는 dict 변환
```

### 4. Output Parser 종류

| Parser | 반환 타입 | 용도 |
|--------|-----------|------|
| 없음 | `AIMessage` | `response.content`로 접근 |
| `StrOutputParser` | `str` | 텍스트 응답 |
| `JsonOutputParser` | `dict` | JSON 구조화 응답 |

### 5. PromptTemplate 중괄호 이스케이프

JSON 예시를 프롬프트에 넣을 때 `{` `}`는 LangChain 변수로 인식되므로 **이중 중괄호**로 이스케이프합니다.

```python
prompt = PromptTemplate.from_template("""
{topic}에 대해 JSON으로 답해줘.
{{
    "title": "제목",
    "summary": "요약"
}}
""")
```

---

## 파일별 설명

| 파일 | 설명 |
|------|------|
| `langchain_role.py` | Message 객체로 역할 + 질문 구성 |
| `langchain_repeat.py` | exit 입력까지 반복 질문 |
| `langchain_prompt_template.py` | 단일 문자열 템플릿, `format()` 사용 |
| `langchain_chat_prompt_template.py` | system/human 역할 분리, 사용자 입력 |
| `langchain_chain.py` | LCEL + `StrOutputParser` + for문 반복 |
| `langchain_jsonparser.py` | LCEL + `JsonOutputParser` + JSON 출력 |
| `.env.example` | API 키 설정 예시 (실제 키 X) |
| `requirements.txt` | 필요 패키지 목록 |

---

## Git에 올리지 않는 파일

`.gitignore`에 의해 아래 파일/폴더는 제외됩니다.

- `.env` — API 키 등 비밀정보
- `.venv/` — 가상환경
- `__pycache__/` — Python 캐시
