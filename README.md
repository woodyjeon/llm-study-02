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
| 7 | `langchain_memory.py` | `InMemoryChatMessageHistory`로 대화 기록 유지 |
| 8 | `langchain_memory2.py` | `MessagesPlaceholder` + Chain으로 맥락 대화 |
| 9 | `langchain_chatbot.py` | 여행 추천 1회 질의 (CLI) |
| 10 | `langchain_chatbot2.py` | 맥락 대화 CLI 챗봇 (`memory2` + `while` 루프) |
| 11 | `langchain_chatbot3.py` | Streamlit 웹 UI 챗봇 (`chatbot2` + `session_state`) |
| 12 | `langchain_chatbot4.py` | 재테크 입문 챗봇 (`pills` + wide 레이아웃 + `{question}`) |
| 13 | `langchain_rag.py` | `PyPDFLoader`로 PDF → Document 변환 (RAG 1단계) |
| 14 | `langchain_rag2.py` | `RecursiveCharacterTextSplitter`로 텍스트 분할 (RAG 2단계) |
| 15 | `langchain_chroma_rag.py` | `OpenAIEmbeddings` + Chroma 벡터 DB 저장 (RAG 3단계) |
| 16 | `langchain_retriever_rag.py` | `as_retriever()`로 유사 문서 검색 (RAG 4단계) |

```bash
python langchain_role.py
python langchain_repeat.py
# ...
python langchain_rag.py                  # PDF 문서 로딩
python langchain_rag2.py                 # 텍스트 청크 분할
python langchain_chroma_rag.py           # 벡터 DB 생성 (먼저 실행)
python langchain_retriever_rag.py        # 저장된 DB에서 유사 문서 검색
streamlit run langchain_chatbot3.py   # 여행 추천 웹 UI
streamlit run langchain_chatbot4.py   # 재테크 입문 웹 UI
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

### 6. Chat Memory (대화 기록)

| 방식 | 파일 | 설명 |
|------|------|------|
| `history.messages` 직접 전달 | `langchain_memory.py` | while 루프 + `llm.invoke(history.messages)` |
| `MessagesPlaceholder` | `langchain_memory2.py` | Chain + 프롬프트에 history 자리 지정 |
| `InMemoryChatMessageHistory` | `langchain_chatbot2.py` | CLI 맥락 대화 (`add_user_message` / `add_ai_message`) |
| `st.session_state` + 메시지 변환 | `langchain_chatbot3.py` | Streamlit UI + `HumanMessage` / `AIMessage` 변환 |
| `st.pills` + `{question}` | `langchain_chatbot4.py` | 주제 빠른 선택 + 재테크 전문가 system 프롬프트 |

```python
# memory.py — 대화 기록을 LLM에 직접 전달
history.add_user_message(user_input)
response = llm.invoke(history.messages)
history.add_ai_message(response.content)

# memory2.py — MessagesPlaceholder로 프롬프트에 history 삽입
prompt = ChatPromptTemplate.from_messages([
    ("system", "역할 지침"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
chain.invoke({"input": question, "history": history.messages})
```

### 7. Streamlit 채팅 UI

| 개념 | 설명 |
|------|------|
| `st.session_state` | 스크립트가 rerun돼도 유지되는 세션 저장소 (대화 기록 보관) |
| `st.chat_input()` | 하단 채팅 입력창 — 입력 시 스크립트 rerun |
| `st.chat_message()` | user / assistant 말풍선 UI |
| `HumanMessage` / `AIMessage` | Streamlit dict → LangChain 메시지 변환 (Chain의 `history`에 전달) |
| `layout="wide"` | 넓은 화면 + `st.columns`로 좌우 패널 분리 |
| `container(border=True)` | 테두리 박스로 설정·채팅 영역 구분 |
| `st.pills` | 버튼 형태 주제 선택 — 클릭 시 미리 정의한 질문 자동 전송 |
| `st.metric` | 질문 수·메시지 수 등 간단한 통계 표시 |

```bash
streamlit run langchain_chatbot3.py   # 여행 추천
streamlit run langchain_chatbot4.py   # 재테크 입문
```

> Streamlit은 입력할 때마다 스크립트 전체를 **처음부터 다시 실행(rerun)** 합니다.
> 따라서 `while` 루프 대신 `session_state`에 대화를 저장해야 합니다.

### 8. chatbot3 vs chatbot4

| 항목 | `chatbot3` | `chatbot4` |
|------|------------|------------|
| 주제 | 여행 추천 | 재테크 입문 |
| user 변수 | `{input}` | `{question}` |
| temperature | `0.1` | `0.7` |
| UI | 단순 채팅 | wide 레이아웃 + pills + metric |
| system 프롬프트 | 짧은 역할 지정 | 5단계 출력 형식 + 위험 고지 |

### 9. RAG — 문서 로딩 (Document Loader)

| 개념 | 설명 |
|------|------|
| **RAG** | Retrieval-Augmented Generation — 외부 문서를 검색해 LLM 답변에 반영 |
| `PyPDFLoader` | PDF 파일을 페이지별 `Document` 리스트로 변환 |
| `Document` | `page_content`(본문) + `metadata`(페이지 번호, 출처 등) |
| `loader.load()` | 파일 전체를 읽어 `List[Document]` 반환 |

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/sample.pdf")
documents = loader.load()   # 페이지마다 Document 1개

for doc in documents:
    print(doc.metadata["page"])   # 페이지 번호
    print(doc.page_content)       # 페이지 본문
```

> `PyPDFLoader`는 pip 패키지가 아닙니다. `pip install langchain-community pypdf`로 설치하세요.

**RAG 파이프라인 흐름:**

```
PDF 로딩 → 텍스트 분할 → 임베딩·벡터 DB → 검색(Retriever) → LLM 답변
  rag.py     rag2.py      chroma_rag.py    retriever_rag.py    (다음 단계)
```

### 10. RAG — 텍스트 분할 (Text Splitter)

| 개념 | 설명 |
|------|------|
| **Chunking** | 긴 문서를 임베딩·검색에 적합한 크기로 분할 |
| `RecursiveCharacterTextSplitter` | 문단 → 문장 → 단어 순으로 재귀 분할 (가장 많이 사용) |
| `chunk_size` | 청크당 최대 문자 수 |
| `chunk_overlap` | 인접 청크 간 겹치는 문자 수 — 문맥 단절 방지 |
| `split_documents()` | `Document` 리스트를 더 작은 `Document` 청크 리스트로 변환 |

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
splits = text_splitter.split_documents(documents)
```

> 페이지 수(예: 10페이지)보다 청크 수(예: 25개)가 많을 수 있습니다. 한 페이지가 `chunk_size`보다 길면 여러 청크로 나뉩니다.

### 11. RAG — 벡터 DB 저장 (Chroma)

| 개념 | 설명 |
|------|------|
| **Embedding** | 텍스트를 고차원 벡터로 변환 — 의미가 비슷한 문장은 벡터도 가깝게 배치 |
| `OpenAIEmbeddings` | OpenAI 임베딩 API (`text-embedding-3-small` 등) |
| `Chroma` | 로컬 벡터 데이터베이스 — 임베딩 저장·유사도 검색 |
| `from_documents()` | Document 청크를 임베딩해 DB **생성** (최초 1회) |
| `persist_directory` | DB 파일이 저장되는 로컬 폴더 (`./chroma_store`) |
| `collection_name` | Chroma 컬렉션 이름 — 로드 시 동일하게 지정 |

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_store",
    collection_name="pdf_collection",
)
```

> ⚠️ 파일명을 `langchain_chroma.py`로 짓면 `langchain_chroma` 패키지와 충돌합니다. `langchain_chroma_rag.py`처럼 다른 이름을 사용하세요.

> `langchain_chroma_rag.py` 실행 전, 스크립트에서 지정한 PDF 파일을 `data/` 폴더에 준비하세요.

### 12. RAG — Retriever 검색

| 개념 | 설명 |
|------|------|
| **Retriever** | 질의(query)를 받아 관련 Document 청크를 반환하는 검색 인터페이스 |
| `as_retriever()` | 벡터스토어를 Retriever로 변환 |
| `search_kwargs k` | 반환할 상위 유사 문서 개수 |
| `retriever.invoke(query)` | 자연어 질의 → 유사 청크 `List[Document]` 반환 |
| `Chroma()` | `persist_directory`에 저장된 기존 DB **로드** (`from_documents` 아님) |

```python
# 기존 DB 로드
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_store",
    collection_name="pdf_collection",
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("문서의 핵심 내용은 무엇인가요?")
```

> `langchain_retriever_rag.py`는 `langchain_chroma_rag.py`를 먼저 실행해 `chroma_store`가 있어야 동작합니다.

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
| `langchain_memory.py` | `InMemoryChatMessageHistory` + while 루프 대화 |
| `langchain_memory2.py` | `MessagesPlaceholder` + Chain 맥락 대화 |
| `langchain_chatbot.py` | `ChatPromptTemplate` + LCEL, 여행지·목적 1회 추천 |
| `langchain_chatbot2.py` | `memory2` 패턴 + CLI 반복 대화 (`exit`로 종료) |
| `langchain_chatbot3.py` | `chatbot2` + Streamlit 채팅 UI (`streamlit run` 실행) |
| `langchain_chatbot4.py` | 재테크 입문 챗봇 — `pills` 주제 선택, wide 레이아웃, `{question}` 변수 |
| `langchain_rag.py` | `PyPDFLoader`로 PDF → `Document` 변환 (RAG 문서 로딩) |
| `langchain_rag2.py` | `RecursiveCharacterTextSplitter`로 Document 청크 분할 (RAG 2단계) |
| `langchain_chroma_rag.py` | `OpenAIEmbeddings` + `Chroma.from_documents()`로 벡터 DB 저장 (RAG 3단계) |
| `langchain_retriever_rag.py` | `Chroma` 로드 + `as_retriever()`로 유사 문서 검색 (RAG 4단계) |
| `data/sample.pdf` | RAG 학습용 샘플 PDF (Git 포함) |
| `chroma_store/` | Chroma 벡터 DB 저장 폴더 (`chroma_rag.py` 실행 시 생성) |
| `.env.example` | API 키 설정 예시 (실제 키 X) |
| `requirements.txt` | 필요 패키지 목록 (`streamlit`, `langchain-community`, `langchain-chroma`, `pypdf` 포함) |

---

## Git에 올리지 않는 파일

`.gitignore`에 의해 아래 파일/폴더는 제외됩니다.

- `.env` — API 키 등 비밀정보
- `.venv/` — 가상환경
- `__pycache__/` — Python 캐시
- `chroma_store/` — Chroma 벡터 DB (실행 시 자동 생성)
- `data/*` — PDF 등 데이터 파일 (`data/sample.pdf`만 예외로 포함)
