# 재테크 입문 대화형 챗봇 (Streamlit + LangChain LCEL)
# chatbot3의 Streamlit UI를 재테크 주제 + Stock Peers 스타일 레이아웃으로 확장
# st.pills로 빠른 주제 선택, {question} 변수로 사용자 질문 전달

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# layout="wide": Stock Peers 데모처럼 좌(설정) / 우(채팅) 2열 레이아웃
st.set_page_config(
    page_title="재테크 입문 상담",
    page_icon=":material/trending_up:",
    layout="wide",
)

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY가 없습니다. .env 파일을 확인하세요.")
    st.stop()

"""
# :material/savings: 재테크 입문 상담

입문자를 위한 예적금·ETF·주식 가이드. 왼쪽에서 주제를 고르거나 아래에 질문하세요.
"""

""

# st.pills 클릭 시 자동으로 보낼 질문 매핑 (키=표시 라벨, 값=실제 질문)
TOPICS = {
    "예적금": "예적금이 뭐고, 입문자에게 왜 좋은지 알려줘",
    "ETF": "ETF 기초 개념과 시작 방법을 알려줘",
    "주식": "주식 투자 기초를 입문자 눈높이로 설명해줘",
    "투자 성향": "안정형·중립형·공격형 투자 성향을 어떻게 파악하나요?",
    "경제 뉴스": "요즘 경제 흐름을 입문자 관점에서 요약해줘",
}

llm = ChatOpenAI(model="gpt-5-nano", temperature=0.7)

# MessagesPlaceholder: system → (이전 대화 history) → user(새 질문) 순서로 메시지 구성
# chatbot3의 {input} 대신 {question} 변수 사용
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 유능한 재테크 전문가야. 재테크 입문자가 이해하기 쉽게 "
            "어려운 용어는 풀어서 설명하고, 아래 형식으로 친절하게 답해줘.\n"
            "1. 기초 개념 설명: 질문과 관련된 예적금 / ETF / 주식 개념을 입문자 눈높이로 설명\n"
            "2. 투자 성향 분석: 안정형 / 중립형 / 공격형 중 어디에 맞는지와 그 이유\n"
            "3. 추천 전략: 성향에 맞는 상품·비중·시작 방법을 단계별로 제시\n"
            "4. 경제 뉴스 요약: 관련된 최근 경제 흐름을 3줄로 핵심만 요약\n"
            "5. 한 줄 조언: 초보자가 주의할 점을 한 문장으로\n\n"
            "투자에는 원금 손실 위험이 있다는 점을 마지막에 꼭 덧붙여줘.",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{question}"),
    ]
)

# LCEL 체인: prompt | llm | parser → 프롬프트 → LLM 호출 → str 변환
chain = prompt | llm | StrOutputParser()

if "messages" not in st.session_state:
    st.session_state.messages = []

# columns([1, 3]): 왼쪽 1/4 설정 패널, 오른쪽 3/4 채팅 패널
left_col, right_col = st.columns([1, 3])

settings = left_col.container(border=True, height="stretch")

with settings:
    # st.pills: 주제 버튼 — 클릭 시 TOPICS dict에서 질문 자동 선택
    topic = st.pills("관심 주제", options=list(TOPICS.keys()), default=None)

    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("last_pill_topic", None)
        st.rerun()

    user_count = sum(1 for m in st.session_state.messages if m["role"] == "user")
    m1, m2 = st.columns(2)
    m1.metric("질문 수", user_count)
    m2.metric("메시지", len(st.session_state.messages))

chat_panel = right_col.container(border=True, height="stretch")

with chat_panel:
    if not st.session_state.messages:
        st.info("왼쪽에서 주제를 고르거나, 아래에 질문을 입력하세요.", icon=":material/info:")

    for msg in st.session_state.messages:
        bubble = st.chat_message(msg["role"])
        bubble.write(msg["content"])

user_input = st.chat_input("재테크에 대해 질문하세요")

# pills 클릭 → TOPICS 질문 / chat_input 입력 → 직접 질문 (같은 pill 재클릭은 무시)
question = user_input
if topic and st.session_state.get("last_pill_topic") != topic:
    st.session_state.last_pill_topic = topic
    question = TOPICS[topic]

if question:
    # 1) 사용자 메시지 저장 및 화면 출력
    st.session_state.messages.append({"role": "user", "content": question})
    bubble = st.chat_message("user")
    bubble.write(question)

    # 2) Streamlit dict → LangChain HumanMessage / AIMessage 변환
    #    messages[:-1]: 현재 입력은 {question}으로 따로 전달 (chatbot3과 동일한 history 패턴)
    history = []
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))

    # 3) Chain 실행 — question(새 질문) + history(이전 대화) 함께 전달
    try:
        with st.spinner("답변 생성 중..."):
            result = chain.invoke({"history": history, "question": question})
    except Exception as e:
        st.session_state.messages.pop()
        st.error(f"API 호출 오류: {e}")
        st.stop()

    # 4) AI 응답 저장 및 화면 출력
    st.session_state.messages.append({"role": "assistant", "content": result})
    bubble = st.chat_message("assistant")
    bubble.write(result)
