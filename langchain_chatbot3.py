# Streamlit + LangChain 챗봇 예제
# chatbot2의 맥락 대화를 Streamlit 웹 UI로 확장
# st.session_state로 대화 기록 유지 → LangChain 메시지로 변환 후 Chain 실행

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# API 키 없으면 Streamlit 화면에 에러 표시 후 종료
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY가 없습니다. .env 파일을 확인하세요.")
    st.stop()

st.set_page_config(page_title="LangChain 여행 추천 챗봇")
st.title("LangChain 여행 추천 챗봇")

llm = ChatOpenAI(model="gpt-5-nano", temperature=0.1)

# MessagesPlaceholder: system → (이전 대화 history) → user(새 질문) 순서로 메시지 구성
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "너는 친절한 여행 플래너 챗봇이야."),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}"),
    ]
)

# prompt | llm | parser: 프롬프트 → LLM 호출 → str 변환
chain = prompt | llm | StrOutputParser()

# st.session_state: Streamlit 재실행(rerun)마다 유지되는 세션 저장소
# messages: {"role": "user"|"assistant", "content": "..."} 형태의 대화 목록
if "messages" not in st.session_state:
    st.session_state.messages = []

# 페이지 로드·재실행 시 저장된 대화를 채팅 UI에 다시 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("여행에 대해 질문하세요")

if user_input:
    # 1) 사용자 메시지 저장 및 화면 출력
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # 2) Streamlit dict → LangChain HumanMessage / AIMessage 변환
    #    messages[:-1]: 현재 입력은 {input}으로 따로 전달 (chatbot2와 동일한 history 패턴)
    history = []
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))

    # 3) Chain 실행 — input(새 질문) + history(이전 대화) 함께 전달
    result = chain.invoke({"history": history, "input": user_input})

    # 4) AI 응답 저장 및 화면 출력
    st.session_state.messages.append({"role": "assistant", "content": result})
    with st.chat_message("assistant"):
        st.write(result)
