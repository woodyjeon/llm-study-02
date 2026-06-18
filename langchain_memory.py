# Chat Memory 예제 (대화 기록 유지)
# InMemoryChatMessageHistory로 이전 대화를 저장 → LLM이 맥락을 기억하며 답변

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()

llm = ChatOpenAI(model="gpt-5-nano", temperature=0.1)

# InMemoryChatMessageHistory: 대화 기록을 메모리에 저장 (프로그램 종료 시 사라짐)
history = InMemoryChatMessageHistory()

while True:
    user_input = input("사용자: ")

    if user_input == "exit":
        break

    # 1) 사용자 메시지를 history에 추가
    history.add_user_message(user_input)

    # 2) 전체 대화 기록(messages)을 LLM에 전달 → 이전 대화 맥락 반영
    response = llm.invoke(history.messages)

    print(f"AI: {response.content}")

    # 3) AI 응답도 history에 추가 → 다음 턴에서 함께 전달됨
    history.add_ai_message(response.content)
