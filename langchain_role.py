# Message 객체 예제 (역할 기반 대화)
# SystemMessage + HumanMessage로 프롬프트를 직접 구성하는 가장 기본적인 방식

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)

load_dotenv()

llm = ChatOpenAI(model="gpt-5-nano", temperature=0.1)

# messages 리스트: 대화 순서대로 메시지를 쌓음
messages = [
    # SystemMessage: AI의 역할/페르소나 설정 (모델이 "누구처럼" 행동할지)
    SystemMessage(content="너는 컴퓨터공학과 조교야. 학생들에게 친절히 답변해줘."),
    # HumanMessage: 사용자(사람)의 질문/입력
    HumanMessage(content="컴퓨터공학과에서는 무엇을 배우나요?"),
]

# invoke()에 messages 리스트를 전달 → AIMessage 응답 반환
response = llm.invoke(messages)

print(response.content)
