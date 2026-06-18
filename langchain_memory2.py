# Chat Memory + MessagesPlaceholder 예제
# ChatPromptTemplate에 대화 기록(history)을 끼워 넣어 맥락 있는 답변 생성

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()

llm = ChatOpenAI(model="gpt-5-nano", temperature=0.1)

history = InMemoryChatMessageHistory()

# MessagesPlaceholder: 프롬프트 중간에 대화 기록을 동적으로 삽입하는 자리
# system → (이전 대화 history) → human(새 질문) 순서로 메시지 구성
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "너는 여행 전문가야. 사용자의 질문에 친절하게 답해줘"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# prompt | llm: 프롬프트(system + history + human) → LLM 호출
chain = prompt | llm

# 미리 정의한 질문 2개로 대화 시뮬레이션 (2번째 질문은 1번째 맥락을 기억해야 함)
inputs = [
    {"input": "부산 2박 3일 여행지 추천해줘"},
    {"input": "맛집 추천해줘"},  # "거기" 맥락 → 부산 맛집으로 이해해야 함
]

for item in inputs:
    # invoke()에 input(새 질문) + history(이전 대화) 함께 전달
    result = chain.invoke({**item, "history": history.messages})

    print(f"사용자: {item['input']}")
    print(f"응답: {result.content}")
    print("-" * 50)

    # 대화 종료 후 history에 user/ai 메시지 추가 → 다음 턴에 맥락으로 사용
    history.add_user_message(item["input"])
    history.add_ai_message(result.content)
