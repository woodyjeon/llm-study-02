# ChatPromptTemplate 예제 (역할 기반 대화형 프롬프트)
# system(역할 설정) + human(사용자 입력) 메시지 구조로 프롬프트 구성

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-5-nano", temperature=0.1)

# from_messages(): 채팅 형식의 프롬프트 생성
# - ("system", ...): AI의 역할/행동 지침
# - ("human", ...): 사용자 메시지, {interest}는 템플릿 변수 (f-string 아님!)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    너는 진로상당 전문가야
    사용자의 관심 분야를 분석해서 적합한 직업 추천해줘.
    """,
        ),
        ("human", "관심 분야: {interest}"),
    ]
)

# prompt | llm: 프롬프트 생성 → LLM 호출을 한 번에 (parser 없으면 AIMessage 반환)
chain = prompt | llm

interest = input("관심 분야를 입력하세요: ")

# invoke()에 템플릿 변수 dict 전달 → {interest} 자리에 실제 값이 들어감
response = chain.invoke({"interest": interest})

print(response.content)
