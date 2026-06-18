# 반복 대화 예제 (while 루프)
# 사용자 입력을 받아 LLM에 계속 질문하는 간단한 챗봇 패턴

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-5-nano", temperature=0.1)

# "exit" 입력 시 루프 종료
while True:
    question = input("질문을 입력하세요: ")
    if question == "exit":
        break

    # 문자열을 직접 invoke → HumanMessage로 자동 변환됨
    # (역할 설정 없이 단순 질문만 보내는 방식)
    response = llm.invoke(question)

    print(response.content)
