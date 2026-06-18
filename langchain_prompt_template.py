# PromptTemplate 예제 (단일 문자열 프롬프트)
# ChatPromptTemplate과 달리 역할 구분 없이 하나의 텍스트 템플릿 사용

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-5-nano", temperature=0.1)

# from_template(): {job}, {subject} 같은 변수 자리를 가진 문자열 템플릿 생성
prompt = PromptTemplate.from_template(
    """
    당신의 {job} 입니다.
    {subject}에 대해 설명하세요.
    """
)

# format(): 템플릿 변수에 실제 값을 넣어 완성된 프롬프트 문자열 반환
# chain 없이 수동으로 prompt → llm 순서를 직접 실행하는 방식
result = prompt.format(job="컴퓨터공학과 교수", subject="인공지능")

# llm.invoke()에 문자열을 직접 전달 (ChatPromptTemplate + chain 방식과 비교해 볼 것)
response = llm.invoke(result)

print(response.content)
