# JsonOutputParser 예제
# LLM 응답(JSON 문자열) → Python dict로 자동 파싱

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-5-nano", temperature=0.1)

# PromptTemplate: {topic}은 템플릿 변수 (f-string 아님, invoke() 시 치환)
# {{ }} : JSON 예시의 중괄호를 리터럴로 출력 (단일 { } 는 LangChain이 변수로 인식)
prompt = PromptTemplate.from_template(
    """ 다음 강의 주제를 JSON 형식으로 출력해줘
    주제: {topic}
    반드시 다음 형식으로 답해주세요.
    {{
        "title": "강의 제목",
        "level": "초급, 중급, 고급",
        "keywords": ["키워드1", "키워드2", "키워드3"],
        "summary": "한줄요약"
    }}
    """
)

# JsonOutputParser: LLM이 반환한 JSON 문자열 → Python dict 변환
# StrOutputParser(str)와 달리 result["title"]처럼 키로 바로 접근 가능
parser = JsonOutputParser()

# LCEL: prompt → llm → parser 를 | 연산자로 연결
# 1) prompt  → {topic}을 실제 값으로 치환해 프롬프트 문자열 생성
# 2) llm     → JSON 형식 AI 응답(AIMessage) 생성
# 3) parser  → AIMessage.content의 JSON → dict 변환
chain = prompt | llm | parser

# invoke()에 템플릿 변수 dict 전달 — topic에는 "LangChain" 같은 값만 넘김
#
# ❌ 버그 예시 (format + invoke 섞기):
#    input_variables = prompt.format(topic="LangChain")  ← 프롬프트 전체 문자열
#    chain.invoke({"topic": input_variables})            ← {topic}에 프롬프트 전체가 들어감
#
# ✅ 올바른 방법: chain이 내부에서 prompt.format()을 처리하므로 값만 넘김
result = chain.invoke({"topic": "LangChain"})

# result는 dict → result["title"], result["keywords"] 등으로 접근
print(result)
