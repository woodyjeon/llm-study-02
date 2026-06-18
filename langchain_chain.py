# LangChain Chain (LCEL) 예제
# prompt → llm → parser 를 파이프(|)로 연결해 하나의 chain-of-work로 실행

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# .env 파일에서 OPENAI_API_KEY 등 환경변수 로드
load_dotenv()

# LLM 모델 생성
# temperature: 0에 가까울수록 일관된 답변, 높을수록 창의적인 답변
llm = ChatOpenAI(model="gpt-5-nano", temperature=0.1)

# ChatPromptTemplate: {topic} 같은 변수 자리를 만들어 두고, invoke() 시 값을 채움
# f-string(f"...{topic}")과 달리, 실행 시점이 아니라 invoke() 호출 시 치환됨
prompt = ChatPromptTemplate.from_template(
    """
    {topic}에 대해 설명해줘
    """
)

# StrOutputParser: LLM 응답(AIMessage 객체)에서 .content 문자열만 추출
# parser 없으면 result가 AIMessage 객체라서 result.content로 접근해야 함
parser = StrOutputParser()

# LCEL(LangChain Expression Language): | 연산자로 단계를 순서대로 연결
# 1) prompt가 {"topic": "인공지능"} → "인공지능에 대해 설명해줘" 문자열 생성
# 2) llm이 해당 문자열을 받아 AI 답변(AIMessage) 생성
# 3) parser가 AIMessage → 순수 문자열(str)로 변환
chain = prompt | llm | parser

# 여러 주제를 리스트로 준비 → for문으로 chain을 반복 호출 (배치 처리 패턴)
topics = [
    "인공지능",
    "피지컬AI",
    "데이터분석"
]

for topic in topics:
    # invoke(): chain에 입력값(dict)을 넘기고 최종 결과(str)를 받음
    result = chain.invoke({"topic": topic})
    print(f"주제: {topic}")
    print(f" 설명: {result}")
    print("-" * 50)
