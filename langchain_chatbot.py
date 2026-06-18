from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-5-nano", temperature=0.1)

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 친절한 여행 플래너 AI야."),
    ("human", """
    여행지: {city}
    여행목적: {purpose}

    아래 형식으로 여행 코스를 추천해줘,

    1. 추천 관광지
    2. 추천 맛집
    3. 하루 일정표
    4. 여행 팁
    """),
])

parser = StrOutputParser()

chain = prompt | llm | parser

city = input("여행지: ")
purpose = input("여행목적: ")

result = chain.invoke({"city": city, "purpose": purpose})

print("\n === AI 추천 결과 ===")
print(result)
