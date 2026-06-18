from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder

load_dotenv()

llm = ChatOpenAI(model="gpt-5-nano", temperature=0.1)

history = InMemoryChatMessageHistory()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "너는 친절한 여행 플래너 AI야."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

parser = StrOutputParser()

chain = prompt | llm | parser

while True:
    user_input = input("사용자: ")
    if user_input.lower() == "exit":
        break

    result = chain.invoke({"input": user_input, "history": history.messages})

    history.add_user_message(user_input)
    history.add_ai_message(result)

    print("\n === AI 추천 결과 ===")
    print("AI :", result)
