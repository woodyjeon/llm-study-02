# RAG 4단계 — Retriever 검색 예제
# langchain_chroma_rag.py로 저장한 Chroma DB에서 질의와 유사한 문서 청크를 검색
# as_retriever()로 벡터스토어를 검색기(Retriever) 인터페이스로 변환
# 사전 준비: python langchain_chroma_rag.py 먼저 실행해 chroma_store 생성

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# 검색 시 쿼리를 벡터로 변환할 임베딩 모델 (DB 생성 시와 동일한 모델 사용)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

# Chroma(): persist_directory에 저장된 기존 DB 로드
# from_documents()는 새 문서 저장용 — documents 인자 없이 호출하면 에러
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_store",
    collection_name="pdf_collection",
)

# as_retriever(): 벡터스토어 → Retriever 인터페이스
# search_kwargs k=3: 코사인 유사도 상위 3개 청크 반환
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3},
)

# 자연어 질의 — 임베딩 후 DB에서 가장 유사한 청크 검색
query = "문서의 핵심 내용은 무엇인가요?"

# invoke(query): 질의와 의미적으로 가장 가까운 Document 청크 리스트 반환
docs = retriever.invoke(query)

print("검색된 문서 수: ", len(docs))

for i, doc in enumerate(docs):
    print(f"\n검색 결과 {i+1}:")
    print("페이지: ", doc.metadata["page"])       # PDF 원본 페이지 번호
    print("콘텐츠: ", doc.page_content[:200])     # 본문 앞 200자만 미리보기
    print("=" * 50)
