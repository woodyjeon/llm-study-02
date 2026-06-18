# RAG 기초 — PDF 문서 로딩 예제
# PyPDFLoader로 PDF를 페이지 단위 Document 리스트로 변환
# RAG(Retrieval-Augmented Generation)의 첫 단계: 외부 문서 불러오기

from langchain_community.document_loaders import PyPDFLoader

# PyPDFLoader: PDF 파일 → LangChain Document 객체 리스트
# 필요 패키지: langchain-community, pypdf (pip install langchain-community pypdf)
loader = PyPDFLoader("data/sample.pdf")

# load(): 각 페이지가 Document 1개 — page_content(본문) + metadata(페이지 번호 등)
documents = loader.load()

for doc in documents:
    print("페이지: ", doc.metadata["page"])
    print(doc.page_content[:200])  # 본문 앞 200자만 미리보기
    print("=" * 50)
