# RAG 2단계 — 텍스트 분할(Chunking) 예제
# PyPDFLoader로 로드한 Document를 작은 청크로 나눔
# RAG에서 임베딩·검색 전에 긴 문서를 적당한 크기로 쪼개는 단계

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1단계: PDF → Document 리스트 (페이지 단위)
loader = PyPDFLoader("data/sample.pdf")
documents = loader.load()

# RecursiveCharacterTextSplitter: 문단·문장·단어 순으로 재귀 분할
# chunk_size: 청크당 최대 문자 수 / chunk_overlap: 인접 청크 간 겹치는 문자 수 (맥락 유지)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# split_documents(): Document 리스트 → 더 작은 Document 청크 리스트
splits = text_splitter.split_documents(documents)

print("원본페이지 수: ", len(documents))
print("분할된 페이지 수: ", len(splits))
