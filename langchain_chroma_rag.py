# RAG 3단계 — Chroma 벡터 DB 저장 예제
# PDF 청크를 OpenAI 임베딩으로 벡터화한 뒤 Chroma에 영구 저장
# langchain_retriever_rag.py에서 이 DB를 불러와 검색에 사용
# 사전 준비: .env에 OPENAI_API_KEY 설정, data/ 폴더에 PDF 파일 배치

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# 1단계: PDF → Document 리스트 (페이지 단위)
loader = PyPDFLoader("data/2040_seoul_plan.pdf")
documents = loader.load()

print("원본페이지 수: ", len(documents))

# 2단계: 긴 문서를 검색에 적합한 크기의 청크로 분할
# chunk_size: 청크당 최대 문자 수 / chunk_overlap: 인접 청크 간 겹치는 문자 수
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

print("분할된 페이지 수: ", len(splits))

# OpenAIEmbeddings: 텍스트 → 벡터 변환 (검색 시에도 동일 모델 사용 필수)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

# persist_directory: 벡터 DB가 저장될 로컬 폴더
# collection_name: Chroma 컬렉션 이름 (로드 시 동일하게 지정)
persist_directory = "./chroma_store"

# from_documents: 청크를 임베딩한 뒤 Chroma DB 생성 및 디스크 저장
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name="pdf_collection",
)

print("Chroma DB 생성 완료")
