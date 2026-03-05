import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DOCS_PATH = "consumer_docs"
DB_PATH = "vector_db"

documents = []

for file in os.listdir(DOCS_PATH):
    if file.lower().endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DOCS_PATH, file))
        documents.extend(loader.load())

print(f"Loaded {len(documents)} pages")


splitter = RecursiveCharacterTextSplitter(chunk_size=1200,chunk_overlap=250,)

chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2")

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(DB_PATH)

print("Vector DB created successfully")
