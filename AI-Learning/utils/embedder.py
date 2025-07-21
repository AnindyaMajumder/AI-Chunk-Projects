import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)

def build_or_load_vectorstore(documents, index_path="index/faiss_store"):
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small")
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    chunks = chunk_docs(documents)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore