from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def chunk_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, 
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]  # Adjust separators as needed
        )
    return splitter.split_documents(documents)

def build_or_load_vectorstore(documents, index_path="index/faiss_store"):
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small")
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  
    
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    chunks = chunk_docs(documents)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore