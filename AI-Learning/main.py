from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import os
import pymupdf
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


# Load PDF documents from a directory
def load_pdfs(pdf_dir):
    documents = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            doc = pymupdf.open(pdf_path)
            pages = ""
            
            for page in doc:
                text = page.get_text()
                pages += str(text)
            
            # Create a Document object with metadata
            documents.append(Document(
                page_content=pages,
                metadata={"source": filename, "type": "pdf"}
            ))
            doc.close()
    
    return documents

# Load training phrases from CSV files
def load_training_phrases(csv_path):
    documents = []
    if os.path.isdir(csv_path):
        for filename in os.listdir(csv_path):
            if filename.endswith(".csv"):
                print(f"Loading training phrases")
                df = pd.read_csv(os.path.join(csv_path, filename), encoding="utf-8")
                for index, row in df.iterrows():
                    content = " ".join(str(value) for value in row.values if pd.notna(value))
                    # Create a Document object with metadata
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": filename, "type": "training_phrase", "row": index}
                    ))
    return documents

# Embedder utility functions
def chunk_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
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

def prompt(claim_no: int, name: str, phone: str, email: str):
    prompt = ChatPromptTemplate.from_template(
        """ 
            You are Benji, a calm and strategic assistant helping users through insurance claims.
            Your personality:
            - Calm, never emotional
            - Strategic like a chess coach
            - Empathetic, warm, and confident

            Always reinforce: “Stay calm. This is a game of chess. The goal is to get paid — not to get angry.”

            Include editable templates when useful. Avoid robotic responses.

            Context:
            {context}

            Conversation history:
            {chat_history}

            User question:
            {question}
            
            CLAIM DETAILS:
            - Claim Number: {claim_no}
            - Claimant Name: {name}
            - Contact Phone: {phone}
            - Contact Email: {email}
            
            Answer as Benji:
        """
    )
    
    return prompt

def model_init():
    load_dotenv()
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        max_tokens=2048,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    return llm

def chaining(claim_no: int, name: str, phone: str, email: str, global_knowledge = "data/", local_knowledge = "upload/", global_store="index/", local_store="locals/"):
    load_dotenv()
    
    # Load global knowledge base
    global_docs = load_pdfs(global_knowledge)
    global_vectorstore = build_or_load_vectorstore(global_docs, os.path.join(global_store, "faiss_store"))
    
    # Load local knowledge base
    local_docs = load_pdfs(local_knowledge)
    local_vectorstore = build_or_load_vectorstore(local_docs, os.path.join(local_store, "faiss_store"))
    
    # Initialize the model
    llm = model_init()
    
    # Create the prompt
    prompt_template = prompt(claim_no, name, phone, email)
    
    def format_inputs(inputs):
        # Retrieve relevant documents
        global_docs = global_vectorstore.as_retriever().invoke(inputs["question"])
        local_docs = local_vectorstore.as_retriever().invoke(inputs["question"])
        
        # Combine contexts
        combined_context = "\n\n".join([doc.page_content for doc in global_docs + local_docs])
        
        return {
            "context": combined_context,
            "chat_history": inputs.get("chat_history", ""),
            "question": inputs["question"],
            "claim_no": inputs["claim_no"],
            "name": inputs["name"],
            "phone": inputs["phone"],
            "email": inputs["email"]
        }
    
    chain = (
        format_inputs
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    return chain

def main():
    claim_no = 123456
    name = "John Doe"
    phone = "123-456-7890"
    email = "johndoe@gmail.com"
    
    chain = chaining(claim_no, name, phone, email)
    result = chain.invoke({
        "claim_no": claim_no,
        "name": name,
        "phone": phone,
        "email": email,
        "question": "This is a sample context for the claim.",
        "chat_history": "Previous conversation history goes here."
    })
    
    print(result)

if __name__ == "__main__":
    main()